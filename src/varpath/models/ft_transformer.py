"""Multi-task FT-Transformer.

Architecture (unchanged from the prototype): every scalar feature becomes a
token ``x_j * W_j + b_j``; a CLS token is prepended; a pre-norm Transformer
encoder is applied; two heads read the CLS embedding – pathogenicity (main
task, BCE with ``pos_weight`` and label smoothing) and panel membership
(auxiliary task, cross-entropy), mixed with weight ``mtl_alpha``.

Epoch selection uses an explicit early-stopping hold-out that the caller
carves from the *training* partition. Refits on a full training partition run
for a fixed number of epochs (median of the inner best epochs) with the same
cosine schedule, so no validation data is needed at refit time.
"""

from __future__ import annotations

import logging
import math
import warnings
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score

from ..config import FTConfig

logger = logging.getLogger("varpath")


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)


class MTLFTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        n_panels: int = 4,
    ):
        super().__init__()
        if d_token % n_heads:
            raise ValueError("d_token must be divisible by n_heads")
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.patho_head = nn.Sequential(
            nn.LayerNorm(d_token), nn.Dropout(dropout), nn.Linear(d_token, 1)
        )
        self.panel_head = nn.Sequential(
            nn.LayerNorm(d_token), nn.Dropout(dropout), nn.Linear(d_token, n_panels)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = self.transformer(torch.cat([cls, tok], dim=1))[:, 0]
        return self.patho_head(h).squeeze(-1), self.panel_head(h)


def select_tokens(X: np.ndarray, y: np.ndarray, budget: int | None) -> np.ndarray:
    """Indices of the ``budget`` highest ANOVA-F features (training data only)."""
    n = X.shape[1]
    if budget is None or budget >= n:
        return np.arange(n)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, _ = f_classif(X, y)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0)
    top = np.argsort(-f, kind="stable")[:budget]
    return np.sort(top)


class FTTransformerClassifier:
    """sklearn-like wrapper with explicit early-stopping data."""

    def __init__(self, cfg: FTConfig, n_panels: int, seed: int = 42, device: str = "cpu"):
        self.cfg = cfg
        self.n_panels = int(n_panels)
        self.seed = int(seed)
        self.device = device
        self.model: MTLFTTransformer | None = None
        self.selected_: np.ndarray | None = None
        self.best_epoch_: int | None = None
        self.history_: list[dict] = []

    # ------------------------------------------------------------ internals
    def _autocast(self):
        if self.device.startswith("cuda") and self.cfg.amp:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def _uses_scaler(self) -> bool:
        return (
            self.device.startswith("cuda") and self.cfg.amp and not torch.cuda.is_bf16_supported()
        )

    def _build(self, n_features: int) -> None:
        torch.manual_seed(self.seed)
        self.model = MTLFTTransformer(
            n_features,
            d_token=self.cfg.d_token,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
            n_panels=self.n_panels,
        ).to(self.device)

    @torch.inference_mode()
    def _predict_selected(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = []
        bs = self.cfg.eval_batch_size
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i : i + bs], dtype=torch.float32, device=self.device)
            with self._autocast():
                logits, _ = self.model(xb)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    # ------------------------------------------------------------------ API
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        panel_idx: np.ndarray,
        pos_weight: float,
        X_es: np.ndarray | None = None,
        y_es: np.ndarray | None = None,
        n_epochs: int | None = None,
    ) -> FTTransformerClassifier:
        """Train with early stopping on (X_es, y_es) or for exactly ``n_epochs``."""
        cfg = self.cfg
        use_es = X_es is not None
        if not use_es and n_epochs is None:
            raise ValueError("Provide early-stopping data or a fixed n_epochs")
        self.selected_ = select_tokens(X, y, cfg.token_budget)
        Xs = np.ascontiguousarray(X[:, self.selected_], dtype=np.float32)
        Xe = np.ascontiguousarray(X_es[:, self.selected_], dtype=np.float32) if use_es else None
        self._build(Xs.shape[1])

        dev = self.device
        Xt = torch.as_tensor(Xs, device=dev)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=dev)
        pt = torch.as_tensor(np.asarray(panel_idx, dtype=np.int64), device=dev)
        bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=dev)
        )
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # T_max is always max_epochs so a fixed-epoch refit follows the same LR path.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
        scaler = torch.amp.GradScaler("cuda") if self._uses_scaler() else None
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        smooth = cfg.label_smoothing

        total = cfg.max_epochs if use_es else int(n_epochs)
        best_score, best_state, best_epoch, since_best = -np.inf, None, total, 0
        self.history_ = []
        for epoch in range(1, total + 1):
            self.model.train()
            perm = torch.randperm(len(Xs), generator=gen).to(dev)
            running = 0.0
            for i in range(0, len(Xs), cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                yb = yt[idx] * (1 - smooth) + smooth / 2
                with self._autocast():
                    p_logit, pan_logit = self.model(Xt[idx])
                    loss = cfg.mtl_alpha * bce(p_logit.float(), yb) + (1 - cfg.mtl_alpha) * ce(
                        pan_logit.float(), pt[idx]
                    )
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    opt.step()
                running += float(loss.detach()) * idx.numel()
            sched.step()
            record = {"epoch": epoch, "train_loss": running / max(len(Xs), 1)}
            if use_es:
                score = self._es_score(y_es, self._predict_selected(Xe))
                record["es_score"] = score
                if score > best_score:
                    best_score, best_epoch, since_best = score, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    since_best += 1
                if since_best >= cfg.patience:
                    self.history_.append(record)
                    break
            self.history_.append(record)
        if use_es and best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_epoch_ = best_epoch if use_es else total
        return self

    @staticmethod
    def _es_score(y: np.ndarray, p: np.ndarray) -> float:
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) == 2:
            return float(roc_auc_score(y, p))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FT-Transformer is not fitted")
        return self._predict_selected(np.ascontiguousarray(X[:, self.selected_], dtype=np.float32))

    # ------------------------------------------------------------ persistence
    def state(self) -> dict:
        return {
            "config": asdict(self.cfg),
            "n_panels": self.n_panels,
            "seed": self.seed,
            "selected": self.selected_.tolist(),
            "best_epoch": self.best_epoch_,
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
        }

    @classmethod
    def from_state(cls, state: dict, device: str = "cpu") -> FTTransformerClassifier:
        obj = cls(FTConfig(**state["config"]), state["n_panels"], state["seed"], device)
        obj.selected_ = np.asarray(state["selected"], dtype=int)
        obj.best_epoch_ = state.get("best_epoch")
        obj._build(len(obj.selected_))
        obj.model.load_state_dict(state["state_dict"])
        obj.model.eval()
        return obj
