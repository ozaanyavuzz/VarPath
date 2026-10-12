"""Persist and reload the deployment model.

Layout of an artifact directory::

    manifest.json        schema, thresholds, grouping, config, seed, versions, checksums
    preprocessor.joblib  fitted FoldPreprocessor (imputer, scalers, encoders, selections)
    xgb_model.ubj        XGBoost booster (native format)
    lgb_model.txt        LightGBM booster (native format)
    ft_model.pt          FT-Transformer state dict + config + selected tokens
    meta_learner.joblib  logistic-regression stacker
    calibrator.joblib    isotonic map (diagnostics only; decisions use raw scores)

Inference goes through :class:`VariantPredictor`, which reuses the same
``ensemble_predict`` function as training.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from . import RESEARCH_DISCLAIMER, __version__
from .config import ExperimentConfig, config_from_dict
from .evaluation.thresholding import apply_threshold
from .models.ft_transformer import FTTransformerClassifier
from .models.stacking import BASE_MODELS, StackedPipeline, ensemble_predict
from .utils import library_versions, resolve_device

logger = logging.getLogger("varpath")

ARTIFACT_FORMAT = 1
_FILES = {
    "preprocessor": "preprocessor.joblib",
    "xgb": "xgb_model.ubj",
    "lgb": "lgb_model.txt",
    "ft": "ft_model.pt",
    "meta": "meta_learner.joblib",
    "calibrator": "calibrator.joblib",
}


class ArtifactError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifacts(
    pipeline: StackedPipeline,
    cfg: ExperimentConfig,
    out_dir: str | Path,
    group_audit: dict | None = None,
    extra: dict | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.preprocessor, out / _FILES["preprocessor"])
    pipeline.xgb_model.save_model(out / _FILES["xgb"])
    pipeline.lgb_booster.save_model(str(out / _FILES["lgb"]))
    torch.save(pipeline.ft_model.state(), out / _FILES["ft"])
    joblib.dump(pipeline.meta_learner, out / _FILES["meta"])
    files = dict(_FILES)
    if pipeline.calibrator is not None:
        joblib.dump(pipeline.calibrator, out / _FILES["calibrator"])
    else:
        files.pop("calibrator")

    schema = pipeline.preprocessor.schema()
    manifest = {
        "artifact_format": ARTIFACT_FORMAT,
        "package_version": __version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": RESEARCH_DISCLAIMER,
        "deployment_threshold": float(pipeline.threshold),
        "deployment_threshold_rule": (
            f"max F{cfg.threshold.beta:g} s.t. specificity >= {cfg.threshold.min_specificity} on "
            "cross-fitted OOF stacked predictions of the full training set"
        ),
        "base_model_thresholds": pipeline.base_thresholds,
        "meta_feature_order": list(BASE_MODELS),
        "feature_schema": schema,
        "feature_names": schema["feature_names"],
        "id_column": cfg.data.id_column,
        "panel_column": cfg.data.panel_column,
        "panels": cfg.data.panels,
        "grouping": {
            "strategy": cfg.grouping.strategy,
            "column": cfg.grouping.column,
            "entity_level": cfg.grouping.entity_level,
            "audit": group_audit,
        },
        "training": pipeline.info,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "library_versions": library_versions(),
        "files": files,
        "sha256": {k: _sha256(out / v) for k, v in files.items()},
    }
    if extra:
        manifest["extra"] = extra
    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=_json_default)
    return out


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class VariantPredictor:
    """Stacked-ensemble inference from a persisted artifact directory."""

    def __init__(self, path: str | Path, device: str = "cpu", verify: bool = True):
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise ArtifactError(f"No manifest.json in {self.path}")
        with open(manifest_path, encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        if self.manifest.get("artifact_format") != ARTIFACT_FORMAT:
            raise ArtifactError("Unsupported artifact format")
        files = self.manifest["files"]
        if verify:
            for key, fname in files.items():
                if _sha256(self.path / fname) != self.manifest["sha256"][key]:
                    raise ArtifactError(f"Checksum mismatch for {fname}")
        self._warn_versions()
        self.config: ExperimentConfig = config_from_dict(self.manifest["config"])
        self.device = resolve_device(device)
        self.preprocessor = joblib.load(self.path / files["preprocessor"])
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(self.path / files["xgb"])
        self.lgb_booster = lgb.Booster(model_file=str(self.path / files["lgb"]))
        state = torch.load(self.path / files["ft"], map_location="cpu", weights_only=True)
        self.ft_model = FTTransformerClassifier.from_state(state, device=self.device)
        self.meta_learner = joblib.load(self.path / files["meta"])
        self.calibrator = (
            joblib.load(self.path / files["calibrator"]) if "calibrator" in files else None
        )
        self.deployment_threshold = float(self.manifest["deployment_threshold"])
        self.feature_names: list[str] = list(self.manifest["feature_names"])
        if self.feature_names != list(self.preprocessor.feature_names_out_):
            raise ArtifactError("Manifest feature order differs from the preprocessor schema")

    def _warn_versions(self) -> None:
        saved = self.manifest.get("library_versions", {})
        now = library_versions()
        for pkg in ("sklearn", "xgboost", "lightgbm", "torch"):
            if saved.get(pkg) and now.get(pkg) and saved[pkg] != now[pkg]:
                logger.warning(
                    "%s version differs from training (%s vs %s)", pkg, now[pkg], saved[pkg]
                )

    @property
    def required_columns(self) -> list[str]:
        return list(self.preprocessor.input_columns_)

    def validate_input(self, df: pd.DataFrame) -> list[str]:
        """Return the list of required columns that are absent."""
        return [c for c in self.preprocessor.numeric_columns_ if c not in df.columns] + [
            c
            for c in (
                self.preprocessor.data_cfg.aa_ref_column,
                self.preprocessor.data_cfg.aa_alt_column,
            )
            if c not in df.columns
        ]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(df)

    def predict_matrix(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return ensemble_predict(
            X, self.xgb_model, self.lgb_booster, self.ft_model, self.meta_learner, self.calibrator
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.transform(df)
        comps = self.predict_matrix(X)
        id_col = self.manifest.get("id_column")
        ids = (
            df[id_col].astype(str).to_numpy()
            if id_col in df.columns
            else np.arange(len(df)).astype(str)
        )
        out = pd.DataFrame({"id": ids})
        panel_col = self.manifest.get("panel_column")
        if panel_col in df.columns:
            out["panel"] = df[panel_col].astype(str).to_numpy()
        out["score"] = comps["stack"]
        out["predicted_label"] = apply_threshold(comps["stack"], self.deployment_threshold)
        out["deployment_threshold"] = self.deployment_threshold
        for k in BASE_MODELS:
            out[f"p_{k}"] = comps[k]
        return out

    def explain(self, df: pd.DataFrame, top_k: int = 5) -> list[list[tuple[str, float, float]]]:
        """Top XGBoost TreeSHAP contributions (feature, contribution, value) per row.

        Contributions are computed on exactly the matrix used for prediction and
        describe the deployment XGBoost base learner, not the full stack.
        """
        X = self.transform(df)
        contrib = self.shap_values(X)
        rows = []
        for i in range(X.shape[0]):
            order = np.argsort(-np.abs(contrib[i]))[:top_k]
            rows.append(
                [(self.feature_names[j], float(contrib[i, j]), float(X[i, j])) for j in order]
            )
        return rows

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.feature_names):
            raise ArtifactError("Matrix width does not match the feature schema")
        contrib = self.xgb_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contrib[:, :-1]


def load_artifacts(path: str | Path, device: str = "cpu", verify: bool = True) -> VariantPredictor:
    return VariantPredictor(path, device=device, verify=verify)
