"""Stacked ensemble fitted on one training partition.

:func:`fit_stacked_pipeline` is the single training procedure of the project.
It is called

* once per outer fold with the outer-training rows (nested evaluation), and
* once with all rows to build the deployment model.

It never receives rows it will later be evaluated on. Inside, the training
partition is split into group-disjoint inner folds that provide:

1. Optuna objectives for XGBoost and LightGBM,
2. early-stopping epochs / boosting rounds,
3. genuine out-of-fold base-model predictions (each row predicted by a model
   that did not see it) used to train the logistic-regression meta learner,
4. cross-fitted stacked predictions used for threshold selection and for the
   optional isotonic recalibration map.

Finally the preprocessor and all base learners are refitted on the whole
training partition with the selected settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..config import ExperimentConfig
from ..evaluation.splits import group_holdout_split, group_kfold_splits, stratification_labels
from ..evaluation.thresholding import select_threshold
from ..features.preprocessing import FoldPreprocessor, detect_label_conflicts
from ..features.tda import TDAFeatureExtractor
from ..utils import derive_seed, resolve_device, timed
from .ft_transformer import FTTransformerClassifier
from .gbdt import (
    InnerFold,
    fit_lgb,
    fit_xgb,
    lgb_predict,
    max_depth_upper,
    positive_weight,
    tune_lgb,
    tune_xgb,
    xgb_predict,
)

logger = logging.getLogger("varpath")

BASE_MODELS = ("ft", "xgb", "lgb")  # column order of the meta-feature matrix
ALL_MODELS = BASE_MODELS + ("stack",)


def panel_indices(panels: np.ndarray, panel_names: list[str]) -> np.ndarray:
    mapping = {p: i for i, p in enumerate(panel_names)}
    return np.array([mapping.get(str(p), 0) for p in panels], dtype=np.int64)


def ensemble_predict(
    X: np.ndarray,
    xgb_model: xgb.XGBClassifier,
    lgb_booster: lgb.Booster,
    ft_model: FTTransformerClassifier,
    meta_learner: LogisticRegression,
    calibrator: IsotonicRegression | None = None,
) -> dict[str, np.ndarray]:
    """Base-model and stacked probabilities for an already transformed matrix."""
    comps = {
        "ft": ft_model.predict_proba(X),
        "xgb": xgb_predict(xgb_model, X),
        "lgb": lgb_predict(lgb_booster, X),
    }
    meta_X = np.column_stack([comps[k] for k in BASE_MODELS])
    comps["stack"] = meta_learner.predict_proba(meta_X)[:, 1]
    if calibrator is not None:
        comps["stack_recalibrated"] = calibrator.predict(comps["stack"])
    return comps


@dataclass
class StackedPipeline:
    preprocessor: FoldPreprocessor
    xgb_model: xgb.XGBClassifier
    lgb_booster: lgb.Booster
    ft_model: FTTransformerClassifier
    meta_learner: LogisticRegression
    calibrator: IsotonicRegression | None
    threshold: float
    base_thresholds: dict[str, float]
    info: dict = field(default_factory=dict)
    training_oof: pd.DataFrame | None = None

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(df)

    def predict_matrix(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return ensemble_predict(
            X, self.xgb_model, self.lgb_booster, self.ft_model, self.meta_learner, self.calibrator
        )

    def predict_components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.predict_matrix(self.transform(df))


def _meta_learner(cfg: ExperimentConfig, seed: int) -> LogisticRegression:
    return LogisticRegression(C=cfg.stacking.C, max_iter=cfg.stacking.max_iter, random_state=seed)


def fit_stacked_pipeline(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    panels: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
    n_splits: int,
    tda_extractor: TDAFeatureExtractor | None = None,
    label: str = "",
) -> StackedPipeline:
    """Fit preprocessing, base learners, meta learner and threshold on training rows."""
    timings: dict[str, float] = {}
    device = resolve_device(cfg.compute.device)
    n_jobs = cfg.compute.n_jobs
    tda = tda_extractor or TDAFeatureExtractor(
        cfg.preprocessing.tda_n_points, cfg.compute.cache_tda
    )
    th = cfg.threshold

    df = df.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    panels = np.asarray(panels).astype(str)
    if not (len(df) == len(y) == len(groups) == len(panels)):
        raise ValueError("df, y, groups and panels must have equal length")

    # 1) conflicting duplicate profiles, detected within this training partition only
    conflict = np.zeros(len(df), dtype=bool)
    if cfg.preprocessing.drop_conflicts:
        conflict = detect_label_conflicts(df, y, cfg.data, cfg.preprocessing.conflict_decimals)
    keep = np.flatnonzero(~conflict)
    dfc = df.iloc[keep].reset_index(drop=True)
    yc, gc, pc = y[keep], groups[keep], panels[keep]
    pic = panel_indices(pc, cfg.data.panels)
    n = len(yc)
    logger.info("%s training rows=%d (removed %d conflicting)", label, n, int(conflict.sum()))

    # 2) inner group-aware splits and per-split preprocessing
    strat = stratification_labels(yc, pc, cfg.cv.stratify_by)
    splits = group_kfold_splits(strat, yc, gc, n_splits, seed)
    folds: list[InnerFold] = []
    with timed("inner_preprocessing", timings):
        for j, (tr, va) in enumerate(splits):
            pre = FoldPreprocessor(cfg.data, cfg.preprocessing, derive_seed(seed, j), tda)
            X_tr = pre.fit_transform(dfc.iloc[tr], yc[tr])
            X_va = pre.transform(dfc.iloc[va])
            es_fit, es_hold = group_holdout_split(
                strat[tr], yc[tr], gc[tr], cfg.cv.early_stopping_fraction, derive_seed(seed, j, 7)
            )
            folds.append(InnerFold(tr, va, X_tr, X_va, yc[tr], yc[va], pic[tr], es_fit, es_hold))
    n_feat = folds[0].X_train.shape[1]

    # 3) GBDT tuning on inner folds
    pw_xgb = positive_weight(yc, cfg.xgb.pos_weight_multiplier)
    pw_lgb = positive_weight(yc, cfg.lgb.pos_weight_multiplier)
    pw_ft = positive_weight(yc, cfg.ft.pos_weight_multiplier)
    depth_xgb = max_depth_upper(n, n_feat, pc, cfg.xgb)
    depth_lgb = max_depth_upper(n, n_feat, pc, cfg.lgb)
    with timed("xgb_tuning", timings):
        xres = tune_xgb(folds, n, cfg.xgb, pw_xgb, depth_xgb, derive_seed(seed, 1), n_jobs)
    with timed("lgb_tuning", timings):
        lres = tune_lgb(folds, n, cfg.lgb, pw_lgb, depth_lgb, derive_seed(seed, 2), n_jobs)
    logger.info(
        "%s XGB inner logloss=%.4f (%d trials, %d pruned) | LGB inner logloss=%.4f (%d trials, %d pruned)",
        label, xres.best_value, xres.n_complete, xres.n_pruned,
        lres.best_value, lres.n_complete, lres.n_pruned,
    )  # fmt: skip

    # 4) FT-Transformer: early stopping on a hold-out inside each inner training fold
    oof_ft = np.full(n, np.nan)
    ft_epochs: list[int] = []
    with timed("ft_inner", timings):
        for j, f in enumerate(folds):
            ft = FTTransformerClassifier(
                cfg.ft, len(cfg.data.panels), derive_seed(seed, j, 11), device
            )
            ft.fit(
                f.X_train[f.es_fit],
                f.y_train[f.es_fit],
                f.panel_train[f.es_fit],
                pw_ft,
                X_es=f.X_train[f.es_hold],
                y_es=f.y_train[f.es_hold],
            )
            ft_epochs.append(int(ft.best_epoch_))
            oof_ft[f.val_idx] = ft.predict_proba(f.X_val)
            logger.info("%s FT inner fold %d: best epoch %d", label, j, ft.best_epoch_)
            del ft

    oof = {"ft": oof_ft, "xgb": xres.oof, "lgb": lres.oof}
    for k, v in oof.items():
        if v is None or not np.isfinite(v).all():
            raise RuntimeError(f"Incomplete inner OOF predictions for {k}")

    # 5) meta learner on genuine OOF base predictions
    meta_X = np.column_stack([oof[k] for k in BASE_MODELS])
    meta = _meta_learner(cfg, seed).fit(meta_X, yc)

    # cross-fitted stacked OOF (meta learner never scores rows it was fitted on)
    stack_oof = np.full(n, np.nan)
    for tr, va in splits:
        m = _meta_learner(cfg, seed).fit(meta_X[tr], yc[tr])
        stack_oof[va] = m.predict_proba(meta_X[va])[:, 1]
    oof["stack"] = stack_oof

    # 6) thresholds from training-side OOF predictions only
    thresholds = {
        k: select_threshold(yc, oof[k], th.beta, th.min_specificity, th.default)[0]
        for k in ALL_MODELS
    }
    calibrator = None
    if cfg.stacking.fit_calibrator:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            stack_oof, yc
        )

    # 7) refit everything on the full training partition
    with timed("full_refit", timings):
        pre_full = FoldPreprocessor(cfg.data, cfg.preprocessing, seed, tda)
        X_full = pre_full.fit_transform(dfc, yc)
        xgb_full = fit_xgb(X_full, yc, xres.params, pw_xgb, derive_seed(seed, 1), n_jobs)
        lgb_full = fit_lgb(X_full, yc, lres.params, pw_lgb, derive_seed(seed, 2), n_jobs)
        n_epochs = max(1, int(np.median(ft_epochs)))
        ft_full = FTTransformerClassifier(
            cfg.ft, len(cfg.data.panels), derive_seed(seed, 99), device
        )
        ft_full.fit(X_full, yc, pic, pw_ft, n_epochs=n_epochs)

    training_oof = pd.DataFrame(
        {"row": keep, "y": yc, "group": gc, "panel": pc, **{f"p_{k}": v for k, v in oof.items()}}
    )
    inner_fold_of = np.full(n, -1)
    for j, (_, va) in enumerate(splits):
        inner_fold_of[va] = j
    training_oof["inner_fold"] = inner_fold_of

    info = {
        "n_input_rows": int(len(df)),
        "n_conflicts_removed": int(conflict.sum()),
        "n_training_rows": int(n),
        "n_groups": int(len(np.unique(gc))),
        "n_inner_splits": int(n_splits),
        "inner_split_sizes": [[int(len(tr)), int(len(va))] for tr, va in splits],
        "n_features_full": int(X_full.shape[1]),
        "ft_tokens": int(len(ft_full.selected_)),
        "pos_weight": {"xgb": pw_xgb, "lgb": pw_lgb, "ft": pw_ft},
        "max_depth_upper": {"xgb": depth_xgb, "lgb": depth_lgb},
        "xgb": {"params": xres.params, "inner_logloss": xres.best_value,
                "trials_complete": xres.n_complete, "trials_pruned": xres.n_pruned},
        "lgb": {"params": lres.params, "inner_logloss": lres.best_value,
                "inner_best_iterations": lres.best_iterations,
                "trials_complete": lres.n_complete, "trials_pruned": lres.n_pruned},
        "ft": {"inner_best_epochs": ft_epochs, "refit_epochs": n_epochs, "device": device},
        "meta_learner": {"features": list(BASE_MODELS),
                         "coef": meta.coef_.ravel().tolist(),
                         "intercept": float(meta.intercept_[0])},
        "thresholds": thresholds,
        "timings_sec": {k: round(v, 2) for k, v in timings.items()},
    }  # fmt: skip
    return StackedPipeline(
        preprocessor=pre_full,
        xgb_model=xgb_full,
        lgb_booster=lgb_full,
        ft_model=ft_full,
        meta_learner=meta,
        calibrator=calibrator,
        threshold=float(thresholds["stack"]),
        base_thresholds={k: float(thresholds[k]) for k in BASE_MODELS},
        info=info,
        training_oof=training_oof,
    )
