"""Nested group-aware cross-validation.

::

    for each outer fold k:
        outer_train, outer_val = group-disjoint split
        pipeline_k = fit_stacked_pipeline(outer_train)      # inner CV lives here
        predictions[outer_val] = pipeline_k.predict(outer_val)
        binary[outer_val] = predictions >= pipeline_k.threshold

``outer_val`` is only ever passed to ``predict``. Its labels are read after
prediction, to store alongside the held-out outputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..data import assert_disjoint_groups
from ..features.tda import TDAFeatureExtractor
from ..models.stacking import ALL_MODELS, fit_stacked_pipeline
from ..utils import derive_seed
from .exploratory import panel_finetune_fold
from .splits import Split, group_kfold_splits, stratification_labels

logger = logging.getLogger("varpath")


@dataclass
class NestedCVResult:
    predictions: pd.DataFrame
    folds: list[dict]
    outer_splits: list[Split]
    exploratory_panel_finetune: list[dict] = field(default_factory=list)

    @property
    def evaluation_fold_thresholds(self) -> dict[int, float]:
        return {f["outer_fold"]: f["thresholds"]["stack"] for f in self.folds}


def make_outer_splits(
    y: np.ndarray, groups: np.ndarray, panels: np.ndarray, cfg: ExperimentConfig
) -> list[Split]:
    strat = stratification_labels(y, panels, cfg.cv.stratify_by)
    return group_kfold_splits(strat, y, groups, cfg.cv.outer_folds, cfg.seed)


def run_outer_fold(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    panels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold: int,
    cfg: ExperimentConfig,
    tda: TDAFeatureExtractor | None = None,
) -> tuple[dict[str, np.ndarray], dict, list[dict]]:
    """Fit on ``train_idx`` only; return held-out probabilities and a fold record."""
    assert_disjoint_groups(groups, train_idx, val_idx)
    start = time.perf_counter()
    seed = derive_seed(cfg.seed, 101, fold)
    pipe = fit_stacked_pipeline(
        df.iloc[train_idx],
        y[train_idx],
        groups[train_idx],
        panels[train_idx],
        cfg,
        seed=seed,
        n_splits=cfg.cv.inner_folds,
        tda_extractor=tda,
        label=f"[outer {fold}]",
    )
    df_val = df.iloc[val_idx]
    X_val = pipe.transform(df_val)
    probs = pipe.predict_matrix(X_val)

    thresholds = {"stack": pipe.threshold, **pipe.base_thresholds}
    record = {
        "outer_fold": int(fold),
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_train_groups": int(len(np.unique(groups[train_idx]))),
        "n_val_groups": int(len(np.unique(groups[val_idx]))),
        "group_intersection": 0,
        "thresholds": thresholds,
        "pipeline": pipe.info,
    }

    exploratory: list[dict] = []
    if cfg.evaluation.panel_finetune:
        keep = pipe.training_oof["row"].to_numpy()
        tr_rows = train_idx[keep]
        X_tr = pipe.transform(df.iloc[tr_rows])
        exploratory = panel_finetune_fold(
            pipe.xgb_model,
            X_tr,
            y[tr_rows],
            panels[tr_rows],
            X_val,
            y[val_idx],
            panels[val_idx],
            cfg.data.panels,
            seed,
            cfg.compute.n_jobs,
        )
        for r in exploratory:
            r["outer_fold"] = int(fold)
    record["elapsed_sec"] = round(time.perf_counter() - start, 1)
    del pipe
    return probs, record, exploratory


def run_nested_cv(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    panels: np.ndarray,
    cfg: ExperimentConfig,
    outer_splits: list[Split] | None = None,
) -> NestedCVResult:
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    panels = np.asarray(panels).astype(str)
    df = df.reset_index(drop=True)
    n = len(df)
    splits = outer_splits or make_outer_splits(y, groups, panels, cfg)
    tda = TDAFeatureExtractor(cfg.preprocessing.tda_n_points, cfg.compute.cache_tda)

    prob_cols = {k: np.full(n, np.nan) for k in (*ALL_MODELS, "stack_recalibrated")}
    thr_cols = {k: np.full(n, np.nan) for k in ALL_MODELS}
    fold_of = np.full(n, -1)
    times_predicted = np.zeros(n, dtype=int)
    folds, exploratory = [], []

    for k, (tr, va) in enumerate(splits):
        logger.info(
            "Outer fold %d/%d: train=%d validation=%d", k + 1, len(splits), len(tr), len(va)
        )
        probs, record, expl = run_outer_fold(df, y, groups, panels, tr, va, k, cfg, tda)
        for name, values in probs.items():
            prob_cols[name][va] = values
        for name in ALL_MODELS:
            thr_cols[name][va] = record["thresholds"][name]
        fold_of[va] = k
        times_predicted[va] += 1
        folds.append(record)
        exploratory.extend(expl)
        logger.info(
            "Outer fold %d done in %.0fs (evaluation threshold %.4f)",
            k + 1, record["elapsed_sec"], record["thresholds"]["stack"],
        )  # fmt: skip

    if not np.all(times_predicted == 1):
        raise RuntimeError("Every row must receive exactly one outer held-out prediction")

    preds = pd.DataFrame(
        {
            "row": np.arange(n),
            "id": df[cfg.data.id_column].astype(str).to_numpy()
            if cfg.data.id_column in df.columns
            else np.arange(n).astype(str),
            "panel": panels,
            "group": groups,
            "outer_fold": fold_of,
            "y": y,
        }
    )
    for name in ALL_MODELS:
        preds[f"p_{name}"] = prob_cols[name]
        preds[f"thr_{name}"] = thr_cols[name]
        preds[f"yhat_{name}"] = (prob_cols[name] >= thr_cols[name]).astype(int)
    if np.isfinite(prob_cols["stack_recalibrated"]).all():
        preds["p_stack_recalibrated"] = prob_cols["stack_recalibrated"]
    return NestedCVResult(preds, folds, splits, exploratory)
