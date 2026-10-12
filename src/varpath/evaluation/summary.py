"""Aggregate evaluation of pooled outer held-out predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..models.stacking import ALL_MODELS
from .calibration import calibration_summary
from .dca import decision_curve
from .metrics import fold_metrics_table, full_metrics, panel_metrics_table
from .statistics import bootstrap_ci, pairwise_comparisons

MODEL_LABELS = {
    "ft": "MTL FT-Transformer",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
    "stack": "Stacking (LR)",
}
COMPARISON_PAIRS = [("xgb", "lgb"), ("xgb", "stack"), ("lgb", "stack"), ("ft", "stack")]


@dataclass
class EvaluationSummary:
    metrics: dict
    fold_table: pd.DataFrame
    panel_table: pd.DataFrame
    comparisons: pd.DataFrame
    dca: pd.DataFrame


def protocol_block(cfg: ExperimentConfig, preds: pd.DataFrame, group_audit: dict | None) -> dict:
    return {
        "profile": cfg.profile,
        "seed": cfg.seed,
        "outer_folds": cfg.cv.outer_folds,
        "inner_folds": cfg.cv.inner_folds,
        "stratify_by": cfg.cv.stratify_by,
        "grouping_strategy": cfg.grouping.strategy,
        "grouping_column": cfg.grouping.column if cfg.grouping.strategy == "column" else None,
        "grouping_entity_level": (group_audit or {}).get("entity_level", cfg.grouping.entity_level),
        "gene_held_out": bool((group_audit or {}).get("gene_held_out_evaluation", False)),
        "n_rows": int(len(preds)),
        "n_groups": int(preds["group"].nunique()),
        "threshold_rule": (
            f"max F{cfg.threshold.beta:g} s.t. specificity >= {cfg.threshold.min_specificity}, "
            "selected on outer-training OOF predictions and applied unchanged to the outer fold"
        ),
        "bootstrap": {
            "iterations": cfg.evaluation.bootstrap_iterations,
            "unit": cfg.evaluation.bootstrap_unit,
            "alpha": cfg.evaluation.ci_alpha,
        },
        "estimate_type": "pooled outer held-out predictions (nested group-aware CV)",
    }


def summarize(
    preds: pd.DataFrame, cfg: ExperimentConfig, group_audit: dict | None = None
) -> EvaluationSummary:
    ev = cfg.evaluation
    y = preds["y"].to_numpy()
    groups = preds["group"].to_numpy() if ev.bootstrap_unit == "group" else None
    overall, point = {}, {}
    for i, m in enumerate(ALL_MODELS):
        s, p = preds[f"p_{m}"].to_numpy(), preds[f"yhat_{m}"].to_numpy()
        point[m] = full_metrics(y, s, p)
        overall[m] = bootstrap_ci(
            y, s, p, groups, ev.bootstrap_iterations, ev.ci_alpha, cfg.seed + i
        )

    comparisons = pd.DataFrame(
        pairwise_comparisons(
            y,
            {m: preds[f"p_{m}"].to_numpy() for m in ALL_MODELS},
            {m: preds[f"yhat_{m}"].to_numpy() for m in ALL_MODELS},
            COMPARISON_PAIRS,
        )
    )

    recal = preds["p_stack_recalibrated"].to_numpy() if "p_stack_recalibrated" in preds else None
    calibration = {
        m: calibration_summary(
            y, preds[f"p_{m}"].to_numpy(), recal if m == "stack" else None, ev.calibration_bins
        )
        for m in ALL_MODELS
    }

    thr_grid = np.linspace(ev.dca_min, ev.dca_max, ev.dca_points)
    dca = decision_curve(y, {m: preds[f"p_{m}"].to_numpy() for m in ALL_MODELS}, thr_grid)

    fold_thr = preds.groupby("outer_fold")["thr_stack"].first()
    metrics = {
        "protocol": protocol_block(cfg, preds, group_audit),
        "primary_model": "stack",
        "held_out": {m: {"point": point[m], "bootstrap": overall[m]} for m in ALL_MODELS},
        "evaluation_fold_thresholds": {int(k): float(v) for k, v in fold_thr.items()},
        "evaluation_threshold_spread": {
            "mean": float(fold_thr.mean()),
            "std": float(fold_thr.std(ddof=0)),
            "min": float(fold_thr.min()),
            "max": float(fold_thr.max()),
        },
        "calibration": calibration,
        "notes": [
            "ROC-AUC and PR-AUC use pooled outer held-out probabilities.",
            "Threshold-dependent metrics use binary predictions produced with each fold's "
            "training-selected threshold; no threshold was tuned on held-out rows.",
            "Recalibrated probabilities come from an isotonic map fitted on training-side "
            "cross-fitted predictions within each outer fold.",
        ],
    }
    return EvaluationSummary(
        metrics=metrics,
        fold_table=fold_metrics_table(preds, list(ALL_MODELS)),
        panel_table=panel_metrics_table(
            preds, list(ALL_MODELS), ev.panel_min_n, ev.panel_exploratory_n
        ),
        comparisons=comparisons,
        dca=dca,
    )
