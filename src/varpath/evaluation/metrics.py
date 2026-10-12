"""Evaluation metrics on held-out predictions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
)

THRESHOLD_METRICS = (
    "f1",
    "f2",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "mcc",
    "balanced_accuracy",
    "accuracy",
)
RANKING_METRICS = ("roc_auc", "pr_auc", "brier")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Threshold-dependent metrics from fixed binary predictions."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    mcc = matthews_corrcoef(y, p) if len(np.unique(y)) > 1 or len(np.unique(p)) > 1 else 0.0
    return {
        "f1": float(fbeta_score(y, p, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y, p, beta=2, zero_division=0)),
        "sensitivity": sens,
        "specificity": spec,
        "precision": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "mcc": float(mcc),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics; NaN when only one class is present."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    out = {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
    if y.size == 0:
        return out
    out["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1)))
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    return out


def full_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    out = {"n": int(y.size), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
    out.update(ranking_metrics(y, y_score))
    out.update(binary_metrics(y, y_pred))
    return out


def reliability_label(n: int, n_pos: int, n_neg: int, exploratory_n: int) -> str:
    if n_pos == 0 or n_neg == 0:
        return "single-class (AUC not defined)"
    if n < exploratory_n:
        return f"exploratory (n<{exploratory_n})"
    if min(n_pos, n_neg) < 10:
        return "exploratory (<10 in minority class)"
    return "reportable"


def panel_metrics_table(
    preds: pd.DataFrame,
    model_prefixes: list[str],
    min_n: int,
    exploratory_n: int,
) -> pd.DataFrame:
    """Per-panel metrics on outer held-out predictions and fold thresholds."""
    rows = []
    for panel, sub in preds.groupby("panel", sort=False):
        for model in model_prefixes:
            n = len(sub)
            if n < min_n:
                continue
            m = full_metrics(sub["y"], sub[f"p_{model}"], sub[f"yhat_{model}"])
            m.update(
                {
                    "panel": panel,
                    "model": model,
                    "reliability": reliability_label(n, m["n_pos"], m["n_neg"], exploratory_n),
                    "near_perfect_auc_flag": bool(
                        not math.isnan(m["roc_auc"]) and m["roc_auc"] >= 0.99
                    ),
                }
            )
            rows.append(m)
    cols = ["panel", "model", "n", "n_pos", "n_neg", "reliability", "near_perfect_auc_flag"]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[cols + [c for c in df.columns if c not in cols]]


def fold_metrics_table(preds: pd.DataFrame, model_prefixes: list[str]) -> pd.DataFrame:
    rows = []
    for fold, sub in preds.groupby("outer_fold"):
        for model in model_prefixes:
            m = full_metrics(sub["y"], sub[f"p_{model}"], sub[f"yhat_{model}"])
            m.update(
                {
                    "outer_fold": int(fold),
                    "model": model,
                    "threshold": float(sub[f"thr_{model}"].iloc[0]),
                }
            )
            rows.append(m)
    df = pd.DataFrame(rows)
    lead = ["outer_fold", "model", "threshold", "n", "n_pos", "n_neg"]
    return df[lead + [c for c in df.columns if c not in lead]]
