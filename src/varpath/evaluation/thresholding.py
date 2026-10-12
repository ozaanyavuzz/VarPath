"""Decision-threshold selection.

The rule is unchanged from the original prototype: among the distinct score
values returned by ``precision_recall_curve``, choose the smallest threshold
maximising F-beta (beta=2) subject to specificity >= ``min_specificity``.

What changed is *where* it is called. In this repository the function only
receives training-side out-of-fold predictions:

* ``evaluation_fold_thresholds`` – one per outer fold, selected from that
  fold's inner OOF predictions and then applied unchanged to the outer
  validation fold;
* ``deployment_threshold`` – selected from the cross-fitted OOF predictions of
  the final model trained on all data. It is never used to report
  performance.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger("varpath")


def fbeta_threshold_scan(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray, beta: float = 2.0
) -> dict[str, np.ndarray]:
    """Confusion counts and F-beta for ``pred = score >= t`` at every threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    all_sorted = np.sort(s)
    pos_sorted = np.sort(s[y == 1])
    n_pos = pos_sorted.size
    n_neg = y.size - n_pos
    pred_pos = y.size - np.searchsorted(all_sorted, t, side="left")
    tp = n_pos - np.searchsorted(pos_sorted, t, side="left")
    fp = pred_pos - tp
    fn = n_pos - tp
    tn = n_neg - fp
    b2 = beta**2
    denom = (1 + b2) * tp + b2 * fn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denom > 0, (1 + b2) * tp / denom, 0.0)
    spec = tn / (tn + fp + 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "fbeta": fbeta, "specificity": spec}


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = 2.0,
    min_specificity: float = 0.60,
    default: float = 0.5,
) -> tuple[float, float]:
    """Return ``(threshold, fbeta_at_threshold)``; call with training-side data only."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    if y.size == 0 or len(np.unique(y)) < 2:
        logger.warning("Threshold selection needs both classes; using default %.3f", default)
        return float(default), 0.0
    _, _, thr = precision_recall_curve(y, s)
    if thr.size == 0:
        return float(default), 0.0
    scan = fbeta_threshold_scan(y, s, thr, beta)
    valid = (scan["specificity"] >= min_specificity) & (scan["fbeta"] > 0)
    if not valid.any():
        return float(default), 0.0
    fb = np.where(valid, scan["fbeta"], -np.inf)
    best = int(np.argmax(fb))  # first maximum in ascending threshold order
    return float(thr[best]), float(fb[best])


def apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_score, dtype=float) >= float(threshold)).astype(int)
