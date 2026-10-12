"""Calibration diagnostics on held-out predictions.

Recalibration maps (isotonic regression) are fitted on training-side
cross-fitted predictions inside each outer fold and applied to that fold's
held-out predictions; they are never fitted on the data they are evaluated on.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(
        np.asarray(y_true).astype(int), np.clip(y_prob, 0, 1), n_bins=n_bins, strategy="uniform"
    )
    return mean_pred, frac_pos


def calibration_summary(
    y_true: np.ndarray, p_raw: np.ndarray, p_recal: np.ndarray | None = None, n_bins: int = 10
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    out = {
        "brier_raw": float(brier_score_loss(y, np.clip(p_raw, 0, 1))),
        "ece_raw": expected_calibration_error(y, p_raw, n_bins),
        "mean_predicted_raw": float(np.mean(p_raw)),
        "observed_prevalence": float(np.mean(y)),
    }
    if p_recal is not None and np.isfinite(p_recal).all():
        out.update(
            {
                "brier_recalibrated": float(brier_score_loss(y, np.clip(p_recal, 0, 1))),
                "ece_recalibrated": expected_calibration_error(y, p_recal, n_bins),
                "mean_predicted_recalibrated": float(np.mean(p_recal)),
            }
        )
    return out
