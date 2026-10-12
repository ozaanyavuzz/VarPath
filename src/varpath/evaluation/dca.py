"""Decision curve analysis (Vickers & Elkin, 2006) on held-out probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    n = y.size
    out = np.empty(len(thresholds))
    for i, pt in enumerate(thresholds):
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        out[i] = tp / n - fp / n * (pt / (1.0 - pt))
    return out


def treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    prev = y.mean()
    t = np.asarray(thresholds, dtype=float)
    return prev - (1.0 - prev) * (t / (1.0 - t))


def decision_curve(
    y_true: np.ndarray, probabilities: dict[str, np.ndarray], thresholds: np.ndarray
) -> pd.DataFrame:
    t = np.asarray(thresholds, dtype=float)
    data = {
        "threshold": t,
        "treat_all": treat_all_net_benefit(y_true, t),
        "treat_none": np.zeros_like(t),
    }
    for name, p in probabilities.items():
        data[name] = net_benefit(y_true, p, t)
    return pd.DataFrame(data)
