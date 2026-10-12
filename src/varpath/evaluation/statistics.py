"""Statistical comparison and uncertainty on held-out predictions.

* DeLong test for paired ROC-AUCs (fast O(n log n) formulation of
  Sun & Xu, 2014).
* McNemar test on fixed held-out binary predictions (continuity-corrected
  chi-square, plus an exact binomial p-value).
* Percentile bootstrap. Probabilities and binary predictions are resampled
  as given; no threshold is re-optimised inside a bootstrap replicate. With
  ``groups`` the resampling unit is the group (cluster bootstrap), which
  respects the correlation between rows that share an entity.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .metrics import binary_metrics, ranking_metrics


def _auc_components(pos: np.ndarray, neg: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    m, n = len(pos), len(neg)
    tx = stats.rankdata(pos)
    ty = stats.rankdata(neg)
    tz = stats.rankdata(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    return float(auc), v10, v01


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return {
            "auc_a": float("nan"),
            "auc_b": float("nan"),
            "diff": float("nan"),
            "se": float("nan"),
            "z": float("nan"),
            "p_value": float("nan"),
        }
    auc_a, v10a, v01a = _auc_components(a[pos], a[neg])
    auc_b, v10b, v01b = _auc_components(b[pos], b[neg])
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = auc_a - auc_b
    if var <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "se": 0.0, "z": 0.0, "p_value": 1.0}
    se = math.sqrt(var)
    z = diff / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": diff,
        "se": se,
        "z": float(z),
        "p_value": float(p),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    a = np.asarray(pred_a).astype(int)
    b = np.asarray(pred_b).astype(int)
    n01 = int(np.sum((a != y) & (b == y)))  # A wrong, B right
    n10 = int(np.sum((a == y) & (b != y)))  # A right, B wrong
    disc = n01 + n10
    if disc == 0:
        return {
            "n_a_wrong_b_right": n01,
            "n_a_right_b_wrong": n10,
            "chi2": 0.0,
            "p_value_chi2": 1.0,
            "p_value_exact": 1.0,
        }
    chi2 = (abs(n01 - n10) - 1) ** 2 / disc
    p_chi2 = float(1.0 - stats.chi2.cdf(chi2, df=1))
    p_exact = float(stats.binomtest(min(n01, n10), disc, 0.5).pvalue)
    return {
        "n_a_wrong_b_right": n01,
        "n_a_right_b_wrong": n10,
        "chi2": float(chi2),
        "p_value_chi2": p_chi2,
        "p_value_exact": p_exact,
    }


def _fast_metrics(y: np.ndarray, s: np.ndarray, p: np.ndarray) -> dict[str, float]:
    rk = ranking_metrics(y, s)
    bm = binary_metrics(y, p)
    return {
        "roc_auc": rk["roc_auc"],
        "pr_auc": rk["pr_auc"],
        "f2": bm["f2"],
        "mcc": bm["mcc"],
        "sensitivity": bm["sensitivity"],
        "specificity": bm["specificity"],
        "balanced_accuracy": bm["balanced_accuracy"],
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Point estimates on the full held-out set with percentile bootstrap CIs."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    p = np.asarray(y_pred).astype(int)
    rng = np.random.default_rng(seed)
    point = _fast_metrics(y, s, p)
    if groups is not None:
        codes, uniq_idx = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        order = np.argsort(uniq_idx, kind="stable")
        bounds = np.searchsorted(uniq_idx[order], np.arange(len(codes) + 1))
        members = [order[bounds[i] : bounds[i + 1]] for i in range(len(codes))]
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(int(n_boot)):
        if groups is None:
            idx = rng.integers(0, len(y), size=len(y))
        else:
            pick = rng.integers(0, len(members), size=len(members))
            idx = np.concatenate([members[i] for i in pick])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        for k, v in _fast_metrics(yb, s[idx], p[idx]).items():
            samples[k].append(v)
    out = {}
    for k, est in point.items():
        arr = np.asarray(samples[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[k] = {
                "estimate": est,
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_boot_valid": 0,
            }
            continue
        out[k] = {
            "estimate": float(est),
            "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "n_boot_valid": int(arr.size),
        }
    return out


def pairwise_comparisons(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
) -> list[dict]:
    """DeLong + McNemar for each pair, with a Bonferroni-adjusted alpha."""
    alpha = 0.05 / max(1, len(pairs))
    rows = []
    for a, b in pairs:
        d = delong_test(y_true, scores[a], scores[b])
        mc = mcnemar_test(y_true, preds[a], preds[b])
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                **{f"delong_{k}": v for k, v in d.items()},
                **{f"mcnemar_{k}": v for k, v in mc.items()},
                "bonferroni_alpha": alpha,
                "delong_significant": bool(np.isfinite(d["p_value"]) and d["p_value"] < alpha),
                "mcnemar_significant": bool(mc["p_value_chi2"] < alpha),
            }
        )
    return rows
