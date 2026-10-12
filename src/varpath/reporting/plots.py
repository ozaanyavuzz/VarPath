"""Figures produced from held-out predictions and the deployment model."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..evaluation.calibration import reliability_curve  # noqa: E402
from ..evaluation.thresholding import fbeta_threshold_scan  # noqa: E402

logger = logging.getLogger("varpath")
COLORS = {"ft": "#8172B3", "xgb": "#1f77b4", "lgb": "#ff7f0e", "stack": "#2ca02c"}
LABELS = {"ft": "FT-Transformer", "xgb": "XGBoost", "lgb": "LightGBM", "stack": "Stacking"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_pr(preds: pd.DataFrame, models: list[str], path: Path, subtitle: str = "") -> Path:
    y = preds["y"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in models:
        p = preds[f"p_{m}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(
            fpr,
            tpr,
            lw=2,
            color=COLORS.get(m),
            label=f"{LABELS.get(m, m)} AUC={roc_auc_score(y, p):.3f}",
        )
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(
            rec,
            prec,
            lw=2,
            color=COLORS.get(m),
            label=f"{LABELS.get(m, m)} AP={average_precision_score(y, p):.3f}",
        )
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].axhline(y.mean(), color="gray", lw=1, ls="--", label=f"prevalence={y.mean():.2f}")
    axes[0].set(
        title="ROC (outer held-out)", xlabel="False positive rate", ylabel="True positive rate"
    )
    axes[1].set(title="Precision-recall (outer held-out)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    if subtitle:
        fig.suptitle(subtitle, fontsize=9)
    return _save(fig, path)


def plot_calibration(preds: pd.DataFrame, models: list[str], path: Path, n_bins: int = 10) -> Path:
    y = preds["y"].to_numpy()
    series = [(m, preds[f"p_{m}"].to_numpy()) for m in models]
    if "p_stack_recalibrated" in preds:
        series.append(("stack_recalibrated", preds["p_stack_recalibrated"].to_numpy()))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in series:
        mp, fp = reliability_curve(y, p, n_bins)
        ax.plot(mp, fp, "o-", lw=2, color=COLORS.get(name, "#555555"),
                label=LABELS.get(name, "Stacking (train-side isotonic)"))  # fmt: skip
    ax.set(
        title="Calibration (outer held-out)",
        xlabel="Mean predicted probability",
        ylabel="Observed frequency",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_threshold_analysis(preds: pd.DataFrame, path: Path, beta: float = 2.0) -> Path:
    """Held-out metric curves across thresholds with the fold-selected thresholds marked.

    The curves are descriptive; the marked thresholds were chosen on
    training-side predictions, not from these curves.
    """
    y = preds["y"].to_numpy()
    p = preds["p_stack"].to_numpy()
    grid = np.linspace(0.01, 0.99, 197)
    scan = fbeta_threshold_scan(y, p, grid, beta)
    tp, fp, fn, tn = (scan[k].astype(float) for k in ("tp", "fp", "fn", "tn"))
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid, f1, lw=2, label="F1")
    ax.plot(grid, scan["fbeta"], lw=2, ls="--", label=f"F{beta:g}")
    ax.plot(grid, recall, lw=2, ls=":", label="Sensitivity")
    ax.plot(grid, scan["specificity"], lw=2, ls="-.", label="Specificity")
    for i, t in enumerate(sorted(preds.groupby("outer_fold")["thr_stack"].first())):
        ax.axvline(
            t,
            color="red",
            lw=1,
            alpha=0.6,
            label="fold thresholds (training-selected)" if i == 0 else None,
        )
    ax.set(
        title="Threshold analysis of stacked held-out scores", xlabel="Threshold", ylabel="Score"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_panel_comparison(panel_table: pd.DataFrame, path: Path, model: str = "stack") -> Path:
    sub = panel_table[panel_table["model"] == model].reset_index(drop=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric, title in zip(
        axes, ["roc_auc", "pr_auc", "f2", "mcc"], ["ROC-AUC", "PR-AUC", "F2", "MCC"]
    ):
        vals = sub[metric].astype(float).fillna(0).to_numpy()
        colors = [
            "#bbbbbb" if not str(r).startswith("reportable") else "#4C72B0"
            for r in sub["reliability"]
        ]
        ax.bar(sub["panel"], vals, color=colors)
        for i, (v, n) in enumerate(zip(sub[metric], sub["n"])):
            txt = "NA" if pd.isna(v) else f"{v:.3f}"
            ax.text(i, (0 if pd.isna(v) else v) + 0.02, f"{txt}\nn={n}", ha="center", fontsize=8)
        ax.set_ylim(min(0, np.nanmin(vals) - 0.05), 1.15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-panel metrics on outer held-out predictions (grey = exploratory / small n)")
    return _save(fig, path)


def plot_dca(dca: pd.DataFrame, models: list[str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    t = dca["threshold"]
    ax.plot(t, dca["treat_all"], "k--", lw=1.5, label="Treat all as pathogenic")
    ax.plot(t, dca["treat_none"], color="gray", lw=1, label="Treat none")
    for m in models:
        ax.plot(t, dca[m], lw=2.2, color=COLORS.get(m), label=LABELS.get(m, m))
    lo = min(dca[models].min().min(), 0) - 0.05
    ax.set_ylim(lo, max(dca["treat_all"].max(), dca[models].max().max()) + 0.05)
    ax.set(
        title="Decision curve analysis (outer held-out)",
        xlabel="Threshold probability",
        ylabel="Net benefit",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    path: Path,
    max_display: int = 20,
) -> Path:
    if shap_values.shape[1] != len(feature_names) or X.shape[1] != len(feature_names):
        raise ValueError("SHAP matrix, data matrix and feature names must have the same width")
    import shap

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, show=False, max_display=max_display
    )
    plt.title("Deployment XGBoost base learner: TreeSHAP summary", fontsize=10)
    fig = plt.gcf()
    return _save(fig, path)


def plot_variant_network(
    preds: pd.DataFrame,
    aa1: np.ndarray,
    aa2: np.ndarray,
    path: Path,
    n_sample: int = 200,
    seed: int = 42,
) -> Path | None:
    """Exploratory: variants linked when same panel and Grantham distance differs by < 20."""
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed; variant network skipped")
        return None
    from ..features.amino_acid import grantham_distance

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(preds), min(n_sample, len(preds)), replace=False)
    sub = preds.iloc[idx].reset_index(drop=True)
    gr = np.array([grantham_distance(a, b) for a, b in zip(aa1[idx], aa2[idx])])
    G = nx.Graph()
    for i, row in sub.iterrows():
        G.add_node(i, panel=row["panel"], p=row["p_stack"])
    pans = sub["panel"].to_numpy()
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            if pans[i] == pans[j] and abs(gr[i] - gr[j]) < 20:
                G.add_edge(i, j)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    uniq = sorted(set(pans))
    cmap = {p: palette[i % len(palette)] for i, p in enumerate(uniq)}
    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(G, k=0.5, seed=seed)
    nx.draw_networkx_nodes(G, pos, node_color=[cmap[G.nodes[n]["panel"]] for n in G.nodes],
                           node_size=[150 + 300 * G.nodes[n]["p"] for n in G.nodes], alpha=0.7, ax=ax)  # fmt: skip
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="gray", ax=ax)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=p.upper()) for p, c in cmap.items()], fontsize=8)
    ax.set_title(
        "Exploratory variant similarity network\n(node size ~ held-out stacked score; colour = panel)"
    )
    ax.axis("off")
    return _save(fig, path)
