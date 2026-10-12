"""End-to-end orchestration: nested evaluation, outputs, deployment model."""

from __future__ import annotations

import datetime as _dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import save_artifacts
from .config import ExperimentConfig, config_from_dict, save_config
from .data import build_groups, group_audit, load_panel_data
from .evaluation.nested_cv import run_nested_cv
from .evaluation.summary import summarize
from .features.amino_acid import normalize_residue_codes
from .features.tda import TDAFeatureExtractor
from .models.stacking import ALL_MODELS, fit_stacked_pipeline
from .reporting import plots
from .reporting.report import read_json, write_csv, write_json
from .utils import (
    configure_torch_threads,
    git_commit,
    library_versions,
    set_global_seed,
    setup_logging,
)

logger = logging.getLogger("varpath")

OUTPUT_FILES = {
    "metrics": "metrics.json",
    "fold_metrics": "fold_metrics.csv",
    "panel_metrics": "panel_metrics.csv",
    "thresholds": "thresholds.json",
    "config": "config_used.yaml",
    "group_audit": "group_audit.json",
    "predictions": "oof_predictions.csv",
    "comparisons": "statistical_tests.csv",
    "dca": "decision_curve.csv",
    "folds": "fold_details.json",
    "exploratory": "exploratory_panel_finetune.csv",
    "run": "run_info.json",
}


def run_info(cfg: ExperimentConfig, started: float, extra: dict | None = None) -> dict:
    info = {
        "profile": cfg.profile,
        "seed": cfg.seed,
        "started_utc": _dt.datetime.fromtimestamp(started, _dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "elapsed_sec": round(time.time() - started, 1),
        "git_commit": git_commit(),
        "library_versions": library_versions(),
    }
    info.update(extra or {})
    return info


def write_evaluation(
    preds: pd.DataFrame,
    cfg: ExperimentConfig,
    out: Path,
    audit: dict | None,
    aa: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Compute summaries from held-out predictions and write tables and figures."""
    summary = summarize(preds, cfg, audit)
    run_tag = {"profile": cfg.profile, "seed": cfg.seed}
    summary.metrics["run"] = run_tag
    write_json(summary.metrics, out / OUTPUT_FILES["metrics"])
    for df, key in (
        (summary.fold_table, "fold_metrics"),
        (summary.panel_table, "panel_metrics"),
        (summary.comparisons, "comparisons"),
        (summary.dca, "dca"),
    ):
        df = df.copy()
        df["profile"], df["seed"] = cfg.profile, cfg.seed
        write_csv(df, out / OUTPUT_FILES[key])

    fig = out / "figures"
    models = list(ALL_MODELS)
    sub = f"profile={cfg.profile} seed={cfg.seed} | {cfg.cv.outer_folds}x{cfg.cv.inner_folds} nested group CV"
    plots.plot_roc_pr(preds, models, fig / "roc_pr.png", sub)
    plots.plot_calibration(preds, models, fig / "calibration.png", cfg.evaluation.calibration_bins)
    plots.plot_threshold_analysis(preds, fig / "threshold_analysis.png", cfg.threshold.beta)
    if not summary.panel_table.empty:
        plots.plot_panel_comparison(summary.panel_table, fig / "panel_comparison.png")
    plots.plot_dca(summary.dca, models, fig / "decision_curve.png")
    if cfg.evaluation.variant_network and aa is not None:
        plots.plot_variant_network(
            preds, aa[0], aa[1], fig / "variant_network_exploratory.png", seed=cfg.seed
        )
    return summary.metrics


def run_experiment(
    cfg: ExperimentConfig,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    fit_deployment: bool = True,
    df: pd.DataFrame | None = None,
) -> dict:
    setup_logging()
    started = time.time()
    set_global_seed(cfg.seed)
    configure_torch_threads(cfg.compute.torch_threads)
    out = Path(output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / OUTPUT_FILES["config"])

    if df is None:
        df = load_panel_data(cfg.data, data_dir)
    df = df.reset_index(drop=True)
    y = df[cfg.data.label_column].astype(int).to_numpy()
    panels = df[cfg.data.panel_column].astype(str).to_numpy()
    groups = build_groups(df, cfg.grouping, cfg.data)
    audit = group_audit(df, groups, cfg.grouping, cfg.data)
    if not audit["gene_or_protein_or_transcript_identifier_available"]:
        logger.warning(
            "No gene/protein/transcript identifier found; evaluation is %s-level held-out, not gene-held-out.",
            audit["entity_level"],
        )

    nested = run_nested_cv(df, y, groups, panels, cfg)
    audit["outer_folds"] = [
        {
            "outer_fold": f["outer_fold"],
            "n_train_groups": f["n_train_groups"],
            "n_val_groups": f["n_val_groups"],
            "train_val_group_intersection": f["group_intersection"],
        }
        for f in nested.folds
    ]
    write_json(audit, out / OUTPUT_FILES["group_audit"])
    write_csv(nested.predictions, out / OUTPUT_FILES["predictions"])
    write_json(nested.folds, out / OUTPUT_FILES["folds"])
    if nested.exploratory_panel_finetune:
        write_csv(
            pd.DataFrame(nested.exploratory_panel_finetune), out / OUTPUT_FILES["exploratory"]
        )

    aa = (
        normalize_residue_codes(df[cfg.data.aa_ref_column]),
        normalize_residue_codes(df[cfg.data.aa_alt_column]),
    )
    metrics = write_evaluation(nested.predictions, cfg, out, audit, aa)

    thresholds = {
        "evaluation_fold_thresholds": {str(f["outer_fold"]): f["thresholds"] for f in nested.folds},
        "deployment_threshold": None,
        "note": "Evaluation thresholds were each selected on outer-training OOF predictions. "
        "The deployment threshold is selected on cross-fitted OOF predictions of the full "
        "training set and is not used to report performance.",
        "profile": cfg.profile,
        "seed": cfg.seed,
    }

    if fit_deployment:
        logger.info("Fitting deployment model on all %d rows", len(df))
        tda = TDAFeatureExtractor(cfg.preprocessing.tda_n_points, cfg.compute.cache_tda)
        pipe = fit_stacked_pipeline(
            df, y, groups, panels, cfg, seed=cfg.seed, n_splits=cfg.cv.inner_folds,
            tda_extractor=tda, label="[deployment]",
        )  # fmt: skip
        thresholds["deployment_threshold"] = pipe.threshold
        thresholds["deployment_base_thresholds"] = pipe.base_thresholds
        art = Path(artifacts_dir or cfg.artifacts_dir)
        save_artifacts(pipe, cfg, art, audit, extra={"output_dir": str(out)})
        write_csv(pipe.training_oof, out / "deployment_training_oof.csv")
        if cfg.shap.enabled:
            write_shap_outputs(pipe, df, cfg, out)
    write_json(thresholds, out / OUTPUT_FILES["thresholds"])
    info = run_info(cfg, started, {"fit_deployment": fit_deployment, "n_rows": int(len(df))})
    write_json(info, out / OUTPUT_FILES["run"])
    logger.info("Finished in %.0fs; outputs in %s", info["elapsed_sec"], out)
    return {"metrics": metrics, "thresholds": thresholds, "output_dir": str(out)}


def write_shap_outputs(pipe, df: pd.DataFrame, cfg: ExperimentConfig, out: Path) -> None:
    """TreeSHAP of the deployment XGBoost on a random sample of training rows."""
    from .evaluation.exploratory import xgb_contributions

    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(df), min(cfg.shap.n_samples, len(df)), replace=False)
    X = pipe.transform(df.iloc[idx])
    names = pipe.preprocessor.feature_names_out_
    if len(names) != X.shape[1]:
        raise RuntimeError("Feature names do not match the transformed matrix")
    sv = xgb_contributions(pipe.xgb_model, X)[:, :-1]
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    write_csv(imp, out / "shap_importance.csv")
    plots.plot_shap_summary(
        sv, X, names, out / "figures" / "shap_summary.png", cfg.shap.max_display
    )


def evaluate_outputs(output_dir: str | Path) -> dict:
    """Recompute summaries and figures from saved held-out predictions (no training)."""
    setup_logging()
    out = Path(output_dir)
    import yaml

    with open(out / OUTPUT_FILES["config"], encoding="utf-8") as fh:
        cfg = config_from_dict(yaml.safe_load(fh))
    preds = pd.read_csv(
        out / OUTPUT_FILES["predictions"], dtype={"id": str, "group": str, "panel": str}
    )
    audit_path = out / OUTPUT_FILES["group_audit"]
    audit = read_json(audit_path) if audit_path.exists() else None
    return write_evaluation(preds, cfg, out, audit)
