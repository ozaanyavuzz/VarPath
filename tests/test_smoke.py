"""Tiny end-to-end run: data loading, nested CV, outputs, deployment artifacts, prediction."""

import json

import numpy as np
import pandas as pd

from varpath.artifacts import load_artifacts
from varpath.config import load_config
from varpath.pipeline import OUTPUT_FILES, evaluate_outputs, run_experiment
from varpath.synthetic import write_synthetic_panels

from .conftest import REPO_ROOT, SIZES, TINY


def test_quick_pipeline_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    cfg = load_config(
        REPO_ROOT / "configs" / "quick.yaml",
        overrides={k: v for k, v in TINY.items() if k != "profile"},
    )
    write_synthetic_panels(
        data_dir, cfg.data.panel_files, sizes=SIZES, n_al=12, n_ek=2, n_cat=3, seed=5
    )
    out, art = tmp_path / "out", tmp_path / "art"
    res = run_experiment(cfg, data_dir=data_dir, output_dir=out, artifacts_dir=art)

    for name in (
        "metrics",
        "fold_metrics",
        "panel_metrics",
        "thresholds",
        "config",
        "group_audit",
        "predictions",
    ):
        assert (out / OUTPUT_FILES[name]).exists(), name
    for fig in (
        "roc_pr",
        "calibration",
        "threshold_analysis",
        "panel_comparison",
        "decision_curve",
        "shap_summary",
    ):
        assert (out / "figures" / f"{fig}.png").exists(), fig

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["run"] == {"profile": "quick", "seed": cfg.seed}
    assert metrics["protocol"]["gene_held_out"] is False
    assert 0 <= metrics["held_out"]["stack"]["point"]["roc_auc"] <= 1

    thresholds = json.loads((out / "thresholds.json").read_text())
    assert set(thresholds["evaluation_fold_thresholds"]) == {"0", "1", "2"}
    assert thresholds["deployment_threshold"] is not None

    audit = json.loads((out / "group_audit.json").read_text())
    assert all(f["train_val_group_intersection"] == 0 for f in audit["outer_folds"])

    preds = pd.read_csv(out / "oof_predictions.csv")
    assert len(preds) == preds["row"].nunique() == audit["n_rows"]

    predictor = load_artifacts(art)
    sample = pd.read_csv(data_dir / cfg.data.panel_files["pah"]).head(4)
    scored = predictor.predict(sample)
    assert np.all((scored["score"] >= 0) & (scored["score"] <= 1))
    assert (scored["deployment_threshold"] == res["thresholds"]["deployment_threshold"]).all()

    again = evaluate_outputs(out)
    for k in ("roc_auc", "pr_auc", "f2", "mcc"):
        assert (
            abs(again["held_out"]["stack"]["point"][k] - metrics["held_out"]["stack"]["point"][k])
            < 1e-9
        )
