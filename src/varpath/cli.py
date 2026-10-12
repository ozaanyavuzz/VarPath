"""Command-line entry points (also used by ``scripts/``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _parse_set(values: list[str] | None) -> dict:
    """Turn ``a.b=1`` overrides into a nested dict (values parsed as YAML)."""
    out: dict = {}
    for item in values or []:
        key, _, raw = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def train_main(argv: list[str] | None = None) -> int:
    from .config import load_config
    from .pipeline import run_experiment

    ap = argparse.ArgumentParser(description="Nested group-aware CV evaluation + deployment model.")
    ap.add_argument("--config", default="configs/standard.yaml")
    ap.add_argument("--data-dir", default=None, help="directory with the panel CSV files")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--skip-deployment", action="store_true", help="only run the nested evaluation")
    ap.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="config override, e.g. cv.outer_folds=3"
    )
    args = ap.parse_args(argv)
    cfg = load_config(args.config, _parse_set(args.set))
    run_experiment(
        cfg, args.data_dir, args.output_dir, args.artifacts_dir, not args.skip_deployment
    )
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    from .pipeline import evaluate_outputs

    ap = argparse.ArgumentParser(
        description="Recompute metrics/figures from saved held-out predictions."
    )
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    metrics = evaluate_outputs(args.output_dir)
    stack = metrics["held_out"]["stack"]["bootstrap"]
    for k in ("roc_auc", "pr_auc", "f2", "mcc"):
        v = stack[k]
        print(f"{k:>8}: {v['estimate']:.4f}  [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from . import RESEARCH_DISCLAIMER
    from .artifacts import load_artifacts
    from .data import read_csv
    from .reporting.report import variant_report_pdf

    ap = argparse.ArgumentParser(description="Score a CSV with a persisted deployment model.")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--explain", type=int, default=0, help="add top-k TreeSHAP features per row")
    ap.add_argument("--report-dir", default=None, help="write one research PDF per row")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    predictor = load_artifacts(args.artifacts, device=args.device)
    df = read_csv(args.input)
    missing = predictor.validate_input(df)
    if missing:
        print(
            f"Input is missing {len(missing)} required columns, e.g. {missing[:5]}", file=sys.stderr
        )
        return 2
    res = predictor.predict(df)
    need_expl = args.explain or args.report_dir
    expl = predictor.explain(df, top_k=max(args.explain, 5)) if need_expl else None
    if args.explain:
        res["top_features"] = [
            "; ".join(f"{n}({c:+.3f})" for n, c, _ in row[: args.explain]) for row in expl
        ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)
    if args.report_dir:
        aa_ref = predictor.preprocessor.data_cfg.aa_ref_column
        aa_alt = predictor.preprocessor.data_cfg.aa_alt_column
        for i, row in res.iterrows():
            variant_report_pdf(
                Path(args.report_dir) / f"report_{row['id']}.pdf",
                row["id"],
                str(row.get("panel", "unknown")),
                float(row["score"]),
                predictor.deployment_threshold,
                str(df[aa_ref].iloc[i]).strip().upper() if pd.notna(df[aa_ref].iloc[i]) else "X",
                str(df[aa_alt].iloc[i]).strip().upper() if pd.notna(df[aa_alt].iloc[i]) else "X",
                expl[i],
            )
    print(f"Wrote {len(res)} predictions to {args.output}")
    print(RESEARCH_DISCLAIMER)
    return 0
