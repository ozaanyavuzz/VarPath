"""Gradio demo for a persisted VarPath deployment model.

Run with ``python app.py --artifacts artifacts/standard``. The demo loads the
model from disk only; it does not depend on any notebook state.

Two tabs:

* **Full-model prediction** – upload a CSV that follows the training schema
  (see ``examples/DATA_SCHEMA.md``); every row is scored by the complete
  stacked pipeline with the persisted deployment threshold.
* **Amino-acid substitution explorer** – descriptors of a residue change.
  This tab is illustrative only and does not produce a model prediction,
  because the model needs the full annotation vector.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from varpath import RESEARCH_DISCLAIMER  # noqa: E402
from varpath.features.amino_acid import (  # noqa: E402
    AA_PROPS,
    BLOSUM62,
    PROP_NAMES,
    STANDARD_AMINO_ACIDS,
    grantham_distance,
)
from varpath.features.structural import AA_STRUCT, STRUCT_NAMES, structural_disruption  # noqa: E402
from varpath.reporting.report import PANEL_CONTEXT, variant_report_pdf  # noqa: E402

AA_NAMES = {
    "A": "Alanine", "C": "Cysteine", "D": "Aspartate", "E": "Glutamate", "F": "Phenylalanine",
    "G": "Glycine", "H": "Histidine", "I": "Isoleucine", "K": "Lysine", "L": "Leucine",
    "M": "Methionine", "N": "Asparagine", "P": "Proline", "Q": "Glutamine", "R": "Arginine",
    "S": "Serine", "T": "Threonine", "V": "Valine", "W": "Tryptophan", "Y": "Tyrosine",
}  # fmt: skip
EXAMPLE_CSV = Path(__file__).resolve().parent / "examples" / "synthetic_sample.csv"


def score_dataframe(predictor, df: pd.DataFrame, make_pdf: bool = False):
    """Score rows with the full pipeline; return (table, markdown, optional pdf path)."""
    missing = predictor.validate_input(df)
    if missing:
        msg = (
            f"**Input rejected.** {len(missing)} required column(s) are missing "
            f"(e.g. `{', '.join(missing[:5])}`). The model needs the complete training schema; "
            "missing columns are not filled in silently."
        )
        return pd.DataFrame(), msg, None
    res = predictor.predict(df)
    expl = predictor.explain(df.head(1), top_k=5)[0]
    thr = predictor.deployment_threshold
    first = res.iloc[0]
    lines = [
        f"> **{RESEARCH_DISCLAIMER}**",
        "",
        f"Scored **{len(res)}** row(s) with the stacked ensemble. "
        f"Deployment threshold: **{thr:.3f}** (selected on training-side cross-fitted predictions).",
        "",
        f"### First row: `{first['id']}`",
        f"- Stacked score: **{first['score']:.3f}** "
        f"({'above' if first['score'] >= thr else 'below'} threshold)",
        f"- Base learners: FT {first['p_ft']:.3f} · XGB {first['p_xgb']:.3f} · LGB {first['p_lgb']:.3f}",
        "",
        "**Largest TreeSHAP contributions (deployment XGBoost base learner):**",
    ]
    lines += [f"- `{n}` = {v:.3g} → {c:+.3f}" for n, c, v in expl]
    panel = str(first.get("panel", ""))
    if panel in PANEL_CONTEXT:
        lines += ["", "**Panel-related contextual information:** " + PANEL_CONTEXT[panel]]
    pdf_path = None
    if make_pdf:
        cfg = predictor.preprocessor.data_cfg
        row = df.iloc[0]
        aa = [
            str(row[c]).strip().upper() if pd.notna(row[c]) else "X"
            for c in (cfg.aa_ref_column, cfg.aa_alt_column)
        ]
        pdf_path = str(Path(tempfile.mkdtemp()) / f"report_{first['id']}.pdf")
        variant_report_pdf(
            pdf_path,
            first["id"],
            panel or "unknown",
            float(first["score"]),
            thr,
            aa[0],
            aa[1],
            expl,
        )
    return res, "\n".join(lines), pdf_path


def describe_substitution(aa1: str, aa2: str) -> str:
    aa1, aa2 = aa1[:1].upper(), aa2[:1].upper()
    header = (
        "> **Illustrative mode.** This mode is illustrative only and does not represent a full "
        "model prediction.\n\n"
    )
    if aa1 == aa2:
        return header + "Reference and alternative residues are identical."
    rows = [
        "| Property | " + f"{aa1} ({AA_NAMES[aa1]}) | {aa2} ({AA_NAMES[aa2]}) | Δ |",
        "|---|---|---|---|",
    ]
    for i, name in enumerate(PROP_NAMES):
        a, b = AA_PROPS[aa1][i], AA_PROPS[aa2][i]
        rows.append(f"| {name} | {a:g} | {b:g} | {b - a:+.2f} |")
    for i, name in enumerate(STRUCT_NAMES):
        a, b = AA_STRUCT[aa1][i], AA_STRUCT[aa2][i]
        rows.append(f"| {name} | {a:g} | {b:g} | {b - a:+.2f} |")
    summary = (
        f"\n\n**Grantham distance:** {grantham_distance(aa1, aa2):.0f} · "
        f"**BLOSUM62:** {BLOSUM62[(aa1, aa2)]} · "
        f"**Structural propensity change:** {structural_disruption(aa1, aa2):.2f}\n\n"
        "These are descriptors of the residue change only; this is not a model prediction."
    )
    return header + "\n".join(rows) + summary


def build_app(predictor):
    import gradio as gr

    choices = [f"{k} — {AA_NAMES[k]}" for k in STANDARD_AMINO_ACIDS]

    def on_predict(file, make_pdf):
        if predictor is None:
            return (
                pd.DataFrame(),
                "No model artifacts loaded. Train a model first (see README).",
                None,
            )
        if file is None:
            return pd.DataFrame(), "Upload a CSV file.", None
        path = file if isinstance(file, str) else file.name
        return score_dataframe(predictor, pd.read_csv(path, low_memory=False), make_pdf)

    with gr.Blocks(title="VarPath — research demo") as demo:
        gr.Markdown(
            f"# VarPath — missense variant pathogenicity (research demo)\n**{RESEARCH_DISCLAIMER}**"
        )
        with gr.Tab("Full-model prediction (CSV)"):
            gr.Markdown(
                "Upload rows that follow the training schema. "
                + (
                    f"An artificial example is provided in `{EXAMPLE_CSV.name}`."
                    if EXAMPLE_CSV.exists()
                    else ""
                )
            )
            with gr.Row():
                inp = gr.File(label="CSV", file_types=[".csv"])
                pdf = gr.Checkbox(label="Research PDF for the first row", value=False)
            btn = gr.Button("Score", variant="primary")
            out_md = gr.Markdown()
            out_tab = gr.Dataframe(label="Predictions")
            out_pdf = gr.File(label="PDF")
            btn.click(on_predict, [inp, pdf], [out_tab, out_md, out_pdf])
            if EXAMPLE_CSV.exists():
                gr.Examples([[str(EXAMPLE_CSV), False]], [inp, pdf])
        with gr.Tab("Amino-acid substitution explorer (illustrative)"):
            a1 = gr.Dropdown(
                choices, value=choices[STANDARD_AMINO_ACIDS.index("R")], label="Reference residue"
            )
            a2 = gr.Dropdown(
                choices, value=choices[STANDARD_AMINO_ACIDS.index("W")], label="Alternative residue"
            )
            md = gr.Markdown()
            gr.Button("Describe").click(
                lambda x, y: describe_substitution(x[0], y[0]), [a1, a2], md
            )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--artifacts", default=os.environ.get("VARPATH_ARTIFACTS", "artifacts/standard")
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    predictor = None
    if (Path(args.artifacts) / "manifest.json").exists():
        from varpath.artifacts import load_artifacts

        predictor = load_artifacts(args.artifacts)
    else:
        print(f"No artifacts found at {args.artifacts}; the prediction tab will be disabled.")
    build_app(predictor).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
