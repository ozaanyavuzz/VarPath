"""Machine-readable outputs and a research-only per-variant PDF summary."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .. import RESEARCH_DISCLAIMER
from ..features.amino_acid import grantham_distance
from ..features.structural import structural_disruption

# Descriptive context only. No diagnostic or therapeutic recommendation is implied.
PANEL_CONTEXT = {
    "master": "Multi-gene reference panel provided with the dataset.",
    "kanser": "Panel labelled 'cancer' in the dataset (hereditary cancer predisposition context).",
    "pah": "Panel labelled 'PAH'; the PAH gene is associated in the literature with phenylketonuria.",
    "cftr": "Panel labelled 'CFTR'; the CFTR gene is associated in the literature with cystic fibrosis.",
}


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2, ensure_ascii=False)
    return path


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _ascii(text: str) -> str:
    table = str.maketrans("çğıöşüÇĞİÖŞÜ→•–—", "cgiosuCGIOSU>*--")
    return str(text).translate(table).encode("latin-1", "replace").decode("latin-1")


def variant_report_pdf(
    path: str | Path,
    variant_id: str,
    panel: str,
    score: float,
    threshold: float,
    aa_ref: str,
    aa_alt: str,
    top_features: list[tuple[str, float, float]],
) -> Path:
    """One-page research summary of a model output (not a clinical report)."""
    from fpdf import FPDF

    predicted = (
        "above threshold (model: pathogenic-like)"
        if score >= threshold
        else "below threshold (model: benign-like)"
    )
    gran = grantham_distance(aa_ref, aa_alt)
    disr = structural_disruption(aa_ref, aa_alt)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    w = pdf.w - pdf.l_margin - pdf.r_margin

    def line(txt: str, size: int = 10, style: str = "", h: float = 6) -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(w, h, _ascii(txt))
        pdf.set_x(pdf.l_margin)

    line("VarPath - research model output", 15, "B", 9)
    pdf.set_fill_color(255, 240, 200)
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(w, 5, _ascii(RESEARCH_DISCLAIMER), border=1, fill=True)
    pdf.set_x(pdf.l_margin)
    pdf.ln(3)
    line("Variant", 12, "B")
    line(f"Identifier: {variant_id}")
    line(f"Panel: {panel.upper()}")
    line(f"Amino-acid change: {aa_ref} -> {aa_alt}")
    line(f"Grantham distance: {gran:.0f}   Structural propensity change: {disr:.2f}")
    pdf.ln(2)
    line("Model output", 12, "B")
    line(f"Stacked ensemble score: {score:.3f}")
    line(f"Deployment threshold: {threshold:.3f}  ->  {predicted}")
    line(
        "The score is the output of a statistical model trained on a competition dataset. "
        "It is not a calibrated risk and has not been clinically validated.",
        9, "I",
    )  # fmt: skip
    pdf.ln(2)
    line("Largest contributions to the XGBoost base learner (TreeSHAP)", 12, "B")
    for name, contrib, value in top_features:
        direction = "towards higher score" if contrib > 0 else "towards lower score"
        line(f"- {name} = {value:.3g}: {direction} ({contrib:+.3f})", 10)
    pdf.ln(2)
    line("Panel-related contextual information", 12, "B")
    line(PANEL_CONTEXT.get(panel, "No context available for this panel."), 10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path
