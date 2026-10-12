"""Residue-level structural propensity descriptors.

Chou-Fasman helix/sheet/turn propensities (Chou & Fasman, 1978), a normalised
relative solvent accessibility and a rigidity index for the reference and
alternative residue, their difference and three derived disruption scores
(18 features).
"""

from __future__ import annotations

import numpy as np

#                Pa     Pb     Pt    RSA   rigid
AA_STRUCT: dict[str, list[float]] = {
    "A": [1.42, 0.83, 0.66, 0.50, 0.72],
    "R": [0.98, 0.93, 0.95, 0.83, 0.45],
    "N": [0.67, 0.89, 1.56, 0.82, 0.52],
    "D": [1.01, 0.54, 1.46, 0.81, 0.55],
    "C": [0.70, 1.19, 1.19, 0.40, 0.88],
    "E": [1.51, 0.37, 0.74, 0.84, 0.50],
    "Q": [1.11, 1.10, 0.98, 0.81, 0.48],
    "G": [0.57, 0.75, 1.56, 0.51, 1.00],
    "H": [1.00, 0.87, 0.95, 0.72, 0.62],
    "I": [1.08, 1.60, 0.47, 0.37, 0.78],
    "L": [1.21, 1.30, 0.59, 0.37, 0.75],
    "K": [1.16, 0.74, 1.01, 0.93, 0.38],
    "M": [1.45, 1.05, 0.60, 0.44, 0.68],
    "F": [1.13, 1.38, 0.60, 0.35, 0.80],
    "P": [0.57, 0.55, 1.52, 0.53, 1.00],
    "S": [0.77, 0.75, 1.43, 0.64, 0.62],
    "T": [0.83, 1.19, 0.96, 0.61, 0.65],
    "W": [1.08, 1.37, 0.96, 0.28, 0.75],
    "Y": [0.69, 1.47, 1.14, 0.47, 0.70],
    "V": [1.06, 1.70, 0.50, 0.36, 0.80],
}
STRUCT_NAMES = ["Pa", "Pb", "Pt", "RSA", "rigid"]

STRUCT_FEATURE_NAMES = (
    [f"str1_{n}" for n in STRUCT_NAMES]
    + [f"str2_{n}" for n in STRUCT_NAMES]
    + [f"dstr_{n}" for n in STRUCT_NAMES]
    + ["struct_disruption", "helix_change", "sheet_change"]
)


def structural_features(aa1: str, aa2: str) -> np.ndarray:
    zero = np.zeros(5)
    s1 = np.asarray(AA_STRUCT.get(aa1, zero), dtype=float)
    s2 = np.asarray(AA_STRUCT.get(aa2, zero), dtype=float)
    diff = s2 - s1
    disruption = float(np.linalg.norm(diff))
    helix_change = abs(s2[0] - s1[0])
    sheet_change = abs(s2[1] - s1[1])
    return np.concatenate([s1, s2, diff, [disruption, helix_change, sheet_change]])


def structural_disruption(aa1: str, aa2: str) -> float:
    return float(structural_features(aa1, aa2)[15])
