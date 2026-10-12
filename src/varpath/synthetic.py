"""Artificial data with the competition file layout.

Values are drawn from simple parametric distributions with a planted signal.
Nothing is derived from real patient records or from the competition files;
the generator exists so that the code, tests and demo run without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features.amino_acid import STANDARD_AMINO_ACIDS, grantham_distance

DEFAULT_PANEL_SIZES = {"master": 240, "kanser": 60, "pah": 50, "cftr": 40}
_GENOTYPES = ["A/A", "C/C", "G/G", "T/T"]
_POPULATIONS = ["popA", "popB", "popC", "popD", "popE"]


def _panel_frame(
    panel: str, n: int, n_al: int, n_ek: int, n_cat: int, rng: np.random.Generator, id_offset: int
) -> pd.DataFrame:
    aa1 = rng.choice(STANDARD_AMINO_ACIDS, n)
    aa2 = np.array([rng.choice([a for a in STANDARD_AMINO_ACIDS if a != r]) for r in aa1])
    gran = np.array([grantham_distance(a, b) for a, b in zip(aa1, aa2)])
    z = rng.normal(size=n) + (gran - 100) / 60.0
    prevalence_shift = {"master": 0.8, "kanser": 0.6, "pah": 1.4, "cftr": 1.2}.get(panel, 0.8)
    label = (rng.random(n) < 1 / (1 + np.exp(-(1.3 * z + prevalence_shift)))).astype(int)

    cols: dict[str, np.ndarray] = {
        "Variant_ID": np.array([f"SYN_{id_offset + i:06d}" for i in range(n)])
    }
    for j in range(1, n_al + 1):
        kind = j % 5
        if kind == 0:
            v = z * rng.uniform(0.3, 1.0) + rng.normal(scale=1.0, size=n)
        elif kind == 1:
            v = rng.lognormal(mean=-3, sigma=1.5, size=n)
        elif kind == 2:
            v = rng.uniform(0, 1, size=n)
        elif kind == 3:
            v = np.full(n, 1.0)  # constant column, as found in real annotation tables
        else:
            v = rng.normal(size=n)
        miss_p = 0.15 + 0.35 * (j % 3 == 0) + 0.15 * (label == 0) * (j % 7 == 0)
        v = np.where(rng.random(n) < miss_p, np.nan, v)
        cols[f"AL_{j}"] = v
    for j in range(1, n_cat + 1):
        pool = _POPULATIONS if j <= 2 else _GENOTYPES
        c = rng.choice(pool, n).astype(object)
        c[rng.random(n) < 0.3] = None
        cols[f"CAT_{j}"] = c
    for j in range(1, n_ek + 1):
        v = z * 0.5 + rng.normal(scale=1.0, size=n) + 5
        cols[f"EK_{j}"] = np.where(rng.random(n) < 0.1, np.nan, v)
    a1 = aa1.astype(object)
    a2 = aa2.astype(object)
    unknown = rng.random(n) < 0.05
    a1[unknown] = None
    a2[unknown] = None
    cols["AA_1"] = a1
    cols["AA_2"] = a2
    cols["Label"] = label
    df = pd.DataFrame(cols)
    # a few rows with no numeric information at all (conflict-prone, as in real tables)
    empty = rng.random(n) < 0.03
    num_cols = [c for c in df.columns if c.startswith(("AL_", "EK_"))]
    df.loc[empty, num_cols] = np.nan
    return df


def make_synthetic_panels(
    sizes: dict[str, int] | None = None,
    n_al: int = 40,
    n_ek: int = 4,
    n_cat: int = 4,
    duplicate_fraction: float = 0.3,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Panel tables; a share of sub-panel variants re-appear in ``master``."""
    rng = np.random.default_rng(seed)
    sizes = sizes or DEFAULT_PANEL_SIZES
    frames: dict[str, pd.DataFrame] = {}
    offset = 0
    for panel, n in sizes.items():
        frames[panel] = _panel_frame(panel, n, n_al, n_ek, n_cat, rng, offset)
        offset += n
    if "master" in frames and duplicate_fraction > 0:
        extra = []
        for panel, df in frames.items():
            if panel == "master":
                continue
            k = int(round(duplicate_fraction * len(df)))
            dup = df.sample(k, random_state=int(rng.integers(1_000_000))).copy()
            num_cols = [
                c
                for c in dup.columns
                if c.startswith(("AL_", "EK_")) and df[c].nunique(dropna=True) > 1
            ]
            noise = rng.normal(scale=0.05, size=(k, len(num_cols)))
            dup[num_cols] = dup[num_cols].to_numpy(dtype=float) * (1 + noise)
            extra.append(dup)
        if extra:
            frames["master"] = pd.concat([frames["master"], *extra], ignore_index=True)
    return frames


def write_synthetic_panels(
    out_dir: str | Path, file_names: dict[str, str], **kwargs
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = make_synthetic_panels(**kwargs)
    paths = {}
    for panel, df in frames.items():
        p = out / file_names[panel]
        df.to_csv(p, index=False)
        paths[panel] = p
    return paths


def synthetic_dataframe(panel_column: str = "panel", **kwargs) -> pd.DataFrame:
    frames = make_synthetic_panels(**kwargs)
    return pd.concat(
        [df.assign(**{panel_column: p}) for p, df in frames.items()], ignore_index=True
    )
