"""Data loading, biological grouping and group auditing."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, GroupingConfig

logger = logging.getLogger("varpath")

# Normalised column-name tokens that would indicate a biological entity.
_ENTITY_NAME_TOKENS: dict[str, tuple[str, ...]] = {
    "gene": ("gene", "genesymbol", "genename", "symbol", "hgnc", "hgncid", "hgncsymbol", "ensg", "geneid", "entrez"),
    "protein": ("protein", "proteinid", "uniprot", "uniprotid", "uniprotkb", "swissprot", "ensp", "refseqprotein"),
    "transcript": ("transcript", "transcriptid", "enst", "refseq", "nmid", "feature"),
}  # fmt: skip

# Value patterns for identifiers that may hide behind anonymised column names.
_ENTITY_VALUE_PATTERNS: dict[str, re.Pattern] = {
    "gene": re.compile(r"^(ENSG\d{6,}|HGNC:\d+)$"),
    "protein": re.compile(r"^(ENSP\d{6,}|NP_\d+(\.\d+)?|[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?)$"),
    "transcript": re.compile(r"^(ENST\d{6,}|NM_\d+(\.\d+)?|XM_\d+(\.\d+)?)$"),
}


class GroupLeakageError(RuntimeError):
    """Raised when a group appears on both sides of a split."""


def read_csv(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        logger.warning("Malformed lines in %s are skipped", path)
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")


def load_panel_data(cfg: DataConfig, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Concatenate panel CSVs, attach the panel label and keep labelled rows only."""
    root = Path(data_dir or cfg.data_dir)
    frames = []
    for panel, fname in cfg.panel_files.items():
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Panel file not found: {path}. See examples/DATA_SCHEMA.md for the expected layout."
            )
        df = read_csv(path)
        if cfg.label_column not in df.columns:
            raise ValueError(f"{path} has no '{cfg.label_column}' column")
        df[cfg.panel_column] = panel
        df[cfg.label_column] = pd.to_numeric(df[cfg.label_column], errors="coerce")
        df = df[df[cfg.label_column].notna()].copy()
        bad = ~df[cfg.label_column].isin([0, 1])
        if bad.any():
            raise ValueError(f"{path}: labels must be 0/1 ({int(bad.sum())} invalid rows)")
        df[cfg.label_column] = df[cfg.label_column].astype(int)
        logger.info(
            "[%s] %d rows | positive=%d negative=%d",
            panel.upper(),
            len(df),
            int(df[cfg.label_column].sum()),
            int((df[cfg.label_column] == 0).sum()),
        )
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.reset_index(drop=True)


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def discover_entity_columns(df: pd.DataFrame, sample: int = 500) -> dict[str, list[str]]:
    """Scan column names and string values for gene/protein/transcript identifiers."""
    found: dict[str, list[str]] = {k: [] for k in _ENTITY_NAME_TOKENS}
    for col in df.columns:
        norm = _normalise_name(col)
        for level, tokens in _ENTITY_NAME_TOKENS.items():
            if norm in tokens or any(norm.startswith(t) and len(t) > 4 for t in tokens):
                found[level].append(str(col))
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        vals = s.dropna().astype(str).head(sample)
        if vals.empty:
            continue
        for level, pat in _ENTITY_VALUE_PATTERNS.items():
            if vals.str.match(pat).mean() > 0.8 and str(col) not in found[level]:
                found[level].append(str(col))
    return found


def row_hash_groups(
    df: pd.DataFrame, data_cfg: DataConfig, decimals: int = 3, buckets: int | None = None
) -> np.ndarray:
    """Duplicate-profile groups (sensitivity analysis only; not a biological entity)."""
    from .features.preprocessing import candidate_columns, coerce_numeric

    num_cols, _ = candidate_columns(df, data_cfg)
    prof = coerce_numeric(df[num_cols]).round(decimals)
    h = pd.util.hash_pandas_object(prof, index=False).to_numpy()
    if buckets:
        h = h % int(buckets)
    return np.asarray([f"h{v}" for v in h], dtype=object)


def build_groups(df: pd.DataFrame, grouping: GroupingConfig, data_cfg: DataConfig) -> np.ndarray:
    if grouping.strategy == "row_hash":
        logger.warning(
            "Grouping by duplicate numeric profile (row_hash). This is a sensitivity analysis "
            "and does NOT hold out genes, proteins or variants."
        )
        return row_hash_groups(df, data_cfg, grouping.row_hash_decimals, grouping.row_hash_buckets)
    col = grouping.column
    if col not in df.columns:
        raise KeyError(
            f"Grouping column '{col}' is not present in the data. Set grouping.column to a real "
            "entity column or explicitly choose grouping.strategy='row_hash' for a sensitivity run."
        )
    values = df[col]
    if values.isna().any():
        raise ValueError(
            f"Grouping column '{col}' contains {int(values.isna().sum())} missing values"
        )
    return values.astype(str).to_numpy(dtype=object)


def group_audit(
    df: pd.DataFrame, groups: np.ndarray, grouping: GroupingConfig, data_cfg: DataConfig
) -> dict:
    """Describe the grouping that the evaluation relies on."""
    g = pd.Series(groups, index=df.index)
    sizes = g.value_counts()
    panels = df[data_cfg.panel_column]
    panels_per_group = panels.groupby(g).nunique()
    labels_per_group = df[data_cfg.label_column].groupby(g).nunique()
    entity_cols = discover_entity_columns(df)
    has_bio_entity = any(entity_cols[k] for k in ("gene", "protein", "transcript"))
    audit = {
        "strategy": grouping.strategy,
        "column": grouping.column if grouping.strategy == "column" else None,
        "entity_level": grouping.entity_level
        if grouping.strategy == "column"
        else "duplicate_profile",
        "n_rows": int(len(df)),
        "n_groups": int(sizes.size),
        "n_multi_row_groups": int((sizes > 1).sum()),
        "max_group_size": int(sizes.max()) if sizes.size else 0,
        "n_groups_spanning_multiple_panels": int((panels_per_group > 1).sum()),
        "n_groups_with_mixed_labels": int((labels_per_group > 1).sum()),
        "detected_entity_columns": entity_cols,
        "gene_or_protein_or_transcript_identifier_available": bool(has_bio_entity),
        "gene_held_out_evaluation": bool(
            grouping.strategy == "column" and grouping.entity_level == "gene"
        ),
        "cross_panel_group_overlap": {},
    }
    panel_sets = {p: set(g[panels == p]) for p in panels.unique()}
    names = sorted(panel_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            audit["cross_panel_group_overlap"][f"{a}|{b}"] = len(panel_sets[a] & panel_sets[b])
    return audit


def assert_disjoint_groups(groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    inter = set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[val_idx])
    if inter:
        raise GroupLeakageError(
            f"{len(inter)} group(s) appear in both training and validation, e.g. {sorted(inter)[:3]}"
        )
