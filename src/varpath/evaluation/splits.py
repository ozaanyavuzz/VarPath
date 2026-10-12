"""Group-aware, stratified-where-possible splitting."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from ..data import assert_disjoint_groups

logger = logging.getLogger("varpath")

Split = tuple[np.ndarray, np.ndarray]


def stratification_labels(
    y: np.ndarray, panels: np.ndarray | None, stratify_by: list[str]
) -> np.ndarray:
    y = np.asarray(y).astype(int)
    parts = []
    if "panel" in stratify_by and panels is not None:
        parts.append(np.asarray(panels).astype(str))
    if "label" in stratify_by or not parts:
        parts.append(y.astype(str))
    lab = parts[0]
    for p in parts[1:]:
        lab = np.char.add(np.char.add(lab.astype(str), "|"), p)
    return lab


def _valid(splits: list[Split], y: np.ndarray) -> bool:
    for tr, va in splits:
        if len(va) == 0 or len(np.unique(y[tr])) < 2:
            return False
    return True


def _shuffled_group_kfold(groups: np.ndarray, n_splits: int, seed: int) -> list[Split]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
    codes = np.array([perm[g] for g in groups])
    idx = np.arange(len(groups))
    return [(tr, va) for tr, va in GroupKFold(n_splits=n_splits).split(idx, groups=codes)]


def group_kfold_splits(
    strat: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[Split]:
    """K group-disjoint folds, stratified on ``strat`` when feasible.

    Falls back to label-only stratification and then to a shuffled GroupKFold
    when a stratified assignment cannot give every training fold both classes.
    """
    groups = np.asarray(groups)
    y = np.asarray(y).astype(int)
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"Only {n_groups} groups available for {n_splits} folds")
    idx = np.arange(len(y))
    candidates = [np.asarray(strat)]
    if not np.array_equal(np.asarray(strat), y.astype(str)):
        candidates.append(y.astype(str))
    splits: list[Split] | None = None
    for lab in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            try:
                cand = [(tr, va) for tr, va in sgkf.split(idx, lab, groups)]
            except ValueError:
                continue
        if _valid(cand, y):
            splits = cand
            break
    if splits is None:
        logger.warning("Stratified group split infeasible; using shuffled GroupKFold")
        splits = _shuffled_group_kfold(groups, n_splits, seed)
    for tr, va in splits:
        assert_disjoint_groups(groups, tr, va)
    return [(np.sort(tr), np.sort(va)) for tr, va in splits]


def group_holdout_split(
    strat: np.ndarray, y: np.ndarray, groups: np.ndarray, fraction: float, seed: int
) -> Split:
    """Group-disjoint (fit, holdout) split of a training partition."""
    n_splits = max(2, int(round(1.0 / fraction)))
    n_splits = min(n_splits, len(np.unique(groups)))
    fit_idx, hold_idx = group_kfold_splits(strat, y, groups, n_splits, seed)[0]
    return fit_idx, hold_idx
