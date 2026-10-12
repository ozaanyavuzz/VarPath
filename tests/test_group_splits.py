import dataclasses

import numpy as np
import pandas as pd
import pytest

from varpath.config import GroupingConfig
from varpath.data import (
    GroupLeakageError,
    assert_disjoint_groups,
    build_groups,
    discover_entity_columns,
    group_audit,
)
from varpath.evaluation.nested_cv import make_outer_splits, run_outer_fold
from varpath.evaluation.splits import group_holdout_split, group_kfold_splits, stratification_labels


def test_synthetic_data_contains_cross_panel_duplicates(tiny_df, arrays):
    _, groups, panels = arrays
    dup = pd.Series(groups).duplicated(keep=False)
    assert dup.any()
    assert pd.Series(panels)[dup].nunique() > 1


def test_outer_splits_are_group_disjoint_and_cover_all_rows(cfg, arrays):
    y, groups, panels = arrays
    splits = make_outer_splits(y, groups, panels, cfg)
    assert len(splits) == cfg.cv.outer_folds
    seen = np.concatenate([va for _, va in splits])
    assert sorted(seen.tolist()) == list(range(len(y)))
    for tr, va in splits:
        assert not set(groups[tr]) & set(groups[va])
        assert len(np.unique(y[tr])) == 2


def test_inner_splits_and_holdout_are_group_disjoint(cfg, arrays):
    y, groups, panels = arrays
    tr, _ = make_outer_splits(y, groups, panels, cfg)[0]
    strat = stratification_labels(y[tr], panels[tr], cfg.cv.stratify_by)
    for itr, iva in group_kfold_splits(strat, y[tr], groups[tr], cfg.cv.inner_folds, 5):
        assert not set(groups[tr][itr]) & set(groups[tr][iva])
        fit, hold = group_holdout_split(strat[itr], y[tr][itr], groups[tr][itr], 0.2, 9)
        g = groups[tr][itr]
        assert not set(g[fit]) & set(g[hold])
        assert len(fit) + len(hold) == len(itr)


def test_inner_splits_used_by_pipeline_are_disjoint(monkeypatch, tiny_df, cfg, arrays):
    import varpath.models.stacking as stacking

    captured = []
    original = stacking.group_kfold_splits

    def spy(strat, y, groups, n_splits, seed):
        out = original(strat, y, groups, n_splits, seed)
        captured.append((groups, out))
        return out

    monkeypatch.setattr(stacking, "group_kfold_splits", spy)
    y, groups, panels = arrays
    tr, va = make_outer_splits(y, groups, panels, cfg)[0]
    run_outer_fold(tiny_df, y, groups, panels, tr, va, 0, cfg)
    assert captured
    for g, splits in captured:
        for itr, iva in splits:
            assert not set(g[itr]) & set(g[iva])
            assert not set(g[iva]) & set(groups[va])


def test_leakage_guard_raises():
    groups = np.array(["a", "a", "b", "c"])
    with pytest.raises(GroupLeakageError):
        assert_disjoint_groups(groups, np.array([0, 2]), np.array([1, 3]))


def test_run_outer_fold_refuses_overlapping_groups(tiny_df, cfg, arrays):
    y, groups, panels = arrays
    dup_group = pd.Series(groups)[pd.Series(groups).duplicated()].iloc[0]
    rows = np.flatnonzero(groups == dup_group)
    others = np.setdiff1d(np.arange(len(y)), rows)
    tr = np.concatenate([others[:100], rows[:1]])
    va = rows[1:]
    with pytest.raises(GroupLeakageError):
        run_outer_fold(tiny_df, y, groups, panels, tr, va, 0, cfg)


def test_splits_are_deterministic(cfg, arrays):
    y, groups, panels = arrays
    a = make_outer_splits(y, groups, panels, cfg)
    b = make_outer_splits(y, groups, panels, cfg)
    for (t1, v1), (t2, v2) in zip(a, b):
        assert np.array_equal(t1, t2) and np.array_equal(v1, v2)
    other = make_outer_splits(y, groups, panels, dataclasses.replace(cfg, seed=cfg.seed + 1))
    assert any(not np.array_equal(v1, v2) for (_, v1), (_, v2) in zip(a, other))


def test_missing_group_column_is_an_error(tiny_df, cfg):
    with pytest.raises(KeyError):
        build_groups(tiny_df, GroupingConfig(strategy="column", column="gene_symbol"), cfg.data)


def test_row_hash_groups_identical_profiles(tiny_df, cfg):
    df = pd.concat([tiny_df.iloc[:5], tiny_df.iloc[:2]], ignore_index=True)
    g = build_groups(df, GroupingConfig(strategy="row_hash"), cfg.data)
    assert g[0] == g[5] and g[1] == g[6]


def test_entity_discovery_by_name_and_value(tiny_df):
    df = pd.DataFrame({"Gene_Symbol": ["BRCA1"], "col_x": ["ENST00000357654"], "AL_1": [0.1]})
    found = discover_entity_columns(df)
    assert "Gene_Symbol" in found["gene"]
    assert "col_x" in found["transcript"]
    none = discover_entity_columns(tiny_df)
    assert not any(none.values())


def test_group_audit_reports_variant_level(tiny_df, cfg, arrays):
    _, groups, _ = arrays
    audit = group_audit(tiny_df, groups, cfg.grouping, cfg.data)
    assert audit["entity_level"] == "variant"
    assert audit["gene_held_out_evaluation"] is False
    assert audit["n_groups_spanning_multiple_panels"] > 0
