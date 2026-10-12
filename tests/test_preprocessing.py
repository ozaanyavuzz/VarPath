import pickle

import numpy as np
import pandas as pd
import pytest

import varpath.features.preprocessing as prep
from varpath.evaluation.nested_cv import make_outer_splits, run_outer_fold
from varpath.features.preprocessing import FoldPreprocessor, SchemaError, detect_label_conflicts


def _split(tiny_df, cfg, arrays):
    y, groups, panels = arrays
    tr, va = make_outer_splits(y, groups, panels, cfg)[0]
    return tr, va, y


def test_fit_never_receives_outer_validation_rows(monkeypatch, tiny_df, cfg, arrays):
    fitted_ids = []
    original = FoldPreprocessor.fit

    def spy(self, df, y):
        fitted_ids.append(set(df[cfg.data.id_column]))
        return original(self, df, y)

    monkeypatch.setattr(FoldPreprocessor, "fit", spy)
    y, groups, panels = arrays
    tr, va = make_outer_splits(y, groups, panels, cfg)[0]
    run_outer_fold(tiny_df, y, groups, panels, tr, va, 0, cfg)
    val_ids = set(tiny_df[cfg.data.id_column].iloc[va])
    assert len(fitted_ids) == cfg.cv.inner_folds + 1
    for ids in fitted_ids:
        assert not ids & val_ids


def test_transform_does_not_change_fitted_state(tiny_df, cfg, arrays):
    tr, va, y = _split(tiny_df, cfg, arrays)
    pre = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(tiny_df.iloc[tr], y[tr])
    before = pickle.dumps(pre)
    pre.transform(tiny_df.iloc[va])
    assert pickle.dumps(pre) == before


def test_missing_indicator_selection_sees_training_labels_only(monkeypatch, tiny_df, cfg, arrays):
    tr, va, y = _split(tiny_df, cfg, arrays)
    seen = []
    original = prep.mutual_info_classif

    def spy(X, yy, **kw):
        seen.append(len(yy))
        return original(X, yy, **kw)

    monkeypatch.setattr(prep, "mutual_info_classif", spy)
    pre = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(tiny_df.iloc[tr], y[tr])
    assert seen == [len(tr)]
    # the selection is a function of training data alone
    y_poison = y.copy()
    y_poison[va] = 1 - y_poison[va]
    pre2 = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(tiny_df.iloc[tr], y_poison[tr])
    assert pre.selected_missing_columns_ == pre2.selected_missing_columns_


def test_encoder_uses_training_categories_and_ignores_unknown(tiny_df, cfg, arrays):
    tr, va, y = _split(tiny_df, cfg, arrays)
    df_tr = tiny_df.iloc[tr].copy()
    df_va = tiny_df.iloc[va].copy()
    df_va["CAT_1"] = "UNSEEN_LEVEL"
    pre = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(df_tr, y[tr])
    assert "UNSEEN_LEVEL" not in pre.encoders_["CAT_1"].categories_[0]
    X = pre.transform(df_va)
    start, stop = pre.block_slices_["categorical"]
    cat1 = [i for i, n in enumerate(pre.feature_names_out_[start:stop]) if n.startswith("CAT_1_")]
    assert np.all(X[:, [start + i for i in cat1]] == 0)


def test_tda_fill_values_are_training_medians(tiny_df, cfg, arrays):
    tr, _, y = _split(tiny_df, cfg, arrays)
    pre = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(tiny_df.iloc[tr], y[tr])
    num = prep.coerce_numeric(tiny_df.iloc[tr][pre.numeric_columns_])
    expected = num.median(skipna=True).to_numpy()
    np.testing.assert_allclose(pre.tda_fill_values_, expected, equal_nan=True)


def test_feature_names_match_transformed_width_and_block_order(tiny_df, cfg, arrays):
    tr, va, y = _split(tiny_df, cfg, arrays)
    pre = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(tiny_df.iloc[tr], y[tr])
    X = pre.transform(tiny_df.iloc[va])
    assert X.shape[1] == len(pre.feature_names_out_)
    assert X.dtype == np.float32
    order = list(pre.block_slices_)
    assert order == ["numeric", "amino_acid", "tda", "missing", "categorical"]
    assert pre.feature_names_out_[pre.block_slices_["amino_acid"][0]] == "aa1_hydro"


def test_constant_columns_are_dropped_from_values_but_not_from_missingness(tiny_df, cfg, arrays):
    tr, _, y = _split(tiny_df, cfg, arrays)
    pre = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(tiny_df.iloc[tr], y[tr])
    assert pre.dropped_numeric_columns_  # generator plants constant columns
    assert set(pre.dropped_numeric_columns_) <= set(pre.numeric_columns_)
    assert not set(pre.dropped_numeric_columns_) & set(pre.value_columns_)


def test_conflict_detection_is_local_to_given_rows(cfg):
    df = pd.DataFrame({"Variant_ID": list("abcd"), "AL_1": [1.0, 1.0, 2.0, 2.0],
                       "AA_1": ["A"] * 4, "AA_2": ["R"] * 4, "panel": ["master"] * 4})  # fmt: skip
    y = np.array([0, 1, 1, 1])
    np.testing.assert_array_equal(
        detect_label_conflicts(df, y, cfg.data), [True, True, False, False]
    )
    np.testing.assert_array_equal(
        detect_label_conflicts(df.iloc[[0, 2, 3]], y[[0, 2, 3]], cfg.data), [False] * 3
    )


def test_missing_numeric_column_raises_schema_error(tiny_df, cfg, arrays):
    tr, va, y = _split(tiny_df, cfg, arrays)
    pre = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(tiny_df.iloc[tr], y[tr])
    with pytest.raises(SchemaError):
        pre.transform(tiny_df.iloc[va].drop(columns=[pre.numeric_columns_[0]]))


def test_transform_is_row_independent(tiny_df, cfg, arrays):
    """A row's features must not depend on the other rows in the batch."""
    tr, va, y = _split(tiny_df, cfg, arrays)
    pre = FoldPreprocessor(cfg.data, cfg.preprocessing, 0).fit(tiny_df.iloc[tr], y[tr])
    full = pre.transform(tiny_df.iloc[va])
    single = pre.transform(tiny_df.iloc[va[3:4]])
    np.testing.assert_allclose(full[3:4], single, atol=1e-5)
