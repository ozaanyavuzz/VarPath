import numpy as np
import pytest

import varpath.models.gbdt as gbdt
import varpath.models.stacking as stacking
from varpath.evaluation.nested_cv import make_outer_splits, run_nested_cv, run_outer_fold
from varpath.models.stacking import fit_stacked_pipeline


class MemorisingClassifier:
    """Returns the stored label for rows seen in fit, 0.5 for unseen rows."""

    def __init__(self, *args, **kwargs):
        self.memory = {}

    def fit(self, X, y, **kwargs):
        self.memory = {np.asarray(r, dtype=np.float32).tobytes(): float(v) for r, v in zip(X, y)}
        return self

    def predict_proba(self, X):
        p = np.array([self.memory.get(np.asarray(r, dtype=np.float32).tobytes(), 0.5) for r in X])
        return np.column_stack([1 - p, p])


def test_every_training_row_gets_exactly_one_inner_oof_prediction(deployment):
    pipe, _ = deployment
    oof = pipe.training_oof
    assert oof["row"].is_unique
    assert (oof["inner_fold"] >= 0).all()
    for col in ("p_ft", "p_xgb", "p_lgb", "p_stack"):
        assert np.isfinite(oof[col]).all()
    assert len(oof) == pipe.info["n_training_rows"]


def test_inner_oof_predictions_come_from_models_that_did_not_see_the_row(
    monkeypatch, tiny_df, cfg, arrays
):
    monkeypatch.setattr(gbdt, "make_xgb", lambda *a, **k: MemorisingClassifier())
    y, groups, panels = arrays
    pipe = fit_stacked_pipeline(tiny_df, y, groups, panels, cfg, seed=3, n_splits=2)
    # a memorising model can only output 0.5 for rows it never trained on
    assert np.allclose(pipe.training_oof["p_xgb"], 0.5)
    # sanity check: the refit model does memorise its own training rows
    X = pipe.transform(tiny_df.iloc[pipe.training_oof["row"]])
    assert not np.allclose(pipe.xgb_model.predict_proba(X)[:, 1], 0.5)


def test_meta_learner_is_trained_only_on_outer_training_rows(monkeypatch, tiny_df, cfg, arrays):
    sizes = []
    original = stacking._meta_learner

    def spy(c, seed):
        lr = original(c, seed)
        fit = lr.fit

        def recording_fit(X, yy, *a, **k):
            sizes.append(len(yy))
            return fit(X, yy, *a, **k)

        lr.fit = recording_fit
        return lr

    monkeypatch.setattr(stacking, "_meta_learner", spy)
    y, groups, panels = arrays
    tr, va = make_outer_splits(y, groups, panels, cfg)[0]
    _, record, _ = run_outer_fold(tiny_df, y, groups, panels, tr, va, 0, cfg)
    assert sizes and max(sizes) == record["pipeline"]["n_training_rows"] <= len(tr)


def test_poisoned_outer_validation_labels_change_nothing(tiny_df, cfg, arrays):
    """Flipping validation labels must not change any training-side decision or prediction."""
    y, groups, panels = arrays
    tr, va = make_outer_splits(y, groups, panels, cfg)[0]
    probs_a, rec_a, _ = run_outer_fold(tiny_df, y, groups, panels, tr, va, 0, cfg)
    y_poison = y.copy()
    y_poison[va] = 1 - y_poison[va]
    df_poison = tiny_df.copy()
    df_poison.loc[df_poison.index[va], cfg.data.label_column] = y_poison[va]
    probs_b, rec_b, _ = run_outer_fold(df_poison, y_poison, groups, panels, tr, va, 0, cfg)
    assert rec_a["thresholds"] == rec_b["thresholds"]
    for key in ("xgb", "lgb", "ft", "meta_learner", "n_features_full"):
        assert rec_a["pipeline"][key] == rec_b["pipeline"][key]
    for k in probs_a:
        np.testing.assert_allclose(probs_a[k], probs_b[k], atol=1e-6)


def test_nested_cv_gives_each_row_one_held_out_prediction(tiny_df, cfg, arrays):
    y, groups, panels = arrays
    res = run_nested_cv(tiny_df, y, groups, panels, cfg)
    preds = res.predictions
    assert len(preds) == len(tiny_df)
    assert preds["row"].is_unique
    assert set(preds["outer_fold"]) == set(range(cfg.cv.outer_folds))
    for m in ("ft", "xgb", "lgb", "stack"):
        assert np.isfinite(preds[f"p_{m}"]).all()
        expected = (preds[f"p_{m}"] >= preds[f"thr_{m}"]).astype(int)
        assert (expected == preds[f"yhat_{m}"]).all()
    for (tr, va), f in zip(res.outer_splits, res.folds):
        assert (preds.loc[va, "thr_stack"] == f["thresholds"]["stack"]).all()
        assert not set(groups[tr]) & set(groups[va])


def test_pipeline_is_deterministic(tiny_df, cfg, arrays, deployment):
    pipe_a, _ = deployment
    y, groups, panels = arrays
    pipe_b = fit_stacked_pipeline(tiny_df, y, groups, panels, cfg, seed=cfg.seed, n_splits=2)
    assert pipe_a.threshold == pytest.approx(pipe_b.threshold)
    a = pipe_a.predict_components(tiny_df.iloc[:40])
    b = pipe_b.predict_components(tiny_df.iloc[:40])
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-6)
