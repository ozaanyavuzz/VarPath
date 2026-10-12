import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, fbeta_score, precision_recall_curve

import varpath.models.stacking as stacking
from varpath.evaluation.nested_cv import make_outer_splits, run_outer_fold
from varpath.evaluation.thresholding import apply_threshold, select_threshold


def legacy_best_threshold(y_true, y_proba, min_spec=0.60):
    """Loop implementation from the competition prototype."""
    _, _, thrs = precision_recall_curve(y_true, y_proba)
    best_f2, best_t = 0.0, 0.5
    for t in thrs:
        yp = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        if tn / (tn + fp + 1e-12) < min_spec:
            continue
        f2 = fbeta_score(y_true, yp, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, float(t)
    return float(best_t), float(best_f2)


@pytest.mark.parametrize("seed", range(6))
def test_vectorised_threshold_matches_legacy_rule(seed):
    rng = np.random.default_rng(seed)
    n = 300
    y = (rng.random(n) < 0.7).astype(int)
    p = np.clip(0.35 * y + rng.normal(0.4, 0.2, n), 0, 1).round(3)  # rounding creates ties
    t_new, f_new = select_threshold(y, p)
    t_old, f_old = legacy_best_threshold(y, p)
    assert t_new == pytest.approx(t_old)
    assert f_new == pytest.approx(f_old)


def test_specificity_constraint_is_respected():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.6).astype(int)
    p = np.clip(0.2 * y + rng.random(400) * 0.8, 0, 1)
    t, _ = select_threshold(y, p, min_specificity=0.7)
    pred = apply_threshold(p, t)
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    assert tn / (tn + fp) >= 0.7


def test_single_class_returns_default():
    assert select_threshold(np.ones(10), np.linspace(0, 1, 10), default=0.4) == (0.4, 0.0)


def test_outer_validation_labels_never_reach_threshold_selection(monkeypatch, tiny_df, cfg, arrays):
    calls = []
    original = stacking.select_threshold

    def spy(y, p, *a, **kw):
        calls.append(np.asarray(y).copy())
        return original(y, p, *a, **kw)

    monkeypatch.setattr(stacking, "select_threshold", spy)
    y, groups, panels = arrays
    tr, va = make_outer_splits(y, groups, panels, cfg)[0]
    _, record, _ = run_outer_fold(tiny_df, y, groups, panels, tr, va, 0, cfg)
    n_train_used = record["pipeline"]["n_training_rows"]
    assert calls and all(len(c) == n_train_used for c in calls)
    assert n_train_used <= len(tr)
