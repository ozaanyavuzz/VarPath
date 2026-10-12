import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from varpath.evaluation.calibration import expected_calibration_error
from varpath.evaluation.dca import decision_curve
from varpath.evaluation.metrics import binary_metrics, panel_metrics_table, reliability_label
from varpath.evaluation.statistics import bootstrap_ci, delong_test, mcnemar_test


def legacy_delong(y_true, pa, pb):
    aa, ab = roc_auc_score(y_true, pa), roc_auc_score(y_true, pb)
    pp_a, np_a = pa[y_true == 1], pa[y_true == 0]
    pp_b, np_b = pb[y_true == 1], pb[y_true == 0]
    n1, n0 = len(pp_a), len(np_a)
    v10a = np.array([np.mean(p > np_a) + 0.5 * np.mean(p == np_a) for p in pp_a])
    v01a = np.array([np.mean(p < pp_a) + 0.5 * np.mean(p == pp_a) for p in np_a])
    v10b = np.array([np.mean(p > np_b) + 0.5 * np.mean(p == np_b) for p in pp_b])
    v01b = np.array([np.mean(p < pp_b) + 0.5 * np.mean(p == pp_b) for p in np_b])
    s = np.cov(v10a, v10b) / n1 + np.cov(v01a, v01b) / n0
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    return aa, ab, 2 * (1 - stats.norm.cdf(abs((aa - ab) / np.sqrt(var))))


def test_fast_delong_matches_reference():
    rng = np.random.default_rng(0)
    y = (rng.random(250) < 0.6).astype(int)
    a = (y * 0.4 + rng.random(250)).round(2)
    b = (y * 0.2 + rng.random(250)).round(2)
    res = delong_test(y, a, b)
    aa, ab, p = legacy_delong(y, a, b)
    assert res["auc_a"] == pytest.approx(aa)
    assert res["auc_b"] == pytest.approx(ab)
    assert res["p_value"] == pytest.approx(p, rel=1e-6)


def test_mcnemar_counts():
    y = np.array([1, 1, 0, 0, 1, 0])
    a = np.array([1, 0, 0, 1, 1, 0])
    b = np.array([1, 1, 0, 0, 0, 0])
    r = mcnemar_test(y, a, b)
    assert r["n_a_wrong_b_right"] == 2 and r["n_a_right_b_wrong"] == 1


def test_bootstrap_uses_fixed_predictions_and_groups():
    rng = np.random.default_rng(1)
    y = (rng.random(200) < 0.5).astype(int)
    s = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    p = (s > 0.5).astype(int)
    groups = np.repeat(np.arange(100), 2)
    ci = bootstrap_ci(y, s, p, groups, n_boot=100, seed=0)
    assert ci["f2"]["estimate"] == pytest.approx(binary_metrics(y, p)["f2"])
    for k in ("roc_auc", "pr_auc", "f2", "mcc"):
        assert ci[k]["ci_lower"] <= ci[k]["estimate"] <= ci[k]["ci_upper"]
    again = bootstrap_ci(y, s, p, groups, n_boot=100, seed=0)
    assert again == ci


def test_ece_includes_probability_one():
    assert expected_calibration_error(np.array([1, 1]), np.array([1.0, 1.0])) == pytest.approx(0.0)


def test_dca_treat_all_matches_formula():
    y = np.array([1, 1, 0, 0])
    d = decision_curve(y, {"m": np.array([0.9, 0.8, 0.2, 0.1])}, np.array([0.5]))
    assert d["treat_all"].iloc[0] == pytest.approx(0.0)
    assert d["m"].iloc[0] == pytest.approx(0.5)


def test_panel_metrics_flags_small_and_single_class_subsets():
    import pandas as pd

    df = pd.DataFrame({
        "panel": ["a"] * 12 + ["b"] * 6,
        "y": [1, 0] * 6 + [1] * 6,
        "p_stack": np.linspace(0, 1, 18),
        "yhat_stack": [1, 0] * 9,
    })  # fmt: skip
    t = panel_metrics_table(df, ["stack"], min_n=5, exploratory_n=30).set_index("panel")
    assert t.loc["a", "reliability"].startswith("exploratory")
    assert t.loc["b", "reliability"].startswith("single-class")
    assert np.isnan(t.loc["b", "roc_auc"])
    assert reliability_label(100, 50, 50, 30) == "reportable"
