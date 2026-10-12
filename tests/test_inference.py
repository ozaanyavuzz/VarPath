import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from varpath.artifacts import load_artifacts
from varpath.cli import predict_main

REPO = Path(__file__).resolve().parents[1]


def test_predict_cli_writes_scores_and_reports(deployment, tiny_df, tmp_path):
    _, path = deployment
    inp = tmp_path / "in.csv"
    tiny_df.head(3).to_csv(inp, index=False)
    out = tmp_path / "pred.csv"
    rc = predict_main(["--artifacts", str(path), "--input", str(inp), "--output", str(out),
                       "--explain", "3", "--report-dir", str(tmp_path / "reports")])  # fmt: skip
    assert rc == 0
    res = pd.read_csv(out)
    assert list(res.columns[:4]) == ["id", "panel", "score", "predicted_label"]
    assert res["score"].between(0, 1).all()
    assert len(list((tmp_path / "reports").glob("*.pdf"))) == 3


def test_predict_cli_rejects_incomplete_schema(deployment, tiny_df, tmp_path):
    _, path = deployment
    predictor = load_artifacts(path)
    inp = tmp_path / "bad.csv"
    tiny_df.head(2).drop(columns=[predictor.preprocessor.numeric_columns_[0]]).to_csv(
        inp, index=False
    )
    assert (
        predict_main(
            ["--artifacts", str(path), "--input", str(inp), "--output", str(tmp_path / "o.csv")]
        )
        == 2
    )


def test_shap_values_explain_the_vector_used_for_prediction(deployment, tiny_df):
    _, path = deployment
    predictor = load_artifacts(path)
    X = predictor.transform(tiny_df.head(10))
    sv = predictor.shap_values(X)
    assert sv.shape == X.shape
    assert len(predictor.feature_names) == X.shape[1]
    margin = predictor.xgb_model.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    bias = predictor.xgb_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, -1]
    np.testing.assert_allclose(sv.sum(axis=1) + bias, margin, atol=1e-3)
    expl = predictor.explain(tiny_df.head(2), top_k=4)
    assert len(expl) == 2 and len(expl[0]) == 4
    assert all(name in predictor.feature_names for name, _, _ in expl[0])


def test_no_hard_coded_decision_threshold_in_sources():
    pattern = re.compile(r">=\s*0\.5\d\d")
    for f in list((REPO / "src").rglob("*.py")) + [REPO / "app.py"]:
        assert not pattern.search(f.read_text(encoding="utf-8")), f


def test_app_helpers_work_without_launching_gradio(deployment, tiny_df):
    import importlib.util

    _, path = deployment
    spec = importlib.util.spec_from_file_location("varpath_app", REPO / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    predictor = load_artifacts(path)
    table, md, _ = app.score_dataframe(predictor, tiny_df.head(3), make_pdf=False)
    assert len(table) == 3
    assert "research" in md.lower()
    desc = app.describe_substitution("R", "W")
    assert "Grantham" in desc and "not a model prediction" in desc.lower()
