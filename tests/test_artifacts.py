import json

import numpy as np
import pytest

from varpath.artifacts import ArtifactError, load_artifacts


def test_save_load_roundtrip_gives_identical_predictions(deployment, tiny_df):
    pipe, path = deployment
    predictor = load_artifacts(path)
    expected = pipe.predict_components(tiny_df)
    got = predictor.predict(tiny_df)
    np.testing.assert_allclose(got["score"], expected["stack"], atol=1e-6)
    for k in ("ft", "xgb", "lgb"):
        np.testing.assert_allclose(got[f"p_{k}"], expected[k], atol=1e-6)
    assert predictor.deployment_threshold == pytest.approx(pipe.threshold)
    np.testing.assert_array_equal(
        got["predicted_label"], (expected["stack"] >= pipe.threshold).astype(int)
    )


def test_feature_order_is_identical_between_training_and_inference(deployment, tiny_df):
    pipe, path = deployment
    predictor = load_artifacts(path)
    assert predictor.feature_names == pipe.preprocessor.feature_names_out_
    np.testing.assert_allclose(predictor.transform(tiny_df), pipe.transform(tiny_df), atol=0)


def test_manifest_contains_required_metadata(deployment):
    _, path = deployment
    m = json.loads((path / "manifest.json").read_text())
    for key in (
        "deployment_threshold",
        "feature_names",
        "feature_schema",
        "grouping",
        "config",
        "seed",
        "library_versions",
        "sha256",
        "base_model_thresholds",
        "training",
    ):
        assert key in m, key
    schema = m["feature_schema"]
    assert schema["n_features"] == len(m["feature_names"])
    assert "selected_missing_indicators" in schema and "categorical_levels" in schema
    assert m["meta_feature_order"] == ["ft", "xgb", "lgb"]
    assert "For research and educational purposes only" in m["disclaimer"]


def test_tampered_artifact_is_rejected(deployment, tmp_path):
    import shutil

    _, path = deployment
    copy = tmp_path / "art"
    shutil.copytree(path, copy)
    with open(copy / "meta_learner.joblib", "ab") as fh:
        fh.write(b"0")
    with pytest.raises(ArtifactError):
        load_artifacts(copy)


def test_inference_does_not_need_training_objects(deployment, tiny_df):
    """The predictor works from files alone (fresh process simulation)."""
    import subprocess
    import sys

    _, path = deployment
    csv = path.parent / "input.csv"
    tiny_df.head(5).to_csv(csv, index=False)
    code = (
        "import sys; from varpath.artifacts import load_artifacts; import pandas as pd;"
        f"p=load_artifacts(r'{path}'); print(p.predict(pd.read_csv(r'{csv}'))['score'].round(6).tolist())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().startswith("[")
