"""Shared fixtures: tiny synthetic data and a CPU-friendly configuration."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from varpath.config import config_from_dict  # noqa: E402
from varpath.data import build_groups  # noqa: E402
from varpath.synthetic import synthetic_dataframe  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]

TINY = {
    "profile": "test",
    "seed": 11,
    "cv": {"outer_folds": 3, "inner_folds": 2},
    "preprocessing": {"imputer_max_iter": 1, "tda_n_points": 20},
    "xgb": {"n_trials": 2, "n_estimators": [10, 20], "pruning": False},
    "lgb": {"n_trials": 2, "n_estimators": [10, 20], "pruning": False, "early_stopping_rounds": 5},
    "ft": {"d_token": 8, "n_heads": 2, "n_layers": 1, "max_epochs": 2, "patience": 1,
           "batch_size": 64, "token_budget": 12},
    "evaluation": {"bootstrap_iterations": 20, "panel_finetune": False, "variant_network": False},
    "shap": {"n_samples": 30},
    "compute": {"device": "cpu"},
}  # fmt: skip
SIZES = {"master": 90, "kanser": 30, "pah": 30, "cftr": 24}


def make_cfg(**overrides):
    import copy

    data = copy.deepcopy(TINY)
    for k, v in overrides.items():
        if isinstance(v, dict):
            data.setdefault(k, {}).update(v)
        else:
            data[k] = v
    return config_from_dict(data)


@pytest.fixture(scope="session")
def cfg():
    return make_cfg()


@pytest.fixture(scope="session")
def tiny_df():
    return synthetic_dataframe(sizes=SIZES, n_al=15, n_ek=3, n_cat=3, seed=3)


@pytest.fixture(scope="session")
def arrays(tiny_df, cfg):
    y = tiny_df[cfg.data.label_column].to_numpy().astype(int)
    panels = tiny_df[cfg.data.panel_column].astype(str).to_numpy()
    groups = build_groups(tiny_df, cfg.grouping, cfg.data)
    return y, groups, panels


@pytest.fixture(scope="session")
def deployment(tiny_df, cfg, arrays, tmp_path_factory):
    """A deployment pipeline fitted once on all tiny rows and persisted."""
    from varpath.artifacts import save_artifacts
    from varpath.models.stacking import fit_stacked_pipeline

    y, groups, panels = arrays
    pipe = fit_stacked_pipeline(tiny_df, y, groups, panels, cfg, seed=cfg.seed, n_splits=2)
    path = tmp_path_factory.mktemp("artifacts")
    save_artifacts(pipe, cfg, path)
    return pipe, path


def assert_allclose(a, b, tol=1e-6):
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0
    )
