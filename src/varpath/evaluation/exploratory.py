"""Exploratory panel-specific fine-tuning.

Panel-specific fine-tuning is exploratory and is not included in the primary
unbiased performance estimate.

The procedure follows the prototype's panel rules (continued boosting from the
fold's XGBoost model; a small fresh model on the 50 most important features for
CFTR), with two changes that keep it honest:

* everything is fitted on the outer-training rows of the current fold and
  scored on that fold's outer-validation rows of the same panel;
* only threshold-free metrics are reported (no threshold is tuned on panel
  validation rows), next to the un-tuned base XGBoost for reference.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import xgboost as xgb

from ..models.gbdt import xgb_predict
from .metrics import ranking_metrics

logger = logging.getLogger("varpath")

PANEL_RULES = {
    "pah": {"n_estimators": 200, "min_depth": 4},
    "cftr": {"fresh_model_top_k": 50},
}
MIN_PANEL_TRAIN = 20


def xgb_contributions(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Exact TreeSHAP contributions from XGBoost (last column = bias)."""
    booster = model.get_booster()
    return booster.predict(xgb.DMatrix(X), pred_contribs=True)


def _depth(n_rows: int, n_features: int) -> int:
    ratio = n_rows / max(n_features, 1)
    return 3 if ratio < 1 else (4 if ratio < 3 else 6)


def panel_finetune_fold(
    base_model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    panels_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    panels_val: np.ndarray,
    panel_names: list[str],
    seed: int,
    n_jobs: int = 1,
) -> list[dict]:
    rows = []
    for pname in panel_names:
        mtr = panels_train == pname
        mva = panels_val == pname
        rec = {"panel": pname, "n_train": int(mtr.sum()), "n_val": int(mva.sum()),
               "n_val_pos": int(y_val[mva].sum()) if mva.any() else 0}  # fmt: skip
        if mtr.sum() < MIN_PANEL_TRAIN or len(np.unique(y_train[mtr])) < 2 or not mva.any():
            rec["status"] = "skipped (insufficient data)"
            rows.append(rec)
            continue
        Xp, yp = X_train[mtr], y_train[mtr]
        spw = float((yp == 0).sum() / max((yp == 1).sum(), 1) + 1e-5)
        rule = PANEL_RULES.get(pname, {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if "fresh_model_top_k" in rule:
                imp = np.abs(xgb_contributions(base_model, Xp)[:, :-1]).mean(axis=0)
                top = np.sort(np.argsort(imp)[-rule["fresh_model_top_k"] :])
                model = xgb.XGBClassifier(
                    n_estimators=30, max_depth=2, reg_alpha=5.0, reg_lambda=5.0, subsample=0.7,
                    colsample_bytree=0.5, scale_pos_weight=spw, random_state=seed,
                    tree_method="hist", n_jobs=n_jobs, verbosity=0,
                )  # fmt: skip
                model.fit(Xp[:, top], yp)
                p_ft = model.predict_proba(X_val[mva][:, top])[:, 1]
                depth = 2
            else:
                depth = max(_depth(len(yp), Xp.shape[1]), rule.get("min_depth", 0))
                model = xgb.XGBClassifier(
                    n_estimators=rule.get("n_estimators", 50), max_depth=depth,
                    scale_pos_weight=spw, random_state=seed, tree_method="hist",
                    n_jobs=n_jobs, verbosity=0,
                )  # fmt: skip
                model.fit(Xp, yp, xgb_model=base_model.get_booster())
                p_ft = model.predict_proba(X_val[mva])[:, 1]
        p_base = xgb_predict(base_model, X_val[mva])
        tuned = ranking_metrics(y_val[mva], p_ft)
        base = ranking_metrics(y_val[mva], p_base)
        rec.update(
            {
                "status": "ok",
                "max_depth": depth,
                "roc_auc_finetuned": tuned["roc_auc"],
                "pr_auc_finetuned": tuned["pr_auc"],
                "roc_auc_base_xgb": base["roc_auc"],
                "pr_auc_base_xgb": base["pr_auc"],
            }
        )
        rows.append(rec)
    return rows
