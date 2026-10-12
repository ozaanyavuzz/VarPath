"""Gradient-boosted tree base learners (XGBoost, LightGBM).

Hyperparameters are tuned with Optuna on *inner* folds of a training
partition. The objective is the mean inner-validation log-loss; the outer
validation fold is not an argument of any function in this module.

LightGBM early stopping uses a group-disjoint hold-out carved out of each
inner training fold. The number of boosting rounds for the refit on the whole
training partition is the median of the inner best iterations of the selected
trial (robust to one unusually short/long inner run).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import log_loss

from ..config import GBDTConfig

logger = logging.getLogger("varpath")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class InnerFold:
    """One inner split of a training partition, already preprocessed."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    panel_train: np.ndarray
    es_fit: np.ndarray  # positions within the inner training fold
    es_hold: np.ndarray


@dataclass
class TuningResult:
    params: dict
    best_value: float
    oof: np.ndarray
    n_complete: int
    n_pruned: int
    best_iterations: list[int] = field(default_factory=list)


def positive_weight(y: np.ndarray, multiplier: float) -> float:
    y = np.asarray(y).astype(int)
    pos = max(int(y.sum()), 1)
    return float((y == 0).sum() / pos) * float(multiplier)


def max_depth_upper(n_samples: int, n_features: int, panels: np.ndarray, cfg: GBDTConfig) -> int:
    """Depth cap from the original prototype, computed from training data only."""
    ratio = n_samples / max(n_features, 1)
    upper = 4 if ratio < 1 else (5 if ratio < 3 else 9)
    if cfg.depth_cap_panel is not None and len(panels):
        frac = float(np.mean(np.asarray(panels).astype(str) == cfg.depth_cap_panel))
        if frac > cfg.depth_cap_min_fraction:
            upper = min(upper, cfg.depth_cap_value)
    return upper


def _depth_bounds(cfg: GBDTConfig, upper: int) -> tuple[int, int]:
    low = min(cfg.max_depth_min, upper)
    return low, upper


def suggest_xgb_params(trial: optuna.Trial, cfg: GBDTConfig, depth_upper: int) -> dict:
    lo, hi = _depth_bounds(cfg, depth_upper)
    return {
        "n_estimators": trial.suggest_int("n_estimators", *cfg.n_estimators),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.learning_rate, log=True),
        "max_depth": trial.suggest_int("max_depth", lo, hi),
        "subsample": trial.suggest_float("subsample", *cfg.subsample),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg.colsample_bytree),
        "reg_alpha": trial.suggest_float("reg_alpha", *cfg.reg_alpha, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *cfg.reg_lambda, log=True),
    }


def suggest_lgb_params(trial: optuna.Trial, cfg: GBDTConfig, depth_upper: int) -> dict:
    lo, hi = _depth_bounds(cfg, depth_upper)
    return {
        "n_estimators": trial.suggest_int("n_estimators", *cfg.n_estimators),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.learning_rate, log=True),
        "max_depth": trial.suggest_int("max_depth", lo, hi),
        "num_leaves": trial.suggest_int("num_leaves", *cfg.num_leaves),
        "subsample": trial.suggest_float("subsample", *cfg.subsample),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg.colsample_bytree),
        "reg_alpha": trial.suggest_float("reg_alpha", *cfg.reg_alpha, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *cfg.reg_lambda, log=True),
    }


def make_xgb(params: dict, pos_weight: float, seed: int, n_jobs: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )


def make_lgb(params: dict, pos_weight: float, seed: int, n_jobs: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **params,
        # Row subsampling only takes effect with a bagging frequency > 0.
        subsample_freq=1,
        scale_pos_weight=pos_weight,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )


def _study(cfg: GBDTConfig, seed: int) -> optuna.Study:
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=max(2, cfg.n_trials // 3), n_warmup_steps=0)
        if cfg.pruning
        else optuna.pruners.NopPruner()
    )
    return optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed), pruner=pruner
    )


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def tune_xgb(
    folds: list[InnerFold],
    n_rows: int,
    cfg: GBDTConfig,
    pos_weight: float,
    depth_upper: int,
    seed: int,
    n_jobs: int,
) -> TuningResult:
    best: dict = {"value": np.inf, "oof": None}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial, cfg, depth_upper)
        oof = np.full(n_rows, np.nan)
        losses = []
        for step, f in enumerate(folds):
            model = make_xgb(params, pos_weight, seed, n_jobs)
            model.fit(f.X_train, f.y_train, verbose=False)
            pred = model.predict_proba(f.X_val)[:, 1]
            oof[f.val_idx] = pred
            losses.append(_logloss(f.y_val, pred))
            trial.report(float(np.mean(losses)), step)
            if step < len(folds) - 1 and trial.should_prune():
                raise optuna.TrialPruned()
        value = float(np.mean(losses))
        if value < best["value"]:
            best.update(value=value, oof=oof)
        return value

    study = _study(cfg, seed)
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    states = [t.state for t in study.trials]
    return TuningResult(
        params=dict(study.best_params),
        best_value=float(study.best_value),
        oof=best["oof"],
        n_complete=states.count(optuna.trial.TrialState.COMPLETE),
        n_pruned=states.count(optuna.trial.TrialState.PRUNED),
    )


def tune_lgb(
    folds: list[InnerFold],
    n_rows: int,
    cfg: GBDTConfig,
    pos_weight: float,
    depth_upper: int,
    seed: int,
    n_jobs: int,
) -> TuningResult:
    best: dict = {"value": np.inf, "oof": None, "iters": None}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgb_params(trial, cfg, depth_upper)
        oof = np.full(n_rows, np.nan)
        losses, iters = [], []
        for step, f in enumerate(folds):
            model = make_lgb(params, pos_weight, seed, n_jobs)
            Xf, yf = f.X_train[f.es_fit], f.y_train[f.es_fit]
            Xh, yh = f.X_train[f.es_hold], f.y_train[f.es_hold]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    Xf,
                    yf,
                    eval_set=[(Xh, yh)],
                    eval_metric="binary_logloss",
                    callbacks=[
                        lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )
            n_iter = int(model.best_iteration_ or params["n_estimators"])
            iters.append(n_iter)
            pred = model.predict_proba(f.X_val, num_iteration=n_iter)[:, 1]
            oof[f.val_idx] = pred
            losses.append(_logloss(f.y_val, pred))
            trial.report(float(np.mean(losses)), step)
            if step < len(folds) - 1 and trial.should_prune():
                raise optuna.TrialPruned()
        value = float(np.mean(losses))
        trial.set_user_attr("best_iterations", iters)
        if value < best["value"]:
            best.update(value=value, oof=oof, iters=iters)
        return value

    study = _study(cfg, seed)
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    states = [t.state for t in study.trials]
    params = dict(study.best_params)
    iters = list(study.best_trial.user_attrs["best_iterations"])
    params["n_estimators"] = max(1, int(np.median(iters)))
    return TuningResult(
        params=params,
        best_value=float(study.best_value),
        oof=best["oof"],
        n_complete=states.count(optuna.trial.TrialState.COMPLETE),
        n_pruned=states.count(optuna.trial.TrialState.PRUNED),
        best_iterations=iters,
    )


def fit_xgb(X, y, params, pos_weight, seed, n_jobs) -> xgb.XGBClassifier:
    model = make_xgb(params, pos_weight, seed, n_jobs)
    model.fit(X, y, verbose=False)
    return model


def fit_lgb(X, y, params, pos_weight, seed, n_jobs) -> lgb.Booster:
    model = make_lgb(params, pos_weight, seed, n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model.booster_


def xgb_predict(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(float)


def lgb_predict(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(X), dtype=float)
