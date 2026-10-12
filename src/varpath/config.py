"""Typed experiment configuration loaded from YAML.

Every tunable quantity of the pipeline lives here so that a run is fully
described by one YAML file (``configs/quick.yaml``, ``standard.yaml``,
``full.yaml``). Unknown keys raise an error instead of being ignored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data"
    # Ordered mapping panel name -> CSV file. The order defines the class
    # indices of the FT-Transformer auxiliary panel head.
    panel_files: dict[str, str] = field(
        default_factory=lambda: {
            "master": "YARISMA_TRAIN_MASTER.csv",
            "kanser": "YARISMA_TRAIN_KANSER.csv",
            "pah": "YARISMA_TRAIN_PAH.csv",
            "cftr": "YARISMA_TRAIN_CFTR.csv",
        }
    )
    label_column: str = "Label"
    id_column: str = "Variant_ID"
    aa_ref_column: str = "AA_1"
    aa_alt_column: str = "AA_2"
    panel_column: str = "panel"
    categorical_prefix: str = "CAT_"
    exclude_columns: list[str] = field(default_factory=list)

    @property
    def panels(self) -> list[str]:
        return list(self.panel_files)

    @property
    def reserved_columns(self) -> set[str]:
        return {
            self.label_column,
            self.id_column,
            self.aa_ref_column,
            self.aa_alt_column,
            self.panel_column,
            *self.exclude_columns,
        }

    def validate(self) -> None:
        if not self.panel_files:
            raise ValueError("data.panel_files must contain at least one panel")


@dataclass
class GroupingConfig:
    # "column": use a real entity column (gene, protein, transcript, variant).
    # "row_hash": duplicate-profile grouping; sensitivity analysis only.
    strategy: str = "column"
    column: str | None = "Variant_ID"
    # Semantic level of ``column``; reported in outputs and README tables.
    entity_level: str = "variant"
    row_hash_decimals: int = 3
    # Legacy prototype used ``hash % 200``; ``None`` keeps exact duplicate groups.
    row_hash_buckets: int | None = None

    def validate(self) -> None:
        if self.strategy not in {"column", "row_hash"}:
            raise ValueError("grouping.strategy must be 'column' or 'row_hash'")
        if self.strategy == "column" and not self.column:
            raise ValueError("grouping.column is required when strategy='column'")
        if self.entity_level not in {"gene", "protein", "transcript", "variant", "other"}:
            raise ValueError("grouping.entity_level has an unsupported value")


@dataclass
class CVConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    stratify_by: list[str] = field(default_factory=lambda: ["label", "panel"])
    # Group-aware hold-out carved from a *training* partition for early stopping.
    early_stopping_fraction: float = 0.15

    def validate(self) -> None:
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("outer_folds and inner_folds must be >= 2")
        if not 0.05 <= self.early_stopping_fraction <= 0.5:
            raise ValueError("cv.early_stopping_fraction must be in [0.05, 0.5]")
        for key in self.stratify_by:
            if key not in {"label", "panel"}:
                raise ValueError("cv.stratify_by accepts 'label' and 'panel'")


@dataclass
class PreprocessingConfig:
    numeric_parse_min_ratio: float = 0.3
    drop_constant_numeric: bool = True
    imputer_max_iter: int = 7
    imputer_n_nearest_features: int = 20
    # Posterior sampling makes a row's imputed value depend on the other rows in
    # the batch and on call order; single deterministic imputation is used by default.
    imputer_sample_posterior: bool = False
    ohe_min_frequency: int = 3
    missing_mi_threshold: float = 0.001
    missing_mi_discrete: bool = False
    drop_conflicts: bool = True
    conflict_decimals: int = 3
    tda_enabled: bool = True
    tda_n_points: int = 50
    # "train_median": fill gaps with training-fold medians (legacy semantics, fitted in-fold)
    # "observed_only": use only observed values of the row (fully row-local)
    tda_missing_strategy: str = "train_median"
    float32: bool = True

    def validate(self) -> None:
        if self.tda_missing_strategy not in {"train_median", "observed_only"}:
            raise ValueError("preprocessing.tda_missing_strategy is invalid")


@dataclass
class GBDTConfig:
    n_trials: int = 10
    pruning: bool = True
    pos_weight_multiplier: float = 1.5
    n_estimators: list[int] = field(default_factory=lambda: [100, 700])
    learning_rate: list[float] = field(default_factory=lambda: [0.01, 0.1])
    max_depth_min: int = 3
    num_leaves: list[int] = field(default_factory=lambda: [15, 127])
    subsample: list[float] = field(default_factory=lambda: [0.6, 1.0])
    colsample_bytree: list[float] = field(default_factory=lambda: [0.6, 1.0])
    reg_alpha: list[float] = field(default_factory=lambda: [0.1, 50.0])
    reg_lambda: list[float] = field(default_factory=lambda: [0.1, 50.0])
    early_stopping_rounds: int = 30
    # Depth cap of the prototype: if the named panel exceeds ``depth_cap_min_fraction``
    # of the training rows, max_depth is capped at ``depth_cap_value``.
    depth_cap_panel: str | None = "cftr"
    depth_cap_min_fraction: float = 0.01
    depth_cap_value: int = 3

    def validate(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")


@dataclass
class FTConfig:
    d_token: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.3
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 128
    eval_batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    mtl_alpha: float = 0.7
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    pos_weight_multiplier: float = 1.5
    # Optional training-fold feature pre-selection for the transformer only
    # (ANOVA F-score). ``None`` feeds every engineered feature as a token.
    token_budget: int | None = None
    amp: bool = True

    def validate(self) -> None:
        if self.d_token % self.n_heads != 0:
            raise ValueError("ft.d_token must be divisible by ft.n_heads")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("ft.max_epochs and ft.patience must be >= 1")


@dataclass
class StackingConfig:
    C: float = 1.0
    max_iter: int = 500
    fit_calibrator: bool = True


@dataclass
class ThresholdConfig:
    beta: float = 2.0
    min_specificity: float = 0.60
    default: float = 0.5


@dataclass
class EvaluationConfig:
    bootstrap_iterations: int = 1000
    bootstrap_unit: str = "group"
    ci_alpha: float = 0.05
    panel_min_n: int = 5
    panel_exploratory_n: int = 30
    calibration_bins: int = 10
    dca_min: float = 0.01
    dca_max: float = 0.80
    dca_points: int = 80
    panel_finetune: bool = True
    variant_network: bool = True

    def validate(self) -> None:
        if self.bootstrap_unit not in {"group", "row"}:
            raise ValueError("evaluation.bootstrap_unit must be 'group' or 'row'")


@dataclass
class ShapConfig:
    enabled: bool = True
    n_samples: int = 300
    max_display: int = 20


@dataclass
class ComputeConfig:
    n_jobs: int = 1
    device: str = "auto"
    torch_threads: int | None = None
    cache_tda: bool = True


@dataclass
class ExperimentConfig:
    profile: str = "standard"
    seed: int = 42
    output_dir: str = "outputs"
    artifacts_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    xgb: GBDTConfig = field(default_factory=GBDTConfig)
    lgb: GBDTConfig = field(default_factory=GBDTConfig)
    ft: FTConfig = field(default_factory=FTConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    def validate(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value) and hasattr(value, "validate"):
                value.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, data: dict[str, Any] | None, path: str = "") -> Any:
    template = cls()
    if data is None:
        return template
    if not isinstance(data, dict):
        raise TypeError(f"Config section '{path or cls.__name__}' must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown config keys in '{path or 'root'}': {unknown}")
    kwargs = {}
    for key, value in data.items():
        default = getattr(template, key)
        if is_dataclass(default):
            kwargs[key] = _build(type(default), value, f"{path}.{key}".strip("."))
        else:
            kwargs[key] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    cfg = _build(ExperimentConfig, data)
    cfg.validate()
    return cfg


def _deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config, optionally deep-merging ``overrides`` on top."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        data = _deep_update(data, overrides)
    return config_from_dict(data)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
