"""Leakage-safe preprocessing.

:class:`FoldPreprocessor` owns *every* data-dependent preprocessing decision:

* which raw columns are numeric (parse ratio on training rows),
* which numeric columns are constant/empty (training rows),
* MICE imputation and robust scaling,
* amino-acid/structural block scaling,
* TDA gap filling (training medians) and TDA block scaling,
* supervised selection of missingness indicators (training labels only),
* one-hot encoding of ``CAT_*`` columns.

``fit`` must only ever receive a training partition; ``transform`` never
updates state. The same fitted object is serialised for deployment, so the
training and inference feature pipelines are identical by construction.

Output block order (kept from the original prototype):
``[numeric, amino-acid + structural, TDA, missingness, categorical]``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from ..config import DataConfig, PreprocessingConfig
from .amino_acid import AA_FEATURE_NAMES, aa_features, normalize_residue_codes, pairwise_matrix
from .structural import STRUCT_FEATURE_NAMES, structural_features
from .tda import TDA_FEATURE_NAMES, TDAFeatureExtractor

logger = logging.getLogger("varpath")

MISSING_TOKEN = "MISSING"


class NotFittedError(RuntimeError):
    pass


class SchemaError(ValueError):
    pass


def _aa_block(aa1: np.ndarray, aa2: np.ndarray) -> np.ndarray:
    return np.hstack(
        [pairwise_matrix(aa1, aa2, aa_features), pairwise_matrix(aa1, aa2, structural_features)]
    )


def candidate_columns(df: pd.DataFrame, data_cfg: DataConfig) -> tuple[list[str], list[str]]:
    """Split raw columns into (numeric candidates, categorical) by name only."""
    reserved = data_cfg.reserved_columns
    cols = [c for c in df.columns if c not in reserved]
    cat = [c for c in cols if str(c).startswith(data_cfg.categorical_prefix)]
    num = [c for c in cols if c not in set(cat)]
    return num, cat


def coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Column-wise ``to_numeric`` returning a float64 frame."""
    out = {}
    for c in frame.columns:
        col = frame[c]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            out[c] = col.astype("float64")
        else:
            out[c] = pd.to_numeric(col, errors="coerce").astype("float64")
    return pd.DataFrame(out, index=frame.index)


def detect_label_conflicts(
    df: pd.DataFrame, y: np.ndarray, data_cfg: DataConfig, decimals: int = 3
) -> np.ndarray:
    """Rows whose rounded numeric profile occurs with more than one label.

    Operates only on the rows it is given; callers pass training partitions so
    that a training row is never dropped because of a validation label.
    """
    num_cols, _ = candidate_columns(df, data_cfg)
    if not num_cols:
        return np.zeros(len(df), dtype=bool)
    prof = coerce_numeric(df[num_cols]).round(decimals)
    keys = pd.util.hash_pandas_object(prof, index=False).to_numpy()
    tmp = pd.DataFrame({"k": keys, "y": np.asarray(y)})
    return (tmp.groupby("k")["y"].transform("nunique") > 1).to_numpy()


class FoldPreprocessor:
    """Fit on a training partition, transform any partition."""

    def __init__(
        self,
        data_cfg: DataConfig,
        cfg: PreprocessingConfig,
        seed: int = 42,
        tda_extractor: TDAFeatureExtractor | None = None,
    ):
        self.data_cfg = data_cfg
        self.cfg = cfg
        self.seed = int(seed)
        self._tda = tda_extractor
        self.fitted_ = False

    # ------------------------------------------------------------------ helpers
    @property
    def tda(self) -> TDAFeatureExtractor:
        if self._tda is None:
            self._tda = TDAFeatureExtractor(self.cfg.tda_n_points)
        return self._tda

    def _check_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_num = [c for c in self.numeric_columns_ if c not in df.columns]
        if missing_num:
            raise SchemaError(
                f"Input is missing {len(missing_num)} numeric column(s) seen during training, "
                f"e.g. {missing_num[:5]}"
            )
        for col in (self.data_cfg.aa_ref_column, self.data_cfg.aa_alt_column):
            if col not in df.columns:
                raise SchemaError(f"Input is missing amino-acid column '{col}'")
        return df

    def _numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        return coerce_numeric(df[self.numeric_columns_])

    def _tda_signals(self, num: pd.DataFrame) -> np.ndarray:
        arr = np.array(num, dtype=float)
        if self.cfg.tda_missing_strategy == "train_median":
            fill = np.broadcast_to(self.tda_fill_values_, arr.shape)
            arr = np.where(np.isnan(arr), fill, arr)
        return arr

    def _categorical(self, df: pd.DataFrame, col: str) -> np.ndarray:
        if col in df.columns:
            s = df[col].astype("object")
            s = s.where(s.notna(), MISSING_TOKEN).astype(str)
        else:
            s = pd.Series([MISSING_TOKEN] * len(df), dtype=str)
        return s.to_numpy().reshape(-1, 1)

    # --------------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> FoldPreprocessor:
        y = np.asarray(y).astype(int)
        if len(df) != len(y):
            raise ValueError("df and y length mismatch")
        cfg = self.cfg
        num_cands, cat_cols = candidate_columns(df, self.data_cfg)

        # 1) numeric typing decided on training rows only
        raw = df[num_cands]
        conv = coerce_numeric(raw)
        keep = []
        for c in num_cands:
            n_obs = int(raw[c].notna().sum())
            if n_obs == 0:
                keep.append(c)  # all-missing in training: only its missingness flag can matter
                continue
            if conv[c].notna().sum() / n_obs >= cfg.numeric_parse_min_ratio:
                keep.append(c)
        self.numeric_columns_ = keep
        num = conv[keep]

        # 2) value columns entering imputation/scaling
        nunique = num.nunique(dropna=True)
        min_unique = 2 if cfg.drop_constant_numeric else 1
        self.value_columns_ = [c for c in keep if nunique[c] >= min_unique]
        self.dropped_numeric_columns_ = [c for c in keep if c not in set(self.value_columns_)]

        # 3) MICE + RobustScaler
        self.imputer_ = None
        self.num_scaler_ = None
        if self.value_columns_:
            self.imputer_ = IterativeImputer(
                random_state=self.seed,
                max_iter=cfg.imputer_max_iter,
                n_nearest_features=min(cfg.imputer_n_nearest_features, len(self.value_columns_)),
                sample_posterior=cfg.imputer_sample_posterior,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp = self.imputer_.fit_transform(np.array(num[self.value_columns_], dtype=float))
            self.num_scaler_ = RobustScaler().fit(imp)

        # 4) amino-acid + structural block
        aa1 = normalize_residue_codes(df[self.data_cfg.aa_ref_column])
        aa2 = normalize_residue_codes(df[self.data_cfg.aa_alt_column])
        self.aa_scaler_ = RobustScaler().fit(_aa_block(aa1, aa2))

        # 5) TDA (fill values learned here; transform is row-local)
        self.tda_scaler_ = None
        if cfg.tda_enabled:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.tda_fill_values_ = np.nanmedian(num.to_numpy(dtype=float), axis=0)
            self.tda_scaler_ = RobustScaler().fit(self.tda.transform(self._tda_signals(num)))

        # 6) supervised missingness-indicator selection (training labels only)
        flags = num.isna().to_numpy().astype(np.int8)
        self.selected_missing_columns_: list[str] = []
        if flags.shape[1] and len(np.unique(y)) == 2:
            varying = flags.std(axis=0) > 0
            mi = np.zeros(flags.shape[1])
            if varying.any():
                mi[varying] = mutual_info_classif(
                    flags[:, varying],
                    y,
                    discrete_features=cfg.missing_mi_discrete,
                    random_state=self.seed,
                )
            self.missing_mi_ = dict(zip(keep, mi.tolist()))
            self.selected_missing_columns_ = [
                c for c, v in zip(keep, mi) if v > cfg.missing_mi_threshold
            ]
        self._missing_positions = [keep.index(c) for c in self.selected_missing_columns_]

        # 7) categorical one-hot encoders
        self.categorical_columns_ = cat_cols
        self.encoders_: dict[str, OneHotEncoder] = {}
        cat_names: list[str] = []
        for col in cat_cols:
            enc = OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=cfg.ohe_min_frequency,
                sparse_output=False,
                dtype=np.float32,
            )
            enc.fit(self._categorical(df, col))
            self.encoders_[col] = enc
            cat_names.extend(enc.get_feature_names_out([col]).tolist())

        # 8) feature schema
        blocks = {
            "numeric": list(self.value_columns_),
            "amino_acid": AA_FEATURE_NAMES + STRUCT_FEATURE_NAMES,
            "tda": TDA_FEATURE_NAMES if cfg.tda_enabled else [],
            "missing": [f"mis_{c}" for c in self.selected_missing_columns_],
            "categorical": cat_names,
        }
        self.feature_names_out_: list[str] = []
        self.block_slices_: dict[str, tuple[int, int]] = {}
        for name, cols in blocks.items():
            start = len(self.feature_names_out_)
            self.feature_names_out_.extend(cols)
            self.block_slices_[name] = (start, len(self.feature_names_out_))
        self.input_columns_ = (
            list(self.numeric_columns_)
            + list(cat_cols)
            + [self.data_cfg.aa_ref_column, self.data_cfg.aa_alt_column]
        )
        self.n_train_ = len(df)
        self.fitted_ = True
        return self

    # --------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise NotFittedError("FoldPreprocessor.transform called before fit")
        self._check_columns(df)
        n = len(df)
        num = self._numeric(df)
        blocks: list[np.ndarray] = []

        if self.value_columns_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp = self.imputer_.transform(np.array(num[self.value_columns_], dtype=float))
            blocks.append(self.num_scaler_.transform(imp))
        else:
            blocks.append(np.zeros((n, 0)))

        aa1 = normalize_residue_codes(df[self.data_cfg.aa_ref_column])
        aa2 = normalize_residue_codes(df[self.data_cfg.aa_alt_column])
        blocks.append(self.aa_scaler_.transform(_aa_block(aa1, aa2)))

        if self.cfg.tda_enabled:
            blocks.append(self.tda_scaler_.transform(self.tda.transform(self._tda_signals(num))))

        flags = num.isna().to_numpy()
        blocks.append(flags[:, self._missing_positions].astype(np.float32))

        for col in self.categorical_columns_:
            blocks.append(self.encoders_[col].transform(self._categorical(df, col)))

        dtype = np.float32 if self.cfg.float32 else np.float64
        out = np.hstack([np.asarray(b, dtype=float) for b in blocks]).astype(dtype)
        if out.shape[1] != len(self.feature_names_out_):
            raise RuntimeError(
                f"Transformed width {out.shape[1]} != schema width {len(self.feature_names_out_)}"
            )
        return out

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return self.fit(df, y).transform(df)

    def schema(self) -> dict:
        if not self.fitted_:
            raise NotFittedError("schema requested before fit")
        return {
            "input_numeric_columns": list(self.numeric_columns_),
            "input_categorical_columns": list(self.categorical_columns_),
            "amino_acid_columns": [self.data_cfg.aa_ref_column, self.data_cfg.aa_alt_column],
            "value_columns": list(self.value_columns_),
            "dropped_constant_numeric_columns": list(self.dropped_numeric_columns_),
            "selected_missing_indicators": list(self.selected_missing_columns_),
            "categorical_levels": {
                c: [str(x) for x in enc.categories_[0]] for c, enc in self.encoders_.items()
            },
            "feature_names": list(self.feature_names_out_),
            "block_slices": {k: list(v) for k, v in self.block_slices_.items()},
            "n_features": len(self.feature_names_out_),
            "tda_enabled": self.cfg.tda_enabled,
            "tda_missing_strategy": self.cfg.tda_missing_strategy,
        }
