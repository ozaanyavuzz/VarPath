"""Topological descriptors of a variant's feature profile.

Each row's numeric annotation vector (in column order) is treated as a 1-D
signal, sub-sampled to ``n_points`` values and delay-embedded in 2-D. Vietoris-
Rips persistent homology (H0, H1) is computed with ``ripser`` and summarised by
13 statistics.

The transform itself is row-local. The only training-derived quantity is how
missing entries are filled before embedding; that step lives in
:class:`varpath.features.preprocessing.FoldPreprocessor`, where the fill values
are learned from the training partition only.
"""

from __future__ import annotations

import hashlib
import logging
import warnings

import numpy as np

logger = logging.getLogger("varpath")

try:  # ripser is a declared dependency; the fallback only keeps imports working
    from ripser import ripser as _ripser

    RIPSER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ripser = None
    RIPSER_AVAILABLE = False

TDA_FEATURE_NAMES = [
    # Counts of finite bars; named after Betti numbers in the original prototype.
    "tda_b0", "tda_b1", "tda_euler",
    "tda_h0_mean", "tda_h0_max", "tda_h0_entropy", "tda_h0_total", "tda_h0_std",
    "tda_h1_mean", "tda_h1_max", "tda_h1_entropy", "tda_h1_total", "tda_h1_std",
]  # fmt: skip
N_TDA = len(TDA_FEATURE_NAMES)


def persistence_stats(dgm: np.ndarray) -> list[float]:
    """Mean, max, entropy, total and std of finite positive lifetimes."""
    if len(dgm) == 0:
        return [0.0] * 5
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life) & (life > 0)]
    if len(life) == 0:
        return [0.0] * 5
    total = life.sum() + 1e-12
    frac = life / total
    entropy = float(-np.sum(frac * np.log(frac + 1e-12)))
    return [
        float(np.mean(life)),
        float(np.max(life)),
        entropy,
        float(life.sum()),
        float(np.std(life)) if len(life) > 1 else 0.0,
    ]


def tda_features(signal: np.ndarray, n_points: int = 50) -> np.ndarray:
    """13 persistent-homology summaries of a 1-D signal (non-finite values dropped)."""
    if not RIPSER_AVAILABLE:
        return np.zeros(N_TDA, dtype=np.float32)
    sig = np.asarray(signal, dtype=float)
    sig = sig[np.isfinite(sig)]
    if len(sig) < 4:
        return np.zeros(N_TDA, dtype=np.float32)
    if len(sig) > n_points:
        sig = sig[np.linspace(0, len(sig) - 1, n_points, dtype=int)]
    pts = np.column_stack([sig[:-1], sig[1:]])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _ripser(pts, maxdim=1)
        dgm0, dgm1 = res["dgms"][0], res["dgms"][1]
        b0 = float(np.sum(np.isfinite(dgm0[:, 1]))) if len(dgm0) else 0.0
        b1 = float(np.sum(np.isfinite(dgm1[:, 1]))) if len(dgm1) else 0.0
        return np.array(
            [b0, b1, b0 - b1] + persistence_stats(dgm0) + persistence_stats(dgm1),
            dtype=np.float32,
        )
    except Exception:  # noqa: BLE001 - degenerate point clouds
        return np.zeros(N_TDA, dtype=np.float32)


class TDAFeatureExtractor:
    """Row-wise TDA with an in-memory cache keyed on the exact filled signal.

    Nested CV evaluates the same row under several fold-specific fill values;
    rows without missing entries produce identical signals and hit the cache.
    """

    def __init__(self, n_points: int = 50, use_cache: bool = True, max_cache: int = 200_000):
        self.n_points = int(n_points)
        self.use_cache = use_cache
        self.max_cache = max_cache
        self._cache: dict[bytes, np.ndarray] = {}
        if not RIPSER_AVAILABLE:
            logger.warning("ripser is not installed; TDA features will be zeros")

    def _key(self, row: np.ndarray) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        h.update(np.ascontiguousarray(row, dtype=np.float64).tobytes())
        h.update(self.n_points.to_bytes(4, "little"))
        return h.digest()

    def transform(self, signals: np.ndarray) -> np.ndarray:
        signals = np.asarray(signals, dtype=float)
        out = np.zeros((signals.shape[0], N_TDA), dtype=np.float32)
        for i, row in enumerate(signals):
            if not self.use_cache:
                out[i] = tda_features(row, self.n_points)
                continue
            key = self._key(row)
            cached = self._cache.get(key)
            if cached is None:
                cached = tda_features(row, self.n_points)
                if len(self._cache) < self.max_cache:
                    self._cache[key] = cached
            out[i] = cached
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state
