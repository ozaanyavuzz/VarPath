"""Small shared helpers: seeding, device selection, logging, provenance."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

LOGGER_NAME = "varpath"
logger = logging.getLogger(LOGGER_NAME)

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "xgboost",
    "lightgbm",
    "torch",
    "optuna",
    "shap",
    "ripser",
    "joblib",
)


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger(LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
    root.setLevel(level)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def configure_torch_threads(n_threads: int | None) -> None:
    if n_threads is None:
        return
    import torch

    torch.set_num_threads(int(n_threads))


def resolve_device(name: str = "auto") -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return name


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except Exception:  # noqa: BLE001 - provenance must never crash a run
            versions[pkg] = "unavailable"
    return versions


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def timed(label: str, store: dict[str, float] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if store is not None:
            store[label] = store.get(label, 0.0) + elapsed
        logger.debug("%s took %.2fs", label, elapsed)


def derive_seed(base: int, *parts: int) -> int:
    """Deterministic child seed for nested loops (kept inside int32 range)."""
    value = int(base)
    for p in parts:
        value = (value * 1_000_003 + int(p) * 7_919 + 17) % 2_147_483_647
    return value
