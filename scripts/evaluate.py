"""Thin wrapper around `varpath.cli.evaluate_main` (usable without installation)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from varpath.cli import evaluate_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(evaluate_main())
