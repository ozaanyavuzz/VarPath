"""Write artificial panel CSVs with the competition layout (no real data involved)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from varpath.config import DataConfig  # noqa: E402
from varpath.synthetic import write_synthetic_panels  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--n-al", type=int, default=40, help="number of AL_* columns (real files: 334)")
    ap.add_argument("--n-ek", type=int, default=4, help="number of EK_* columns (real files: 9)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = write_synthetic_panels(
        args.out_dir, DataConfig().panel_files, n_al=args.n_al, n_ek=args.n_ek, seed=args.seed
    )
    for panel, p in paths.items():
        print(panel, p)
