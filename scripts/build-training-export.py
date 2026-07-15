#!/usr/bin/env python3
"""Build local training_frame.parquet from bronze listings CSV (ETL only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from house_price_ml.data.training_data import build_training_export  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export assembled training frame for make train")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "sample" / "listings.csv",
        help="Bronze listings CSV (default: data/sample/listings.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "sample" / "training_frame.parquet",
        help="Output parquet path (default: data/sample/training_frame.parquet)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        print("Run: make seed", file=sys.stderr)
        sys.exit(1)

    frame, metadata = build_training_export(args.input, args.output)
    print(f"Wrote {len(frame)} training rows to {args.output}")
    print(f"  rejected_rows={metadata['rejected_rows']}, source={metadata['source_file']}")


if __name__ == "__main__":
    main()
