"""
scripts/regenerate_dat.py — Rebuild .dat memmap files from Zarr archive.

Since the exact .dat binary layout is complex and project-specific, this script:
  1. Reads arrays from ZarrArchive for the requested split.
  2. Saves them as numpy .npz files in the output directory.
  3. Prints instructions to re-run parse_tfrecord.py for full .dat regeneration.

Usage:
    python scripts/regenerate_dat.py \\
        --zarr-root data/zarr/cylinder_flow_fpc \\
        --out-dir   data/cylinder_flow \\
        --split     train
"""

import argparse
import sys
from pathlib import Path

# Make project root importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.zarr_archive import ZarrArchive  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate .dat files from Zarr archive.")
    p.add_argument("--zarr-root", required=True, help="Root directory of the Zarr archive.")
    p.add_argument("--out-dir", required=True, help="Output directory for .npz intermediates.")
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to regenerate (default: train).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    zarr_root = Path(args.zarr_root)
    out_dir = Path(args.out_dir)
    split = args.split

    archive = ZarrArchive(zarr_root)

    if not archive.exists(split):
        print(f"[ERROR] Split '{split}' not found in Zarr archive at {zarr_root}.")
        print("        Check that the sentinel file exists: "
              f"{zarr_root}/{split}.zarr.ok")
        sys.exit(1)

    print(f"Reading split '{split}' from {zarr_root} ...")
    result = archive.read_split(split)

    if len(result) == 4:
        positions, velocities, node_types, pressures = result
    else:
        positions, velocities, node_types = result
        pressures = None

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{split}_from_zarr.npz"

    save_kwargs = dict(positions=positions, velocities=velocities, node_types=node_types)
    if pressures is not None:
        save_kwargs["pressures"] = pressures

    import numpy as np
    np.savez_compressed(str(npz_path), **save_kwargs)
    print(f"Saved intermediate .npz → {npz_path}")
    print(f"  positions  : {positions.shape}  dtype={positions.dtype}")
    print(f"  velocities : {velocities.shape}  dtype={velocities.dtype}")
    print(f"  node_types : {node_types.shape}  dtype={node_types.dtype}")
    if pressures is not None:
        print(f"  pressures  : {pressures.shape}  dtype={pressures.dtype}")

    print()
    print("=" * 60)
    print("NOTE: This script produces .npz intermediates only.")
    print("To fully regenerate .dat memmap files, re-run parse_tfrecord.py")
    print("against the original TFRecord files, e.g.:")
    print()
    print("    python parse_tfrecord.py --dataset cylinder_flow \\")
    print("        --input-dir  <path-to-tfrecords> \\")
    print("        --output-dir data/cylinder_flow")
    print("=" * 60)


if __name__ == "__main__":
    main()
