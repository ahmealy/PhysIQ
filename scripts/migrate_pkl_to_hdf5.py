#!/usr/bin/env python3
"""
scripts/migrate_pkl_to_hdf5.py — Migrate PKL result files to HDF5.

Usage:
    python scripts/migrate_pkl_to_hdf5.py [--result-dir result] [--dry-run] [--delete-pkl]
"""
import argparse
from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.pkl_repository import PklResultRepository
from storage.hdf5_repository import HDF5ResultRepository


def migrate(result_dir: Path, dry_run: bool, delete_pkl: bool):
    pkl_repo  = PklResultRepository(result_dir)
    hdf5_repo = HDF5ResultRepository(result_dir)
    names = pkl_repo.list()
    print(f"Found {len(names)} PKL files in {result_dir}")
    migrated, skipped, errors = 0, 0, 0
    for name in names:
        if hdf5_repo.exists(name):
            print(f"  SKIP  {name} (HDF5 already exists)")
            skipped += 1
            continue
        print(f"  {'DRY ' if dry_run else ''}MIGRATE  {name}")
        if not dry_run:
            try:
                preds, targets, coords, meta = pkl_repo.load(name)
                hdf5_repo.save(name, preds, targets, coords, meta)
                if delete_pkl:
                    pkl_repo.delete(name)
                    print(f"           deleted {name}.pkl")
                migrated += 1
            except Exception as e:
                print(f"  ERROR  {name}: {e}")
                errors += 1
    print(f"\nDone. migrated={migrated}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-pkl", action="store_true")
    args = parser.parse_args()
    migrate(Path(args.result_dir), args.dry_run, args.delete_pkl)
