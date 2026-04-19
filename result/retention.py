"""
result/retention.py — Result directory cleanup utility.

Usage:
    python -m result.retention --keep 10
    python -m result.retention --keep 10 --dry-run
    python -m result.retention --keep 10 --result-dir /path/to/result

Keeps the N most-recently modified result files and deletes the rest.
Dry-run mode (default) prints what would be deleted without removing anything.
"""

import argparse
import os
import sys
from pathlib import Path


RESULT_EXTENSIONS = (".pkl", ".h5")   # support both legacy PKL and future HDF5


def prune(result_dir: Path, keep: int = 10, dry_run: bool = True) -> list[Path]:
    """
    Delete oldest result files, keeping the `keep` most recent.

    Args:
        result_dir:  Directory containing result files.
        keep:        Number of most-recent files to keep.
        dry_run:     If True, print actions without deleting.

    Returns:
        List of files that were (or would be) deleted.
    """
    if not result_dir.exists():
        print(f"[retention] Result directory does not exist: {result_dir}", file=sys.stderr)
        return []

    files = [
        p for p in result_dir.iterdir()
        if p.is_file() and p.suffix in RESULT_EXTENSIONS
    ]

    if not files:
        print(f"[retention] No result files found in {result_dir}")
        return []

    # Sort by modification time — oldest first
    files.sort(key=lambda p: p.stat().st_mtime)

    to_delete = files[:-keep] if keep > 0 else files
    to_keep   = files[-keep:] if keep > 0 else []

    print(f"[retention] Found {len(files)} file(s). Keeping {len(to_keep)}, "
          f"{'would delete' if dry_run else 'deleting'} {len(to_delete)}.")

    for f in to_delete:
        size_mb = f.stat().st_size / (1024 * 1024)
        if dry_run:
            print(f"  [dry-run] would delete: {f.name}  ({size_mb:.1f} MB)")
        else:
            f.unlink()
            print(f"  [deleted] {f.name}  ({size_mb:.1f} MB)")

    if dry_run and to_delete:
        print("[retention] Run without --dry-run to actually delete.")

    return to_delete


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune old result files, keeping the N most recent."
    )
    parser.add_argument(
        "--keep", type=int, default=10,
        help="Number of most-recent result files to keep (default: 10)"
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("result"),
        help="Path to result directory (default: result/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print what would be deleted without removing anything"
    )
    args = parser.parse_args()

    if args.keep < 0:
        parser.error("--keep must be >= 0")

    prune(args.result_dir, keep=args.keep, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
