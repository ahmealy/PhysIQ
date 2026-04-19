"""Stage 4: Write output files and update data/manifest.json."""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

MANIFEST_PATH = Path("data/manifest.json")


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"splits": {}, "created": datetime.now(timezone.utc).isoformat()}


def _save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))


def write_npz(data: dict, stats: dict, out_dir: Path, split: str) -> Path:
    """Save data + stats as .npz in out_dir. Update manifest.json. Returns Path to npz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_ingest.npz"
    np.savez_compressed(
        out_path,
        **{k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    )

    manifest = _load_manifest()
    manifest["splits"][split] = {
        "npz": str(out_path),
        "stats": stats,
        "num_trajectories": int(data["positions"].shape[0]),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)
    return out_path
