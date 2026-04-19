from pathlib import Path
from typing import Union

_CONFIG_PATH = Path("runs/storage_config.json")
_DEFAULT_RESULT_DIR = Path("result")


def _load_config() -> dict:
    """Load storage config from runs/storage_config.json if it exists."""
    if _CONFIG_PATH.exists():
        import json
        return json.loads(_CONFIG_PATH.read_text())
    return {}


class StorageFactory:
    @staticmethod
    def create(result_dir: Union[Path, str, None] = None) -> "ResultRepository":
        cfg = _load_config()
        backend = cfg.get("result_backend", "pkl")
        rdir = Path(result_dir) if result_dir else Path(cfg.get("result_dir", str(_DEFAULT_RESULT_DIR)))

        if backend == "hdf5":
            from storage.hdf5_repository import HDF5ResultRepository
            return HDF5ResultRepository(rdir)
        else:
            from storage.pkl_repository import PklResultRepository
            return PklResultRepository(rdir)


def get_repository(result_dir=None):
    """Convenience function — returns the configured repository."""
    return StorageFactory.create(result_dir)
