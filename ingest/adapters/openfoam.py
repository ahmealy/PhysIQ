"""OpenFOAM adapter stub — not yet implemented."""
from pathlib import Path


class OpenFOAMAdapter:
    name = "OpenFOAM"

    def __init__(self, data_dir: str | Path = "."):
        self._data_dir = Path(data_dir)

    @property
    def source_path(self) -> Path:
        return self._data_dir

    def list_splits(self) -> list[str]:
        raise NotImplementedError(
            "OpenFOAM adapter not yet implemented. "
            "Contribute at github.com/meshGraphNets/meshGraphNets_pytorch"
        )

    def load_split(self, split: str) -> dict:
        raise NotImplementedError(
            "OpenFOAM adapter not yet implemented. "
            "Contribute at github.com/meshGraphNets/meshGraphNets_pytorch"
        )
