"""IngestPipeline — orchestrates stages for one or more splits."""
from pathlib import Path
from ingest.protocols import SolverAdapter
from ingest.stages import harvest, validate, normalise, write, index


class IngestPipeline:
    def __init__(self, adapter: SolverAdapter, out_dir: str | Path = "data"):
        self._adapter = adapter
        self._out_dir = Path(out_dir)

    def run(self, splits: list[str] | None = None, rebuild_index: bool = True) -> dict:
        """
        Run full pipeline for each split.
        Returns summary dict: {split: {"status": "ok"|"error", "npz": path, ...}}
        """
        if splits is None:
            splits = self._adapter.list_splits()
        results = {}
        for split in splits:
            try:
                data = harvest.harvest(self._adapter, split)
                data = validate.validate(data, split)
                data, stats = normalise.normalise(data)
                npz_path = write.write_npz(data, stats, self._out_dir, split)
                results[split] = {"status": "ok", "npz": str(npz_path)}
                print(f"  ✓ {split}: {data['positions'].shape[0]} trajectories → {npz_path}")
            except Exception as e:
                results[split] = {"status": "error", "error": str(e)}
                print(f"  ✗ {split}: {e}")

        if rebuild_index:
            index.rebuild_confidence_index()

        return results
