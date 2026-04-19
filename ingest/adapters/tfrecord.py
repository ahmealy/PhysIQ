"""TFRecord adapter — reads already-parsed .dat memmap files."""
import numpy as np
from pathlib import Path


class TFRecordAdapter:
    name = "TFRecord"

    def __init__(self, data_dir: str | Path, domain: str = "cylinder_flow"):
        self._data_dir = Path(data_dir)
        self._domain = domain

    @property
    def source_path(self) -> Path:
        return self._data_dir

    def list_splits(self) -> list[str]:
        return ["train", "valid", "test"]

    def load_split(self, split: str) -> dict:
        """Load split from existing .dat memmap files. Raises RuntimeError if not parsed yet."""
        data_path = self._data_dir / f"{split}.dat"
        meta_path = self._data_dir / f"{split}.npz"

        if not data_path.exists() or not meta_path.exists():
            raise RuntimeError(
                f"Parsed data not found for split '{split}' in {self._data_dir}.\n"
                "Run parse_tfrecord.py first to generate the .dat and .npz files:\n"
                "  python parse_tfrecord.py --data_dir <raw_data_dir> --output_dir <output_dir>"
            )

        # Check sentinel (indicates complete, non-interrupted parse)
        sentinel = str(data_path) + ".ok"
        if not Path(sentinel).exists():
            raise RuntimeError(
                f"Missing parse sentinel: {sentinel}\n"
                "The .dat file may be incomplete (parse_tfrecord.py was interrupted).\n"
                "Re-run parse_tfrecord.py to regenerate the data files."
            )

        meta_keys = ("pos", "node_type", "cells", "indices", "cindices", "all_velocity_shape")
        tmp = np.load(meta_path, allow_pickle=True)
        meta = {key: tmp[key] for key in meta_keys if key in tmp}

        vel_shape = meta["all_velocity_shape"]  # [total_nodes, T, D]
        velocities_raw = np.memmap(data_path, dtype="float32", mode="r", shape=vel_shape)

        # Reconstruct per-trajectory arrays using indices
        indices = meta["indices"]
        num_trajectories = len(indices) - 1
        T = vel_shape[1]
        N_per_traj = indices[1] - indices[0]  # assume uniform node count
        D = vel_shape[2]

        velocities = np.stack([
            velocities_raw[indices[i]:indices[i + 1]]  # [N, T, D]
            for i in range(num_trajectories)
        ], axis=0)  # [S, N, T, D] -> transpose to [S, T, N, D]
        velocities = velocities.transpose(0, 2, 1, 3)  # [S, T, N, D]

        # positions: use pos metadata (static per trajectory)
        pos = meta["pos"]  # [total_nodes, D]
        positions_list = []
        for i in range(num_trajectories):
            p = pos[indices[i]:indices[i + 1]]  # [N, D]
            # Repeat for T+1 timesteps (static mesh positions)
            positions_list.append(np.stack([p] * (T + 1), axis=0))  # [T+1, N, D]
        positions = np.stack(positions_list, axis=0)  # [S, T+1, N, D]

        node_type = meta["node_type"]  # [total_nodes, 1]
        node_types = np.stack([
            node_type[indices[i]:indices[i + 1]]  # [N, 1]
            for i in range(num_trajectories)
        ], axis=0)  # [S, N, 1]
        node_types = node_types.squeeze(-1)  # [S, N]

        result = {
            "positions": positions.astype(np.float32),
            "velocities": velocities.astype(np.float32),
            "node_types": node_types.astype(np.float32),
            "metadata": {k: v for k, v in meta.items()},
        }

        # Optional pressure field
        pressure_path = self._data_dir / f"{split}_pressure.dat"
        if pressure_path.exists():
            pressure_sentinel = str(pressure_path) + ".ok"
            if Path(pressure_sentinel).exists():
                pressure_shape = (vel_shape[0], vel_shape[1], 1)
                pressures_raw = np.memmap(pressure_path, dtype="float32", mode="r",
                                          shape=pressure_shape)
                pressures = np.stack([
                    pressures_raw[indices[i]:indices[i + 1]]  # [N, T, 1]
                    for i in range(num_trajectories)
                ], axis=0)  # [S, N, T, 1]
                pressures = pressures.transpose(0, 2, 1, 3)  # [S, T, N, 1]
                result["pressures"] = pressures.astype(np.float32)

        return result
