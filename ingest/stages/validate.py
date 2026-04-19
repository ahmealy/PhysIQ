"""Stage 2: Validate shapes, dtypes, and required keys."""
import numpy as np

REQUIRED_KEYS = {"positions", "velocities", "node_types"}


def validate(data: dict, split: str) -> dict:
    """
    Checks:
    - Required keys present
    - Arrays are numpy ndarrays
    - positions.shape[0] == velocities.shape[0] (same number of trajectories)
    - node_types.ndim == 2
    Returns data unchanged if valid, raises ValueError with descriptive message if not.
    """
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(
            f"[{split}] Missing required keys: {missing}. "
            f"Got: {set(data.keys())}"
        )

    for key in REQUIRED_KEYS:
        if not isinstance(data[key], np.ndarray):
            raise ValueError(
                f"[{split}] Expected numpy ndarray for '{key}', "
                f"got {type(data[key]).__name__}"
            )

    if data["positions"].shape[0] != data["velocities"].shape[0]:
        raise ValueError(
            f"[{split}] Trajectory count mismatch: "
            f"positions has {data['positions'].shape[0]} trajectories but "
            f"velocities has {data['velocities'].shape[0]}."
        )

    if data["node_types"].ndim != 2:
        raise ValueError(
            f"[{split}] node_types must be 2-dimensional [S, N], "
            f"got shape {data['node_types'].shape}"
        )

    return data
