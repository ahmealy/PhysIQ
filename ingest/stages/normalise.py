"""Stage 3: Compute normalization statistics (mean/std)."""
import numpy as np


def compute_stats(data: dict) -> dict:
    """
    Compute per-field mean and std over all samples.
    Returns dict with keys: vel_mean, vel_std, pos_mean, pos_std
    (pressure_mean, pressure_std if pressures present)
    """
    stats = {}
    vel = data["velocities"]  # [S, T, N, D]
    stats["vel_mean"] = float(vel.mean())
    stats["vel_std"] = float(vel.std()) + 1e-8
    pos = data["positions"]
    stats["pos_mean"] = float(pos.mean())
    stats["pos_std"] = float(pos.std()) + 1e-8
    if "pressures" in data:
        p = data["pressures"]
        stats["pressure_mean"] = float(p.mean())
        stats["pressure_std"] = float(p.std()) + 1e-8
    return stats


def normalise(data: dict) -> tuple[dict, dict]:
    """Compute stats. Return (data, stats_dict) — data arrays unchanged
    (normalisation is tracked, not applied in-place)."""
    stats = compute_stats(data)
    return data, stats
