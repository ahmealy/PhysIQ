"""
/dataset/info and /dataset/samples endpoints.
"""

import asyncio
import os
import sys

import numpy as np
from fastapi import APIRouter, HTTPException

from api.state import DOMAINS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

router = APIRouter(prefix="/dataset")

# Cache computed samples result so repeated page loads are instant
_samples_cache: dict = {}

# Cache for npz metadata (24 MB, slow to decompress) — keyed by npz path
_meta_cache: dict = {}


def _load_meta(npz_path: str):
    """Load and cache a .npz metadata file. Avoids re-decompressing on every request."""
    if npz_path not in _meta_cache:
        _meta_cache[npz_path] = np.load(npz_path, allow_pickle=True)
    return _meta_cache[npz_path]


@router.get("/info")
def dataset_info(domain: str = "cylinder_flow", split: str = "train"):
    if domain not in DOMAINS:
        raise HTTPException(404, "Unknown domain: %s" % domain)
    cfg = DOMAINS[domain]
    if not cfg["available"]:
        raise HTTPException(400, "Domain '%s' is not available yet" % domain)

    if domain == "flag_simple":
        data_dir = cfg["data_dir"]
        index_path = os.path.join(data_dir, f"{split}_index.npz")
        if not os.path.exists(index_path):
            raise HTTPException(404, "Cloth index not found: %s" % index_path)
        idx = np.load(index_path)
        n_traj = int(idx["n_traj"])
        steps = idx["steps_per_traj"]  # array of per-traj step counts
        timesteps = int(np.median(steps)) if hasattr(steps, '__len__') else int(steps)
        return {
            "domain": domain, "split": split,
            "num_trajectories": n_traj,
            "timesteps_per_trajectory": timesteps,
            "dt": cfg["dt"],
            "total_samples": int(np.sum(steps - 1)) if hasattr(steps, '__len__') else n_traj * (timesteps - 1),
        }
    else:
        data_dir = cfg["data_dir"]
        npz_path = os.path.join(data_dir, "%s.npz" % split)
        if not os.path.exists(npz_path):
            raise HTTPException(404, "Dataset not found: %s" % npz_path)
        meta = np.load(npz_path, allow_pickle=True)
        indices = meta["indices"]
        n_trajectories = len(indices) - 1
        velocity_shape = meta["all_velocity_shape"]
        timesteps = int(velocity_shape[1])
        return {
            "domain": domain, "split": split,
            "num_trajectories": n_trajectories,
            "timesteps_per_trajectory": timesteps,
            "dt": cfg["dt"],
            "total_samples": n_trajectories * (timesteps - 1),
        }


def _mesh_quality_stats(pos: np.ndarray, cells: np.ndarray) -> dict:
    """
    Compute triangle mesh quality metrics from node positions and connectivity.

    Args:
        pos:   [N, 2] or [N, 3] float32 — node coordinates
        cells: [F, 3] int32 — triangle face indices (local, 0-based)

    Returns dict with:
        aspect_ratio_mean, aspect_ratio_p95, aspect_ratio_max — edge length ratio stats
        n_degenerate  — triangles with zero or near-zero area (area < 1e-12)
        n_faces       — total face count
        quality_ok    — True if p95 aspect ratio < 10 and no degenerate elements
    """
    v0 = pos[cells[:, 0]]   # [F, 2or3]
    v1 = pos[cells[:, 1]]
    v2 = pos[cells[:, 2]]

    # Edge lengths
    e0 = np.linalg.norm(v1 - v0, axis=-1)   # [F]
    e1 = np.linalg.norm(v2 - v1, axis=-1)
    e2 = np.linalg.norm(v0 - v2, axis=-1)

    edges = np.stack([e0, e1, e2], axis=1)   # [F, 3]
    max_e = edges.max(axis=1)
    min_e = edges.min(axis=1)

    # Aspect ratio = longest / shortest edge (≥ 1 always; equilateral = 1)
    aspect = max_e / np.maximum(min_e, 1e-12)

    # Degenerate triangles: area ≈ 0
    # Use cross product magnitude for area (works for both 2D and 3D)
    if pos.shape[1] == 2:
        # 2D: signed area via 2D cross product
        cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) \
              - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        area = np.abs(cross) * 0.5
    else:
        # 3D: area = 0.5 * ||(v1-v0) × (v2-v0)||
        d1 = v1 - v0
        d2 = v2 - v0
        cross3 = np.cross(d1, d2)
        area = np.linalg.norm(cross3, axis=-1) * 0.5

    n_degenerate = int(np.sum(area < 1e-12))
    n_faces = len(cells)

    ar_mean = float(np.mean(aspect))
    ar_p95  = float(np.percentile(aspect, 95))
    ar_max  = float(np.max(aspect))

    quality_ok = (ar_p95 < 10.0) and (n_degenerate == 0)

    return {
        "aspect_ratio_mean": round(ar_mean, 2),
        "aspect_ratio_p95":  round(ar_p95,  2),
        "aspect_ratio_max":  round(ar_max,  2),
        "n_degenerate":      n_degenerate,
        "n_faces":           n_faces,
        "quality_ok":        quality_ok,
    }



    """CPU/IO-heavy computation for CFD domain — always called via run_in_executor."""
    meta = _load_meta(npz_path)
    indices = meta["indices"]
    n_trajectories = len(indices) - 1
    shape = tuple(meta["all_velocity_shape"])

    all_velocity = np.memmap(dat_path, dtype="float32", mode="r", shape=shape)
    vel_t0_all = np.array(all_velocity[:, 0, :])        # single contiguous memmap read
    vel_mag_all = np.linalg.norm(vel_t0_all, axis=-1)

    # Per-trajectory stats
    traj_means = np.array([
        float(vel_mag_all[int(indices[i]):int(indices[i + 1])].mean())
        for i in range(n_trajectories)
    ])
    # Node count per trajectory
    node_counts = np.array([
        int(indices[i + 1]) - int(indices[i])
        for i in range(n_trajectories)
    ])

    sample_size = min(20, n_trajectories)
    sampled = np.linspace(0, n_trajectories - 1, sample_size, dtype=int)
    sample_nodes = np.concatenate([
        vel_mag_all[int(indices[i]):int(indices[i + 1])] for i in sampled
    ])
    energy_vals = 0.5 * sample_nodes ** 2

    v_counts, v_edges = np.histogram(sample_nodes, bins=50)
    velocity_bins = [{"bin": round(float(v_edges[i]), 6), "count": int(v_counts[i])} for i in range(len(v_counts))]

    e_counts, e_edges = np.histogram(energy_vals, bins=50)
    energy_bins = [{"bin": round(float(e_edges[i]), 6), "count": int(e_counts[i])} for i in range(len(e_counts))]

    nc_counts, nc_edges = np.histogram(node_counts, bins=min(30, len(set(node_counts.tolist()))))
    node_count_bins = [{"bin": int(nc_edges[i]), "count": int(nc_counts[i])} for i in range(len(nc_counts))]

    mean_v = float(traj_means.mean())
    std_v  = float(traj_means.std()) if traj_means.std() > 0 else 1.0
    z_scores = (traj_means - mean_v) / std_v

    outliers = [
        {"trajectory": i, "mean_v": round(float(traj_means[i]), 6),
         "z_score": round(float(z_scores[i]), 3), "flag": bool(abs(z_scores[i]) > 3.0)}
        for i in range(n_trajectories)
    ]
    flagged = [o for o in outliers if o["flag"]]
    unflagged_sample = [o for o in outliers if not o["flag"]][:max(0, 50 - len(flagged))]
    outlier_table = sorted(flagged + unflagged_sample, key=lambda x: abs(x["z_score"]), reverse=True)

    # Node type breakdown from mesh metadata
    try:
        node_type_raw = meta["node_type"]  # [N_total, 1] or [N_total]
        nt = node_type_raw.flatten().astype(np.int32)
        node_type_counts = {}
        for name, val in [("NORMAL",0),("OBSTACLE",1),("AIRFOIL",2),("HANDLE",3),
                          ("INFLOW",4),("OUTFLOW",5),("WALL_BOUNDARY",6)]:
            c = int(np.sum(nt == val))
            if c > 0:
                node_type_counts[name] = c
    except Exception:
        node_type_counts = {}

    # Mesh quality — sample first trajectory
    mesh_quality: dict = {}
    try:
        if "pos" in meta and "cells" in meta and "cindices" in meta:
            c_start = int(meta["cindices"][0])
            c_end   = int(meta["cindices"][1])
            pos0    = meta["pos"][int(meta["indices"][0]):int(meta["indices"][1])].astype(np.float32)
            cells0  = meta["cells"][c_start:c_end].astype(np.int32)
            mesh_quality = _mesh_quality_stats(pos0, cells0)
    except Exception:
        pass

    return {
        "velocity_bins":    velocity_bins,
        "energy_bins":      energy_bins,
        "node_count_bins":  node_count_bins,
        "outliers":         outlier_table,
        "n_trajectories":   n_trajectories,
        "total_nodes":      int(node_counts.sum()),
        "mean_nodes":       round(float(node_counts.mean()), 1),
        "node_type_counts": node_type_counts,
        "mesh_quality":     mesh_quality,
    }


def _compute_samples_flag(data_dir: str, split: str) -> dict:
    """CPU-heavy computation for flag_simple (cloth) domain.
    Cloth data is stored per-trajectory as individual .npz files under {data_dir}/{split}/.
    Index file: {data_dir}/{split}_index.npz — contains n_traj, steps_per_traj.
    """
    index_path = os.path.join(data_dir, f"{split}_index.npz")
    split_dir  = os.path.join(data_dir, split)
    idx = np.load(index_path)
    n_traj = int(idx["n_traj"])
    steps_per_traj = idx["steps_per_traj"].tolist()

    # Sample up to 20 trajectories for stats
    sample_size = min(20, n_traj)
    sampled = np.linspace(0, n_traj - 1, sample_size, dtype=int)

    pos_mags = []
    node_counts = []

    for ti in range(n_traj):
        traj_path = os.path.join(split_dir, f"traj_{ti:05d}.npz")
        if not os.path.exists(traj_path):
            node_counts.append(0)
            continue
        try:
            traj = np.load(traj_path)
            world_pos = traj["world_pos"]   # [T, N, 3]
            N = world_pos.shape[1]
            node_counts.append(N)
            if ti in sampled:
                pos_t0 = world_pos[0]       # [N, 3]
                mag = np.linalg.norm(pos_t0, axis=-1)  # [N]
                pos_mags.append(mag)
        except Exception:
            node_counts.append(0)

    node_counts_arr = np.array(node_counts, dtype=np.int32)
    if pos_mags:
        sample_nodes = np.concatenate(pos_mags)
    else:
        sample_nodes = np.zeros(1)

    traj_means = np.array([
        float(np.linalg.norm(np.load(os.path.join(split_dir, f"traj_{i:05d}.npz"))["world_pos"][0], axis=-1).mean())
        if os.path.exists(os.path.join(split_dir, f"traj_{i:05d}.npz")) else 0.0
        for i in range(n_traj)
    ])
    energy_vals = 0.5 * sample_nodes ** 2

    v_counts, v_edges = np.histogram(sample_nodes, bins=50)
    velocity_bins = [{"bin": round(float(v_edges[i]), 6), "count": int(v_counts[i])} for i in range(len(v_counts))]

    e_counts, e_edges = np.histogram(energy_vals, bins=50)
    energy_bins = [{"bin": round(float(e_edges[i]), 6), "count": int(e_counts[i])} for i in range(len(e_counts))]

    nc_unique = len(set(node_counts_arr.tolist()))
    nc_counts, nc_edges = np.histogram(node_counts_arr, bins=min(30, max(1, nc_unique)))
    node_count_bins = [{"bin": int(nc_edges[i]), "count": int(nc_counts[i])} for i in range(len(nc_counts))]

    mean_v = float(traj_means.mean())
    std_v  = float(traj_means.std()) if traj_means.std() > 0 else 1.0
    z_scores = (traj_means - mean_v) / std_v

    outliers = [
        {"trajectory": i, "mean_v": round(float(traj_means[i]), 6),
         "z_score": round(float(z_scores[i]), 3), "flag": bool(abs(z_scores[i]) > 3.0)}
        for i in range(n_traj)
    ]
    flagged    = [o for o in outliers if o["flag"]]
    unflagged  = [o for o in outliers if not o["flag"]][:max(0, 50 - len(flagged))]
    outlier_table = sorted(flagged + unflagged, key=lambda x: abs(x["z_score"]), reverse=True)

    valid_nc = node_counts_arr[node_counts_arr > 0]

    # Node type breakdown — load from first available trajectory
    node_type_counts = {}
    mesh_quality: dict = {}
    try:
        first_path = os.path.join(split_dir, "traj_00000.npz")
        if os.path.exists(first_path):
            traj0 = np.load(first_path)
            nt = traj0["node_type"].flatten().astype(np.int32)
            for name, val in [("NORMAL",0),("HANDLE",3)]:
                c = int(np.sum(nt == val))
                if c > 0:
                    node_type_counts[name] = c
            # Mesh quality on world_pos at t=0 (3D cloth shape)
            if "world_pos" in traj0 and "cells" in traj0:
                pos3d  = traj0["world_pos"][0].astype(np.float32)   # [N, 3]
                cells0 = traj0["cells"].astype(np.int32)             # [F, 3]
                mesh_quality = _mesh_quality_stats(pos3d, cells0)
    except Exception:
        pass

    return {
        "velocity_bins":    velocity_bins,
        "energy_bins":      energy_bins,
        "node_count_bins":  node_count_bins,
        "outliers":         outlier_table,
        "n_trajectories":   n_traj,
        "total_nodes":      int(valid_nc.sum()) if len(valid_nc) else 0,
        "mean_nodes":       round(float(valid_nc.mean()), 1) if len(valid_nc) else 0,
        "node_type_counts": node_type_counts,
        "mesh_quality":     mesh_quality,
    }


@router.get("/samples")
async def dataset_samples(domain: str = "cylinder_flow", split: str = "test"):
    """Return velocity/energy histograms and Z-score outlier table.
    Heavy numpy work runs in a thread so the event loop stays responsive."""
    cache_key = (domain, split)
    if cache_key in _samples_cache:
        return _samples_cache[cache_key]

    if domain not in DOMAINS:
        raise HTTPException(404, "Unknown domain: %s" % domain)

    cfg = DOMAINS[domain]
    if not cfg["available"]:
        raise HTTPException(400, "Domain '%s' is not available yet" % domain)

    data_dir = cfg["data_dir"]

    # Branch: cloth (flag_simple) uses per-trajectory npz files; CFD uses memmap
    if domain == "flag_simple":
        index_path = os.path.join(data_dir, f"{split}_index.npz")
        if not os.path.exists(index_path):
            raise HTTPException(404, "Cloth index file not found: %s" % index_path)
        result = await asyncio.get_event_loop().run_in_executor(
            None, _compute_samples_flag, data_dir, split
        )
    else:
        npz_path = os.path.join(data_dir, "%s.npz" % split)
        if not os.path.exists(npz_path):
            raise HTTPException(404, "Dataset not found: %s" % npz_path)
        dat_path = npz_path.replace(".npz", ".dat")
        if not os.path.exists(dat_path):
            raise HTTPException(404, "Velocity data file not found: %s" % dat_path)
        result = await asyncio.get_event_loop().run_in_executor(
            None, _compute_samples, npz_path, dat_path
        )
    _samples_cache[cache_key] = result
    return result


@router.get("/mesh_preview")
async def mesh_preview(domain: str = "cylinder_flow", trajectory: int = 0):
    """Return mesh positions, faces, and field values for one sample frame for visualization."""
    if domain not in DOMAINS:
        raise HTTPException(404, "Unknown domain: %s" % domain)
    cfg = DOMAINS[domain]
    if not cfg["available"]:
        raise HTTPException(400, "Domain '%s' is not available yet" % domain)

    if domain == "flag_simple":
        data_dir = cfg["data_dir"]
        traj_path = os.path.join(data_dir, "train", f"traj_{trajectory:05d}.npz")
        if not os.path.exists(traj_path):
            raise HTTPException(404, "Trajectory not found: %s" % traj_path)
        traj = np.load(traj_path)
        faces = traj["cells"].astype(int).tolist()         # [F, 3]
        world_pos_t0 = traj["world_pos"][0]                # [N, 3]  — actual cloth shape at t=0
        # Use world_pos x,y as 2D display coordinates (real cloth shape, not flat UV grid)
        positions_2d = world_pos_t0[:, :2].tolist()        # [N, 2]  x,y in world space
        # Color by z (height) — shows the 3D drape in the 2D projection
        field_values = world_pos_t0[:, 2].tolist()         # [N]     z coordinate
        node_type = traj["node_type"].flatten().astype(int).tolist()
        return {
            "positions": positions_2d,
            "faces": faces,
            "field_values": field_values,
            "node_type": node_type,
            "n_nodes": len(positions_2d),
            "n_faces": len(faces),
        }
    else:
        data_dir = cfg["data_dir"]
        npz_path = os.path.join(data_dir, "train.npz")
        if not os.path.exists(npz_path):
            raise HTTPException(404, "Dataset not found")
        # Use cached meta — avoids re-decompressing the 24 MB npz on every request
        meta = _load_meta(npz_path)
        indices = meta["indices"]
        if trajectory >= len(indices) - 1:
            raise HTTPException(400, "Trajectory index out of range")
        start, end = int(indices[trajectory]), int(indices[trajectory + 1])
        pos = meta["pos"][start:end].tolist()          # [N, 2]
        node_type = meta["node_type"][start:end].flatten().astype(int).tolist()

        # Slice cells for this trajectory using cindices (cell index array, parallel to indices).
        # Cell node indices are already local (0-based within each trajectory).
        if "cells" in meta and "cindices" in meta:
            c_start = int(meta["cindices"][trajectory])
            c_end   = int(meta["cindices"][trajectory + 1])
            cells_local = meta["cells"][c_start:c_end].tolist()
        elif "cells" in meta:
            # Fallback: cells use global indices — filter and remap
            cells_raw = meta["cells"].astype(np.int32)
            mask = (cells_raw >= start) & (cells_raw < end)
            cells_local = (cells_raw[mask.all(axis=1)] - start).tolist()
        else:
            cells_local = []

        # Load velocity at t=0 via memmap — only reads the needed slice
        velocity_shape = tuple(meta["all_velocity_shape"])
        dat_path = npz_path.replace(".npz", ".dat")
        all_velocity = np.memmap(dat_path, dtype="float32", mode="r", shape=velocity_shape)
        vel_t0 = np.array(all_velocity[start:end, 0, :])  # [N, 2]
        field_values = np.linalg.norm(vel_t0, axis=-1).tolist()

        return {
            "positions":    pos,
            "faces":        cells_local,
            "field_values": field_values,
            "node_type":    node_type,
            "n_nodes":      end - start,
            "n_faces":      len(cells_local),
        }


@router.post("/flag_outliers")
async def flag_outliers(domain: str = "cylinder_flow", split: str = "test"):
    """
    Write result/outlier_mask_{split}.npy — a boolean array [n_trajectories]
    where True = flagged as outlier (|z-score| > 3σ).
    Clears the samples cache so next load reflects any data changes.
    """
    cache_key = (domain, split)
    if cache_key not in _samples_cache:
        # Need to compute first
        if domain not in DOMAINS:
            raise HTTPException(404, "Unknown domain: %s" % domain)
        cfg = DOMAINS[domain]
        data_dir = cfg["data_dir"]
        npz_path = os.path.join(data_dir, "%s.npz" % split)
        dat_path = npz_path.replace(".npz", ".dat")
        result = await asyncio.get_event_loop().run_in_executor(
            None, _compute_samples, npz_path, dat_path
        )
        _samples_cache[cache_key] = result

    data = _samples_cache[cache_key]
    outliers = data["outliers"]
    n = data["n_trajectories"]

    mask = np.zeros(n, dtype=bool)
    for o in outliers:
        if o["flag"]:
            mask[o["trajectory"]] = True

    os.makedirs("result", exist_ok=True)
    mask_path = os.path.join("result", f"outlier_mask_{split}.npy")
    np.save(mask_path, mask)

    flagged_ids = [o["trajectory"] for o in outliers if o["flag"]]
    return {
        "status":      "saved",
        "path":        mask_path,
        "n_flagged":   int(mask.sum()),
        "flagged_ids": flagged_ids,
    }
