"""
rollout_ssh.py — SSH-compatible rollout runner.

Reads a JSON config file (same schema as runs/ui_rollout_config.json produced by
the FastAPI server), runs the GNN rollout, and prints SSE-format lines to stdout.

Usage:
    python -u rollout_ssh.py --config /abs/path/to/runs/ui_rollout_config.json

stdout lines:
    data: {"type":"progress","step":N,"total":T}\n\n
    data: {"type":"done",...}\n\n      OR
    data: {"type":"error","message":"..."}\n\n

The -u flag is required to disable output buffering.
"""

import sys
# Force line-buffered stdout so SSE lines flush immediately over SSH
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch_geometric.transforms as T

# Allow imports from project root (same logic as the API server)
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def _sse(obj: dict) -> None:
    """Print a single SSE event to stdout."""
    print("data: %s\n" % json.dumps(obj), flush=True)


def _run_rollout(cfg: dict, req: dict) -> dict:
    """CFD (cylinder_flow) rollout — mirrors _run_rollout_sync in api/routes/rollout.py."""
    from api.state import DOMAINS, get_model
    from dataset import FpcDataset
    from utils.utils import NodeType

    domain = req["domain"]
    domain_cfg = DOMAINS[domain]
    device = req.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    checkpoint_path = domain_cfg["checkpoint"]
    model = get_model(checkpoint_path, device)
    target_field = getattr(model, "target_field", "velocity")
    field_slice = slice(1, 2) if target_field == "pressure" else slice(1, 3)

    dataset = FpcDataset(domain_cfg["data_dir"], split=req.get("split", "test"))
    n_steps = dataset.num_sampes_per_tra
    traj_idx = req["trajectory_index"]

    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    predicted_velocity = None
    boundary_mask = None
    predicteds, targets_list = [], []
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(n_steps):
            idx = traj_idx * n_steps + i
            graph = dataset[idx]
            graph = transformer(graph)
            graph = graph.to(device)

            if boundary_mask is None:
                node_type = graph.x[:, 0]
                fluid = torch.logical_or(
                    node_type == NodeType.NORMAL,
                    node_type == NodeType.OUTFLOW,
                )
                boundary_mask = torch.logical_not(fluid)

            if predicted_velocity is not None:
                graph.x[:, field_slice] = predicted_velocity.detach()

            next_v = graph.y
            predicted_velocity = model(graph, velocity_sequence_noise=None)
            predicted_velocity[boundary_mask] = next_v[boundary_mask]

            predicteds.append(predicted_velocity.detach().cpu().numpy())
            targets_list.append(next_v.detach().cpu().numpy())

            if i % 20 == 0 or i == n_steps - 1:
                _sse({"type": "progress", "step": i + 1, "total": n_steps})

    elapsed = time.perf_counter() - t_start
    sim_time = n_steps * domain_cfg.get("dt", 0.01)
    speedup = sim_time / (elapsed + 1e-12)

    predicted_arr = np.stack(predicteds)
    targets_arr   = np.stack(targets_list)
    crds          = graph.pos.cpu().numpy()

    sq_diff       = np.square(predicted_arr - targets_arr).reshape(n_steps, -1)
    per_step_rmse = np.sqrt(np.mean(sq_diff, axis=1))

    os.makedirs("result", exist_ok=True)
    pkl_path = "result/result%d.pkl" % traj_idx
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], crds, {
            "domain":       domain,
            "target_field": target_field,
        }], f)

    return {
        "elapsed_seconds": round(elapsed, 3),
        "speedup":         round(speedup, 2),
        "pkl_path":        pkl_path,
        "rmse_final":      float(per_step_rmse[-1]),
        "similarity_score": None,
        "confidence_score": None,
    }


def _run_cloth_rollout(cfg: dict, req: dict) -> dict:
    """Cloth (flag_simple) rollout — mirrors _run_cloth_rollout_sync in api/routes/rollout.py."""
    from api.state import DOMAINS, get_model
    from dataset.flag_dataset import FlagDataset
    from utils.utils import NodeType

    domain_cfg = DOMAINS["flag_simple"]
    device = req.get("device", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = get_model(domain_cfg["checkpoint"], device)
    data_dir = domain_cfg.get("data_dir", "data_flag")
    dataset = FlagDataset(data_dir, split=req.get("split", "test"))
    traj_idx = req["trajectory_index"]
    n_steps = int(dataset.steps_per_traj[traj_idx])

    prev_world = None
    cur_world  = None
    predicteds, targets_list = [], []
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(n_steps):
            idx = int(dataset._cum_steps[traj_idx]) + i
            graph = dataset[idx]
            graph = graph.to(device)

            if cur_world is not None:
                graph.world_pos = cur_world.detach()
                graph.x = torch.cat([cur_world.detach(), graph.x[:, 3:]], dim=-1)
                graph.prev_x = prev_world.detach()

            prev_world = graph.world_pos.clone()
            next_world = model(graph)
            node_type = graph.x[:, 3].long()
            handle_mask = (node_type == NodeType.HANDLE)
            next_world[handle_mask] = graph.y[handle_mask]

            predicteds.append(next_world.detach().cpu().numpy())
            targets_list.append(graph.y.detach().cpu().numpy())
            cur_world = next_world

            if i % 20 == 0 or i == n_steps - 1:
                _sse({"type": "progress", "step": i + 1, "total": n_steps})

    elapsed = time.perf_counter() - t_start
    sim_time = n_steps * 0.01
    speedup  = sim_time / (elapsed + 1e-12)

    predicted_arr = np.stack(predicteds)
    targets_arr   = np.stack(targets_list)
    mesh_pos = np.asarray(dataset.mesh_pos_list[traj_idx], dtype=np.float32)

    sq = np.square(predicted_arr - targets_arr).reshape(n_steps, -1)
    per_step_rmse = np.sqrt(np.mean(sq, axis=1))

    os.makedirs("result", exist_ok=True)
    pkl_path = "result/flag_result%d.pkl" % traj_idx
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], mesh_pos, {
            "domain":       "flag_simple",
            "target_field": "velocity",
        }], f)

    return {
        "elapsed_seconds": round(elapsed, 3),
        "speedup":         round(speedup, 2),
        "pkl_path":        pkl_path,
        "rmse_final":      float(per_step_rmse[-1]),
        "similarity_score": None,
        "confidence_score": None,
    }


def main():
    parser = argparse.ArgumentParser(description="SSH-compatible rollout runner")
    parser.add_argument("--config", required=True, help="Path to ui_rollout_config.json")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            req = json.load(f)
    except Exception as e:
        _sse({"type": "error", "message": "Cannot read config: %s" % e})
        sys.exit(1)

    # Change to project root so relative paths (checkpoints/, data/, result/) work
    os.chdir(_ROOT)

    try:
        if req.get("domain") == "flag_simple":
            result = _run_cloth_rollout({}, req)
        else:
            result = _run_rollout({}, req)
        _sse({"type": "done", **result})
    except Exception as e:
        import traceback
        _sse({"type": "error", "message": "%s\n%s" % (e, traceback.format_exc())})
        sys.exit(1)


if __name__ == "__main__":
    main()
