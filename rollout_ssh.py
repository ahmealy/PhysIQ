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
import datetime
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


def _ts() -> str:
    """Compact timestamp suffix: YYYYMMDD_HHMMSS"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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

    _raw_graph = dataset[traj_idx * n_steps]
    faces = _raw_graph.face.numpy().T.astype(np.int32)  # [F, 3]

    # Initialise Poisson corrector once before the loop (opt-in, CFD only)
    corrector = None
    if req.get("poisson_correction", False):
        from physics.poisson_pressure import PoissonPressureCorrector
        crds_init = _raw_graph.pos.numpy()  # [N, 2]
        corrector = PoissonPressureCorrector(crds_init)
        print("Poisson pressure corrector initialised (LU factorised)", flush=True)

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

            # GT velocity at t — read BEFORE any overwrite (always from dataset)
            gt_velocity_t = graph.x[:, field_slice].detach().cpu().numpy()

            if predicted_velocity is not None:
                graph.x[:, field_slice] = predicted_velocity.detach()

            # Autoregressive input at t: GT at t=0, own prediction from t=1 onward
            rollout_velocity_t = graph.x[:, field_slice].detach().cpu().numpy()

            next_v = graph.y
            predicted_velocity = model(graph, velocity_sequence_noise=None)

            # Apply Poisson pressure correction before boundary pin (opt-in)
            if corrector is not None:
                vel_np = predicted_velocity.detach().cpu().numpy()  # [N, 2]
                vel_np = corrector.correct(vel_np)
                predicted_velocity = torch.from_numpy(vel_np).to(predicted_velocity.device)

            predicted_velocity[boundary_mask] = next_v[boundary_mask]

            # predicteds[t] = autoregressive input at t  (matches DeepMind trajectory[t])
            # targets[t]    = GT at t                    (matches DeepMind inputs['velocity'][t])
            predicteds.append(rollout_velocity_t)
            targets_list.append(gt_velocity_t)

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

    # Confidence score — requires domain-scoped embedding_index built after training
    confidence_score = None
    _confidence_debug = ""
    _domain_slug = domain.replace("_", "")  # cylinderflow, flagsimple
    index_path = os.path.join("runs", "embedding_index_%s.pkl" % _domain_slug)
    _confidence_debug += "cwd=%s index=%s exists=%s" % (os.getcwd(), index_path, os.path.exists(index_path))
    if os.path.exists(index_path):
        try:
            from confidence.index import NearestNeighborIndex, IndexStaleError
            from model.embedding import extract_embedding

            _index = NearestNeighborIndex.load(
                index_path, expected_checkpoint=checkpoint_path)
            # Dual-frame embedding: concat(frame_0, frame_warmup)
            # frame_0    → geometry + boundary conditions
            # frame_warmup → early flow dynamics
            _warmup = min(5, n_steps - 1)
            _g0 = dataset[traj_idx * n_steps]
            _gw = dataset[traj_idx * n_steps + _warmup]
            _g0 = transformer(_g0)
            _gw = transformer(_gw)
            _emb = extract_embedding(model, _g0, device=device, graph_warmup=_gw)
            confidence_score = float(_index.query(_emb))
            _confidence_debug += " frame=0+%d emb_norm=%.4f diameter=%.4f score=%.4f" % (
                _warmup, float((_emb ** 2).sum() ** 0.5), _index.train_diameter, confidence_score)
        except IndexStaleError as _ce:
            _confidence_debug += " STALE_INDEX=%s" % _ce
        except Exception as _ce:
            _confidence_debug += " ERROR=%s" % _ce

    os.makedirs("result", exist_ok=True)
    pkl_path = "result/result_traj%d_%s.pkl" % (traj_idx, _ts())
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], crds, {
            "domain":             domain,
            "target_field":       target_field,
            "confidence_score":   confidence_score,
            "faces":              faces,
            "speedup":            round(speedup, 2),
            "elapsed_seconds":    round(elapsed, 3),
            "poisson_correction": req.get("poisson_correction", False),
        }], f)

    return {
        "elapsed_seconds":   round(elapsed, 3),
        "speedup":           round(speedup, 2),
        "pkl_path":          pkl_path,
        "rmse_final":        float(per_step_rmse[-1]),
        "similarity_score":  None,
        "confidence_score":  round(confidence_score, 3) if confidence_score is not None else None,
        "_confidence_debug": _confidence_debug,
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

            # Record cur_world BEFORE stepping — matches DeepMind cloth_eval.py:
            #   trajectory[t] = cur_pos (input at t), compared against GT world_pos at t
            gt_world_t = dataset[idx].world_pos.numpy()   # GT at t, unaffected by rollout
            predicteds.append(graph.world_pos.detach().cpu().numpy())
            targets_list.append(gt_world_t)

            prev_world = graph.world_pos.clone()
            next_world = model(graph)

            # Pin HANDLE nodes to cur_pos (fully autoregressive, matches DeepMind)
            node_type = graph.x[:, 3].long()
            handle_mask = (node_type != NodeType.NORMAL)
            next_world[handle_mask] = graph.world_pos[handle_mask]

            cur_world = next_world

            if i % 20 == 0 or i == n_steps - 1:
                _sse({"type": "progress", "step": i + 1, "total": n_steps})

    elapsed = time.perf_counter() - t_start
    sim_time = n_steps * 0.01
    speedup  = sim_time / (elapsed + 1e-12)

    predicted_arr = np.stack(predicteds)
    targets_arr   = np.stack(targets_list)

    # Load mesh_pos and cells directly from npz — FlagDataset has no mesh_pos_list attribute
    _traj_path = os.path.join(dataset._split_dir, "traj_%05d.npz" % traj_idx)
    mesh_pos = np.load(_traj_path)["mesh_pos"].astype(np.float32)
    cells    = np.load(_traj_path)["cells"].astype(np.int32)

    sq = np.square(predicted_arr - targets_arr).reshape(n_steps, -1)
    per_step_rmse = np.sqrt(np.mean(sq, axis=1))

    # Confidence score — requires domain-scoped embedding_index built after training
    confidence_score = None
    _confidence_debug = ""
    index_path = os.path.join("runs", "embedding_index_flagsimple.pkl")
    _confidence_debug += "cwd=%s index=%s exists=%s" % (os.getcwd(), index_path, os.path.exists(index_path))
    if os.path.exists(index_path):
        try:
            from confidence.index import NearestNeighborIndex, IndexStaleError
            from model.embedding import extract_embedding

            _cloth_ckpt = domain_cfg.get("checkpoint", "")
            _index = NearestNeighborIndex.load(
                index_path, expected_checkpoint=_cloth_ckpt)
            # Cloth frame 0 IS the design (world_pos at t=0) — correct to use it
            _first_idx = int(dataset._cum_steps[traj_idx])
            _first_graph = dataset[_first_idx]
            _emb = extract_embedding(model, _first_graph, device=device)
            confidence_score = float(_index.query(_emb))
            _confidence_debug += " emb_norm=%.4f diameter=%.4f score=%.4f" % (
                float((_emb ** 2).sum() ** 0.5), _index.train_diameter, confidence_score)
        except IndexStaleError as _ce:
            _confidence_debug += " STALE_INDEX=%s" % _ce
        except Exception as _ce:
            _confidence_debug += " ERROR=%s" % _ce

    os.makedirs("result", exist_ok=True)
    pkl_path = "result/flag_result_traj%d_%s.pkl" % (traj_idx, _ts())
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], mesh_pos, {
            "domain":           "flag_simple",
            "target_field":     "world_pos",
            "confidence_score": confidence_score,
            "faces":            cells,
            "speedup":          round(speedup, 2),
            "elapsed_seconds":  round(elapsed, 3),
        }], f)

    return {
        "elapsed_seconds":   round(elapsed, 3),
        "speedup":           round(speedup, 2),
        "pkl_path":          pkl_path,
        "rmse_final":        float(per_step_rmse[-1]),
        "similarity_score":  None,
        "confidence_score":  round(confidence_score, 3) if confidence_score is not None else None,
        "_confidence_debug": _confidence_debug,
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
