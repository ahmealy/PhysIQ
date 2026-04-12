import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as T
from tqdm import tqdm

from dataset import FpcDataset
from model.simulator import Simulator
from utils.utils import NodeType
from model.flag_simulator import FlagSimulator
from dataset.flag_dataset import FlagDataset


def rollout_error(predicteds, targets, rollout_index=0, save_dir='result'):
    """
    Compute per-step RMSE between predicted and target velocity fields.
    Saves a plot to save_dir/rollout_error_<rollout_index>.png.

    Returns:
        per_step_rmse: np.ndarray of shape [T] — RMSE at each timestep
    """
    number_len = targets.shape[0]

    # Per-step RMSE: sqrt(mean squared error across all nodes at each timestep)
    squared_diff = np.square(predicteds - targets)           # [T, N, 2]
    per_step_mse = np.mean(squared_diff.reshape(number_len, -1), axis=1)  # [T]
    per_step_rmse = np.sqrt(per_step_mse)                    # [T]

    # Print at every 50 steps
    for step in range(0, number_len, 50):
        print('rollout rmse @ step %d: %.2e' % (step, per_step_rmse[step]))

    # Save RMSE plot
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_step_rmse, linewidth=1.5, color='steelblue')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('RMSE (velocity)')
    ax.set_title('Rollout RMSE vs Timestep — Trajectory %d' % rollout_index)
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(save_dir, 'rollout_error_%d.png' % rollout_index)
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('RMSE plot saved to %s' % plot_path)

    return per_step_rmse


@torch.no_grad()
def rollout(model, dataset, transformer, rollout_index=0, device='cuda:0'):
    """
    Autoregressive rollout for a single trajectory.

    Args:
        model:          trained Simulator
        dataset:        FpcDataset (test split)
        transformer:    PyG transform compose (FaceToEdge + Cartesian + Distance)
        rollout_index:  which trajectory in the dataset to roll out
        device:         torch device string

    Returns:
        result:     [predicted_velocities, target_velocities] — each [T, N, 2]
        crds:       node positions [N, 2]
        elapsed:    wall-clock seconds for the full rollout
    """
    num_samples_per_tra = dataset.num_sampes_per_tra
    # Grab raw faces before transformer runs
    _raw_graph = dataset[rollout_index * num_samples_per_tra]
    _faces = _raw_graph.face.numpy().T.astype(np.int32)  # [F, 3]
    predicted_velocity = None
    boundary_mask = None
    predicteds = []
    targets = []

    t_start = time.perf_counter()

    for i in tqdm(range(num_samples_per_tra), desc='Rollout trajectory %d' % rollout_index):
        index = rollout_index * num_samples_per_tra + i
        graph = dataset[index]
        graph = transformer(graph)
        graph = graph.to(device)

        # Build boundary mask once from step 0 — assumes fixed topology (cylinder_flow).
        # For domains with dynamic node types, this would need to be re-evaluated each step.
        if boundary_mask is None:
            node_type = graph.x[:, 0]
            fluid_mask = torch.logical_or(
                node_type == NodeType.NORMAL,
                node_type == NodeType.OUTFLOW
            )
            boundary_mask = torch.logical_not(fluid_mask)

        # Swap in own prediction (skip on first step — use ground truth as seed)
        if predicted_velocity is not None:
            graph.x[:, 1:3] = predicted_velocity.detach()

        next_v = graph.y  # ground truth at t+1

        predicted_velocity = model(graph, velocity_sequence_noise=None)

        # Pin boundary nodes back to ground truth
        predicted_velocity[boundary_mask] = next_v[boundary_mask]

        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())

    elapsed = time.perf_counter() - t_start

    # NOTE: crds saved from the last frame only — valid for cylinder_flow (fixed mesh).
    # For deformable-mesh domains, node positions change per timestep and would need
    # per-frame saving.
    crds = graph.pos.cpu().numpy()
    result = [np.stack(predicteds), np.stack(targets)]

    # Save result pkl
    os.makedirs('result', exist_ok=True)
    pkl_path = 'result/result%d.pkl' % rollout_index
    with open(pkl_path, 'wb') as f:
        pickle.dump([result, crds, {
            "domain":          "cylinder_flow",
            "target_field":    "velocity",
            "faces":           _faces,
            "speedup":         round((n_steps * 0.01) / elapsed, 2),
            "elapsed_seconds": round(elapsed, 3),
        }], f)
    print('Result saved to %s' % pkl_path)

    return result, crds, elapsed


@torch.no_grad()
def rollout_cloth(model, dataset, rollout_index: int = 0, device: str = "cpu"):
    """
    Autoregressive rollout for cloth (flag_simple) using Verlet integration.
    Saves result to result/flag_result{rollout_index}.pkl
    """
    steps_per_traj = dataset.steps_per_traj[rollout_index]
    predicteds = []
    targets_list = []
    prev_world = None
    cur_world  = None

    t_start = time.perf_counter()

    for i in tqdm(range(steps_per_traj), desc="Rollout cloth trajectory %d" % rollout_index):
        cum = dataset._cum_steps
        idx = int(cum[rollout_index]) + i
        graph = dataset[idx]
        graph = graph.to(device)

        if cur_world is not None:
            graph.world_pos = cur_world.detach()
            graph.x = torch.cat([cur_world.detach(), graph.x[:, 3:]], dim=-1)
            graph.prev_x    = prev_world.detach()

        # Record cur_world BEFORE stepping — matches DeepMind cloth_eval.py which
        # writes cur_pos (the input) to the trajectory, not the prediction.
        # trajectory[t] = position fed as input at step t (= prediction from step t-1)
        predicteds.append(graph.world_pos.detach().cpu().numpy())
        targets_list.append(graph.y.detach().cpu().numpy())

        prev_world = graph.world_pos.clone()
        next_world = model(graph)   # [N, 3]

        # Pin HANDLE nodes to cur_pos — matches DeepMind: next_pos = where(NORMAL, pred, cur_pos)
        # Using cur_pos (not GT) keeps the rollout fully autoregressive for all nodes.
        node_type = graph.x[:, 3].long()
        handle_mask = (node_type != NodeType.NORMAL)
        next_world[handle_mask] = graph.world_pos[handle_mask]

        cur_world = next_world

    elapsed = time.perf_counter() - t_start
    predicted_arr = np.stack(predicteds)    # [T, N, 3]
    targets_arr   = np.stack(targets_list)  # [T, N, 3]

    sq = np.square(predicted_arr - targets_arr).reshape(steps_per_traj, -1)
    per_step_rmse = np.sqrt(np.mean(sq, axis=1))
    for step in range(0, steps_per_traj, 50):
        print("rollout position rmse @ step %d: %.2e" % (step, per_step_rmse[step]))

    _npz = np.load(os.path.join(dataset._split_dir, f"traj_{rollout_index:05d}.npz"))
    mesh_pos = _npz["mesh_pos"].astype(np.float32)
    cells = _npz["cells"].astype(np.int32)  # [F, 3]
    os.makedirs("result", exist_ok=True)
    pkl_path = "result/flag_result%d.pkl" % rollout_index
    with open(pkl_path, "wb") as f:
        pickle.dump([[predicted_arr, targets_arr], mesh_pos, {
            "domain":          "flag_simple",
            "target_field":    "world_pos",
            "faces":           cells,
            "speedup":         round((steps_per_traj * 0.01) / elapsed, 2),
            "elapsed_seconds": round(elapsed, 3),
        }], f)
    print("Result saved to %s" % pkl_path)
    return [predicted_arr, targets_arr], mesh_pos, elapsed


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MeshGraphNets rollout and evaluation')
    parser.add_argument('--gpu',        type=int,   default=0)
    parser.add_argument('--model_dir',  type=str,   default='checkpoints/best_model.pth')
    parser.add_argument('--test_split', type=str,   default='test')
    parser.add_argument('--rollout_num', type=int,  default=1)
    parser.add_argument('--domain', type=str, default='cylinder_flow',
                        choices=['cylinder_flow', 'flag_simple'])
    args = parser.parse_args()

    device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.gpu) if torch.cuda.is_available() else None

    if args.domain == 'flag_simple':
        model = FlagSimulator(message_passing_num=15, device=device)
        ckpt = torch.load(args.model_dir, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        dataset = FlagDataset('data_flag', split=args.test_split)
        for i in range(args.rollout_num):
            rollout_cloth(model, dataset, rollout_index=i, device=device)
    else:
        # existing CFD code
        # Load model
        simulator = Simulator(
            message_passing_num=15,
            node_input_size=11,
            edge_input_size=3,
            device=device
        )
        state_dict = torch.load(args.model_dir, map_location=device, weights_only=False)
        simulator.load_state_dict(state_dict['model_state_dict'])
        simulator.eval()
        print('Model loaded from %s (epoch %d, valid loss %.2e)' % (
            args.model_dir, state_dict['epoch'], state_dict['valid_loss']
        ))

        # Prepare dataset and transforms
        transformer = T.Compose([
            T.FaceToEdge(),
            T.Cartesian(norm=False),
            T.Distance(norm=False)
        ])
        dataset = FpcDataset('data', split=args.test_split)
        print('Dataset: %d trajectories x %d steps' % (
            len(dataset) // dataset.num_sampes_per_tra, dataset.num_sampes_per_tra
        ))

        # Run rollouts
        for i in range(args.rollout_num):
            print('\n' + '='*60)
            result, crds, elapsed = rollout(simulator, dataset, transformer,
                                            rollout_index=i, device=device)
            n_steps = result[0].shape[0]
            sim_time = n_steps * 0.01   # dt = 0.01s per step
            print('\nInference: %d steps in %.2fs  (%.1fx faster than real-time sim)' % (
                n_steps, elapsed, sim_time / elapsed
            ))
            print('='*60)
            rollout_error(result[0], result[1], rollout_index=i)


    



