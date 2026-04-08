import torch
import numpy as np
import pytest
from torch_geometric.data import Data


def _make_cloth_graph(N=20, F=30):
    """Synthetic cloth graph matching FlagDataset output format."""
    world_pos  = torch.randn(N, 3)
    prev_world = torch.randn(N, 3)
    mesh_pos   = torch.randn(N, 2)
    node_type  = torch.zeros(N, 1)  # all NORMAL

    # x: concat(world_pos, node_type)
    x = torch.cat([world_pos, node_type], dim=-1)  # [N, 4]

    # Triangular faces [3, F]
    face = torch.randint(0, N, (3, F))
    # Target: next world_pos
    y = torch.randn(N, 3)

    return Data(
        x=x,
        prev_x=prev_world,
        pos=mesh_pos,
        world_pos=world_pos,
        face=face,
        y=y,
    )


def test_flag_simulator_training_shapes():
    """Training forward pass returns (predicted_acc_norm, target_acc_norm) both [N, 3]."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    sim.train()

    graph = _make_cloth_graph(N=20, F=30)

    predicted, target = sim(graph)
    N = 20
    assert predicted.shape == (N, 3), f"Expected ({N}, 3), got {predicted.shape}"
    assert target.shape    == (N, 3), f"Expected ({N}, 3), got {target.shape}"


def test_flag_simulator_inference_shape():
    """Inference forward pass returns next world_pos [N, 3]."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    sim.eval()

    graph = _make_cloth_graph(N=20, F=30)
    with torch.no_grad():
        next_pos = sim(graph)
    assert next_pos.shape == (20, 3)


def test_flag_simulator_verlet_integration():
    """Inference: next_pos = 2*cur - prev + acc (Verlet)."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    sim.eval()

    graph = _make_cloth_graph(N=5, F=6)
    world_pos  = graph.world_pos.clone()
    prev_world = graph.prev_x.clone()

    with torch.no_grad():
        next_pos = sim(graph)

    # acc must be finite (not NaN/Inf)
    acc_implied = next_pos - 2 * world_pos + prev_world
    assert torch.isfinite(acc_implied).all()
    assert torch.isfinite(next_pos).all()


def test_flag_simulator_node_features_size():
    """Node input size is exactly 12 (vel[3] + one_hot[9])."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    assert sim.node_input_size == 12


def test_flag_simulator_edge_features_size():
    """Edge input size is exactly 7."""
    from model.flag_simulator import FlagSimulator
    sim = FlagSimulator(message_passing_num=2, device="cpu")
    assert sim.edge_input_size == 7
