import torch
import numpy as np
import pytest
from torch_geometric.data import Data
import torch_geometric.transforms as T


def _make_pressure_graph(N=30, F=20):
    """Pressure mode graph: graph.x = [N, 2] (node_type + p)."""
    node_type = torch.zeros(N, 1)
    pressure  = torch.randn(N, 1)
    x = torch.cat([node_type, pressure], dim=-1)  # [N, 2]
    face = torch.randint(0, N, (3, F))
    return Data(x=x, pos=torch.randn(N, 2), face=face, y=torch.randn(N, 1))


def _apply_transforms(graph):
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    return tfm(graph)


def test_simulator_pressure_training_shapes():
    """Pressure Simulator training: returns (pred_acc_norm, target_acc_norm) both [N, 1]."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2,
        node_input_size=10,
        edge_input_size=3,
        device="cpu",
        target_field="pressure",
    )
    sim.train()
    graph = _apply_transforms(_make_pressure_graph())
    noise = torch.randn(graph.x.shape[0], 1) * 0.02
    pred, target = sim(graph, velocity_sequence_noise=noise)
    N = graph.x.shape[0]
    assert pred.shape   == (N, 1), f"Expected ({N}, 1), got {pred.shape}"
    assert target.shape == (N, 1), f"Expected ({N}, 1), got {target.shape}"


def test_simulator_pressure_inference_shape():
    """Pressure Simulator inference: returns predicted pressure [N, 1]."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2,
        node_input_size=10,
        edge_input_size=3,
        device="cpu",
        target_field="pressure",
    )
    sim.eval()
    graph = _apply_transforms(_make_pressure_graph())
    with torch.no_grad():
        out = sim(graph, velocity_sequence_noise=None)
    assert out.shape[1] == 1, f"Pressure output should have 1 dim, got {out.shape}"


def test_simulator_velocity_unchanged():
    """Default velocity Simulator is unaffected by pressure changes."""
    from model.simulator import Simulator
    sim = Simulator(message_passing_num=2, node_input_size=11,
                    edge_input_size=3, device="cpu")
    assert sim.target_field == "velocity"
    assert sim._output_normalizer._acc_sum.shape[-1] == 2


def test_simulator_pressure_normalizer_sizes():
    """Pressure mode: node_normalizer size=10, output_normalizer size=1."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=10,
        edge_input_size=3, device="cpu", target_field="pressure",
    )
    assert sim._output_normalizer._acc_sum.shape[-1] == 1
    assert sim._node_normalizer._acc_sum.shape[-1] == 10


def test_simulator_pressure_frames_slice():
    """Pressure Simulator extracts frames from graph.x[:, 1:2] (1 column, not 2)."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=10,
        edge_input_size=3, device="cpu", target_field="pressure",
    )
    sim.eval()
    graph = _apply_transforms(_make_pressure_graph(N=5, F=6))
    with torch.no_grad():
        out = sim(graph, velocity_sequence_noise=None)
    assert out.shape == (5, 1)
