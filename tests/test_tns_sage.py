import torch
import pytest
from torch_geometric.data import Data
from model.model import EncoderProcesserDecoder


def _graph(N=50, E=200, node_feat=11, edge_feat=3):
    return Data(
        x=torch.randn(N, node_feat),
        edge_attr=torch.randn(E, edge_feat),
        edge_index=torch.randint(0, N, (2, E)),
        num_nodes=N,
    )


# ── TNSBlock via EncoderProcesserDecoder ─────────────────────────────────────

def test_tns_epd_output_shape():
    m = EncoderProcesserDecoder(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        output_size=2, architecture="tns", tns_heads=4,
    )
    out = m(_graph())
    assert out.shape == (50, 2), f"Expected (50,2) got {out.shape}"


def test_tns_invalid_heads_raises():
    """hidden_size=128, heads=3 → 128 % 3 != 0 → AssertionError."""
    with pytest.raises(AssertionError, match="divisible"):
        EncoderProcesserDecoder(
            message_passing_num=2, node_input_size=11, edge_input_size=3,
            architecture="tns", tns_heads=3,
        )


def test_tns_output_size_1():
    m = EncoderProcesserDecoder(
        message_passing_num=2, node_input_size=10, edge_input_size=3,
        output_size=1, architecture="tns", tns_heads=4,
    )
    out = m(_graph(node_feat=10))
    assert out.shape == (50, 1)


# ── SAGEBlock via EncoderProcesserDecoder ────────────────────────────────────

def test_sage_epd_output_shape():
    m = EncoderProcesserDecoder(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        output_size=2, architecture="sage",
    )
    out = m(_graph())
    assert out.shape == (50, 2)


def test_sage_max_aggr():
    m = EncoderProcesserDecoder(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        architecture="sage", sage_aggr="max",
    )
    out = m(_graph())
    assert out.shape == (50, 2)


# ── All three produce same output shape ──────────────────────────────────────

def test_all_three_same_output_shape():
    g = _graph()
    for arch, kwargs in [
        ("gn",   {}),
        ("tns",  {"tns_heads": 4}),
        ("sage", {}),
    ]:
        m = EncoderProcesserDecoder(
            message_passing_num=2, node_input_size=11, edge_input_size=3,
            architecture=arch, **kwargs,
        )
        out = m(Data(
            x=g.x.clone(), edge_attr=g.edge_attr.clone(),
            edge_index=g.edge_index.clone(), num_nodes=50,
        ))
        assert out.shape == (50, 2), f"{arch}: {out.shape}"


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="Unknown architecture"):
        EncoderProcesserDecoder(
            message_passing_num=2, node_input_size=11, edge_input_size=3,
            architecture="badarch",
        )


# ── Existing GNS default still works (no architecture arg) ───────────────────

def test_gn_default_unchanged():
    """No regression: existing callers that don't pass architecture must still work."""
    m = EncoderProcesserDecoder(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
    )
    out = m(_graph())
    assert out.shape == (50, 2)


def test_tns_backward_pass():
    """Gradients must flow through TNSBlock to all parameters."""
    m = EncoderProcesserDecoder(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        architecture="tns", tns_heads=4,
    )
    m.train()
    out = m(_graph())
    out.sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_sage_backward_pass():
    """Gradients must flow through SAGEBlock to all parameters."""
    m = EncoderProcesserDecoder(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        architecture="sage",
    )
    m.train()
    out = m(_graph())
    out.sum().backward()
    for name, p in m.named_parameters():
        # SAGEBlock does not consume edge_attr in message-passing, so the edge
        # encoder receives no gradient — this is expected by design.
        if "eb_encoder" in name:
            continue
        assert p.grad is not None, f"No gradient for {name}"


# ── Simulator wrapper ─────────────────────────────────────────────────────────

import torch_geometric.transforms as T


def _sim_graph(N=30, F=20):
    """CFD-style graph with faces for transform pipeline."""
    node_type = torch.zeros(N, 1)
    vel       = torch.randn(N, 2)
    x = torch.cat([node_type, vel], dim=-1)
    face = torch.randint(0, N, (3, F))
    return Data(x=x, pos=torch.randn(N, 2), face=face, y=torch.randn(N, 2))


def _apply_cfd_transforms(graph):
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    return tfm(graph)


def test_simulator_tns_inference_shape():
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        device="cpu", architecture="tns", tns_heads=4,
    )
    sim.eval()
    graph = _apply_cfd_transforms(_sim_graph())
    with torch.no_grad():
        out = sim(graph, velocity_sequence_noise=None)
    assert out.shape == (graph.x.shape[0], 2), f"Expected ({graph.x.shape[0]}, 2) got {out.shape}"


def test_simulator_tns_training_shapes():
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        device="cpu", architecture="tns", tns_heads=4,
    )
    sim.train()
    graph = _apply_cfd_transforms(_sim_graph())
    noise = torch.zeros(graph.x.shape[0], 2)
    pred, tgt = sim(graph, velocity_sequence_noise=noise)
    N = graph.x.shape[0]
    assert pred.shape == (N, 2)
    assert tgt.shape  == (N, 2)


def test_simulator_sage_inference_shape():
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        device="cpu", architecture="sage",
    )
    sim.eval()
    graph = _apply_cfd_transforms(_sim_graph())
    with torch.no_grad():
        out = sim(graph, velocity_sequence_noise=None)
    assert out.shape == (graph.x.shape[0], 2)


def test_simulator_gn_default_unchanged():
    """Existing callers that don't pass architecture must still work."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=11, edge_input_size=3, device="cpu",
    )
    assert sim.architecture == "gn"
    sim.eval()
    graph = _apply_cfd_transforms(_sim_graph())
    with torch.no_grad():
        out = sim(graph, velocity_sequence_noise=None)
    assert out.shape == (graph.x.shape[0], 2)


def test_simulator_sage_gradient_flows():
    """Verify gradients reach encoder node-branch weights for SAGE."""
    from model.simulator import Simulator
    sim = Simulator(
        message_passing_num=2, node_input_size=11, edge_input_size=3,
        device="cpu", architecture="sage",
    )
    sim.train()
    graph = _apply_cfd_transforms(_sim_graph())
    noise = torch.zeros(graph.x.shape[0], 2)
    pred, tgt = sim(graph, velocity_sequence_noise=noise)
    loss = (pred - tgt).pow(2).mean()
    loss.backward()
    # Check node encoder — it IS in the gradient path for SAGE
    enc_param = list(sim.model.encoder.nb_encoder.parameters())[0]
    assert enc_param.grad is not None
    assert not torch.isnan(enc_param.grad).any()
