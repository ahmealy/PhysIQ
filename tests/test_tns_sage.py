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
