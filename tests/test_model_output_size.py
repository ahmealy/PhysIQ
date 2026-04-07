import pytest
import torch
from torch_geometric.data import Data
from model.model import EncoderProcesserDecoder


def test_output_size_param_velocity():
    """EncoderProcesserDecoder with output_size=2 produces [N, 2] output."""
    model = EncoderProcesserDecoder(
        message_passing_num=2,
        node_input_size=11,
        edge_input_size=3,
        output_size=2,
    )
    N, E = 10, 20
    graph = Data(
        x=torch.randn(N, 11),
        edge_attr=torch.randn(E, 3),
        edge_index=torch.randint(0, N, (2, E)),
    )
    out = model(graph)
    assert out.shape == (N, 2)


def test_output_size_param_cloth():
    """EncoderProcesserDecoder with output_size=3 produces [N, 3] output."""
    model = EncoderProcesserDecoder(
        message_passing_num=2,
        node_input_size=12,
        edge_input_size=7,
        output_size=3,
    )
    N, E = 10, 20
    graph = Data(
        x=torch.randn(N, 12),
        edge_attr=torch.randn(E, 7),
        edge_index=torch.randint(0, N, (2, E)),
    )
    out = model(graph)
    assert out.shape == (N, 3)


def test_output_size_param_pressure():
    """EncoderProcesserDecoder with output_size=1 produces [N, 1] output."""
    model = EncoderProcesserDecoder(
        message_passing_num=2,
        node_input_size=10,
        edge_input_size=3,
        output_size=1,
    )
    N, E = 10, 20
    graph = Data(
        x=torch.randn(N, 10),
        edge_attr=torch.randn(E, 3),
        edge_index=torch.randint(0, N, (2, E)),
    )
    out = model(graph)
    assert out.shape == (N, 1)
