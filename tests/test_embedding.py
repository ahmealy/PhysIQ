import torch
import numpy as np
import pytest
from torch_geometric.data import Data
import torch_geometric.transforms as T


def _make_cfd_graph(N=30, E=60):
    """Synthetic CFD graph: node_type + velocity, edges with 3 features."""
    node_type = torch.zeros(N, 1)
    velocity  = torch.randn(N, 2)
    x = torch.cat([node_type, velocity], dim=-1)  # [N, 3]
    face = torch.randint(0, N, (3, E // 3))       # rough triangles
    return Data(x=x, pos=torch.randn(N, 2), face=face)


def test_extract_embedding_shape():
    """extract_embedding returns [128] numpy array."""
    from model.simulator import Simulator
    from model.embedding import extract_embedding
    import torch_geometric.transforms as T

    sim = Simulator(message_passing_num=2, node_input_size=11,
                    edge_input_size=3, device="cpu")
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    graph = _make_cfd_graph()
    graph = tfm(graph)

    emb = extract_embedding(sim, graph, device="cpu")
    assert isinstance(emb, np.ndarray), "embedding must be numpy array"
    assert emb.shape == (128,), f"expected shape (128,), got {emb.shape}"


def test_extract_embedding_is_finite():
    """Embedding values must be finite (no NaN/Inf)."""
    from model.simulator import Simulator
    from model.embedding import extract_embedding
    import torch_geometric.transforms as T

    sim = Simulator(message_passing_num=2, node_input_size=11,
                    edge_input_size=3, device="cpu")
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    graph = _make_cfd_graph()
    graph = tfm(graph)

    emb = extract_embedding(sim, graph, device="cpu")
    assert np.isfinite(emb).all(), "embedding contains NaN or Inf"


def test_extract_embedding_no_grad():
    """extract_embedding must not modify any model parameters."""
    from model.simulator import Simulator
    from model.embedding import extract_embedding
    import torch_geometric.transforms as T

    sim = Simulator(message_passing_num=2, node_input_size=11,
                    edge_input_size=3, device="cpu")
    tfm = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    params_before = [p.clone() for p in sim.parameters()]
    graph = _make_cfd_graph()
    graph = tfm(graph)
    extract_embedding(sim, graph, device="cpu")
    params_after = list(sim.parameters())

    for before, after in zip(params_before, params_after):
        assert torch.allclose(before, after), "parameters changed during embedding extraction"
