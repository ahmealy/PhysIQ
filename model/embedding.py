"""
Encoder-only forward pass for confidence score computation.

extract_embedding() runs only the encoder on a single graph,
mean-pools over NORMAL nodes, and returns a [128] numpy vector.

Supports both:
  - ``Simulator``      (cylinder_flow / pressure mode)
  - ``FlagSimulator``  (flag_simple cloth mode)

The cloth path calls ``FlagSimulator._build_graph()`` to compute edge attributes
and ``FlagSimulator._build_node_features()`` to build node features, mirroring
the forward() method exactly.
"""
import numpy as np
import torch
from utils.utils import NodeType


def _extract_embedding_cfd(simulator, graph, device: str) -> np.ndarray:
    """
    Encoder embedding for a CFD (cylinder_flow / pressure) Simulator.

    Requires graph to have edge_index and edge_attr already
    (apply FaceToEdge + Cartesian + Distance transforms first).
    """
    simulator.eval()
    with torch.no_grad():
        graph = graph.clone().to(device)

        # Extract node_type and velocity/pressure from x
        node_type = graph.x[:, 0:1]   # [N, 1]
        frames    = graph.x[:, 1:3]   # [N, 2]  (velocity) or 1:2 (pressure)

        # Detect pressure mode: node_input_size == 10 means pressure (9+1)
        if simulator.node_input_size == 10:
            frames = graph.x[:, 1:2]  # [N, 1]

        node_attr       = simulator.update_node_attr(frames, node_type)
        graph.x         = node_attr
        graph.edge_attr = simulator.edge_normalizer(graph.edge_attr, False)

        encoded       = simulator.model.encoder(graph)                # [N, 128]
        node_type_idx = node_type.squeeze(-1).long()
        normal_mask   = (node_type_idx == int(NodeType.NORMAL))

        if normal_mask.sum() == 0:
            embedding = encoded.x.mean(dim=0)
        else:
            embedding = encoded.x[normal_mask].mean(dim=0)

    return embedding.cpu().numpy()


def _extract_embedding_cloth(simulator, graph, device: str) -> np.ndarray:
    """
    Encoder embedding for a FlagSimulator (cloth).

    Builds edge features and node features exactly as FlagSimulator.forward()
    does in eval mode, then runs only the encoder.
    """
    simulator.eval()
    with torch.no_grad():
        graph = graph.clone().to(device)

        # Replicate FlagSimulator._build_graph()
        graph = simulator._build_graph(graph)

        # Extract node_type BEFORE graph.x is overwritten
        node_type_col = graph.x[:, 3:4].squeeze(-1).long()   # [N]

        # Replicate FlagSimulator._build_node_features()
        node_feats      = simulator._build_node_features(graph, node_type_col)
        graph.x         = simulator._node_normalizer(node_feats, False)
        graph.edge_attr = simulator._edge_normalizer(graph.edge_attr, False)

        encoded     = simulator.model.encoder(graph)          # [N, 128]
        normal_mask = (node_type_col == int(NodeType.NORMAL))

        if normal_mask.sum() == 0:
            embedding = encoded.x.mean(dim=0)
        else:
            embedding = encoded.x[normal_mask].mean(dim=0)

    return embedding.cpu().numpy()


def extract_embedding(simulator, graph, device: str) -> np.ndarray:
    """
    Run only the encoder on a single graph and return a [128] embedding.

    Automatically dispatches to the correct implementation based on the
    simulator type (Simulator vs FlagSimulator).

    Args:
        simulator:  Simulator or FlagSimulator instance
        graph:      PyG Data object
                    - CFD: must have edge_index + edge_attr
                      (apply FaceToEdge + Cartesian + Distance first)
                    - Cloth: raw Data with world_pos + face + pos is OK;
                      edge features are built internally.
        device:     torch device string, e.g. "cpu" or "cuda:0"

    Returns:
        np.ndarray of shape [128]
    """
    # Import here to avoid circular imports; both are thin wrappers around
    # EncoderProcesserDecoder so isinstance checks are safe.
    from model.flag_simulator import FlagSimulator   # noqa: F401

    if isinstance(simulator, FlagSimulator):
        return _extract_embedding_cloth(simulator, graph, device)
    else:
        return _extract_embedding_cfd(simulator, graph, device)
