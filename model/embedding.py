"""
Encoder-only forward pass for confidence score computation.

extract_embedding() returns a [256] numpy vector formed by concatenating
the encoder outputs from two frames:
  - frame 0: captures mesh geometry and boundary conditions (the "design")
  - frame N_WARMUP (default 5): captures early flow dynamics after the inlet
    has propagated through the domain (the "physics response")

This dual-frame approach avoids two failure modes of single-frame embedding:
  - Frame 0 only: two designs with identical geometry but different inlet
    velocities look identical at t=0 before any advection occurs.
  - Frame N only: geometry information (cylinder position, size) has been
    partially washed out by the dynamics.

Pooling is over ALL nodes (not just NORMAL) because:
  - WALL_BOUNDARY nodes carry cylinder geometry directly.
  - INFLOW/OUTFLOW nodes carry boundary condition information.
  - Excluding them discards the structural information that distinguishes
    different designs.

Supports both:
  - ``Simulator``      (cylinder_flow / pressure mode)
  - ``FlagSimulator``  (flag_simple cloth mode)
    For cloth, frame 0 IS the design (world_pos at t=0), so only one frame
    is used and the embedding is [128] rather than [256].
"""
import numpy as np
import torch
from utils.utils import NodeType

# Number of warm-up steps for the physics-aware frame in CFD embedding.
# Must match the value used in confidence/build_index.py.
CFD_WARMUP_FRAMES = 5


def _encode_single_graph(simulator, graph, device: str) -> np.ndarray:
    """
    Run the CFD encoder on a single pre-transformed graph.
    Returns a [128] numpy vector, pooled over ALL nodes.
    """
    simulator.eval()
    with torch.no_grad():
        graph = graph.clone().to(device)

        node_type = graph.x[:, 0:1]   # [N, 1]
        frames    = graph.x[:, 1:3]   # [N, 2]  velocity

        # Detect pressure mode: node_input_size == 10 means pressure (9+1)
        if simulator.node_input_size == 10:
            frames = graph.x[:, 1:2]  # [N, 1]

        node_attr       = simulator.update_node_attr(frames, node_type)
        graph.x         = node_attr
        graph.edge_attr = simulator.edge_normalizer(graph.edge_attr, False)

        encoded = simulator.model.encoder(graph)   # [N, 128]

        # Pool over ALL nodes — wall/boundary nodes carry geometry and BC info
        embedding = encoded.x.mean(dim=0)

    return embedding.cpu().numpy()


def _extract_embedding_cfd(simulator, graph_t0, graph_tw, device: str) -> np.ndarray:
    """
    Dual-frame CFD embedding: concat(encoder(frame_0), encoder(frame_warmup)).

    Returns a [256] vector:
      - First 128 dims: geometry + boundary conditions (from frame 0)
      - Last  128 dims: early flow dynamics (from frame CFD_WARMUP_FRAMES)

    Both graphs must already have edge_index and edge_attr
    (apply FaceToEdge + Cartesian + Distance transforms first).
    """
    emb_t0 = _encode_single_graph(simulator, graph_t0, device)
    emb_tw = _encode_single_graph(simulator, graph_tw, device)
    return np.concatenate([emb_t0, emb_tw], axis=0)   # [256]


def _extract_embedding_cloth(simulator, graph, device: str) -> np.ndarray:
    """
    Encoder embedding for a FlagSimulator (cloth).

    Cloth frame 0 IS the design (world_pos at t=0), so a single-frame
    [128] embedding is correct and sufficient. Pools over ALL nodes —
    HANDLE nodes define the attachment geometry.

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

        encoded = simulator.model.encoder(graph)   # [N, 128]

        # Pool over ALL nodes — HANDLE nodes define the attachment geometry
        embedding = encoded.x.mean(dim=0)

    return embedding.cpu().numpy()


def extract_embedding(simulator, graph, device: str,
                      graph_warmup=None) -> np.ndarray:
    """
    Run only the encoder and return an embedding vector.

    For CFD (Simulator): returns [256] = concat(frame_0_emb, frame_warmup_emb).
        graph       — frame 0 (geometry + BCs), pre-transformed
        graph_warmup — frame CFD_WARMUP_FRAMES (early dynamics), pre-transformed.
                       If None, falls back to single-frame [128] embedding.

    For cloth (FlagSimulator): returns [128] from frame 0 only.
        graph       — frame 0 (world_pos at t=0)
        graph_warmup — ignored

    Args:
        simulator:    Simulator or FlagSimulator instance
        graph:        PyG Data object (frame 0 for CFD; the design frame for cloth)
        device:       torch device string, e.g. "cpu" or "cuda:0"
        graph_warmup: Optional warm-up frame for CFD dual-frame embedding

    Returns:
        np.ndarray of shape [256] (CFD dual-frame) or [128] (CFD single or cloth)
    """
    from model.flag_simulator import FlagSimulator   # noqa: F401

    if isinstance(simulator, FlagSimulator):
        return _extract_embedding_cloth(simulator, graph, device)
    else:
        if graph_warmup is not None:
            return _extract_embedding_cfd(simulator, graph, graph_warmup, device)
        else:
            # Fallback: single-frame [128] — used when warmup frame unavailable
            return _encode_single_graph(simulator, graph, device)
