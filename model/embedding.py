"""
Encoder-only forward pass for confidence score computation.

extract_embedding() runs only the encoder on a single (already-transformed) graph,
mean-pools over NORMAL nodes, and returns a [128] numpy vector.
"""
import numpy as np
import torch
from utils.utils import NodeType


def extract_embedding(simulator, graph, device: str) -> np.ndarray:
    """
    Run only the encoder on a single graph.
    Returns mean-pooled NORMAL node embedding of shape [128].

    Args:
        simulator:  Simulator (must have .model.encoder, .update_node_attr,
                    and .edge_normalizer)
        graph:      PyG Data — must already have edge_index and edge_attr
                    (apply FaceToEdge + Cartesian + Distance transforms first)
        device:     torch device string

    Returns:
        np.ndarray of shape [128]
    """
    simulator.eval()

    with torch.no_grad():
        graph = graph.to(device)

        # Extract node_type and velocity from x, exactly as Simulator.forward does
        node_type = graph.x[:, 0:1]   # [N, 1]
        frames    = graph.x[:, 1:3]   # [N, 2]

        # Build normalized node features (replicates the inference branch of forward)
        node_attr = simulator.update_node_attr(frames, node_type)  # [N, node_input_size]
        graph.x = node_attr

        # Normalize edge attributes
        edge_attr = graph.edge_attr                                 # [E, edge_input_size]
        graph.edge_attr = simulator.edge_normalizer(edge_attr, False)

        # Run ONLY the encoder — skip processor and decoder
        encoded = simulator.model.encoder(graph)                    # Data with x: [N, 128]

        # Mean-pool over NORMAL nodes (NodeType.NORMAL == 0)
        node_type_idx = node_type.squeeze(-1).long()                # [N]
        normal_mask = (node_type_idx == int(NodeType.NORMAL))       # [N] bool

        if normal_mask.sum() == 0:
            # Fallback: pool over all nodes if none are NORMAL
            embedding = encoded.x.mean(dim=0)
        else:
            embedding = encoded.x[normal_mask].mean(dim=0)          # [128]

    return embedding.cpu().numpy()
