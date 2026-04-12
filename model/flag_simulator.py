"""
FlagSimulator — cloth physics simulator using Verlet integration.

Matches DeepMind cloth_model.py exactly:
    node_input_size = 12  (velocity[3] + one_hot(node_type, 9)[9])
    edge_input_size = 7   (rel_world[3] + |rel_world|[1] + rel_mesh[2] + |rel_mesh|[1])
    output_size     = 3   (3D acceleration)

Training target: acc = world_pos_next - 2*world_pos + world_pos_prev   (Verlet)
Inference:       world_pos_next = 2*world_pos - world_pos_prev + acc
                 HANDLE nodes pinned to world_pos (boundary condition)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge

from .model import EncoderProcesserDecoder
from utils import normalization
from utils.utils import NodeType


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class FlagSimulator(nn.Module):
    """
    Cloth simulator: wraps EncoderProcesserDecoder with Verlet integration.

    Node input:  12 = (world_pos_t - world_pos_{t-1})[3] + one_hot(node_type, 9)[9]
    Edge input:   7 = rel_world[3] + |rel_world|[1] + rel_mesh[2] + |rel_mesh|[1]
    Output:       3 = predicted acceleration in 3D
    """
    node_input_size: int = 12
    edge_input_size: int = 7
    output_size:     int = 3

    def __init__(self, message_passing_num: int = 15, device: str = "cpu") -> None:
        super(FlagSimulator, self).__init__()

        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
        ).to(device)

        self._output_normalizer = normalization.Normalizer(
            size=self.output_size, name="flag_output_normalizer", device=device
        )
        self._node_normalizer = normalization.Normalizer(
            size=self.node_input_size, name="flag_node_normalizer", device=device
        )
        self._edge_normalizer = normalization.Normalizer(
            size=self.edge_input_size, name="flag_edge_normalizer", device=device
        )

        self.model.apply(_init_weights)
        self._face_to_edge = FaceToEdge(remove_faces=False)
        print("FlagSimulator initialized")

    def _build_graph(self, graph: Data) -> Data:
        """
        Convert face-based graph to edge-based and build cloth edge features.

        Edge features [E, 7] — matches DeepMind cloth_model.py order:
            rel_world[3]  — relative 3D world-space position (sender - receiver)
            |rel_world|[1]— norm of rel_world
            rel_mesh[2]   — relative 2D mesh-space position (sender - receiver)
            |rel_mesh|[1] — norm of rel_mesh
        """
        graph = self._face_to_edge(graph)
        edge_index = graph.edge_index  # [2, E]
        senders, receivers = edge_index[0], edge_index[1]

        mesh_pos  = graph.pos         # [N, 2]
        world_pos = graph.world_pos   # [N, 3]

        rel_world  = world_pos[senders] - world_pos[receivers]   # [E, 3]
        world_norm = torch.norm(rel_world, dim=-1, keepdim=True)  # [E, 1]
        rel_mesh   = mesh_pos[senders]  - mesh_pos[receivers]    # [E, 2]
        mesh_norm  = torch.norm(rel_mesh,  dim=-1, keepdim=True)  # [E, 1]

        # DeepMind order: [rel_world_3, |world|_1, rel_mesh_2, |mesh|_1]
        edge_attr = torch.cat([rel_world, world_norm, rel_mesh, mesh_norm], dim=-1)  # [E, 7]
        graph.edge_attr = edge_attr
        return graph

    def _build_node_features(self, graph: Data, node_type_col: torch.Tensor) -> torch.Tensor:
        """
        Node features [N, 12]:
            velocity[3]  = world_pos_t - world_pos_{t-1}
            one_hot[9]   = one_hot(node_type, num_classes=9)

        Args:
            graph: cloth Data object with world_pos and prev_x attributes
            node_type_col: [N] int64 tensor of node type indices (extracted before x overwrite)
        """
        world_pos  = graph.world_pos   # [N, 3]
        prev_world = graph.prev_x      # [N, 3]

        velocity  = world_pos - prev_world                            # [N, 3]
        one_hot   = F.one_hot(node_type_col, num_classes=9).float()  # [N, 9]
        node_feats = torch.cat([velocity, one_hot], dim=-1)           # [N, 12]
        return node_feats

    def forward(self, graph: Data, velocity_sequence_noise: torch.Tensor = None):
        """
        Training (model.training == True):
            Returns (predicted_acc_norm, target_acc_norm) — both [N, 3].
            Loss should be MSE on NodeType.NORMAL nodes only.

        Inference (model.eval()):
            Returns next_world_pos [N, 3] via Verlet integration.
        """
        # Move graph to model device to avoid CPU/CUDA mismatch with normalizer buffers
        device = next(self.model.parameters()).device
        graph = graph.to(device)

        graph = self._build_graph(graph)

        world_pos  = graph.world_pos    # [N, 3]
        prev_world = graph.prev_x       # [N, 3]

        # Extract node_type BEFORE graph.x is overwritten by normalizer
        node_type_col = graph.x[:, 3:4].squeeze(-1).long()  # [N]

        node_feats = self._build_node_features(graph, node_type_col)

        # Optional noise injection on the velocity component only (first 3 dims).
        # node_feats is [N, 12] = velocity[3] + one_hot[9].
        # velocity_sequence_noise is [N, 3] — add only to the velocity slice.
        if velocity_sequence_noise is not None:
            node_feats = node_feats.clone()
            node_feats[:, :3] = node_feats[:, :3] + velocity_sequence_noise

        graph.x    = self._node_normalizer(node_feats, self.training)
        graph.edge_attr = self._edge_normalizer(graph.edge_attr, self.training)

        predicted_acc_norm = self.model(graph)   # [N, 3]

        if self.training:
            target_world    = graph.y
            target_acc      = target_world - 2.0 * world_pos + prev_world  # Verlet
            target_acc_norm = self._output_normalizer(target_acc, self.training)
            return predicted_acc_norm, target_acc_norm
        else:
            acc = self._output_normalizer.inverse(predicted_acc_norm)
            next_world_pos = 2.0 * world_pos - prev_world + acc
            # Pin HANDLE nodes (flag corners) to their current position —
            # matches DeepMind cloth_eval.py: next_pos = where(mask, prediction, cur_pos)
            handle_mask = (node_type_col != NodeType.NORMAL).unsqueeze(-1)  # [N, 1]
            next_world_pos = torch.where(handle_mask, world_pos, next_world_pos)
            return next_world_pos
