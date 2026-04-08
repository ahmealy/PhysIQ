import torch.nn.init as init

import torch.nn as nn
import torch
from torch_geometric.data import Data

from .model import EncoderProcesserDecoder
from utils import normalization

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class Simulator(nn.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        device: str,
        target_field: str = "velocity",
    ) -> None:
        super(Simulator, self).__init__()

        if target_field not in ("velocity", "pressure"):
            raise ValueError("target_field must be 'velocity' or 'pressure'")

        self.target_field    = target_field
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size

        # output_size: 2 for velocity, 1 for pressure
        output_size = 1 if target_field == "pressure" else 2

        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            output_size=output_size,
        ).to(device)

        self._output_normalizer = normalization.Normalizer(
            size=output_size, name="output_normalizer", device=device
        )
        self._node_normalizer = normalization.Normalizer(
            size=node_input_size, name="node_normalizer", device=device
        )
        self.edge_normalizer = normalization.Normalizer(
            size=edge_input_size, name="edge_normalizer", device=device
        )

        self.model.apply(init_weights)
        print("Simulator model initialized")

    def update_node_attr(self, frames: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        Construct and normalize node features.

        Args:
            frames: [N, 2] velocity OR [N, 1] pressure
            types:  [N, 1] node type indices

        Returns:
            Normalized node attributes [N, node_input_size]
            (node_input_size = 11 for velocity, 10 for pressure)
        """
        node_type = types.squeeze(-1).long()                                    # [N]
        one_hot   = torch.nn.functional.one_hot(node_type, num_classes=9)      # [N, 9]
        node_feats = torch.cat([frames, one_hot.float()], dim=-1)               # [N, 11] or [N, 10]
        return self._node_normalizer(node_feats, self.training)

    @staticmethod
    def velocity_to_acceleration(noised_frames: torch.Tensor,
                                  next_frames: torch.Tensor) -> torch.Tensor:
        """Compute change: next - current. Works for both velocity and pressure."""
        return next_frames - noised_frames

    def _frames_slice(self) -> slice:
        """Return the slice of graph.x that contains the field (velocity or pressure)."""
        if self.target_field == "pressure":
            return slice(1, 2)   # graph.x[:, 1:2] — pressure [N, 1]
        return slice(1, 3)       # graph.x[:, 1:3] — velocity [N, 2]

    def forward(self, graph: Data, velocity_sequence_noise: torch.Tensor):
        """
        Forward pass.

        Training:
            Returns (predicted_change_norm, target_change_norm) — both [N, output_size]
            velocity: output_size=2 (acceleration), pressure: output_size=1 (pressure change)

        Inference:
            Returns predicted next velocity [N, 2] or next pressure [N, 1]
        """
        node_type = graph.x[:, 0:1]                    # [N, 1]
        frames    = graph.x[:, self._frames_slice()]   # [N, 2] or [N, 1]

        if self.training:
            assert velocity_sequence_noise is not None, "Noise must be provided during training"
            noised_frames = frames + velocity_sequence_noise   # [N, 2] or [N, 1]
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x   = node_attr

            edge_attr       = self.edge_normalizer(graph.edge_attr, self.training)
            graph.edge_attr = edge_attr

            predicted_norm = self.model(graph)   # [N, output_size]

            target_change      = self.velocity_to_acceleration(noised_frames, graph.y)
            target_change_norm = self._output_normalizer(target_change, self.training)

            return predicted_norm, target_change_norm

        else:
            # Inference
            node_attr = self.update_node_attr(frames, node_type)
            graph.x   = node_attr

            edge_attr       = self.edge_normalizer(graph.edge_attr, self.training)
            graph.edge_attr = edge_attr

            predicted_norm  = self.model(graph)                                # [N, output_size]
            delta           = self._output_normalizer.inverse(predicted_norm)  # [N, output_size]
            next_value      = frames + delta                                   # v_{t+1} or p_{t+1}
            return next_value
