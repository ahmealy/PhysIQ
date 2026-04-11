"""
GnnScorer — wraps MeshGraphNets Simulator for physics-accurate candidate scoring.

Used in Deep mode of the Generate subsystem. Runs an adaptive rollout per
candidate (20-step chunks, stops when drag Δ < 1e-3, hard cap 200 steps)
and extracts drag as mean x-velocity over OUTFLOW nodes at the final step.

Lazy-loaded and cached via api.state.get_gnn_scorer().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

_CHUNK      = 20      # steps per convergence check
_MAX_STEPS  = 200     # hard cap
_CONV_DELTA = 1e-3    # convergence threshold on drag change


@dataclass
class GnnScore:
    """Result of one adaptive GNN rollout."""
    gnn_predicted_value: float
    converged: bool   # False if _MAX_STEPS cap was hit


class GnnScorer:
    """
    Scores CFD design candidates using the MeshGraphNets GNN simulator.

    Parameters
    ----------
    checkpoint_path : str
        Path to best_model.pth (cylinder_flow checkpoint).
    device : str
        Torch device string, e.g. "cpu" or "cuda:0".
    """

    def __init__(self, checkpoint_path: str, device: str) -> None:
        from api.state import get_model

        # Reuse the cached simulator from api.state — avoids double-loading.
        self.simulator   = get_model(checkpoint_path, device=device)
        self.device      = device
        self.transformer = T.Compose([
            T.FaceToEdge(),
            T.Cartesian(norm=False),
            T.Distance(norm=False),
        ])
        logger.info("GnnScorer initialized (device=%s)", device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        graphs: list[Data],
        device: str | None = None,
    ) -> list[GnnScore]:
        """
        Score a list of PyG graphs via adaptive rollout.

        Each graph is scored independently. If a graph raises during rollout
        it gets GnnScore(gnn_predicted_value=float('nan'), converged=False)
        so callers can detect failure via math.isnan().

        NOTE: GnnScorer is not safe to use concurrently with training — eval()
        mode is set here once for the whole batch rather than per-candidate.
        """
        # Set eval mode once per batch, not once per candidate.
        self.simulator.eval()

        dev = device or self.device
        results: list[GnnScore] = []

        for i, graph in enumerate(graphs):
            try:
                score = self._adaptive_rollout(graph, dev)
                results.append(score)
            except Exception as exc:
                logger.warning("GnnScorer: rollout failed for candidate %d: %s", i, exc)
                results.append(GnnScore(gnn_predicted_value=float("nan"), converged=False))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_rollout(self, graph: Data, device: str) -> GnnScore:
        """
        Run autoregressive rollout in _CHUNK-step chunks.
        Stops when |drag_t - drag_{t-_CHUNK}| < _CONV_DELTA or steps >= _MAX_STEPS.
        """
        from utils.utils import NodeType

        # simulator.eval() is called once per batch in score_candidates(), not here.

        graph = graph.clone()
        if graph.edge_index is None:
            graph = self.transformer(graph)

        # Bug 3: validate that the transform actually produced connectivity.
        if graph.edge_index is None and graph.face is None:
            raise ValueError(
                "_adaptive_rollout: graph has no edge_index and no face after "
                "transform — cannot run rollout. Supply a graph with pre-built "
                "edges or a face tensor so FaceToEdge can construct them."
            )

        graph = graph.to(device)

        # Extract node_type from x[:,0]
        node_type = graph.x[:, 0].long()   # [N]

        # evolving_mask: NORMAL and OUTFLOW nodes whose velocity is predicted
        # each step. boundary_mask pins all non-fluid-dynamic nodes
        # (OBSTACLE, WALL_BOUNDARY, INFLOW, AIRFOIL, HANDLE, …) to their
        # original values so they are never overwritten by the simulator output.
        evolving_mask = (node_type == int(NodeType.NORMAL)) | \
                        (node_type == int(NodeType.OUTFLOW))
        boundary_mask = ~evolving_mask     # [N] bool

        # Save the full x layout once — Simulator.forward() overwrites graph.x
        # in-place with normalized features, so we must restore the original
        # [node_type | vel_x | vel_y | …] schema before every forward call.
        original_x = graph.x.clone()      # shape [N, F]

        # Current velocity — columns 1:3 of x (cylinder_flow schema)
        current_vel = graph.x[:, 1:3].clone()   # [N, 2]

        prev_drag   = None
        converged   = False
        total_steps = 0
        drag        = 0.0

        with torch.no_grad():
            while total_steps < _MAX_STEPS:
                chunk_end = min(total_steps + _CHUNK, _MAX_STEPS)

                for _ in range(chunk_end - total_steps):
                    # Restore the original x layout, then inject current velocity.
                    # This is necessary because Simulator.forward() mutates
                    # graph.x to hold normalized features; without this restore
                    # the next write would corrupt the wrong tensor columns.
                    graph.x = original_x.clone()
                    graph.x[:, 1:3] = current_vel

                    next_vel = self.simulator(graph, velocity_sequence_noise=None)  # [N,2]

                    # Pin boundary nodes to their original values
                    next_vel[boundary_mask] = current_vel[boundary_mask]
                    current_vel = next_vel

                total_steps = chunk_end
                drag = self._extract_drag(current_vel, node_type)

                if prev_drag is not None and abs(drag - prev_drag) < _CONV_DELTA:
                    converged = True
                    break

                prev_drag = drag

        return GnnScore(gnn_predicted_value=float(drag), converged=converged)

    @staticmethod
    def _extract_drag(velocity: torch.Tensor, node_type: torch.Tensor) -> float:
        """
        Extract drag proxy as mean |vx| over OUTFLOW nodes.
        Falls back to all nodes if no OUTFLOW nodes exist.
        """
        from utils.utils import NodeType

        outflow_mask = (node_type == int(NodeType.OUTFLOW))

        if outflow_mask.sum() == 0:
            logger.debug("_extract_drag: no OUTFLOW nodes found, using all nodes")
            vx = velocity[:, 0]
        else:
            vx = velocity[outflow_mask, 0]

        return float(vx.abs().mean().item())
