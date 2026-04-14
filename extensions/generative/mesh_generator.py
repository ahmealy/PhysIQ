"""
Mesh Generator
===============
Converts CVAE design parameters back into PyG ``Data`` objects
ready for the existing MeshGraphNets predictor.

Two domains are supported:

CFD (cylinder_flow)
-------------------
Input:  ``(cx, cy, r, v_inlet)``
Output: PyG Data with nodes at mesh positions, node types set
        (WALL_BOUNDARY for cylinder/walls, INFLOW, OUTFLOW, NORMAL),
        and edge attributes ready for the Simulator.

The mesh is built by:
    1. Place ``n_cyl`` points on the cylinder circle boundary.
    2. Create a rectangular background grid over [0,L]×[0,H].
    3. Remove background points that are inside the cylinder.
    4. Merge cylinder + background points.
    5. Compute Delaunay triangulation.
    6. Assign node types from position.

Cloth (flag_simple)
-------------------
Input:  ``world_pos [N, 3]`` initial cloth positions (from PCA inverse-transform)
Output: PyG Data with fixed mesh topology (loaded from reference trajectory),
        node types preserved from the reference mesh.

The cloth mesh generator simply replaces ``world_pos`` in a reference Data
object — no remeshing is needed because all cloth trajectories share the same
triangle connectivity.

Design principles
-----------------
- **Single Responsibility**: ``CFDMeshBuilder`` and ``ClothMeshBuilder`` each
  do one thing; ``MeshGeneratorFactory`` selects the right one.
- **Open / Closed**: add a new domain by subclassing ``BaseMeshBuilder``
  and registering it in the factory.
- **Liskov Substitution**: ``CFDMeshBuilder`` and ``ClothMeshBuilder`` are
  drop-in replacements for each other via ``BaseMeshBuilder.build()``.
- **Dependency Inversion**: callers depend on ``BaseMeshBuilder``, not on
  concrete implementations.
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from scipy.spatial import Delaunay
from abc import ABC, abstractmethod
from typing import Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.utils import NodeType   # noqa: E402

_INFLOW_TYPE  = int(NodeType.INFLOW)   # 4
_OUTFLOW_TYPE = int(NodeType.OUTFLOW)  # 5


# ---------------------------------------------------------------------------
# Differentiable injection helper
# ---------------------------------------------------------------------------

def _inject_scalar_differentiable(
    feat_col: torch.Tensor,  # [N] current feature column
    node_mask: torch.Tensor, # [N] bool mask of nodes to overwrite
    new_val: torch.Tensor,   # [1] or [] differentiable scalar to inject
) -> torch.Tensor:
    """Replace feat_col values at masked nodes with new_val, preserving autograd."""
    delta = torch.zeros_like(feat_col)
    idx   = node_mask.nonzero(as_tuple=True)[0]
    delta = delta.index_put(
        (idx,),
        new_val.expand(idx.shape[0]) - feat_col[idx].detach(),
    )
    return feat_col.detach() + delta


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMeshBuilder(ABC):
    """Abstract interface for domain-specific mesh construction."""

    @abstractmethod
    def build(self, *args, **kwargs) -> Data:
        """
        Build a PyG Data object ready for MeshGraphNets.

        The returned graph must have:
            graph.pos        [N, 2]   node 2-D coordinates
            graph.face       [3, F]   triangle indices (int64)
            graph.x          [N, d]   node features (node_type + field)
            graph.y          [N, d]   regression target (dummy zeros for generation)
        """


# ---------------------------------------------------------------------------
# CFD mesh builder
# ---------------------------------------------------------------------------

class CFDMeshBuilder(BaseMeshBuilder):
    """
    Builds a cylinder_flow CFD mesh from (cx, cy, r, v_inlet).

    .. deprecated:: sample path, 2026-04-13
        No longer used in CFDDesignSampler.sample().
        Thumbnails and OOD scoring now use RealMeshLookup + ParamSpaceOOD
        respectively.
        Retained for: MeshGeneratorFactory registry, params_to_graph() helper,
        and CLI.

    Algorithm
    ---------
    1. Generate ``n_cyl`` points on the cylinder circle (WALL_BOUNDARY).
    2. Create a regular background grid over the domain.
    3. Remove grid points inside the cylinder (distance < r).
    4. Classify boundary nodes (INFLOW, OUTFLOW, WALL_BOUNDARY for walls).
    5. Merge + Delaunay triangulate all points.
    6. Build PyG Data object (edge_index / edge_attr built by T.FaceToEdge etc.).
    """

    # DeepMind cylinder_flow domain dimensions
    DOMAIN_L: float = 1.6    # length (x-axis)
    DOMAIN_H: float = 0.41   # height (y-axis)
    BOUNDARY_EPS: float = 2e-3

    def __init__(self,
                 n_cyl:     int = 24,     # points on cylinder circumference
                 grid_nx:   int = 60,     # background grid columns
                 grid_ny:   int = 16,     # background grid rows
                 transform: Optional[T.Compose] = None) -> None:
        self._n_cyl   = n_cyl
        self._grid_nx = grid_nx
        self._grid_ny = grid_ny
        # Default transforms: same as training pipeline
        self._transform = transform or T.Compose([
            T.FaceToEdge(),
            T.Cartesian(norm=False),
            T.Distance(norm=False),
        ])

    def _make_cylinder_pts(self, cx: float, cy: float,
                           r: float) -> np.ndarray:
        """Return [n_cyl, 2] boundary points on the cylinder."""
        angles = np.linspace(0, 2 * np.pi, self._n_cyl, endpoint=False)
        return np.column_stack([
            cx + r * np.cos(angles),
            cy + r * np.sin(angles),
        ]).astype(np.float32)

    def _make_background_grid(self) -> np.ndarray:
        """Return [n_grid, 2] regular grid points over the domain."""
        xs = np.linspace(0, self.DOMAIN_L, self._grid_nx, dtype=np.float32)
        ys = np.linspace(0, self.DOMAIN_H, self._grid_ny, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _assign_node_types(self, pts: np.ndarray,
                           cx: float, cy: float, r: float,
                           v_inlet: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign node types and initial velocity field.

        Returns
        -------
        node_type: [N, 1] int-valued float tensor
        x_field:   [N, 3]  concat(node_type, vx, vy)
        """
        N   = len(pts)
        eps = self.BOUNDARY_EPS
        L   = self.DOMAIN_L
        H   = self.DOMAIN_H

        nt = np.zeros(N, dtype=np.int64)   # default: NORMAL = 0

        # Cylinder surface (WALL_BOUNDARY)
        dist_cyl = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        cyl_mask = dist_cyl < r + eps
        nt[cyl_mask] = int(NodeType.WALL_BOUNDARY)

        # Top/bottom walls (WALL_BOUNDARY)
        wall_mask = (
            (np.abs(pts[:, 1] - 0.0) < eps) |
            (np.abs(pts[:, 1] - H)   < eps)
        )
        nt[wall_mask] = int(NodeType.WALL_BOUNDARY)

        # Inflow (left edge)
        inflow_mask = np.abs(pts[:, 0] - 0.0) < eps
        nt[inflow_mask] = int(NodeType.INFLOW)

        # Outflow (right edge)
        outflow_mask = np.abs(pts[:, 0] - L) < eps
        nt[outflow_mask] = int(NodeType.OUTFLOW)

        # Initial velocity field: v_inlet on inflow nodes, 0 elsewhere
        vx = np.zeros(N, dtype=np.float32)
        vx[inflow_mask] = v_inlet

        nt_float = nt.reshape(-1, 1).astype(np.float32)
        x_field  = np.concatenate(
            [nt_float, vx.reshape(-1, 1), np.zeros((N, 1), dtype=np.float32)],
            axis=-1
        )  # [N, 3]
        return nt_float, x_field

    def build(self, cx: float, cy: float, r: float,
              v_inlet: float) -> Data:
        """
        Build a cylinder_flow PyG Data object.

        Args:
            cx, cy:  cylinder centre (normalised coordinates)
            r:       cylinder radius
            v_inlet: inlet velocity magnitude

        Returns:
            PyG Data with pos, face, x, y and after-transform edge_index / edge_attr
        """
        # 1. Cylinder boundary points
        cyl_pts = self._make_cylinder_pts(cx, cy, r)

        # 2. Background grid (exclude interior of cylinder)
        grid_pts = self._make_background_grid()
        dist_to_cyl = np.sqrt((grid_pts[:, 0] - cx) ** 2 +
                               (grid_pts[:, 1] - cy) ** 2)
        grid_pts = grid_pts[dist_to_cyl >= r]  # remove interior points

        # 3. Merge all points
        pts = np.vstack([cyl_pts, grid_pts]).astype(np.float32)  # [N, 2]

        # 4. Delaunay triangulation
        tri  = Delaunay(pts)
        # Remove triangles whose centroid is inside the cylinder
        centroids = pts[tri.simplices].mean(axis=1)   # [F, 2]
        d_cent    = np.sqrt((centroids[:, 0] - cx) ** 2 +
                            (centroids[:, 1] - cy) ** 2)
        valid_tri = tri.simplices[d_cent >= r]         # [F_valid, 3]

        # 5. Assign node types + initial field
        _, x_field = self._assign_node_types(pts, cx, cy, r, v_inlet)

        # 6. Build PyG Data
        graph = Data(
            pos  = torch.from_numpy(pts),
            face = torch.from_numpy(valid_tri.T.astype(np.int64)),
            x    = torch.from_numpy(x_field),
            y    = torch.zeros(len(pts), 2, dtype=torch.float32),  # dummy
        )

        # Apply transforms (FaceToEdge + Cartesian + Distance)
        graph = self._transform(graph)
        return graph


# ---------------------------------------------------------------------------
# Cloth mesh builder
# ---------------------------------------------------------------------------

class ClothMeshBuilder(BaseMeshBuilder):
    """
    Builds a cloth Data object for FlagSimulator from a new initial world_pos.

    Since all flag_simple trajectories share the same mesh topology, we load
    a reference trajectory and simply swap in the new initial world position.
    """

    def __init__(self, reference_traj_path: str) -> None:
        """
        Args:
            reference_traj_path: path to any traj_XXXXX.npz file from which
                                 mesh_pos, node_type, cells are extracted.
        """
        ref = np.load(reference_traj_path)
        self._mesh_pos  = ref["mesh_pos"].astype(np.float32)    # [N, 2]
        self._node_type = ref["node_type"].astype(np.float32)   # [N, 1]
        self._cells     = ref["cells"].astype(np.int64)          # [F, 3]
        self._N         = len(self._mesh_pos)

    def build(self, world_pos: np.ndarray) -> Data:
        """
        Build a cloth Data object with new initial world position.

        Args:
            world_pos: [N, 3] new initial cloth world position

        Returns:
            PyG Data ready for FlagSimulator.forward() in eval mode
        """
        if world_pos.shape != (self._N, 3):
            raise ValueError(
                f"world_pos must have shape ({self._N}, 3), "
                f"got {world_pos.shape}"
            )

        wp  = torch.from_numpy(world_pos.astype(np.float32))   # [N, 3]
        nt  = torch.from_numpy(self._node_type)                 # [N, 1]
        mp  = torch.from_numpy(self._mesh_pos)                  # [N, 2]
        face = torch.from_numpy(self._cells.T)                  # [3, F]

        # Node features: concat(world_pos_t, node_type) → [N, 4]
        x = torch.cat([wp, nt], dim=-1)   # [N, 4]

        graph = Data(
            x          = x,
            prev_x     = wp.clone(),      # no previous frame: prev = current
            pos        = mp,
            world_pos  = wp,
            face       = face,
            y          = torch.zeros_like(wp),  # dummy target
        )
        return graph


# ---------------------------------------------------------------------------
# Real mesh lookup (nearest-neighbour by geometry)
# ---------------------------------------------------------------------------

class RealMeshLookup:
    """
    Finds the nearest real CFD training mesh for given design parameters.

    Used for CFD gradient coupling — loads a real OpenFOAM mesh and injects
    v_inlet as a differentiable tensor so ``∂drag/∂v_inlet`` is non-zero.

    The lookup is a nearest-neighbour search over (cx, cy, r) in the training
    set.  v_inlet is excluded because it is the gradient variable; only
    cylinder geometry determines which real mesh topology to use.

    Algorithm
    ---------
    1. ``__init__``: load ``data/design_params.npy`` [N, 4] and compute
       per-column min/max for L2-normalisation.
    2. ``find_nearest(cx, cy, r)``: normalise query, compute L2 to all rows,
       return the argmin trajectory index.
    3. ``load_mesh_for_trajectory(traj_idx, v_inlet, dataset, device)``:
       load the first timestep of that trajectory from ``FpcDataset``,
       apply PyG transforms, replace INFLOW node x-velocity with the
       differentiable ``v_inlet`` tensor.
    """

    def __init__(self, dataset_path: str = 'data') -> None:
        """
        Args:
            dataset_path: Directory containing ``design_params.npy``.
        """
        self._params: Optional[np.ndarray] = None   # [N, 3]  (cx, cy, r only)
        self._norm_params: Optional[np.ndarray] = None
        self._p_min: Optional[np.ndarray] = None
        self._p_max: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

        dp_path = os.path.join(dataset_path, 'design_params.npy')
        if not os.path.exists(dp_path):
            # Graceful degradation — lookup will always return trajectory 0
            return

        try:
            params = np.load(dp_path)          # [N, 4]: [cx, cy, r, v_inlet]
            if params.ndim != 2 or params.shape[1] < 3:
                return
            geom = params[:, :3].astype(np.float64)   # cx, cy, r only
            p_min = geom.min(axis=0)
            p_max = geom.max(axis=0)
            scale = p_max - p_min
            # Avoid division by zero for degenerate dimensions
            scale[scale < 1e-12] = 1.0
            self._params      = geom
            self._p_min       = p_min
            self._p_max       = p_max
            self._scale       = scale  # stored for use in find_nearest
            self._norm_params = (geom - p_min) / scale    # [N, 3] ∈ [0,1]
        except Exception:
            pass   # silent — _params stays None, find_nearest returns 0

    @property
    def available(self) -> bool:
        """True when design_params.npy was loaded successfully."""
        return self._params is not None

    def find_nearest(self, cx: float, cy: float, r: float) -> int:
        """
        Find the trajectory index whose cylinder geometry is closest to
        (cx, cy, r) under L2 distance on normalised coordinates.

        Args:
            cx, cy: cylinder centre
            r:      cylinder radius

        Returns:
            Trajectory index (0-based) into the training dataset.
            Returns 0 if design_params were not loaded.
        """
        if self._norm_params is None or self._p_min is None:
            return 0

        query  = np.array([cx, cy, r], dtype=np.float64)
        scale  = self._scale
        q_norm = (query - self._p_min) / scale           # [3]
        diffs  = self._norm_params - q_norm               # [N, 3]
        dists  = (diffs ** 2).sum(axis=1)                 # [N]
        return int(np.argmin(dists))

    def load_mesh_for_trajectory(
        self,
        trajectory_index: int,
        v_inlet: torch.Tensor,   # [1] differentiable scalar tensor
        dataset,                 # FpcDataset
        device: str = 'cpu',
    ) -> Data:
        """
        Load the real mesh for ``trajectory_index`` and inject ``v_inlet``.

        The graph is loaded from the first timestep of the trajectory so the
        initial velocity field is as clean as possible.  INFLOW nodes
        (node_type == NodeType.INFLOW == 4) have their x-velocity feature
        replaced by ``v_inlet`` to maintain differentiability.

        Args:
            trajectory_index: which trajectory to load (row in dataset).
            v_inlet:          [1] or scalar differentiable tensor for inlet
                              velocity.  Gradient flows through this tensor.
            dataset:          loaded FpcDataset (or compatible).
            device:           torch device string.

        Returns:
            PyG ``Data`` with ``edge_index`` / ``edge_attr`` already built
            (T.FaceToEdge + T.Cartesian + T.Distance applied).
            ``graph.x[:, 1]`` for INFLOW nodes is set to ``v_inlet``.
        """
        import torch_geometric.transforms as _T

        transformer = _T.Compose([
            _T.FaceToEdge(),
            _T.Cartesian(norm=False),
            _T.Distance(norm=False),
        ])

        # First timestep of this trajectory:
        #   dataset index = traj_idx * num_sampes_per_tra + 0
        n_steps   = dataset.num_sampes_per_tra
        # Clamp to valid range
        traj_idx  = max(0, min(trajectory_index,
                               len(dataset) // n_steps - 1))
        item_idx  = traj_idx * n_steps   # step 0

        graph = dataset[item_idx]          # Data(x, pos, face, y)
        graph = transformer(graph)
        graph = graph.to(device)

        # v_inlet injection: replace vx for INFLOW nodes
        # x layout: [node_type, vx, vy]  (shape [N, 3])
        inflow_mask = (graph.x[:, 0] == _INFLOW_TYPE)   # [N] bool

        # Build a new x tensor that shares storage for all non-injected
        # entries and uses v_inlet for INFLOW vx, preserving grad_fn.
        v_inlet_dev = v_inlet.to(device)   # keep on correct device

        # We need a differentiable replacement; use torch.where on a
        # pre-built "all v_inlet" tensor so autograd can trace through.
        vx_col = graph.x[:, 1].clone()
        # scatter v_inlet into INFLOW positions via masked_fill-alike
        # but staying differentiable w.r.t. v_inlet:
        inflow_idx = inflow_mask.nonzero(as_tuple=True)[0]   # indices
        if inflow_idx.numel() > 0:
            # Differentiable splice via module-level helper
            vx_new = _inject_scalar_differentiable(vx_col, inflow_mask, v_inlet_dev)

            # Reconstruct x with differentiable vx column
            graph.x = torch.cat([
                graph.x[:, 0:1],         # node_type  [N, 1]
                vx_new.unsqueeze(1),      # vx         [N, 1] — differentiable
                graph.x[:, 2:3],          # vy         [N, 1]
            ], dim=1)                     # [N, 3]

        return graph


# ---------------------------------------------------------------------------
# Factory (Open / Closed)
# ---------------------------------------------------------------------------

class MeshGeneratorFactory:
    """
    Creates the appropriate mesh builder for a given domain.

    Adding a new domain requires only:
        1. Subclassing BaseMeshBuilder
        2. Registering it here
    """

    _registry: dict[str, type[BaseMeshBuilder]] = {
        "cylinder_flow": CFDMeshBuilder,
        "flag_simple":   ClothMeshBuilder,
    }

    @classmethod
    def create(cls, domain: str, **kwargs) -> BaseMeshBuilder:
        if domain not in cls._registry:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[domain](**kwargs)

    @classmethod
    def register(cls, domain: str,
                 builder_cls: type[BaseMeshBuilder]) -> None:
        """Register a new domain mesh builder."""
        cls._registry[domain] = builder_cls


# ---------------------------------------------------------------------------
# Convenience functions (used by API route)
# ---------------------------------------------------------------------------

def params_to_graph(cx: float, cy: float, r: float, v_inlet: float,
                    **kwargs) -> Data:
    """Convert CFD design parameters to a predictor-ready PyG graph."""
    builder = CFDMeshBuilder(**kwargs)
    return builder.build(cx, cy, r, v_inlet)


def world_pos_to_graph(world_pos: np.ndarray,
                       reference_traj_path: str) -> Data:
    """Convert cloth initial world position to a predictor-ready PyG graph."""
    builder = ClothMeshBuilder(reference_traj_path=reference_traj_path)
    return builder.build(world_pos)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smoke-test the mesh generator."
    )
    p.add_argument("--domain", default="cylinder_flow",
                   choices=["cylinder_flow", "flag_simple"])
    p.add_argument("--cx",      type=float, default=0.3)
    p.add_argument("--cy",      type=float, default=0.15)
    p.add_argument("--r",       type=float, default=0.05)
    p.add_argument("--v-inlet", type=float, default=0.5)
    p.add_argument("--ref-traj", default="data_flag/train/traj_00000.npz")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.domain == "cylinder_flow":
        print(f"Building CFD mesh: cx={args.cx} cy={args.cy} "
              f"r={args.r} v_inlet={args.v_inlet}")
        graph = params_to_graph(args.cx, args.cy, args.r, args.v_inlet)
        print(f"  Nodes     : {graph.num_nodes}")
        print(f"  Edges     : {graph.num_edges}")
        print(f"  pos shape : {graph.pos.shape}")
        print(f"  x shape   : {graph.x.shape}")
        print(f"  edge_attr : {graph.edge_attr.shape}")
    else:
        print(f"Building cloth mesh from reference: {args.ref_traj}")
        ref  = np.load(args.ref_traj)
        wp0  = ref["world_pos"][0].astype(np.float32)
        graph = world_pos_to_graph(wp0, reference_traj_path=args.ref_traj)
        print(f"  Nodes      : {graph.num_nodes}")
        print(f"  face shape : {graph.face.shape}")
        print(f"  x shape    : {graph.x.shape}")
        print(f"  world_pos  : {graph.world_pos.shape}")


if __name__ == "__main__":
    main()
