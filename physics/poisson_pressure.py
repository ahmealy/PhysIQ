"""
physics/poisson_pressure.py — Helmholtz projection for GNN velocity fields.

After a GNN predicts velocity v*, this module corrects it to be divergence-free:
    1. Assemble mesh Laplacian L (sparse, from mesh edges)
    2. Compute divergence b = ∇·v* (per node, using k-NN finite differences)
    3. Solve  L p = b  via sparse LU factorisation
    4. Correct: v_corrected = v* - ∇p

The LU factorisation is cached per mesh topology (same mesh → reuse factorisation
for all timesteps). This makes correction cost O(N) per timestep after the one-time
O(N log N) factorisation.

Usage:
    corrector = PoissonPressureCorrector(crds, edges)
    v_corrected = corrector.correct(v_predicted)          # single timestep [N, 2]
    v_all_corrected = corrector.correct_series(v_series)  # all timesteps [T, N, 2]
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree


class PoissonPressureCorrector:
    """
    Helmholtz projection corrector: removes divergent component from a velocity field
    by solving the pressure Poisson equation with a sparse LU factorisation.

    The LU factorisation is computed once in __init__ and reused for every timestep,
    making per-timestep correction O(N) after the one-time O(N log N) setup.
    """

    def __init__(
        self,
        crds: np.ndarray,
        edges: np.ndarray | None = None,
        k_neighbors: int = 7,
        regularise: float = 1e-6,
    ):
        """
        Args:
            crds:        [N, 2] node coordinates
            edges:       [E, 2] edge index pairs (optional — if None, built from k-NN)
            k_neighbors: number of nearest neighbors for gradient estimation
            regularise:  small diagonal added to Laplacian to ensure non-singular
        """
        crds = np.asarray(crds, dtype=np.float64)
        N = crds.shape[0]

        self._crds = crds
        self._k = k_neighbors
        self._N = N

        # Build k-NN tree for gradient / divergence computations
        self._tree = cKDTree(crds)
        _, idxs = self._tree.query(crds, k=k_neighbors)
        self._neighbor_idxs = idxs[:, 1:]  # [N, k-1], exclude self

        # Precompute geometry matrices for vectorised grad/div solves
        dr = crds[self._neighbor_idxs] - crds[:, np.newaxis, :]  # [N, k-1, 2]
        drT = dr.transpose(0, 2, 1)                                # [N, 2, k-1]
        A = drT @ dr                                               # [N, 2, 2]
        eye = 1e-6 * np.eye(2, dtype=np.float64)
        self._A = A + eye[np.newaxis]     # [N, 2, 2]  regularised
        self._drT = drT                   # [N, 2, k-1]

        # Build Laplacian and factorise
        if edges is None:
            edges = self._build_knn_edges(crds, k_neighbors)
        L = self._build_laplacian(crds, edges, N, regularise)
        self._lu = splu(L)

    # ── Edge construction ─────────────────────────────────────────────────────

    @staticmethod
    def _build_knn_edges(crds: np.ndarray, k: int) -> np.ndarray:
        """Build undirected edge list from k-NN graph.

        Returns [E, 2] array of unique undirected edges (i < j).
        """
        tree = cKDTree(crds)
        _, idxs = tree.query(crds, k=k)  # [N, k]
        N = crds.shape[0]
        edge_set = set()
        for i in range(N):
            for j in idxs[i, 1:]:       # skip self (index 0)
                a, b = int(i), int(j)
                edge_set.add((min(a, b), max(a, b)))
        return np.array(sorted(edge_set), dtype=np.int64)

    # ── Laplacian ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_laplacian(
        crds: np.ndarray,
        edges: np.ndarray,
        N: int,
        regularise: float,
    ) -> sp.csc_matrix:
        """Build sparse inverse-distance-squared weighted graph Laplacian.

        Weights w_ij = 1 / ||x_i - x_j||^2 make the discrete operator consistent
        with the finite-difference divergence and gradient estimators (which also
        implicitly scale as 1/h²).

        For each edge (i, j) with weight w:
            L[i,i] += w,  L[j,j] += w,  L[i,j] -= w,  L[j,i] -= w

        Diagonal is shifted by `regularise` to prevent singularity.
        Row 0 is pinned to ground (Dirichlet BC) to remove null space.

        Returns scipy CSC matrix of shape [N, N].
        """
        ei, ej = edges[:, 0], edges[:, 1]
        dx = crds[ei] - crds[ej]               # [E, 2]
        dist2 = np.maximum(np.sum(dx ** 2, axis=1), 1e-20)  # [E], avoid /0
        w = 1.0 / dist2                         # [E] inverse-distance-squared weights

        # Off-diagonal contributions
        rows_off = np.concatenate([ei, ej])
        cols_off = np.concatenate([ej, ei])
        data_off = -np.concatenate([w, w])

        # Diagonal: weighted degree of each node
        deg = np.zeros(N, dtype=np.float64)
        np.add.at(deg, ei, w)
        np.add.at(deg, ej, w)
        rows_diag = np.arange(N)
        data_diag = deg + regularise

        rows = np.concatenate([rows_diag, rows_off])
        cols = np.concatenate([rows_diag, cols_off])
        data = np.concatenate([data_diag, data_off])

        L = sp.csc_matrix((data, (rows, cols)), shape=(N, N))

        # Pin node 0: clear row 0, set diagonal to 1 (Dirichlet BC)
        L = L.tolil()
        L[0, :] = 0.0
        L[0, 0] = 1.0
        return L.tocsc()

    # ── Divergence ────────────────────────────────────────────────────────────

    def _compute_divergence(self, vel: np.ndarray) -> np.ndarray:
        """Compute per-node divergence ∇·v using vectorised k-NN finite differences.

        Args:
            vel: [N, 2] velocity field

        Returns:
            div: [N] divergence at each node
        """
        dv = vel[self._neighbor_idxs] - vel[:, np.newaxis, :]  # [N, k-1, 2]
        rhs = self._drT @ dv.astype(np.float64)                 # [N, 2, 2]
        grad = np.linalg.solve(self._A, rhs)                    # [N, 2, 2]
        # div = ∂vx/∂x + ∂vy/∂y = grad[:,0,0] + grad[:,1,1]
        return (grad[:, 0, 0] + grad[:, 1, 1]).astype(np.float64)

    # ── Gradient ──────────────────────────────────────────────────────────────

    def _compute_gradient(self, p: np.ndarray) -> np.ndarray:
        """Compute pressure gradient ∇p at each node via k-NN finite differences.

        Args:
            p: [N] scalar pressure field

        Returns:
            grad_p: [N, 2]  [∂p/∂x, ∂p/∂y] at each node
        """
        # dp[i, j] = p[neighbor_j] - p[i]  → [N, k-1]
        dp = p[self._neighbor_idxs] - p[:, np.newaxis]           # [N, k-1]
        # rhs: [N, 2, 1]  (single scalar field)
        rhs = self._drT @ dp[:, :, np.newaxis].astype(np.float64)  # [N, 2, 1]
        grad_p = np.linalg.solve(self._A, rhs)                      # [N, 2, 1]
        return grad_p[:, :, 0]  # [N, 2]

    # ── Public API ────────────────────────────────────────────────────────────

    def correct(self, vel: np.ndarray) -> np.ndarray:
        """Apply Helmholtz projection to a single velocity field.

        Args:
            vel: [N, 2] predicted velocity (may have non-zero divergence)

        Returns:
            v_corrected: [N, 2] divergence-free velocity
        """
        vel = np.asarray(vel, dtype=np.float64)
        b = self._compute_divergence(vel)
        b[0] = 0.0          # ground node BC: p=0 at node 0
        # L is the positive semi-definite graph Laplacian (L = -Δ),
        # so the Poisson equation Δp = ∇·v becomes L p = -∇·v
        p = self._lu.solve(-b)
        grad_p = self._compute_gradient(p)
        return (vel - grad_p).astype(vel.dtype)

    def correct_series(self, vel_series: np.ndarray) -> np.ndarray:
        """Apply Helmholtz projection to every timestep.

        Args:
            vel_series: [T, N, 2] predicted velocity series

        Returns:
            v_corrected: [T, N, 2]
        """
        vel_series = np.asarray(vel_series, dtype=np.float64)
        return np.stack([self.correct(vel_series[t]) for t in range(vel_series.shape[0])])

    def divergence_rms(self, vel: np.ndarray) -> float:
        """RMS of |∇·v| across all nodes — quality metric for incompressibility.

        Returns:
            rms: float, 0.0 means perfectly divergence-free
        """
        div = self._compute_divergence(np.asarray(vel, dtype=np.float64))
        return float(np.sqrt(np.mean(div ** 2)))
