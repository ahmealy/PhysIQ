"""
tests/test_poisson_pressure.py — unit tests for PoissonPressureCorrector.

Synthetic mesh: N=50 nodes on a 5×10 regular grid.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from unittest.mock import patch, MagicMock

from physics.poisson_pressure import PoissonPressureCorrector

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def grid_crds():
    """50-node regular grid on [0,1]²."""
    xs, ys = np.mgrid[0:1:5j, 0:1:10j]
    return np.column_stack([xs.ravel(), ys.ravel()])  # [50, 2]


@pytest.fixture
def corrector(grid_crds):
    return PoissonPressureCorrector(grid_crds, k_neighbors=7)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_corrector_reduces_divergence(corrector, grid_crds):
    """Synthetic field v=[x, -y] has divergence ≈ 0 analytically but GNN noise
    can add divergent components.  Use a purely divergent field and check reduction.
    """
    N = grid_crds.shape[0]
    # Pure divergent field: v = [x, y]  →  ∇·v = 2 everywhere
    vel = grid_crds.copy().astype(np.float64)  # [N, 2]: vx=x, vy=y

    rms_before = corrector.divergence_rms(vel)
    v_corr = corrector.correct(vel)
    rms_after = corrector.divergence_rms(v_corr)

    assert rms_before > 1e-3, "Test field should have non-zero divergence"
    assert rms_after < rms_before, "Correction should reduce divergence"


def test_correct_series_shape(corrector, grid_crds):
    """correct_series([T, N, 2]) must return the same shape."""
    N = grid_crds.shape[0]
    T = 5
    rng = np.random.default_rng(0)
    vel_series = rng.standard_normal((T, N, 2))

    result = corrector.correct_series(vel_series)
    assert result.shape == (T, N, 2)


def test_divergence_rms_zero_for_curl_field(corrector, grid_crds):
    """Pure curl field v=(-y, x) is analytically divergence-free.
    Corrector should barely change it (small numerical residual only).
    """
    N = grid_crds.shape[0]
    x, y = grid_crds[:, 0], grid_crds[:, 1]
    vel = np.column_stack([-y, x]).astype(np.float64)

    rms_before = corrector.divergence_rms(vel)
    assert rms_before < 0.05, f"Curl field divergence too large: {rms_before}"

    v_corr = corrector.correct(vel)
    rms_after = corrector.divergence_rms(v_corr)

    # Should not significantly increase divergence for already-div-free field
    assert rms_after < rms_before + 0.05, "Corrector should not worsen div-free field"


def test_laplacian_sparse(grid_crds):
    """Laplacian should be CSC, shape [N, N], and row sums near zero (except node 0)."""
    N = grid_crds.shape[0]
    edges = PoissonPressureCorrector._build_knn_edges(grid_crds, k=7)
    L = PoissonPressureCorrector._build_laplacian(grid_crds, edges, N, regularise=1e-6)

    assert isinstance(L, sp.csc_matrix), "Laplacian must be CSC"
    assert L.shape == (N, N), f"Expected ({N},{N}), got {L.shape}"

    # Row sums for non-pinned nodes should be approximately regularise (small)
    L_dense = L.toarray()
    row_sums = L_dense[1:].sum(axis=1)   # skip pinned row 0
    assert np.all(np.abs(row_sums) < 1e-4), \
        f"Row sums should be ≈0 (regularise only), got max={np.abs(row_sums).max()}"

    # Pinned node: row 0 is [1, 0, 0, ...]
    assert L_dense[0, 0] == pytest.approx(1.0)
    assert np.all(L_dense[0, 1:] == 0.0)


def test_knn_edges_symmetric(grid_crds):
    """k-NN edges should be undirected: represented as sorted (i<j) pairs, all unique."""
    edges = PoissonPressureCorrector._build_knn_edges(grid_crds, k=7)
    assert edges.ndim == 2 and edges.shape[1] == 2

    # All edges stored as (i < j)
    assert np.all(edges[:, 0] < edges[:, 1]), "All edges should have i < j"

    # No duplicates
    edge_set = set(map(tuple, edges.tolist()))
    assert len(edge_set) == len(edges), "Edges should be unique"


def test_correct_preserves_boundary(corrector, grid_crds):
    """After correction, the field magnitude should not blow up (norm ratio < 2.0)."""
    N = grid_crds.shape[0]
    rng = np.random.default_rng(42)
    vel = rng.standard_normal((N, 2))

    v_corr = corrector.correct(vel)
    norm_before = np.linalg.norm(vel)
    norm_after  = np.linalg.norm(v_corr)

    ratio = norm_after / (norm_before + 1e-12)
    assert ratio < 2.0, f"Correction blew up field: norm ratio = {ratio:.3f}"


def test_divergence_reduction_endpoint_structure(grid_crds):
    """Mock _load_pkl_physics and call the endpoint; verify response keys."""
    from fastapi.testclient import TestClient

    T, N = 25, grid_crds.shape[0]
    rng = np.random.default_rng(7)
    predicted = rng.standard_normal((T, N, 2)).astype(np.float32)
    targets   = rng.standard_normal((T, N, 2)).astype(np.float32)

    with patch("api.routes.physics._load_pkl_physics", return_value=(predicted, targets, grid_crds.astype(np.float32))):
        from api.routes.physics import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/results/fake.pkl/physics/corrected_divergence")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    for key in ("divergence_before", "divergence_after", "divergence_reduction_pct", "correction_norm"):
        assert key in data, f"Missing key: {key}"

    assert len(data["divergence_before"]) == T
    assert len(data["divergence_after"])  == T
    assert isinstance(data["divergence_reduction_pct"], float)
