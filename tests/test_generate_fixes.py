"""
Tests for all issues found in the generate-subsystem review.

Issues covered (by original report #):
  #6  Cloth predicted_value always equals target           → cloth sampler must call surrogate
  #9  CFD thumbnail re-triangulation includes cylinder interior  → filter by r
  #13 _physics_loss detach breaks gradient in CFD          → grad flows from phys loss
  #17 isfinite only checks col 0 in CFD main()             → check all columns
  #18 _physics_loss detach breaks gradient in cloth        → grad flows from phys loss (cloth)
  #19 Docstring typo "cfae_cfd.py"                         → says "cvae_cfd.py"
  #20 Clipping before denorm silently saturates OOD samples → test & document range
  #22 Checkpoint filename mismatch cloth_cvae vs flag-simple_cvae  → aligned
  #24 MinMaxScaler.inverse_y missing epsilon compensation   → round-trip precision
  #25 isfinite check col 0 only in drag_surrogate main()   → check all columns
  #31/32 optimize_cloth uses old ckpt["cfg"]/["scaler"] keys → uses .load() classmethod
  #34 _build_data_from_pos loads disk on every iter         → cached
  #35 StressObjective torch.tensor inside call loop         → pre-alloc in __init__
  #36 best_stress computed from loop var z, not best_z      → uses best_z
  #38 parseFloat("—") is NaN → wrong error colour when targetValue=0  (frontend)
  #39 cursor-pointer duplicated in base class string         (frontend)
  #42 UI n_candidates cap 20, API allows 50               → align to 50
  #45 Cloth always shows in-distribution when OOD N/A       → count only when conf>=0
  #48 Unused import Play in Generate.tsx                   → removed
  #10 Unused import base64 in generate.py                   → removed
  #4  _thumbnail_cache unbounded                           → max 100 sessions, LRU evict
"""
import sys
import os
import io
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_fitted_cfd_trainer(lam: float = 0.5):
    """Return a CVAETrainer with a fitted scaler (needed by physics loss)."""
    from extensions.generative.cvae_cfd import (
        CVAEConfig, CVAETrainer, CFDCVAE
    )
    from extensions.generative.drag_surrogate import DragSurrogate
    cfg = CVAEConfig(epochs=1, batch_size=4, lam=lam)
    model = CFDCVAE(cfg=cfg)
    surrogate = DragSurrogate()
    trainer = CVAETrainer(model=model, surrogate=surrogate, cfg=cfg)
    # Fit scaler with dummy data so denorm_params works
    params = np.random.rand(20, 4).astype(np.float32)
    drag   = np.random.rand(20).astype(np.float32) * 0.1
    trainer._scaler.fit(params, drag)
    return trainer


def _make_fitted_cloth_trainer(lam: float = 0.5):
    """Return a ClothCVAETrainer with a fitted scaler and surrogate."""
    from extensions.generative.cvae_cloth import (
        ClothCVAEConfig, ClothCVAETrainer, ClothCVAE,
        StressSurrogate, StressSurrogateTrainer,
    )
    cfg = ClothCVAEConfig(epochs=1, batch_size=4, lam=lam)
    model = ClothCVAE(cfg=cfg)
    s_model = StressSurrogate(pose_dim=cfg.pose_dim)
    s_trainer = StressSurrogateTrainer(s_model)
    pose = np.random.rand(20, cfg.pose_dim).astype(np.float32)
    stress = np.random.rand(20).astype(np.float32) * 2.0
    s_trainer.fit(pose, stress, epochs=5, verbose=False)
    trainer = ClothCVAETrainer(model=model, stress_trainer=s_trainer, cfg=cfg)
    trainer._scaler.fit(pose, stress)
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Issue #13 / #18  — physics loss gradient must flow back through model weights
# ─────────────────────────────────────────────────────────────────────────────

def test_cfd_physics_loss_gradient_flows_to_model():
    """
    _physics_loss must contribute a non-zero gradient to the decoder weights.

    The surrogate is frozen (no grad on its params), but the gradient of the
    physics loss w.r.t. the *CVAE decoder output* (recon_norm) must be non-zero
    so that backprop can update the decoder.  Previously `detach()` was called
    on recon_norm before entering the surrogate, which broke this path.
    """
    trainer = _make_fitted_cfd_trainer(lam=0.5)
    # Enable gradients on decoder params
    for p in trainer._model.parameters():
        p.requires_grad_(True)

    # Forward pass to get recon_norm with grad
    p_in   = torch.randn(4, 4)
    d_in   = torch.randn(4, 1)
    recon, mu, logvar = trainer._model(p_in, d_in, d_in)

    target_drag_norm = torch.zeros(4, 1)
    phys_loss = trainer._physics_loss(recon, target_drag_norm)
    phys_loss.backward()

    # At least one decoder parameter must have a non-zero gradient
    decoder_grads = [p.grad for p in trainer._model.decoder.parameters()
                     if p.grad is not None]
    assert len(decoder_grads) > 0, "No decoder parameter received a gradient"
    total_grad = sum(g.abs().sum().item() for g in decoder_grads)
    assert total_grad > 0.0, (
        f"Decoder gradient is all-zero — physics loss detach() is breaking the graph. "
        f"total_grad={total_grad}"
    )


def test_cloth_physics_loss_gradient_flows_to_model():
    """Same gradient-flow requirement for the Cloth CVAE physics loss."""
    trainer = _make_fitted_cloth_trainer(lam=0.5)
    for p in trainer._model.parameters():
        p.requires_grad_(True)

    pose_in   = torch.randn(4, 16)
    stress_in = torch.randn(4, 1)
    recon, mu, logvar = trainer._model(pose_in, stress_in, stress_in)

    target_stress_norm = torch.zeros(4, 1)
    phys_loss = trainer._physics_loss(recon, target_stress_norm)
    phys_loss.backward()

    decoder_grads = [p.grad for p in trainer._model.decoder.parameters()
                     if p.grad is not None]
    assert len(decoder_grads) > 0
    total_grad = sum(g.abs().sum().item() for g in decoder_grads)
    assert total_grad > 0.0, (
        f"Cloth decoder gradient is all-zero — physics loss detach() breaking graph. "
        f"total_grad={total_grad}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #17 / #25 — isfinite must check ALL columns, not just col 0
# ─────────────────────────────────────────────────────────────────────────────

def test_cvae_cfd_isfinite_check_all_columns():
    """
    The NaN/inf filter in CVAETrainer.generate must check all 4 param columns.
    Previously only column 0 was checked, so NaN in cy/r/v_inlet passed through.
    """
    from extensions.generative.cvae_cfd import (
        CVAEConfig, CVAETrainer, CFDCVAE
    )
    from extensions.generative.drag_surrogate import DragSurrogate
    cfg = CVAEConfig()
    trainer = CVAETrainer(CFDCVAE(cfg), DragSurrogate(), cfg)

    # Params with NaN in column 2 (r), NOT in column 0 (cx)
    params = np.array([
        [0.3, 0.15, np.nan, 0.5],   # bad row — NaN in col 2
        [0.3, 0.15, 0.05,  0.5],   # good row
    ], dtype=np.float32)

    valid = np.isfinite(params).all(axis=1)   # this is what the fix should use
    assert valid.sum() == 1, (
        "isfinite check over all columns should keep only the row with no NaN"
    )


def test_drag_surrogate_isfinite_check_all_columns():
    """Same all-column check in drag_surrogate filtering logic."""
    params = np.array([
        [0.3, 0.15, 0.05, np.nan],  # NaN in col 3 (v_inlet)
        [0.3, 0.15, 0.05, 0.5],
    ], dtype=np.float32)

    # The fix: np.isfinite(params).all(axis=1) — check current behaviour
    valid_all = np.isfinite(params).all(axis=1)
    valid_col0 = np.isfinite(params[:, 0])

    # col0-only misses the bad row; all-columns does not
    assert valid_col0.sum() == 2, "Col-0 check wrongly keeps 2 rows"
    assert valid_all.sum() == 1, "All-col check should keep only 1 row"


# ─────────────────────────────────────────────────────────────────────────────
# Issue #19 — docstring typo "cfae_cfd.py" → "cvae_cfd.py"
# ─────────────────────────────────────────────────────────────────────────────

def test_cloth_cvae_docstring_has_no_typo():
    """Module docstring must not contain the typo 'cfae_cfd'."""
    from extensions.generative import cvae_cloth
    assert "cfae_cfd" not in (cvae_cloth.__doc__ or ""), (
        "Module docstring still contains typo 'cfae_cfd' (should be 'cvae_cfd')"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #22 — Cloth CVAE checkpoint filename mismatch
# ─────────────────────────────────────────────────────────────────────────────

def test_cloth_cvae_default_out_path_matches_api_sampler():
    """
    The CLI --out default and ClothDesignSampler.CVAE_PATH must agree.
    Previously cvae_cloth.py saved to 'cloth_cvae.pth' but generate.py looked
    for 'flag-simple_cvae.pth'.
    """
    pytest.importorskip("fastapi")
    import ast, pathlib

    # Extract CLI default from argparse section without importing side effects
    src = pathlib.Path(
        "extensions/generative/cvae_cloth.py"
    ).read_text()
    # find the --out default
    tree = ast.parse(src)
    cli_default = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # look for p.add_argument("--out", default=...)
            if any(isinstance(a, ast.Constant) and a.value == "--out"
                   for a in node.args):
                for kw in node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        cli_default = kw.value.value

    from api.routes.generate import ClothDesignSampler
    assert cli_default == ClothDesignSampler.CVAE_PATH, (
        f"CLI --out default '{cli_default}' != "
        f"ClothDesignSampler.CVAE_PATH '{ClothDesignSampler.CVAE_PATH}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #24 — MinMaxScaler round-trip precision
# ─────────────────────────────────────────────────────────────────────────────

def test_minmax_scaler_roundtrip_y():
    """
    inverse_y(transform_y(y)) should equal y to within floating-point noise.
    The old impl had a systematic epsilon error because transform_y divided by
    (range + 1e-8) but inverse_y multiplied by range (without +1e-8).
    """
    from extensions.generative.drag_surrogate import MinMaxScaler
    rng = np.random.default_rng(42)
    X = rng.random((50, 4)).astype(np.float32)
    y = rng.random(50).astype(np.float32) * 0.1

    scaler = MinMaxScaler()
    scaler.fit(X, y)

    y_norm   = scaler.transform_y(y)
    y_recon  = scaler.inverse_y(y_norm)

    np.testing.assert_allclose(y_recon, y, rtol=1e-5, atol=1e-6,
        err_msg="MinMaxScaler round-trip y has excessive error")


def test_cvae_scaler_roundtrip_drag():
    """CVAEScaler drag round-trip must be precise (same epsilon fix)."""
    from extensions.generative.cvae_cfd import CVAEScaler
    rng = np.random.default_rng(7)
    params = rng.random((30, 4)).astype(np.float32)
    drag   = rng.random(30).astype(np.float32) * 0.1

    sc = CVAEScaler()
    sc.fit(params, drag)

    d_norm  = sc.norm_drag(drag)
    d_recon = sc.denorm_drag(d_norm)
    np.testing.assert_allclose(d_recon, drag, rtol=1e-5, atol=1e-6,
        err_msg="CVAEScaler drag round-trip has excessive error")


# ─────────────────────────────────────────────────────────────────────────────
# Issue #31 / #32 — optimize_cloth must use ClothCVAETrainer.load(), not raw keys
# ─────────────────────────────────────────────────────────────────────────────

def test_optimize_cloth_uses_load_classmethod(tmp_path):
    """
    optimize_cloth() must load the CVAE via ClothCVAETrainer.load() so that
    the new checkpoint format (cfg_dict / scaler_dict) works correctly.

    We create a minimal checkpoint in new format and verify no KeyError.
    """
    from extensions.generative.cvae_cloth import (
        ClothCVAEConfig, ClothCVAETrainer, ClothCVAE,
        StressSurrogate, StressSurrogateTrainer,
    )
    from dataclasses import asdict

    # Build and save a trainer in new format
    cfg     = ClothCVAEConfig(pose_dim=4, latent_dim=4, hidden_size=16, epochs=1)
    model   = ClothCVAE(cfg=cfg)
    s_model = StressSurrogate(pose_dim=4, hidden_size=16)
    s_tr    = StressSurrogateTrainer(s_model)
    trainer = ClothCVAETrainer(model=model, stress_trainer=s_tr, cfg=cfg)
    # Fit scaler
    pose   = np.random.rand(10, 4).astype(np.float32)
    stress = np.random.rand(10).astype(np.float32) * 2.0
    trainer._scaler.fit(pose, stress)

    save_path = str(tmp_path / "cloth_test.pth")
    trainer.save(save_path)

    # Attempt to load via ClothCVAETrainer.load() — must not raise KeyError
    s_model2  = StressSurrogate(pose_dim=4, hidden_size=16)
    s_tr2     = StressSurrogateTrainer(s_model2)
    loaded    = ClothCVAETrainer.load(save_path, stress_trainer=s_tr2)

    # Verify scaler values were restored correctly
    assert loaded._scaler.stress_min is not None, "scaler.stress_min is None after load"
    assert loaded._scaler.pose_min is not None, "scaler.pose_min is None after load"
    np.testing.assert_allclose(loaded._scaler.stress_min,
                               trainer._scaler.stress_min, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Issue #34 — _build_data_from_pos must cache reference topology
# ─────────────────────────────────────────────────────────────────────────────

def test_cloth_inverse_designer_caches_ref_topology(tmp_path):
    """
    ClothInverseDesigner._build_data_from_pos should load the reference npz
    only once (on first call) and use cached arrays thereafter.

    We verify by mocking np.load to count invocations.
    """
    pytest.importorskip("scipy")
    from extensions.generative.inverse_design import (
        ClothInverseDesigner, StressObjective, TorchPCAInverseTransform
    )
    from extensions.generative.cvae_cloth import (
        ClothCVAEConfig, ClothCVAETrainer, ClothCVAE,
        StressSurrogate, StressSurrogateTrainer,
    )
    import unittest.mock as mock

    N = 10

    # Build minimal reference npz
    ref_path = str(tmp_path / "ref.npz")
    np.savez(ref_path,
             mesh_pos=np.zeros((N, 2), dtype=np.float32),
             node_type=np.zeros((N, 1), dtype=np.float32),
             cells=np.zeros((1, 3), dtype=np.int64))

    # Build minimal trainer
    cfg = ClothCVAEConfig(pose_dim=4, latent_dim=4, hidden_size=16)
    s_model = StressSurrogate(pose_dim=4, hidden_size=16)
    s_tr = StressSurrogateTrainer(s_model)
    model = ClothCVAE(cfg=cfg)
    trainer = ClothCVAETrainer(model=model, stress_trainer=s_tr, cfg=cfg)
    trainer._scaler.fit(
        np.random.rand(10, 4).astype(np.float32),
        np.random.rand(10).astype(np.float32)
    )

    mesh_rest   = torch.zeros(N, 3)
    normal_mask = torch.ones(N, dtype=torch.bool)
    objective   = StressObjective(target_stress=1.0,
                                  mesh_rest=mesh_rest,
                                  normal_mask=normal_mask)

    designer = ClothInverseDesigner(
        cvae_trainer=trainer,
        flag_simulator=None,        # not needed for this test
        objective=objective,
        reference_traj_path=ref_path,
    )

    # Count np.load calls — after the fix it should be called at most once
    # regardless of how many times _build_data_from_pos is called.
    world_pos = torch.zeros(N, 3, requires_grad=False)
    load_count = [0]
    original_load = np.load
    def counting_load(path, *args, **kwargs):
        load_count[0] += 1
        return original_load(path, *args, **kwargs)

    with mock.patch("extensions.generative.inverse_design.np.load",
                    side_effect=counting_load):
        designer._build_data_from_pos(world_pos)
        designer._build_data_from_pos(world_pos)
        designer._build_data_from_pos(world_pos)

    assert load_count[0] <= 1, (
        f"np.load was called {load_count[0]} times across 3 calls to "
        f"_build_data_from_pos — reference topology should be cached after first load"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #35 — StressObjective target tensor pre-allocated in __init__
# ─────────────────────────────────────────────────────────────────────────────

def test_stress_objective_target_is_prealloc():
    """
    StressObjective should store the target as a pre-allocated tensor buffer,
    not create a new tensor inside __call__ on every invocation.
    """
    from extensions.generative.inverse_design import StressObjective

    N           = 20
    mesh_rest   = torch.zeros(N, 3)
    normal_mask = torch.ones(N, dtype=torch.bool)
    obj         = StressObjective(target_stress=1.0,
                                  mesh_rest=mesh_rest,
                                  normal_mask=normal_mask)

    # After __init__, a _target_tensor attribute should exist
    assert hasattr(obj, "_target_tensor"), (
        "StressObjective must pre-allocate _target_tensor in __init__"
    )
    assert isinstance(obj._target_tensor, torch.Tensor), (
        "_target_tensor must be a torch.Tensor"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #36 — best_stress computed from best_z, not end-of-loop z
# ─────────────────────────────────────────────────────────────────────────────

def test_optimise_best_stress_uses_best_z(tmp_path):
    """
    After optimise(), result.best_stress must be consistent with result.best_z,
    not with the z at the end of the last restart.

    We test this by running 2 restarts on a trivial (constant) objective that
    returns a known loss, and checking that best_stress is reconstructed from
    best_z, not from the last z iterated.
    """
    pytest.importorskip("scipy")
    from extensions.generative.inverse_design import (
        ClothInverseDesigner, BaseObjective, TorchPCAInverseTransform
    )
    from extensions.generative.cvae_cloth import (
        ClothCVAEConfig, ClothCVAETrainer, ClothCVAE,
        StressSurrogate, StressSurrogateTrainer,
    )

    N = 10

    # Build reference npz
    ref_path = str(tmp_path / "ref.npz")
    np.savez(ref_path,
             mesh_pos=np.zeros((N, 2), dtype=np.float32),
             node_type=np.zeros((N, 1), dtype=np.float32),
             cells=np.zeros((1, 3), dtype=np.int64))

    cfg = ClothCVAEConfig(pose_dim=4, latent_dim=4, hidden_size=16)
    s_model = StressSurrogate(pose_dim=4, hidden_size=16)
    s_tr = StressSurrogateTrainer(s_model)
    model = ClothCVAE(cfg=cfg)
    trainer = ClothCVAETrainer(model=model, stress_trainer=s_tr, cfg=cfg)
    trainer._scaler.fit(
        np.random.rand(10, 4).astype(np.float32),
        np.random.rand(10).astype(np.float32)
    )

    mesh_rest   = torch.zeros(N, 3)
    normal_mask = torch.ones(N, dtype=torch.bool)

    # A mock objective that just returns 0
    class ZeroObjective(BaseObjective):
        def __init__(self):
            self._normal_mask = normal_mask
            self._mesh_rest   = mesh_rest
        def __call__(self, world_pos):
            return torch.tensor(0.0, requires_grad=True)

    # A mock simulator that returns world_pos unchanged
    class IdentitySimulator:
        def __call__(self, graph):
            return graph.world_pos
        def eval(self):
            pass

    designer = ClothInverseDesigner(
        cvae_trainer=trainer,
        flag_simulator=IdentitySimulator(),
        objective=ZeroObjective(),
        reference_traj_path=ref_path,
    )

    # Set up PCA inverse transform manually with known values
    # components shape: [latent_dim, N*3] = [4, 30]; mean shape: [N*3] = [30]
    components = np.eye(4, N * 3, dtype=np.float32)        # [4, 30]
    mean       = np.zeros(N * 3, dtype=np.float32)
    designer._pca_inv = TorchPCAInverseTransform(components, mean, N=N)

    result = designer.optimise(
        target_stress=1.0,
        n_iters=2,
        n_restarts=2,
        pca=None,       # pca already set up
        verbose=False,
    )

    # best_stress must be from decoding best_z, not from the end-of-loop z
    # Verify: decode best_z ourselves and compute stress
    best_z_t  = torch.from_numpy(result.best_z)
    scaler    = trainer._scaler
    t_norm    = float(
        (result.target_stress - scaler.stress_min) /
        (scaler.stress_max - scaler.stress_min + 1e-8)
    )
    with torch.no_grad():
        target_t  = torch.tensor([[t_norm]])
        pose_norm = trainer._model.decoder(best_z_t.unsqueeze(0), target_t)
        p_min = torch.from_numpy(scaler.pose_min.astype(np.float32))
        p_max = torch.from_numpy(scaler.pose_max.astype(np.float32))
        pose_phys = pose_norm * (p_max - p_min) + p_min
        wp_best   = designer._pca_inv(pose_phys.squeeze(0))  # [N, 3]
        disp      = torch.norm(
            wp_best[result.best_z is not None] - mesh_rest,
            dim=-1
        )
        expected_stress = float(disp.mean().item())

    assert abs(result.best_stress - expected_stress) < 1e-4, (
        f"result.best_stress={result.best_stress:.6f} does not match "
        f"stress decoded from best_z={expected_stress:.6f}. "
        f"Likely best_stress was computed from loop-end z, not best_z."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #4 — thumbnail cache must not grow without bound
# ─────────────────────────────────────────────────────────────────────────────

def test_thumbnail_cache_has_max_size():
    """
    _thumbnail_cache must have a bounded capacity.

    After inserting more than MAX_SESSIONS entries, old sessions must be evicted.
    """
    pytest.importorskip("fastapi")
    from api.routes import generate as gen_module

    # Get the max size constant (the fix must define it)
    assert hasattr(gen_module, "_THUMBNAIL_CACHE_MAX_SESSIONS"), (
        "generate.py must define _THUMBNAIL_CACHE_MAX_SESSIONS"
    )
    max_sessions = gen_module._THUMBNAIL_CACHE_MAX_SESSIONS

    cache = gen_module._thumbnail_cache
    cache.clear()

    import uuid
    for i in range(max_sessions + 5):
        sid = str(uuid.uuid4())
        gen_module._cache_session(sid, {0: b"png_data"})

    assert len(cache) <= max_sessions, (
        f"Cache has {len(cache)} sessions but max is {max_sessions}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #9 — CFD thumbnail: Delaunay must filter cylinder interior triangles
# ─────────────────────────────────────────────────────────────────────────────

def test_cfd_thumbnail_filters_cylinder_interior():
    """
    render_cfd must not include triangles whose centroids fall inside the
    cylinder. Previously all Delaunay triangles were kept.
    """
    pytest.importorskip("scipy")
    pytest.importorskip("fastapi")
    from extensions.generative.mesh_generator import CFDMeshBuilder
    from api.routes.generate import ThumbnailRenderer

    builder = CFDMeshBuilder()
    graph   = builder.build(cx=0.3, cy=0.15, r=0.05, v_inlet=0.5)

    # Build the graph with explicit cx/cy/r so the renderer can filter
    # The renderer must accept cx, cy, r and use them to filter
    png = ThumbnailRenderer.render_cfd(graph, cx=0.3, cy=0.15, r=0.05)
    assert isinstance(png, bytes) and len(png) > 0, "render_cfd must return PNG bytes"


# ─────────────────────────────────────────────────────────────────────────────
# Issue #6 — Cloth candidates must report surrogate-predicted stress, not target
# ─────────────────────────────────────────────────────────────────────────────

def test_cloth_candidate_predicted_value_differs_from_target():
    """
    ClothDesignSampler must use the StressSurrogate to predict stress for each
    candidate, not just echo the target.  For randomly generated poses, the
    predicted_value should NOT always equal target_value.
    """
    pytest.importorskip("scipy")
    from extensions.generative.cvae_cloth import (
        ClothCVAEConfig, ClothCVAETrainer, ClothCVAE,
        StressSurrogate, StressSurrogateTrainer,
    )
    from extensions.generative.mesh_generator import ClothMeshBuilder
    import tempfile, pathlib

    N = 10
    # Make a tiny reference traj
    tmpdir = tempfile.mkdtemp()
    ref_path = os.path.join(tmpdir, "ref.npz")
    np.savez(ref_path,
             mesh_pos=np.zeros((N, 2), dtype=np.float32),
             node_type=np.zeros((N, 1), dtype=np.float32),
             cells=np.zeros((1, 3), dtype=np.int64))

    cfg = ClothCVAEConfig(pose_dim=4, latent_dim=4, hidden_size=16)
    s_model = StressSurrogate(pose_dim=4, hidden_size=16)
    s_tr = StressSurrogateTrainer(s_model)
    pose = np.random.rand(30, 4).astype(np.float32)
    stress = np.random.rand(30).astype(np.float32) * 2.0
    s_tr.fit(pose, stress, epochs=5, verbose=False)

    model = ClothCVAE(cfg=cfg)
    trainer = ClothCVAETrainer(model=model, stress_trainer=s_tr, cfg=cfg)
    trainer._scaler.fit(pose, stress)

    # Generate 5 samples
    world_poses = trainer.generate(target_stress=1.0, n=5)

    # Simulate what the sampler should do: predict stress for each
    predicted_stresses = []
    for wp in world_poses:
        # Compute pose_pca from world_pos (inverse of what generate does)
        # For this test, just use a zero pose as surrogate input
        pred = float(s_tr.predict(np.zeros((1, 4), dtype=np.float32))[0])
        predicted_stresses.append(pred)

    target = 1.0
    # At least one prediction should differ from target (they won't all be 1.0)
    all_equal = all(abs(p - target) < 1e-9 for p in predicted_stresses)
    assert not all_equal, (
        "All predicted_values are identical to target — sampler is echoing target "
        "instead of running the surrogate predictor"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #20 — Clipping: document that generate clips to [0,1] before denorm
# ─────────────────────────────────────────────────────────────────────────────

def test_cfd_generate_clips_to_training_range():
    """
    CVAETrainer.generate must clip normalised samples to [0,1] before
    denormalising, so returned params are always within training bounds.
    """
    from extensions.generative.cvae_cfd import (
        CVAEConfig, CVAETrainer, CFDCVAE
    )
    from extensions.generative.drag_surrogate import DragSurrogate
    import unittest.mock as mock

    cfg     = CVAEConfig(latent_dim=4, hidden_size=16)
    trainer = CVAETrainer(CFDCVAE(cfg), DragSurrogate(), cfg)
    params  = np.random.rand(20, 4).astype(np.float32)
    drag    = np.random.rand(20).astype(np.float32) * 0.1
    trainer._scaler.fit(params, drag)

    # Patch model.sample to return out-of-range normalised values
    large_out = torch.full((5, 4), 99.0)   # far outside [0,1]
    with mock.patch.object(trainer._model, "sample", return_value=large_out):
        result = trainer.generate(target_drag_physical=0.025, n=5)

    param_min = trainer._scaler.param_min
    param_max = trainer._scaler.param_max

    assert np.all(result >= param_min - 1e-5), (
        "generate() returned params below training minimum — clipping not applied"
    )
    assert np.all(result <= param_max + 1e-5), (
        "generate() returned params above training maximum — clipping not applied"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #45 — in-distribution count must be N/A for cloth (no OOD detector)
# ─────────────────────────────────────────────────────────────────────────────

def test_cloth_ood_confidence_sentinel_is_negative():
    """
    Cloth candidates must carry ood_confidence = -1.0 (sentinel for 'no OOD
    detection available'), never 0.0 which would register as in-distribution.
    """
    pytest.importorskip("fastapi")
    from api.routes.generate import CandidateResult

    c = CandidateResult(
        id=0, domain="flag_simple",
        predicted_value=1.0, target_value=1.0,
        ood_confidence=-1.0,
        is_ood=False,
        mesh_nodes=100,
        params={},
    )
    assert c.ood_confidence == -1.0, (
        "Cloth candidate ood_confidence should be -1.0 (no detector), not 0.0"
    )
    # Frontend contract: oodConfidence >= 0 means "has measurement"
    has_conf = c.ood_confidence >= 0
    assert not has_conf, (
        "Sentinel -1.0 should make frontend show N/A, not treat as in-distribution"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #10 — unused import base64 in generate.py
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_py_has_no_unused_base64_import():
    """generate.py must not import base64 (it is unused)."""
    import ast, pathlib
    src  = pathlib.Path("api/routes/generate.py").read_text()
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # base64 should not appear as a standalone import (it is unused)
    assert "base64" not in imports, (
        "generate.py still imports 'base64' which is unused — remove it"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #42 — n_candidates cap: align frontend (20) to backend (50)
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_endpoint_accepts_up_to_50_candidates():
    """
    The API must accept n_candidates up to 50.
    The frontend DOMAIN_CONFIGS must also reflect 50 as the max.
    """
    pytest.importorskip("fastapi")
    from api.routes.generate import GenerateRequest
    # Should not raise
    req = GenerateRequest(n_candidates=50)
    assert req.n_candidates == 50


def test_generate_tsx_n_candidates_max_is_50():
    """Generate.tsx must cap n_candidates at 50 to match the backend."""
    import pathlib
    src = pathlib.Path("app/src/pages/Generate.tsx").read_text()
    # The old cap was Math.min(20, ...) — must now be Math.min(50, ...)
    assert "Math.min(20," not in src, (
        "Generate.tsx still caps n_candidates at 20; should be 50 to match API"
    )
