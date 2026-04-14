"""
Tests for extract_true_drag() and DragSurrogateTrainer.fit() extensions.

Covers:
  1. extract_true_drag() returns a finite float when simulator/dataset behave.
  2. extract_true_drag() returns NaN when the simulator raises.
  3. fit() uses true_drag_labels for the overridden indices.
  4. fit() falls back to analytical formula when true_drag_labels is None.
  5. fit() ignores NaN values in true_drag_labels (analytical fallback per row).
  6. fit() accepts y=None and auto-computes analytical labels.
"""
from __future__ import annotations

import copy
import math
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_surrogate_trainer(n: int = 20):
    """Return a fresh (DragSurrogate, DragSurrogateTrainer) pair."""
    from extensions.generative.drag_surrogate import (
        DragSurrogate, DragSurrogateTrainer, SurrogateConfig,
    )
    cfg     = SurrogateConfig(epochs=5, batch_size=8)
    model   = DragSurrogate(config=cfg)
    trainer = DragSurrogateTrainer(surrogate=model, config=cfg, device='cpu')
    return model, trainer


def _make_params(n: int = 20) -> np.ndarray:
    """[N, 4] design params with valid values."""
    rng = np.random.default_rng(0)
    params = rng.uniform(
        [0.1, 0.15, 0.03, 0.5],
        [0.9, 0.35, 0.08, 1.5],
        size=(n, 4),
    ).astype(np.float32)
    return params


# ---------------------------------------------------------------------------
# extract_true_drag tests
# ---------------------------------------------------------------------------

class TestExtractTrueDrag:

    def _make_mock_dataset_and_graph(self, n_nodes=60, n_steps=10):
        """
        Build a mock FpcDataset and a mock graph that the simulator will see.

        OUTFLOW nodes are the last 10 nodes (type == 5).
        Simulator.forward returns a constant vx = 0.4 for all nodes.
        """
        import torch_geometric.transforms as T
        from torch_geometric.data import Data

        # ── Minimal graph that survives FaceToEdge + Cartesian + Distance ───
        # We need pos (node positions) and a face attribute for FaceToEdge.
        # Use 3 triangles connecting the first 9 nodes.
        n_nodes = max(n_nodes, 9)
        pos   = torch.rand(n_nodes, 2)
        face  = torch.tensor([[0, 3, 6], [1, 4, 7], [2, 5, 8]], dtype=torch.long).T  # [3, 3]

        # node_type: nodes 50+ are OUTFLOW (type 5)
        node_type = torch.zeros(n_nodes, dtype=torch.float)
        node_type[50:] = 5.0  # OUTFLOW

        # x = [node_type, vx, vy, ...] — at minimum 3 columns
        x = torch.zeros(n_nodes, 3)
        x[:, 0] = node_type
        x[:, 1] = 0.1  # initial vx

        # y = ground-truth next velocity [N, 2] (used to pin boundary nodes)
        y = torch.zeros(n_nodes, 2)

        graph = Data(x=x, pos=pos, face=face, y=y)

        # Mock dataset — return a fresh deepcopy each call so in-place
        # mutations (e.g. graph.x[:, 1:3] = predicted_velocity) from one
        # step do not bleed into the next step.
        dataset               = MagicMock()
        dataset.num_sampes_per_tra = n_steps
        dataset.__getitem__   = MagicMock(
            side_effect=lambda idx: copy.deepcopy(graph)
        )

        return dataset, graph, node_type

    def test_returns_finite_float(self):
        """extract_true_drag returns a finite float with a working simulator."""
        from extensions.generative.drag_surrogate import extract_true_drag

        dataset, graph, node_type = self._make_mock_dataset_and_graph()

        # Simulator returns vx=0.4 for every node
        def fake_forward(graph, velocity_sequence_noise=None):
            n = graph.x.shape[0]
            out = torch.zeros(n, 2)
            out[:, 0] = 0.4   # vx = 0.4
            return out

        simulator             = MagicMock(side_effect=fake_forward)
        simulator.eval        = MagicMock()

        drag = extract_true_drag(simulator, dataset, trajectory_index=0,
                                 device='cpu', steady_state_frac=0.5)

        assert isinstance(drag, float), "should return float"
        assert math.isfinite(drag),     "should be finite"
        # All nodes return |vx|=0.4 — so drag should be ~0.4
        assert abs(drag - 0.4) < 0.05, f"expected ~0.4, got {drag}"

    def test_returns_nan_on_simulator_exception(self):
        """extract_true_drag returns NaN when the simulator raises."""
        from extensions.generative.drag_surrogate import extract_true_drag

        dataset, graph, node_type = self._make_mock_dataset_and_graph()

        simulator             = MagicMock(side_effect=RuntimeError("boom"))
        simulator.eval        = MagicMock()

        drag = extract_true_drag(simulator, dataset, trajectory_index=0, device='cpu')

        assert math.isnan(drag), "should return NaN on failure"

    def test_uses_outflow_nodes(self):
        """OUTFLOW nodes (type 5) determine drag; other nodes are ignored."""
        from extensions.generative.drag_surrogate import extract_true_drag

        dataset, graph, node_type = self._make_mock_dataset_and_graph(n_nodes=60, n_steps=5)

        # Simulator: OUTFLOW nodes (50+) get vx=0.7, others get vx=0.1
        def fake_forward(graph, velocity_sequence_noise=None):
            n  = graph.x.shape[0]
            out = torch.zeros(n, 2)
            # detect OUTFLOW from current graph.x[:, 0]
            nt  = graph.x[:, 0]
            out[:, 0] = torch.where(nt == 5, torch.tensor(0.7), torch.tensor(0.1))
            return out

        simulator             = MagicMock(side_effect=fake_forward)
        simulator.eval        = MagicMock()

        drag = extract_true_drag(simulator, dataset, trajectory_index=0,
                                 device='cpu', steady_state_frac=1.0)

        assert math.isfinite(drag)
        # Should be close to 0.7 (OUTFLOW) not 0.1 (other nodes)
        assert abs(drag - 0.7) < 0.1, f"expected ~0.7 (OUTFLOW vx), got {drag}"

    def test_steady_state_frac_uses_last_steps(self):
        """Verify that steady_state_frac=0.5 averages the LAST 50% of steps, not first."""
        from extensions.generative.drag_surrogate import extract_true_drag

        total_steps = 10
        dataset, base_graph, _ = self._make_mock_dataset_and_graph(
            n_nodes=60, n_steps=total_steps
        )

        # Track how many times the simulator has been called
        call_count = [0]

        def mock_forward(graph, velocity_sequence_noise=None):
            call_count[0] += 1
            step = call_count[0]
            # First half of steps → vx=0.1; second half → vx=0.9
            vx = 0.1 if step <= total_steps // 2 else 0.9
            out = torch.zeros(graph.x.shape[0], 2)
            out[:, 0] = vx
            return out

        simulator       = MagicMock(side_effect=mock_forward)
        simulator.eval  = MagicMock()

        drag = extract_true_drag(
            simulator=simulator,
            dataset=dataset,
            trajectory_index=0,
            device='cpu',
            steady_state_frac=0.5,
        )

        # steady_state_frac=0.5 must use the LAST 50% (steps 6-10, vx=0.9)
        # NOT the first 50% (steps 1-5, vx=0.1)
        assert math.isfinite(drag), f"drag should be finite, got {drag}"
        assert drag > 0.5, (
            f"Expected ~0.9 (last steps averaged), got {drag}. "
            "steady_state_frac may be slicing from the front instead of the back."
        )


# ---------------------------------------------------------------------------
# DragSurrogateTrainer.fit() tests
# ---------------------------------------------------------------------------

class TestDragSurrogateTrainerFit:

    def test_fit_with_analytical_labels_only(self):
        """fit(X, y) trains normally — baseline backward-compatible path."""
        _, trainer = _make_surrogate_trainer()
        X = _make_params(30)
        from extensions.generative.drag_surrogate import DragProxyComputer
        y = DragProxyComputer()(X)

        losses = trainer.fit(X, y, verbose=False)

        assert len(losses) == 5   # 5 epochs
        assert all(math.isfinite(l) for l in losses)

    def test_fit_y_none_auto_computes_labels(self):
        """fit(X, y=None) computes analytical labels automatically."""
        _, trainer = _make_surrogate_trainer()
        X = _make_params(20)

        losses = trainer.fit(X, verbose=False)   # y omitted

        assert len(losses) == 5
        assert all(math.isfinite(l) for l in losses)

    def test_true_drag_labels_override_analytical(self):
        """When true_drag_labels provided, those rows use physics values."""
        _, trainer = _make_surrogate_trainer(n=20)
        X = _make_params(20)

        # Override first 5 rows with a very large drag value
        true_drag = {i: 999.0 for i in range(5)}

        # We intercept what label array gets fed to the scaler
        captured = {}

        _orig_fit = trainer._scaler.fit

        def patched_scaler_fit(X_arr, y_arr):
            captured['y'] = y_arr.copy()
            return _orig_fit(X_arr, y_arr)

        trainer._scaler.fit = patched_scaler_fit

        trainer.fit(X, true_drag_labels=true_drag, verbose=False)

        assert 'y' in captured, "scaler.fit was not called"
        y_used = captured['y']
        # First 5 rows should reflect the overridden value 999.0
        for i in range(5):
            assert abs(y_used[i] - 999.0) < 1.0, (
                f"row {i}: expected ~999.0, got {y_used[i]}"
            )
        # Remaining rows should NOT be 999.0 (they use analytical formula)
        for i in range(5, 20):
            assert y_used[i] < 500.0, (
                f"row {i}: should use analytical, not override; got {y_used[i]}"
            )

    def test_fallback_when_true_drag_labels_none(self):
        """When true_drag_labels=None, uses analytical formula for all rows."""
        from extensions.generative.drag_surrogate import DragProxyComputer
        _, trainer = _make_surrogate_trainer()
        X = _make_params(20)
        analytical = DragProxyComputer()(X)

        captured = {}
        _orig_fit = trainer._scaler.fit

        def patched_scaler_fit(X_arr, y_arr):
            captured['y'] = y_arr.copy()
            return _orig_fit(X_arr, y_arr)

        trainer._scaler.fit = patched_scaler_fit

        trainer.fit(X, true_drag_labels=None, verbose=False)

        assert 'y' in captured
        np.testing.assert_allclose(captured['y'], analytical, rtol=1e-5)

    def test_nan_in_true_drag_labels_skipped(self):
        """NaN entries in true_drag_labels are ignored (analytical used instead)."""
        from extensions.generative.drag_surrogate import DragProxyComputer
        _, trainer = _make_surrogate_trainer()
        X = _make_params(10)
        analytical = DragProxyComputer()(X)

        # Pass NaN for all indices — should behave identically to no labels
        true_drag = {i: float('nan') for i in range(10)}

        captured = {}
        _orig_fit = trainer._scaler.fit

        def patched_scaler_fit(X_arr, y_arr):
            captured['y'] = y_arr.copy()
            return _orig_fit(X_arr, y_arr)

        trainer._scaler.fit = patched_scaler_fit

        trainer.fit(X, true_drag_labels=true_drag, verbose=False)

        assert 'y' in captured
        np.testing.assert_allclose(captured['y'], analytical, rtol=1e-5)
