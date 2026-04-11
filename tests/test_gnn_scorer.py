"""Tests for GnnScorer adaptive rollout and drag extraction."""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


def test_extract_drag_from_outflow_nodes():
    """_extract_drag returns mean x-velocity magnitude over OUTFLOW nodes."""
    from extensions.generative.gnn_scorer import GnnScorer

    n_nodes = 50
    velocity = torch.zeros(n_nodes, 2)
    velocity[40:, 0] = 0.5   # OUTFLOW nodes: vx = 0.5
    velocity[40:, 1] = 0.3   # vy = 0.3

    node_type = torch.zeros(n_nodes, dtype=torch.long)
    node_type[40:] = 5  # NodeType.OUTFLOW = 5

    drag = GnnScorer._extract_drag(velocity, node_type)
    assert abs(drag - 0.5) < 1e-5


def test_extract_drag_fallback_no_outflow():
    """_extract_drag falls back to all nodes if no OUTFLOW nodes exist."""
    from extensions.generative.gnn_scorer import GnnScorer

    n_nodes = 20
    velocity = torch.full((n_nodes, 2), 0.4)
    node_type = torch.zeros(n_nodes, dtype=torch.long)  # all NORMAL

    drag = GnnScorer._extract_drag(velocity, node_type)
    assert abs(drag - 0.4) < 1e-5


def test_gnn_scorer_score_candidates_returns_one_per_graph():
    """score_candidates returns exactly one GnnScore per input graph."""
    from extensions.generative.gnn_scorer import GnnScorer, GnnScore
    from torch_geometric.data import Data

    scorer = GnnScorer.__new__(GnnScorer)
    scorer.device = "cpu"
    scorer.simulator = MagicMock()   # score_candidates() calls simulator.eval() once per batch

    fake_score = GnnScore(gnn_predicted_value=0.3, converged=True)
    graphs = [MagicMock(), MagicMock(), MagicMock()]

    with patch.object(scorer, '_adaptive_rollout', return_value=fake_score):
        results = scorer.score_candidates(graphs, device="cpu")

    assert len(results) == 3
    assert all(isinstance(r, GnnScore) for r in results)
    assert all(r.gnn_predicted_value == 0.3 for r in results)


def test_generate_request_has_mode_field():
    """GenerateRequest accepts mode='quick' and mode='deep'."""
    from api.routes.generate import GenerateRequest
    req_quick = GenerateRequest(mode="quick")
    req_deep  = GenerateRequest(mode="deep")
    assert req_quick.mode == "quick"
    assert req_deep.mode  == "deep"


def test_candidate_result_has_gnn_fields():
    """CandidateResult has the 4 new optional GNN fields defaulting to None/False."""
    from api.routes.generate import CandidateResult
    c = CandidateResult(
        id=0, domain="cylinder_flow",
        predicted_value=0.03, target_value=0.025,
        ood_confidence=-1.0, is_ood=False,
        mesh_nodes=100, params={},
    )
    assert c.gnn_predicted_value is None
    assert c.score_gap           is None
    assert c.gnn_converged       is None
    assert c.gnn_failed          is False


def test_get_gnn_scorer_returns_cached_instance():
    """get_gnn_scorer returns the same object on repeated calls (cached)."""
    import importlib
    import api.state as state_module

    # Clear cache before test
    from api.state import clear_gnn_scorer_cache
    clear_gnn_scorer_cache()

    from unittest.mock import patch, MagicMock
    mock_scorer = MagicMock()

    with patch('api.state.GnnScorer', return_value=mock_scorer) as MockClass:
        from api.state import get_gnn_scorer
        s1 = get_gnn_scorer("checkpoints/best_model.pth", "cpu")
        s2 = get_gnn_scorer("checkpoints/best_model.pth", "cpu")

        assert s1 is s2
        assert MockClass.call_count == 1  # constructed only once
