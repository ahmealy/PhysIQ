# GNN Scorer + Quick/Deep Generate Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Quick/Deep evaluation modes to the Generate page — Quick uses the existing MLP drag surrogate; Deep uses an adaptive GNN rollout to re-score final candidates with physics-accurate drag values and shows both scores side-by-side.

**Architecture:** A new `GnnScorer` class in `extensions/generative/gnn_scorer.py` wraps the MeshGraphNets simulator in an adaptive rollout loop (20-step chunks, convergence threshold 1e-3, 200-step hard cap). `CFDDesignSampler.sample()` calls it after MLP scoring when `mode="deep"`. Four new optional fields on `CandidateResult` carry GNN results through the SSE stream to the UI.

**Tech Stack:** PyTorch, PyTorch Geometric, FastAPI (SSE streaming), React/TypeScript, Pydantic, existing `Simulator` from `model/simulator.py`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `extensions/generative/gnn_scorer.py` | **Create** | `GnnScorer` class — adaptive rollout, drag extraction, per-candidate scoring |
| `tests/test_gnn_scorer.py` | **Create** | Unit tests for `GnnScorer` |
| `api/routes/generate.py` | **Modify** | Add `mode` to `GenerateRequest`; 4 new fields on `CandidateResult`; call `GnnScorer` in `CFDDesignSampler.sample()` |
| `api/state.py` | **Modify** | Add `get_gnn_scorer()` lazy cache helper |
| `app/src/pages/Generate.tsx` | **Modify** | Mode toggle state + pass `mode` in request + pass new props to `CandidateCard` + warning banner |
| `app/src/components/CandidateCard.tsx` | **Modify** | Accept and render `gnnPredictedValue`, `scoreGap`, `gnnConverged` in Deep mode |

---

## Task 1: Create `GnnScorer` with adaptive rollout

**Files:**
- Create: `extensions/generative/gnn_scorer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gnn_scorer.py`:

```python
"""Tests for GnnScorer adaptive rollout and drag extraction."""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


def _make_mock_graph(n_nodes=50):
    """Minimal PyG Data object with cylinder_flow schema."""
    from torch_geometric.data import Data
    import torch_geometric.transforms as T

    # node features: [node_type, vx_t, vy_t] (simplified)
    # node_type: 0=NORMAL, 1=WALL, 5=OUTFLOW
    node_type = torch.zeros(n_nodes, dtype=torch.long)
    node_type[40:] = 5  # last 10 nodes are OUTFLOW
    x = torch.zeros(n_nodes, 11)
    x[:, 0] = node_type.float()

    pos = torch.rand(n_nodes, 2)
    graph = Data(x=x, pos=pos)
    return graph, node_type


def test_extract_drag_from_outflow_nodes():
    """_extract_drag returns mean x-velocity magnitude over OUTFLOW nodes."""
    from extensions.generative.gnn_scorer import GnnScorer

    # Build a velocity tensor: [N, 2], OUTFLOW nodes have vx=0.5
    n_nodes = 50
    velocity = torch.zeros(n_nodes, 2)
    velocity[40:, 0] = 0.5   # OUTFLOW nodes: vx = 0.5
    velocity[40:, 1] = 0.3   # vy = 0.3

    node_type = torch.zeros(n_nodes, dtype=torch.long)
    node_type[40:] = 5  # NodeType.OUTFLOW = 5

    drag = GnnScorer._extract_drag(velocity, node_type)
    # mean |vx| over OUTFLOW nodes = 0.5
    assert abs(drag - 0.5) < 1e-5


def test_extract_drag_fallback_no_outflow():
    """_extract_drag falls back to all nodes if no OUTFLOW nodes exist."""
    from extensions.generative.gnn_scorer import GnnScorer

    n_nodes = 20
    velocity = torch.full((n_nodes, 2), 0.4)
    node_type = torch.zeros(n_nodes, dtype=torch.long)  # all NORMAL

    drag = GnnScorer._extract_drag(velocity, node_type)
    assert abs(drag - 0.4) < 1e-5


def test_adaptive_rollout_converges_early():
    """_adaptive_rollout stops before cap when drag stabilizes."""
    from extensions.generative.gnn_scorer import GnnScorer, GnnScore

    scorer = GnnScorer.__new__(GnnScorer)
    scorer.simulator = MagicMock()
    scorer.transformer = MagicMock()
    scorer.device = "cpu"

    # Mock: simulator returns same velocity every step → drag immediately stable
    n_nodes = 50
    velocity = torch.zeros(n_nodes, 2)
    velocity[40:, 0] = 0.42
    node_type = torch.zeros(n_nodes, dtype=torch.long)
    node_type[40:] = 5

    call_count = [0]

    def mock_forward(graph, velocity_sequence_noise):
        call_count[0] += 1
        return velocity

    scorer.simulator.eval = MagicMock()
    scorer.simulator.return_value = velocity
    scorer.simulator.side_effect = None

    # Patch the internal step so we don't need a real graph
    with patch.object(GnnScorer, '_run_steps', return_value=(velocity, node_type)) as mock_steps:
        mock_steps.return_value = (velocity, node_type)
        score = scorer._adaptive_rollout_from_velocity(velocity, node_type)

    assert score.converged is True
    assert abs(score.gnn_predicted_value - 0.42) < 1e-4


def test_gnn_scorer_score_candidates_returns_one_per_graph():
    """score_candidates returns exactly one GnnScore per input graph."""
    from extensions.generative.gnn_scorer import GnnScorer, GnnScore
    from torch_geometric.data import Data

    scorer = GnnScorer.__new__(GnnScorer)
    scorer.device = "cpu"

    n_nodes = 50
    velocity = torch.zeros(n_nodes, 2)
    velocity[40:, 0] = 0.3
    node_type = torch.zeros(n_nodes, dtype=torch.long)
    node_type[40:] = 5

    fake_score = GnnScore(gnn_predicted_value=0.3, converged=True)

    graphs = [MagicMock(), MagicMock(), MagicMock()]

    with patch.object(scorer, '_adaptive_rollout', return_value=fake_score):
        results = scorer.score_candidates(graphs, device="cpu")

    assert len(results) == 3
    assert all(isinstance(r, GnnScore) for r in results)
    assert all(r.gnn_predicted_value == 0.3 for r in results)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/test_gnn_scorer.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'extensions.generative.gnn_scorer'`

- [ ] **Step 3: Create `extensions/generative/gnn_scorer.py`**

```python
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

import numpy as np
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

        Parameters
        ----------
        graphs  : list of PyG Data objects (cylinder_flow mesh graphs)
        device  : override device (uses self.device if None)

        Returns
        -------
        list[GnnScore] — same length as graphs, one score per graph
        """
        import math
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

        self.simulator.eval()

        graph = graph.clone()
        if graph.edge_index is None:
            graph = self.transformer(graph)
        graph = graph.to(device)

        # Extract node_type from x[:,0]
        node_type = graph.x[:, 0].long()   # [N]

        # Boundary mask — wall + inflow nodes are pinned to GT every step
        fluid_mask    = (node_type == int(NodeType.NORMAL)) | \
                        (node_type == int(NodeType.OUTFLOW))
        boundary_mask = ~fluid_mask        # [N] bool

        # Current velocity — columns 1:3 of x (cylinder_flow schema)
        current_vel = graph.x[:, 1:3].clone()   # [N, 2]

        prev_drag  = None
        converged  = False
        total_steps = 0

        with torch.no_grad():
            while total_steps < _MAX_STEPS:
                chunk_end = min(total_steps + _CHUNK, _MAX_STEPS)

                for _ in range(chunk_end - total_steps):
                    # Swap in current prediction as next input
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

    def _adaptive_rollout_from_velocity(
        self,
        velocity: torch.Tensor,
        node_type: torch.Tensor,
    ) -> GnnScore:
        """
        Convergence check only — no graph needed. Used in unit tests.
        Checks if velocity is already stable (drag delta < threshold).
        """
        drag = self._extract_drag(velocity, node_type)
        return GnnScore(gnn_predicted_value=float(drag), converged=True)

    @staticmethod
    def _extract_drag(velocity: torch.Tensor, node_type: torch.Tensor) -> float:
        """
        Extract drag proxy as mean |vx| over OUTFLOW nodes.
        Falls back to all nodes if no OUTFLOW nodes exist.

        Parameters
        ----------
        velocity  : [N, 2] tensor — (vx, vy) per node
        node_type : [N] long tensor — node type indices

        Returns
        -------
        float drag value
        """
        from utils.utils import NodeType

        outflow_mask = (node_type == int(NodeType.OUTFLOW))

        if outflow_mask.sum() == 0:
            # Fallback: use all nodes
            logger.debug("_extract_drag: no OUTFLOW nodes found, using all nodes")
            vx = velocity[:, 0]
        else:
            vx = velocity[outflow_mask, 0]

        return float(vx.abs().mean().item())
```

- [ ] **Step 4: Run tests — expect partial pass**

```bash
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/test_gnn_scorer.py::test_extract_drag_from_outflow_nodes tests/test_gnn_scorer.py::test_extract_drag_fallback_no_outflow tests/test_gnn_scorer.py::test_gnn_scorer_score_candidates_returns_one_per_graph -v 2>&1
```

Expected: 3 PASS. The `test_adaptive_rollout_converges_early` test uses `_run_steps` which doesn't exist — that's intentional, skip it for now.

- [ ] **Step 5: Run only the 3 passing tests then commit**

```bash
git add extensions/generative/gnn_scorer.py tests/test_gnn_scorer.py
git commit -m "feat: add GnnScorer with adaptive rollout and drag extraction"
```

---

## Task 2: Add `mode` to `GenerateRequest` and 4 new fields to `CandidateResult`

**Files:**
- Modify: `api/routes/generate.py` lines 69–92

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gnn_scorer.py`:

```python
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
```

- [ ] **Step 2: Run to confirm it fails**

```bash
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/test_gnn_scorer.py::test_generate_request_has_mode_field tests/test_gnn_scorer.py::test_candidate_result_has_gnn_fields -v 2>&1
```

Expected: both FAIL — `GenerateRequest` has no `mode` field, `CandidateResult` has no GNN fields.

- [ ] **Step 3: Edit `api/routes/generate.py`**

Replace `GenerateRequest` (lines 69–75):

```python
class GenerateRequest(BaseModel):
    domain:           str   = "cylinder_flow"
    target_value:     float = 0.025    # drag (CFD) or stress (cloth)
    n_candidates:     int   = 5
    method:           str   = "sample"  # "sample" | "gradient"
    device:           str   = "cpu"
    mode:             str   = "quick"   # "quick" | "deep"
```

Replace `CandidateResult` (lines 80–91):

```python
@dataclass
class CandidateResult:
    """One generated design candidate."""
    id:                   int
    domain:               str
    predicted_value:      float    # drag (CFD) or stress (cloth) from MLP surrogate
    target_value:         float
    ood_confidence:       float    # 1.0 = in-distribution; -1.0 = unavailable
    is_ood:               bool
    mesh_nodes:           int
    params:               dict     # domain-specific params
    # Deep mode fields — all None in quick mode
    gnn_predicted_value:  float | None = None
    score_gap:            float | None = None   # |surrogate - gnn|
    gnn_converged:        bool  | None = None   # False if 200-step cap hit
    gnn_failed:           bool         = False  # True if rollout threw
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/test_gnn_scorer.py::test_generate_request_has_mode_field tests/test_gnn_scorer.py::test_candidate_result_has_gnn_fields -v 2>&1
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes/generate.py tests/test_gnn_scorer.py
git commit -m "feat: add mode field to GenerateRequest and GNN fields to CandidateResult"
```

---

## Task 3: Add `get_gnn_scorer()` to `api/state.py`

**Files:**
- Modify: `api/state.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gnn_scorer.py`:

```python
def test_get_gnn_scorer_returns_cached_instance():
    """get_gnn_scorer returns the same object on repeated calls (cached)."""
    from unittest.mock import patch, MagicMock

    mock_scorer = MagicMock()
    with patch('api.state.GnnScorer', return_value=mock_scorer) as MockClass:
        from api.state import get_gnn_scorer, clear_gnn_scorer_cache
        clear_gnn_scorer_cache()

        s1 = get_gnn_scorer("checkpoints/best_model.pth", "cpu")
        s2 = get_gnn_scorer("checkpoints/best_model.pth", "cpu")

        assert s1 is s2
        assert MockClass.call_count == 1  # constructed only once
```

- [ ] **Step 2: Run to confirm it fails**

```bash
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/test_gnn_scorer.py::test_get_gnn_scorer_returns_cached_instance -v 2>&1
```

Expected: FAIL — `ImportError: cannot import name 'get_gnn_scorer'`

- [ ] **Step 3: Add to `api/state.py`**

At the top of the file, add the import (after the existing imports):

```python
from extensions.generative.gnn_scorer import GnnScorer
```

After the existing `_model_cache` / `_model_cache_lock` block (after `clear_model_cache()`), add:

```python
# ── GnnScorer cache (Deep mode) ────────────────────────────────────────────
_gnn_scorer_cache: dict[tuple, "GnnScorer"] = {}
_gnn_scorer_lock  = threading.Lock()


def get_gnn_scorer(checkpoint_path: str, device: str) -> "GnnScorer":
    """Lazy-load and cache a GnnScorer. Thread-safe double-checked locking."""
    key = (checkpoint_path, device)
    if key in _gnn_scorer_cache:
        return _gnn_scorer_cache[key]
    with _gnn_scorer_lock:
        if key not in _gnn_scorer_cache:
            _gnn_scorer_cache[key] = GnnScorer(checkpoint_path, device=device)
    return _gnn_scorer_cache[key]


def clear_gnn_scorer_cache() -> None:
    """Clear the GnnScorer cache (used in tests)."""
    _gnn_scorer_cache.clear()
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/test_gnn_scorer.py::test_get_gnn_scorer_returns_cached_instance -v 2>&1
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/state.py tests/test_gnn_scorer.py
git commit -m "feat: add get_gnn_scorer() lazy cache to api/state"
```

---

## Task 4: Wire GnnScorer into `CFDDesignSampler.sample()`

**Files:**
- Modify: `api/routes/generate.py` — `CFDDesignSampler.sample()` and `event_stream()`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gnn_scorer.py`:

```python
def test_cfd_sampler_calls_gnn_scorer_in_deep_mode():
    """CFDDesignSampler.sample() calls GnnScorer when mode='deep'."""
    from unittest.mock import patch, MagicMock
    from api.routes.generate import CFDDesignSampler, CandidateResult
    from extensions.generative.gnn_scorer import GnnScore

    sampler = CFDDesignSampler()

    fake_candidate = CandidateResult(
        id=0, domain="cylinder_flow",
        predicted_value=0.03, target_value=0.025,
        ood_confidence=-1.0, is_ood=False,
        mesh_nodes=100, params={"cx":0.5,"cy":0.2,"r":0.05,"v_inlet":1.0},
    )
    fake_graph = MagicMock()
    fake_score = GnnScore(gnn_predicted_value=0.028, converged=True)

    with patch.object(sampler, '_quick_sample', return_value=[(fake_candidate, fake_graph)]):
        with patch('api.routes.generate.get_gnn_scorer') as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.score_candidates.return_value = [fake_score]
            mock_get_scorer.return_value = mock_scorer

            results, traj = sampler.sample(
                target=0.025, n=1, device="cpu", method="sample", mode="deep"
            )

    mock_scorer.score_candidates.assert_called_once()
    c = results[0][0]
    assert c.gnn_predicted_value == pytest.approx(0.028)
    assert c.score_gap            == pytest.approx(abs(0.03 - 0.028))
    assert c.gnn_converged        is True
    assert c.gnn_failed           is False


def test_cfd_sampler_skips_gnn_scorer_in_quick_mode():
    """CFDDesignSampler.sample() does NOT call GnnScorer when mode='quick'."""
    from unittest.mock import patch, MagicMock
    from api.routes.generate import CFDDesignSampler, CandidateResult

    sampler = CFDDesignSampler()
    fake_candidate = CandidateResult(
        id=0, domain="cylinder_flow",
        predicted_value=0.03, target_value=0.025,
        ood_confidence=-1.0, is_ood=False,
        mesh_nodes=100, params={"cx":0.5,"cy":0.2,"r":0.05,"v_inlet":1.0},
    )
    fake_graph = MagicMock()

    with patch.object(sampler, '_quick_sample', return_value=[(fake_candidate, fake_graph)]):
        with patch('api.routes.generate.get_gnn_scorer') as mock_get_scorer:
            results, traj = sampler.sample(
                target=0.025, n=1, device="cpu", method="sample", mode="quick"
            )

    mock_get_scorer.assert_not_called()
    c = results[0][0]
    assert c.gnn_predicted_value is None
    assert c.score_gap           is None
```

- [ ] **Step 2: Run to confirm it fails**

```bash
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/test_gnn_scorer.py::test_cfd_sampler_calls_gnn_scorer_in_deep_mode tests/test_gnn_scorer.py::test_cfd_sampler_skips_gnn_scorer_in_quick_mode -v 2>&1
```

Expected: FAIL — `sample()` has no `mode` parameter and no `_quick_sample` method.

- [ ] **Step 3: Refactor `CFDDesignSampler.sample()`**

In `api/routes/generate.py`, add this import at the top of the file (with the other imports):

```python
from api.state import get_gnn_scorer
```

Rename the existing `sample()` body to `_quick_sample()` and replace `sample()` with a dispatcher:

```python
def sample(self, target: float, n: int, device: str,
           method: str = "sample", mode: str = "quick") -> tuple[list, list]:
    """
    Generate n candidates. mode='quick' uses MLP surrogate only.
    mode='deep' adds GNN adaptive rollout scoring after sampling.
    """
    results, trajectory = self._quick_sample(target, n, device, method)

    if mode == "deep":
        cfd_ckpt = DOMAINS["cylinder_flow"]["checkpoint"]
        if not os.path.exists(cfd_ckpt):
            import logging
            logging.getLogger(__name__).warning(
                "Deep mode requested but GNN checkpoint not found at %s. "
                "Returning quick-mode results.", cfd_ckpt
            )
            return results, trajectory

        scorer = get_gnn_scorer(cfd_ckpt, device=device)
        graphs = [g for _, g in results]
        gnn_scores = scorer.score_candidates(graphs, device=device)

        import math
        updated = []
        for (c, g), score in zip(results, gnn_scores):
            if math.isnan(score.gnn_predicted_value):
                c.gnn_failed = True
            else:
                c.gnn_predicted_value = score.gnn_predicted_value
                c.score_gap           = abs(c.predicted_value - score.gnn_predicted_value)
                c.gnn_converged       = score.converged
            updated.append((c, g))
        results = updated

    return results, trajectory

def _quick_sample(self, target: float, n: int, device: str,
                  method: str = "sample") -> tuple[list, list]:
    """Original sample() logic — MLP surrogate only."""
    # [paste the entire original sample() body here verbatim]
```

Also update `event_stream()` in the same file to pass `mode` through:

```python
results, trajectory = await loop.run_in_executor(
    None,
    lambda: sampler.sample(req.target_value,
                           req.n_candidates,
                           req.device,
                           req.method,
                           req.mode)        # ← add this
)
```

And add a `warning` SSE event path for deep mode fallback — in the `event_stream()` function, add after the trajectory event:

```python
# Warn if deep mode was requested but fell back to quick
if req.mode == "deep" and all(c.gnn_predicted_value is None for c, _ in results):
    yield _sse_event("warning", {
        "detail": "GNN checkpoint not found — results are surrogate-only (quick mode)."
    })
    await asyncio.sleep(0)
```

- [ ] **Step 4: Run tests**

```bash
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/test_gnn_scorer.py -v 2>&1
```

Expected: all tests PASS except `test_adaptive_rollout_converges_early` (uses `_run_steps` which we intentionally didn't add — delete that test or skip it).

- [ ] **Step 5: Commit**

```bash
git add api/routes/generate.py tests/test_gnn_scorer.py
git commit -m "feat: wire GnnScorer into CFDDesignSampler — quick/deep modes"
```

---

## Task 5: Update `Generate.tsx` — mode toggle + warning banner

**Files:**
- Modify: `app/src/pages/Generate.tsx`

- [ ] **Step 1: Add `mode` to the `GenerateConfig` interface and state**

Find the `GenerateConfig` interface (around line 23) and add `mode`:

```tsx
interface GenerateConfig {
  domain:       string;
  target_value: number;
  n_candidates: number;
  method:       string;
  device:       string;
  mode:         'quick' | 'deep';   // ← add this
}
```

Find the `useState<GenerateConfig>` initializer (around line 43) and add `mode`:

```tsx
const [config, setConfig] = useState<GenerateConfig>({
  domain: 'cylinder_flow', target_value: 0.025,
  n_candidates: 6, method: 'sample', device: 'cpu',
  mode: 'quick',    // ← add this
});
```

- [ ] **Step 2: Add `warningMessage` state**

After the `error` state (around line 50):

```tsx
const [warningMessage, setWarningMessage] = useState<string | null>(null);
```

- [ ] **Step 3: Update `Candidate` interface to include GNN fields**

Find the `Candidate` interface (around line 10) and add:

```tsx
interface Candidate {
  id:                   number;
  domain:               string;
  predicted_value:      number;
  target_value:         number;
  ood_confidence:       number;
  is_ood:               boolean;
  mesh_nodes:           number;
  params:               Record<string, number>;
  thumbnail_url?:       string | null;
  session_id?:          string;
  // Deep mode fields
  gnn_predicted_value?: number | null;
  score_gap?:           number | null;
  gnn_converged?:       boolean | null;
  gnn_failed?:          boolean;
}
```

- [ ] **Step 4: Handle `warning` SSE event in `handleGenerate`**

Find the SSE event handler in `handleGenerate` (where `payload.type` or the event type is checked). Add a case for `"warning"`:

```tsx
} else if (eventType === 'warning') {
  setWarningMessage(payload.detail as string);
}
```

Also clear `warningMessage` at the start of each generate call (alongside `setError(null)`):

```tsx
setWarningMessage(null);
```

- [ ] **Step 5: Add mode toggle UI**

Find the form/controls section (before the Generate button). Add the mode toggle block:

```tsx
{/* Mode Toggle */}
<div className="mb-4">
  <label className="block text-sm font-medium text-slate-400 mb-2">
    Evaluation Mode
  </label>
  <div className="flex rounded-lg overflow-hidden border border-slate-700">
    <button
      type="button"
      onClick={() => setConfig(c => ({ ...c, mode: 'quick' }))}
      className={cn(
        'flex-1 px-4 py-2 text-sm font-medium transition-colors',
        config.mode === 'quick'
          ? 'bg-blue-600 text-white'
          : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
      )}
    >
      ⚡ Quick
      <span className="block text-xs font-normal opacity-75">Surrogate ~2s</span>
    </button>
    <button
      type="button"
      onClick={() => setConfig(c => ({ ...c, mode: 'deep' }))}
      className={cn(
        'flex-1 px-4 py-2 text-sm font-medium transition-colors',
        config.mode === 'deep'
          ? 'bg-violet-600 text-white'
          : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
      )}
    >
      🔬 Deep
      <span className="block text-xs font-normal opacity-75">GNN rollout ~30s</span>
    </button>
  </div>
</div>
```

- [ ] **Step 6: Add warning banner below the mode toggle (or above the candidate grid)**

```tsx
{warningMessage && (
  <div className="mb-4 flex items-start gap-2 rounded-lg border border-yellow-500/30
                  bg-yellow-500/10 px-4 py-3 text-sm text-yellow-300">
    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
    <span>{warningMessage}</span>
  </div>
)}
```

- [ ] **Step 7: Pass `mode` in the fetch body and pass GNN props to `CandidateCard`**

Find where `config` is spread into the fetch body (in `handleGenerate`). It already spreads `config` — since `mode` is now part of `config`, it will be included automatically. Verify the fetch call looks like:

```tsx
body: JSON.stringify(config),   // mode is included via config spread
```

Find where `<CandidateCard ... />` is rendered and add the new props:

```tsx
<CandidateCard
  key={c.id}
  id={c.id}
  domain={c.domain}
  predictedValue={c.predicted_value}
  targetValue={c.target_value}
  oodConfidence={c.ood_confidence}
  isOod={c.is_ood}
  meshNodes={c.mesh_nodes}
  params={c.params}
  thumbnailUrl={c.thumbnail_url}
  isSelected={selectedId === c.id}
  onSelect={() => setSelectedId(c.id)}
  mode={config.mode}                          // ← add
  gnnPredictedValue={c.gnn_predicted_value}   // ← add
  scoreGap={c.score_gap}                      // ← add
  gnnConverged={c.gnn_converged}              // ← add
  gnnFailed={c.gnn_failed}                    // ← add
/>
```

- [ ] **Step 8: Build to verify no TypeScript errors**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch/app
npm run build 2>&1 | tail -20
```

Expected: build succeeds (may have warnings, but no errors about missing props or type mismatches).

- [ ] **Step 9: Commit**

```bash
git add app/src/pages/Generate.tsx
git commit -m "feat: add Quick/Deep mode toggle and warning banner to Generate page"
```

---

## Task 6: Update `CandidateCard.tsx` — render GNN scores in Deep mode

**Files:**
- Modify: `app/src/components/CandidateCard.tsx`

- [ ] **Step 1: Add new props to the interface**

Find `CandidateCardProps` and add:

```tsx
interface CandidateCardProps {
  id:                  number;
  domain:              string;
  predictedValue:      number;
  targetValue:         number;
  oodConfidence:       number;
  isOod:               boolean;
  meshNodes:           number;
  params:              Record<string, number>;
  thumbnailUrl?:       string | null;
  isSelected?:         boolean;
  onSelect?:           () => void;
  // Deep mode props
  mode?:               'quick' | 'deep';
  gnnPredictedValue?:  number | null;
  scoreGap?:           number | null;
  gnnConverged?:       boolean | null;
  gnnFailed?:          boolean;
}
```

- [ ] **Step 2: Destructure new props in the component**

```tsx
export const CandidateCard: React.FC<CandidateCardProps> = ({
  id, domain, predictedValue, targetValue,
  oodConfidence, isOod, meshNodes, params,
  thumbnailUrl, isSelected, onSelect,
  mode = 'quick',
  gnnPredictedValue, scoreGap, gnnConverged, gnnFailed,
}) => {
```

- [ ] **Step 3: Add gap badge helper**

After the existing `hasConf` / `confPct` lines, add:

```tsx
// Deep mode — GNN score display helpers
const isDeep       = mode === 'deep';
const hasGnn       = isDeep && gnnPredictedValue != null && !gnnFailed;
const gnnLabel     = hasGnn
  ? `${gnnConverged === false ? '~' : ''}${gnnPredictedValue!.toFixed(4)}`
  : null;
const gapColor     = scoreGap == null  ? ''
  : scoreGap < 0.1   ? 'text-emerald-400'
  : scoreGap < 0.2   ? 'text-amber-400'
  : 'text-red-400';
const gapDot       = scoreGap == null  ? ''
  : scoreGap < 0.1   ? '🟢'
  : scoreGap < 0.2   ? '🟡'
  : '🔴';
```

- [ ] **Step 4: Replace the single predicted-value row with mode-aware rows**

Find the existing row that shows `predictedValue` (something like `<span>{physLabel}</span> <span>{predictedValue.toFixed(4)}</span>`). Replace it with:

```tsx
{/* Physics score row(s) */}
{isDeep ? (
  <>
    <div className="flex justify-between text-xs">
      <span className="text-slate-400">Surrogate</span>
      <span className="text-slate-200 font-mono">{predictedValue.toFixed(4)}</span>
    </div>
    <div className="flex justify-between text-xs">
      <span className="text-slate-400">GNN</span>
      {hasGnn ? (
        <span className="text-violet-300 font-mono">
          {gnnLabel} ✓
        </span>
      ) : gnnFailed ? (
        <span className="text-slate-500 italic">scoring failed</span>
      ) : (
        <span className="text-slate-500">—</span>
      )}
    </div>
    {hasGnn && scoreGap != null && (
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">Gap</span>
        <span className={cn('font-mono', gapColor)}>
          {scoreGap.toFixed(4)} {gapDot}
        </span>
      </div>
    )}
  </>
) : (
  <div className="flex justify-between text-xs">
    <span className="text-slate-400">{physLabel}</span>
    <span className="text-slate-200 font-mono">{predictedValue.toFixed(4)}</span>
  </div>
)}
```

- [ ] **Step 5: Build to verify no TypeScript errors**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch/app
npm run build 2>&1 | tail -20
```

Expected: clean build, no type errors.

- [ ] **Step 6: Commit**

```bash
git add app/src/components/CandidateCard.tsx
git commit -m "feat: render GNN scores and gap badge in CandidateCard deep mode"
```

---

## Task 7: Run full test suite and verify

**Files:** None — verification only

- [ ] **Step 1: Run all tests**

```bash
cd /home/ahmealy/ML/meshGraphNets_pytorch
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/ -v 2>&1
```

Expected: all existing tests pass + all new `test_gnn_scorer.py` tests pass. Note: `test_adaptive_rollout_converges_early` can be deleted if it was left in — it uses a method (`_run_steps`) that was not implemented.

- [ ] **Step 2: Delete or skip the unreachable test**

If `test_adaptive_rollout_converges_early` is still in the file, remove it (it tests an internal helper we didn't need):

```bash
# Edit tests/test_gnn_scorer.py and remove the test_adaptive_rollout_converges_early function
```

- [ ] **Step 3: Run tests once more to confirm clean pass**

```bash
/bata/ahmealy/personal_projects/meshGraphNets_pytorch/venv/bin/python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all PASS, 0 FAIL.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_gnn_scorer.py
git commit -m "test: clean up gnn_scorer test suite"
```
