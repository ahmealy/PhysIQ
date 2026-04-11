# Design: GNN Scorer + Quick/Deep Generate Modes

**Date:** 2026-04-11  
**Status:** Approved  
**Scope:** `extensions/generative/`, `api/routes/generate.py`, `api/state.py`, `app/src/pages/Generate.tsx`, `app/src/components/CandidateCard.tsx`

---

## Problem

The Generate subsystem currently uses only a lightweight MLP drag surrogate to score candidates. This surrogate is fast (~1ms) but coarse — it maps 4 scalar parameters `(cx, cy, r, v_inlet)` to a single drag proxy value and accumulates approximation error. Users have no way to get physics-accurate scoring without running a full external CFD solver.

MeshGraphNets (the GNN at the core of this project) is itself a surrogate — but a much higher-fidelity one. It takes the full mesh graph as input and predicts the velocity/pressure field on every node. Extracting drag from a short GNN rollout gives a score much closer to real CFD than the MLP can provide.

---

## Goal

Add two evaluation modes to the Generate page:

- **Quick mode** — existing behavior, MLP surrogate only, ~2s total
- **Deep mode** — MLP surrogate for gradient descent (fast optimization), then GNN adaptive rollout to re-score final candidates with physics-accurate drag values, ~30s total

Show both scores in Deep mode so users can see surrogate vs GNN agreement (disagreement = uncertainty signal).

---

## Architecture

### Data Flow

```
GenerateRequest(mode="quick"|"deep")
        │
        ▼
CFDDesignSampler.sample()
        │
        ├─ CVAE generates N candidates
        │
        ├─ MLP surrogate scores all N       ← always runs (both modes)
        │   └─ gradient descent in latent space (surrogate only, always)
        │
        ├─ [mode="deep" only]
        │   └─ GnnScorer.score_candidates(candidates, graphs, device)
        │         ├─ adaptive rollout per candidate
        │         │   └─ runs in 20-step chunks, stops when drag Δ < 1e-3
        │         │   └─ hard cap: 200 steps
        │         └─ returns gnn_predicted_value per candidate
        │
        └─ CandidateResult:
              predicted_value      (surrogate, always present)
              gnn_predicted_value  (float | None — None in quick mode)
              score_gap            (|surrogate - gnn| | None — None in quick mode)
              gnn_converged        (bool | None — False if 200-step cap hit)
```

### Why gradient descent always uses the MLP surrogate

Running 150 gradient descent iterations with GNN rollouts would require 150 × ~5s = ~12 minutes per candidate. The MLP surrogate is the right tool for optimization — it's differentiable and instant. The GNN is used only as a final verifier/scorer on the finished candidates.

---

## New File: `extensions/generative/gnn_scorer.py`

```python
class GnnScorer:
    """
    Wraps the MeshGraphNets simulator to score CFD design candidates
    via adaptive rollout. Used in Deep mode only.

    Lazy-loaded and cached in API state.
    """
    def __init__(self, checkpoint_path: str, device: str)

    def score_candidates(
        self,
        graphs: list[Data],
        device: str,
    ) -> list[GnnScore]
    # Returns one GnnScore per graph:
    #   gnn_predicted_value: float
    #   converged: bool  (False if 200-step cap hit)

    def _adaptive_rollout(self, graph: Data, device: str) -> tuple[float, bool]
    # Runs rollout in 20-step chunks.
    # Stops when |drag_t - drag_{t-20}| < 1e-3 or steps >= 200.
    # Returns (drag_value, converged).
```

Drag is extracted as mean x-velocity magnitude over OUTFLOW nodes at the final rollout step — same convention as `drag_surrogate.py`.

---

## Changes to Existing Files

### `api/routes/generate.py`

- `GenerateRequest` gets `mode: Literal["quick", "deep"] = "quick"`
- `CandidateResult` dataclass gets four new optional fields:
  - `gnn_predicted_value: float | None = None`
  - `score_gap: float | None = None`
  - `gnn_converged: bool | None = None`
  - `gnn_failed: bool = False`
- `CFDDesignSampler.sample()` accepts `mode` and calls `GnnScorer.score_candidates()` after MLP scoring when `mode == "deep"`
- If GNN scoring fails on one candidate: that candidate gets `gnn_predicted_value=None` and a `gnn_failed=True` flag; other candidates unaffected
- If checkpoint missing in deep mode: emit `warning` SSE event and fall back to quick mode automatically
- `GnnScorer` instantiated once and cached (lazy) — not recreated per request

### `api/state.py`

- Add `gnn_scorer: GnnScorer | None = None` to cached state
- Add `get_gnn_scorer(device) -> GnnScorer` helper following same pattern as `get_model()`

### `app/src/pages/Generate.tsx`

- Add mode toggle UI (see UI section below)
- Pass `mode` in `GenerateRequest` body
- Pass `gnn_predicted_value`, `score_gap`, `gnn_converged` as props to `CandidateCard`
- Show yellow warning banner if `warning` SSE event received (e.g. GNN fallback)

### `app/src/components/CandidateCard.tsx`

- Accept new optional props: `gnnPredictedValue`, `scoreGap`, `gnnConverged`
- **Quick mode** — render same as today (no change)
- **Deep mode** — render additional rows:
  - `Surrogate: {predicted_value}`
  - `GNN: {gnn_predicted_value} ✓` (with `~` prefix if `!gnnConverged`)
  - `Gap: {score_gap}` with color badge:
    - 🟢 green if gap < 0.1
    - 🟡 yellow if 0.1 ≤ gap < 0.2
    - 🔴 red if gap ≥ 0.2

---

## UI Design

### Mode Toggle (top of Generate page, above Generate button)

```
┌─────────────────────────────────────────┐
│  Evaluation Mode                        │
│  ⚡ Quick  |  🔬 Deep                   │
│  Surrogate ~2s  |  GNN rollout ~30s     │
└─────────────────────────────────────────┘
```

### CandidateCard — Deep Mode

```
┌──────────────────────────────┐
│  [mesh heatmap thumbnail]    │
│                              │
│  Surrogate:  0.42            │
│  GNN:        0.38  ✓        │
│  Gap:        0.04  🟢        │
│  Conf:       87%             │
└──────────────────────────────┘
```

If GNN did not converge: `GNN: ~0.41 ✓` (tilde prefix).  
If GNN scoring failed: `GNN: scoring failed` in muted text.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| GNN rollout throws on one candidate | That candidate: `gnn_predicted_value=null`, `gnn_failed=true`. Others unaffected. |
| `best_model.pth` missing in deep mode | Emit `warning` SSE event, fall back to quick mode automatically. UI shows yellow banner. |
| Convergence cap hit (200 steps) | Return drag at step 200, set `gnn_converged=false`. UI shows `~` prefix on GNN score. |
| Quick mode selected | `gnn_predicted_value` and `score_gap` are `null`. `CandidateCard` does not render GNN rows. |

---

## What is NOT in scope

- Gradient descent using GNN (too slow — MLP always used for optimization)
- Cloth domain Deep mode (GNN scoring only for CFD / cylinder_flow)
- Ensemble blending (scores are shown side by side, not averaged)
- Training a new model — uses existing `best_model.pth`
