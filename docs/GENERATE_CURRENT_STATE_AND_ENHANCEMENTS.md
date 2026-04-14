# Generate Pipeline — Current State & Enhancement Proposals

> **Purpose:** Precise technical snapshot of every component in the generate pipeline as it
> exists in code today, followed by concrete enhancement proposals. Written for external
> brainstorming — no fluff, exact numbers and class names throughout.
>
> **Last major rearchitecting:** 2026-04-14 — `ParamSpaceOOD` (param-space OOD replaces
> embedding-based OOD in generate pipeline), `RealMeshLookup` for thumbnails (real mesh
> replaces synthetic `CFDMeshBuilder` in sample path), confidence index rebuilt + staleness
> warning added. Prior: CFD gradient coupling via `RealMeshLookup`, cloth K=5 BPTT fix,
> `free_bits=0.05`, LHS sampling, `extract_true_drag`, mode removal.

---

## Table of Contents

1. [End-to-End Data Flow](#1-end-to-end-data-flow)
2. [Component Inventory](#2-component-inventory)
   - [2.1 Data Extraction](#21-data-extraction)
   - [2.2 Surrogate Models](#22-surrogate-models)
   - [2.3 CVAE Models](#23-cvae-models)
   - [2.4 Mesh Generators & RealMeshLookup](#24-mesh-generators--realmeshtlookup)
   - [2.5 Cloth Inverse Design](#25-cloth-inverse-design)
   - [2.6 Backend API](#26-backend-api)
   - [2.7 Frontend](#27-frontend)
3. [What the Two Generation Methods Actually Do](#3-what-the-two-generation-methods-actually-do)
4. [Known Limitations in Current Code](#4-known-limitations-in-current-code)
5. [Enhancement Proposals](#5-enhancement-proposals)

---

## 1. End-to-End Data Flow

### CFD (cylinder_flow) — Sample method

```
User: domain=cylinder_flow, target_drag=0.025, method=sample
                │
                ▼
  CVAETrainer.load("checkpoints/cfd_cvae.pth")
  DragSurrogateTrainer.load("checkpoints/drag_surrogate.pth")
                │
  z ~ LHS(N(0, I))  shape [n_candidates, 16]   ← Latin Hypercube Sampling
                │
  CVAEDecoder(z, target_drag) → params_norm [n, 4]
  clip([0,1]) → CVAEScaler.denorm_params → params_phys [n, 4]
                │
  For each candidate:
    DragSurrogate(params_phys[i]) → predicted_drag  (MLP, ~1ms)
                │
  Sort by |predicted_drag - target|, keep top-n
                │
  RealMeshLookup.find_nearest(cx, cy, r) → traj_idx
  RealMeshLookup.load_mesh_for_trajectory(traj_idx, v_inlet, dataset)
    → real OpenFOAM PyG Data (~1883 nodes, non-uniform, finer near cylinder)
                │
  ThumbnailRenderer.render_cfd(graph) → PNG bytes
                │
  OOD check: ParamSpaceOOD.score(cx, cy, r, v_inlet)
    → 4-D param KDTree on data/design_params.npy
    → ood_confidence [0,1] or -1.0 if design_params.npy missing
                │
  SSE stream: event:candidate  data:{id, predicted_value, ood_confidence, ...}
                │
  event:done  data:{best_id}
```

### CFD — Gradient method (new: CFD–GNN coupling via RealMeshLookup)

```
User: method=gradient, domain=cylinder_flow
                │
  n_restarts=3, n_iters=150, lr=0.05
  Per restart:
    z = randn(16), requires_grad=True
    Adam([z], lr=0.05)
    150 steps:
      params_norm = CFDDecoder(z, target_norm)       [4]
      params_phys = CVAEScaler.denorm(params_norm)   [4]  (cx, cy, r, v_inlet)

      ── RealMeshLookup path (primary) ────────────────────────────────────────
      RealMeshLookup.find(cx, cy, r)
        → normalised L2 distance on design_params.npy [N_traj, 4]
        → nearest training trajectory index k
        → load real OpenFOAM mesh from traj_k.npz     (~1876 nodes)
        → inject v_inlet as differentiable tensor into INFLOW node features

      GNN rollout:
        steps 0 … K-2 (no_grad warm-up):
          next_vel = Simulator(graph)  [N, 2]
          pin boundary nodes
        step K-1 (with grad):
          next_vel = Simulator(graph)  [N, 2]   ← gradient flows here

      drag = mean(|vx|) over OUTFLOW nodes  [scalar]
      loss = (drag - target)²
      loss.backward()
        gradient path: z → CFDDecoder → v_inlet (tensor) → GNN → drag → loss
        NOTE: cx, cy, r have zero gradient through the discrete lookup step

      ── DragSurrogate fallback (if design_params.npy or simulator ckpt absent) ──
      drag_n  = SurrogateScaler.norm(params_phys)
      drag    = DragSurrogate(drag_n)               [1]
      loss    = (drag - target_norm)²
      ─────────────────────────────────────────────────────────────────────────

      loss.backward(); step
  best_z = restart with min |loss[-1]|
  n_candidates = best_z + randn * noise_scale=0.25  [n, 16]
  → decode each → score → sort → stream
```

### Cloth (flag_simple) — Sample method

```
User: domain=flag_simple, target_stress=1.0, method=sample
                │
  ClothCVAETrainer.load("checkpoints/flag-simple_cvae.pth")
  PosePCA.load("data_flag/train/cloth_pca.pkl")
                │
  z ~ LHS(N(0, I)) [n, 16]             ← Latin Hypercube Sampling
  ClothDecoder(z, target_stress) → pose_pca_norm [n, 16]
  CVAEScaler.denorm → pose_pca_phys [n, 16]
  PosePCA.inverse_transform → world_pos [n, 1579, 3]
                │
  For each:
    StressSurrogate(pose_pca_phys[i]) → predicted_stress
    ClothMeshBuilder.build(world_pos[i]) → PyG Data (1579 nodes, 3028 faces)
    ThumbnailRenderer.render_cloth(graph) → PNG
                │
  SSE stream candidates
```

### Cloth — Gradient method (new: K=5 BPTT, velocity gradient fixed)

```
ClothInverseDesigner.optimise(target_stress, n_iters=80, n_restarts=2, lr=0.05)
  Per restart:
    z = randn(16), requires_grad=True
    Adam([z], lr=0.05)
    80 steps:
      pose_pca_norm = ClothDecoder(z, target_norm)         [1, 16]
      pose_pca_phys = denorm(pose_pca_norm)                 [1, 16]
      world_pos = TorchPCAInverseTransform(pose_pca_phys)  [1579, 3]  ← grad flows

      _rollout(world_pos, K=ROLLOUT_STEPS):         ← ClothInverseDesigner.ROLLOUT_STEPS = 5
        pos = world_pos
        for step in range(K):
          graph = _build_data_from_pos(pos)
            x = cat([pos, node_type])                [N, 4]
            prev_x = pos  (NO .detach() — grad flows through velocity term)
          pos = FlagSimulator(graph)                 [1579, 3]  ← grad flows each step
        return pos   (final position after K steps)

      stress = mean(‖final_pos[NORMAL] - mesh_rest[NORMAL]‖)
      loss = (stress - target)²
      loss.backward() → ∂loss/∂z via full K=5 BPTT

  noise_scale=0.20 for diversity
```

---

## 2. Component Inventory

### 2.1 Data Extraction

#### `shape_extractor.py` — CFD

| Item | Value |
|---|---|
| **Output** | `design_params.npy` shape `[N_traj, 4]` |
| **Columns** | `[cx, cy, r, v_inlet]` |
| **Circle fit** | Kåsa algebraic least-squares (`np.linalg.lstsq`) |
| **Disambiguation** | WALL_BOUNDARY nodes within `eps=1e-3` of domain edge = wall, not cylinder |
| **v_inlet** | Mean x-velocity of INFLOW nodes at `t=0` |

#### `cloth_extractor.py` — Cloth

| Item | Value |
|---|---|
| **Output** | `cloth_pose_pca.npy [N, 16]`, `cloth_stress.npy [N]`, `cloth_pca.npz` |
| **PCA solver** | `np.linalg.svd(full_matrices=False)` — no sklearn |
| **n_components** | 16 |
| **Cloth node count N** | 1579 (fixed) |
| **Flattened space** | 1579 × 3 = 4737 dims |
| **Stress computation** | `mean(‖world_pos[-1][NORMAL] − rest_3d[NORMAL]‖₂)` — final timestep only |
| **Stress node filter** | NORMAL nodes only (excludes HANDLE) |

---

### 2.2 Surrogate Models

#### `DragSurrogate` (CFD)

```
Input:  [B, 4]  (cx, cy, r, v_inlet)  — min-max normalised
        Linear(4→64) → ReLU
        Linear(64→64) → ReLU
        Linear(64→1)
Output: [B]  drag_proxy (normalised)

Training: Adam, lr=1e-3, CosineAnnealingLR(T_max=200), MSE loss
Scaler:   MinMaxScaler — x_min/x_max [4], y_min/y_max scalar
Checkpoint keys: state_dict, scaler_dict, config_dict
```

**Drag label source (updated):**
- **Primary:** `extract_true_drag(simulator, dataset, traj_idx, device, steady_state_frac=0.3)`
  in `drag_surrogate.py` — runs full GNN rollout, finds OUTFLOW nodes, averages `|vx|` over
  the last 30% of timesteps (steady-state window).
- **Fallback:** Analytical formula `r × v² / (1 − 2r/H)`, H=0.41 m — used when simulator
  checkpoint is absent.
- `DragSurrogateTrainer.fit(X, y=None, true_drag_labels=None)` accepts a pre-computed dict
  `{traj_idx: drag_value}` to override the analytical formula.
- `train_cvae.py` CFDStrategy attempts to load simulator checkpoint and extract true labels
  before surrogate training.

#### `StressSurrogate` (Cloth)

```
Input:  [B, 16]  pose_pca — min-max normalised
        Linear(16→64) → ReLU
        Linear(64→64) → ReLU
        Linear(64→1)
Output: [B]  stress_proxy

Training: Adam, lr=1e-3, NO scheduler (unlike DragSurrogate), 200 epochs
Batch size: hardcoded 64 inside fit() loop
```

---

### 2.3 CVAE Models

#### `CFDCVAE` — Encoder

```
Input:  [B, 5]  = cat(params[B,4], drag_actual[B,1])
        Linear(5→64) → ReLU → Linear(64→64) → ReLU
        → fc_mu(64→16)       → μ [B, 16]
        → fc_logvar(64→16)   → log σ² [B, 16]
```

#### `CFDCVAE` — Decoder

```
Input:  [B, 17]  = cat(z[B,16], target_drag[B,1])
        Linear(17→64) → ReLU → Linear(64→64) → ReLU → Linear(64→4)
Output: [B, 4]  params_normalised  (unbounded — no final activation)
```

#### `CFDCVAE` — Training config

| Param | Value |
|---|---|
| latent_dim | 16 |
| hidden_size | 64 |
| alpha (recon) | 1.0 |
| beta (KL) | 1e-3 |
| lam (physics) | 0.5 |
| lr | 1e-3 |
| epochs | 300 |
| batch_size | 64 |
| val_split | 0.1 |
| free_bits | **0.05** (prevents posterior collapse — was 0.0) |
| scheduler | CosineAnnealingLR(T_max=300) |

#### `CFDCVAE.sample()` — LHS sampling

```python
# Primary path (scipy available):
self._lhs_sampler = qmc.LatinHypercube(d=latent_dim)  # cached as instance attr
z_unit = self._lhs_sampler.random(n=n_candidates)      # [n, 16] uniform [0,1]
z = torch.tensor(ndtri(z_unit), dtype=torch.float32)   # map to N(0,I)

# Fallback (scipy absent):
z = torch.randn(n_candidates, latent_dim)
```

#### `ClothCVAE` — Differences from CFDCVAE

| Param | CFD | Cloth |
|---|---|---|
| Encoder input dim | 5 (params[4]+drag[1]) | 17 (pose_pca[16]+stress[1]) |
| Decoder output dim | 4 (cx,cy,r,v_inlet) | 16 (pose_pca components) |
| hidden_size | **64** | **128** |
| free_bits | **0.05** | **0.05** |
| LHS sampler | `self._lhs_sampler` cached | `self._lhs_sampler` cached |
| Post-decode step | None | PCA⁻¹ → world_pos [N,3] |
| Physics loss guard | Always applied | Skipped if `lam <= 0.0` |
| Stress surrogate scheduler | CosineAnnealingLR | **No scheduler** |

#### Loss formula (both CVAEs)

```
L = 1.0 × MSE(recon, params_gt)                     [reconstruction]
  + 1e-3 × KL(q(z|θ,d) ‖ N(0,I))                   [regularisation]
  + 0.5  × MSE(surrogate(denorm(recon)), target)     [physics consistency]

KL with free_bits=0.05:
  kl_per_dim = -0.5 × (1 + logvar - μ² - exp(logvar))   [B, 16]
  kl_clamped = max(kl_per_dim, 0.05)                     [B, 16]  ← floor per dim
  KL = mean(kl_clamped)
  → forces encoder to use all 16 latent dims (no collapsed dims)

Physics gradient flow:
  recon_norm → denorm (tensor ops) → re-norm to surrogate scale
  → surrogate.forward() [NOT no_grad] → denorm → re-norm → MSE
  Surrogate weights FROZEN (train=False) but gradients PASS THROUGH
  → trains the DECODER through the surrogate
```

---

### 2.4 Mesh Generators, RealMeshLookup & Confidence

#### `CFDMeshBuilder` *(deprecated in sample path)*

| Item | Value |
|---|---|
| Domain dimensions | 1.6 × 0.41 m |
| Cylinder boundary pts | n_cyl=24, uniformly spaced on circle |
| Background grid | grid_nx=60 × grid_ny=16 = 960 pts |
| Interior removal | pts where `dist_to_cylinder < r` removed |
| Triangulation | `scipy.spatial.Delaunay` (NOT differentiable) |
| Post-filter | Triangles with centroid inside cylinder removed |
| Node features | `x [N, 3]` = `[node_type, vx=v_inlet or 0, vy=0]` |
| Typical output | ~973 nodes, ~5492 edges |
| Transform | `FaceToEdge → Cartesian(norm=False) → Distance(norm=False)` |
| **Role** | ~~Used for thumbnail rendering and sample-method scoring~~ **DEPRECATED in sample path** (2026-04-14). Thumbnails now use `RealMeshLookup`; OOD now uses `ParamSpaceOOD`. Retained for: `MeshGeneratorFactory` registry, `params_to_graph()` helper, and CLI. |

Node type assignments:
```
dist_cyl < r + 2e-3      → WALL_BOUNDARY
|y - 0| < 2e-3           → WALL_BOUNDARY
|y - 0.41| < 2e-3        → WALL_BOUNDARY
|x - 0| < 2e-3           → INFLOW  (vx = v_inlet)
|x - 1.6| < 2e-3         → OUTFLOW
else                     → NORMAL
```

#### `RealMeshLookup` (gradient method + thumbnails)

**File:** `extensions/generative/mesh_generator.py`

```
Purpose: find the nearest real OpenFOAM training trajectory for a given
         (cx, cy, r) design, load its mesh, optionally inject v_inlet as a
         differentiable tensor.

Used in TWO places:
  1. CFDDesignSampler.sample()  — thumbnail rendering (real mesh visual quality)
  2. CFDDesignSampler._gradient_sample() — GNN gradient coupling (differentiable v_inlet)

Construction:
  design_params = np.load("design_params.npy")  [N_traj, 4]
  Normalisation: per-column min-max on cols 0:3 (cx, cy, r)

Lookup: find_nearest(cx, cy, r)
  query = normalise([cx, cy, r])             [3]
  dists = ‖design_params_norm[:, :3] - query‖₂  [N_traj]
  k     = argmin(dists)
  returns: traj index k (int)

Mesh loading: load_mesh_for_trajectory(traj_idx, v_inlet_tensor, dataset, device)
  npz = load traj_k.npz  (~1883 nodes, real OpenFOAM topology, non-uniform
                           spacing — denser near cylinder)
  For thumbnails: pass dummy v_inlet (no grad needed)
  For gradient:   pass differentiable v_inlet_tensor from decoder
  graph.x[INFLOW, 1] = v_inlet_tensor  (spliced via _inject_scalar_differentiable)
  returns: PyG Data with edge_index/edge_attr (FaceToEdge+Cartesian+Distance applied)

Differentiability (gradient mode):
  v_inlet_tensor IS differentiable → ∂drag/∂v_inlet → ∂loss/∂z (via CFDDecoder)
  k (traj index) is a discrete argmin → cx, cy, r have ZERO gradient through lookup
```

#### `ClothMeshBuilder`

| Item | Value |
|---|---|
| Reference topology | Loaded from `traj_00000.npz` once at init |
| Node count | N=1579 (fixed, all trajectories) |
| Face count | F=3028 (fixed) |
| Node features | `x [N, 4]` = `[wx, wy, wz, node_type]` |
| What changes | Only `world_pos [N,3]` — topology never changes |
| `prev_x` | `world_pos` (no `.detach()` — differentiable in gradient mode) |

#### `ParamSpaceOOD` *(new — generate pipeline OOD)*

**File:** `extensions/confidence/ood_detector.py`

```
Purpose: determine whether generated design params are within the training
         distribution, using a 4-D KDTree over (cx, cy, r, v_inlet).

Why param-space (not mesh-space):
  - If we used RealMeshLookup meshes for OOD, we'd always query training meshes
    by definition → always in-distribution.
  - If we used synthetic CFDMeshBuilder meshes, embedding distance from training
    GNN embeddings → always OOD (domain shift).
  - Correct question: "Are these *design params* in the training envelope?"

Construction: ParamSpaceOOD(dataset_path='data')
  params = np.load("data/design_params.npy")  [N, 4]  (cx, cy, r, v_inlet)
  Per-column min/max normalisation → norm_params [N, 4] ∈ [0, 1]^4
  scipy.spatial.KDTree(norm_params)
  train_diameter = 95th percentile of leave-one-out NN distances
    (same formula as NearestNeighborIndex.build() in confidence/index.py)

Scoring: score(cx, cy, r, v_inlet) → OODResult
  q_norm = (query - p_min) / scale             [4] ∈ [0,1]^4
  d_min = KDTree.query(q_norm, k=1)
  confidence = clip(1 - d_min / (train_diameter + 1e-12), 0, 1)
  is_ood = confidence < threshold (default 0.3)

OODResult.embedding field repurposed to store q_norm [4] (normalised params).
Returns confidence=-1.0 if design_params.npy missing (graceful degradation).
```

#### Two Distinct Confidence Systems

| Context | Class | Input | Question answered |
|---------|-------|-------|-------------------|
| **Generate** (sample path) | `ParamSpaceOOD` | 4-D `(cx, cy, r, v_inlet)` param KDTree | Are these design params within the training envelope? |
| **Predict** (rollout route) | `NearestNeighborIndex` | 128-D GNN embedding KDTree | Is this test trajectory similar to training trajectories? |

The predict path (`api/routes/rollout.py`) uses `NearestNeighborIndex` from
`runs/embedding_index_cylinderflow.pkl` — conceptually correct because test trajectories
are genuinely unseen.  A **staleness warning** is logged if the checkpoint is newer than
the index by >60 s (guards against stale index after checkpoint retraining).

---

### 2.5 Cloth Inverse Design

**File:** `extensions/generative/inverse_design.py`

#### Full differentiable chain (updated — K=5 BPTT, velocity grad fixed)

```
z [16]  requires_grad=True
  ↓  ClothDecoder.forward(z, target_stress_norm)
pose_pca_norm [1, 16]
  ↓  denorm via CVAEScaler (tensor arithmetic — grad flows)
pose_pca_phys [1, 16]
  ↓  TorchPCAInverseTransform (registered buffers — grad flows)
     flat = z_pca @ components + mean  → [4737]
world_pos [1579, 3]
  ↓  _rollout(world_pos, K=5):
     for step in range(5):
       graph = _build_data_from_pos(pos):
         x = cat([pos, node_type.unsqueeze(-1)])  [N, 4]  ← pos in graph
         prev_x = pos                             ← NO detach — grad flows through velocity
       pos = FlagSimulator(graph)  eval mode      [1579, 3]  ← grad flows each step
     return pos                                   ← final position after K=5 steps
  ↓  StressObjective:
     disp = ‖final_pos[NORMAL] - mesh_rest[NORMAL]‖  [M]
     stress = disp.mean()
     loss = MSE(stress, target_tensor)
scalar loss → loss.backward() → ∂loss/∂z  (via full K=5 BPTT)
```

| Param | Value |
|---|---|
| ROLLOUT_STEPS (class const) | **5** (K=5 BPTT) |
| n_restarts | 2 |
| n_iters per restart | 80 |
| lr | 0.05 |
| Optimizer | Adam |
| Objective | `MSE(mean_displacement, target)` — K=5 full rollout |
| Result | `OptimisationResult(best_z, best_stress, trajectory, n_iters)` |

---

### 2.6 Backend API

**File:** `api/routes/generate.py`

#### Endpoints

```
POST /api/generate
  Body: GenerateRequest {domain, target_value, n_candidates[1-50], method, device}
  Response: SSE stream

  Events emitted:
    "trajectory"  {values: [float]}          gradient mode — loss per iter
    "candidate"   {id, domain, predicted_value, target_value, ood_confidence,
                   is_ood, mesh_nodes, params, thumbnail_url, session_id}
    "warning"     {detail: str}              missing GNN or surrogate checkpoint
    "done"        {best_id, session_id}
    "error"       {detail: str}

GET  /api/generate/thumbnail/{session_id}/{candidate_id}
  Returns: image/png  (400×300 px, matplotlib)
  Cache: in-memory dict, max 100 sessions (LRU insertion-order)

POST /api/generate/rollout/{session_id}/{candidate_id}?n_steps=50&device=cpu
  Returns: {pkl_filename: "generate_{session[:8]}_{candidate_id}.pkl"}
  Effect: runs full rollout, saves pkl, navigatable via /visualize?file=...
```

**Removed from GenerateRequest:** `mode` field (deep/quick toggle gone).  
**Removed from CandidateResult:** `gnn_predicted_value`, `score_gap`, `gnn_converged`, `gnn_failed`.  
**Removed events:** `"gnn_score"` event no longer emitted.

#### Checkpoint paths (hardcoded)

```python
CFDDesignSampler.CFD_CVAE_PATH   = "checkpoints/cfd_cvae.pth"
CFDDesignSampler.SURROGATE_PATH  = "checkpoints/drag_surrogate.pth"
CFDDesignSampler.SIMULATOR_PATH  = "checkpoints/simulator_cylinderflow.pth"  # for gradient
CFDDesignSampler.DESIGN_PARAMS   = "data_cylinderflow/train/design_params.npy"
ClothDesignSampler.CVAE_PATH     = "checkpoints/flag-simple_cvae.pth"
ClothDesignSampler.PCA_PATH      = "data_flag/train/cloth_pca.pkl"
ClothDesignSampler.STRESS_PATH   = "data_flag/train/cloth_stress.npy"
ClothDesignSampler.REF_TRAJ      = "data_flag/train/traj_00000.npz"
ParamSpaceOOD data:   "data/design_params.npy"              ← generate OOD (param-space)
Embedding index:      "runs/embedding_index_cylinderflow.pkl" ← predict confidence (128-D)
Embedding index leg:  "runs/embedding_index.pkl"            ← legacy fallback
```

#### Gradient descent hyperparams

| | CFD | Cloth |
|---|---|---|
| n_restarts | 3 | 2 |
| n_iters | 150 | 80 |
| lr | 0.05 | 0.05 |
| noise_scale (diversity) | 0.25 | 0.20 |

---

### 2.7 Frontend

**Files:** `app/src/pages/Generate.tsx`, `app/src/components/CandidateCard.tsx`

#### Config state (persisted to localStorage)

```typescript
{ domain: 'cylinder_flow', target_value: 0.025, n_candidates: 6,
  method: 'sample', device: 'cpu' }
```

Note: `mode` field removed from config state.

#### Domain ranges

```
cylinder_flow: target ∈ [0.001, 0.15]  step=0.001  default=0.025
flag_simple:   target ∈ [0.2,   2.6]   step=0.05   default=1.0
```

#### CandidateCard display logic

```
Both methods:  shows single physics score — "Drag proxy" / "Stress proxy"
               + surrogate prediction value
OOD badge:     N/A / ⚠ OOD / ✓ X% conf
Analyze btn:   only shown when sessionId present → navigates to /visualize
```

No GNN score row, no score gap row, no deep/quick toggle UI.

---

## 3. What the Two Generation Methods Actually Do

### Sample method — what happens step by step

1. Load CVAE + surrogate from disk (each generate call reloads from file)
2. Draw `n_candidates` z vectors via LHS mapping to N(0,I) — better latent coverage than randn
3. Decode all to params via CVAE decoder, score all with DragSurrogate (~1ms each)
4. Sort by `|predicted - target|`, keep top n_candidates
5. For each: load nearest real OpenFOAM mesh via `RealMeshLookup` (thumbnail), render
   thumbnail, OOD check via `ParamSpaceOOD.score(cx, cy, r, v_inlet)` (4-D param KDTree)
6. Stream via SSE — candidates appear in score order

### Gradient method — what each domain does

**CFD (cylinder_flow):**

Gradient flows through two possible chains depending on checkpoint availability:

*Primary chain (RealMeshLookup active):*
```
z → CFDDecoder → (cx, cy, r, v_inlet)
                              ↓
                  RealMeshLookup.find(cx, cy, r)   ← discrete, no grad for geometry
                              ↓
                  load real OpenFOAM mesh traj_k
                  inject v_inlet as differentiable tensor
                              ↓
                  GNN warm-up (K-1 steps, no_grad)
                  GNN final step (with grad)
                              ↓
                  drag = mean(|vx|) at OUTFLOW nodes
                              ↓
                  loss = (drag - target)²
```
The gradient path: `z → CFDDecoder.v_inlet_output → v_inlet_tensor → GNN → drag → loss`.
Note: cx, cy, r are optimised indirectly — only v_inlet carries a gradient.

*Fallback chain (DragSurrogate):*
```
z → CFDDecoder → params_phys → DragSurrogate → loss
```
Used when `design_params.npy` or simulator checkpoint is absent.

**Cloth (flag_simple):**

Full differentiable chain through physics simulator with K=5 BPTT:
```
z → ClothDecoder → PCA⁻¹ → world_pos
  → _rollout(K=5): [_build_data_from_pos → FlagSimulator] × 5 steps
  → StressObjective → loss
```
All 5 GNN steps contribute to the gradient. The velocity term (`prev_x = pos`, no detach)
is now in the gradient graph — the simulator sees accurate velocity across roll steps.

Diversity in both domains: adds Gaussian noise to best_z (`σ=0.25` CFD, `σ=0.20` cloth).

---

## 4. Known Limitations in Current Code

| # | Location | Limitation | Status | Impact |
|---|---|---|---|---|
| 1 | `RealMeshLookup` | cx, cy, r have **zero gradient** through discrete argmin lookup — only v_inlet is differentiable in CFD gradient mode | **Architectural** | Geometry parameters not directly optimised; gradient only via v_inlet |
| 2 | `inverse_design.py` | K=5 BPTT may be too short for some cloth configurations that take 50-200 steps to reach steady state | **Remaining** | Stress estimate less accurate for stiff or highly-draped poses |
| 3 | `cloth_extractor.py` | Stress label uses `world_pos[-1]` (final frame only) — misses dynamic behaviour and transient stress peaks | **Remaining** | Surrogate trained on potentially unrepresentative stress values |
| 4 | `generate.py` | Checkpoints reloaded from disk on every generate call | **Remaining** | Slow startup per request (~0.5s) |
| 5 | `generate.py` | OOD index paths hardcoded, two fallback paths | **Remaining** | Fragile path resolution |
| 6 | `generate.py` | Gradient mode diversity = best_z + noise — all candidates correlated around single point | **Remaining** | Low geometric diversity in gradient mode |
| 7 | `generate.py` | In-memory thumbnail cache, max 100 sessions — no disk persistence | **Remaining** | Cache lost on server restart |
| 8 | `StressSurrogateTrainer` | No LR scheduler (unlike DragSurrogateTrainer) | **Remaining** | May converge slowly |
| 9 | `train_cvae.py` | No held-out test set — CVAE validated on same data as training | **Remaining** | No independent quality measurement |
| 10 | `Generate.tsx` | Config persisted to localStorage — domain-specific defaults not restored on switch | **Remaining** | Target slider may show wrong domain range |
| 11 | `PosePCA` / `TorchPCAInverseTransform` | PCA captures only linear deformation modes — wrinkles, folds are nonlinear | **Architectural** | Generated cloth shapes limited to linear combinations of training poses |
| 12 | `drag_surrogate.py` | `extract_true_drag` requires simulator checkpoint at training time — if absent, falls back to analytical formula | **Remaining** | Drag labels may be formula-based if checkpoint not present during training |
| 13 | `CFDMeshBuilder` | ~~Synthetic Delaunay mesh (~973 nodes) still used for thumbnails and sample-mode scoring — not the real mesh~~ | **✅ FIXED** | Thumbnails now load real OpenFOAM mesh via `RealMeshLookup` (~1883 nodes, non-uniform); OOD uses `ParamSpaceOOD` (4-D param space) |

**Previously fixed (documented for reference):**

| # | What was fixed |
|---|---|
| F1 | ~~CFD deep mode always NaN~~ — Fixed: `RealMeshLookup` loads real OpenFOAM mesh; GNN rollout converges |
| F2 | ~~`prev_x = world_pos.detach()`~~ — Fixed: detach removed; velocity gradient now flows in `_build_data_from_pos` |
| F3 | ~~Single-step cloth objective~~ — Fixed: `_rollout()` helper with K=5 BPTT replaces single-step evaluation |
| F4 | ~~`free_bits=0.0` posterior collapse risk~~ — Fixed: `free_bits=0.05` in both `CVAEConfig` and `ClothCVAEConfig` |
| F5 | ~~Pure `randn` latent sampling~~ — Fixed: LHS (Latin Hypercube Sampling) via `scipy.stats.qmc`, cached `self._lhs_sampler` |
| F6 | ~~Surrogate trained on analytical formula only~~ — Fixed: `extract_true_drag` provides GNN-rollout drag labels; `DragSurrogateTrainer.fit()` accepts `true_drag_labels` dict |
| F7 | ~~Deep/quick mode overcomplication~~ — Fixed: `mode` field removed from `GenerateRequest`; `GnnScorer` post-hoc scoring phase removed |
| F8 | ~~CFD generate OOD always 0 / always flagged OOD~~ — Fixed: `ParamSpaceOOD` replaces `OODDetector`; queries 4-D param KDTree instead of mismatched GNN embedding space |
| F9 | ~~Synthetic mesh thumbnails (CFDMeshBuilder, ~973 nodes)~~ — Fixed: `RealMeshLookup.load_mesh_for_trajectory()` loads nearest real OpenFOAM mesh (~1883 nodes) |
| F10 | ~~Stale confidence index (predict route)~~ — Fixed: `embedding_index_cylinderflow.pkl` rebuilt; staleness warning logged if checkpoint is >60 s newer than index |

---

## 5. Enhancement Proposals

Status legend: ✅ DONE · 🔲 PENDING

---

### ✅ P1 — CFD Gradient Coupling via RealMeshLookup *(DONE)*

**Was:** `CFDMeshBuilder` produced a uniform Delaunay mesh that the GNN had never seen.
Deep mode always returned NaN drag values.

**Implemented:** `RealMeshLookup` in `extensions/generative/mesh_generator.py`:
- Finds nearest real training trajectory by (cx, cy, r) using normalised L2 on `design_params.npy`
- Loads actual OpenFOAM mesh (~1876 nodes) from `traj_k.npz`
- Injects `v_inlet` as a differentiable tensor into INFLOW node features
- GNN rollout: K-1 warm-up steps (no_grad) + 1 final step with gradient
- Drag extracted from OUTFLOW nodes: `mean(|vx|)`
- Falls back to DragSurrogate MLP when checkpoints absent

**Remaining limitation:** cx, cy, r carry no gradient through the discrete lookup.
Only v_inlet is differentiable. See P1b below for the full-differentiable path.

---

### ✅ P2 — Cloth Gradient Mode: K=5 Full BPTT Objective *(DONE)*

**Was:** `StressObjective` evaluated stress from a single GNN forward step.
Cloth takes ~50-200 steps to reach steady state.

**Implemented:** `_rollout()` helper in `inverse_design.py`:
```python
ClothInverseDesigner.ROLLOUT_STEPS = 5   # class constant

def _rollout(self, world_pos, K):
    pos = world_pos
    for _ in range(K):
        graph = self._build_data_from_pos(pos)   # no detach on prev_x
        pos = self.simulator(graph)
    return pos
```
K=5 BPTT replaces the single-step evaluation. Gradient flows through all 5 steps.

**Remaining limitation:** K=5 may still be too short for complex cloth poses.
See P2b below for K=20+ longer BPTT.

---

### ✅ P3 — Enable `free_bits=0.05` in Both CVAEs *(DONE)*

**Was:** `free_bits=0.0` in both CVAEs — posterior collapse possible, some latent dims unused.

**Implemented:** `free_bits=0.05` in both `CVAEConfig` (CFD) and `ClothCVAEConfig` (cloth).
Per-dimension KL is floored at 0.05, forcing the encoder to use all 16 latent dimensions.

---

### ✅ P4 — LHS Sampling in Both CVAEs *(DONE)*

**Was:** `CFDCVAE.sample()` drew `z ~ N(0, I)` purely random — candidates could cluster.

**Implemented:** Both `CFDCVAE.sample()` and `ClothCVAE.sample()` use LHS via
`scipy.stats.qmc.LatinHypercube`. The sampler is cached as `self._lhs_sampler` (instance
attribute, created once). Fallback to `torch.randn` if scipy is unavailable.

---

### ✅ P5 — True Drag Labels for Surrogate *(DONE)*

**Was:** `DragSurrogate` trained on analytical proxy `r × v² / (1−2r/H)` — ignores viscosity,
Reynolds effects, wake dynamics.

**Implemented:**
- `extract_true_drag(simulator, dataset, traj_idx, device, steady_state_frac=0.3)` in
  `drag_surrogate.py`: runs full GNN rollout, finds OUTFLOW nodes, averages `|vx|` over the
  last 30% of timesteps.
- `DragSurrogateTrainer.fit(X, y=None, true_drag_labels=None)`: accepts `{traj_idx: drag_value}`
  dict to override analytical labels.
- `train_cvae.py` CFDStrategy: attempts to load simulator checkpoint and extract true labels
  before surrogate training; falls back to formula if checkpoint absent.

---

### ✅ P6 — Mode Simplification: Remove Deep/Quick Toggle *(DONE)*

**Was:** `GenerateRequest.mode` field (`'quick'` / `'deep'`) triggered a post-hoc GNN scoring
phase (`GnnScorer`) that always returned NaN for CFD and was never implemented for cloth.

**Implemented:** `mode` field removed from `GenerateRequest`. `GnnScorer` deep-mode phase
removed from the pipeline. `CandidateResult` no longer carries `gnn_predicted_value`,
`score_gap`, `gnn_converged`, `gnn_failed`. CandidateCard shows a single physics score.
`method` field remains: `'sample'` = surrogate scoring, `'gradient'` = gradient descent.

---

### 🔲 P7 — Fully Differentiable CFD: All Parameters, Not Just v_inlet *(PENDING)*

**Problem:** `RealMeshLookup` couples the gradient to a real mesh but the geometric parameters
(cx, cy, r) carry zero gradient — only v_inlet is differentiable through the GNN.

**Proposed — differentiable mesh deformation:**
```
Target: make the GNN input fully differentiable w.r.t. (cx, cy, r, v_inlet)

Option A — Mesh sizing field MLP + Gmsh:
  (cx, cy, r, v_inlet, query_x, query_y) → MLP → σ (edge length at point)
  Gmsh sizing field → adaptive mesh matching training distribution (~1876 nodes)
  Inject (cx, cy, r) influence through mesh node positions

Option B — Soft nearest-neighbour lookup (differentiable k-NN):
  dists = ‖design_params_norm - query‖₂     [N_traj]
  weights = softmax(-dists / τ)             [N_traj]  (temperature τ)
  v_inlet_eff = Σ_k weights_k × v_inlet_k  (convex combination)
  → approximate differentiable path for geometry params
  Limitation: topology of blended meshes undefined — only scalar fields blend cleanly
```

**Effort:** 1-2 weeks for Gmsh path; 1-2 days for soft-lookup approximation.

---

### 🔲 P8 — Longer Cloth BPTT (K=20+) *(PENDING)*

**Problem:** `ROLLOUT_STEPS=5` may be insufficient for cloth poses that take 50-200 steps
to reach steady state (stiff configurations, large initial draping displacement).

**Proposed:**
```python
ClothInverseDesigner.ROLLOUT_STEPS = 20   # or adaptive

# Or: truncated BPTT with detached warm-up:
pos_warmup = world_pos.detach()
for _ in range(K_warmup):                  # K_warmup = 45 steps, no_grad
    pos_warmup = simulator(_build_data(pos_warmup.detach()))
pos = pos_warmup.detach().requires_grad_(False)
# then K=5-10 steps with full grad
```

**Tradeoff:** Each additional BPTT step multiplies gradient computation cost.
K=20 full BPTT is ~4× more expensive than K=5. Warm-up + K=5 tail is preferred.
**Effort:** 0.5 days.

---

### 🔲 P9 — Graph VAE Replacing PCA for Cloth *(PENDING)*

**Problem:** PCA captures only linear deformation modes. Wrinkles, folds, and wave patterns
are nonlinear — PCA misses them. All generated cloth shapes are linear combinations of
training poses.

**Proposed:** GCN encoder/decoder on the fixed cloth mesh graph:
```
Encoder: world_pos [N, 3]  (node features on fixed mesh graph)
  → 3× GCNConv(3→64) + ReLU
  → global_mean_pool → [64]
  → Linear(64→16) [μ],  Linear(64→16) [log σ]

Decoder: z [16] + target_stress [1]
  → broadcast z to each of N nodes → [N, 17]
  → 3× GCNConv(17→64) + ReLU
  → Linear(64→3) per node → world_pos_recon [N, 3]
```

The graph connectivity (3028 faces) is fixed. Only vertex positions are encoded/decoded.
`TorchPCAInverseTransform` replaced by GCN decoder — gradient mode still fully differentiable.
**Effort:** 1-2 weeks.

---

### 🔲 P10 — True Multi-Trajectory Drag Dataset *(PENDING)*

**Problem:** `extract_true_drag` relies on the GNN simulator, which itself may have systematic
errors vs. ground-truth OpenFOAM. The surrogate ultimately approximates a learned model,
not the CFD solver.

**Proposed:**
1. Export actual pressure/velocity fields from OpenFOAM simulations as ground truth
2. Integrate `∮ p n̂ · x̂ dA` over cylinder surface for true pressure drag
3. Add viscous drag from wall shear stress
4. Save as `design_drag_true.npy [N_traj]`
5. Re-train DragSurrogate and CFDCVAE physics loss on these ground-truth values

**Why different from extract_true_drag:** Eliminates the GNN approximation layer entirely
from the label pipeline. Requires access to raw OpenFOAM output files.
**Effort:** 1-2 weeks (data processing pipeline + retraining).

---

### 🔲 P11 — Beta Annealing During CVAE Training *(PENDING)*

**Problem:** Fixed `beta=1e-3` throughout training. The KL term competes with reconstruction
early in training → may slow convergence or cause poor reconstruction quality.

**Proposed:**
```python
beta_t = min(cfg.beta, cfg.beta * epoch / beta_warmup_epochs)   # linear warmup
loss = alpha * recon + beta_t * kl + lam * phys
```
Standard VAE technique (Bowman et al. 2015). Start `beta=0`, ramp to `1e-3` over first 100
epochs. Improves reconstruction quality at no architecture cost.
**Effort:** 1 day.

---

### 🔲 P12 — Independent Multi-Start Gradient Descent *(PENDING)*

**Problem:** Current gradient mode generates diversity by adding noise to a single `best_z`:
```python
z_p = best_z + torch.randn_like(best_z) * noise_scale   # correlated candidates
```
All candidates cluster around the same latent-space point.

**Proposed — independent restarts per candidate:**
```python
results = []
for i in range(n_candidates):
    z0 = torch.randn(latent_dim)   # independent init per candidate
    z_opt = optimise(z0, target, n_iters)
    results.append(z_opt)
```
True diversity — not noise around one optimum. Cost: n_candidates × n_iters gradient steps.
**Effort:** 1 day.

---

### 🔲 P13 — CFD Mesh Generation via Sizing Field MLP + Gmsh *(PENDING)*

**Problem:** `CFDMeshBuilder` produces a uniform Delaunay grid. Real OpenFOAM meshes have
adaptive density — denser near the cylinder wall and wake. Thumbnails and sample-mode
OOD scores use the synthetic mesh.

**Proposed:**
```
(cx, cy, r, v_inlet, query_x, query_y) [6]
  → 3-layer MLP (~2000 params)
  → σ  (target edge length at this point)

Training data: existing training meshes
  For each node in each traj_k.npz:
    input  = (cx, cy, r, v_inlet, node_x, node_y)
    target = min edge length touching this node

At generation:
  MLP predicts σ on background grid
  Gmsh API: set σ as background mesh field
  → Delaunay + refinement → adaptive mesh like OpenFOAM (~1876 nodes)
  → thumbnails and OOD scores use realistic mesh
```

Why not graph generative models (GRAN, DiGress, GDSS): max node limits (~40–200 nodes),
memory issues, require thousands of training graphs. Sizing MLP + Gmsh is reliable, fast
(~50ms), and quality is guaranteed by Delaunay theory.
**Effort:** 1-2 weeks.

---

### 🔲 P14 — Multi-Objective Pareto Generation *(PENDING)*

**Problem:** Single scalar target forces a trade-off. Real designs may need
"low drag AND structural stability."

**Proposed — vector-conditioned CVAE:**
```python
# Decoder input: cat(z[16], target_drag[1], target_pressure[1])  → [18]
# Physics loss:  α₁·drag_loss + α₂·pressure_loss
```

**UI addition:** 2D scatter of (drag, pressure) for generated candidates, Pareto front
highlighted. User clicks a point on the Pareto front to select a design.
**Effort:** 2-3 weeks.

---

### 🔲 P15 — Active Learning Loop *(PENDING)*

**Problem:** CVAE and surrogate trained once on fixed dataset and never updated.
Novel generated designs (near OOD boundary) have less reliable predictions.

**Proposed:**
```
1. Generate N candidates
2. Identify K highest-uncertainty candidates:
   - High OOD distance (low ood_confidence)
   - OR: high deviation between surrogate and GNN rollout drag
3. Run actual CFD simulation on those K (offline, expensive)
4. Add (params, real_drag) to dataset
5. Re-train surrogate on augmented data
6. Re-train CVAE with updated physics consistency
```

**Effort:** 2-4 weeks (research contribution).

---

### ✅ P16 — Param-Space OOD for Generate Pipeline *(DONE)*

**Was:** `CFDDesignSampler.sample()` passed each synthetic `CFDMeshBuilder` graph through
the GNN encoder (128-D embedding) and queried a KDTree built from real training embeddings.
Domain shift between synthetic Delaunay meshes and real OpenFOAM meshes → GNN embedding
always far from training → `confidence ≈ 0.0` → every candidate flagged OOD.

**Implemented:** `ParamSpaceOOD` in `extensions/confidence/ood_detector.py`:
- Loads `data/design_params.npy` [N, 4] (cx, cy, r, v_inlet)
- Normalises all 4 columns to [0, 1], builds `scipy.spatial.KDTree`
- `confidence = clip(1 − d_min / train_diameter, 0, 1)` where `train_diameter` is the
  95th-percentile leave-one-out NN distance (identical formula to `NearestNeighborIndex`)
- Wired into `CFDDesignSampler.__init__` as `self._param_ood`; called in `sample()` as
  `self._param_ood.score(cx, cy, r, v_in)`
- No GNN or mesh required; gracefully degrades to `confidence=-1.0` if file missing

**Why param-space is correct here:** If we used `RealMeshLookup` meshes for OOD we'd always
query training meshes by definition → always in-distribution. The right question is whether
the *design params* are in the training envelope.

---

### ✅ P17 — Real Mesh Thumbnails via RealMeshLookup *(DONE)*

**Was:** `CFDDesignSampler.sample()` called `CFDMeshBuilder.build(cx, cy, r, v_inlet)` to
generate a synthetic Delaunay triangulation (~973 nodes, uniform spacing) for thumbnails.
Visual quality was poor — no cylinder wake structure, no boundary layer refinement.

**Implemented:** `RealMeshLookup.load_mesh_for_trajectory()` in `sample()`:
- `find_nearest(cx, cy, r)` → closest real training trajectory index by normalised L2
- `load_mesh_for_trajectory(traj_idx, dummy_v_inlet, dataset, device='cpu')` → real
  OpenFOAM PyG Data (~1883 nodes, non-uniform spacing, finer near cylinder)
- Thumbnail rendered from the real mesh → accurate velocity-magnitude heatmap
- Gracefully skips thumbnail (sets `graph=None`) if `RealMeshLookup` is unavailable

`CFDMeshBuilder` is retained for `MeshGeneratorFactory` registry, `params_to_graph()`
helper, and CLI smoke test but is no longer used in `CFDDesignSampler.sample()`.

---

| # | Enhancement | Files | Effort | Status |
|---|---|---|---|---|
| P1 | CFD gradient coupling via RealMeshLookup | `mesh_generator.py`, `generate.py` | 1 day | ✅ DONE |
| P2 | Cloth K=5 BPTT objective | `inverse_design.py` | 0.5 days | ✅ DONE |
| P3 | free_bits=0.05 | `cvae_cfd.py`, `cvae_cloth.py` | 30 min | ✅ DONE |
| P4 | LHS sampling | `cvae_cfd.py`, `cvae_cloth.py` | 5 min | ✅ DONE |
| P5 | True drag labels (extract_true_drag) | `drag_surrogate.py`, `train_cvae.py` | 1-2 weeks | ✅ DONE |
| P6 | Remove deep/quick mode toggle | `generate.py`, `Generate.tsx`, `CandidateCard.tsx` | 1 day | ✅ DONE |
| P16 | Param-space OOD for generate pipeline | `ood_detector.py`, `generate.py` | 0.5 days | ✅ DONE |
| P17 | Real mesh thumbnails via RealMeshLookup | `generate.py`, `mesh_generator.py` | 0.5 days | ✅ DONE |
| P7 | Fully differentiable CFD (all params) | `mesh_generator.py`, `generate.py` | 1-2 weeks | 🔲 PENDING |
| P8 | Longer cloth BPTT (K=20+) | `inverse_design.py` | 0.5 days | 🔲 PENDING |
| P9 | Graph VAE replacing PCA (cloth) | `cvae_cloth.py`, `cloth_extractor.py` | 1-2 weeks | 🔲 PENDING |
| P10 | True multi-trajectory drag dataset (OpenFOAM) | `drag_surrogate.py`, `train_cvae.py` | 1-2 weeks | 🔲 PENDING |
| P11 | Beta annealing during CVAE training | `train_cvae.py` | 1 day | 🔲 PENDING |
| P12 | Independent multi-start gradient descent | `generate.py` | 1 day | 🔲 PENDING |
| P13 | Sizing field MLP + Gmsh (CFD mesh) | `mesh_generator.py` | 1-2 weeks | 🔲 PENDING |
| P14 | Multi-objective Pareto generation | `cvae_cfd.py`, `Generate.tsx` | 2-3 weeks | 🔲 PENDING |
| P15 | Active learning loop | new `extensions/active_learning/` | 2-4 weeks | 🔲 PENDING |

**Highest ROI next steps:**
1. P8 (longer cloth BPTT, 0.5 days) — low cost, meaningful accuracy gain
2. P7-soft-lookup (differentiable geometry approx, 1-2 days) — unlocks full geometry gradient
3. P12 (independent multi-start, 1 day) — true diversity in gradient mode

---

*Document updated 2026-04-14 — ParamSpaceOOD (generate OOD), real-mesh thumbnails*
*(RealMeshLookup in sample path), confidence index rebuilt + staleness warning.*
*Prior update 2026-04-13 — rearchitected generate pipeline.*
*All class names, file paths, tensor shapes, and hyperparameters are exact values from source.*
