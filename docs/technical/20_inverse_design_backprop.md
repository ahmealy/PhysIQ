---
tags: [physicsai, inverse-design, backprop, gradient, cvae, bptt, surrogate]
created: 2026-04-27
aliases: [inverse-backprop, gradient-chain, inverse-design-gradient]
---

# Inverse Design — Differentiable Path (CFD vs Cloth)

Both domains optimise a single latent vector `z ∈ R¹⁶` via Adam. The gradient path from loss back to `z` is completely different because CFD uses a surrogate MLP while cloth uses full BPTT through the GNN.

---

## The Shared Setup (Both Domains)

```
z = torch.randn([16], requires_grad=True)   ← only leaf variable

All other networks are frozen (eval mode, no weight updates):
  - CVAE decoder
  - DragSurrogate / FlagSimulator
  - Normalizers

Only z moves. Gradients flow THROUGH frozen networks to reach z.
```

Adam optimiser: `lr=0.05`, 3 restarts.

---

## CFD — Surrogate MLP Path

### Why not the GNN?

`_use_gnn = False` is hardcoded in `CFDDesignSampler._gradient_sample()`.

The GNN path is **implemented** but **always disabled** for two reasons:

```
z → decoder → (cx, cy, r, v_inlet)
                      │
                      ▼
    RealMeshLookup.find_nearest(cx, cy, r)
    = KDTree argmin → integer index
    ← DISCRETE: ∂index/∂(cx,cy,r) = 0
```

Only `v_inlet` would survive as gradient through the GNN. With cx/cy/r having zero gradient, the optimiser can barely steer `z`. The surrogate MLP is fully differentiable through all 4 parameters with well-matched units — strictly better for optimisation.

### Full Chain (CFD)

```mermaid
flowchart TD
    Z["z  [16]\nrequires_grad=True"]
    DEC["CVAE Decoder  frozen, eval\nFC: 17→64→ReLU→64→ReLU→4\ninput: cat(z[16], target_drag_norm[1])"]
    PN["params_norm  [4]\ncx, cy, r, v_inlet  in [0,1]"]
    PP["params_phys  [4]\naffine denorm: × (p_max−p_min) + p_min\n∂/∂params_norm ✓"]
    SI["surr_input  [4]\naffine renorm into surrogate scale\n∂/∂params_phys ✓"]
    SURR["DragSurrogate MLP  frozen, eval\nFC: 4→64→ReLU→64→ReLU→1→clamp≥0\n∂/∂surr_input ✓"]
    DN["drag_norm  [1]"]
    DP["drag_phys  [1]\naffine denorm: × (y_max−y_min) + y_min\n∂/∂drag_norm ✓"]
    LOSS["loss = MSE(drag_phys, target_drag)\nscalar"]
    ADAM["loss.backward()\nAdam.step() on z\n150 iters × 3 restarts = 450 total steps"]

    Z --> DEC --> PN --> PP --> SI --> SURR --> DN --> DP --> LOSS --> ADAM
```

### Tensor shapes at each step

| Step | Tensor | Shape |
|---|---|---|
| Leaf | `z` | `[16]` |
| Decoder input | `cat(z, target_norm)` | `[17]` |
| Decoder output | `params_norm` | `[4]` |
| Denorm | `params_phys` | `[4]` |
| Renorm | `surr_input` | `[1, 4]` |
| Surrogate output | `drag_norm` | `[1]` |
| Denorm | `drag_phys` | `[1]` |
| Loss | scalar | `[]` |

Every operation is a matrix multiply or affine transform — PyTorch autograd traces through all of them automatically. The gradient `∂loss/∂z` flows back through the surrogate MLP weights (which are fixed scalars at this point) then through the decoder weights (also fixed) to reach `z`.

---

## Cloth — BPTT Through GNN

### Why the GNN (not a surrogate)?

Cloth has a **continuous** latent→position mapping via PCA inverse transform:

```
z → decoder → PCA coefficients → PCA⁻¹ (matrix multiply) → world_pos [N, 3]
```

PCA inverse transform is an exact differentiable matrix multiply — no discrete lookup. All 4 parameters are differentiable all the way to `z`. The GNN rollout is the only model that can compute physically meaningful stress (displacement from rest), so BPTT through it is both possible and necessary.

`StressSurrogate` MLP exists but is only used during **CVAE training** as the physics consistency loss term — it is **not used** during inverse design optimisation.

### Full Chain (Cloth)

```mermaid
flowchart TD
    Z2["z  [16]\nrequires_grad=True"]
    DEC2["ClothCVAE Decoder  frozen, eval\nFC: 17→128→ReLU→128→ReLU→K=16\ninput: cat(z[16], target_stress_norm[1])"]
    PN2["pose_norm  [16]\nnormalised PCA coefficients"]
    PP2["pose_phys  [16]\naffine denorm via ClothCVAEScaler\n∂/∂pose_norm ✓"]
    PCA["TorchPCAInverseTransform\nflat = z_pca @ components + mean\n.view(N, 3)\n∂/∂pose_phys ✓  — pure matrix multiply"]
    WP["world_pos  [N, 3]\ninitial cloth node positions\ngrad_fn=ViewBackward"]

    subgraph BPTT["BPTT rollout — K=5 steps, NO detach"]
        S1["Step 1\n_build_data_from_pos(world_pos, prev=world_pos.clone())\nvelocity = world_pos − prev_x  ← grad flows through both\nFlagSimulator.forward(graph)\n→ next_pos_1  [N,3]"]
        S2["Step 2\n_build_data_from_pos(next_pos_1, prev=world_pos)\nFlagSimulator.forward(graph)\n→ next_pos_2  [N,3]"]
        S3["Steps 3, 4, 5\nsame — no detach on current_pos or prev_pos"]
        FP["final_pos  [N, 3]"]
        S1 --> S2 --> S3 --> FP
    end

    OBJ["StressObjective\ndisp = ‖final_pos[normal_mask] − mesh_rest‖₂  [N_normal]\nstress = disp.mean()\n∂/∂final_pos ✓"]
    LOSS2["loss = MSE(stress, target_stress)\nscalar"]
    ADAM2["loss.backward()\nAdam.step() on z\n100 iters × 3 restarts"]

    Z2 --> DEC2 --> PN2 --> PP2 --> PCA --> WP --> BPTT --> OBJ --> LOSS2 --> ADAM2
```

### Why prev_x must NOT be detached

Inside `_build_data_from_pos`, the velocity feature is:

```python
delta_pos = world_pos - prev_x   # [N, 3]
```

If `prev_x` were detached, the gradient would be cut at that edge feature, losing half the velocity signal's contribution. The code explicitly preserves the grad_fn:

```python
if prev_x is None:
    prev_x = world_pos.clone()   # clone() keeps grad_fn — NOT detach()
```

### Tensor shapes through the cloth chain

| Step | Tensor | Shape |
|---|---|---|
| Leaf | `z` | `[16]` |
| Decoder input | `cat(z, target_norm)` | `[17]` |
| Decoder output | `pose_norm` | `[16]` |
| Denorm | `pose_phys` | `[16]` |
| PCA inverse | `world_pos` | `[N, 3]` |
| After each GNN step | `next_pos` | `[N, 3]` |
| Displacement | `disp` | `[N_normal]` |
| Stress | scalar | `[]` |
| Loss | scalar | `[]` |

---

## Side-by-Side Comparison

| | CFD (`cylinder_flow`) | Cloth (`flag_simple`) |
|---|---|---|
| **Leaf variable** | `z [16]` | `z [16]` |
| **Physics model in loop** | `DragSurrogate` MLP (4→64→64→1) | `FlagSimulator` GNN × 5 rollout steps |
| **GNN used?** | ❌ `_use_gnn = False` | ✅ Always — full BPTT |
| **Why no GNN for CFD** | Discrete KDTree mesh lookup → ∂/∂(cx,cy,r)=0 | N/A — PCA inverse is continuous |
| **Surrogate role** | Replaces GNN during optimisation | CVAE training only — not used here |
| **Loss** | `MSE(surrogate_drag, target_drag)` | `MSE(mean_‖pos−rest‖, target_stress)` |
| **Gradient path length** | z → 2-layer decoder → 3-layer MLP → loss | z → 2-layer decoder → PCA⁻¹ → 5×(15-layer GNN) → stress → loss |
| **All params differentiable?** | ✅ All 4: cx,cy,r,v_inlet | ✅ Full N×3 position field |
| **Iterations** | 150 × 3 restarts = 450 steps | 100 × 3 restarts = 300 steps (80 × 2 from API) |

---

## What the gradient actually updates

In both cases, `loss.backward()` computes `∂loss/∂z`. The frozen network weights receive `∂loss/∂weights` but are **never stepped** — only `z` moves.

```
Frozen (computed but not updated):
  ∂loss/∂decoder_weights    ← passes through but Adam ignores
  ∂loss/∂surrogate_weights  ← CFD only
  ∂loss/∂GNN_weights        ← cloth only

Updated:
  ∂loss/∂z  →  Adam.step()  →  z_{t+1} = z_t - lr · m̂/√v̂
```

This is the same mechanism as feature inversion / style transfer in image models — you backprop through a frozen network to optimise an input, not the weights.
