---
tags: [physicsai, inverse-design, cvae, generate, drag, lhs]
created: 2026-04-25
aliases: [inverse-design, cvae-pipeline, generate-pipeline]
---

# Inverse Design Pipeline

How PhysIQ finds cylinder geometries that produce a target drag — from user input to candidate mesh.

---

## Overview

```mermaid
flowchart TD
    USER["User: target drag  e.g. Cd = 0.3"]

    subgraph TRAIN_TIME["Built once — before Generate is used"]
        SE["shape_extractor.py\nfor every training trajectory:\n  fit circle to WALL_BOUNDARY nodes\n  → cx, cy, r\n  v_inlet = mean vx of INFLOW nodes\n  → [cx, cy, r, v_inlet] per trajectory"]
        DS["DragSurrogate MLP\n[cx, cy, r, v_inlet] → drag\n4 → 64 → 64 → 1\ntrained on (params, true_drag) pairs"]
        CV["CFDCVAE\nEncoder: [cx,cy,r,v_inlet, drag] → μ[16], σ[16]\nDecoder: [z[16], target_drag] → [cx,cy,r,v_inlet]\nloss = recon + β·KL + λ·physics_consistency\nfree_bits=0.05 prevents posterior collapse"]
        RL["RealMeshLookup\nKDTree over training [cx,cy,r] vectors\nsnaps generated params to nearest real mesh"]
        OOD["ParamSpaceOOD\nKDTree over training [cx,cy,r,v_inlet]\nconfidence = clip(1 - d_min/train_diam, 0,1)"]
        SE --> DS
        SE --> CV
        SE --> RL
        SE --> OOD
    end

    subgraph SAMPLE["Phase 1 — CVAE Sampling  method=sample"]
        LHS["Latin Hypercube Sampling\nz ~ LHS(n, d=16)\n→ uniform coverage of latent space\ntransformed to N(0,I) via scipy.stats.norm.ppf"]
        DEC["CVAE Decoder\n[z[16], target_drag] → [cx,cy,r,v_inlet]"]
        RML["RealMeshLookup.find_nearest(cx,cy,r)\nreturns trajectory index of nearest real mesh\nguarantees valid triangulation"]
        SURR["DragSurrogate.predict(cx,cy,r,v_inlet)\ninstant drag estimate  ~0.1 ms"]
        CONF["ParamSpaceOOD.score(cx,cy,r,v_inlet)\nconfidence 0–1"]
    end

    subgraph GRADIENT["Phase 2 — Gradient Refinement  method=gradient"]
        ADAM["Adam on z ∈ R¹⁶\nminimise (drag_pred - target)²"]
        BPTT["K=5 BPTT through GNN\nsteps 1..K-1 detached, step K differentiable\ngradient flows: z → decoder → params → mesh → GNN → drag"]
        RML2["RealMeshLookup.find_nearest()\nafter each Adam step — keep mesh valid"]
    end

    subgraph SSE["SSE stream to browser"]
        CAND["event: candidate\n{id, cx, cy, r, v_inlet, drag_pred, confidence, thumbnail_png}"]
    end

    USER --> LHS
    CV --> DEC
    LHS --> DEC
    DEC --> RML --> SURR & CONF
    SURR & CONF --> CAND
    USER --> ADAM
    ADAM --> BPTT --> RML2 --> SURR & CONF
    CAND --> SSE
```

---

## Design Parameters — What They Are

Every CFD inverse design candidate is described by 4 scalars:

```
[cx, cy, r, v_inlet]

cx      — cylinder centre x-position (normalised [0,1] of domain width)
cy      — cylinder centre y-position (normalised [0,1] of domain height)
r       — cylinder radius (normalised units)
v_inlet — inlet flow velocity magnitude (m/s)
```

These are **reverse-engineered from the training mesh**, not stored in the TFRecord:

```mermaid
flowchart LR
    MESH["Training mesh\nnode positions + node_type"]
    WALL["WALL_BOUNDARY nodes\n= cylinder surface + top/bottom walls"]
    SPLIT["Separate:\n  top/bottom wall  →  near y=0 or y=1\n  interior wall    →  cylinder surface"]
    FIT["Least-squares circle fit\n(x-cx)²+(y-cy)²=r²\n→ cx, cy, r"]
    VIN["INFLOW nodes at t=0\nv_inlet = mean(vx)"]
    PARAMS["CylinderParams(cx, cy, r, v_inlet)"]

    MESH --> WALL --> SPLIT --> FIT & VIN --> PARAMS
```

---

## CVAE Architecture

```mermaid
flowchart LR
    subgraph ENCODER["Encoder  training only"]
        EP["[cx,cy,r,v_inlet, drag_actual]\n[B, 5]"]
        EH["FC 5→64 → ReLU → FC 64→64 → ReLU"]
        MU["μ [B, 16]"]
        LV["log σ [B, 16]"]
        EP --> EH --> MU & LV
    end

    subgraph REPARAM["Reparameterisation"]
        Z["z = μ + σ · ε\nε ~ N(0,I)\n[B, 16]"]
        MU & LV --> Z
    end

    subgraph DECODER["Decoder  training + inference"]
        DP["[z[16], target_drag[1]]\n[B, 17]"]
        DH["FC 17→64 → ReLU → FC 64→64 → ReLU"]
        OUT["[cx, cy, r, v_inlet]\n[B, 4]"]
        Z --> DP --> DH --> OUT
    end
```

### Training loss

```
L = α · ‖params_recon - params_gt‖²          reconstruction
  + β · KL(q(z|params,drag) ‖ N(0,I))        regularisation  (free_bits=0.05)
  + λ · |drag_surrogate(params_recon) - drag_target|   physics consistency
```

**free_bits = 0.05**: KL per latent dimension is clamped to at least 0.05. Prevents the encoder from collapsing unused latent dimensions to the prior (posterior collapse), ensuring all 16 dimensions carry information.

**Physics consistency term**: the decoder is forced, during training, to produce parameters whose drag (as estimated by the surrogate) matches the target. This makes the decoder learn the physics mapping — not just reconstruction.

---

## Drag Surrogate

```mermaid
flowchart LR
    IN["[cx, cy, r, v_inlet]\n[B, 4]  min-max scaled"]
    H1["FC 4→64 → ReLU"]
    H2["FC 64→64 → ReLU"]
    OUT["drag prediction\n[B, 1]  scaled"]
    IN --> H1 --> H2 --> OUT
```

Trained on `(params, true_drag)` pairs extracted from training trajectories. `true_drag` is computed from the last 10% of timesteps of each GNN rollout (steady-state average).

Used for:
1. **CVAE physics consistency loss** during CVAE training
2. **Instant candidate scoring** at generate time — avoids running a full 600-step GNN rollout per candidate

---

## Latin Hypercube Sampling

```mermaid
flowchart LR
    RAND["Random Normal\nz ~ N(0,I)  independent\nclumps in high-density regions\npoor coverage of latent space tails"]
    LHS["Latin Hypercube\ndivides [0,1]ⁿ into n equal slices per dim\nguarantees one sample per slice\ntransformed: z = norm.ppf(lhs_sample)\nuniform coverage of all 16 dims"]
    RAND -. "replaced by" .-> LHS
```

With n=10–20 candidates and a 16-dimensional latent space, random Normal sampling tends to cluster near the origin. LHS partitions each dimension into n equal-probability intervals and places exactly one sample per interval — much better coverage of the latent space with the same budget.

---

## RealMeshLookup — Why Generated Params Must Be Snapped

```mermaid
flowchart TD
    GEN["CVAE generates cx=0.31, cy=0.52, r=0.08"]
    PROB["Problem:\nBuilding a mesh from scratch requires Delaunay triangulation.\nArbitrary (cx,cy,r) may produce:\n  - degenerate triangles\n  - invalid node counts\n  - meshes the GNN has never seen"]
    SNAP["RealMeshLookup.find_nearest(cx, cy, r)\nKDTree over training [cx,cy,r] vectors\nreturns index of nearest real training trajectory"]
    REAL["Real training mesh loaded from .npz\n  valid triangulation\n  correct node count\n  GNN has seen this geometry"]
    INJ["Inject v_inlet:\n  set INFLOW node velocities to new v_inlet\n  (only scalar change — mesh topology unchanged)"]
    GRAPH["Valid PyG Data graph\nready for GNN rollout"]

    GEN --> PROB --> SNAP --> REAL --> INJ --> GRAPH
```

This is the key design choice that makes gradient refinement possible: instead of differentiating through mesh construction (impossible — Delaunay is not differentiable), we snap to a real mesh and only differentiate through the GNN rollout.

---

## Gradient Refinement — BPTT K=5

```mermaid
flowchart TD
    Z0["z ∈ R¹⁶  (initial: LHS sample)"]
    DEC["CVAE Decoder\nz + target_drag → [cx,cy,r,v_inlet]"]
    SNAP["RealMeshLookup.find_nearest(cx,cy,r)\n→ real mesh graph"]
    INJ["inject v_inlet into graph"]

    subgraph ROLLOUT["GNN Rollout  K=5 steps"]
        S1["step 1  detached\nno gradient"]
        S2["step 2  detached"]
        S3["step 3  detached"]
        S4["step 4  detached"]
        S5["step K  differentiable\ngradient flows here"]
    end

    DRAG["drag = mean force on\nWALL_BOUNDARY nodes at step K"]
    LOSS["loss = (drag - target_drag)²"]
    ADAM["Adam.step() on z\n∂loss/∂z via chain rule:\nz → decoder → params → GNN step K → drag"]
    SNAP2["snap to nearest real mesh again\n(params changed → may need new mesh)"]

    Z0 --> DEC --> SNAP --> INJ --> S1 --> S2 --> S3 --> S4 --> S5
    S5 --> DRAG --> LOSS --> ADAM --> Z0
    ADAM --> SNAP2
```

**Why K-1 detached steps?** Running all K steps differentiably would require storing intermediate activations for all K steps — high memory. Only the last step is differentiable, which is enough: the gradient signal tells the optimiser which direction to move `z` to reduce drag at the final step.

**Why K=5 and not K=1?** One step isn't enough physics — the flow hasn't developed. Five steps gives enough temporal context for the drag signal to be meaningful without excessive memory.

---

## Confidence Scoring — Two Different OOD Checks

The system has two distinct OOD mechanisms that answer different questions:

| | ParamSpaceOOD (Generate page) | NearestNeighborIndex (Predict page) |
|---|---|---|
| Space | 4D parameter space [cx,cy,r,v_inlet] | 256D embedding space |
| Question | "Is this cylinder geometry unusual?" | "Is this simulation unusual?" |
| KDTree input | training param vectors | training GNN embeddings |
| Used for | candidate confidence in Generate | rollout confidence in Predict |

```mermaid
flowchart LR
    subgraph PARAM["ParamSpaceOOD — Generate"]
        PV["[cx, cy, r, v_inlet]\n4D vector"]
        PKD["KDTree over\ntraining param vectors"]
        PS["score = clip(1 - d_min/train_diam, 0,1)"]
        PV --> PKD --> PS
    end

    subgraph EMBED["NearestNeighborIndex — Predict"]
        EV["concat(encode(frame_0), encode(frame_5))\n256D embedding"]
        EKD["KDTree over\ntraining embeddings"]
        ES["score = clip(1 - d_min/train_diam, 0,1)"]
        EV --> EKD --> ES
    end
```

---

## Full Tensor Shape Summary

| Step | Tensor | Shape | Notes |
|---|---|---|---|
| CVAE encoder input | [params, drag] | [B, 5] | 4 design params + 1 drag |
| CVAE latent | μ, log σ | [B, 16] | 16-dim latent space |
| LHS samples | z | [n, 16] | n = n_candidates |
| CVAE decoder input | [z, target_drag] | [n, 17] | |
| CVAE decoder output | params | [n, 4] | cx, cy, r, v_inlet |
| DragSurrogate input | params (scaled) | [n, 4] | min-max normalised |
| DragSurrogate output | drag | [n, 1] | |
| ParamSpaceOOD query | params | [1, 4] | per candidate |
| GNN rollout output | velocity field | [N, 2] per step | used to compute drag |
