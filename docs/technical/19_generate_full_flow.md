---
tags: [physicsai, inverse-design, generate, cvae, gradient, sse, flow]
created: 2026-04-25
aliases: [generate-flow, candidate-generation, inverse-design-flow]
---

# Generate — Full Flow from Button Press to Candidates

Everything that happens from the moment the user clicks **Generate** to the moment candidates appear in the browser. Both `sample` and `gradient` methods covered in full.

---

## Top-Level Overview

```mermaid
flowchart TD
    USER["User clicks Generate\ndomain=cylinder_flow\ntarget_drag=0.3\nn_candidates=10\nmethod=sample OR gradient"]

    API["POST /api/generate\nFastAPI route\ncreates session_id = uuid4()"]

    SSE["SSE stream opens\ntext/event-stream\nevents: progress · candidate · trajectory · done · error"]

    subgraph THREAD["Thread pool  asyncio.run_in_executor"]
        SAMP["CFDDesignSampler.sample()\n↓ branches on method"]
    end

    subgraph RENDER["Main event loop  after thread returns"]
        THUMB["render thumbnail per candidate\nThumbnailRenderer.render_cfd()"]
        STREAM["yield event: candidate\none by one as each renders"]
    end

    DONE["event: done  {best_id}"]

    USER --> API --> SSE --> THREAD --> RENDER --> DONE
```

---

## Path A — `method="sample"` (CVAE + LHS)

```mermaid
flowchart TD
    TARGET["target_drag = 0.3\nn = 10"]

    LHS["Latin Hypercube Sampling\nLatinHypercube(d=16).random(n=10)\n→ 10 points in [0,1]¹⁶  uniform coverage\n→ norm.ppf() transforms to N(0,I) scale\nz_lhs  [10, 16]"]

    DEC["CVAE Decoder  (frozen, eval mode)\nfor each of 10 z vectors:\n  input: [z[16], target_drag[1]] → [B, 17]\n  FC 17→64→ReLU→64→ReLU→4\n  output: params_normalised [4]"]

    DENORM["Denormalise\nparams_phys = params_n × (p_max - p_min) + p_min\n→ [cx, cy, r, v_inlet]  physical units\nclip cx,cy,r to valid domain bounds"]

    MESH["RealMeshLookup.find_nearest(cx, cy, r)\nKDTree over training [cx,cy,r] vectors\n→ trajectory index of nearest real mesh\nload_mesh_for_trajectory(idx, v_inlet)\n→ PyG Data graph with real triangulation"]

    SURR["DragSurrogate.predict(cx, cy, r, v_inlet)\nMLP 4→64→64→1\n→ drag_pred  scalar\n~0.1 ms per candidate"]

    OOD["ParamSpaceOOD.score(cx, cy, r, v_inlet)\nKDTree over training param vectors\nd_min = nearest training point in 4D\nconfidence = clip(1 - d_min/train_diam, 0,1)"]

    SORT["sort all 10 by |drag_pred - target|\nreturn top n"]

    TARGET --> LHS --> DEC --> DENORM --> MESH --> SURR & OOD --> SORT
```

### What each z[16] is

`z` is the CVAE latent code — a 16-dimensional vector in a learned continuous space where similar cylinder designs are nearby. It has no human-interpretable dimensions; the CVAE encoder mapped thousands of `(cx,cy,r,v_inlet)` training examples into this space during training.

Sampling `z ~ N(0,I)` via LHS gives us 10 points that are spread evenly across the latent space rather than clumped near the origin (which random Normal would produce). Each z, when decoded with a target drag condition, produces a different cylinder geometry that the CVAE believes could achieve that drag.

---

## Path B — `method="gradient"` (Adam in latent space)

```mermaid
flowchart TD
    TARGET["target_drag = 0.3"]

    INIT["z = torch.randn([16])\nrequires_grad = True\nrandom starting point in latent space"]

    subgraph LOOP["Adam optimisation loop  n_restarts=3 × n_iters=150"]
        DEC2["CVAE Decoder  (frozen, eval)\n[z[16], target_drag_norm[1]] → params_norm [4]"]
        DENORM2["denorm: params_phys [4]  cx,cy,r,v_inlet"]
        NORM2["renorm into surrogate input space [4]"]
        SURR2["DragSurrogate MLP  (frozen, eval)\n4→64→64→1\n→ drag_norm"]
        DENORM3["denorm → drag_phys  scalar"]
        LOSS["MSE loss = (drag_phys - target)²"]
        BACK["loss.backward()\n∂loss/∂z via chain rule through:\n  denorm → surrogate → norm → denorm → decoder → z\nAdam.step() updates z"]
        DEC2 --> DENORM2 --> NORM2 --> SURR2 --> DENORM3 --> LOSS --> BACK --> DEC2
    end

    BEST["best_z* = z with lowest |drag - target|\nacross all 3 restarts"]

    DIVERSE["sample n diverse candidates around best_z*:\nfor i in range(n):\n    z_i = best_z + randn_like(best_z) × 0.10\n    params_i = decoder(z_i, target_drag)\n→ n variations of the optimal design"]

    MESH2["RealMeshLookup.find_nearest(cx,cy,r)\n→ nearest real training mesh per candidate"]
    OOD2["ParamSpaceOOD.score()\nconfidence per candidate"]

    TARGET --> INIT --> LOOP --> BEST --> DIVERSE --> MESH2 & OOD2
```

### Why gradient does NOT flow through the GNN

The GNN path is implemented but disabled (`_use_gnn = False`). The reason:

```
z → decoder → (cx, cy, r, v_inlet)
                       │
                       ▼
       RealMeshLookup.find_nearest(cx, cy, r)
       = KDTree argmin  →  integer index
       ← DISCRETE: ∂index/∂(cx,cy,r) = 0
```

Only `v_inlet` would survive as a differentiable signal through the GNN. With 3 of 4 parameters having zero gradient, plus unit-scale mismatches between GNN drag proxy and target drag, the optimiser can barely steer `z`. The surrogate is fully differentiable through all 4 parameters and well-scaled — it's strictly better for optimisation.

### Gradient chain (surrogate path, what actually runs)

```
z [16]  requires_grad=True
  ↓  CVAE decoder  (FC layers, ∂/∂z ✓)
params_norm [4]
  ↓  affine denorm  (∂/∂params_norm ✓)
params_phys [4]  (cx, cy, r, v_inlet)
  ↓  affine renorm into surrogate space  (∂/∂params_phys ✓)
params_surr [4]
  ↓  DragSurrogate MLP  FC 4→64→64→1  (∂/∂params_surr ✓)
drag_norm [1]
  ↓  affine denorm  (∂/∂drag_norm ✓)
drag_phys [1]
  ↓  MSE  = (drag_phys - target)²
loss
  ↓  .backward()
∂loss/∂z  →  Adam updates z
```

Every step is matrix multiplications and affine transforms — PyTorch autograd traces through automatically. 150 iterations × 3 restarts = 450 total gradient steps.

### After optimisation — n diverse candidates

Gradient mode finds **one** optimal `z*`. To give the user n candidates, small noise is added:

```python
z_i = best_z + torch.randn_like(best_z) * 0.10
```

`noise_scale=0.10` with `latent_dim=16` gives `E[‖noise‖] = 0.10 × √16 = 0.40` — a modest perturbation that keeps candidates near the optimum while varying the geometry. These are n variations of the same intent, not n independent designs (unlike sample mode).

---

## Path A vs Path B — When to use which

| | `method="sample"` | `method="gradient"` |
|---|---|---|
| How z is found | LHS covers latent space broadly | Adam searches for best z |
| n candidates | n independent designs, diverse geometries | n variations around one optimal design |
| Drag accuracy | Depends on where the decoder points | Explicitly minimised — closer to target |
| Speed | Fast (n decoder calls) | Slower (450 gradient steps) |
| Best for | "Show me diverse options near target drag" | "Find the design closest to exactly Cd=0.3" |

---

## After sampling — rendering and streaming (both paths)

```mermaid
sequenceDiagram
    participant T as Thread (sampler.sample)
    participant L as Event loop
    participant B as Browser

    T->>L: returns (results, trajectory)  list of (CandidateResult, graph)
    L->>B: event: trajectory  {values: [drag step 0..149]}  (gradient only)

    loop for each candidate i in results
        L->>L: ThumbnailRenderer.render_cfd(graph, cx, cy, r)
        note over L: draw mesh triangles + cylinder circle\ncolour by node type  →  PNG bytes
        L->>L: store PNG in _thumbnail_cache[session_id][i]
        L->>B: event: candidate  {id, cx, cy, r, v_inlet,\n  drag_pred, confidence, thumbnail_url,\n  session_id, ...}
        note over B: card appears immediately\nuser sees candidates one by one
    end

    L->>B: event: done  {best_id: i_with_min_error}
```

### SSE event types

| Event | Payload | When |
|---|---|---|
| `progress` | `{phase, step, done, total}` | Before sampling starts, during rendering |
| `trajectory` | `{values: [float]}` | Gradient mode only — loss curve |
| `candidate` | Full CandidateResult + thumbnail_url + session_id | Once per candidate as it renders |
| `done` | `{best_id}` | After all candidates streamed |
| `warning` | `{detail}` | e.g. CVAE not trained yet |
| `error` | `{detail}` | Exception during sampling |

### Thumbnail URL

`thumbnail_url = /api/generate/thumbnail/{session_id}/{candidate_id}`

The PNG is stored in an in-process dict `_thumbnail_cache[(session_id, id)]`. The browser fetches it separately after receiving the candidate event. Session is kept for 5 minutes then evicted.

---

## Full end-to-end — sample mode

```mermaid
flowchart LR
    U["User\nGenerate button"]
    R["POST /api/generate\n{domain, target, n, method}"]
    S["session_id = uuid4()"]
    P1["event: progress\n'Sampling from CVAE...'"]
    LHS2["LHS → z [n,16]"]
    DEC3["CVAE decoder\nz → params [n,4]"]
    RM["RealMeshLookup\nparams → real mesh graph × n"]
    SC["DragSurrogate\nparams → drag_pred × n"]
    OD["ParamSpaceOOD\nparams → confidence × n"]
    SO["sort by |drag_pred - target|"]
    TH["ThumbnailRenderer\ngraph → PNG × n"]
    C1["event: candidate {0}"]
    C2["event: candidate {1}"]
    CN["event: candidate {n-1}"]
    DN["event: done {best_id}"]

    U --> R --> S --> P1 --> LHS2 --> DEC3 --> RM & SC & OD --> SO --> TH --> C1 --> C2 --> CN --> DN
```

## Full end-to-end — gradient mode

```mermaid
flowchart LR
    U2["User\nGenerate button"]
    R2["POST /api/generate\n{method=gradient}"]
    P2["event: progress\n'Optimising in latent space...'"]
    Z0["z = randn([16])"]
    OPT["Adam 150 iters × 3 restarts\nz → decoder → surrogate → loss → ∂z"]
    BZ["best_z*"]
    DIV["n perturbed z around best_z*"]
    DEC4["CVAE decoder\nz_i → params_i [4] × n"]
    RM2["RealMeshLookup\nparams → real mesh × n"]
    SC2["DragSurrogate\nparams → drag_pred × n"]
    OD2["ParamSpaceOOD\nparams → confidence × n"]
    TR["event: trajectory\n{values: loss curve}"]
    TH2["ThumbnailRenderer → PNG × n"]
    CD["event: candidate × n"]
    DN2["event: done"]

    U2 --> R2 --> P2 --> Z0 --> OPT --> BZ --> DIV --> DEC4 --> RM2 & SC2 & OD2 --> TR --> TH2 --> CD --> DN2
```
