---
tags: [physicsai, subsystem, deep-dive]
created: 2026-04-10
aliases: [predictor, meshgraphnets, simulator]
---

# Predictor Subsystem — Deep-Dive Technical Reference

> [!info] Audience
> You know what an MLP, a loss function, backprop, and gradient descent are.
> You do **not** need to know graph neural networks, message passing, or physics simulation.
> Everything is introduced from scratch.

## Quick summary

- Replaces expensive PDE solvers with a single neural network forward pass
- Uses **message passing** — nodes exchange 128-dim vectors with neighbours, 15 rounds
- Two variants: `Simulator` (CFD, velocity/pressure) and `FlagSimulator` (cloth, 3D positions)
- Predicts **change** (Δv or acceleration), not absolute values — more stable training
- Cloth uses Verlet integration; HANDLE nodes are pinned as boundary conditions

---

## Table of Contents

1. [What this subsystem does](#1-what-this-subsystem-does)
2. [The core problem: why normal neural networks don't work here](#2-the-core-problem-why-normal-neural-networks-dont-work-here)
3. [Message passing — the key idea](#3-message-passing--the-key-idea)
4. [Architecture: EncoderProcessorDecoder](#4-architecture-encoderprocessordecoder)
5. [CFD Simulator in detail](#5-cfd-simulator-in-detail)
6. [Cloth FlagSimulator in detail](#6-cloth-flagsimulator-in-detail)
7. [Code snippets](#7-code-snippets)
8. [Normalizers](#8-normalizers)
9. [Tradeoffs](#9-tradeoffs)
10. [Potential enhancements](#10-potential-enhancements)

---

## 1. What this subsystem does

The predictor subsystem is the heart of MeshGraphNets. Given a snapshot of a physical simulation — either a fluid-dynamics (CFD) mesh with per-node velocities and pressures, or a cloth mesh with per-vertex 3D positions — it produces the **next** state of every node in the mesh one timestep later. It does this by learning from thousands of ground-truth simulation trajectories: rather than solving the underlying partial differential equations (which is expensive), the network learns to *imitate* the simulator's output. At inference time a single forward pass replaces a costly numerical solver, enabling near-real-time rollouts over hundreds of timesteps.

---

## 2. The core problem: why normal neural networks don't work here

A standard MLP requires a **fixed-size input vector**. If you tried to feed a mesh directly into an MLP you would immediately hit a wall: every mesh in the dataset has a different number of nodes (and therefore a different-size input). A 2 000-node airfoil mesh and a 12 000-node flag mesh cannot share the same linear layer — the weight matrix dimensions would have to change per sample.

You might think "just pad to the maximum mesh size." Two problems:

1. **Wasteful and brittle.** Most entries would be zero-padding, and you'd need to decide a hard maximum forever.
2. **No spatial awareness.** Even if you concatenated all node features into one giant vector, the MLP would have no idea that node 42 is *adjacent* to node 43 and completely disconnected from node 9 000. Physical laws are *local*: a node's velocity next timestep depends mostly on its neighbours, not on distant parts of the mesh.

**Graphs are the solution.** A graph is just a set of nodes and a set of edges that say which nodes are connected. Crucially, the graph structure is *separate* from the data tensors — you can have a 2 000-node graph and a 12 000-node graph and process both with the **same** neural network weights, because the weights operate per-node and per-edge, not on the whole mesh at once. The network learns *local interaction rules* that apply everywhere on the mesh regardless of total size.

---

## 3. Message passing — the key idea

Imagine a large company where every employee (node) knows their own workload (feature vector). Management wants each person to update their self-assessment based on what their direct colleagues are experiencing. They run the following protocol:

> [!note] Round 1
> Every employee writes a memo to each of their direct contacts. The memo contains their own current state *and* some description of their relationship (e.g. "we share project X"). Each recipient reads all incoming memos, **sums** them up, and uses that aggregate plus their own state to update their self-assessment.

Run that 15 times. After 15 rounds, each employee's state has been influenced not just by their immediate neighbours, but transitively by everyone within 15 hops. In a mesh where physical effects travel a few edges per timestep, 15 rounds is enough to propagate information across the length of the domain.

That is **message passing** — or in the literature, a Graph Network (GN) step. In code:

- A **message** is a **vector** (128 numbers). It is produced by an MLP that takes as input: the sender's latent state, the receiver's latent state, and the edge's latent state — all concatenated.
- **Aggregation** means summing all incoming messages at each node (implemented via `scatter(..., reduce='sum')` in `blocks.py`).
- The node then updates its own latent state by running another MLP on `[its_own_state || aggregated_messages]`.
- Both node latents and edge latents are updated every round. After each round the new values are **added** to the old values (residual connection), so the network can learn incremental refinements.

Nothing about this protocol requires knowing the total number of nodes in advance — the same MLP weights process every node and every edge, making the whole thing size-agnostic.

---

## 4. Architecture: EncoderProcessorDecoder

Every simulator in this codebase wraps a single class: `EncoderProcesserDecoder` (`model/model.py`). It has three stages.

```
Raw features                 Latent space (128-dim)              Predictions
─────────────────────────────────────────────────────────────────────────────

  Nodes: [N, F_n]              Nodes: [N, 128]         Nodes: [N, output_size]
  Edges: [E, F_e]              Edges: [E, 128]
  Graph structure              Graph structure

       │                            │                          │
       ▼                            ▼                          ▼
  ┌─────────┐   ──────────►   ┌───────────┐  ──────────►  ┌─────────┐
  │ ENCODER │                 │ PROCESSOR │                │ DECODER │
  │  2-layer│                 │ 15× GnBlock               │ 2-layer │
  │   MLP   │                 │ (message  │                │   MLP   │
  │ per node│                 │ passing)  │                │ no LayerNorm
  │ per edge│                 └───────────┘                └─────────┘
  └─────────┘
  + LayerNorm                 Each GnBlock:
                              1. EdgeBlock: MLP(sender||receiver||edge) → new edge
                              2. NodeBlock: MLP(node||Σ_edges) → new node
                              3. Residual add: new = old + update
```

**Stage 1 — Encoder.** Two independent MLPs (one for nodes, one for edges), each with 2 hidden layers of 128 units + LayerNorm, project raw features into a uniform 128-dimensional latent space. After this step every node and every edge carries exactly 128 numbers regardless of how many raw features it started with.

**Stage 2 — Processor.** A sequence of 15 identical `GnBlock` modules (each with its own learned weights) runs message passing. In each block: edges are updated first (they "observe" both endpoint states), then nodes aggregate the updated messages, then residual connections preserve gradient flow. Two alternative processor architectures (`TNSBlock`, `SAGEBlock`) are available — see Section 4b.

**Stage 3 — Decoder.** A single MLP projects each node's final 128-dim latent to the output size (2 for CFD velocity, 1 for CFD pressure, 3 for cloth position). No LayerNorm here — the raw output is a predicted normalized change, which is then de-normalized to physical units.

---

## 4b. Processor Architecture Variants

`model/simulator.py` `__init__` accepts an `architecture='gn'/'tns'/'sage'` parameter that controls which block type fills the processor. The architecture name is stored inside the checkpoint so that the stale-index check can verify architecture consistency.

The Train page UI exposes GN / TNS / SAGE buttons. The Predict page shows an architecture selector — each option shows the best checkpoint (lowest validation loss) trained with that architecture, and is disabled if no checkpoint has been trained for it yet.

### GnBlock (default — Graph Network)

The standard block described in Section 4. Sum aggregation over all incoming edges. Best default for production use: stable gradients, well-understood failure modes, no special training setup required. Learning rate `1e-4`, no gradient clipping needed.

### TNSBlock (Transformer-based Node-update)

Replaces scatter-sum aggregation with **multi-head attention** over incoming edge embeddings:

- **Q** = current node embedding, **K/V** = incoming edge embeddings
- The node learns *which* neighbours to attend to rather than summing all equally
- Attention weights are interpretable: a high weight on edge (j → i) means "node j's message strongly influenced node i's update at this step"

**Problem:** Attention over variable-degree neighbourhoods causes gradient scale issues — nodes with many neighbours (high-degree interior mesh nodes) accumulate attention logits across many keys, producing different gradient magnitudes than low-degree nodes. This makes training unstable with standard learning rates.

**Required training adjustments:**
- Clipped learning rate: `3e-5` (vs `1e-4` for GN)
- Gradient clipping: `clip_grad_norm_(parameters, max_norm=1.0)`

**When to prefer TNS over GN:**
- Long-range dependencies where knowing *which* edges dominate matters
- Interpretability use cases — attention weights directly answer "which neighbouring nodes drove this update?"
- Ablation studies on attention vs sum aggregation

Lives in `model/model.py` as `TNSBlock`.

### SAGEBlock (GraphSAGE-style)

**Mean aggregation**: sum incoming neighbour messages, then divide by the node's degree, then concatenate with the node's own state and project:

```
msg_agg = sum(neighbour_msgs) / degree(node)
v'_i    = Linear([v_i || msg_agg])
```

**Why mean instead of sum:** Cloth meshes have highly variable node valence — corner nodes have 2–3 edges, interior nodes have 6 or more. Sum aggregation gives high-degree interior nodes disproportionately large aggregated signals, causing those nodes to dominate gradient updates and making the network inconsistently sensitive to mesh resolution. Mean aggregation normalises by degree, so every node's update signal is on the same scale regardless of its number of neighbours.

**Properties:**
- More parameter-efficient than GnBlock (no separate edge update MLP — edge features are used directly as messages)
- More gradient-stable than TNSBlock (no attention softmax over variable-length sequences)
- Best suited to cloth / irregular mesh domains where valence variance is high

Lives in `model/model.py` as `SAGEBlock`.

---

## 5. CFD Simulator in detail

**File:** `model/simulator.py` — class `Simulator`

### Node features

| Component | Dims | Description |
|-----------|------|-------------|
| Velocity (or pressure) | 2 (or 1) | Current field value at node |
| One-hot node type | 9 | Encodes NORMAL, OBSTACLE, AIRFOIL, INFLOW, OUTFLOW, WALL_BOUNDARY, etc. |
| **Total** | **11** (velocity) / **10** (pressure) | Input to node encoder MLP |

The one-hot encoding is critical: boundary nodes (inflow, walls, obstacles) obey different physics rules than interior fluid nodes. Without it, the network would have to infer boundary conditions from context alone.

### Edge features

Edges connect mesh nodes that share a mesh element face. Each directed edge carries:

| Component | Dims | Description |
|-----------|------|-------------|
| Δx, Δy | 2 | Relative position: `pos[sender] - pos[receiver]` |
| Distance | 1 | `‖Δx, Δy‖` |
| **Total** | **3** | Input to edge encoder MLP |

Using *relative* positions (not absolute coordinates) makes the network translation-invariant: it generalises to meshes placed anywhere in space.

### Output

| `target_field` | Output shape | Meaning |
|----------------|-------------|---------|
| `"velocity"` | `[N, 2]` | 2D velocity field (u, v) |
| `"pressure"` | `[N, 1]` | Scalar pressure field |

### Verlet-style prediction: predicting Δv, not v

The network does **not** predict the next velocity directly. It predicts the *change* (Δv = v_{t+1} − v_t), which is numerically much smaller and smoother than the raw velocity, making the regression task easier. At inference:

```python
delta      = output_normalizer.inverse(predicted_norm)  # de-normalize
next_value = frames + delta                             # Verlet: v_{t+1} = v_t + Δv
```

### Training noise injection for stability

A known failure mode of one-step-trained simulators is **compounding error**: small mistakes at each step accumulate into large drift over long rollouts. To mitigate this, Gaussian noise is added to the input velocity *during training only*:

```python
noised_frames = frames + velocity_sequence_noise   # training only
```

The noise is sampled from a zero-mean Gaussian and zeroed out on non-NORMAL nodes (boundaries must stay exact). This forces the network to learn to recover from small perturbations — a form of data augmentation that simulates the distribution shift encountered during autoregressive rollouts.

---

## 6. Cloth FlagSimulator in detail

**File:** `model/flag_simulator.py` — class `FlagSimulator`

The cloth simulator operates in 3D and uses a richer feature set to capture both the *current deformed shape* (world space) and the *rest shape* (mesh space).

### Node features

| Component | Dims | Description |
|-----------|------|-------------|
| Velocity | 3 | `world_pos_t − world_pos_{t−1}` (finite-difference velocity in 3D) |
| One-hot node type | 9 | NORMAL (free cloth) vs HANDLE (pinned corners) |
| **Total** | **12** | |

Note that "velocity" here is computed on-the-fly from two consecutive position snapshots — there is no explicit velocity field stored in the dataset.

### Edge features

The flag simulator uses **two coordinate systems** simultaneously:

| Component | Dims | Description |
|-----------|------|-------------|
| `rel_world` | 3 | `world_pos[sender] − world_pos[receiver]` — deformed 3D geometry |
| `‖rel_world‖` | 1 | Euclidean norm in world space |
| `rel_mesh` | 2 | `mesh_pos[sender] − mesh_pos[receiver]` — rest-shape 2D UV coordinates |
| `‖rel_mesh‖` | 1 | Euclidean norm in mesh space |
| **Total** | **7** | |

The mesh-space features encode the **rest length** of each spring (cloth edge). The difference between rest length and current world-space length is what generates elastic restoring forces in real cloth — by providing both, the network can learn to approximate this relationship without any explicit physics formula.

### Verlet integration

The training target is **acceleration** in the Verlet sense:

```
acc = pos_{t+1} − 2·pos_t + pos_{t−1}
```

This is the second finite difference of position — equivalent to `(pos_{t+1} − pos_t) − (pos_t − pos_{t−1})`, i.e. the *change in velocity*. At inference, the inverse Verlet update reconstructs the next position:

```python
acc            = output_normalizer.inverse(predicted_norm)
next_world_pos = 2.0 * world_pos - prev_world + acc
```

This is the discrete analogue of `x_{t+1} = 2x_t − x_{t−1} + a·Δt²` from classical mechanics.

### Node pinning (HANDLE boundary condition)

The flag's corners are pinned — they are animated externally (the flag is being waved) and must not be moved by the physics network. After computing `next_world_pos` for all nodes, the pinned nodes are overwritten:

```python
handle_mask    = (node_type_col != NodeType.NORMAL).unsqueeze(-1)  # [N, 1]
next_world_pos = torch.where(handle_mask, world_pos, next_world_pos)
```

`torch.where(mask, a, b)` selects `a` where the mask is True, `b` elsewhere. So pinned nodes keep `world_pos` (their current position), while free nodes get the network's predicted `next_world_pos`. This is equivalent to a Dirichlet boundary condition.

---

## 6b. Poisson Pressure Correction

### The Problem

After each GNN rollout step, the predicted velocity field may not exactly satisfy the **incompressibility constraint** ∇·u = 0. A real fluid has zero velocity divergence everywhere (mass is conserved — no fluid is created or destroyed at any node). The GNN approximates this but does not enforce it exactly. Small divergence errors at each step **accumulate over a 600-step rollout**, eventually producing unphysical results: drift in the pressure field, spurious sources/sinks of fluid, and velocity fields that no longer resemble realistic flow.

### The Fix: Helmholtz-Hodge Decomposition

Any vector field can be uniquely decomposed into a divergence-free part and a curl-free (gradient) part:

```
u = u_div_free + ∇φ
```

To project u onto the divergence-free subspace:
1. Solve the **Poisson equation**: ∇²φ = ∇·u
2. Subtract the gradient: u_corrected = u − ∇φ

By construction, ∇·u_corrected = 0.

### Discretisation

On the mesh graph, the Poisson equation is discretised as a **sparse linear system**:

```
L φ = b
```

where **L** is the graph Laplacian built from the 5-NN neighbourhood (same neighbourhood used for edges) and **b** is the discrete divergence of the predicted velocity at each node.

### Implementation in `physics/poisson_pressure.py`

The key computational trick is to **factorize the Laplacian once per rollout** using a sparse LU decomposition, then solve cheaply at each step:

```python
class PoissonPressureCorrector:
    def __init__(self, graph):
        L = build_graph_laplacian(graph)          # sparse [N, N]
        self.lu = scipy.sparse.linalg.splu(L)     # O(N^1.5) — done ONCE

    def correct(self, velocity: torch.Tensor) -> torch.Tensor:
        b = compute_divergence(velocity, self.graph)   # O(N)
        phi = torch.from_numpy(self.lu.solve(b.numpy()))  # O(N) — fast solve
        return velocity - compute_gradient(phi, self.graph)
```

**Complexity:** `splu` factorisation is O(N^1.5) and runs once at rollout start. Each of the 600 `lu.solve(b)` calls is O(N). Total overhead is dominated by the factorisation, not the per-step solves.

### Wiring

- `physics/poisson_pressure.py` — `PoissonPressureCorrector` class
- `rollout.py` — correction applied after each GNN step when enabled
- `api/routes/rollout.py` — same, for the API rollout path

### Activation

| Interface | How to enable |
|-----------|--------------|
| CLI | `--poisson_correction` flag |
| UI | Checkbox on the CFD rollout panel |

**Default: OFF.** The correction is not enabled by default for three reasons:
1. **Overhead:** adds ~10–15% to rollout time (dominated by the O(N^1.5) LU factorisation).
2. **Domain restriction:** assumes incompressibility — not valid for cloth, combustion, or any compressible-flow domain.
3. **Conservative default:** most training trajectories were generated with a solver that already enforces incompressibility; the GNN typically produces small enough divergence errors that correction does not change results visibly for short rollouts.

Enable it for long CFD rollouts (400+ steps) where pressure drift is noticeable.

---

## 7. Code snippets

### 7a. `update_node_attr` — building node features for CFD

```python
# simulator.py
def update_node_attr(self, frames: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """
    frames: [N, 2] velocity  OR  [N, 1] pressure
    types:  [N, 1] integer node-type indices
    Returns: normalized node features [N, 11] or [N, 10]
    """
    node_type = types.squeeze(-1).long()                         # [N]
    one_hot   = torch.nn.functional.one_hot(node_type, num_classes=9)  # [N, 9]
    node_feats = torch.cat([frames, one_hot.float()], dim=-1)    # [N, 11] or [N, 10]
    return self._node_normalizer(node_feats, self.training)
```

### 7b. `forward` — training vs. inference split

```python
# simulator.py (abbreviated)
def forward(self, graph, velocity_sequence_noise):
    node_type = graph.x[:, 0:1]
    frames    = graph.x[:, self._frames_slice()]   # velocity or pressure

    if self.training:
        noised_frames = frames + velocity_sequence_noise   # inject noise
        graph.x       = self.update_node_attr(noised_frames, node_type)
        graph.edge_attr = self.edge_normalizer(graph.edge_attr, self.training)

        predicted_norm     = self.model(graph)
        target_change      = self.velocity_to_acceleration(noised_frames, graph.y)
        target_change_norm = self._output_normalizer(target_change, self.training)
        return predicted_norm, target_change_norm   # caller computes MSE loss

    else:  # inference
        graph.x       = self.update_node_attr(frames, node_type)
        graph.edge_attr = self.edge_normalizer(graph.edge_attr, self.training)

        predicted_norm = self.model(graph)
        delta          = self._output_normalizer.inverse(predicted_norm)
        next_value     = frames + delta              # v_{t+1} = v_t + Δv
        return next_value
```

### 7c. `_frames_slice` — which columns hold the physical field

```python
# simulator.py
def _frames_slice(self) -> slice:
    if self.target_field == "pressure":
        return slice(1, 2)   # graph.x[:, 1:2] — pressure  [N, 1]
    return slice(1, 3)       # graph.x[:, 1:3] — velocity  [N, 2]
    # column 0 is always the integer node type
```

### 7d. `FlagSimulator._build_graph` — cloth edge feature construction

```python
# flag_simulator.py
def _build_graph(self, graph: Data) -> Data:
    graph = self._face_to_edge(graph)       # convert triangle faces → directed edges
    senders, receivers = graph.edge_index[0], graph.edge_index[1]

    mesh_pos  = graph.pos        # [N, 2]  — rest shape (UV)
    world_pos = graph.world_pos  # [N, 3]  — current deformed shape

    rel_world  = world_pos[senders] - world_pos[receivers]   # [E, 3]
    world_norm = torch.norm(rel_world, dim=-1, keepdim=True)  # [E, 1]
    rel_mesh   = mesh_pos[senders]  - mesh_pos[receivers]    # [E, 2]
    mesh_norm  = torch.norm(rel_mesh,  dim=-1, keepdim=True)  # [E, 1]

    # Final edge feature: [rel_world(3) | |world|(1) | rel_mesh(2) | |mesh|(1)] = 7 dims
    graph.edge_attr = torch.cat([rel_world, world_norm, rel_mesh, mesh_norm], dim=-1)
    return graph
```

### 7e. Verlet integration at inference (FlagSimulator)

```python
# flag_simulator.py — inference branch
acc            = self._output_normalizer.inverse(predicted_acc_norm)  # de-normalize
next_world_pos = 2.0 * world_pos - prev_world + acc                   # Verlet step

# Pin HANDLE nodes (animated flag corners) to their current position
handle_mask    = (node_type_col != NodeType.NORMAL).unsqueeze(-1)     # [N, 1] bool
next_world_pos = torch.where(handle_mask, world_pos, next_world_pos)
```

---

## 8. Normalizers

### Why not just compute mean/std once from the dataset?

You could compute dataset-wide statistics offline and bake them in. But this implementation uses **online running-statistics normalizers** (`utils/normalization.py`) instead. Here's why that matters:

**What `Normalizer` does:**
- Maintains running accumulators: `_acc_sum`, `_acc_sum_squared`, `_acc_count` — all registered as `nn.Module` buffers so they are saved/loaded with the model checkpoint.
- On each forward call during training (`accumulate=True`), it updates these buffers from the current batch.
- Normalizes as: `(x − mean) / max(std, ε)` where ε = 1e-8 prevents division by zero.
- Provides an `inverse()` method to undo the normalization.
- Stops accumulating after 10⁶ updates (a safety cap to prevent numerical precision issues with very large sums).

**Why this is better than fixed normalization:**

| Concern | Fixed stats | Running stats |
|---------|------------|---------------|
| Dataset statistics unavailable at init | Must pre-compute a pass | No pre-pass needed |
| Different normalization per feature dim | Same | Yes — each dim gets its own mean/std |
| Portable with checkpoint | Must save separately | Saved inside the model |
| Fine-tuning on a new mesh distribution | Stats are stale | Adapts automatically (up to cap) |

There are **three separate normalizers** per simulator: one for node features, one for edge features, and one for output (predicted changes). This is important because these three quantities live in completely different numerical ranges — velocities might be O(1), edge distances O(0.01), and acceleration targets O(0.001).

---

## 9. Tradeoffs

1. **Fixed 128-dim latent, fixed 15 steps — both are heuristics.** The choice of 128-dimensional latent space and 15 message-passing rounds comes from the original DeepMind paper and works well on their benchmarks, but there is no principled derivation. A coarser mesh might need only 8 steps; a very fine mesh with long-range physics might need 30. Both are hyperparameters that require ablation per domain.

2. **One-step training, multi-step rollout mismatch.** The model is trained to predict one step at a time with ground-truth inputs, but at inference it feeds its own predictions back as inputs. This **distribution shift** causes error accumulation — the model sees slightly wrong inputs at step 2, worse inputs at step 3, and so on. Noise injection during training partially mitigates this but does not eliminate it.

3. **No temporal context beyond two frames.** The CFD simulator uses only the current frame; the cloth simulator uses only the current and previous frame (to compute velocity). Neither model has memory of what happened three or more steps ago. Vortices that develop over many timesteps, or resonant oscillation modes, may be poorly captured.

4. **GN sum aggregation is mesh-resolution sensitive; TNS attention is gradient-unstable; SAGE mean is the pragmatic middle ground.** GnBlock sums all neighbours equally — coarsely-meshed regions with fewer neighbours get a weaker aggregated signal than finely-meshed regions, which can cause the network to behave inconsistently across non-uniform meshes. TNSBlock solves the weighting problem via attention but introduces gradient scale instability in variable-degree neighbourhoods, requiring a 3× lower learning rate and explicit gradient clipping — making it slower to converge and harder to tune. SAGEBlock's mean aggregation normalises by degree and is stable at standard learning rates, but discards edge-update information that GnBlock preserves. **GN is the production default** because its failure modes (resolution sensitivity) are well-understood and manageable; TNS's gradient instability is harder to diagnose and more likely to cause silent training divergence on new datasets.

5. **Static graph topology.** The edge set is built once from the mesh connectivity at the start of each rollout and never changes. For the cloth simulator this means the rest-shape mesh edges are permanent — the model cannot simulate tearing, folding contact, or self-collision where new contacts form between previously disconnected nodes.

6. **Node type vocabulary is fixed at 9.** The `one_hot(node_type, num_classes=9)` encoding is hard-coded. Adding a new boundary condition type (e.g. a porous membrane) requires retraining from scratch with a modified feature size — there is no mechanism to extend the vocabulary without architectural changes.

7. **All normalizer statistics are global across the rollout.** The same normalizer applies to timestep 1 and timestep 500. If the physical state drifts significantly (e.g. a cloth that has fallen out of frame), the normalizer statistics from early training may become poorly calibrated for the late-rollout distribution.

---

## 10. Potential Enhancements

1. **Multi-step rollout training (unrolling).** Instead of always training with ground-truth inputs, periodically unroll the model for K steps and backpropagate through time. This directly penalizes compounding error and teaches the model to produce states that are *stable inputs* for itself. Cost: K× memory and compute per update, but substantially better long-rollout fidelity.

2. **Adaptive message passing depth.** Use an early-exit mechanism: after each GnBlock, compute a convergence criterion (e.g. ‖new_x − old_x‖ < threshold). For simple timesteps (small changes) the forward pass terminates in fewer than 15 steps, saving compute. For challenging timesteps it can run more. This connects to ideas in adaptive computation time (ACT).

3. **Temporal attention over last K frames.** Replace the two-frame velocity estimate with a small transformer (or GRU) that reads the last K node states. This gives the model a richer notion of "is this node accelerating or decelerating?" — information that is critical near flow separation zones or for cloth oscillation modes. The temporal window K is a new hyperparameter, but even K=4 could capture meaningful momentum.

4. **Attention-based message aggregation.** Replace the sum aggregation in `NodeBlock` with a learned dot-product attention over incoming messages. This allows the node to weight nearby influential neighbours more heavily than distant or irrelevant ones — analogous to how graph attention networks (GATs) work. Particularly useful for irregular meshes where node degree varies widely.

5. **Physics-informed loss terms.** Add soft physics constraints to the loss: e.g. penalise violations of mass conservation (divergence of velocity field for CFD), or penalise cloth edge lengths that stretch far beyond their rest lengths. These terms do not require labelled data — they are computed from the model's own predictions — and can meaningfully regularize the model in data-sparse regions.

6. **Hierarchical multi-resolution processing.** Coarsen the mesh into a hierarchy (fine → medium → coarse), run message passing at each level, and upsample back. Long-range information that takes 15 hops to propagate on the fine mesh can be communicated in 3–4 hops on the coarse mesh. This is the spatial analogue of a U-Net and can dramatically reduce the steps needed for accurate long-range propagation.

7. **Mesh-adaptive training with variable noise schedules.** The current noise injection uses a fixed standard deviation for all nodes and all timesteps. A curriculum that starts with high noise (aggressive regularization) and anneals to near-zero noise as training progresses — inspired by diffusion model schedules — could give the network a smoother learning signal early while converging to high-fidelity predictions late in training.

8. **GNN explainability via `torch_geometric.explain`.** PyG 2.3+ ships a first-class explainability package that can produce node masks and edge masks — indicating which parts of the mesh most influenced a given prediction. Algorithms like `GNNExplainer`, `PGExplainer`, and `CaptumExplainer` are supported.

   **Important caveat for this project:** The `GNNExplainer` and `PGExplainer` algorithms hook into PyG's `MessagePassing` base class to intercept intermediate representations. This project's `EdgeBlock` and `NodeBlock` are plain `nn.Module` subclasses — they do message passing *conceptually* (nodes exchange information along edges via `scatter`) but do **not** subclass `torch_geometric.nn.MessagePassing`. This means `GNNExplainer` cannot hook into the forward pass without first refactoring the blocks to inherit from `MessagePassing`.

   Two paths forward:

   - **Refactor `EdgeBlock` / `NodeBlock` to subclass `MessagePassing`.** The math is identical — the only change is moving the `scatter` call into `aggregate()` and the MLP into `message()` / `update()`. This unlocks the full `torch_geometric.explain` ecosystem with no changes to the model behaviour.

   - **Use `CaptumExplainer` instead.** Captum's attribution methods (Integrated Gradients, GradientShap) work on any `nn.Module` — no `MessagePassing` required. They compute feature importance by measuring how much each input feature (node position, node type, edge length) contributes to the output, via gradient attribution. This is the lower-friction path:

   ```python
   from torch_geometric.explain import Explainer, CaptumExplainer

   explainer = Explainer(
       model=simulator,
       algorithm=CaptumExplainer('IntegratedGradients'),
       explanation_type='model',
       node_mask_type='attributes',
       edge_mask_type='object',
       model_config=dict(
           mode='regression',
           task_level='node',
           return_type='raw',
       ),
   )
   explanation = explainer(graph.x, graph.edge_index)
   explanation.visualize_graph()
   ```

   **When this is most useful:** On a steady-state SER model predicting drag directly from geometry in a single forward pass (see [[neural-simulator-model-types]]). GNNExplainer on that model would reveal which mesh regions drive the drag prediction — actionable design sensitivity information. On the transient rollout predictor, every edge participates in propagating the PDE at every step, so edge masks mostly highlight the cylinder and wake — regions you already knew were important.

   > [!info] API stability warning
   > The `torch_geometric.explain` API is still evolving — PyG documents it as subject to change. Pin your PyG version if you depend on specific explainer behaviour.

## See also

- [[SUBSYSTEM_DATA]] — feeds training data into this predictor
- [[SUBSYSTEM_CONFIDENCE]] — uses this predictor's encoder embeddings to score OOD distance
- [[SUBSYSTEM_GENERATOR]] — generates candidate inputs that are fed into this predictor for verification
