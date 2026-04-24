---
tags: [physicsai, subsystem, deep-dive]
created: 2026-04-10
aliases: [generator, cvae, inverse-design]
---

# Generator Subsystem — Deep-Dive Technical Reference

## Quick summary

- Learns the *inverse* mapping: physics target → design parameters
- Phase 0 extracts (cx, cy, r, v_inlet) from CFD data via algebraic circle fit
- CVAE encoder maps design+physics → latent z[16]; decoder maps z+target → design
- Surrogate MLP (~1000× faster than rollout) provides physics signal during CVAE training
- Latent-space sampling uses **Latin Hypercube Sampling** (not pure random) to avoid clustering
- Gradient descent in latent space works for cloth (differentiable chain) **and** CFD (via `RealMeshLookup`)
- CFD gradient refinement uses K=5 BPTT through GNN steps, not Delaunay

**Files covered:** `shape_extractor.py`, `drag_surrogate.py`, `cvae_cfd.py`,
`cloth_extractor.py`, `cvae_cloth.py`, `mesh_generator.py`, `inverse_design.py`

---

## 1. What This Subsystem Does

The Generator subsystem solves the **inverse design problem**: given a desired
physics outcome (a target drag force for a CFD channel flow, or a target
deformation stress for a draped cloth), generate a concrete physical design
that achieves it. This is the reverse of everything else in PhysicsAI — the
predictor subsystem starts from a shape and rolls forward in time; the
generator subsystem starts from a physics *goal* and works backward to a
shape. The result is a closed design loop: a user specifies what they want the
physics to do, and the system proposes geometries or configurations that meet
that specification, ready to be fed straight into the MeshGraphNets predictor
for verification.

---

## 2. The Inverse Design Problem

The natural direction for a neural network is **forward**: given a design, predict
the resulting physics. The predictor does exactly this — encode a mesh, run
message-passing, decode velocities or accelerations.

Inverse design asks the opposite question: *given a desired physics outcome,
what design should I use?* The naive approach — just invert the network — fails
for a fundamental reason: **the mapping from physics to design is one-to-many**.
Many different cylinder radii, positions, and inlet velocities can all produce
the same drag value. If you ask "what design gives drag = 0.03?", there is no
single correct answer; there is an entire family of valid answers. Trying to
train a plain MLP to predict a design from a drag value forces the network to
average over all valid answers, producing a blurry, physically implausible
output.

This is precisely the problem that **Variational Autoencoders (VAEs)** were
designed to solve. A VAE does not map a condition to one point — it maps a
condition to a *distribution* over plausible answers. Sampling from that
distribution gives you different valid designs, all consistent with the physics
target. The Conditional VAE (CVAE) extends this by making the distribution
explicitly conditioned on the target physics value, so every sample is
steered toward the goal.

---

## 3. Phase 0: Data Extraction — What Design Parameters Are

Before training any generative model, each trajectory in the dataset must be
reduced to a compact, fixed-length vector that describes the design. This is
the role of `shape_extractor.py` (CFD) and `cloth_extractor.py` (cloth).

### CFD: Circle Fit + Inlet Velocity

A cylinder_flow trajectory has thousands of mesh nodes, but the underlying
design is described by just four numbers: **`(cx, cy, r, v_inlet)`** — the
cylinder centre, radius, and inlet velocity.

**Extracting `(cx, cy, r)` — the Kåsa algebraic circle fit.**
The cylinder surface nodes are labelled `WALL_BOUNDARY` in the dataset, but
so are the top and bottom channel walls. The extractor first discards any
`WALL_BOUNDARY` node that lies within `1e-3` mesh units of the domain boundary
edges, leaving only the cylinder surface nodes. It then fits a circle to those
points using the Kåsa algebraic method.

The geometric identity `(x - cx)² + (y - cy)² = r²` can be expanded and
rearranged into a linear system:

```
A @ [2·cx, 2·cy, cx² + cy² - r²]ᵀ  =  x² + y²

where  A = [x₁  y₁  1]
           [x₂  y₂  1]
           [ ⋮   ⋮  ⋮]
```

Solving this with `numpy.linalg.lstsq` gives the three unknowns in one shot.
The radius is recovered as `r = sqrt(result[2] + cx² + cy²)`.

**Extracting `v_inlet`.**
`INFLOW` nodes (left boundary of the channel) carry the inlet velocity at
`t = 0`. The extractor takes the mean x-velocity across all inflow nodes:

```python
v_inlet = np.mean(velocity[inflow_mask, 0])
```

The result is a `[N_traj, 4]` array — one compact design vector per trajectory,
saved as `design_params.npy`.

### Cloth: PCA on Initial World Positions

The cloth (flag_simple) dataset has a different structure: the mesh **topology**
is fixed (always 1579 nodes, always the same triangles), but the initial
**world position** — how the cloth is draped in 3D space — varies across
trajectories. Each initial pose is a `[1579, 3]` array, which flattened gives
`4737` numbers. That is too many to use directly as a design vector.

The key insight is that these 4737 numbers are not independent. A cloth can
only drape in a limited number of physically plausible ways; most of the
variation is captured by a low-dimensional subspace.

**PCA** finds that subspace. Conceptually: given 1200 training cloth poses,
PCA finds the 16 directions in 4737-dimensional space along which the poses
vary the most. The first component might capture "how much the cloth sags
overall"; the second might capture "how much it twists to the left vs. right";
and so on. With 16 components, the reconstruction explains >95% of the total
variance across all training poses.

```python
# cloth_extractor.py — PosePCA.fit()
X_c = X - X.mean(axis=0)          # centre the data
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
components = Vt[:16]               # [16, 4737]  top-16 directions

# transform: [1, 4737] → [1, 16]
pose_pca = (world_pos_flat - mean) @ components.T
```

Each trajectory is now represented as a `[16]` PCA coefficient vector, stored
in `cloth_pose_pca.npy`. The `PosePCA` object (components + mean) is saved
separately because it is needed at generation time to invert the transform.

---

## 4. The Surrogate Model — The Fast Physics Proxy

### What It Is

`DragSurrogate` is a 3-layer MLP that maps the four CFD design parameters
`(cx, cy, r, v_inlet)` to a single scalar `drag_proxy`:

```
Input [4] → FC(4→64) → ReLU → FC(64→64) → ReLU → FC(64→1) → drag_proxy
```

### Why It Exists

Training the CVAE requires a **physics consistency loss** at every gradient
step: the designs the CVAE generates must actually achieve the target drag.
Computing true drag requires running a full MeshGraphNets rollout — build the
mesh, execute 15 message-passing steps across thousands of edges, accumulate
the pressure field, integrate drag over the cylinder surface. That takes
roughly 100 ms per call. With batch size 64 and thousands of epochs, that is
days of training time.

The surrogate runs in **microseconds** — roughly 1000× faster — enabling the
physics loss to be evaluated at every training step without any special
hardware.

### The Drag Proxy Formula

The surrogate is not arbitrary; it is pre-trained to fit the analytical
**blockage-corrected drag proxy**:

```
drag_proxy = r × v_inlet² / (1 − 2r/H)
```

Each term has physical meaning:

| Term | What it represents |
|------|-------------------|
| `r` | Obstacle size — a larger cylinder blocks more flow, increasing drag |
| `v_inlet²` | Dynamic pressure — drag scales with the square of velocity (same as Bernoulli) |
| `1 − 2r/H` | Blockage correction — a cylinder near the walls of a narrow channel (height H = 0.41 m) creates extra confinement; as `2r/H → 1`, the correction diverges, reflecting the physical choking of the channel |

### Honest Limitation

This is an **analytical approximation**, not true CFD drag. It correlates
monotonically with the real pressure-based drag coefficient from a MeshGraphNets
rollout but is not calibrated in Newtons. The physics consistency loss during
CVAE training therefore provides directional guidance, not precise physical accuracy.

---

## 5. The CVAE — The Generative Model

### Building Up From Autoencoder to CVAE

**Autoencoder:** An encoder compresses an input to a small code; a decoder
reconstructs the input from that code. Training minimises reconstruction error.
Limitation: the code is a single point — you cannot sample new designs by
picking arbitrary points in code space.

**VAE (Variational Autoencoder):** The encoder now outputs a *distribution*
over codes — a mean `μ` and log-variance `log_σ` — instead of a point. During
training, a code is sampled from this distribution. The KL divergence loss
forces the distribution to stay close to a standard normal `N(0, I)`. At
inference time, you can sample codes from `N(0, I)` and the decoder produces
valid designs. The code space is now smooth and navigable.

**CVAE (Conditional VAE):** The decoder is additionally given a condition
(the target drag), so each sample is steered toward a specific physics outcome.
This is the piece that enables inverse design.

### Architecture

**Encoder** (`CVAEEncoder`):

```
[design_params (4D), drag_actual (1D)]
       → FC(5 → 64) → ReLU
       → FC(64 → 64) → ReLU
       → split → μ[16],  log_σ[16]
```

The encoder maps a *known* design-plus-drag pair to a region of latent space,
not a point. This is the VAE trick: by parameterising uncertainty explicitly,
the model learns that slightly different designs can produce the same drag.

**Reparameterisation** (`CFDCVAE.reparameterise`):

```python
z = mu + eps * torch.exp(0.5 * logvar)   # eps ~ N(0, I)
```

Sampling is non-differentiable, which would break backpropagation. The
reparameterisation trick rewrites the sample as a deterministic function of `μ`,
`log_σ`, and a fixed noise vector `ε`. Gradients now flow through `μ` and `log_σ`
back to the encoder, enabling end-to-end training.

**Decoder** (`CVAEDecoder`):

```
[z (16D), target_drag (1D)]
       → FC(17 → 64) → ReLU
       → FC(64 → 64) → ReLU
       → FC(64 → 4)
       → (cx, cy, r, v_inlet)
```

The conditioning on `target_drag` is what makes this a *conditional* VAE. At
inference the encoder is discarded; you sample `z` and feed it to the
decoder alongside your desired drag.

### Latent Sampling — Latin Hypercube Sampling (LHS)

At inference time, `n_candidates` latent vectors `z` must be sampled from the
prior. Naïve `torch.randn` can **cluster**: with 10–20 samples, pure random
draws sometimes leave entire regions of the latent space unrepresented.

Both `cvae_cfd.py` and `cvae_cloth.py` use **Latin Hypercube Sampling**
instead:

- Each latent dimension is divided into `n_candidates` equal strata.
- Exactly one sample is drawn from each stratum, then dimensions are randomly
  permuted across samples.
- Result: every stratum of every dimension is guaranteed to be covered.

```python
# Module-level sampler (created once, reused):
_lhs_sampler = scipy.stats.qmc.LatinHypercube(d=latent_dim)

# At generate time:
unit_samples = _lhs_sampler.random(n=n_candidates)          # [n, 16] in [0,1]
z = torch.tensor(scipy.stats.norm.ppf(unit_samples), dtype=torch.float32)
```

The sampler is cached as a module-level `_lhs_sampler` so it is not
reconstructed on every call.

### The Three-Part Loss

```python
# From CVAETrainer.fit() — cvae_cfd.py
recon_loss = F.mse_loss(recon, p_b)               # α = 1.0
kl_loss    = -0.5 * mean(1 + logvar - mu² - exp(logvar))  # β = 1e-3
phys_loss  = F.mse_loss(drag_surrogate(recon), target_drag)  # λ = 0.5

loss = alpha * recon_loss + beta * kl_loss + lam * phys_loss
optim.zero_grad()
loss.backward()
optim.step()
```

| Loss term | Purpose | Weight |
|-----------|---------|--------|
| **Reconstruction** `‖θ_recon − θ_gt‖²` | The decoded params must match the input params — forces the encoder/decoder pair to be an accurate round-trip | α = 1.0 |
| **KL divergence** `−½Σ(1 + log_σ − μ² − exp(log_σ))` | Push the latent distribution toward `N(0,I)` so the space is smooth and we can sample from it freely at inference | β = 1e-3 |
| **Physics consistency** `‖drag_surrogate(θ_recon) − target_drag‖` | The generated design must actually achieve the target drag, not just reconstruct the input | λ = 0.5 |

The small β for the KL term is deliberate — it relaxes the regularisation
slightly (a "β-VAE" style setting), allowing the latent space to retain more
design information at the cost of slightly less smoothness.

The cloth CVAE (`cvae_cloth.py`) follows an identical structure with
`(pose_pca[K], stress)` in place of `(design_params[4], drag)` and a
`StressSurrogate` MLP in place of the analytical drag proxy.

### Free-Bits KL — Preventing Posterior Collapse

A well-known failure mode of VAEs is **posterior collapse**: some latent
dimensions become completely ignored during training — the encoder outputs
`μ ≈ 0, σ ≈ 1` for those dimensions (i.e. the posterior matches the prior
exactly), contributing KL ≈ 0, and the decoder learns not to use those
dimensions at all. The result is a latent space that is effectively
lower-dimensional than intended, reducing design diversity in generated samples.

The **free-bits** technique (Kingma et al., 2016) prevents this by imposing a
*minimum KL contribution per latent dimension*. Instead of summing raw KL
values across all dimensions, each dimension's KL is clamped from below:

```
kl_per_dim[i] = -0.5 × mean_batch(1 + logvar_i − μ_i² − exp(logvar_i))
kl_free[i]    = max(kl_per_dim[i], free_bits)
kl_loss       = Σ_i  kl_free[i]
```

With `free_bits = 0.05`, any latent dimension whose posterior is too close to
the prior contributes a fixed cost of `0.05` to the KL loss. The encoder
receives a gradient signal that pushes it to *use* that dimension — because
not using it costs the same as a small non-zero KL. Dimensions that are already
well above the threshold are unaffected; only collapsed dimensions are nudged.

| Setting | Behaviour |
|---------|-----------|
| `free_bits = 0.0` (default) | Standard KL — identical to the formula in the table above. Posterior collapse is possible. |
| `free_bits = 0.05` | Recommended starting value. Prevents collapse with minimal regularisation overhead. |
| `free_bits > 0.2` | Aggressively prevents collapse but can interfere with reconstruction quality if the KL floor is too large relative to the reconstruction loss. |

**Implementation note.** When `free_bits > 0`, the KL loss is computed as a
**sum over dimensions** (not mean), because the floor must be applied
*per-dimension before* aggregating. With `free_bits = 0`, the implementation
falls back to the scalar mean for numerical compatibility with the baseline
formula.

```python
# From CVAETrainer._kl_loss() — cvae_cfd.py
def _kl_loss(self, mu, logvar):
    free_bits = self._cfg.free_bits
    if free_bits <= 0.0:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
    return torch.sum(torch.clamp(kl_per_dim, min=free_bits))
```

The `free_bits` field is exposed as a hyperparameter on both `CVAEConfig` and
`ClothCVAEConfig`, **defaulting to `0.05`** (changed from `0.0`). The default
was updated because posterior collapse was observed in practice with `0.0` on
the cloth CVAE. To revert to the standard KL, pass `free_bits=0.0` explicitly.
Behaviour is covered by `tests/test_free_bits_kl.py` (11 tests).

---

## 6. Mesh Generation — Bridging CVAE Output to the Predictor

The CVAE outputs compact parameter vectors. The MeshGraphNets predictor
consumes PyG `Data` objects with full mesh topology. `mesh_generator.py`
bridges this gap.

### CFD: Full Remeshing via Delaunay

Given `(cx, cy, r, v_inlet)`:

1. **Place 24 points** on the cylinder circumference at equal angles.
2. **Build a 60×16 background grid** over the domain `[0, 1.6] × [0, 0.41]`.
3. **Remove background points** whose distance to `(cx, cy)` is less than `r`
   (they would be inside the cylinder).
4. **Merge** cylinder boundary points + remaining grid points.
5. **`scipy.spatial.Delaunay`** triangulates all merged points; triangles whose
   centroid falls inside the cylinder radius are discarded.
6. **Assign node types** by position: left edge → `INFLOW`, right edge →
   `OUTFLOW`, cylinder surface + top/bottom walls → `WALL_BOUNDARY`, interior
   → `NORMAL`.
7. Apply `T.FaceToEdge() + T.Cartesian() + T.Distance()` to build edge
   indices and attributes matching the training pipeline.

### Cloth: No Remeshing — PCA Inverse Transform Only

Cloth trajectories always have exactly 1579 nodes with identical triangle
connectivity. There is no remeshing. The generator simply:

1. Takes the `[K]` PCA coefficient vector from the CVAE.
2. Applies `PosePCA.inverse_transform`: `world_pos_flat = pose_pca @ components + mean`
3. Reshapes to `[1579, 3]`.
4. Slots the result into a reference `Data` object, replacing only `world_pos`
   and `x` while keeping `mesh_pos`, `node_type`, and `face` from any stored
   reference trajectory.

### Why the Asymmetry Previously Existed — and How CFD Was Unblocked

The CFD pipeline involves a **combinatorial step** (Delaunay triangulation) that has no gradient. This *was* the reason gradient refinement was unavailable for CFD. The cloth pipeline involves only **matrix multiplications** (the PCA inverse transform and MLP decoder), which are fully differentiable.

CFD gradient refinement is now implemented via **`RealMeshLookup`** — a strategy that sidesteps the Delaunay problem entirely. See §7 for both implementations.

---

## 7. Gradient Descent in Latent Space

The subsystem supports gradient-based inverse design for **both** cloth and CFD,
using different strategies to handle differentiability.

### 7a. Cloth — Full Differentiable Chain

The cloth subsystem supports a more powerful inverse design strategy: instead of
just sampling from the CVAE and hoping one sample hits the target, it
**optimises** the latent vector `z` by gradient descent through the full physics
simulation chain.

### The Full Differentiable Chain (Cloth)

```
z [16]
  │  ClothDecoder (MLP — differentiable)
  ↓
pose_pca [16]
  │  PCA inverse transform (matrix multiply — differentiable)
  ↓
world_pos [1579, 3]
  │  FlagSimulator.forward() in eval mode (MeshGraphNets — differentiable)
  ↓
next_world_pos [1579, 3]
  │  StressObjective (norm + mean — differentiable)
  ↓
stress_loss  →  ∂loss/∂z via autograd
```

Every arrow in this chain is a differentiable operation. PyTorch's autograd can
therefore compute `∂stress_loss/∂z` and use it to nudge `z` in the direction
that reduces the gap between predicted stress and target stress.

### Why CFD Needed a Different Approach

In the CFD pipeline, step 5 of mesh generation is `scipy.spatial.Delaunay` — a
combinatorial algorithm that decides *which* points connect to *which* triangles.
This decision has no continuous derivative: there is no gradient for "which
triangles should form". Differentiating through Delaunay is impossible, so CFD
gradient descent cannot go through mesh generation.

### 7b. CFD — Gradient Refinement via RealMeshLookup

CFD gradient refinement is implemented by **avoiding Delaunay entirely**.
`RealMeshLookup` (in `extensions/generative/mesh_generator.py`) builds a
KDTree over the design parameter vectors of all training meshes:

```
generated (cx, cy, r, v_inlet)
        │
        ▼  KDTree.query — nearest training param vector
nearest real training mesh (valid PyG Data, pre-built Delaunay)
        │
        ▼  GNN forward pass (differentiable)
predicted physics
        │
        ▼  drag objective
loss → ∂loss/∂z via autograd
```

The key insight: **we never differentiate through mesh generation**. We snap the
generated parameters to the nearest real training mesh, then differentiate through
the GNN forward pass only. The snapped mesh is a fully valid `PyG Data` object
because it came from actual training data.

**K=5 BPTT:** rather than rolling out the full 600 GNN steps (which would require
600× more activation memory), the refinement loop runs K=5 steps, computes drag
from the average of the last 3 steps, and backpropagates through those 5 steps:

```python
for step in range(K):        # K = 5
    state = gnn(state, mesh)
    if step >= K - 3:
        drag_samples.append(drag_from_state(state))

loss = (mean(drag_samples) - target_drag) ** 2
loss.backward()              # gradients through 5 GNN steps only
```

| Choice | Reason |
|---|---|
| K=5 not full 600 | Memory scales with K; 5 steps gives enough physics context without 600× activation overhead |
| Last 3 of 5 steps for drag | Early steps are transient; averaging later steps gives a more stable drag estimate |
| Snap to training mesh | Guarantees valid topology; avoids Delaunay entirely |

### The Cloth Optimisation Loop

```python
# From ClothInverseDesigner.optimise() — inverse_design.py
for restart in range(n_restarts):               # 3 random restarts
    z = torch.randn(latent_dim, requires_grad=True)
    optim = torch.optim.Adam([z], lr=0.05)

    for it in range(n_iters):                   # 100 iterations
        optim.zero_grad()
        world_pos = decode_pose(z, target_norm) # CVAE decoder + PCA^{-1}
        graph     = build_data(world_pos)       # mesh scaffold (non-diff)
        next_pos  = flag_simulator(graph)       # MeshGraphNets forward
        loss      = stress_objective(next_pos)  # (stress - target)²
        loss.backward()                         # ∂loss/∂z
        optim.step()

# Keep the z with the lowest final loss across all restarts
```

Three restarts guard against local minima in the latent space; the restart with
the lowest final loss is returned.

### `TorchPCAInverseTransform`

The PCA inverse transform must stay inside the autograd graph. A plain NumPy
operation would detach the gradient. `TorchPCAInverseTransform` wraps the PCA
`components` and `mean` arrays as `torch.nn.Module` buffers:

```python
class TorchPCAInverseTransform(torch.nn.Module):
    def __init__(self, components, mean, N):
        super().__init__()
        self.register_buffer("components", torch.from_numpy(components))
        self.register_buffer("mean",       torch.from_numpy(mean))
        self._N = N

    def forward(self, z_pca):
        flat = z_pca @ self.components + self.mean  # gradients flow here
        return flat.view(self._N, 3)
```

Registering as buffers (not parameters) means they are not updated by the
optimiser — only `z` is optimised. But because they are proper tensors in the
computation graph, `loss.backward()` traces through them correctly.

---

## 8. OOD Scoring for Generated Candidates

Each generated candidate is scored for out-of-distribution risk **before** being shown to the user. The generator subsystem uses a different OOD strategy than the Predict page.

### ParamSpaceOOD (`extensions/confidence/ood_detector.py`)

`ParamSpaceOOD` builds a KDTree over the **training design parameter vectors**
`(cx, cy, r, v_inlet)`:

```python
score = clip(1 - d / train_diameter, 0, 1)
```

where `d` is the Euclidean distance from the candidate's parameter vector to its
nearest neighbour in the training set, and `train_diameter` is the 95th-percentile
pairwise distance of training parameters (used for normalisation).

| Score | Interpretation |
|---|---|
| ~1.0 | Candidate geometry is close to a training geometry |
| ~0.0 | Candidate geometry is far from anything seen during training |

**Why parameter space, not embedding space?**
The Predict page uses a GNN embedding-space OOD detector — it asks "is this
*simulation state* similar to training states?" The generator's OOD question is
different: "is this *geometry design* similar to training geometries?" A
parameter-space KDTree answers this directly and is far cheaper (no GNN forward
pass needed). The two detectors coexist and serve different purposes.

---

## 9. SSH Dispatch for Generate

If `runs/ssh_config.json` exists, the generate route offloads computation to a
remote GPU machine instead of running locally:

```
Browser → POST /api/generate
        → api/routes/generate.py
        → write GenerateRequest to runs/ui_generate_config.json
        → SSH to GPU machine
        → run generate_ssh.py on remote
        → generate_ssh.py streams named SSE events back through SSH tunnel
        → browser receives events
```

`generate_ssh.py` uses **named SSE events**:

```
event: candidate
data: { ...candidate payload... }

event: done
data: { "best_id": 2, "session_id": "abc123" }
```

Named events (vs. unnamed `data:` events used in rollout SSE) are required here
because the generate stream carries multiple event types — candidates and done —
that the frontend must handle differently. The rollout SSE has only one event
type (a frame update) so unnamed events are sufficient.

---

## 10. Tradeoffs

| # | Tradeoff | Detail |
|---|----------|--------|
| 1 | **Surrogate is proxy, not true drag** | `drag_proxy = r·v²/(1−2r/H)` correlates with real drag but is not calibrated in Newtons. Physics consistency during CVAE training provides directional guidance only; generated designs may not hit exact drag targets in a full CFD simulation. |
| 2 | **Fixed latent dimensionality** | Latent dim = 16 is a hand-tuned hyperparameter for both CVAEs. Too small and the latent space is too compressed to represent design diversity; too large and the space becomes harder to sample from. There is no automatic selection. |
| 3 | **Circle fit assumes single cylinder** | `CircleFitter` fits exactly one circle. A domain with two obstacles or a non-circular obstacle (NACA airfoil, square bluff body) would produce incorrect or meaningless parameters. |
| 4 | **CFD gradient refinement snaps to training mesh** | `RealMeshLookup` guarantees valid topology by snapping to the nearest real training mesh. This means refinement can only reach parameter combinations that have a close neighbour in the training set — extrapolation to truly novel geometries may snap to a poor proxy mesh. |
| 5 | **Cloth PCA assumes linearity** | PCA models the cloth pose manifold as a linear subspace. If the true manifold is nonlinear (e.g., large-deflection folds), 16 linear components may not capture all valid poses, and the CVAE can only generate poses reachable by linear combinations of the training data's principal directions. |
| 6 | **CVAE is bounded by training distribution** | The CVAE can only generate designs within the range of the training data. Extrapolation — requesting a drag far outside the training range, or a cloth stress never seen during training — may produce degenerate or physically implausible parameters. |

---

## 11. Potential Enhancements

| # | Enhancement | Benefit |
|---|-------------|---------|
| 1 | **Use true MeshGraphNets rollout for physics loss** | Replace the analytical drag proxy with an actual forward pass through `CylinderSimulator` during CVAE training. Slower (requires GPU batching tricks), but the physics consistency loss would be exactly calibrated — not an approximation. |
| 2 | **Learnable shape encoder instead of circle fit** | Train a small GNN or PointNet to extract design parameters directly from node positions, removing the single-cylinder assumption. The encoder could generalise to multi-obstacle or non-circular geometries. |
| 3 | **Mesh morphing for CFD gradient descent** | Instead of re-triangulating from scratch for each new `(cx, cy, r)`, deform an existing template mesh by moving nodes smoothly (e.g., via radial basis function interpolation or a learned deformation field). All steps remain differentiable with respect to the design parameters. |
| 4 | **Normalizing flows instead of VAE** | Flows (e.g., RealNVP, Glow) model the exact posterior rather than a variational approximation. This produces sharper samples with no blurring from the KL regularisation, at the cost of more complex architecture and training. |
| 5 | **Bayesian optimisation over CVAE latent space** | Instead of Adam gradient descent, treat the latent space as a black-box search space and apply a Gaussian Process surrogate + acquisition function (e.g., Expected Improvement). This is more sample-efficient when the physics evaluation is expensive and non-differentiable. |
| 6 | **Multi-objective inverse design** | The current physics loss optimises a single scalar (drag or stress). Extending the CVAE to condition on a tuple `(drag, lift)` and using a Pareto-front sampler at inference time would enable simultaneous multi-objective optimisation — for example, minimising drag while maintaining a target lift coefficient. |

## See also

- [[SUBSYSTEM_PREDICTOR]] — used as the physics oracle during gradient-descent inverse design (cloth) and for final candidate verification
- [[SUBSYSTEM_CONFIDENCE]] — OOD scores are computed on each generated design before presenting it to the user
- [[SUBSYSTEM_DATA]] — raw trajectories are consumed in Phase 0 to extract compact design parameters
- [[FOR_THE_USER]] — plain-language explanation of what this subsystem does and why
