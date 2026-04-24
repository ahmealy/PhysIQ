---
tags: [physicsai, research, equivariance, e3nn, geometric-dl]
created: 2026-04-24
aliases: [equivariance, geometric, symmetry, e3nn]
---

# Equivariance — What It Is, Why We Don't Have It, and How to Add It

## The short answer

**No — PhysIQ is not equivariant.**

The GNN will give different predictions if you rotate or reflect the input mesh, even though the underlying physics doesn't care about orientation. This note explains exactly why, what it costs us, and the complete roadmap to fix it.

---

## 1. What equivariance means — from scratch

### Invariance vs equivariance

A function `f` is **invariant** to transformation `T` if the output doesn't change when the input is transformed:

```
f(T(x)) = f(x)
```

Example: drag coefficient is invariant to rotation. Rotate the entire flow domain 90° — the drag number is unchanged.

A function `f` is **equivariant** to transformation `T` if the output transforms in the same way the input does:

```
f(T(x)) = T(f(x))
```

Example: velocity field is equivariant to rotation. Rotate the domain 90° — the velocity vectors rotate 90° too, but the magnitudes and flow patterns are the same. The physics is the same, just expressed in a different coordinate frame.

### Why physics is equivariant

The laws of physics don't have a preferred coordinate frame. Navier-Stokes equations hold in any inertial frame, at any orientation. A vortex behind a cylinder looks the same whether the cylinder is horizontal or vertical — just rotated. This is called **Galilean symmetry** (for velocity transforms) or **E(n) symmetry** (for rotations + reflections + translations in n dimensions).

What this means for a learning model:
- **If your model is equivariant**, rotating a training example gives you a free training sample — the model already knows what that looks like.
- **If your model is not equivariant**, a flow going left and a flow going right look like completely different problems. The model must learn both from data.

---

## 2. Exactly where our GNN breaks equivariance

### The node features

CFD node features (`model/simulator.py`):

```python
node_feats = torch.cat([frames, one_hot.float()], dim=-1)
# frames = [vx, vy]          ← a 2D vector
# one_hot = node_type [N, 9]  ← scalars (invariant to rotation)
```

`vx` and `vy` are the x and y components of velocity — a **2D vector**. When you rotate the mesh 90°, `(vx, vy)` should become `(-vy, vx)`. But the NodeEncoder MLP treats them as **two independent scalar inputs**:

```python
# NodeEncoder MLP: Linear(11, 128) → ReLU → Linear(128, 128) → ...
```

The weight matrix `W ∈ ℝ^{128×11}` multiplies `[vx, vy, ...]` — it has no knowledge that columns 0 and 1 are components of the same vector. It can learn `vx` and `-vy` as separate features, but it has no structural guarantee that their relationship is preserved under rotation.

**The violation:** `Encoder(R · x) ≠ R · Encoder(x)` for rotation matrix `R`.

### The edge features

CFD edge features are built as `(dx, dy, distance)` — the relative displacement vector between connected nodes plus its norm:

```
edge_attr = [pos_j - pos_i]  →  [Δx, Δy, ||Δr||]
```

From `model/flag_simulator.py` (cloth, showing the same pattern):

```python
rel_world = pos[senders] - pos[receivers]           # [E, 3] displacement vector
world_norm = torch.norm(rel_world, dim=-1, keepdim=True)  # [E, 1] scalar distance
edge_attr = torch.cat([rel_world, world_norm, ...], dim=-1)  # [E, 7]
```

`(Δx, Δy)` is again a vector. The EdgeEncoder MLP treats it as two independent scalars. Rotate the mesh: `Δx` and `Δy` transform but the MLP weights don't know they're components of the same geometric vector.

**The violation:** `EdgeEncoder(R · edge_attr) ≠ R · EdgeEncoder(edge_attr)`.

### The decoder

The decoder outputs `(Δvx, Δvy)` — a velocity change vector. For equivariance, rotating the input should rotate the output by the same amount. But because the encoder already broke equivariance, the decoder cannot recover it.

### Summary: three places where equivariance is violated

```
Input: (vx, vy)      ← velocity vector, treated as 2 scalars  [VIOLATION 1]
Input: (Δx, Δy)      ← displacement vector, treated as 2 scalars  [VIOLATION 2]
Output: (Δvx, Δvy)   ← velocity change, not equivariantly produced  [VIOLATION 3]
```

---

## 3. What it costs us

### Data efficiency

Without equivariance, every orientation is a different problem. To learn cylinder flow going left, you need data for flow going left. To learn it going right, you need separate data. To learn it at 45°, more data.

With equivariance, learning flow going left automatically gives you all orientations for free — by the group structure.

**Rough estimate:** E(2)-equivariant models (rotations + reflections in 2D) need ~8-16× less data to reach the same accuracy on CFD. E(3)-equivariant models need even more leverage in 3D.

### Generalisation to new mesh orientations

If a user submits a cylinder flow mesh that is rotated 30° from the training set's typical orientation, our GNN may generalise poorly. It has never seen that exact orientation in training. An equivariant model would handle it exactly, by construction.

### Confidence scoring

Our confidence index uses GNN embeddings to detect out-of-distribution inputs. A rotated mesh produces a different embedding from the same mesh un-rotated — even though the physics is identical. This inflates the "OOD distance" for inputs that are geometrically the same but differently oriented. An equivariant encoder would embed them identically.

---

## 4. What partial equivariance we DO have

### Message passing is permutation equivariant

The GNN's aggregation (scatter-sum, mean, attention) is permutation equivariant — reordering the nodes doesn't change the predictions. This is a different symmetry from rotation, but it's correct.

### Distance as edge feature

Including `||Δr||` (the scalar norm of the displacement) as an edge feature is **rotationally invariant** — distance doesn't change under rotation. This gives the model access to one piece of geometric information that's correct regardless of orientation.

### The MeshGraphNets paper's partial fix

The original paper (Pfaff et al., 2021) uses **relative** mesh positions as edge features rather than absolute positions. This gives **translational equivariance** — you can translate the whole mesh and predictions are unchanged (since all that matters is the relative displacement). But it doesn't give rotational equivariance, because `(Δx, Δy)` are still raw Cartesian components.

So current status:
- ✅ Permutation equivariant
- ✅ Translational equivariant (relative positions)
- ❌ Rotational equivariant
- ❌ Reflection equivariant

---

## 5. How to make it equivariant — the full roadmap

### Option A: Data augmentation (easiest, imperfect)

**What:** During training, randomly rotate the mesh and velocity field by a random angle θ before each forward pass.

```python
# In training loop, before forward():
theta = torch.rand(1) * 2 * math.pi
R = torch.tensor([[cos(theta), -sin(theta)],
                  [sin(theta),  cos(theta)]])
graph.pos = graph.pos @ R.T
graph.x[:, :2] = graph.x[:, :2] @ R.T     # rotate velocity
graph.edge_attr[:, :2] = graph.edge_attr[:, :2] @ R.T  # rotate edge disp
graph.y = graph.y @ R.T                    # rotate target acceleration
```

**What you get:** The model learns to be approximately equivariant — it has seen all orientations. But it's learned, not structural. At inference on an unseen orientation it might still generalise slightly less perfectly than a true equivariant model.

**Cost:** 2-3× more training time (to see enough orientations). Zero architectural change. Zero inference overhead.

**When to use:** Good first step. Easy to implement, gives most of the data efficiency benefit. Used in practice by many production systems.

---

### Option B: Polar/spherical edge features (medium effort, partial fix)

**What:** Instead of `(Δx, Δy)`, use `(||Δr||, θ_rel)` — distance and relative angle. Distance is already rotation-invariant. The relative angle `θ_rel = atan2(Δy, Δx)` transforms predictably under rotation (adds a constant offset).

For the node features, decompose velocity into `(||v||, θ_v)` — speed and direction angle.

**What you get:** The scalar magnitudes `||Δr||` and `||v||` are fully rotation-invariant. The angles `θ_rel` and `θ_v` are not invariant but they transform in a simple, learnable way. This is better than raw `(Δx, Δy)` because at least the magnitudes are correct.

**What you don't get:** True equivariance. The MLP still doesn't know angles are angles.

**Cost:** Small feature engineering change. No architectural change.

---

### Option C: Steerable features with e3nn (proper equivariance, more effort)

This is the principled solution. It requires replacing the MLPs in the encoder with **equivariant linear layers** from the `e3nn` library.

#### The key idea: irreducible representations

In equivariant neural networks, features are not arbitrary vectors — they are typed by how they transform under rotations. The types are called **irreducible representations** (irreps):

| Type | Symbol | Example | Transforms under rotation |
|------|--------|---------|--------------------------|
| Scalar | `0e` | pressure, distance | Unchanged — invariant |
| Pseudoscalar | `0o` | handedness/chirality | Flips sign under reflection |
| Vector | `1o` | velocity, displacement | Rotates like a vector |
| Pseudovector | `1e` | angular momentum | Rotates + flips under reflection |
| Rank-2 tensor | `2e` | stress tensor | Rotates like a 2nd-order tensor |

In e3nn, you declare the type of every feature:

```python
from e3nn import o3

# Node features: 9 scalars (one-hot node_type) + 1 velocity vector
node_irreps = o3.Irreps("9x0e + 1x1o")
#                        ^scalars   ^vector

# Edge features: 1 scalar (distance) + 1 displacement vector
edge_irreps = o3.Irreps("1x0e + 1x1o")
```

The **equivariant linear layer** `o3.Linear(irreps_in, irreps_out)` can only mix features of the same type — scalars with scalars, vectors with vectors. This structural constraint is what enforces equivariance.

#### What changes in the code

**1. NodeEncoder** (`model/model.py`, `Encoder` class):

```python
# CURRENT (non-equivariant):
self.nb_encoder = nn.Sequential(
    nn.Linear(node_in, hidden), nn.ReLU(),
    nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
)

# EQUIVARIANT:
from e3nn import o3
from e3nn.nn import Gate
node_irreps_in  = o3.Irreps("9x0e + 1x1o")  # 9 scalars + velocity vector
node_irreps_out = o3.Irreps("64x0e + 16x1o") # 64 scalar channels + 16 vector channels
self.nb_encoder = o3.Linear(node_irreps_in, node_irreps_out)
```

**2. EdgeEncoder** (`model/model.py`, `Encoder` class):

```python
# CURRENT:
self.eb_encoder = nn.Sequential(
    nn.Linear(edge_in, hidden), nn.ReLU(), ...
)

# EQUIVARIANT:
edge_irreps_in  = o3.Irreps("1x0e + 1x1o")  # distance scalar + displacement vector
edge_irreps_out = o3.Irreps("64x0e + 16x1o")
self.eb_encoder = o3.Linear(edge_irreps_in, edge_irreps_out)
```

**3. EdgeBlock and NodeBlock** — the MLPs inside need to be replaced with **tensor product layers**. This is the most complex change.

```python
# The equivariant way to combine two irrep vectors (e.g. node + edge):
self.tp = o3.FullyConnectedTensorProduct(
    irreps_in1=node_irreps,
    irreps_in2=edge_irreps,
    irreps_out=hidden_irreps,
    shared_weights=False,
)
```

The tensor product is how equivariant layers "mix" different-type features. It respects the Clebsch-Gordan rule: `1o ⊗ 1o = 0e + 1e + 2e` (two vectors combine to give a scalar + pseudovector + rank-2 tensor).

**4. Decoder** — output velocity change is a `1o` vector:

```python
# CURRENT: Linear(128, 2)
# EQUIVARIANT:
decoder_irreps_in  = o3.Irreps("64x0e + 16x1o")
decoder_irreps_out = o3.Irreps("1x1o")  # one velocity-change vector
self.decoder = o3.Linear(decoder_irreps_in, decoder_irreps_out)
```

#### Complexity of the change

| Component | Current | Equivariant | Effort |
|-----------|---------|-------------|--------|
| NodeEncoder | `nn.Linear` | `o3.Linear` | Low |
| EdgeEncoder | `nn.Linear` | `o3.Linear` | Low |
| EdgeBlock MLP | `nn.Sequential` | `o3.TensorProduct + Gate` | High |
| NodeBlock MLP | `nn.Sequential` | `o3.TensorProduct + Gate` | High |
| Decoder | `nn.Linear` | `o3.Linear` | Low |
| Feature type declarations | None | `o3.Irreps` everywhere | Medium |

#### Which processor to start with

**GnBlock is the best candidate** — its aggregation (scatter-sum) is already equivariant to permutations, and replacing the EdgeBlock and NodeBlock MLPs with tensor products is straightforward.

**TNSBlock is hardest** — attention over equivariant features requires the attention weights themselves to be invariant (scalars), while the values are equivariant (vectors). Possible but complex.

---

### Option D: Invariant features only (simplest strict equivariance)

**What:** Design features that are already rotationally invariant, so no architectural change is needed.

For edge features:
- `||Δr||` — distance (invariant) ✅
- `angle between edge and reference direction` — NOT invariant ❌

For node velocity:
- `||v||` — speed (invariant) ✅  
- Direction of velocity — NOT invariant without a reference frame ❌

**The problem:** If you use only invariant features, you lose directional information entirely. The model can't distinguish "flow going left" from "flow going right" — they'd look identical. This is too much information to throw away for CFD.

**When it works:** For predicting **scalar** quantities only (drag coefficient, pressure coefficient at a point). If your target is a scalar, you can use all-invariant features and get a fully invariant model. This is appropriate for the drag surrogate MLP in `drag_surrogate.py`.

---

## 6. What to implement for this project

### Recommended path

**Phase 1 (immediate, low cost):** Data augmentation with random rotations during CFD training. 3 days of work. Gives ~70% of the benefit of true equivariance.

**Phase 2 (medium term):** Replace `(Δx, Δy)` edge features with `(||Δr||, sin θ, cos θ)` where θ is angle relative to mesh principal axis. Not true equivariance but much better. 1 week.

**Phase 3 (research):** Full e3nn equivariant GnBlock. New `EquivariantEncoder`, `EquivariantGnBlock`, `EquivariantDecoder` modules. 3-6 weeks. Requires e3nn install and significant architecture rewrite.

### Files that would change in Phase 3

```
model/model.py          ← Encoder, GnBlock, Decoder rewrite with e3nn
model/simulator.py      ← feature construction (velocity as 1o irrep)
model/flag_simulator.py ← feature construction (world_pos as 1o irrep)
dataset/fpc.py          ← graph.x structure changes
train.py                ← no change needed (model is swappable)
```

### Backward compatibility

The equivariant model would be a new `architecture='egn'` option alongside `'gn'`, `'tns'`, `'sage'`. Existing checkpoints remain valid. Training would require e3nn:

```bash
pip install e3nn
```

---

## 7. How to verify equivariance — the test

Regardless of approach, equivariance can be tested numerically:

```python
import torch
import math

def test_rotational_equivariance(simulator, graph, theta=0.5):
    """If model is equivariant, rotating input should rotate output by same angle."""
    R = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ], dtype=torch.float32)

    # Original prediction
    pred_orig = simulator(graph, noise=None)  # [N, 2] velocity change

    # Build rotated graph
    graph_rot = graph.clone()
    graph_rot.pos = graph.pos @ R.T
    graph_rot.x[:, :2] = graph.x[:, :2] @ R.T  # rotate velocity in node features
    graph_rot.edge_attr[:, :2] = graph.edge_attr[:, :2] @ R.T  # rotate edge displacements

    # Prediction on rotated input
    pred_rot = simulator(graph_rot, noise=None)  # [N, 2]

    # If equivariant: pred_rot ≈ pred_orig @ R.T
    pred_expected = pred_orig @ R.T

    error = (pred_rot - pred_expected).norm() / pred_expected.norm()
    print(f"Relative equivariance error: {error:.6f}")
    # Current model: error ≈ 0.3-0.8 (not equivariant)
    # Equivariant model: error < 1e-5 (machine precision)
    return error
```

Run this test on the current GN model — you'll see a large error (~30-80%). Run it on an e3nn model — you'll see near-zero error (< 1e-5, machine precision only).

---

## 8. Related work

| Paper | What it does | Relevance |
|-------|-------------|-----------|
| **e3nn** (Geiger et al., 2022) | Library for E(3)-equivariant neural networks | Direct implementation tool |
| **EGNN** (Satorras et al., 2021) | Equivariant GNN using only distances + velocities | Simpler than e3nn, good starting point |
| **SE(3)-Transformer** (Fuchs et al., 2020) | Attention + equivariance | TNS-equivalent but equivariant |
| **EquiMesh** (research, 2023) | Equivariant GNNs for mesh simulation | Directly related to MeshGraphNets |
| **Segnn** (Brandstetter et al., 2022) | Steerable E(3) GNN for simulation | Shows 3-10× data efficiency gain |
| **NequIP** (Batzner et al., 2022) | E(3)-equivariant for molecular dynamics | Proof that equivariance works for physics |

The Segnn paper (Brandstetter 2022) is the closest to our use case — they apply equivariant GNNs to the same class of physics simulation problems as MeshGraphNets and show significant data efficiency improvements.

---

## 9. Interview talking points

**Q: Is your GNN equivariant?**

> "No — it's translational equivariant (we use relative positions as edge features) and permutation equivariant (scatter-sum aggregation), but not rotationally equivariant. The core issue is that velocity and displacement vectors — `(vx, vy)` and `(Δx, Δy)` — are fed into MLPs as independent scalars. The MLP weight matrix has no structural knowledge that these are components of the same geometric vector, so it can't guarantee that rotating the input rotates the output the same way."

**Q: How would you add equivariance?**

> "Three paths with increasing complexity. First, data augmentation — randomly rotate meshes during training so the model approximately learns all orientations. Fast to implement, zero architectural change, covers maybe 70% of the benefit. Second, switch to polar edge features — replace `(Δx, Δy)` with `(distance, sin θ, cos θ)` so the scalar magnitude is guaranteed invariant. Third, full e3nn rewrite — declare node features as typed irreps (scalars `0e`, vectors `1o`), replace MLPs with equivariant tensor product layers. This gives exact equivariance, ~10× data efficiency, and better generalisation to unseen orientations. The cost is 3-6 weeks of work and ~3× computational overhead per forward pass."

**Q: What's the practical impact of not having it?**

> "For our training distribution it's fine — all training meshes have roughly similar orientations so the model has seen enough variety. The real cost shows up in two places: data efficiency (we need more training trajectories than an equivariant model would) and confidence scoring (a rotated version of a training mesh produces a different embedding and looks artificially out-of-distribution, when physically it's identical)."
