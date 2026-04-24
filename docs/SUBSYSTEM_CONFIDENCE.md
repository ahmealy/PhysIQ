---
tags: [physicsai, subsystem, deep-dive]
created: 2026-04-10
aliases: [confidence, ood, embedding]
---

# Confidence & OOD Detection Subsystem

## Quick summary

- Detects when a generated design is unlike anything in the training set
- Extracts embedding from the predictor's encoder (full-mesh mean-pool over all nodes):
  - **CFD**: dual-frame 256-dim (frame 0 → [128] + frame 5 → [128], concatenated)
  - **Cloth**: single-frame 128-dim
- KD-tree lookup: find nearest training embedding → compute normalised distance
- Score formula: `confidence = clip(1 - d / train_diameter, 0, 1)`
- `train_diameter` = 95th percentile of 5-NN distances within the training set (robust to outliers)
- Threshold 0.3: designs scoring below 30% confidence are flagged OOD
- Index stores a SHA-256[:16] checkpoint hash; raises `IndexStaleError` if the checkpoint has changed since the index was built

**Source modules:**
`confidence/index.py` · `confidence/build_index.py` · `model/embedding.py` · `extensions/confidence/ood_detector.py`

---

## 1. What This Subsystem Does

Given a generated or user-submitted mesh design — a cylinder with a certain radius and position, or a cloth patch in some initial pose — this subsystem decides how much to trust the physics predictor's output for that design. It extracts a compact numerical "fingerprint" of the mesh, compares that fingerprint against fingerprints stored from every training mesh, and produces a **confidence score in [0, 1]**. A score near 1.0 means the mesh looks like something the model has seen before, so its prediction is likely reliable. A score near 0.0 means the mesh is unlike anything in the training set — the model is, effectively, making something up. This is the "am I making something up?" alarm that sits between the predictor and the user-facing result.

---

## 2. The Core Problem: Distribution Shift

### Neural networks are interpolators, not extrapolators

A neural network generalises between examples it has seen — it does not extrapolate safely to regions it has never encountered. A CFD model trained on cylinders with radii in the range **0.04–0.12 m** has learned flow patterns that exist within that geometric envelope. Feed it a cylinder with radius **0.40 m** and the network has no principled basis for its answer. The geometry falls completely outside the manifold of training data.

The dangerous part: **the model will not refuse to answer.** It will produce a number — a velocity field, a pressure distribution — that looks plausible in format but can be wildly wrong in value. There is no built-in "I don't know" signal.

### This is not the same as softmax confidence

In classification models you sometimes hear "use the softmax probability as a confidence score." That trick does not apply here. This project predicts a **physics field** (velocity, pressure, or 3D acceleration) — it is a regression problem. There is no softmax, no probability distribution over classes, no natural output that says "uncertain." We need an entirely separate mechanism layered on top of the predictor.

### The mechanism we need

We need a way to ask: *does this input look like the training data?* That question lives in the **input space**, not the output space, and it requires comparing the new mesh against the distribution of training meshes — distribution shift detection.

---

## 3. The Embedding Approach

### Meshes as points in embedding space

The predictor is an **Encoder → Processor → Decoder** architecture (`EncoderProcesserDecoder` in `model/model.py`). The encoder compresses normalised node and edge features into a 128-dimensional latent vector for each node. Crucially, this compression happens **before** the 15-step message-passing processor and before any decoding. The encoder has learned to map "what kind of mesh is this?" into a compact representation.

**Key insight:** Two meshes that are geometrically similar — cylinders of similar radii, or cloth patches in similar poses — will produce encoder outputs that cluster close together in the embedding space. A mesh with a radically different geometry will land far away from that cluster. For CFD, the dual-frame encoding means that two identical geometries with different inlet velocities will also be separated — their frame-5 embeddings diverge as the flow begins to develop differently.

### Building the reference library

After training, `confidence/build_index.py` iterates over all training samples, calls `extract_embedding()` on each one, and stacks the results into an array of shape **[N_train, 256]** (CFD) or **[N_train, 128]** (cloth). For the standard datasets this is approximately 1,000 meshes. These embeddings are the **reference library** — the map of what the model has seen.

### Why mean-pool over all nodes (full-mesh pooling)?

Each call to the encoder produces one 128-dimensional vector **per node**. To get a single fingerprint for the whole mesh, we average (mean-pool) across **all** nodes — not just NORMAL nodes. Full-mesh mean pooling is **permutation-equivariant**: the fingerprint is invariant to node ordering, which is the correct inductive bias for an unordered mesh. Including boundary nodes is safe here because the normalizer has already placed all node types in a comparable numerical range, and the one-hot type encoding makes each node type distinguishable in latent space.

---

## 4. `extract_embedding()` in Detail

`model/embedding.py` provides a single public function, `extract_embedding(simulator, graph, device)`, which dispatches to one of two private implementations depending on whether the simulator is a `Simulator` (CFD) or a `FlagSimulator` (cloth).

### CFD path (`_extract_embedding_cfd`) — dual-frame 256-dim

CFD uses a **dual-frame** encoding to capture inlet velocity boundary conditions. Two simulations with identical geometry but different inlet velocities look identical at frame 0 (the mesh is the same, the initial velocity field may be uniform), but differ by frame 5 (`CFD_WARMUP_FRAMES = 5`) because early flow development has begun to encode the inlet velocity into the interior field. Encoding only frame 0 would make those two simulations indistinguishable — a confidence system could not tell them apart.

The procedure:

1. Extract `node_type` from `graph.x[:, 0:1]` and the physics field (velocity or pressure).
2. Call `simulator.update_node_attr(frame_0, node_type)`, run `simulator.model.encoder(graph)`, mean-pool over all nodes → **[128] vector** for frame 0.
3. Repeat for frame 5 (the warmup frame) → **[128] vector** for frame 5.
4. Concatenate the two vectors → **[256] embedding**.

The resulting [256]-dim vector captures both the geometry (frame 0 encodes the mesh shape) and the early flow physics (frame 5 encodes how the flow responds to the inlet condition). This is sufficient to disambiguate geometrically identical designs with different boundary conditions.

### Cloth path (`_extract_embedding_cloth`) — single-frame 128-dim

Cloth does not need a warmup frame. Cloth initial conditions are **fully captured by positions**: a cloth patch's initial pose, rest shape, and handle locations completely determine its subsequent trajectory under the learned dynamics. There is no hidden boundary condition that only manifests after several steps.

The path mirrors `FlagSimulator.forward()` precisely:

```python
graph = simulator._build_graph(graph)                        # FaceToEdge + build [E, 7] edge attrs
node_type_col = graph.x[:, 3:4].squeeze(-1).long()          # extract type BEFORE x is overwritten
node_feats = simulator._build_node_features(graph, node_type_col)   # velocity[3] + one_hot[9]
graph.x = simulator._node_normalizer(node_feats, False)     # normalise in eval mode
graph.edge_attr = simulator._edge_normalizer(graph.edge_attr, False)

encoded = simulator.model.encoder(graph)                     # [N, 128]
embedding = encoded.x.mean(dim=0)                            # [128] — full-mesh mean pool
```

### Why must preprocessing exactly replicate `forward()`?

The encoder weights were optimised during training to operate on **normalised** features — zero-meaned, unit-variance, via running statistics accumulated during the training loop. If you feed the encoder raw unnormalised features, you are presenting inputs that are orders of magnitude outside the distribution the weights expect. The resulting embedding is meaningless garbage, and any distance computed from it is meaningless too. Every transformation applied in `forward()` must be applied identically here.

---

## 5. The KD-Tree: Fast Nearest-Neighbor Lookup

### The problem

We have ~1,000 training embeddings, each a point in 128-dimensional space. When a new mesh arrives, we need to find the **single closest** training embedding to the new mesh's embedding. That distance tells us how "alone" the new mesh is in the training data landscape.

### Brute force is fine — until it isn't

Computing all 1,000 Euclidean distances takes O(N × D) operations: multiply-and-add for every dimension of every training point. For N=1,000 and D=128 that's 128,000 operations — completely fine. But at N=100,000 that's 12.8 million per query, and if you are scoring designs interactively at inference time, it starts to matter.

### KD-tree structure

A **KD-tree** is a spatial indexing structure that recursively partitions the embedding space using axis-aligned splits. Building it costs O(N log N) and is done **once** after training. Each query then costs O(log N) in the best case by pruning branches that cannot contain the nearest neighbor.

`confidence/index.py` uses **`scipy.spatial.KDTree`** as the default backend. After building, the index computes the training diameter (see Section 6) and serialises to `runs/embedding_index.pkl`.

An **optional C++ backend** (`confidence/kdtree.cpp`, bound via pybind11) is also available. `NearestNeighborIndex.build()` tries to import `_kdtree.KDTree` and falls back to scipy silently if the extension is not compiled:

```python
try:
    from _kdtree import KDTree as CppKDTree
    self._cpp_tree = CppKDTree(self.embeddings)
    self.backend = "cpp"
except ImportError:
    pass  # scipy fallback already built
```

### The curse of dimensionality caveat

KD-trees are asymptotically optimal in low dimensions (D ≤ ~20). At D=128 the algorithm degrades toward a linear scan because bounding hypercubes in 128 dimensions almost always intersect the query ball. In theory this is bad. In practice it is acceptable here because **training embeddings cluster tightly** — similar training geometries produce nearby embeddings — so the tree pruning still eliminates most branches. The benchmark (`confidence/benchmark.py`) shows scipy KDTree query times well under 0.1 ms for N=10,000, which is fast enough for interactive use.

---

## 6. Confidence Score Formula

Once we have the nearest-neighbor distance `d_min`, we need to convert it to an interpretable score. The formula in `NearestNeighborIndex.query()` is:

```
confidence = clip(1 - d_min / train_diameter, 0, 1)
```

Implemented as:

```python
return float(np.clip(1.0 - d_min / (self.train_diameter + 1e-12), 0.0, 1.0))
```

### What is `train_diameter`?

The **train diameter** is the **95th percentile of 5-NN distances within the training set**. It is computed in `NearestNeighborIndex.build()`:

```python
dists, _ = self._scipy_tree.query(self.embeddings, k=6)  # k=6: skip self + 5 neighbours
self.train_diameter = float(np.percentile(dists[:, 1:6], 95))
```

We query each training point against its 5 nearest neighbours (skipping the self-match at index 0) and take the 95th percentile of all those distances. This is a robust measure of how spread out the training embeddings are within their own neighbourhood structure.

**Why 95th percentile and not the maximum?**

The maximum nearest-neighbor distance in the training set is brittle: a single outlier training sample with an unusual geometry can inflate the diameter and make the score overconfident for everything else. The 95th percentile is a robust estimate that tolerates a small fraction of outlier training examples without distorting the scale for the majority.

| Scenario | `d_min` vs `train_diameter` | Confidence |
|---|---|---|
| New mesh is very close to a training mesh | `d_min ≪ train_diameter` | Near 1.0 |
| New mesh is about as far as typical training scatter | `d_min ≈ train_diameter` | Near 0.0 |
| New mesh is far beyond any training mesh | `d_min ≫ train_diameter` | Clipped at 0.0 |

If your new mesh's embedding sits at a distance that is smaller than what is typical **between** training meshes themselves, you are safely in-distribution. If it is much larger, you have left the training manifold.

### Intuition

| Scenario | `d_min` vs `train_diameter` | Confidence |
|---|---|---|
| New mesh is very close to a training mesh | `d_min ≪ train_diameter` | Near 1.0 |
| New mesh is about as far as typical training scatter | `d_min ≈ train_diameter` | Near 0.0 |
| New mesh is far beyond any training mesh | `d_min ≫ train_diameter` | Clipped at 0.0 |

If your new mesh's embedding sits at a distance smaller than what is typical **between** training meshes themselves, you are safely in-distribution. If it is much larger, you have left the training manifold.

---

## 6b. Stale Index Detection

The embedding index (`runs/embedding_index.pkl`) is built from a specific trained checkpoint. If the checkpoint is later retrained — even with the same filename — the index embeddings are stale and distances computed against them are meaningless.

### How it works

`confidence/index.py` computes a **SHA-256[:16] hash** of the checkpoint file at index-build time and stores it inside the serialised index:

```python
def checkpoint_hash(path: str) -> str:
    """Read checkpoint bytes, compute SHA-256, return first 16 hex chars."""
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()[:16]
```

`confidence/build_index.py` calls this and writes the hash into the index at build time:

```python
index.checkpoint_hash = checkpoint_hash(args.checkpoint)
```

At load time, `NearestNeighborIndex.load(expected_checkpoint="path/to/model.pth")` recomputes the hash of the checkpoint currently on disk and compares:

```python
if self.checkpoint_hash != checkpoint_hash(expected_checkpoint):
    raise IndexStaleError(
        f"Index was built from checkpoint hash {self.checkpoint_hash!r} "
        f"but current checkpoint hashes to {live_hash!r}. Rebuild the index."
    )
```

### Why fail-fast instead of a warning or silent rebuild?

- **Silent rebuild:** surprises the user with a 5-minute delay mid-inference; the delay is invisible and the cause is unclear.
- **Warning:** easily ignored, especially in a pipeline or automated evaluation script. The stale index continues to silently produce wrong confidence scores.
- **Fail-fast (`IndexStaleError`):** forces a deliberate, explicit rebuild. Production correctness outweighs convenience. The error message names the checkpoint and tells the user exactly what to do.

---

## 7. OODDetector Wrapper

`extensions/confidence/ood_detector.py` provides the public-facing API that the rest of the application uses.

### Class hierarchy

```
BaseOODScorer (ABC)
    └─ KDTreeScorer          ← wraps NearestNeighborIndex.query()
         └─ [future: FAISSScorer, MahalanobisScorer, EnsembleScorer ...]

OODDetector
    ├── scorer:    BaseOODScorer  ← dependency-injected
    ├── simulator: Simulator | FlagSimulator
    ├── device:    str
    └── threshold: float (default 0.3)
```

The `OODDetector` depends on the **abstract** `BaseOODScorer`, not on `KDTreeScorer` directly. This means the scoring backend can be swapped (FAISS, Mahalanobis distance, ensemble variance) by passing a different scorer — without any change to `OODDetector` itself.

### `OODResult` dataclass

```python
@dataclass
class OODResult:
    confidence: float       # [0, 1]: how in-distribution is this mesh?
    is_ood:     bool        # True if confidence < threshold
    threshold:  float       # the threshold used (default 0.3)
    embedding:  np.ndarray  # [128] cloth / [256] CFD — the mesh's fingerprint
```

### The threshold

The default threshold of **0.3** means: if the confidence score is below 30%, the mesh is flagged as out-of-distribution and the API warns the user not to trust the prediction. This value was chosen empirically — it sits far enough below the typical in-distribution score that false positives (spuriously flagging valid designs) are rare, while still catching designs with radii, shapes, or poses that are far outside the training envelope.

### Typical usage

```python
detector = OODDetector.from_index_file(
    "runs/embedding_index.pkl",
    simulator=simulator,
    device="cpu"
)
result = detector.score(generated_graph)  # OODResult
if result.is_ood:
    print(f"Warning: confidence={result.confidence:.2f} — prediction may be unreliable")
```

---

## 7b. Theoretical Framing: JEPA Connection

This subsystem is **JEPA-adjacent** in the sense of LeCun (2022), *Joint Embedding Predictive Architectures*. The core idea in JEPA is to train an encoder that maps inputs into a space where semantically similar inputs cluster together — not in pixel/node space, but in a learned abstract representation space.

Here, the GNN encoder embeds full simulations into a latent space where **similar physics clusters together**: two cylinders of similar radii and similar inlet velocities land near each other in 256-dim space; a wildly different geometry lands far away. The confidence system then asks: *is this test point in the support of the training distribution?*

This is equivalent to **density estimation** in the embedding space: "is the new point in a region of space that has training-set density?" The KD-tree nearest-neighbor distance is a non-parametric density proxy — high density (many training points nearby) implies low distance implies high confidence. Mahalanobis distance and FAISS are alternative estimators for the same underlying question.

---

## 7c. Separate Confidence for Predict vs Generate Pages

There are **two distinct OOD questions** depending on context. They use different detectors backed by different feature spaces.

### Predict page — embedding-space OOD

The Predict page asks: *"given a specific mesh, how trustworthy is the predictor's rollout?"* This is answered by the standard `OODDetector` described above: a 256-dim GNN encoding of the simulation (dual-frame CFD, or single-frame cloth) compared against training-set embeddings via KD-tree. The embedding captures both geometry *and* physics boundary conditions.

### Generate page — parameter-space OOD

The Generate page asks a different question: *"is this design geometry similar to the training geometries?"* This is answered by `ParamSpaceOOD` in `extensions/confidence/ood_detector.py`. It builds a **KDTree over training design parameter vectors** — for CFD designs: `(cx, cy, r, v_inlet)` — and finds the nearest training design in that 4-dimensional parameter space.

```python
class ParamSpaceOOD:
    """KDTree over (cx, cy, r, v_inlet) training vectors — parameter-space OOD."""
```

This is a deliberately cheaper, more interpretable check: "have we trained on similar cylinder shapes and similar flow speeds?" It does not involve running the GNN at all. A design with `r=0.40` (3× outside training range) fails this check immediately, without needing an encoder forward pass.

**Summary:**

| Page | Detector | Feature space | Question |
|------|----------|---------------|----------|
| Predict | `OODDetector` (KDTreeScorer) | 256-dim GNN embedding | Is this mesh's physics in-distribution? |
| Generate | `ParamSpaceOOD` | 4-dim `(cx, cy, r, v_inlet)` | Is this design geometry in-distribution? |

---

## 8. When Does It Fire in Practice?

| Scenario | Expected Confidence | Reasoning |
|---|---|---|
| Cylinder, r=0.40 m (training max ≈ 0.12 m) | **Low (< 0.3)** | Geometry 3× outside training range; embedding lands far from all training points |
| Cylinder, r=0.08 m, cx=0.5, cy=0.2 | **High (> 0.7)** | Mid-range radius, standard position; very similar to many training samples |
| Cloth pose = linear interpolation of two training poses | **High (> 0.7)** | Encoder embedding is roughly a linear blend of two nearby training embeddings |
| Cloth with extreme deformation (10× training stress range) | **Low (< 0.3)** | Velocity features far exceed the normalizer's learned scale; embedding is an outlier |

---

## 9. Tradeoffs

1. **KD-trees degrade in high dimensions.** D=128 is well above the asymptotic-efficiency threshold for KD-trees (~20 dimensions). The structure still performs well in practice due to training-set clustering, but for very large N or more uniformly distributed embeddings, query time approaches linear.

2. **The confidence threshold (0.3) is a hyperparameter chosen empirically.** There is no theoretical derivation for this value. A dataset with tighter training distributions might warrant 0.5; a more diverse one might tolerate 0.15. It should be re-evaluated whenever the training data changes significantly.

3. **Embeddings come from the encoder only — they don't capture what the full network thinks.** The encoder sees normalized features and produces a compressed representation. The processor (15 message-passing steps) may still behave unpredictably for some inputs whose encoder embeddings appear in-distribution. A mesh that happens to encode close to a training point is not guaranteed to produce accurate processor output.

4. **We score at t=0 only — not across the rollout.** The OOD check is performed on the initial state of the mesh. During a rollout, the simulation evolves: a mesh that starts in-distribution can drift into OOD territory as time progresses (e.g., a fluid simulation that develops an unexpected vortex, or a cloth that stretches to extreme deformations). The current score does not detect that mid-rollout drift.

5. **The confidence score does not tell you *how wrong* the prediction is — only that it might be.** A confidence of 0.1 means "far from training data" but does not quantify the magnitude of the prediction error. The model could be 10% wrong or 300% wrong; the confidence score cannot distinguish. It is a flag, not an error bound.

---

## 10. Potential Enhancements

1. **FAISS `IndexIVFPQ` for million-scale embedding search.** Facebook's FAISS library uses inverted-file indexing with product quantization to support approximate nearest-neighbor search over millions of embeddings in milliseconds. If the training corpus scales to 100k+ designs, a FAISS-backed scorer can be dropped in by implementing `BaseOODScorer`.

2. **Mahalanobis distance.** The current score uses raw Euclidean distance — it treats all dimensions equally. Mahalanobis distance weights each dimension by the inverse covariance of the training distribution, accounting for the fact that some embedding dimensions vary much more than others across the training set. This tends to produce sharper, better-calibrated OOD scores at the cost of computing and inverting a 128×128 covariance matrix.

3. **Ensemble variance as uncertainty.** Train K independent instances of the predictor (e.g., K=5) with different random seeds. At inference, run all K models and measure the variance of their predictions. High variance across ensemble members is a signal that the model is uncertain — without needing a separate OOD index at all. Expensive at training time, but produces uncertainty estimates that reflect the full network's behavior, not just the encoder.

4. **MC Dropout for cheap uncertainty.** Leave dropout layers active during inference and run the forward pass K times. The variance of the K output fields is an approximation of the model's epistemic (knowledge) uncertainty. This is cheaper than a full ensemble and requires no changes to the index infrastructure — just run the existing model K times with `model.train()` mode enabled during inference.

5. **Per-timestep rollout embeddings.** Instead of scoring only the initial state, extract embeddings at every timestep of a rollout and track how the confidence score evolves over time. A trajectory that starts confident but whose confidence drops below threshold at step 50 signals that the simulation has drifted into OOD territory mid-rollout. This could be exposed as a confidence time-series in the UI alongside the predicted field animation.

## See also

- [[SUBSYSTEM_PREDICTOR]] — this subsystem borrows the predictor's encoder to extract 128-dim mesh embeddings
- [[SUBSYSTEM_GENERATOR]] — every generated candidate design is scored by this subsystem before being shown to the user
- [[SUBSYSTEM_API]] — the OOD detector is called from the `/api/generate` route for every candidate
