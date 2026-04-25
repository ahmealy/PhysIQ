---
tags: [physicsai, data, memmap, hdf5, confidence, train_diameter]
created: 2026-04-25
aliases: [memmap, dat-npz, train-diameter, confidence-scoring-detail]
---

# Training Data Access & Confidence Score Mechanics

Two questions answered in full: how `.dat` + `.npz` gives O(1) random access,
and exactly how `train_diameter` is computed and used in confidence scoring.

---

## Part 1 — How `.dat` + `.npz` gives O(1) access

### The layout on disk

After `parse_tfrecord.py` runs, you get two files per split:

```
data/
  train.dat       ← flat binary array, shape [N_total, T, D], float32
  train.npz       ← metadata: node indices, mesh positions, connectivity
  train.dat.ok    ← sentinel: parse completed without interruption
```

**`N_total`** is the sum of node counts across ALL trajectories packed end-to-end in the node dimension. If you have 1000 trajectories each with ~1823 nodes, `N_total ≈ 1,823,000`.

**`T`** = 600 (timesteps per trajectory).

**`D`** = 2 for velocity `(vx, vy)`, or 1 for pressure.

The `.npz` holds the **CSR-style index arrays** that say where each trajectory starts and ends:

```python
indices  = [0, 1823, 3651, 5490, ...]   # node-space boundaries
cindices = [0, 3604, 7218, ...]          # cell-space boundaries (for mesh faces)
pos      = [N_total, 2]                  # mesh node positions (same for all t)
node_type= [N_total, 1]                  # node type labels
cells    = [C_total, 3]                  # triangle connectivity
```

### The memmap

```python
# dataset/fpc.py  __init__()
vel_shape = self.meta["all_velocity_shape"]   # (N_total, 600, 2)
self.fp = np.memmap(data_path, dtype="float32", mode="r", shape=vel_shape)
```

`np.memmap` maps the file into the process's **virtual address space** — it does NOT read the file into RAM. It registers a virtual memory region and tells the OS: "when I access address X, load the corresponding bytes from the file."

The entire 8 GB file is *addressable* but not *loaded*. The OS page cache brings in only the pages that are actually touched.

### __getitem__ — the O(1) path

```python
def __getitem__(self, index: int) -> Data:
    # Step 1: integer division to decode (trajectory, timestep) — O(1) arithmetic
    tra_index        = index // self.num_sampes_per_tra   # which trajectory
    tra_sample_index = index %  self.num_sampes_per_tra   # which timestep

    # Step 2: look up byte offsets from .npz — O(1) array index
    tra_start = self.meta["indices"][tra_index]
    tra_end   = self.meta["indices"][tra_index + 1]

    # Step 3: read velocity — O(1) pointer arithmetic + one pread() syscall
    velocity = self.fp[tra_start:tra_end, tra_sample_index]   # [N_traj, 2]
    target   = self.fp[tra_start:tra_end, tra_sample_index + 1]
```

Step 3 compiles to:

```
byte_offset = (tra_start * T * D  +  tra_sample_index * D) * 4 bytes
address     = mmap_base_ptr + byte_offset
```

That's **one pointer addition and one memory fetch** — O(1) regardless of dataset size. The OS decides whether those bytes are already in the page cache (free) or need a disk read (~0.1 ms for SSD).

### What the OS does behind the scenes

```
First access to trajectory 42, step 317:
  1. CPU calculates byte_offset = 42_start * 600 * 2 * 4 + 317 * 2 * 4
  2. OS page fault: that 4KB page is not in RAM yet
  3. OS reads the page from disk into page cache
  4. Subsequent accesses to the same page: cache hit, ~100 ns

Training loop shuffles randomly over all samples:
  Hot trajectories stay in page cache (OS manages this automatically)
  Cold trajectories cause page faults → disk reads
  With 8 DataLoader workers prefetching: pipeline hides the latency
```

### Why the .npz is loaded fully into RAM

The `.npz` is tiny — just the index arrays and mesh topology, maybe 50–200 MB depending on dataset. It's loaded fully at init:

```python
tmp = np.load(meta_path, allow_pickle=True)
self.meta = {key: tmp[key] for key in meta_keys}
```

This is intentional. The index arrays are accessed on every single `__getitem__` call (`indices[tra_index]`). If they were memmaped too, every index lookup would be a separate memory fetch. Loading them into RAM gives guaranteed O(1) with no page fault risk.

---

## Part 2 — Could this be HDF5? Would it be better?

### For training data: No

HDF5 stores data in **compressed chunks**. Every random `__getitem__` during training would:

1. Seek to the right chunk → same as memmap
2. **Decompress the entire chunk** → extra CPU work (gzip is not random-access within a chunk)
3. Copy the decompressed bytes → extra allocation

```
memmap random access:       ~0.05 ms  (page cache read, no decompression)
HDF5 gzip random access:    ~3–8 ms   (decompress full chunk first)
→ HDF5 is 60–160× slower per sample for random access
```

With 8 DataLoader workers and a fast GPU, those milliseconds add up — the GPU ends up idle waiting for data.

### For results storage: Yes, and we already use it

The **access pattern is completely different** for rollout results:

| | Training data | Rollout results |
|--|--|--|
| Write pattern | Once (parse time) | Once (per rollout) |
| Read pattern | Millions of random (traj, t) reads | Occasional: load frame 47, load metadata |
| What matters | Seek speed | Compression ratio, partial reads |

`HDF5ResultRepository` stores results with chunk shape `(1, N, D)` — one chunk per timestep:

```python
dset.create_dataset("velocity", shape=(600, N, 2),
                    chunks=(1, N, 2), compression="gzip", compression_opts=4)
```

`load_timestep(key, t=47)` reads **exactly one chunk** — only timestep 47 is decompressed. The other 599 timesteps are never touched. This is what makes the Visualize page fast: frame scrubbing reads one chunk at a time, not the full file.

### The rule of thumb

| Use case | Right format | Why |
|----------|-------------|-----|
| Training DataLoader random access | `.dat` memmap | O(1) seek, no decompression |
| Rollout results, partial timestep reads | HDF5 `chunks=(1,N,D)` | Per-chunk gzip, efficient frame scrubbing |
| Cloud storage / archival | Zarr Blosc/LZ4 | S3-native, LZ4 is 10× faster than gzip |

### What if you had to use HDF5 for training?

Switch to **LZ4 compression** and chunk by whole trajectory:

```python
# chunk = one full trajectory = one decompression per sample
chunks=(1, 600, 2), compression="lz4"
```

LZ4 decompresses at ~3 GB/s (vs gzip at ~300 MB/s), so one trajectory chunk (~1 MB) decompresses in ~0.3 ms. Still 6× slower than memmap, but workable. This is what Zarr does under the hood.

---

## Part 3 — train_diameter in confidence scoring

### What it is

`train_diameter` is a **single scalar** that represents how spread out the training embeddings are in embedding space. It acts as the normalisation constant in the confidence score formula:

```python
confidence = clip(1 - d_min / train_diameter, 0, 1)
```

Where `d_min` = distance from the test embedding to its nearest training neighbour.

If `d_min = 0`: test point is identical to a training point → score = 1.0 (fully in-distribution)  
If `d_min = train_diameter`: test point is as far as a "typical" training point is from its neighbours → score = 0.0 (out-of-distribution)  
If `d_min > train_diameter`: score clips to 0.0

### Exactly how it's computed

From `confidence/index.py`:

```python
# After building the KDTree over training embeddings:

# Query each training point against itself + 5 nearest neighbours
# k=6: index 0 = the point itself (distance 0), indices 1-5 = 5 nearest neighbours
dists, _ = self._scipy_tree.query(self.embeddings, k=6)

# dists[:, 5] = distance from each training point to its 5th nearest neighbour
# Take the 95th percentile of those distances
self.train_diameter = float(np.percentile(dists[:, 5], 95))
```

Step by step:

```
1. embeddings: [N_train, 256] — one 256-dim vector per training trajectory

2. For each training trajectory i:
   find its 6 nearest neighbours in the training set
   (k=6 because index 0 is the point itself, distance=0 → skip it)
   d_5nn[i] = distance to the 5th nearest training neighbour

3. train_diameter = 95th percentile of {d_5nn[0], d_5nn[1], ..., d_5nn[N-1]}
```

### Why 5th nearest neighbour (k=5)?

Using the **5th nearest** rather than the 1st gives a more stable measure of local density. The 1st nearest neighbour distance is noisy — two nearly identical trajectories in the training set would give near-zero distances. The 5th nearest neighbour captures the "typical radius of the local neighbourhood" — how far you need to go to find 5 similar training examples.

### Why 95th percentile instead of max or mean?

```
max:  sensitive to outliers — one weird isolated training sample
      would inflate train_diameter → everything looks in-distribution
      
mean: hides the tail — most training points are close together,
      but the mean is pulled up by sparse regions

95th percentile: ignores the 5% most isolated training points (outliers)
                 but still reflects the "outer boundary" of the main cluster
                 robust to a few unusual training trajectories
```

Concrete example:

```
Training set: 1000 trajectories
d_5nn values (sorted): [0.12, 0.14, 0.15, ..., 0.98, 1.2, 4.5, 8.1]
                                                 ^95th ^outliers

95th percentile ≈ 0.98   ← train_diameter
max             = 8.1    ← would be misleading
mean            ≈ 0.45   ← hides the sparse outer region
```

### How it flows into the score

```python
# At inference (confidence/index.py  score())
def score(self, embedding: np.ndarray) -> float:
    d_min, _ = self._scipy_tree.query(embedding.reshape(1, -1), k=1)
    return float(np.clip(1.0 - d_min / (self.train_diameter + 1e-12), 0.0, 1.0))
```

```
Test embedding: 256-dim vector from GNN encoder on test simulation

d_min = distance to nearest training embedding in 256-dim space

confidence = clip(1 - d_min / 0.98, 0, 1)

d_min = 0.0   → score = 1.00  (identical to a training simulation)
d_min = 0.49  → score = 0.50  (halfway to the training boundary)
d_min = 0.98  → score = 0.00  (at the boundary — OOD threshold)
d_min = 1.50  → score = 0.00  (clipped, clearly OOD)
```

### The full picture — where train_diameter fits

```
BUILD TIME (once, after training):
  confidence/build_index.py
    for each training trajectory:
      embed(frame_0) → [128]
      embed(frame_5) → [128]
      concat         → [256]
    KDTree.build(all_embeddings)
    query(all_embeddings, k=6) → compute d_5nn
    train_diameter = percentile(d_5nn[:, 5], 95)
    save index (embeddings + tree + train_diameter + checkpoint_hash)

INFERENCE TIME (per prediction):
  NearestNeighborIndex.score(test_embedding)
    d_min = KDTree.query(test_embedding, k=1)
    return clip(1 - d_min / train_diameter, 0, 1)
```

### Is k=5 the right choice? Could you score OOD while still being similar to training data?

This is worth unpacking carefully because **two different k values serve two different roles**:

```python
# k=6 used for train_diameter (calibration, once at build time)
dists, _ = self._scipy_tree.query(self.embeddings, k=6)
train_diameter = np.percentile(dists[:, 5], 95)

# k=1 used for scoring (inference, per request)
d_min, _ = self._scipy_tree.query(test_embedding, k=1)
score = clip(1 - d_min / train_diameter, 0, 1)
```

`train_diameter` is only a **scale reference** — it answers "how big is a typical local neighbourhood in training space?" The actual score is determined entirely by `d_min`, which is distance to your **single nearest** training point.

#### When the concern doesn't apply

If `d_min` is large enough to give score ≈ 0, you are genuinely far from every training point — farther than the typical 5-NN radius. If you were close to any training point, `d_min` would be small and the score would be high, regardless of `train_diameter`.

#### When the concern IS valid — "surrounded but unmatched"

The real vulnerability is a different failure mode:

```
Test point sits in the middle of a training cluster,
moderately close to many training points,
but not very close to any single one.

d_min = 0.7   train_diameter = 0.98
score = 1 - 0.7/0.98 = 0.29  → flagged as borderline OOD

Reality: 50 training neighbours at distance 0.7–0.9
         Test point is well inside the training distribution
```

Using k=1 for `d_min` misses this case. A k-NN **density score** would catch it:

```python
# average distance to k nearest, not just the closest one
dists, _ = tree.query(test_embed, k=5)
density_score = 1 - dists.mean() / train_diameter
```

#### Is k=5 for train_diameter principled?

No — it's a heuristic from the k-NN classification literature (same reasoning as choosing k=5 for k-NN classifiers: small enough to capture local density, large enough to be stable). Different k values produce very different thresholds:

| k for train_diameter | Effect |
|---|---|
| k=1 | `train_diameter` is tiny → nearly everything flags OOD |
| k=5 | Moderate — reflects "how far until you have 5 similar examples" |
| k=20 | `train_diameter` grows → threshold looser → harder to flag OOD |

The choice of 5 is not derived from the physics or the embedding geometry. It's a reasonable default.

#### What would be better

Three alternatives, in increasing complexity:

**1. k-NN density score** (most impactful fix — changes the scoring formula):
```python
dists, _ = tree.query(test_embed, k=5)
score = clip(1 - dists.mean() / train_diameter, 0, 1)
```

**2. Adaptive local threshold** — instead of one global `train_diameter`, use the local 5-NN radius of the nearest training point:
```python
# local_5nn_dists[i] = 5th-NN distance for training point i (precomputed at build time)
nearest_idx = tree.query(test_embed, k=1)[1][0]
local_diam = local_5nn_dists[nearest_idx]
score = clip(1 - d_min / local_diam, 0, 1)
```
Dense clusters get a tight threshold; sparse regions get a looser one.

**3. Tune k at build time** — use a held-out validation set with known-good and known-OOD examples, sweep k ∈ {1, 5, 10, 20}, pick the k that best separates them by AUROC. If FAISS is active (≥100k trajectories), k=20 queries are essentially free.

#### Current limitation summary

The k=5 choice for `train_diameter` is reasonable. The more significant design limitation is using k=1 for `d_min` at scoring time — that's what misses "surrounded but unmatched" test points. If users report low confidence on clearly reasonable inputs, the first fix to try is switching from k=1 to a k-NN density score.

---

### Stale index detection

`train_diameter` is computed FROM a specific set of model weights (the encoder that produced the embeddings). If the model is retrained, the embedding space changes — the old `train_diameter` is meaningless. This is why the index stores a SHA-256 hash of the checkpoint:

```python
# On load:
stored_hash   = index["checkpoint_hash"]   # e.g. "a3f7b291c4..."
expected_hash = checkpoint_hash(path)       # recompute from current file

if stored_hash != expected_hash:
    raise IndexStaleError("Model has been retrained — rebuild the confidence index")
```

`train_diameter` is stored inside the index file alongside the embeddings and hash, so it doesn't need to be recomputed at inference time.
