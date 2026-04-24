---
tags: [physicsai, subsystem, deep-dive]
created: 2026-04-10
aliases: [data, dataset, fpc-dataset]
---

# Data Processing Subsystem — Deep Dive

> [!info] Audience
> This document assumes familiarity with PyTorch `Dataset` and basic deep learning concepts (tensors, batches, loss functions). No prior knowledge of graph neural networks or computational fluid dynamics is required.

## Quick summary

- CFD data: flat memmap `.dat` + `.npz` metadata with CSR-style `indices` array
- Cloth data: one `.npz` per trajectory (`traj_XXXXX.npz`)
- Each `__getitem__` returns one `(t → t+1)` transition as a PyG `Data` object
- `np.memmap` means the full dataset never loads into RAM — OS pages in what's needed
- One-time preprocessing: TFRecords → `.dat`/`.npz` via `parse_tfrecord.py`

---

## 1. What this subsystem does

The data processing subsystem is responsible for turning raw physics simulation recordings — stored on disk as binary arrays — into PyTorch Geometric `Data` objects that the model can consume one sample at a time. It abstracts away two completely different on-disk formats (one for fluid flow, one for cloth simulation) behind a unified `Dataset.__getitem__` interface. Each returned sample is a single timestep transition: given the mesh state at time *t*, the network must predict the mesh state at time *t+1*. Everything else in the training loop — batching, shuffling, edge construction — happens downstream; this subsystem's sole job is to load the right nodes, connectivity, and fields for a given `(trajectory, timestep)` pair and pack them into a graph.

---

## 2. Raw data format

### What is a trajectory?

A **trajectory** is a time series of mesh states for one simulation run. Think of it as a short movie of a physical system: each frame is a snapshot of every mesh node's position and associated physical quantities (velocity, pressure, or 3-D world position). A single dataset split (train / valid / test) contains hundreds of trajectories, each typically 600 timesteps long.

### Common per-node fields

| Field | Shape | Description |
|---|---|---|
| `pos` / `mesh_pos` | `[N, 2]` | 2-D rest-configuration coordinates of each node (fixed for the whole trajectory) |
| `node_type` | `[N, 1]` | Integer label classifying each node (see §4) |
| `cells` | `[F, 3]` | Triangle connectivity — each row is three node indices forming one triangle |
| `velocity` | `[N, T, 2]` | 2-D velocity at each node across all T timesteps (CFD) |
| `world_pos` | `[T, N, 3]` | 3-D absolute position of each node at each timestep (cloth) |

`N` is the number of nodes, `F` the number of triangle faces, and `T` the trajectory length.

### CFD format — flat memmap + metadata

The cylinder-flow (`FpcDataset`) data is stored as two files per split:

- **`{split}.dat`** — a flat binary file treated as a memory-mapped array of shape `[N_total, T, 2]`, where `N_total` is the *sum* of node counts across **all** trajectories concatenated in the node dimension. Velocity values are `float32`.
- **`{split}.npz`** — a compressed NumPy archive containing:
  - `pos` `[N_total, 2]` — all mesh positions stacked
  - `node_type` `[N_total, 1]` — all node type labels stacked
  - `cells` `[F_total, 3]` — all triangle indices stacked (with a separate `cindices` offset array for the cell dimension)
  - `indices` `[n_traj + 1]` — the key: a cumulative offset array such that trajectory `i` occupies node rows `indices[i] : indices[i+1]` in every stacked array
  - `all_velocity_shape` — tuple `(N_total, T, 2)` needed to re-open the memmap with the correct shape

The `indices` trick is essentially a compressed sparse row (CSR) pointer array reused for variable-length axis slicing. It means all trajectories share one big flat file even though they have different node counts.

```
indices = [0, 1847, 3601, 5520, ...]   ← cumulative node counts
                    ↑
         trajectory 1 has nodes 1847..3600 in the flat arrays
```

### Cloth format — one file per trajectory

The flag-simple (`FlagDataset`) data uses a much simpler layout:

- **`{split}/traj_00000.npz`**, `traj_00001.npz`, … — one compressed NumPy file per trajectory, each containing `world_pos [T, N, 3]`, `mesh_pos [N, 2]`, `node_type [N, 1]`, `cells [F, 3]`.
- **`{split}_index.npz`** — a small index file with `n_traj` (total trajectory count) and `steps_per_traj` (array of per-trajectory lengths, since cloth trajectories can vary in length).

Each trajectory file is fully self-contained: load it and you have everything needed to build any sample from that trajectory.

---

## 3. Dataset `__getitem__` in detail

### CFD — `FpcDataset.__getitem__`

The dataset is indexed by a flat integer `index` ranging over *all* `(trajectory, timestep)` pairs. The first step is to reverse-map this flat index back to a trajectory and an in-trajectory timestep:

```python
tra_index        = index // self.num_sampes_per_tra   # which trajectory
tra_sample_index = index % (self.tra_len - 1)         # which timestep t within it
```

`num_sampes_per_tra` equals `T - 1` because the last timestep has no *t+1* to predict. With those two indices in hand:

1. **Slice static fields** from the metadata arrays using the `indices` and `cindices` offset arrays:
   ```python
   tra_start_index  = self.meta["indices"][tra_index]
   tra_end_index    = self.meta["indices"][tra_index + 1]
   pos       = self.meta["pos"][tra_start_index:tra_end_index]        # [N, 2]
   node_type = self.meta["node_type"][tra_start_index:tra_end_index]  # [N, 1]
   cells     = self.meta["cells"][ctra_start_index:ctra_end_index]    # [F, 3]
   ```

2. **Slice the velocity (or pressure) at timesteps `t` and `t+1`** from the memmap — this is the only disk I/O in the hot path:
   ```python
   tra_velocity = self.fp[tra_start_index:tra_end_index, tra_sample_index]      # [N, 2]
   tra_target   = self.fp[tra_start_index:tra_end_index, tra_sample_index + 1]  # [N, 2]
   ```

3. **Build node features and target**, then wrap in a PyG `Data` object:
   ```python
   x = np.concatenate([node_type, tra_velocity], axis=-1)  # [N, 3]
   y = tra_target                                            # [N, 2]
   graph = Data(x=..., pos=..., face=cells.T, y=...)
   ```

### Pressure mode branch

When `target_field == "pressure"`, the dataset opens a second memmap (`{split}_pressure.dat`, shape `[N_total, T, 1]`) and substitutes pressure for velocity in both input features and target:

```python
if self.target_field == "pressure":
    pressure_t   = self.fp_pressure[tra_start_index:tra_end_index, tra_sample_index]     # [N, 1]
    pressure_tp1 = self.fp_pressure[tra_start_index:tra_end_index, tra_sample_index + 1] # [N, 1]
    x = np.concatenate([node_type, pressure_t], axis=-1)   # [N, 2]
    y = pressure_tp1                                         # [N, 1]
else:
    tra_velocity = self.fp[tra_start_index:tra_end_index, tra_sample_index]      # [N, 2]
    tra_target   = self.fp[tra_start_index:tra_end_index, tra_sample_index + 1]  # [N, 2]
    x = np.concatenate([node_type, tra_velocity], axis=-1)  # [N, 3]
    y = tra_target                                            # [N, 2]
```

Everything else (pos, cells, graph construction) is identical for both modes.

### Cloth — `FlagDataset.__getitem__`

Because cloth trajectories can have variable lengths, `FlagDataset` uses a cumulative step count array (`_cum_steps`) and `np.searchsorted` to map a flat index to `(traj_idx, t)`:

```python
traj_idx = int(np.searchsorted(self._cum_steps[1:], index, side="right"))
t        = index - self._cum_steps[traj_idx]
```

Then the trajectory `.npz` is loaded on demand and three timesteps are extracted:

- `world_pos_t` — current frame `[N, 3]`
- `world_pos_tp1` — next frame `[N, 3]` → regression target `y`
- `world_pos_prev` — previous frame `[N, 3]` (clamped to `t` at the first frame, encoding zero velocity)

The returned graph carries more fields than the CFD case because the cloth model needs both the current 3-D position and the 2-D rest-configuration mesh coordinates:

```
graph.x         [N, 4]  concat(world_pos_t, node_type)
graph.prev_x    [N, 3]  world_pos_{t-1}
graph.pos       [N, 2]  mesh_pos (2-D rest config)
graph.world_pos [N, 3]  world_pos_t
graph.y         [N, 3]  world_pos_{t+1}
```

---

## 4. Node types

Every node in a mesh carries an integer **node type** label. This label encodes the node's physical role and determines how the model (and the simulation boundary conditions) must treat it. Passing it as a node feature lets the GNN learn type-specific update rules without any hard-coded branching.

| Constant | Value | Used in | Meaning |
|---|---|---|---|
| `NORMAL` | 0 | Both | Interior mesh node; fully governed by the learned dynamics |
| `OBSTACLE` | 1 | CFD | Surface of the cylinder; enforced no-slip boundary (fixed velocity = 0) |
| `AIRFOIL` | 2 | CFD | Airfoil surface variant of obstacle nodes |
| `HANDLE` | 3 | Cloth | Corner/edge nodes that are **pinned** — position is held fixed by the simulation |
| `INFLOW` | 5 | CFD | Inlet boundary; velocity is overwritten with a prescribed inlet profile `v_inlet` |
| `OUTFLOW` | 6 | CFD | Outlet boundary; pressure is set to zero (open boundary) |
| `WALL_BOUNDARY` | 6 | CFD | Channel wall; treated as a no-slip surface |

**Why HANDLE nodes matter:** In cloth simulation, the flag is attached to a pole at its leading edge. Those attachment nodes are type `HANDLE` — the model's predicted displacement for them is discarded at rollout time and replaced with the known fixed position. Without this, the flag would drift off the pole within a few steps.

**Why INFLOW nodes matter:** At the left boundary of the CFD domain, fluid enters at a fixed velocity profile. During rollout, the model's output for INFLOW nodes is overridden with `v_inlet` each step, enforcing the physical inlet condition regardless of what the network predicts.

---

## 5. Memory layout — the memmap trick

The full CFD training split contains on the order of **1–2 GB** of velocity data. Loading it into a NumPy array at startup would consume all available RAM on most workstations and make multi-worker `DataLoader` processes impossible.

`np.memmap` solves this by mapping the file into the process's **virtual address space** without reading it into physical RAM. The OS kernel manages a page cache: when a slice like `self.fp[1847:3600, 42]` is accessed, only the 4 KB pages covering that byte range are fetched from disk and cached. Pages that are no longer needed are silently evicted when memory pressure rises.

```
Disk file: train.dat  [N_total=287,000 nodes × T=600 steps × 2 floats × 4 bytes ≈ 1.38 GB]
                            │
                     np.memmap (mode='r')
                            │
           ┌────────────────┴──────────────────┐
           │  Virtual address window            │
           │  OS pages in only what you touch   │
           └───────────────────────────────────┘
                     ↑ DataLoader workers share the same fd
```

**Tradeoff:** Random access into a cold memmap file causes **page faults** (disk seeks). This is slower than RAM but fits on any machine. With PyTorch's `DataLoader(num_workers=4)`, four worker processes each keep their own recently-accessed pages warm, which partially amortises the cost for nearby indices within the same trajectory.

> [!note] memmap vs. HDF5 — different tools for different access patterns
> **memmap is correct for training data** (random access dominates — the DataLoader jumps to arbitrary `(trajectory, timestep)` pairs during shuffled iteration; the OS page cache handles this efficiently).
> **HDF5 is correct for rollout results** (partial reads matter — the Visualize page loads individual frames on demand; `load_timestep(key, t)` reads exactly one chunk without deserialising the whole file). See §10 (Storage Layer) for the HDF5 implementation.

---

## 6. Code snippets

### `FpcDataset.__getitem__` — index arithmetic and memmap slice

```python
def __getitem__(self, index: int) -> Data:
    # --- Step 1: unpack flat index ---
    tra_index        = index // self.num_sampes_per_tra   # trajectory number
    tra_sample_index = index % (self.tra_len - 1)         # timestep t within trajectory

    # --- Step 2: node/cell ranges from the offset arrays ---
    tra_start_index  = self.meta["indices"][tra_index]
    tra_end_index    = self.meta["indices"][tra_index + 1]
    ctra_start_index = self.meta["cindices"][tra_index]
    ctra_end_index   = self.meta["cindices"][tra_index + 1]

    pos       = self.meta["pos"][tra_start_index:tra_end_index]       # [N, 2]
    node_type = self.meta["node_type"][tra_start_index:tra_end_index] # [N, 1]
    cells     = self.meta["cells"][ctra_start_index:ctra_end_index]   # [F, 3]

    # --- Step 3: velocity vs. pressure branch ---
    if self.target_field == "pressure":
        pressure_t   = self.fp_pressure[tra_start_index:tra_end_index, tra_sample_index]
        pressure_tp1 = self.fp_pressure[tra_start_index:tra_end_index, tra_sample_index + 1]
        x = np.concatenate([node_type, pressure_t], axis=-1)   # [N, 2]
        y = pressure_tp1                                         # [N, 1]
    else:
        tra_velocity = self.fp[tra_start_index:tra_end_index, tra_sample_index]      # [N, 2]
        tra_target   = self.fp[tra_start_index:tra_end_index, tra_sample_index + 1]  # [N, 2]
        x = np.concatenate([node_type, tra_velocity], axis=-1)  # [N, 3]
        y = tra_target                                            # [N, 2]

    # --- Step 4: wrap in PyG Data ---
    return Data(
        x    = torch.as_tensor(x.copy(),    dtype=torch.float32),
        pos  = torch.as_tensor(pos.copy(),  dtype=torch.float32),
        face = torch.as_tensor(cells.T.copy(), dtype=torch.int64),
        y    = torch.as_tensor(y.copy(),    dtype=torch.float32),
    )
```

> [!note] Why `.copy()`?
> A memmap slice returns a view into the page cache. Passing a view to `torch.as_tensor` would keep the page locked until the tensor is freed. Copying to a fresh array releases the page immediately and avoids subtle memory leaks across workers.

---

## 7. Data pipeline: from raw TFRecords to `.dat` / `.npz`

DeepMind's original MeshGraphNets release distributes data as **TFRecord** files — TensorFlow's binary serialisation format. Parsing them at training time has two problems: it is slow (each record must be decoded from a protobuf), and it requires a TensorFlow installation that is incompatible with the project's PyTorch environment.

The one-time preprocessing pipeline solves this:

```
DeepMind TFRecord files
        │
        ▼
parse_tfrecord.py (requires tensorflow < 1.15 in a separate venv)
        │
        ├─ CFD: iterate all trajectories
        │    ├─ transpose velocity [T,N,2] → [N,T,2], write into pre-allocated memmap
        │    ├─ collect pos / node_type / cells lists
        │    └─ write split.npz (stacked arrays + indices offsets)
        │
        └─ Cloth: write one traj_XXXXX.npz per trajectory
                  + {split}_index.npz with n_traj and steps_per_traj
```

After preprocessing, TensorFlow is never touched again. Training uses only NumPy `.npz` and `.dat` files, which load in microseconds and require no parsing overhead.

---

## 8. Tradeoffs

| # | Tradeoff | Impact |
|---|---|---|
| 1 | **Fixed timestep stride (always t → t+1)** | The model sees only one-step transitions. It learns single-step accuracy but may accumulate errors over long rollouts because it was never trained to correct multi-step drift. |
| 2 | **No multi-step supervision** | Training loss is computed purely on the *next* state. The model has no gradient signal from states further ahead, making rollout stability entirely an emergent property. |
| 3 | **Memmap random access patterns** | When `DataLoader` shuffles with `shuffle=True`, consecutive indices often span different trajectories and distant node ranges, causing scattered page faults. Sequential or trajectory-grouped sampling would be faster but breaks i.i.d. assumptions. |
| 4 | **Variable N per trajectory prevents naive batching** | Different trajectories have different node counts. PyTorch's default `collate_fn` cannot stack tensors of different sizes, so a custom collator or PyG's `Batch.from_data_list` is required. Without it, batch size is effectively forced to 1. |
| 5 | **Pressure data doubles storage** | Storing the pressure memmap alongside velocity doubles the disk footprint for CFD data. Pressure is only needed for the `target_field="pressure"` training mode; it is wasted space for the default velocity mode. |
| 6 | **No in-memory caching for cloth** | `FlagDataset` opens a new `.npz` file on every `__getitem__` call. For small datasets this is fine, but for large splits it creates repeated decompression overhead. |

---

## 9. Potential enhancements

1. **Multi-step rollout supervision** — instead of always predicting t → t+1, randomly sample a rollout horizon k ∈ {1, 2, …, K} and unroll the model k steps, accumulating loss at each step. This directly trains the model to stay stable over longer horizons and is the standard fix for error accumulation.

2. **HDF5 instead of flat memmap** — replace the `.dat` + `.npz` pair with a single HDF5 file (via `h5py`). HDF5 supports named datasets, chunked storage with configurable chunk shape, and built-in compression. Chunking along the trajectory-step dimension (e.g., chunk shape `[N_traj, 1, 2]`) dramatically improves random-access performance compared to a flat file because each disk read fetches exactly one timestep per trajectory rather than pulling in unneeded adjacent nodes.

3. **Graph batching with `PyG Batch.from_data_list`** — PyTorch Geometric's `Batch` object handles variable-size graphs by renumbering node indices and concatenating everything into one large disconnected graph. Using a custom `collate_fn` that calls `Batch.from_data_list` enables batch sizes > 1, improving GPU utilisation significantly.

4. **Dataset augmentation for cloth** — cloth simulation is invariant to global rotation and reflection in 3-D. Applying random SO(3) rotations to `world_pos` and `mesh_pos` at load time effectively multiplies the dataset size, improves generalisation to unseen flag orientations, and is cheap to implement as a transform passed to the dataset.

5. **Streaming from cloud storage** — for large-scale training on cloud VMs where local disk is expensive, the per-trajectory `.npz` layout used by `FlagDataset` maps naturally to an object store (e.g., S3 or GCS). A thin `fsspec`-backed loader could replace `np.load(traj_path)` with a streaming read, enabling training without a full local copy of the dataset.

6. **Trajectory-grouped sampling** — replace uniform random shuffling with a two-level sampler: first pick a random trajectory, then pick a random timestep within it. This keeps consecutive batches in the same node-range of the memmap, dramatically reducing page-fault rate while preserving approximate i.i.d. sampling at the trajectory level.

---

## 10. Storage Layer — ResultRepository Pattern

Rollout results must be saved to disk and re-loaded on demand (e.g. by the Visualize page, which loads individual timestep frames). The `storage/` folder provides a clean abstraction over three different backends behind a single Protocol.

### `storage/protocols.py` — the Protocol

```python
@runtime_checkable
class ResultRepository(Protocol):
    def save(self, key: str, data: Any) -> None: ...
    def load(self, key: str) -> Any: ...
    def load_timestep(self, key: str, t: int) -> Any: ...
    def list(self) -> list[str]: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def get_path(self, key: str) -> Path: ...
```

**Why `Protocol` instead of `ABC`:** Structural typing — any class that defines the right seven methods satisfies `ResultRepository` without inheriting from it. `isinstance(repo, ResultRepository)` returns `True` for all compliant implementations, including duck-typed mocks in tests. No import of the base class is required, which avoids coupling new backend implementations to this module.

### `storage/pkl_repository.py` — pickle backend

`PklResultRepository` serialises each result with `pickle`. Simple, zero extra dependencies, no queryable structure. Suitable for development and local one-off runs. Not suitable for partial reads.

### `storage/hdf5_repository.py` — HDF5 backend

`HDF5ResultRepository` writes gzip-level-4 compressed HDF5 files (via `h5py`). The dataset is stored with chunk shape `(1, N, D)` — one chunk per timestep. This makes `load_timestep(key, t)` a single-chunk read:

```python
def load_timestep(self, key: str, t: int) -> np.ndarray:
    with h5py.File(self.get_path(key), "r") as f:
        return f["data"][t]   # reads exactly one (1, N, D) chunk
```

The file is **not** deserialised in full — the OS fetches only the bytes for chunk `t`. This is the correct backend for the Visualize page, which scrubs through frames interactively. Contrast with memmap (§5), which is correct for training-time random access across trajectories.

### `storage/factory.py` — backend selection

`StorageFactory.create()` reads `runs/storage_config.json` and returns the configured backend:

```json
{ "result_backend": "hdf5" }
```

| Config value | Backend returned |
|---|---|
| `"pkl"` | `PklResultRepository` |
| `"hdf5"` | `HDF5ResultRepository` |
| `"zarr"` | `ZarrArchive` |

**Adding a new backend:** implement the seven-method Protocol, add one line to the factory's dispatch dict. No changes needed in any caller.

### `storage/zarr_archive.py` — cloud-native archival

`ZarrArchive` uses Blosc/LZ4 compression. Each chunk is stored as an independent file, which maps cleanly to S3 objects (one HTTP PUT per chunk). A `.zarr.ok` sentinel file (same pattern as `.dat.ok`) marks a completed write. Used for long-term archival and cloud push; not the default backend for interactive use.

---

## 11. Ingest Pipeline — Adding New Solvers

The `ingest/` folder implements the **Open/Closed Principle** for new physics solvers: the pipeline is open for extension (new adapters) but closed for modification (no changes to the pipeline stages themselves).

### `ingest/protocols.py` — SolverAdapter Protocol

```python
class SolverAdapter(Protocol):
    def list_splits(self) -> list[str]: ...
    def load_split(self, split: str) -> dict: ...
    @property
    def source_path(self) -> Path: ...
    @property
    def name(self) -> str: ...
```

Any class with these four members is a valid adapter.

### Provided adapters

| File | Class | What it wraps |
|---|---|---|
| `ingest/adapters/tfrecord.py` | `TFRecordAdapter` | Existing `.dat` memmap loading (backwards compatibility) |
| `ingest/adapters/openfoam.py` | `OpenFOAMAdapter` | Stub — raises `NotImplementedError`. Template for future OpenFOAM support. |

Adding support for a new solver means writing one new adapter class. The pipeline stages are untouched.

### `ingest/pipeline.py` — five composable stages

`IngestPipeline.run(adapter)` chains five independent stages:

```
adapter.list_splits() + adapter.load_split()
        │ harvest.py
        ↓
raw data dict
        │ validate.py — check shapes, NaN/Inf, field completeness
        ↓
validated data
        │ normalise.py — compute running stats, normalise features
        ↓
normalised data
        │ write.py — write .npz files + manifest.json
        ↓
written files
        │ index.py — update DVC dvc.yaml
        ↓
DVC index updated
```

Each stage is a separate module with a single `run(data) → data` function. This makes stages individually testable and reorderable.

---

## 12. DVC Data Versioning

DVC (Data Version Control) adds a **dependency graph** on top of git for large data files.

- DVC tracks that `train.dat` was produced **from** `train.tfrecord` **by** `parse_tfrecord.py`.
- If the source changes, DVC knows the output is stale → `dvc repro` reruns only the affected stages.
- Unlike **git-lfs**, which just stores blobs, DVC understands the full pipeline graph.
- `dvc push` uploads `.dat` files to a configured S3/GCS remote, keeping the git repository small (only `.dvc` pointer files are committed).

The `index.py` ingest stage updates `dvc.yaml` automatically so new adapter outputs are immediately tracked.

---

## 13. Result Retention

```
python -m result.retention --keep 10 [--dry-run]
```

`result/retention.py` keeps the N most recent results (ordered by creation time) and deletes the oldest.

- `--keep N` — number of results to retain (default 10).
- `--dry-run` — print what *would* be deleted without actually deleting anything.

> [!tip] Always implement `--dry-run` for destructive operations
> Any script that deletes data should support a preview mode. Users can audit the output before committing to the delete. `result.retention` is the project's reference implementation of this pattern.

---

## See also

- [[SUBSYSTEM_PREDICTOR]] — consumes the `Data` objects produced by this subsystem during training and rollout
- [[SUBSYSTEM_GENERATOR]] — also reads raw trajectories in Phase 0 to extract compact design parameters
