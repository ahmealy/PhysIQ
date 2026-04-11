# MeshGraphNets Pipeline — From Raw Simulation to Production at Billion Scale

> A step-by-step walkthrough you can narrate while demoing the UI. Each section covers what the system does today, the design decisions behind it, and what you would change to scale to billions of mesh nodes across thousands of simulations.

---

## Stage 1 — Data Ingestion & Harvest

### What the system does today

Physics simulation data arrives as TFRecord files produced by DeepMind's open-source simulator. The ingestion scripts convert them into two domain-specific formats:

- **CFD / Cylinder Flow** — a single memory-mapped binary (`train.dat`, `test.dat`) plus a companion index file (`train.npz`, `test.npz`). The memmap holds the entire velocity field as a contiguous `float32` array of shape `[total_nodes, T, 2]`. The index file records `indices` (cumulative node offsets per trajectory) and `all_velocity_shape`.
- **Cloth / Flag Simple** — per-trajectory `.npz` files (`traj_00000.npz`, ...) each holding `world_pos [T, N, 3]`, `velocity [T, N, 3]`, `node_type [N]`, and edge arrays. A small sidecar `{split}_index.npz` records `n_traj` and `steps_per_traj`.

Both domains share a single `DOMAINS` dict in `api/state.py` that carries the data directory, checkpoint path, time-step size (`dt`), and availability flag. A domain is marked unavailable at startup if the data directory does not exist.

**Design choice — why two different layouts:**  
CFD has a fixed mesh topology across all trajectories (the cylinder mesh never changes), so one big memmap with a flat node offset index is optimal — a single `np.memmap` open and a slice gives you any trajectory instantly. Cloth has variable node count per trajectory (different cloth resolutions), so per-trajectory files are the natural fit.

### Billion-scale upgrade path

| Pain point | Today | At billion scale |
|---|---|---|
| Format heterogeneity (OpenFOAM, VTK, FEniCS, STL, CGNS) | Only TFRecord/npz | Unified ingest layer with format adapters (meshio, PyVista, pyvista-readers for VTK/CGNS/ExodusII). Write once to a canonical format. |
| Storage layout | Single-host memmap | **Zarr** backed by object storage (S3 / GCS). Zarr is chunk-compressed, cloud-native, and supports partial reads. Chunks of `[1_000, T, C]` align with per-trajectory access patterns. |
| Metadata catalog | A Python dict | A metadata database (SQLite → Postgres at scale) storing per-trajectory bounding box, node count, hash, simulation parameters. Enables filtered queries ("give me all trajectories with Re > 500 and > 10k nodes") without touching data files. |
| Ingestion speed | Single-threaded Python | Apache Beam / Spark pipeline reading raw sim outputs in parallel, writing Zarr shards. Easily parallelises across hundreds of workers. |
| Data versioning | None — files are overwritten | DVC (Data Version Control) or Delta Lake for append-only, versioned, auditable datasets. Critical when simulations are regenerated with corrected boundary conditions. |

---

## Stage 2 — Storage Layer

### What the system does today

**CFD:** One memory-mapped file (`train.dat`) holds all trajectories end-to-end. `np.memmap` with `mode='r'` opens a view into the file without copying it into RAM; the OS virtual memory system pages in only the slices you touch. A single contiguous read `all_velocity[:, 0, :]` (all nodes, time step 0) for Dataset Studio statistics takes milliseconds even for large datasets.

**Cloth:** Per-trajectory `traj_{i:05d}.npz` files. `np.load` decompresses the file on demand. Slightly higher latency per access but avoids loading the entire dataset for partial queries.

**Checkpoints:** A single PyTorch `state_dict` `.pt` file per domain. Epoch metadata (loss curve, best validation loss) is appended to a plain `.log` file and parsed on demand by the Train page via server-sent events.

### Billion-scale upgrade path

**The core tension at scale:** You cannot fit billions of mesh nodes in RAM. You need a storage format that supports:
1. **Partial reads** — fetch only the trajectories or timesteps you need for a batch.
2. **Columnar access** — for statistics (Dataset Studio), read only velocity channel without loading position or edge data.
3. **Compression** — mesh data is spatially smooth and highly compressible (3–10× with zstd).

**Recommended stack:**

```
Raw sim outputs  →  Zarr (chunked, compressed, sharded)
                        ↓
                   Metadata DB (PostgreSQL + PostGIS for spatial queries)
                        ↓
                   Petastorm / WebDataset  →  DataLoader
```

**Zarr specifics:** Store each physical field as its own Zarr array, chunked along the trajectory and time dimensions. Example chunk: `chunks=(1, 10, None)` means one trajectory, 10 timesteps, all nodes — aligns perfectly with rollout-style access. Use `zarr.consolidate_metadata()` so listing the dataset does not hit object-storage LIST calls.

**HDF5 vs Zarr:** HDF5 has a global file lock — only one writer at a time. Zarr is lock-free and designed for concurrent cloud writes. For read-heavy inference, both are fine; for training pipelines writing checkpoints and metrics in parallel, Zarr wins.

**Checkpoint management at scale:** Replace single `.pt` files with a checkpoint registry (a small DB table: `epoch, val_loss, path, created_at`). Keep the top-K checkpoints, auto-prune the rest. Tools: `torch.distributed.checkpoint` for tensor-parallel sharded checkpoints, or Hugging Face `safetensors` for fast, memory-mapped weight loading.

---

## Stage 3 — Preprocessing & Graph Construction

### What the system does today

**Mesh → Graph conversion:**  
For CFD, the mesh topology is pre-computed once and embedded in the TFRecord. Each mesh node becomes a graph node; each face-neighbour relationship becomes a directed edge (edges are added in both directions so message passing is bidirectional). Edge features are computed on the fly in `encode_process_decode.py`:
- **Relative displacement** `Δx = x_receiver − x_sender` (2D for CFD, 3D for cloth)
- **Displacement norm** `‖Δx‖`

Node features include the raw physical field (velocity or world position) concatenated with one-hot node type (interior node, boundary, obstacle surface).

**Normalisation:**  
The `Normalizer` class (`normalizer.py`) is an `nn.Module` that tracks running mean and variance as non-trainable buffers. It updates online during training (Welford-style) and freezes during evaluation. Storing the normaliser inside the model means normalisation statistics travel with checkpoints automatically.

**K-d tree for OOD detection:**  
After training, the system builds a k-d tree (`scipy.spatial.KDTree`) over the training set's latent-space representations of mesh nodes. At inference, query distance to the k-th nearest training point estimates out-of-distribution confidence. The system ships a C++ k-d tree with a pool allocator exposed via pybind11; `scipy` is the Python fallback. This is the spatial indexing component most directly relevant to interview questions.

### Billion-scale upgrade path

**Graph construction is the bottleneck:**  
For unstructured meshes without pre-computed connectivity (raw point clouds from LiDAR or FEM outputs), you must build the k-NN graph at preprocessing time. Building a k-d tree over 1 billion points is possible but serial — construction is `O(n log n)` and takes ~10 minutes for 100M points in single-threaded scipy.

**Strategies:**

| Problem | Solution |
|---|---|
| k-d tree construction too slow | **FAISS** (Facebook AI Similarity Search) — GPU-accelerated approximate nearest-neighbour. Builds IVF (inverted file) index over billions of vectors in minutes on a V100. For exact k-NN on meshes (small n, high accuracy needed), **nanoflann** (header-only C++ k-d tree) is 3–5× faster than scipy. |
| Graph too large for one GPU | **Graph partitioning** (METIS, ParMETIS). Partition the mesh into N subgraphs with minimal cut edges; assign one partition per GPU. X-MeshGraphNet (NVIDIA PhysicsNeMo) implements this. |
| Repeated preprocessing overhead | Cache processed graphs as Zarr or PyG `Data` objects. Use a content-addressed cache (hash of mesh file + preprocessing params) so re-running the pipeline hits cache instead of recomputing. |
| Variable mesh resolution per trajectory | Use PyTorch Geometric `DataLoader` with `follow_batch` to batch graphs of different sizes into a single disconnected super-graph. The batch vector `data.batch` tracks which nodes belong to which graph. |
| Feature drift between simulations | Fit the `Normalizer` on a representative 10% sample first; then freeze and apply to the full dataset. For streaming ingestion, use a distributed Welford accumulator (reduce `count`, `mean`, `M2` across workers). |

**Geometric preprocessing tricks:**
- **Surface normals** computed via cross-products of face edges, area-weighted per vertex — adds geometric context without extra simulation data.
- **Laplacian smoothing check** — if the input mesh has degenerate elements (zero-area faces), the Laplacian eigenspectrum will have near-zero eigenvalues; flag these trajectories before training.
- **Adaptive remeshing features** — for cloth, track the ratio `‖Δx‖ / rest_length` per edge; edges stretched beyond 2× rest length are candidates for dynamic subdivision in the simulator.

---

## Stage 4 — Training

### What the system does today

**Training loop:**  
`train.py` uses a standard PyTorch loop: sample a random training trajectory, pick a random timestep `t`, run `model(graph_t)` to predict `Δfield`, integrate to get `field_{t+1}`, compare to ground truth with `MSELoss`, backpropagate, clip gradients (`max_norm=1.0`), step the Adam optimiser.

**Noise injection:**  
Before each forward pass, Gaussian noise `σ=0.02` is added to the input field. This is the key training trick from the MeshGraphNets paper — it makes the model robust to its own prediction errors during rollout (since at rollout time, step `t` input is the model's own noisy prediction, not clean ground truth).

**Monitoring:**  
Training metrics stream to the UI via server-sent events (SSE). The backend appends JSON lines to a log file; the `_parse_log` function scans it from the last-read offset. The Train page renders a live loss curve with Recharts.

**Fresh start vs. resume:**  
A checkbox on the Train page sets `fresh_start=True` in the POST body. The backend deletes the checkpoint file and clears the in-memory model cache before launching the training subprocess, ensuring epoch 1 restarts with random weights.

### Billion-scale upgrade path

**The fundamental limit:** A single GPU can hold graphs with ~1M nodes in batch (at 128-dim latent, 15 message-passing rounds). Billion-node meshes require multi-GPU or multi-node training.

**Scaling strategies, in order of complexity:**

1. **Data parallelism (DDP)** — each GPU holds a copy of the model and processes different trajectory batches. Gradients are all-reduced across GPUs after each step. Launch with `torchrun --nproc_per_node=8`. Near-linear scaling to ~8 GPUs; communication overhead dominates beyond that.

2. **Graph partitioning + distributed message passing** — partition the mesh with METIS; assign partitions to GPUs. Boundary edges require a halo exchange (send node embeddings across GPU boundaries before each message-passing round). This is what X-MeshGraphNet implements. Scales to arbitrary mesh size but adds significant engineering complexity.

3. **Mixed precision (BF16/FP16)** — halves memory, speeds up matrix multiplications. Use `torch.autocast('cuda', dtype=torch.bfloat16)`. BF16 preferred over FP16 for training stability (larger dynamic range).

4. **Gradient checkpointing** — instead of storing all intermediate activations for the backward pass, recompute them on the fly. Halves activation memory at the cost of ~30% extra compute. Use `torch.utils.checkpoint.checkpoint` on the processor's message-passing blocks.

5. **Flash Attention / optimised scatter-gather** — if self-attention is used in the processor (e.g., graph transformer variant), Flash Attention 2 fuses the attention kernel to avoid materialising the full `N×N` attention matrix. For message passing, fused CUDA kernels for scatter (e.g., `torch_scatter`) eliminate intermediate Python overhead.

6. **Curriculum learning** — start training on small meshes (fast iterations, many epochs), then progressively increase mesh resolution. The model generalises to larger meshes without ever needing a giant batch of billion-node graphs.

**Monitoring at scale:**  
Replace the SSE log-file approach with a proper experiment tracker: **Weights & Biases** or **MLflow**. Both support distributed runs — each worker logs its shard metrics; the tracker aggregates. Store model checkpoints in S3/GCS with version tags.

---

## Stage 5 — Validation & Physics Consistency

### What the system does today

**Rollout validation:**  
The system runs a full autoregressive rollout on the test set: it feeds the model its own predictions step-by-step for the full trajectory length (600 timesteps for CFD, ~250 for cloth). Per-step RMSE is computed and displayed on the Visualize page. The RMSE curve should stay flat or grow slowly — exponential growth means the model is unstable.

**Out-of-distribution detection:**  
The OOD confidence system (`confidence.py`) builds a k-d tree over latent representations of training nodes. At inference, the query node's distance to its k-th nearest training neighbour is normalised to a `[0, 1]` confidence score. Nodes far from the training distribution get low confidence — the UI colours them differently. This is purely data-driven; no physics knowledge is encoded in the OOD check.

**Generate / Inverse Design validation:**  
The Generate page runs both a physics surrogate score (drag, area) and optionally a GNN rollout in deep mode. Deep mode is expensive (full 600-step rollout per candidate) but gives a physics-consistent validation signal, not just a surrogate estimate.

### Billion-scale upgrade path

**Physics consistency is the hard part of physics-AI.** Accuracy metrics (RMSE, MAE) are necessary but not sufficient. You need:

**1. Conservation law monitoring:**  
For incompressible CFD, check `∇·v = 0` (divergence-free velocity). Compute discrete divergence at each node after each predicted step. If `mean(|∇·v|) > ε`, the prediction is physically inconsistent. This is a cheap post-hoc check; you can add it as a validation metric without retraining.

**2. Equivariance testing:**  
MeshGraphNets are not inherently equivariant — they encode absolute positions as node features. Test equivariance empirically: rotate the mesh by 90°, re-run inference, check that the velocity field rotates by 90° too. If not, the model has memorised coordinate frames from training data. Fix by encoding only relative positions and displacement vectors (which the current implementation already does for edge features).

**3. Long-horizon stability test:**  
Run rollouts 10× longer than training length. Plot the energy spectrum. A stable model maintains the correct energy cascade; an unstable one will show energy pile-up at high frequencies (aliasing) before blowing up to NaN. The system already handles NaN gracefully (sanitised to `null` in the API), so the UI won't crash, but the physics is broken.

**4. PINN-based fine-tuning at inference:**  
A powerful production technique: after the GNN predicts a field, run a few steps of gradient descent minimising the PDE residual (e.g., Navier-Stokes residual computed via automatic differentiation). This corrects small violations of conservation laws without a full solver. It is much faster than training a PINN from scratch since the GNN already provides a near-correct initialisation.

**5. Shadow deployment:**  
Run the ML model in parallel with the ground-truth CFD solver on 5% of new simulations. Compute divergence from solver ground truth over time. When drift exceeds a threshold, trigger retraining with the new simulation data.

---

## Stage 6 — Serving & Deployment

### What the system does today

**API server:** FastAPI application with four route groups (`/train`, `/dataset`, `/results`, `/generate`, `/simulate`). Runs as a local process. Model inference is synchronous in the request handler — no async inference queue.

**Model caching:** `api/state.py` uses double-checked locking to cache the loaded PyTorch model in memory. First request loads from disk (~2s); subsequent requests hit the in-memory cache. A `clear_model_cache()` function is called when training restarts to avoid serving stale weights.

**Result caching:** `results.py` uses an LRU dict (keyed by `(filename, mtime)`) to cache parsed pkl files. A fresh rollout invalidates the cache automatically because the mtime changes.

**Frontend:** React SPA built with Vite. Connects to the FastAPI backend via `/api/` prefix (proxied in `vite.config.ts`). Canvas 2D for mesh rendering (chosen over SVG because it handles >100k nodes without DOM explosion). Recharts for time-series plots.

### Billion-scale upgrade path

**The production path for a physics-AI model:**

```
Trained PyTorch model
        ↓
  torch.jit.script (TorchScript) or torch.export
        ↓
  ONNX export (optional, for cross-framework deployment)
        ↓
  NVIDIA Triton Inference Server
        ↓
  REST / gRPC API  →  client applications
```

**Why Triton:**  
Triton Inference Server handles dynamic batching (accumulate requests for 5ms, batch them together), multi-GPU scheduling, model versioning (A/B test model v1 vs v2 with traffic splitting), and GPU utilisation monitoring. It supports TorchScript, ONNX, and TensorRT backends.

**TensorRT optimisation:**  
For fixed-mesh deployments (the cylinder mesh never changes), TensorRT can fuse and quantise the GNN graph into a single optimised CUDA kernel. Typical speedup: 3–8× over vanilla PyTorch inference. The tradeoff: TensorRT engines are tied to a specific GPU architecture and mesh size.

**Coupling with solvers (the real production use case):**  
In practice, ML models are not used as standalone solvers. They are coupled with traditional CFD/FEM solvers in three patterns:

| Role | Description | Example |
|---|---|---|
| **Surrogate** | Replace expensive inner loop entirely | Replace Navier-Stokes solver with GNN for design optimisation sweeps |
| **Initialiser** | Provide warm-start solution | GNN predicts steady-state; solver converges in 10× fewer iterations |
| **Corrector** | Refine coarse solver output | Run coarse solver at 10× larger timestep; GNN corrects the error field |

**Horizontal scaling:**  
FastAPI is stateless except for the in-memory model cache. Wrapping it in a Docker container and deploying behind a load balancer (NGINX, or Kubernetes Ingress) gives horizontal scaling. The model cache should move to a shared service (Redis with binary serialisation, or a dedicated model-serving pod) so all API replicas share the same loaded weights.

**Streaming inference for long rollouts:**  
The current SSE-based streaming (used for training logs) can be extended to inference: stream each predicted timestep back to the client as it is computed rather than waiting for the full rollout. The Visualize page already polls individual frames via `/api/results/frame?t=N` — replacing this with a streamed WebSocket would eliminate per-frame HTTP overhead for long trajectories.

---

## Spatial Indexing at Billion Scale — Deep Dive

> This section is specifically for the k-d tree / spatial indexing interview question.

### What the system does today

`confidence.py` implements a k-d tree for OOD detection:
1. At the end of training, run inference on the entire training set and collect the 128-dim latent node embeddings.
2. Build `scipy.spatial.KDTree(training_embeddings)`.
3. At inference time, query `tree.query(test_embedding, k=5)` to get the 5 nearest training neighbours.
4. Normalise the query distance to a `[0,1]` confidence score.

The C++ implementation (`kdtree.cpp` exposed via pybind11) uses a pool allocator to reduce `malloc`/`free` overhead during tree construction. It is 2–4× faster than scipy for large `n` because it avoids Python object overhead and uses cache-friendly memory layout.

### Why k-d trees and what are their limits

**How k-d trees work:**
- Recursively split the point set along alternating axes (x → y → z → x → …), choosing the median point as the split.
- At each internal node, store the split axis and value; left subtree holds points below, right holds points above.
- Construction: `O(n log n)`. Query for k nearest neighbours: `O(k log n)` average case.
- For 3D meshes, the constant factor is small — k-d trees dominate in practice for `n < 10M`.

**Where k-d trees break down:**
- **High dimensionality** (`d > 20`): query time degrades to `O(n)` (the curse of dimensionality). The latent space here is 128-dimensional — k-d trees are not optimal, but the structure of learned embeddings tends to cluster well so in practice they still work.
- **Dynamic point sets**: inserting new points into a balanced k-d tree requires rebalancing. For streaming data, maintain a buffer and periodically rebuild.
- **Billion-point datasets**: a k-d tree over 1B 128-dim vectors requires ~200 GB of RAM. Use FAISS instead.

**FAISS at billion scale:**
- **IVF (Inverted File)** index: partition the space into `nlist` Voronoi cells (clusters). To query, search only `nprobe` nearest cells. Construction: `O(n)` after k-means clustering. Query: `O(nprobe * n/nlist)`.
- **PQ (Product Quantisation)** compression: compress each 128-dim vector into 8 bytes (16× compression). Enables billion-vector indexes to fit in RAM.
- **HNSW (Hierarchical Navigable Small World)**: graph-based ANN structure. Faster queries than IVF+PQ but higher memory. Best for < 100M vectors where accuracy matters.

**Octrees for point cloud rendering:**
- The Visualize page renders mesh nodes with Canvas 2D. For meshes with >100k nodes, only the visible nodes (within the viewport frustum) should be rendered.
- An octree subdivides 3D space into 8 child cubes recursively. Level-of-detail (LOD) rendering: at zoom-out, render only the octree root (1 point per cell); as you zoom in, descend the tree and render finer cells.
- At billion-point scale (e.g., LiDAR point clouds), **COPC (Cloud-Optimized Point Cloud)** format stores an octree-indexed LAZ file where coarse LOD data lives at the file head — streaming readers fetch only what the current zoom level requires.

---

## Quick Numbers Table

| Metric | Today (demo scale) | Production scale |
|---|---|---|
| CFD trajectories | ~1,000 test / ~1,000 train | 100,000+ per parameter sweep |
| Nodes per mesh | ~1,800 (cylinder) | 100k–10M (aircraft wing, turbine) |
| Timesteps per trajectory | 600 | 600–10,000 |
| Total training nodes | ~1M | 100B+ |
| k-d tree build time | <1s (scipy) | 10 min (1B pts, single thread) → 30s (FAISS GPU) |
| k-NN query time | <1ms | 1μs (FAISS IVF+PQ, GPU) |
| Rollout latency (600 steps) | ~8s (CPU) | <100ms (TensorRT, A100) |
| Model size | ~2M parameters | 10M–100M (deeper processors, wider latent) |
| Storage per trajectory | ~10 MB (npz) | 1–10 GB (full 3D unstructured mesh) |
| Training time (100 epochs) | ~30 min (CPU) | 2–48 hours (multi-GPU, large mesh) |

---

## Design Pattern Summary

| Decision | Choice made | Alternative | Why this one |
|---|---|---|---|
| Graph format | PyTorch Geometric `Data` | DGL, plain adjacency matrix | PyG has the richest ecosystem for mesh-based GNNs and native batching of variable-size graphs |
| Storage layout | memmap + npz for CFD; per-traj npz for cloth | Zarr, HDF5 | Simplicity at demo scale; Zarr is the upgrade path |
| Spatial index | C++ k-d tree (pybind11) + scipy fallback | FAISS, nanoflann | Balance of speed and portability; FAISS for >10M points |
| Normalisation | Online Welford in nn.Module | Pre-computed statistics file | Travels with checkpoint; works for streaming data |
| API streaming | SSE for training logs | WebSocket | SSE is simpler (HTTP, unidirectional); sufficient for log streaming |
| Frontend rendering | Canvas 2D | WebGL / Three.js | Canvas handles 10k–100k nodes well; WebGL needed beyond that |
| NaN handling | Sanitise to `null` at API boundary | Clip in rollout | Preserves raw simulation data integrity; frontend renders `—` gracefully |
| OOD confidence | k-d tree over latent space | Ensemble disagreement, Mahalanobis distance | Single model, no retraining; O(log n) query per node at inference time |
