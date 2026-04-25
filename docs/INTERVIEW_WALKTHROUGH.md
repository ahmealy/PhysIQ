# MeshGraphNets — Interview Walkthrough Script

> **How to use**: One tab, second screen. Read the **bold lines** naturally as you navigate.
> Regular text = reminders of what's on screen or what to do next.
> Don't read the code or tables aloud — they're your safety net if asked to go deeper.

---

## BEFORE YOU OPEN THE UI

*Take a breath. Start here.*

---

**"I built a full-stack physics simulation platform based on DeepMind's MeshGraphNets paper. The core idea is replacing a traditional CFD or finite-element solver — which can take hours per run — with a graph neural network that learns the underlying physics from simulation data and can roll forward a complete trajectory in milliseconds at inference time.**

**On top of that I built an interactive web application where you can train the model, run predictions, visualize them as animated mesh plots, and generate novel mesh designs that hit a target physics value — all from a browser."**

*Now open the UI.*

---

## PAGE: Dataset

*Navigate to the Dataset page. Let the interviewer see the data summary.*

---

### Why Graphs?

**"Traditional simulators like OpenFOAM discretize space into unstructured meshes. A mesh is naturally a graph — nodes are spatial points, edges are mesh connectivity. CNNs can't handle irregular meshes because there's no grid to slide a filter over. A GNN can.**

**The key insight from the paper is that physics is local. A node's next state depends only on its immediate neighbors. Message passing encodes exactly that inductive bias — each round propagates information one hop further, so after 15 steps you've covered a 15-edge interaction radius, which is enough to capture the relevant physics for both CFD and cloth."**

---

### The Two Domains

**"I implemented two physical domains. The first is cylinder flow — 2D incompressible Navier-Stokes, the classic von Kármán vortex street behind a cylinder. The second is a deformable cloth simulation. They share the same GNN architecture but differ in their physics integrators and edge features."**

*Point to the node type legend if visible.*

**"Node types encode boundary conditions directly in the features. A NORMAL node gets its velocity updated by the GNN. An INFLOW node has its velocity fixed. The model learns to respect these constraints — it's how the physics boundaries get embedded without hard-coding them separately."**

---

### Data Pipeline

*If asked about the data format, here's the story:*

**"The original dataset from DeepMind is in TFRecord format — Google's protobuf-based binary. It requires an old version of TensorFlow to parse, so I wrote a one-time conversion script that reads the TFRecords and writes NumPy memmap files. After that the entire training loop is pure PyTorch with zero TensorFlow dependency."**

**"The full training set is too large to fit in RAM, so I used NumPy memory maps. The OS manages paging — the training loop reads exactly the slice it needs for each sample without materializing the full array. This lets the system scale to datasets that exceed available memory."**

**"The memmap layout is node-first: shape [N_total, T, D]. A companion .npz file holds CSR-style index arrays — cumulative node counts per trajectory — so any (trajectory, timestep) pair maps to a byte offset in O(1) arithmetic. No scan, no decompression, just one pointer addition and one memory fetch."**

---

## PAGE: Train

*Navigate to the Train page. Don't start training — just walk through the config.*

---

### Model Architecture — Three Options

**"The architecture follows the paper: Encoder, Processor, Decoder. But I implemented three interchangeable Processor blocks — you pick the one that fits the problem."**

```
Input graph  {x: [N,11], edge_attr: [E,3], edge_index: [2,E]}
      │
  ┌───▼──────────────────────────────────┐
  │  ENCODER                             │
  │  node MLP:  11  → 128 → 128 → 128   │  + LayerNorm
  │  edge MLP:   3  → 128 → 128 → 128   │  + LayerNorm
  └─────────────────────────────────────┘
      │  latent graph {x: [N,128], edge_attr: [E,128]}
      │
  ┌───▼──────────────────────────────────┐  ×15
  │  PROCESSOR  — one of:                │
  │    GnBlock (default)                 │
  │    TNSBlock (attention)              │
  │    SAGEBlock (mean aggregation)      │
  └─────────────────────────────────────┘
      │  updated latent graph
      │
  ┌───▼──────────────────────────────────┐
  │  DECODER                             │
  │  node MLP:  128 → 128 → 128 → 2     │  NO LayerNorm
  └─────────────────────────────────────┘
      │
  Predicted velocity residual  [N, 2]
```

---

### GnBlock — The Paper Baseline

**"GnBlock is the default. It has two steps per round. First the EdgeBlock — concatenate the sender embedding, receiver embedding, and current edge embedding, pass through an MLP, get an updated edge. Then the NodeBlock — scatter-sum all incoming updated edges at each node, concatenate with the node's own embedding, pass through another MLP. Both steps add a residual connection."**

> Use when: medium mesh, proven baseline, you want to match the paper exactly.

---

### TNSBlock — Attention Aggregation

**"TNSBlock replaces the fixed scatter-sum with multi-head dot-product attention. Instead of summing all neighbors equally, it learns which neighbors matter most for each node. The query is the current node embedding, keys and values are incoming edge embeddings. The attention weights are learned — they tell you which edges are driving the physics at each node."**

> Use when: you care about interpretability (which edges are most influential) or when some neighbors dominate the dynamics and equal weighting wastes capacity. Needs lower LR (3e-5) and gradient clipping (norm 1.0) to train stably.

*If asked about implementation details:*

**"It's a direct drop-in swap in the Processor. The same Encoder and Decoder are used — only the aggregation changes. I unit-tested the block in isolation to confirm the output shapes are identical to GnBlock, so the rest of the pipeline doesn't know which block is active."**

---

### SAGEBlock — Mean Aggregation for Irregular Meshes

**"SAGEBlock uses mean aggregation — it divides the summed neighbor features by the degree of each node. This is critical for cloth and other meshes where node valence varies widely. Corner nodes have 2–3 neighbors; interior nodes have 6+. With sum aggregation, high-degree nodes dominate. With mean aggregation, each node's update is normalized by how many neighbors it has, so the scale stays consistent regardless of local mesh density."**

> Use when: large meshes, irregular valence, cloth — any domain where topology varies significantly across the mesh.

---

### Why Residuals?

*If the interviewer asks:*

**"Without residuals, 15 sequential matrix multiplications would cause the signal to either vanish or explode by the time it reaches the bottom of the stack. Residual connections let each GnBlock learn what to add to the current state rather than what the state should be — that's a much easier learning target and it gives gradients a direct path back through the network."**

---

### Why No LayerNorm in the Decoder?

*If asked:*

**"The Encoder needs to normalize before features enter the latent space because they come in at very different scales. The Decoder output is a velocity change that feeds directly into a physics integrator. If I normalized the output I'd clip or distort its magnitude, and the Euler integration step would produce wrong trajectories."**

---

### Online Normalizer

**"One design decision I'm proud of: normalization statistics are computed online during training rather than as a pre-processing step. Each normalizer accumulates a running mean and variance over the first million batches using Welford's algorithm — numerically stable, single pass. After that the statistics freeze. This means the normalization is baked into the model checkpoint — there's no separate stats file to manage or accidentally lose."**

---

### Training Loop

**"The loss is MSE on velocity change — what the paper calls acceleration — computed only on NORMAL and OUTFLOW nodes. Boundary nodes like INFLOW are fixed by physics so there's nothing to predict there."**

**"One thing the paper emphasizes is noise injection during training. I add Gaussian noise to the input velocity field at each step. The reason is train-test distribution mismatch — during training the model sees clean ground-truth inputs, but during rollout it sees its own previous predictions which accumulate error. Adding noise at training time closes that gap."**

**"For cloth specifically I match DeepMind's training protocol more closely: exponential learning rate decay per step — gamma = 0.1^(1/5,000,000) — with a floor at 1e-6. And the first 1000 steps are a normalizer warmup — the forward pass runs but the backward pass is skipped, so the normalizers accumulate statistics before any gradient updates happen."**

**"For multi-GPU training I use PyTorch DistributedDataParallel with NCCL. Each process owns one GPU, gradients are all-reduced across GPUs after each backward pass. Only rank 0 writes to TensorBoard and saves checkpoints — otherwise you get corruption from concurrent writes to the same file."**

---

## PAGE: Predict / Analyze

*Navigate to Predict. Pick a trajectory and run a rollout, or show a pre-computed result.*

---

### Autoregressive Rollout

**"Inference is autoregressive. The model predicts the velocity change at each timestep, we add it to the current velocity, feed that as the next input, and repeat for all 600 timesteps. The model never sees the ground truth during rollout — each prediction is built on the previous prediction."**

**"Running a rollout triggers a POST request. On the backend, the server runs the full T-step loop, saves the predicted trajectory, ground truth, and mesh coordinates as a pickle file, and streams progress events back to the UI via Server-Sent Events so the user sees a progress bar rather than a frozen screen."**

---

### Poisson Pressure Correction

*If the interviewer asks about physics correctness or post-processing:*

**"There's an optional Poisson pressure correction on the rollout page. The raw GNN output is a velocity field that may not satisfy the divergence-free constraint — ∇·u = 0 — required by incompressible flow. The correction solves the pressure Poisson equation: ∇²p = (ρ/Δt)∇·u\*. Discretized on the mesh that becomes a sparse linear system L·p = rhs, where L is the graph Laplacian."**

**"I factorize L once using scipy's sparse LU — splu — which costs O(n^1.5) for sparse systems. Then for each of the 600 timesteps I call lu.solve(rhs), which is just a forward-backward substitution — O(n). Factor once, solve 600 times. The correction is opt-in because it adds latency and isn't always necessary — for smooth, low-Reynolds-number flows the GNN already satisfies incompressibility well enough."**

---

### Visualization

*Navigate to the Visualize page.*

**"The visualization shows three animated panels side by side: ground truth, prediction, and absolute error. The color map is Turbo for velocity and red for error. The mesh renderer is pure Canvas 2D — no WebGL for the CFD case. For cloth I added a Three.js 3D viewer with two side-by-side panels — ground truth and prediction — with linked cameras so dragging one rotates both simultaneously."**

**"I chose Canvas over SVG for CFD because a 2000-node mesh has roughly 4000 triangles. In SVG, that's 4000 DOM elements being re-rendered every animation frame. In Canvas, all 4000 triangles are drawn in a single imperative pass with no DOM overhead."**

**"Per-frame data is loaded on demand via a GET request — not all 600 frames at once. The backend uses an LRU cache keyed by filename and modification time. When you open the page, the browser fires three parallel requests for metadata, RMSE, and frame zero. Without caching each request independently deserializes an 18-megabyte pickle. With the cache, the first request pays that cost and the other two return instantly."**

---

## PAGE: Generate

*Navigate to Generate. Show the target drag input and the candidate cards.*

---

### The Problem This Solves

**"The Generate page inverts the simulation. Instead of asking 'given this mesh, what are the physics?' it asks 'given a target drag value, what mesh design achieves it?' That's the inverse design problem."**

---

### Design Parameters — cx, cy, r, v_inlet

**"Every candidate is described by four scalars: cylinder centre x, cylinder centre y, radius, and inlet velocity. These aren't stored in the dataset — I reverse-engineer them from the mesh by fitting a circle to the WALL_BOUNDARY nodes using least-squares, then reading inlet velocity from the INFLOW nodes at timestep zero."**

---

### CVAE + Surrogate

**"The pipeline has two stages. First, a Conditional VAE maps cylinder design parameters to a 16-dimensional latent space, conditioned on a target drag value. At generation time I sample from the prior, condition on the target, decode to design parameters."**

**"Second, a lightweight MLP surrogate — four inputs, two 64-unit hidden layers, one output — predicts drag from design parameters. This lets me score all candidates instantly without running a full GNN rollout."**

**"The CVAE loss has three terms: reconstruction loss, KL divergence regularisation with free_bits=0.05 to prevent posterior collapse, and a physics consistency term — the surrogate must agree that the reconstructed design produces the target drag. That third term forces the decoder to learn the physics, not just the geometry."**

---

### RealMeshLookup — Why Generated Params Are Snapped

**"The CVAE decoder outputs continuous parameters — cx=0.31, cy=0.52, r=0.08. Building a mesh from scratch requires Delaunay triangulation, which isn't differentiable and can produce degenerate triangles. Instead, I maintain a KDTree over all training meshes' (cx, cy, r) vectors. Generated params are snapped to the nearest real training mesh. Only v_inlet is changed — injected into the INFLOW nodes. This guarantees a valid triangulation the GNN has seen during training."**

---

### Latin Hypercube Sampling

**"For the sample method, I use Latin Hypercube Sampling instead of random Normal to draw points in the 16-dimensional latent space. LHS divides each dimension into n equal-probability intervals and places exactly one sample per interval. With n=10 candidates and 16 dimensions, random Normal would cluster near the origin. LHS gives uniform coverage of the latent space, so you get more diverse designs with the same budget."**

---

### Two Generation Methods

**"There are two methods. The default — sample — draws n independent LHS points from the latent space, decodes each to design parameters, scores with the surrogate, and returns all n as diverse candidates."**

**"The gradient method does something more targeted. Adam runs directly on a single latent vector z for 150 iterations, minimising the squared difference between surrogate-predicted drag and the target. Three random restarts find the best z\*. Then n diverse candidates are produced by adding small noise around z\* — variations of the same optimal design rather than independent samples."**

---

### Why Gradient Doesn't Flow Through the GNN

*If asked about the gradient path in detail:*

**"The gradient chain is: z → CVAE decoder → design params → DragSurrogate MLP → drag → loss → ∂z. Every step is matrix multiplications and affine transforms — fully differentiable."**

**"There's an implemented GNN path but it's disabled. The problem is RealMeshLookup.find_nearest() — a KDTree argmin that returns an integer index. Integers aren't differentiable. So cx, cy, r get zero gradient through the mesh lookup. Only v_inlet would survive. With three of four parameters having no gradient, plus unit-scale mismatches between GNN drag proxy and target drag, the surrogate path is strictly better for optimisation."**

---

### OOD Confidence

*Point to the confidence badge on a candidate card.*

**"Every candidate has a confidence score. The GNN will always produce a number — there's nothing in the architecture that says 'I don't know'. The confidence score warns when a design is outside the training distribution."**

**"For the Generate page I use parameter-space OOD — a KDTree over training (cx, cy, r, v_inlet) vectors. Distance in 4D parameter space tells you whether this cylinder geometry was seen during training. For the Predict page I use embedding-space OOD — the test simulation is run through the frozen GNN encoder to get a 128-dim embedding, then compared against a KDTree of training embeddings. These answer different questions: parameter OOD asks 'is this geometry unusual?', embedding OOD asks 'does this simulation behave like training simulations?'"**

---

### Confidence Index — Three-Tier Backend

**"The embedding index has a three-tier backend hierarchy with automatic fallback:**

**First tier: FAISS IndexFlatL2. If faiss-cpu is installed, we use this. FAISS applies AVX-512 SIMD — it processes 16 floats per CPU cycle. At high dimensions KD-tree pruning becomes ineffective because the curse of dimensionality means almost all points are equidistant. FAISS brute-force with SIMD wins from about N=1000 upwards.**

**Second tier: C++ KDTree via pybind11. If the extension is compiled but FAISS isn't installed, we use this. The C++ tree uses a pool allocator — pre-sized to 2n+1 nodes at build time, zero heap allocations during queries.**

**Third tier: scipy KDTree. Always available, always the fallback. train_diameter is always computed using scipy regardless of which query backend wins — scipy is the ground truth."**

---

### SSE Streaming

**"Candidates stream to the UI incrementally via Server-Sent Events. SSE is unidirectional — the server pushes, the client reads — which is exactly what you need for a long-running computation where you want progressive updates. The user sees results arriving one by one. Thumbnails are rendered server-side and served via a separate GET endpoint, so each candidate event carries a URL, not raw bytes."**

---

## IF ASKED: Where LU Decomposition Fits — and Why MeshGraphNets Replaces It

*This is a strong talking point — it shows you understand the physics solver this GNN is replacing.*

---

**"To understand what MeshGraphNets does, it helps to understand what it replaces. A traditional Finite Element solver for cloth or a CFD pressure solver both reduce to the same thing at their core: solving a linear system Ax = b."**

**"In FEM cloth simulation, the stiffness matrix K encodes the spring network of the mesh. At each timestep you solve K·u = f. With implicit Euler integration the system becomes (M + h²K)·u_{t+1} = M·u_t + h·f_ext. You factorize the left matrix once with LU — cost O(n³) for dense, O(n^1.5) for sparse — and for every subsequent timestep you only pay O(n) for the triangular solve. Pay factorization once, reuse across thousands of timesteps."**

**"In CFD, the incompressibility constraint ∇·u = 0 is enforced by solving a pressure Poisson equation — ∇²p = (ρ/Δt)∇·u\*. Discretized on the mesh that becomes L·p = rhs, where L is the graph Laplacian. L is sparse symmetric positive definite — Cholesky rather than full LU, half the cost."**

**"MeshGraphNets replaces these solves in the main simulation loop. The GNN learns state_t → state_{t+1} directly from data. At inference there's no matrix assembly, no factorization, no triangular solve — just a forward pass through the graph network.**

**I do still use sparse LU in one place: the optional Poisson pressure correction post-processing step. After the GNN generates the velocity field, if the user wants to enforce incompressibility, I build the graph Laplacian, call scipy.sparse.linalg.splu once, and call lu.solve for each of the 600 timesteps. Factor once, solve 600 times — exactly the same pattern as the FEM solver, just applied as a correction layer on top of the GNN output rather than as the primary solver."**

**"One active research direction is hybrid approaches — use the GNN as a preconditioner for Conjugate Gradient. The GNN gives you a warm start that's close to the true solution, and CG corrects the residual. You get GNN speed plus solver accuracy. That's where I'd take this project next."**

---

## IF ASKED: Backend Design

---

**"The backend is FastAPI. All shared mutable state lives in a single state module — the loaded model, the training process handle, the GNN scorer. Each is a module-level singleton."**

**"Model loading uses double-checked locking: a fast path that checks for the cached model without acquiring a lock, and a slow path that acquires the lock and checks again before loading. This avoids lock contention on every inference request while remaining thread-safe when two requests race to load the same checkpoint."**

**"The domain sampler uses a Strategy pattern. There's a base class with the generation interface, and concrete implementations for CFD and cloth. Adding a new physics domain means implementing one class and adding one entry to a registry dictionary — no existing code changes. That's the Open/Closed principle directly applied."**

**"Training launch has a sentinel file mechanism. There's a window between an SSH-based remote GPU launch being initiated and the actual PID being written to disk. During that window, a second call would spawn a duplicate process. The sentinel is written synchronously at the very start of the launch handler — before any async work — and expires after 120 seconds. Any concurrent request that sees the sentinel gets a 409 immediately."**

---

## IF ASKED: Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Repository** | `ResultRepository` Protocol → PKL / HDF5 / Zarr | Swap storage backend in one config line, callers unchanged |
| **Strategy** | `BaseDesignSampler` → CFD / Cloth | Swap physics domain without touching generation logic |
| **Open/Closed** | `_DOMAIN_SAMPLERS` registry, `IngestPipeline` + `SolverAdapter` | New domain or solver = new class + one line, no existing changes |
| **Double-checked locking** | `state.py` `_model_cache` | Thread-safe model loading with minimal lock contention |
| **LRU Cache** | `_LRUCache` in `results.py` | Bounded memory, auto-invalidates on file change (keyed by mtime) |
| **SSE / Observer** | `AsyncGenerator` → `StreamingResponse` | Progressive UI updates without polling |
| **Pool allocator** | C++ KDTree build | Zero heap fragmentation after construction, deterministic query latency |
| **Fallback chain** | FAISS → C++ → scipy in `NearestNeighborIndex` | Performance when available, portability always |
| **Sentinel files** | `.dat.ok`, `.zarr.ok`, launch guard | Fail-fast on corrupt/partial writes and duplicate process launches |

---

## IF ASKED: Modular Code and Unit Tests

**"Every subsystem is independently testable. The confidence index builds from embeddings, queries by vector, saves to pickle, and loads back — all tested in isolation without touching the training loop or the API. The dataset tests verify the shape contract: does `__getitem__` return a PyG Data object with the right node and edge dimensions?"**

**"I follow TDD: tests before or alongside implementation. For the three processor blocks — GnBlock, TNSBlock, SAGEBlock — I wrote shape tests first, confirmed they failed, then implemented. The tests also verify the residual connection: output shape must match input shape, otherwise the residual add would crash."**

---

## IF ASKED: Performance and Algorithms

**"The main performance decisions: memmap for data that doesn't fit in RAM — O(1) random access via byte-offset arithmetic, no decompression. LRU cache on result pickles to avoid repeated 18MB deserialization. On-demand frame loading instead of all 600 frames at once. And the FAISS/C++/scipy fallback chain for nearest-neighbor queries."**

**"The C++ KDTree pool allocator is worth explaining. Standard new/delete per node creates heap fragmentation and cache misses during tree traversal. The pool pre-allocates a contiguous block for all 2n+1 possible nodes at build time. Every subsequent allocation is a pointer bump — O(1) with no fragmentation. During query, the traversal walks physically contiguous memory — better CPU cache utilization and predictable latency."**

---

## IF ASKED: C++ / Python Integration

**"Python everywhere except one place: the k-d tree query. The hot path during design generation calls the confidence index for every candidate. Python's GIL and object allocation overhead add measurable latency per query. The C++ extension eliminates both — pybind11 calls release the GIL during the native call, and the pool allocator means no Python object allocation happens on the C++ side."**

**"The pybind11 binding accepts a NumPy float32 array, passes the raw data pointer to the C++ KDTree, and returns a Python float. No data copying. Import is guarded by try/except — if the extension hasn't been compiled, code silently falls back to scipy. The extension is a performance optimization, not a requirement."**

---

## IF ASKED: Scalability

**"The current design is intentionally monolithic — one FastAPI process hosting everything, state in module-level singletons. That's fine for a demo or small team."**

**"The first thing I'd split at scale is training — long-running GPU-bound job that blocks an API worker. Move it to a job queue like Celery with the API just submitting and polling."**

**"For inference at scale, the model should be served by a dedicated inference server like TorchServe or Triton that batches requests and manages GPU memory independently from the API tier."**

**"For the confidence index, at millions of embeddings exact NN becomes slow even for FAISS FlatL2. That's where FAISS IVFFlat makes sense — cluster training embeddings with k-means, search only the nearest cluster. Approximate but fast. For our use case with a few thousand training trajectories, exact search is fine."**

---

## RAPID-FIRE ANSWERS

*Have these ready if the interviewer fires short questions at you.*

**"Why autoregressive rollout?"** → *"That's how you get a trajectory — feed each prediction as the next input to unroll over time. The downside is error accumulation. Noise injection during training closes that distribution gap."*

**"Why 15 message-passing steps?"** → *"Empirically from the paper. Each step propagates information one hop. 15 hops covers the physical interaction radius at the mesh densities in the dataset."*

**"Why CVAE over GAN for generation?"** → *"Structured latent space — I can sample, interpolate, and optimise in it via gradient descent. GANs don't give you that. Training is also more stable."*

**"Does the project use LU decomposition?"** → *"Yes — in the optional Poisson pressure correction. After the GNN generates the velocity field, scipy.sparse.linalg.splu factorizes the graph Laplacian once per rollout. Then lu.solve is called 600 times — one per timestep. Factor once, solve many. The GNN replaces the main time-stepping solver; LU handles the incompressibility correction layer on top."*

**"Why FAISS over C++ KDTree for production?"** → *"At 128 dimensions, KD-tree pruning degrades — curse of dimensionality. FAISS brute-force with AVX-512 SIMD beats tree traversal from N≈1000 upwards. The C++ tree was the first implementation, demonstrates pybind11 integration, and is the active backend when FAISS isn't installed."*

**"Why three processor architectures?"** → *"GnBlock is the paper default — correct and proven. TNSBlock adds attention when you want to know which neighbors are driving the physics. SAGEBlock handles irregular valence — important for cloth where corner nodes have 2 neighbors and interior nodes have 6."*

**"memmap vs HDF5?"** → *"memmap gives O(1) random access with no decompression — the OS maps the file into virtual address space and pages in only what's touched. HDF5 with gzip chunks would be 60–160× slower for random access at training time because every read decompresses a full chunk. I do use HDF5 for rollout results with chunk=(1, N, D) — one chunk per timestep — because there the access pattern is sequential partial reads, which is exactly what HDF5 chunks are designed for."*

**"GNN error over time?"** → *"Error grows roughly linear for the first 200–300 steps then accelerates. That's characteristic of autoregressive error accumulation — small prediction errors compound. It's the fundamental limitation of the surrogate approach."*

**"Why LHS instead of random Normal for sampling?"** → *"With n=10 candidates in 16 dimensions, random Normal clusters near the origin — poor coverage of the latent space tails. LHS partitions each dimension into n equal-probability intervals, exactly one sample per interval. You get uniform coverage with the same budget, which means more diverse candidate designs."*

**"What is train_diameter?"** → *"A scale reference for the confidence score. Computed once at index build time: for each training embedding, find its 5th nearest neighbour in the training set. Take the 95th percentile of those distances. That's train_diameter — how big a typical local neighbourhood is in embedding space. The confidence score normalises d_min by it: score = clip(1 - d_min/train_diameter, 0, 1)."*

---

## QUICK NUMBERS

*Glance at these if you need to cite a specific value.*

| Fact | Value |
|---|---|
| Hidden dimension | 128 |
| Message-passing rounds | 15 |
| MLP layers per block | 3 |
| Processor options | GnBlock (sum), TNSBlock (attention), SAGEBlock (mean) |
| CFD node features | 11 (velocity×2 + node_type one-hot×9) |
| CFD edge features | 3 (Δx, Δy, distance) |
| Cloth node features | 12 (velocity×3 + node_type one-hot×9) |
| Cloth edge features | 7 (rel_world×3 + ‖rel_world‖ + rel_mesh×2 + ‖rel_mesh‖) |
| Rollout timesteps (CFD) | 600 (= 6 s × 0.01 s/step, covers 6–12 vortex shedding cycles) |
| CVAE latent dim | 16 |
| Design params | 4: cx, cy, r, v_inlet |
| Confidence index backends | FAISS → C++ KDTree → scipy (priority order) |
| Confidence embedding dim | 128 (dual-frame CFD: frame 0 + warmup frame, single encoder output) |
| train_diameter | 95th percentile of 5-NN distances in training embedding set |
| Pool allocator size | 2n+1 nodes pre-allocated at build |
| LRU cache capacity | 8 pickle files |
| Drag surrogate | 4 → 64 → 64 → 1 |
| LHS latent samples | n = n_candidates, d = 16 |
| Gradient descent | 150 iterations × 3 restarts = 450 total steps |
| Poisson correction | splu once per rollout, lu.solve × 600 steps |
| Cloth LR decay | 0.1^(step/5,000,000) per step, floor 1e-6 |
| Normalizer warmup (cloth) | 1000 steps, forward pass only |
| Cloth noise std | 3e-3 (DeepMind protocol) |
| CFD noise std | 2e-2 |
| Duplicate launch guard | Sentinel file, 120s TTL |
| free_bits (CVAE KL) | 0.05 per latent dimension |
| BPTT window (gradient refine) | K=5 steps (1 differentiable, K-1 detached) |

---

*End of script. You know this project deeply — the script is just your sequence guide.*
