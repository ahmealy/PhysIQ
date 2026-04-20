# MeshGraphNets — Interview Walkthrough Script
### Three-Level Structure: Stop at any level based on time and audience

> **How to use**: One tab, second screen.
> **Level 1** = anyone, 3–5 min. Stop here if time is short or audience is non-technical.
> **Level 2** = software engineers, 5–10 min more. Design patterns, architecture, C++/Python.
> **Level 3** = ML engineers or deep technical drill, 5–10 min more. GNN internals, training, physics.
> Read the **bold lines** naturally. Regular text = reminders. Don't read code aloud.

---

# ━━━ LEVEL 1 — THE STORY (3–5 min) ━━━
### What you built, why it matters, what you can show

---

## Opening — say this before opening the UI

**"I built a full-stack physics simulation platform based on a DeepMind research paper called MeshGraphNets.**

**The core problem: traditional physics simulators — the kind used in engineering for fluid dynamics or cloth — can take hours per run. The idea is to replace that solver with a neural network that learns the underlying physics from simulation data, and can reproduce a full trajectory in milliseconds at inference time.**

**I built everything end to end: the data pipeline, the model, the training loop, a REST API, and a React frontend where you can train, run predictions, visualize the results, and generate novel mesh designs that hit a target physics value — all from a browser."**

*Open the UI. Navigate to Dataset.*

---

## The Two Physics Domains

**"I implemented two domains. Cylinder flow — fluid dynamics, a 2D mesh with ~2000 nodes simulating vortices behind a cylinder. And cloth — a deformable 3D mesh simulating fabric falling under gravity. They share the same model architecture but different physics integrators."**

*Point at the mesh preview and node type breakdown.*

**"Each node has a type — Normal, Inflow, Outflow, Wall. These encode the boundary conditions directly in the input features. The model learns to respect them: Inflow nodes stay fixed, Normal nodes get updated. That's how physics constraints enter the network."**

---

## What You Can Do in the UI

*Navigate each page as you say it.*

**"Train page — configure and launch training, watch the loss curve live. The model trains on GPU, logs go to TensorBoard, everything streams back to the browser."**

**"Predict page — pick a test trajectory, run a rollout. The backend runs the full 600-step simulation and saves the result. You get a progress bar via server-sent events. The model architecture buttons — GN, TNS, SAGE — let you pick which trained variant to use; they're disabled if that architecture hasn't been trained yet, so the UI always reflects what's actually available."**

**"Visualize — three animated panels side by side: ground truth, prediction, error. You can scrub through timesteps. For cloth there's a 3D viewer with linked cameras — drag one panel and the other rotates with it."**

**"Generate — the inverse design problem. You give a target drag value, and the system proposes mesh geometries that achieve it. Candidates stream in one by one as they're scored."**

---

## The One-Sentence Pitch

**"The system replaces an hours-long numerical solver with a millisecond neural network, wrapped in a production-quality full-stack application."**

*Pause. If they want more, continue to Level 2.*

---

# ━━━ LEVEL 2 — SOFTWARE DESIGN (5–10 min) ━━━
### Architecture, patterns, data pipeline, Docker, C++/Python

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│  React + TypeScript frontend  (Vite)                │
│  Canvas 2D mesh renderer  |  Three.js 3D cloth view │
└──────────────────────┬──────────────────────────────┘
                       │  REST + SSE
┌──────────────────────▼──────────────────────────────┐
│  FastAPI backend                                    │
│  ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐  │
│  │ /train   │ │/rollout │ │/results  │ │/generate│ │
│  └──────────┘ └─────────┘ └──────────┘ └────────┘  │
│  state.py — model cache, training handle, scorer   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  ML layer                                           │
│  Simulator (GNN)  |  CVAE  |  PoissonCorrector      │
│  NearestNeighborIndex  |  Normalizers               │
└─────────────────────────────────────────────────────┘
```

**"The backend is FastAPI. All shared mutable state — loaded model, training process handle, confidence index — lives in a single `state.py` module as singletons. Routers are thin: they validate input, delegate to the ML layer, and stream results back."**

---

## Data Pipeline — From Raw Data to Training

**"The original dataset from DeepMind is in TFRecord format — Google's protobuf-based binary that requires an old version of TensorFlow to parse. I wrote a one-time conversion script that reads the TFRecords and writes NumPy memmap files. After that the entire pipeline is pure PyTorch with zero TensorFlow dependency."**

**"I chose memory-mapped files for training data rather than HDF5. The training loop does millions of random trajectory accesses — pick a random trajectory, pick a random timestep. With memmaps, the OS manages paging: you access a slice and the kernel fetches just that page from disk. With HDF5, each random access decompresses a chunk first. For training-time random access, memmaps win."**

**"After the parse script completes it writes a sentinel file — `.dat.ok`. On startup, the dataset class checks for the sentinel before using the `.dat` file. Without this: a crash mid-parse leaves a partial binary file that looks valid but contains garbage. Silent corruption is the worst kind. The sentinel makes failure loud: no sentinel means no data access, clear error."**

**"For data versioning I use DVC — Data Version Control. The distinction from git-lfs matters: git-lfs just stores blobs externally. DVC understands that `train.dat` was produced FROM `train.tfrecord` BY `parse_tfrecord.py`. If the source changes, DVC knows the output is stale. It's pipeline-aware versioning, not just large-file storage."**

---

## Storage Layer — Repository Pattern

**"Results — predicted trajectories, ground truth — are stored as files. Early on that was just scattered `pickle.dump()` calls. When I wanted to add HDF5 support for partial timestep reads, I'd have had to find every call site."**

**"The fix is a Repository Pattern. I defined a `ResultRepository` Protocol — a Python structural interface — with seven methods: `save`, `load`, `load_timestep`, `list`, `exists`, `delete`, `get_path`. Then two concrete implementations: `PklResultRepository` for legacy pickle, and `HDF5ResultRepository` for compressed chunked storage. A `StorageFactory` reads a config file and returns whichever backend is configured."**

**"The key design choice: Protocol, not ABC. Python's `@runtime_checkable Protocol` means any class with the right method signatures satisfies the interface without inheriting from it. This enables structural duck typing with type checking — a test mock doesn't need to inherit from anything, it just needs the right methods. `isinstance(repo, ResultRepository)` still returns True."**

**"HDF5 adds something memmaps can't: efficient partial reads. Results are chunked `(1, N, D)` — one timestep per chunk. `load_timestep(key, t)` reads exactly one chunk from disk, not the whole file. When the Visualize page requests frame 47, it doesn't deserialize all 600 frames."**

---

## SOLID Principles — Concrete Examples

**"I applied all five SOLID principles. The ones I'd highlight:"**

**"Single Responsibility: every module does one thing. The encoder encodes, the confidence index scores, the ingest pipeline ingests. The early `rollout.py` had storage, inference, and display logic all mixed — extracting `ResultRepository` isolated the storage concern and made each piece independently testable."**

**"Open/Closed: the ingest pipeline accepts any solver format via a `SolverAdapter` Protocol — four methods: `list_splits`, `load_split`, `source_path`, `name`. Adding OpenFOAM support means writing one new adapter class. No existing pipeline stage changes. The `StorageFactory` has the same property: adding Zarr storage is a new class plus one line in the factory."**

**"Dependency Inversion: API routes are typed against `ResultRepository` the Protocol, never `PklResultRepository` the concrete class. The route doesn't know or care what's on disk — it calls `repo.save()` and the factory's choice handles the rest."**

---

## Design Patterns — say the ones that come up naturally

**Strategy + Open/Closed** *(Generate page, domain switching)*:

**"The design generation logic uses a Strategy pattern. There's a `BaseDesignSampler` abstract class with one interface. CFD and cloth each have their own concrete implementation. Adding a new physics domain means writing one class and adding one line to a registry dictionary — no existing code changes."**

**Double-checked locking** *(model loading)*:

**"Model loading uses double-checked locking. Fast path: check if the model is already cached without acquiring a lock — this is the common case and it's free. Slow path: acquire the lock, check again, then load. This avoids lock contention on every inference request while staying thread-safe when two requests race to load the same checkpoint."**

**LRU Cache** *(results page)*:

**"The pickle result files are 18MB each. When the Visualize page opens, the browser fires three parallel requests — metadata, RMSE curve, and frame 0 — all hitting the same file. Without caching that's three independent deserialization passes, 1–2 seconds each. An LRU cache keyed by filename and modification time means the first request pays the cost and the other two are instant."**

**Filesystem mutex / sentinel** *(duplicate training guard)*:

**"There's a subtle concurrency bug in remote GPU training: SSH takes 5–15 seconds to start, so there's a window between the API responding 'training started' and the actual PID file being written. A second `/train/start` call in that window would see no PID and spawn a duplicate process. I handle this with a sentinel file — written synchronously at the very start of the launch handler, with a 120-second TTL. Any concurrent request that sees the sentinel gets a 409 immediately. It's essentially a distributed mutex using the filesystem."**

---

## C++ / Python Integration — the k-d Tree

**"The confidence scoring uses a nearest-neighbor index. For every candidate design at inference time, we query: how close is this to anything in the training set? If it's far away, the model is extrapolating — low confidence."**

**"The query is on the hot path — it runs for every candidate during generation. I implemented the k-d tree in C++ with a pool allocator: instead of `new`/`delete` per node, the pool pre-allocates a contiguous block for all 2n+1 nodes at build time. After construction there are zero heap allocations during queries. No GC pressure, no heap fragmentation, deterministic latency."**

**"It's exposed to Python via pybind11 as a drop-in replacement for scipy. The data flows as raw NumPy float32 pointers — no copying. And there's a three-tier fallback: FAISS if installed, C++ if compiled, scipy always."**

*If asked why the three tiers:*

**"FAISS is the fastest at high embedding dimensions because it uses AVX-512 SIMD and at high dimensions brute-force with vectorization beats tree traversal. The C++ tree is exact and fast enough for small training sets. Scipy is always available. You get maximum performance when you've done the setup, and it just works otherwise."**

---

## Docker Deployment

**"The project is fully Dockerized. There are two containers: a Python/FastAPI API container and a React frontend served by nginx. The frontend uses a multi-stage build — stage one is a Node.js build environment that runs `npm run build` and produces static files, stage two is a bare nginx image that just serves those static files. The build tools don't end up in the final image: it goes from ~490MB to ~22MB."**

**"The API container uses CPU-only PyTorch. GPU training runs on a separate machine via SSH dispatch — the API writes a config JSON, SSH-launches the training or rollout script on the GPU machine, and streams SSE progress back. This avoids the complexity of NVIDIA Container Runtime and GPU driver compatibility in Docker."**

**"There's one subtle nginx config that matters for streaming: `proxy_buffering off` and `chunked_transfer_encoding on`. Without those, nginx buffers the entire SSE response and flushes it at the end — the user sees a frozen screen for ten minutes and then all progress events arrive at once, which is useless. Those two lines are what makes live streaming work through the proxy."**

---

## Frontend Performance

**"The mesh renderer is Canvas 2D, not SVG. A 2000-node CFD mesh has ~4000 triangles. SVG would create 4000 DOM elements and re-render them every frame — the browser can't keep up. Canvas draws all triangles in a single imperative pass with no DOM overhead, well above 60fps. For cloth I used Three.js with WebGL because cloth is 3D and you need real normals and lighting to make it legible."**

**"Server-Sent Events for all streaming — training logs, rollout progress, design candidates. SSE is one-directional push, which is exactly what you need here. WebSockets would add bidirectional complexity with no benefit."**

---

## Testing

**"Every subsystem has its own test file. The tests focus on contracts: correct output shapes, scores in valid ranges, save/load round trips. TDD — tests written before or alongside implementation. When I added pressure mode to the simulator, shape tests immediately caught every place the output-size assumption had leaked. That's the payoff: regressions surface as test failures, not silent wrong results."**

**"The Repository Pattern tests include explicit protocol conformance checks — `isinstance(repo, ResultRepository)` must return True for every implementation. If someone writes a new backend that's missing a method, the test fails immediately."**

*Pause. If they want the ML internals, continue to Level 3.*

---

# ━━━ LEVEL 3 — ML INTERNALS (5–10 min) ━━━
### GNN architecture, training details, physics connections

---

## The GNN — Encoder / Processor / Decoder

**"The architecture has three stages. The Encoder takes the raw node features — 11 dimensions for CFD: a 9-dimensional one-hot for node type plus 2D velocity — and maps everything to a uniform 128-dimensional latent space using separate MLPs for nodes and edges. LayerNorm after the encoder normalizes the scale differences between feature types."**

**"The Processor runs 15 rounds of message passing. Each round: the EdgeBlock concatenates sender, receiver, and current edge embeddings, passes them through an MLP, gets an updated edge. The NodeBlock scatter-aggregates all updated incoming edges at each node, concatenates with the node embedding, passes through an MLP. Both add residual connections."**

**"The Decoder is a single linear projection from 128 to 2 — the predicted velocity change. No LayerNorm at the output, because that feeds directly into a physics integrator and we need unconstrained real values."**

---

## Three Processor Architectures

**"I implemented three interchangeable processor blocks — same encoder, same decoder, same training loop, different aggregation strategy."**

> **GnBlock** — scatter-sum aggregation. The paper default. Fast, proven, works well for CFD.

> **TNSBlock** — multi-head attention. The query is the current node embedding, keys and values are incoming edge embeddings. The attention weights are learned — they tell you which neighbors are driving the physics at each node. Use when interpretability matters or when a few edges dominate.

> **SAGEBlock** — mean aggregation. Divides the aggregated neighbor features by the local node degree. This is important for cloth: corner nodes have 2–3 neighbors, interior nodes have 6+. With sum aggregation high-degree nodes dominate. Mean aggregation normalizes by valence, keeping gradients well-scaled regardless of mesh topology.

**"Switching between them is one config line. The Predict page shows GN / TNS / SAGE buttons — each button shows the best checkpoint for that architecture (lowest validation loss), and is disabled if that architecture hasn't been trained. That way the UI always reflects what actually exists on disk."**

---

## Why Residuals in the Processor?

**"Without residuals, 15 sequential MLPs act like 15 matrix multiplications — gradients vanish or explode before they reach the first layer. Residuals let each block learn what to add to the current state, not what the state should be. That's a much easier target, and it gives gradients a direct path back through the stack."**

---

## Online Normalizer — Why It Matters

**"Normalization statistics are computed online during training, not as a preprocessing step. The normalizers accumulate running mean and variance for the first million batches, then freeze. This means the normalization is baked into the checkpoint — no separate stats file to manage or lose. The clamp before sqrt prevents NaN from floating-point underflow when variance approaches zero."**

---

## Training — Noise Injection and Cloth Protocol

**"Noise injection is the key training trick from the paper. During training I add Gaussian noise to the input velocity field at each step. This simulates the accumulated prediction error of an autoregressive rollout — the model sees slightly noisy inputs at train time, so it's not shocked by its own imperfect outputs at test time. Without this, the model overfits to clean ground-truth inputs and error explodes after a few dozen rollout steps."**

**"For cloth I follow DeepMind's training protocol more closely: exponential LR decay per step — gamma = 0.1^(1/5,000,000), floor at 1e-6. And a 1000-step normalizer warmup at the start: the forward pass runs, normalizers accumulate statistics, but the backward pass is skipped. Cloth world positions span a much wider range than CFD velocities — the warmup ensures the normalizers have seen the full range before the first gradient update."**

---

## Confidence Scoring — Embedding Space + JEPA Connection

**"The GNN will always produce a prediction — there's nothing in the architecture that says 'I don't know'. For a mesh the model was never trained on, the output might be garbage with no warning."**

**"The solution is a nearest-neighbor index over the training set's embedding space. During index build, every training trajectory is run through the frozen GNN encoder, node embeddings are mean-pooled to a single vector, and stored in the index. At inference, the test input is encoded the same way. The confidence score is: one minus the distance to the nearest training neighbor, normalized by the training set's own spread."**

**"For CFD there's a subtlety: two simulations with identical mesh geometry but different inlet velocities look identical at frame 0 — same node positions, same initial velocity. The model can't distinguish them by geometry alone. I use dual-frame embeddings: concatenate the frame-0 encoding with the frame-5 encoding. By frame 5, the boundary conditions have propagated into the mesh interior, so the two flows look different. Single frame embeds 128 dimensions; dual frame gives 256."**

**"The index stores a SHA-256 hash of the checkpoint it was built from. On load, it recomputes the hash and raises `IndexStaleError` if they don't match. This prevents a subtle production bug: retrain the model, forget to rebuild the index, and all your confidence scores are based on an old model's embedding space — silently wrong. Fail-fast is the right call here."**

**"This is JEPA-adjacent — Joint Embedding Predictive Architectures. The idea is to learn a representation space where similar inputs cluster together, and use distance in that space as a proxy for uncertainty. Our encoder is essentially doing that: similar physics → nearby embeddings → high confidence."**

---

## Poisson Pressure Correction — Physics as a Post-Processing Layer

**"One issue with a GNN surrogate is that it doesn't strictly enforce physical constraints. Incompressible flow requires the velocity field to be divergence-free: ∇·u = 0. The GNN learns an approximation — each step introduces a small divergence error that accumulates over 600 timesteps."**

**"The fix comes from the Helmholtz-Hodge decomposition: any vector field can be split into a divergence-free part and a gradient part. So u_predicted = u_divergence_free + ∇φ. If I can find φ, I can subtract it and get a physically consistent field. Finding φ means solving the Poisson equation: ∇²φ = ∇·u_predicted."**

**"Discretized on the mesh, that's a sparse linear system: L·φ = b, where L is the graph Laplacian. L is sparse, symmetric, and positive definite after fixing one pressure reference node as a Dirichlet boundary condition. I factorize it once per rollout with sparse LU — `scipy.sparse.linalg.splu` — and then for each of the 600 timesteps I call `lu.solve(b)`. Factor once at O(n^1.5), solve 600 times at O(n) each."**

**"So yes — LU decomposition IS used at runtime in this project, as a physics enforcement layer on top of the GNN. It's opt-in via a UI checkbox because it adds ~10-15% overhead and only makes sense for incompressible flow."**

---

## The Physics the GNN Replaces — Traditional Solvers

**"To really understand what the GNN does, it helps to know what it replaces. A traditional FEM cloth solver at each timestep solves:**

```
(M + h²K) · u_{t+1}  =  M · u_t + h · f_ext
```

**K is the stiffness matrix assembled from the mesh. M is the mass matrix. This is a linear system — you factorize the left side once with LU decomposition, cost O(n³), then each timestep is just two triangular solves, O(n²). The factorization is reused because K only changes if the mesh topology changes."**

**"For CFD, the pressure Poisson equation ∇²p = rhs becomes L·p = rhs where L is the graph Laplacian — sparse, symmetric positive definite. There you'd use Cholesky rather than LU, which costs half as much by exploiting symmetry. For large meshes you'd use Conjugate Gradient to avoid fill-in entirely."**

**"MeshGraphNets replaces both of these with a single GNN forward pass. No matrix assembly, no factorization, no triangular solve — unless you add the Poisson correction layer, which adds one LU solve back in to enforce incompressibility. The tradeoff is approximation error that accumulates over the rollout — the RMSE chart shows it's roughly linear for the first 200–300 steps, then accelerates as errors compound."**

**"One research direction I find interesting: use the GNN as a preconditioner for CG. The GNN gives you a warm start close to the true solution, and CG corrects the residual. You'd get GNN speed for most steps, and solver accuracy where it matters."**

---

## Inverse Design — CVAE + Gradient Descent

**"The Generate tab inverts the simulation. A Conditional VAE learns the distribution of design parameters — cylinder position, radius, inlet velocity — conditioned on drag. At generation time: sample from the prior, condition on the target drag, decode to design parameters."**

**"I use Latin Hypercube Sampling instead of pure random normal sampling. With 10–20 candidates, pure random can cluster — five samples in one region, none in another. LHS divides each latent dimension into N equal strata and samples once per stratum. It guarantees that every region of the latent space is represented. At small sample counts, coverage matters more than unbiasedness."**

**"There are two generation strategies. Sampling draws from the CVAE prior — fast, diverse, good for exploration. Gradient mode treats the latent vector as a learnable parameter: the chain from latent → CVAE decoder → design params → MLP surrogate → drag is fully differentiable, so I run Adam on the latent vector directly, minimizing squared error against the target. This gives a higher-quality answer for a specific target but less diversity."**

**"Every candidate gets a confidence score using param-space OOD detection — a KDTree over the training set's design parameter vectors, not the GNN embeddings. The GNN embedding space is used on the Predict page for simulation-level confidence. The param-space OOD is used on the Generate page for candidate-level confidence. Different questions, different indices."**

---

## Quick Numbers — cite if asked

| Fact | Value |
|------|-------|
| GNN hidden dim | 128 |
| Message-passing rounds | 15 |
| Processor options | GnBlock / TNSBlock / SAGEBlock |
| CFD node features | 11 (9 one-hot + 2D velocity) |
| CFD edge features | 3 (Δx, Δy, distance) |
| Cloth edge features | 7 (world-space + mesh-space relative coords) |
| Rollout timesteps | 600 (CFD) |
| CVAE latent dim | 16 |
| CVAE LHS candidates | 10–20 (stratified, not random) |
| free_bits | 0.05 nats (posterior collapse guard) |
| Drag surrogate | 4 → 64 → 64 → 1 |
| Confidence backends | FAISS → C++ KDTree → scipy |
| CFD confidence embedding | 256-dim dual-frame (frame 0 + frame 5) |
| Cloth confidence embedding | 128-dim single-frame |
| Confidence train_diameter | 95th percentile of 5-NN distances in training set |
| Poisson correction | Opt-in, sparse LU on 5-NN graph Laplacian |
| BPTT steps (gradient descent) | K=5 |
| C++ pool size | 2n+1 nodes pre-allocated |
| LRU cache | 8 pkl files, keyed by (filename, mtime) |
| Cloth LR decay | 0.1^(step / 5,000,000), floor 1e-6 |
| Cloth normalizer warmup | 1000 steps (forward only) |
| Cloth noise std | 3e-3 |
| CFD noise std | 2e-2 |
| Duplicate launch TTL | 120 s sentinel file |
| Storage backends | PKL (legacy) → HDF5 (chunked, partial reads) → Zarr (cloud-native) |
| HDF5 chunk shape | (1, N, D) — one timestep per chunk |

---

*End of script. Level 1 → Level 2 → Level 3. Stop wherever they stop asking.*
