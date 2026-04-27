# PhysIQ Technical Reference

> **Vault home** — start here. This index covers every technical note in the vault.

---

## What This Vault Is

This is the living technical reference for **PhysIQ / MeshGraphNets-PyTorch** — a physics simulation surrogate that uses Graph Neural Networks to replace expensive numerical solvers for fluid dynamics, cloth simulation, and related PDE-governed systems.

The vault is written as a series of *build-up stories*: each note starts from first principles and explains not just *what* was built, but *why*, *what alternatives existed*, and *what the tradeoffs cost us*. Reading these notes end-to-end will give you a complete mental model of the system.

**How to navigate**:
- If you're new: start with [[03_system_architecture]], then [[04_gnn_architecture]]
- If you're debugging: go directly to the relevant component note
- If you're preparing for a technical interview: use the [[#Quick Reference by Topic]] section below

---

## Note Index

| Note | What You'll Learn |
|------|-------------------|
| [[01_problem_statement]] | Why physics simulation is expensive, what a surrogate model is, and what PhysIQ promises to do |
| [[02_dataset_and_mesh]] | How simulation meshes are structured, what the training trajectories contain, coordinate systems, node/edge features |
| [[03_system_architecture]] | The full system overview: GNN core, FastAPI backend, React frontend, Docker deployment, GPU dispatch |
| [[04_gnn_architecture]] | The encoder-processor-decoder GNN in detail: node/edge MLPs, message passing, three processor variants (GN/TNS/SAGE), noise injection |
| [[05_confidence_scoring]] | How PhysIQ knows when to trust its own predictions: embedding extraction, KDTree nearest-neighbor lookup, OOD detection |
| [[06_inverse_design]] | The CVAE-based inverse design loop: how to find simulation inputs that produce a desired output, Latin Hypercube Sampling, surrogate optimization |
| [[07_poisson_correction_lu]] | The optional incompressibility correction: Poisson pressure equation, sparse LU decomposition, when to use it and when not to |
| [[08_data_pipeline]] | Full data lifecycle: raw TFRecord ingestion, preprocessing, normalization, PKL/HDF5/Zarr storage via the Repository Pattern, memmap training access |
| [[09_experiment_tracking]] | DVC pipelines, MLflow runs, checkpoint naming conventions, best-model tracking, how to reproduce any historical experiment |
| [[10_api_layer]] | FastAPI routes, SSE streaming for training progress, job management, background task lifecycle, error handling |
| [[11_frontend]] | React + TypeScript + Vite frontend: component architecture, SSE event consumption, live training dashboard, result visualization |
| [[12_testing_strategy]] | Test pyramid: unit tests for GNN components, integration tests for API routes, end-to-end smoke tests, physics regression tests |
| [[13_docker_deployment]] | Docker setup: multi-stage frontend build, CPU-only API container, SSH GPU dispatch pattern, nginx SSE configuration, Compose profiles |
| [[14_design_decisions_tradeoffs]] | Every major architectural decision: alternatives considered, rationale, and explicit costs accepted |
| [[15_equivariance]] | Is the GNN equivariant? What that means, exactly where it breaks, 3-phase roadmap to add E(2)/E(3) equivariance with e3nn, test to verify |
| [[16_data_access_and_confidence]] | How `.dat` + `.npz` gives O(1) random access (memmap mechanics, byte-offset arithmetic), why HDF5 is wrong for training, and exactly how `train_diameter` is computed and used in confidence scoring |
| [[17_tfrecord_to_graph_pipeline]] | Full pipeline from raw TFRecord to GNN training: TFRecord fields, parse_tfrecord.py flatten, FpcDataset O(1) access, PyG transformer (FaceToEdge → Cartesian → Distance), Simulator.forward normalise/encode/process/decode — with tensor shapes at every stage and CFD vs cloth side-by-side |
| [[18_inverse_design_pipeline]] | Inverse design end-to-end: how cx/cy/r/v_inlet are extracted, CVAE encoder/decoder architecture, CVAE training loss (recon + KL + physics consistency), Latin Hypercube Sampling, RealMeshLookup snapping, K=5 BPTT gradient refinement, ParamSpaceOOD vs NearestNeighborIndex — all with Mermaid diagrams |
| [[19_generate_full_flow]] | Full flow from user clicking Generate to candidates in browser: sample path (LHS → CVAE decode → mesh → surrogate → OOD → SSE stream) and gradient path (Adam in latent space → why GNN gradient is disabled → surrogate chain → diverse candidates around best_z*) with all SSE events explained |
| [[20_inverse_design_backprop]] | The full differentiable path for both domains: CFD (z → CVAE decoder → DragSurrogate MLP → loss, _use_gnn=False, why discrete KDTree kills cx/cy/r gradient) vs Cloth (z → CVAE decoder → PCA⁻¹ → 5×GNN BPTT → stress → loss), tensor shapes at every step, why prev_x must not be detached |
| [[21_training_backprop]] | How the GNN learns: CFD (MSE on normalized Δv, noise injection, NORMAL+OUTFLOW mask) vs Cloth (MSE on Verlet acceleration, sum-xyz loss, HANDLE mask, edges rebuilt every step), what weights get updated, what stays frozen, and how this differs from inverse design backprop |

---

## Quick Reference by Topic

Use this section when a specific question arises and you need the right note immediately.

### Architecture & Core ML

| Question | Go to |
|---|---|
| "How does the GNN work? Walk me through the architecture." | [[04_gnn_architecture]] |
| "Why GNNs and not CNNs for this problem?" | [[14_design_decisions_tradeoffs#1. GNN on Graphs vs CNN on Grids]] |
| "What are the three processor variants and when would you use each?" | [[04_gnn_architecture]] · [[14_design_decisions_tradeoffs#2. Three Processor Variants vs Single Architecture]] |
| "How do you handle error accumulation in autoregressive rollout?" | [[14_design_decisions_tradeoffs#3. Autoregressive Rollout vs Direct Multi-Step Prediction]] · [[14_design_decisions_tradeoffs#4. Noise Injection]] |
| "Why predict accelerations instead of velocities?" | [[14_design_decisions_tradeoffs#5. Predict Accelerations vs Absolute Velocities]] |
| "What is BPTT and how do you use it here?" | [[14_design_decisions_tradeoffs#10. BPTT K=5 Steps]] |

### Data & Storage

| Question | Go to |
|---|---|
| "Tell me about your data pipeline." | [[08_data_pipeline]] |
| "How do you store simulation results? Why not just pickle?" | [[14_design_decisions_tradeoffs#7. Storage Evolution PKL → HDF5 → Zarr]] · [[08_data_pipeline]] |
| "What is the Repository Pattern and why use it here?" | [[08_data_pipeline]] · [[14_design_decisions_tradeoffs#11. Protocol vs ABC]] |
| "How do you handle training data access at scale?" | [[14_design_decisions_tradeoffs#14. Memmap for Training Data]] |
| "How do you version datasets?" | [[09_experiment_tracking]] · [[14_design_decisions_tradeoffs#13. DVC vs git-lfs]] |

### Physics & Correctness

| Question | Go to |
|---|---|
| "Where does LU decomposition appear in the system?" | [[07_poisson_correction_lu]] |
| "What is the Poisson pressure correction and why is it opt-in?" | [[07_poisson_correction_lu]] · [[14_design_decisions_tradeoffs#6. Poisson Correction Opt-In vs Always-On]] |
| "How do you ensure the model produces physically consistent outputs?" | [[07_poisson_correction_lu]] · [[04_gnn_architecture]] |
| "Is your GNN equivariant?" | [[15_equivariance]] |
| "How would you add rotational equivariance?" | [[15_equivariance#5. How to Make It Equivariant]] |
| "What is e3nn and how does it work?" | [[15_equivariance#Option C: Steerable Features with e3nn]] |

### Inference & Deployment

| Question | Go to |
|---|---|
| "How do you handle train-test distribution shift at inference time?" | [[05_confidence_scoring]] |
| "How do you know when the model's predictions can be trusted?" | [[05_confidence_scoring]] |
| "How does inverse design work?" | [[06_inverse_design]] |
| "Why Latin Hypercube Sampling for the CVAE?" | [[14_design_decisions_tradeoffs#9. Latin Hypercube Sampling vs Random Normal]] |
| "How do you deploy this?" | [[13_docker_deployment]] |
| "Why not run CUDA inside Docker?" | [[13_docker_deployment#CPU-Only Containers and SSH GPU Dispatch]] · [[14_design_decisions_tradeoffs#12. SSH GPU Dispatch]] |

### API & Frontend

| Question | Go to |
|---|---|
| "How does the live training progress work technically?" | [[10_api_layer]] · [[13_docker_deployment#SSE Through nginx]] |
| "What is SSE and how do you stream training events?" | [[10_api_layer]] · [[11_frontend]] |
| "Why does nginx need special configuration for SSE?" | [[13_docker_deployment#SSE Through nginx The Buffering Problem]] |
| "How is the frontend built and served?" | [[11_frontend]] · [[13_docker_deployment#Multi-Stage Frontend Build]] |

### Design Philosophy

| Question | Go to |
|---|---|
| "Why did you use X instead of Y?" (general) | [[14_design_decisions_tradeoffs]] |
| "What tradeoffs did you consciously accept?" | [[14_design_decisions_tradeoffs#Decisions Pending Under Review]] |
| "How do you test a physics simulation surrogate?" | [[12_testing_strategy]] |
| "How do you track experiments and ensure reproducibility?" | [[09_experiment_tracking]] |

---

## System at a Glance

The full system in one diagram. Follow the data from raw simulation to deployed surrogate.

```mermaid
flowchart TD
    subgraph Data["📁 Data Layer"]
        TF[TFRecord / HDF5 files]
        MEMMAP[numpy memmap\ntraining index]
        DVC[DVC remote\ndataset versions]
        TF --> MEMMAP
        TF --> DVC
    end

    subgraph Training["🧠 Training"]
        LOADER[DataLoader\nrandom mini-batch]
        NOISE[Noise injection\nε ~ N(0, σ²)]
        ENC[Encoder\nnode + edge MLPs]
        PROC["Processor\nGN / TNS / SAGE\nL layers of message passing"]
        DEC[Decoder\nMLP → Δv]
        INTEG[Euler integration\nv_{t+1} = v_t + Δv]
        BPTT[BPTT K=5\ngradient + Adam step]
        CKPT[Checkpoint\n.pt file]

        LOADER --> NOISE --> ENC --> PROC --> DEC --> INTEG --> BPTT
        BPTT -->|"update weights"| ENC
        BPTT --> CKPT
    end

    subgraph Inference["🔍 Inference"]
        ROLL[Autoregressive rollout\n600 steps]
        CONF[Confidence scorer\nKDTree on embeddings]
        POISSON["Poisson correction\n∇²p = ∇·v\nLU solve (opt-in)"]
        INV[Inverse design\nCVAE + LHS sampling]

        CKPT --> ROLL --> POISSON --> CONF
        INV --> ROLL
    end

    subgraph API["🌐 API Layer (FastAPI)"]
        ROUTES[REST routes\nPOST /train, GET /predict]
        SSE[SSE stream\ntraining progress events]
        JOB[Job manager\nasync background tasks]
        GPU[SSH → GPU host\nremote training dispatch]

        ROUTES --> JOB --> GPU
        GPU -->|stdout stream| SSE
        ROLL --> ROUTES
    end

    subgraph Frontend["🖥️ Frontend (React + Vite)"]
        DASH[Training dashboard\nlive loss chart]
        VIZ[Result visualizer\nmesh + velocity field]
        INVUI[Inverse design UI\nparameter explorer]

        SSE --> DASH
        ROUTES --> VIZ
        ROUTES --> INVUI
    end

    subgraph Docker["🐳 Docker (Compose)"]
        APICONT[api container\nCPU-only, port 8000]
        NGINX[nginx container\nport 80, SSE proxy]

        APICONT -->|reverse proxy| NGINX
    end

    subgraph Experiments["📊 Experiment Tracking"]
        MLFLOW[MLflow\nmetrics + artifacts]
        DVC2[DVC\npipeline stages]

        BPTT --> MLFLOW
        CKPT --> DVC2
    end

    MEMMAP --> LOADER
    API --> APICONT
    Frontend --> NGINX
```

---

## Key Numbers (Quick Reference)

| Quantity | Value |
|---|---|
| Typical mesh size | 1,000 – 50,000 nodes |
| Node features | 5–7 (position x/y, velocity vx/vy, node type, ...) |
| Edge features | 3 (Δx, Δy, distance) |
| Rollout length (training) | 600 timesteps |
| Rollout length (inference) | User-configurable; default 600 |
| Processor layers L | 15 (GN default) |
| Latent dimension | 128 |
| BPTT window K | 5 |
| Noise σ | ~0.003 × feature_std |
| LHS candidates (inverse design) | 10–20 |
| KDTree → FAISS threshold | 100k trajectories |
| API container size | ~1.2 GB |
| Frontend image size | ~22 MB |

---

## How Notes Are Written

Each note follows the **build-up story** structure:

1. **The problem** — what challenge or requirement motivated this component
2. **The solution** — what we built, with code snippets and diagrams
3. **The tradeoffs** — what alternatives were considered, what costs were accepted

If you find a note that doesn't explain *why* — only *what* — that's a gap to fill.

---

*Last updated: 2026-04-19 · Vault maintained alongside `main` branch of `meshGraphNets_pytorch`.*
