# PhysIQ — Data Flow & Architecture Diagrams

---

## 1. Full Data Flow (Raw → Train → Predict → Generate)

```mermaid
flowchart TD
    subgraph RAW["📦 RAW DATA  (DeepMind)"]
        TF1["TFRecord\ncylinder_flow"]
        TF2["TFRecord\nflag_simple"]
    end

    subgraph PARSE["⚙️ PARSE  (one-time, needs TF)"]
        P1["parse_tfrecord.py"]
        P2["parse_flag_tfrecord.py"]
    end

    subgraph STORAGE_TRAIN["💾 TRAINING DATA"]
        DAT["train/valid/test .dat\n(numpy memmap, float32)"]
        NPZ["train/valid/test .npz\n(metadata: shapes, mesh, indices)"]
        SENT1[".dat.ok sentinels\n(crash-safe parse guard)"]
        DVC["DVC versioning\n→ remote S3 / GCS"]
    end

    subgraph DATASET["🔄 DATASET LAYER"]
        FPC["dataset/fpc.py\nnp.memmap  O(1) random access\nPyG Data objects per timestep"]
    end

    subgraph TRAIN["🧠 TRAINING"]
        NORM["Online Welford Normalizer\n(baked into checkpoint)"]
        NOISE["Noise injection\n(train-test gap fix)"]
        GNN["GNN — MeshGraphNets\n\nEncoder  →  Processor ×15  →  Decoder\nMLP 11→128    GnBlock  (sum)    MLP 128→2\n+ LayerNorm   TNSBlock (attention)\n              SAGEBlock (mean)"]
        LOSS["MSE loss on Δvelocity\n(NORMAL + OUTFLOW nodes)"]
    end

    subgraph CKPT["📌 CHECKPOINT (.pth)"]
        CPT["weights + normalizer stats\n+ architecture tag (gn/tns/sage)"]
        IDX["NearestNeighborIndex\ndual-frame 256-dim embeddings\nSHA-256 stale check"]
    end

    subgraph PREDICT["🔮 PREDICT"]
        ROLL["rollout.py / rollout_ssh.py\n600 autoregressive GNN steps"]
        POISSON["Poisson Pressure Correction\n(opt-in)\nsparse LU  ∇·u = 0\nfactor once → solve 600×"]
        OOD["OOD Confidence\n256-dim embed vs KDTree\nscore 0–1"]
    end

    subgraph GENERATE["🎨 GENERATE  (Inverse Design)"]
        CVAE["CVAE\n(mesh_params, drag) → z[16]\nLatin Hypercube Sampling\nfree_bits=0.05"]
        GRAD["Gradient Refinement\nAdam on z, BPTT K=5\nRealMeshLookup → valid mesh"]
        SURR["Drag Surrogate MLP\n4→64→64→1\nParamSpaceOOD confidence"]
    end

    subgraph RESULTS["🗄️ RESULTS STORAGE"]
        FACT["StorageFactory\nreads runs/storage_config.json"]
        PKL["PklRepository\n.pkl  legacy"]
        HDF5["HDF5Repository\n.h5  gzip\nchunk (1,N,D)\npartial timestep reads"]
        ZARR["ZarrArchive\n.zarr  Blosc/LZ4\ncloud-native  S3-ready\n.zarr.ok sentinel"]
    end

    subgraph FRONTEND["🖥️ FRONTEND  (React + Vite)"]
        DS["Dataset Studio\nmesh preview\nnode type breakdown"]
        TR["Train\nSSE loss stream\nTensorBoard"]
        PR["Predict\narch buttons GN/TNS/SAGE\nconfidence badge"]
        VIZ["Visualize\nCanvas 2D CFD\nThree.js 3D cloth\nlinked cameras"]
        GEN["Generate\ncandidates stream SSE\nthumbnails + OOD score"]
    end

    TF1 --> P1
    TF2 --> P2
    P1 --> DAT & NPZ & SENT1
    P2 --> DAT & NPZ & SENT1
    DAT & NPZ --> DVC
    DAT & NPZ --> FPC
    FPC --> NORM --> NOISE --> GNN --> LOSS
    GNN --> CPT
    CPT --> IDX
    CPT --> ROLL
    CPT --> CVAE
    CPT --> GRAD
    ROLL --> POISSON --> OOD
    OOD --> FACT
    CVAE --> SURR
    GRAD --> SURR
    SURR --> GEN
    FACT --> PKL & HDF5 & ZARR
    PKL & HDF5 & ZARR --> VIZ
    CPT --> PR
    FPC --> DS
    GNN --> TR
```

---

## 2. Storage Layer — Repository Pattern (Multi-Format Support)

```mermaid
classDiagram
    class ResultRepository {
        <<Protocol>>
        +save(key, data) None
        +load(key) dict
        +load_timestep(key, t) dict
        +list() list[str]
        +exists(key) bool
        +delete(key) bool
        +get_path(key) Path
    }

    class PklResultRepository {
        -result_dir: Path
        +save(key, data)
        +load(key)
        +load_timestep(key, t)
        +list()
        +exists(key)
        +delete(key)
        +get_path(key)
    }

    class HDF5ResultRepository {
        -result_dir: Path
        -chunk_shape: tuple (1,N,D)
        -compression: gzip-4
        +save(key, data)
        +load(key)
        +load_timestep(key, t)  ← reads 1 chunk only
        +list()
        +exists(key)
        +delete(key)
        +get_path(key)
    }

    class ZarrArchive {
        -zarr_root: Path
        -codec: Blosc/LZ4
        -sentinel: .zarr.ok
        +write_split(split, positions, velocities)
        +read_split(split)
    }

    class StorageFactory {
        +create(result_dir) ResultRepository
        -reads: runs/storage_config.json
        -backend: pkl | hdf5 | zarr
    }

    ResultRepository <|.. PklResultRepository : implements
    ResultRepository <|.. HDF5ResultRepository : implements
    StorageFactory --> ResultRepository : creates
    StorageFactory ..> PklResultRepository
    StorageFactory ..> HDF5ResultRepository
```

**Why Protocol, not ABC?**
Any class with the right method signatures satisfies `ResultRepository` — no inheritance needed.
`isinstance(repo, ResultRepository)` returns `True` for all implementations.
Add a new backend (e.g. `S3ResultRepository`) → write one class, add one line in `StorageFactory`. Zero changes to callers.

---

## 3. Ingest Pipeline — Open/Closed for New Solvers

```mermaid
flowchart LR
    subgraph ADAPTERS["SolverAdapter Protocol"]
        A1["TFRecordAdapter\n✅ implemented"]
        A2["OpenFOAMAdapter\n🔲 stub"]
        A3["AnsysAdapter\n🔲 future"]
    end

    subgraph PIPELINE["IngestPipeline.run()"]
        S1["1 Harvest\nadapter.list_splits()\nadapter.load_split()"]
        S2["2 Validate\nshapes, NaN/Inf check"]
        S3["3 Normalise\nrunning stats"]
        S4["4 Write\n.npz + manifest.json"]
        S5["5 Index\nDVC dvc.yaml update"]
    end

    A1 --> S1
    A2 --> S1
    A3 --> S1
    S1 --> S2 --> S3 --> S4 --> S5
```

**Adding a new solver** = implement 4 methods (`list_splits`, `load_split`, `source_path`, `name`).
No existing pipeline stage changes. This is the **Open/Closed Principle** directly applied.

---

## 4. Training Data Format Decision

```mermaid
flowchart LR
    subgraph FORMATS["Training data format options"]
        M["memmap .dat\n✅ CHOSEN\nO(1) random seek\nno decompression\nOS page cache"]
        H["HDF5\ngzip chunks\nhigher read latency\nfor random access"]
        Z["Zarr\nBlosc/LZ4\ncloud-native\nbest for streaming"]
    end

    subgraph USE["Use case"]
        TR["Training DataLoader\nmillions of random\ntrajectory accesses"]
        RES["Rollout results\nsequential write\npartial timestep reads"]
        ARCH["Long-term archive\ncloud push\nDVC remote"]
    end

    M --> TR
    H --> RES
    Z --> ARCH
```

**Rule of thumb:**
- **Random access at training time** → memmap `.dat`
- **Partial reads of results** → HDF5 with chunk `(1, N, D)`
- **Cloud storage / archival** → Zarr with Blosc/LZ4

---

## 5. Confidence Scoring Flow

```mermaid
sequenceDiagram
    participant U as User (browser)
    participant API as FastAPI
    participant GNN as Simulator (frozen)
    participant IDX as NearestNeighborIndex
    participant KD as KDTree (FAISS/C++/scipy)

    note over IDX: Built once after training<br/>SHA-256 hash of checkpoint stored

    U->>API: POST /rollout {trajectory}
    API->>GNN: encode(graph_t0) → embed[128]
    API->>GNN: encode(graph_t5) → embed[128]
    GNN-->>API: concat → embed[256]
    API->>IDX: load(expected_hash=checkpoint_sha256)
    IDX-->>API: ✅ hash matches / ❌ IndexStaleError
    API->>KD: query(embed[256], k=5)
    KD-->>API: d_min (distance to nearest training point)
    API-->>U: confidence = clip(1 - d_min/train_diameter, 0, 1)
```

---

## 6. Inverse Design Flow

```mermaid
flowchart TD
    T["Target drag  e.g. Cd = 0.3"]

    subgraph PHASE1["Phase 1 — CVAE Sampling"]
        LHS["Latin Hypercube Sample\nz ~ LHS(n=10-20) in latent[16]"]
        DEC["CVAE Decoder\n(z, target_drag) → mesh_params"]
        RML["RealMeshLookup\nsnap to nearest valid training mesh\n(KDTree on param space)"]
    end

    subgraph PHASE2["Phase 2 — Gradient Refinement"]
        ADAM["Adam on z\nBPTT K=5 through GNN\nminimise (drag_pred - target)²"]
    end

    subgraph SCORE["Scoring"]
        SURR["Drag Surrogate MLP\n4→64→64→1\ninstant score per candidate"]
        POOD["ParamSpaceOOD\nKDTree on training param vectors\nconfidence 0–1"]
    end

    SSE["SSE stream to browser\ncandidates arrive one by one\nthumbnail + drag score + confidence"]

    T --> LHS --> DEC --> RML --> ADAM
    ADAM --> SURR & POOD
    SURR & POOD --> SSE
```

---

## Key Design Patterns Summary

| Pattern | Where used | Why |
|---------|-----------|-----|
| **Repository** | `ResultRepository` Protocol → PKL / HDF5 / Zarr | Swap storage backend in one config line, callers unchanged |
| **Factory** | `StorageFactory.create()` | Centralised creation, config-driven |
| **Protocol (structural typing)** | `ResultRepository`, `SolverAdapter` | Duck typing with type safety, no inheritance needed |
| **Strategy** | `BaseDesignSampler` → CFD / Cloth | Swap physics domain without touching generation logic |
| **Open/Closed** | `IngestPipeline` + `SolverAdapter` | New solver = new adapter class, zero existing changes |
| **Sentinel files** | `.dat.ok`, `.zarr.ok` | Fail-fast on corrupt/partial writes |
| **LRU Cache** | Model loading, result pickle | Amortise expensive deserialization across repeated requests |
| **Double-checked locking** | `api/state.py` model cache | Thread-safe loading with minimal lock contention |
| **SSE / Observer** | Train, Rollout, Generate routes | Progressive UI updates without polling |
| **Fail-fast** | `IndexStaleError`, sentinel check | Surface problems loudly, never silently wrong |
