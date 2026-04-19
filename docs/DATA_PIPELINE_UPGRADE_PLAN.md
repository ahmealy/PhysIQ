# Data Pipeline Upgrade Plan

Status: IN PROGRESS  
Started: 2026-04-19  
Last updated: 2026-04-19

---

## New Dependencies (all local, no cloud)

```bash
pip install dvc zarr h5py cachetools
```

---

## Phase 1 — Quick Wins ✅ COMPLETE
*No architecture changes, all standalone. Deploy same day.*

| # | What | Files | Status |
|---|---|---|---|
| 1.1 | `.dat.ok` sentinel — crash protection on parse | `parse_tfrecord.py`, `dataset/fpc.py` | ✅ |
| 1.2 | `num_workers=4` + `pin_memory=True` in DataLoader | `train.py` | ✅ |
| 1.3 | Result retention CLI — `python -m result.retention --keep 10` | new `result/retention.py` | ✅ |
| 1.4 | LRU model cache (max 3) | `api/state.py` | ✅ |
| 1.5 | DVC init — track `data/*.dat`, `data/*.npz` | `.dvc/`, `data/*.dvc` | ✅ |

---

## Phase 2 — Storage Layer ✅ COMPLETE
*Additive — nothing breaks, old files keep working.*

| # | What | Files | Status |
|---|---|---|---|
| 2.1 | `ResultRepository` Protocol + `PklRepository` wrapper | `storage/protocols.py`, `storage/pkl_repository.py` | ✅ |
| 2.2 | Wire Repository into API routes | `api/routes/results.py`, `api/routes/physics.py` | ✅ |
| 2.3 | `HDF5Repository` — compressed rollout files, partial timestep reads | `storage/hdf5_repository.py` | ✅ |
| 2.4 | Migration script — `scripts/migrate_pkl_to_hdf5.py` | scripts/ | ✅ |
| 2.5 | `StorageFactory` — swap backend via config key `result_backend=hdf5` | `storage/factory.py` | ✅ |
| 2.6 | Zarr archive layer — write alongside memmap during parse | `storage/zarr_archive.py` | ✅ |
| 2.7 | `scripts/regenerate_dat.py` — rebuild `.dat` from Zarr if lost | scripts/ | ✅ |

---

## Phase 3 — Ingest Pipeline ✅ COMPLETE
*Replaces monolithic `parse_tfrecord.py` with composable stages.*

| # | What | Files | Status |
|---|---|---|---|
| 3.1 | `SolverAdapter` Protocol + `TFRecordAdapter` | `ingest/protocols.py`, `ingest/adapters/tfrecord.py` | ✅ |
| 3.2 | `IngestPipeline` + stages: harvest → validate → normalise → write → index | `ingest/pipeline.py`, `ingest/stages/*.py` | ✅ |
| 3.3 | Dataset manifest — `data/manifest.json` | `ingest/stages/write.py` | ✅ |
| 3.4 | Auto-rebuild confidence index post-ingest | `ingest/stages/index.py` | ✅ |
| 3.5 | `parse_tfrecord.py` kept as-is (shim deferred) | `parse_tfrecord.py` | ✅ |
| 3.6 | `OpenFOAMAdapter` stub | `ingest/adapters/openfoam.py` | ✅ |

---

## Final Directory Structure

```
meshGraphNets_pytorch/
  ingest/                    ← NEW Phase 3
    __init__.py
    protocols.py
    pipeline.py
    adapters/
      tfrecord.py
      openfoam.py
    stages/
      harvest.py
      validate.py
      normalise.py
      write.py
      index.py

  storage/                   ← NEW Phase 2
    __init__.py
    protocols.py
    pkl_repository.py
    hdf5_repository.py
    zarr_archive.py
    factory.py

  result/
    retention.py             ← NEW Phase 1
    *.h5                     ← Phase 2 new rollouts
    *.pkl                    ← legacy, kept during transition

  scripts/                   ← NEW Phase 2
    migrate_pkl_to_hdf5.py
    regenerate_dat.py

  api/
    model_cache.py           ← NEW Phase 1

  data/
    train.dat                (unchanged)
    train.dat.ok             ← NEW Phase 1 sentinel
    manifest.json            ← NEW Phase 3
    archive.zarr/            ← NEW Phase 2
```

---

## Design Patterns

| Pattern | Phase | Location |
|---|---|---|
| Repository | 2 | `storage/protocols.py` |
| Strategy | 2 | `storage/factory.py` |
| Adapter | 3 | `ingest/adapters/` |
| Pipeline | 3 | `ingest/pipeline.py` |
| Factory | 2 | `storage/factory.py` |
| Sentinel/Guard | 1 | `.dat.ok` files |
| LRU Cache | 1 | `api/model_cache.py` |

---

## Cutover Sequence

```
Day 0   Phase 1 complete     → deploy, zero behaviour change
Day 1   DVC init             → dvc add data/*.dat; git commit
Day 2   Repo Pattern only    → result_backend="pkl", no change visible
Day 3   HDF5 backend         → dry-run migrate, then flip config
Day 4   Zarr archive         → additive write, .dat unchanged
Day 5   Verify Zarr output   → cmp new .dat vs old .dat (byte-identical)
Day 7   Ingest pipeline      → run alongside old parse_tfrecord.py
Day 9   Manifest + DVC tag   → dvc add data/archive.zarr; git tag v2.0-data
Day 10  Deprecate old parse  → parse_tfrecord.py becomes 3-line shim
```

---

## Rollback

| Phase | How to rollback |
|---|---|
| Phase 1 | Delete sentinel check, revert num_workers — trivial |
| Phase 2 repo | Set `result_backend="pkl"` in config — instant |
| Phase 2 HDF5 | PKL fallback in `_resolve()` keeps old files serving |
| Phase 2 Zarr | Zarr write is after memmap — memmap always succeeds first |
| Phase 3 | `parse_tfrecord.py` shim still works; revert to original |

---

## What Never Changes

- `dataset/fpc.py` — interface frozen, training unaffected
- `data/train.dat` format — Zarr generates identical bytes
- `data/train.npz` — manifest is additive
- Model architecture — zero coupling to data pipeline
- Existing `.pkl` files — HDF5 repo falls back to PKL
