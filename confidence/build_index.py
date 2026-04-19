"""
Standalone script to build the confidence index from a saved checkpoint.

Usage:
    python -m confidence.build_index \
        --checkpoint checkpoints/best_model.pth \
        --split train \
        --output runs/embedding_index.pkl \
        --domain cylinder_flow \
        --data_dir data

The index is saved to runs/embedding_index.pkl (or --output path).
A SHA-256 hash of the checkpoint file is stored inside the index so
that stale-index detection works at query time.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confidence.index import NearestNeighborIndex, checkpoint_hash
from model.embedding import extract_embedding, CFD_WARMUP_FRAMES


def main():
    parser = argparse.ArgumentParser(description="Build confidence embedding index")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split",      type=str, default="train")
    parser.add_argument("--output",     type=str, default="runs/embedding_index.pkl")
    parser.add_argument("--domain",     type=str, default="cylinder_flow",
                        choices=["cylinder_flow", "flag_simple"])
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--device",      type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Randomly subsample this many frames from the dataset. "
                             "5000 is more than enough for a KD-tree OOD index. "
                             "Pass -1 to use all samples (may take hours).")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Load checkpoint and simulator
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    domain = ckpt.get("domain", args.domain)

    if domain == "flag_simple":
        from model.flag_simulator import FlagSimulator
        simulator = FlagSimulator(message_passing_num=15, device=device)
    else:
        from model.simulator import Simulator
        node_input_size = ckpt.get("node_input_size", 11)
        edge_input_size = ckpt.get("edge_input_size", 3)
        architecture    = ckpt.get("architecture", "gn")
        simulator = Simulator(
            message_passing_num=15,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            architecture=architecture,
            device=device,
        )

    # Load only the keys that match the current model exactly (shape + name).
    # This handles checkpoints saved with an older/deeper architecture — e.g. a
    # 4-layer MLP decoder vs the current 3-layer decoder.  The decoder is NOT
    # used for embeddings (only the encoder and processor are needed), so it is
    # safe to skip any keys that don't fit.
    ckpt_sd      = ckpt["model_state_dict"]
    current_sd   = simulator.state_dict()
    filtered_sd  = {
        k: v for k, v in ckpt_sd.items()
        if k in current_sd and current_sd[k].shape == v.shape
    }
    skipped = [k for k in ckpt_sd if k not in filtered_sd]
    if skipped:
        print("Warning: skipping %d checkpoint keys with shape/name mismatch "
              "(decoder not needed for embeddings): %s%s" % (
                  len(skipped), skipped[:3], " ..." if len(skipped) > 3 else ""))
    simulator.load_state_dict(filtered_sd, strict=False)
    simulator.eval()
    print("Loaded checkpoint from %s (epoch %d)" % (args.checkpoint, ckpt.get("epoch", 0)))

    # Compute checkpoint hash for stale-index detection at query time
    ckpt_hash = checkpoint_hash(args.checkpoint)
    print("Checkpoint hash: %s" % ckpt_hash)

    # Load dataset
    if domain == "flag_simple":
        from dataset.flag_dataset import FlagDataset
        dataset = FlagDataset(args.data_dir, split=args.split)
        transformer = None
    else:
        from dataset import FpcDataset
        dataset = FpcDataset(data_root=args.data_dir, split=args.split)
        transformer = T.Compose([
            T.FaceToEdge(),
            T.Cartesian(norm=False),
            T.Distance(norm=False),
        ])

    n_total = len(dataset)

    # For CFD: one dual-frame embedding per trajectory.
    # concat(encoder(frame_0), encoder(frame_warmup)) captures both geometry+BCs
    # and early flow dynamics, making embeddings discriminative across inlet
    # conditions as well as cylinder geometry.
    if domain != "flag_simple" and hasattr(dataset, "num_sampes_per_tra"):
        steps = dataset.num_sampes_per_tra
        n_traj = n_total // steps
        warmup = min(CFD_WARMUP_FRAMES, steps - 1)
        traj_list = list(range(n_traj))
        # Subsample trajectories if needed
        if args.max_samples > 0 and len(traj_list) > args.max_samples:
            rng      = np.random.default_rng(seed=42)
            traj_list = sorted(rng.choice(len(traj_list), size=args.max_samples, replace=False).tolist())
            print("CFD: subsampling to %d trajectories (dual-frame: 0 + %d)." % (len(traj_list), warmup))
        else:
            print("CFD: %d trajectories × %d steps → dual-frame embedding "
                  "(frame 0 + frame %d) per trajectory."
                  % (n_traj, steps, warmup))
        frame0_indices = [t * steps          for t in traj_list]
        framew_indices = [t * steps + warmup for t in traj_list]
        is_cfd_dual = True
    else:
        # Cloth or unknown: single-frame, subsample flat frames as before
        is_cfd_dual = False
        if args.max_samples > 0 and n_total > args.max_samples:
            rng     = np.random.default_rng(seed=42)
            indices = sorted(rng.choice(n_total, size=args.max_samples, replace=False).tolist())
            print("Dataset: %d samples → subsampling %d for index build." % (n_total, args.max_samples))
        else:
            indices = list(range(n_total))
            print("Dataset: %d samples. Extracting embeddings..." % n_total)

    embeddings = []
    if is_cfd_dual:
        for i0, iw in tqdm(zip(frame0_indices, framew_indices),
                           total=len(frame0_indices), desc="Extracting embeddings"):
            g0 = dataset[i0]
            gw = dataset[iw]
            if transformer is not None:
                g0 = transformer(g0)
                gw = transformer(gw)
            emb = extract_embedding(simulator, g0, device=device, graph_warmup=gw)
            embeddings.append(emb)
    else:
        for i in tqdm(indices, desc="Extracting embeddings"):
            graph = dataset[i]
            if transformer is not None:
                graph = transformer(graph)
            emb = extract_embedding(simulator, graph, device=device)
            embeddings.append(emb)

    embeddings = np.stack(embeddings)   # [N_train, 128]
    print("Embeddings shape: %s" % str(embeddings.shape))

    index = NearestNeighborIndex()
    index.build(embeddings)
    index.checkpoint_hash = ckpt_hash   # store for stale-index detection
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    index.save(args.output)
    print("Index saved to %s (backend: %s, diameter: %.4f, N=%d, ckpt_hash: %s)" % (
        args.output, index.backend, index.train_diameter, len(embeddings), ckpt_hash))


if __name__ == "__main__":
    main()
