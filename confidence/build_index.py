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

from confidence.index import NearestNeighborIndex
from model.embedding import extract_embedding


def main():
    parser = argparse.ArgumentParser(description="Build confidence embedding index")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split",      type=str, default="train")
    parser.add_argument("--output",     type=str, default="runs/embedding_index.pkl")
    parser.add_argument("--domain",     type=str, default="cylinder_flow",
                        choices=["cylinder_flow", "flag_simple"])
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--device",     type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
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
        simulator = Simulator(
            message_passing_num=15,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            device=device,
        )

    simulator.load_state_dict(ckpt["model_state_dict"])
    simulator.eval()
    print("Loaded checkpoint from %s (epoch %d)" % (args.checkpoint, ckpt.get("epoch", 0)))

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

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Dataset: %d samples. Extracting embeddings..." % len(dataset))

    embeddings = []
    for graph in tqdm(loader, desc="Extracting embeddings"):
        graph = graph[0] if isinstance(graph, list) else graph  # unbatch single item
        if transformer is not None:
            graph = transformer(graph)
        emb = extract_embedding(simulator, graph, device=device)
        embeddings.append(emb)

    embeddings = np.stack(embeddings)   # [N_train, 128]
    print("Embeddings shape: %s" % str(embeddings.shape))

    index = NearestNeighborIndex()
    index.build(embeddings)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    index.save(args.output)
    print("Index saved to %s (backend: %s, diameter: %.4f, N=%d)" % (
        args.output, index.backend, index.train_diameter, len(embeddings)))


if __name__ == "__main__":
    main()
