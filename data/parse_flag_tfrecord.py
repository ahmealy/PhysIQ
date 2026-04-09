# -*- encoding: utf-8 -*-
"""
Parse flag_simple TFRecords into numpy arrays.

Usage:
    python data/parse_flag_tfrecord.py

Requires: tensorflow<1.15, data_flag/{train,valid,test}.tfrecord

Outputs — one .npz per trajectory, streamed to disk:
    data_flag/{split}/traj_{i:05d}.npz  contains:
        world_pos  [T, N, 3]  float32
        mesh_pos   [N, 2]     float32
        node_type  [N, 1]     int32
        cells      [F, 3]     int32

Index files (tiny — one int per trajectory):
    data_flag/{split}_index.npz  — n_traj, steps_per_traj array
"""
import functools
import json
import os

import numpy as np

DATA_DIR = "data_flag"


def _tf_parse(proto, meta):
    """Parses a trajectory from tf.Example — same pattern as parse_tfrecord.py."""
    import tensorflow as tf
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format: %s" % field["type"])
        out[key] = data
    return out


def load_dataset(split, data_dir=DATA_DIR):
    import tensorflow as tf
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"meta.json not found at {meta_path}\n"
            "Download flag_simple dataset first:\n"
            "  bash meshgraphnets/download_dataset.sh flag_simple data_flag"
        )
    with open(meta_path) as fp:
        meta = json.load(fp)

    # Validate required fields are present in this meta.json
    required = {"world_pos", "mesh_pos", "node_type", "cells"}
    if "features" in meta:
        present = set(meta["features"].keys())
    else:
        present = set(meta.get("field_names", []))
    missing = required - present
    if missing:
        raise ValueError(
            f"meta.json at {meta_path} is missing flag_simple fields: {missing}\n"
            "Ensure you are using the flag_simple dataset's meta.json, not another domain's."
        )

    ds = tf.data.TFRecordDataset(os.path.join(data_dir, split + ".tfrecord"))
    ds = ds.map(functools.partial(_tf_parse, meta=meta), num_parallel_calls=1)
    ds = ds.prefetch(1)
    return ds


def parse_split(split: str, data_dir: str = DATA_DIR):
    """Parse one split, writing one .npz per trajectory. Idempotent via index file."""
    split_dir  = os.path.join(data_dir, split)
    index_path = os.path.join(data_dir, f"{split}_index.npz")

    if os.path.exists(index_path):
        print(f"[{split}] Index already exists, skipping.")
        return

    os.makedirs(split_dir, exist_ok=True)
    print(f"[{split}] Parsing — streaming one .npz per trajectory to {split_dir}/")

    ds = load_dataset(split, data_dir=data_dir)
    steps_per_traj = []

    for idx, d in enumerate(ds):
        world_pos = d["world_pos"].numpy()    # [T, N, 3]
        mesh_pos  = d["mesh_pos"].numpy()[0]  # [N, 2] — static, step 0
        node_type = d["node_type"].numpy()[0] # [N, 1] — static, step 0
        cells     = d["cells"].numpy()[0]     # [F, 3] — static, step 0

        traj_path = os.path.join(split_dir, f"traj_{idx:05d}.npz")
        # np.savez_compressed appends .npz automatically, so use a tmp stem without .npz
        traj_tmp_stem = os.path.join(split_dir, f"traj_{idx:05d}.tmp")
        np.savez_compressed(
            traj_tmp_stem,
            world_pos=world_pos.astype(np.float32),
            mesh_pos=mesh_pos.astype(np.float32),
            node_type=node_type.astype(np.int32),
            cells=cells.astype(np.int32),
        )
        # np.savez_compressed writes to traj_tmp_stem + ".npz"
        os.replace(traj_tmp_stem + ".npz", traj_path)

        steps_per_traj.append(world_pos.shape[0])  # T

        if idx % 10 == 0:
            print(f"  [{split}] {idx} trajectories written...")

    n_traj = len(steps_per_traj)
    print(f"[{split}] {n_traj} trajectories parsed.")

    # Write index file — tiny, marks split as complete
    index_tmp_stem = index_path[:-4] + ".tmp"  # strip .npz, add .tmp stem
    np.savez_compressed(
        index_tmp_stem,
        n_traj=np.array(n_traj, dtype=np.int64),
        steps_per_traj=np.array(steps_per_traj, dtype=np.int32),
    )
    os.replace(index_tmp_stem + ".npz", index_path)
    print(f"[{split}] Index written to {index_path}")


if __name__ == "__main__":
    import tensorflow as tf
    from packaging import version

    if version.parse(tf.__version__) >= version.parse("1.15"):
        raise RuntimeError(
            f"TensorFlow {tf.__version__} found. This script requires tensorflow 1.x (< 1.15).\n"
            "The enable_eager_execution() API is only available in TF 1.x.\n"
            "Install in a separate env: pip install 'tensorflow==1.14.0'"
        )

    os.makedirs(DATA_DIR, exist_ok=True)

    tf.enable_resource_variables()   # type: ignore
    tf.enable_eager_execution()      # type: ignore

    for split in ["train", "valid", "test"]:
        parse_split(split)

    print("Done. Re-run train.py with domain=flag_simple to use the parsed data.")
