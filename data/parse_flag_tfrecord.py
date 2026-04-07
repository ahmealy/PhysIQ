# -*- encoding: utf-8 -*-
"""
Parse flag_simple TFRecords into numpy arrays.

Usage:
    python data/parse_flag_tfrecord.py

Requires: tensorflow<1.15, data_flag/{train,valid,test}.tfrecord

Outputs per split:
    data_flag/{split}_pos.npz   — world_pos [T, N, 3] per trajectory, stacked
    data_flag/{split}_mesh.npz  — mesh_pos [N, 2], node_type [N, 1], cells [F, 3]
                                   (per trajectory, stored as ragged lists)
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
    with open(os.path.join(data_dir, "meta.json")) as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(data_dir, split + ".tfrecord"))
    ds = ds.map(functools.partial(_tf_parse, meta=meta), num_parallel_calls=1)
    ds = ds.prefetch(1)
    return ds


def parse_split(split: str, data_dir: str = DATA_DIR):
    """Parse one split and write output files. Idempotent (skips if already exists)."""
    pos_path  = os.path.join(data_dir, f"{split}_pos.npz")
    mesh_path = os.path.join(data_dir, f"{split}_mesh.npz")

    if os.path.exists(pos_path) and os.path.exists(mesh_path):
        print(f"[{split}] Output files already exist, skipping.")
        return

    print(f"[{split}] Parsing...")
    ds = load_dataset(split, data_dir=data_dir)

    all_world_pos  = []  # list of [T, N, 3] per trajectory
    all_mesh_pos   = []  # list of [N, 2] per trajectory
    all_node_type  = []  # list of [N, 1] per trajectory
    all_cells      = []  # list of [F, 3] per trajectory

    for idx, d in enumerate(ds):
        world_pos  = d["world_pos"].numpy()    # [T, N, 3]
        mesh_pos   = d["mesh_pos"].numpy()[0]  # [N, 2] — static, use step 0
        node_type  = d["node_type"].numpy()[0] # [N, 1] — static, use step 0
        cells      = d["cells"].numpy()[0]     # [F, 3] — static, use step 0

        all_world_pos.append(world_pos)
        all_mesh_pos.append(mesh_pos)
        all_node_type.append(node_type)
        all_cells.append(cells)

        if idx % 10 == 0:
            print(f"  [{split}] Parsed {idx} trajectories...")

    print(f"[{split}] {len(all_world_pos)} trajectories parsed.")

    # Save world_pos as ragged (different N per trajectory possible)
    # Use object dtype arrays for ragged storage
    np.savez_compressed(
        pos_path,
        world_pos=np.array(all_world_pos, dtype=object),  # [n_traj] of [T, N, 3]
    )
    np.savez_compressed(
        mesh_path,
        mesh_pos=np.array(all_mesh_pos, dtype=object),    # [n_traj] of [N, 2]
        node_type=np.array(all_node_type, dtype=object),  # [n_traj] of [N, 1]
        cells=np.array(all_cells, dtype=object),          # [n_traj] of [F, 3]
    )
    print(f"[{split}] Saved to {pos_path} and {mesh_path}")


if __name__ == "__main__":
    import tensorflow as tf
    from packaging import version

    if version.parse(tf.__version__) >= version.parse("1.15"):
        raise RuntimeError(
            f"TensorFlow {tf.__version__} found. This script requires tensorflow<1.15.\n"
            "Install in a separate env: pip install 'tensorflow<1.15'"
        )

    os.makedirs(DATA_DIR, exist_ok=True)

    tf.enable_resource_variables()   # type: ignore
    tf.enable_eager_execution()      # type: ignore

    for split in ["train", "valid", "test"]:
        parse_split(split)

    print("Done. Re-run train.py with domain=flag_simple to use the parsed data.")
