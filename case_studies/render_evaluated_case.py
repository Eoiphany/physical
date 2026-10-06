"""Render raw versus protocol-evaluated maps for one completed case."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radiomap_physics import load_ground_truth_path_loss_db, load_json, load_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--tx-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_json(Path(args.config))
    scene = load_scene(args.data_root, args.scene_id, config)
    sample = Path(args.run_dir) / f"scene_{args.scene_id}_tx_{args.tx_id}" / "physics_arrays"
    raw = np.load(sample / "physics_prior_db.npy")
    gt = load_ground_truth_path_loss_db(args.data_root, args.scene_id, args.tx_id, config)
    minimum = float(config["dataset_labels"]["png_gray_mapping"]["pathgain_db_range"][0])
    evaluated = np.where(scene.height_map_m > 0.0, minimum, raw)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    panels = [
        (scene.height_map_m > 0.0, "Building mask", "gray"),
        (raw, "Raw FSPL + diffraction", "viridis"),
        (evaluated, f"Evaluated (building={minimum:g} dB)", "viridis"),
        (gt, "Ground truth", "viridis"),
    ]
    for ax, (value, title, cmap) in zip(axes, panels):
        image = ax.imshow(value, cmap=cmap, origin="upper", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{config.get('dataset', 'Dataset')} scene={args.scene_id} tx={args.tx_id}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
