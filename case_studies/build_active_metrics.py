"""Rebuild the active cross-dataset summary without Boston or UCLA.

The script reads raw arrays from completed runs, applies the protocol-level
building-interior minimum only to the derived evaluation map, and emits a
small CSV/JSON/figure summary.  It deliberately does not modify historical
experiment directories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radiomap_physics import load_ground_truth_path_loss_db, load_json, load_scene


def _metric_row(dataset: str, config_path: Path, data_root: Path, run_dir: Path, scene_id: str, tx_id: str) -> dict:
    config = load_json(config_path)
    scene = load_scene(data_root, scene_id, config)
    sample = run_dir / f"scene_{scene_id}_tx_{tx_id}"
    arrays = sample / "physics_arrays"
    raw = np.load(arrays / "physics_prior_db.npy")
    los = np.load(arrays / "los_mask.npy").astype(bool)
    gt = load_ground_truth_path_loss_db(data_root, scene_id, tx_id, config)
    building_min_db = float(config["dataset_labels"]["png_gray_mapping"]["pathgain_db_range"][0])
    evaluated = np.where(scene.height_map_m > 0.0, building_min_db, raw).astype(np.float64)
    error = evaluated - gt.astype(np.float64)
    nlos = ~los
    resolved = np.load(arrays / "resolved_method_map.npy")
    cd_valid = np.load(arrays / "cd_validity_mask.npy").astype(bool)
    fallback = np.load(arrays / "fallback_to_rd_mask.npy").astype(bool)
    unresolved = np.load(arrays / "unresolved_nlos_mask.npy").astype(bool)
    return {
        "dataset": dataset,
        "status": "metrics",
        "config": str(config_path.relative_to(Path(__file__).resolve().parents[1]).as_posix()),
        "scene_id": scene_id,
        "tx_id": tx_id,
        "total_pixels": int(raw.size),
        "nlos_ratio": float(np.mean(nlos)),
        "los_ratio": float(np.mean(los)),
        "los_fspl_pixels": int(np.sum(resolved == "los-fspl")),
        "owr_cd_pixels": int(np.sum(resolved == "owr-cd")),
        "owr_rd_pixels": int(np.sum(resolved == "owr-rd")),
        "cd_valid_nlos_ratio": float(np.mean(cd_valid & nlos)),
        "fallback_to_rd_ratio": float(np.mean(fallback)),
        "unresolved_nlos_ratio": float(np.mean(unresolved)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "los_rmse_db": float(np.sqrt(np.mean(error[los] ** 2))) if np.any(los) else float("nan"),
        "nlos_rmse_db": float(np.sqrt(np.mean(error[nlos] ** 2))) if np.any(nlos) else float("nan"),
        "prediction_min_db": float(np.min(evaluated)),
        "raw_prediction_min_db": float(np.min(raw)),
        "below_gt_min_ratio": float(np.mean(evaluated < float(np.min(gt)))),
        "building_interior_min_db": building_min_db,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiomapseer-root", required=True)
    parser.add_argument("--urbanradio3d-root", required=True)
    parser.add_argument("--usc-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    rows = [
        _metric_row(
            "RadioMapSeer",
            repo / "configs" / "radiomapseer.json",
            Path(args.radiomapseer_root),
            repo / "runs" / "cross_dataset_physics_prior" / "radiomapseer" / "los_fspl_auto_nlos_20261006",
            "0",
            "0",
        ),
        _metric_row(
            "UrbanRadio3D",
            repo / "configs" / "rm_data.json",
            Path(args.urbanradio3d_root),
            repo / "runs" / "cross_dataset_physics_prior" / "rm_data" / "los_fspl_auto_nlos_20261006_evaluated",
            "0",
            "0",
        ),
        _metric_row(
            "USC",
            repo / "configs" / "usc.json",
            Path(args.usc_root),
            repo / "runs" / "cross_dataset_physics_prior" / "usc" / "los_fspl_auto_nlos_20261006",
            "1",
            "0",
        ),
    ]
    pp_summary = load_json(
        repo / "runs" / "cross_dataset_physics_prior" / "ppdata5d" / "T06C0D0001_n02_f00_z00_proxy" / "case_summary.json"
    )
    rows.append(
        {
            "dataset": "PPData5D",
            "status": pp_summary["metric_status"],
            "sample": pp_summary["sample"],
            "metric_limitation": pp_summary["metric_limitation"],
            "los_ratio": pp_summary["los_ratio"],
            "fallback_to_rd_ratio": pp_summary["fallback_to_rd_ratio"],
            "resolved_method_counts": pp_summary["auto_method_counts"],
        }
    )
    payload = {
        "experiment": "active_los_fspl_auto_nlos_20261006",
        "excluded_datasets": ["Boston", "UCLA"],
        "resolved_logic": "LOS -> FSPL; NLOS -> per-Rx OWR-CD or OWR-RD by relative height; strict CD fallback is diagnosed explicitly.",
        "building_interior_evaluation_rule": "height_map_m > 0 pixels are replaced by the config minimum signed pathgain only in derived evaluation maps; raw solver arrays are preserved.",
        "datasets": rows,
    }
    (output / "active_metrics_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_rows = [row for row in rows if row.get("status") == "metrics"]
    fields = ["dataset", "scene_id", "tx_id", "nlos_ratio", "los_ratio", "rmse_db", "mae_db", "bias_db", "los_rmse_db", "nlos_rmse_db", "prediction_min_db", "raw_prediction_min_db", "below_gt_min_ratio", "cd_valid_nlos_ratio", "fallback_to_rd_ratio", "unresolved_nlos_ratio"]
    with (output / "active_metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in metric_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    names = [row["dataset"] for row in metric_rows]
    x = np.arange(len(names))
    for offset, key in enumerate(("rmse_db", "mae_db")):
        axes[0].bar(x + (offset - 0.5) * 0.32, [row[key] for row in metric_rows], width=0.32, label=key)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("dB")
    axes[0].set_title("Active evaluated-map errors")
    axes[0].legend()
    axes[1].bar(x - 0.2, [row["los_rmse_db"] for row in metric_rows], width=0.4, label="LOS RMSE")
    axes[1].bar(x + 0.2, [row["nlos_rmse_db"] for row in metric_rows], width=0.4, label="NLOS RMSE")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("dB")
    axes[1].set_title("LOS / NLOS RMSE")
    axes[1].legend()
    fig.suptitle("Active deterministic Physics Prior case studies")
    fig.savefig(output / "active_metrics_overview.png", dpi=220)
    fig.savefig(output / "active_metrics_overview.pdf")
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
