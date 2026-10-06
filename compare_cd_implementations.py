"""Compare the PDF-compatible OWR-CD chain with local-segment UTD OWR-CD.

The legacy side is loaded from an existing ``run_physics_prior.py`` output so
the comparison reproduces the user's reference PDF exactly.  The corrected
side reuses the same one-way geometry search but evaluates each corner only
after the complete chain is known, using adjacent propagation segments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radiomap_physics import (
    _associate_footprints_with_components,
    _building_component_labels,
    _extract_footprint_polygons,
    _normalize_footprint_polygons,
    free_space_path_loss_db,
    load_ground_truth_path_gain_db,
    load_json,
    load_scene,
    solve_owr_cd_local_segment_diffraction,
)


def _metric(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, method: str, scope: str) -> dict:
    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        return {"method": method, "scope": scope, "pixel_count": 0}
    error = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    return {
        "method": method,
        "scope": scope,
        "pixel_count": int(valid.sum()),
        "rmse_db": float(np.sqrt(np.mean(error * error))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "prediction_min_db": float(np.min(prediction[valid])),
        "prediction_below_gt_min_ratio": float(np.mean(prediction[valid] < np.min(target[valid]))),
    }


def _plot(
    output: Path,
    height_map: np.ndarray,
    gt: np.ndarray,
    legacy_prior: np.ndarray,
    local_prior: np.ndarray,
    legacy_diff: np.ndarray,
    local_diff: np.ndarray,
    corner_count: np.ndarray,
    unresolved: np.ndarray,
    tx_row: float,
    tx_col: float,
    config: dict,
) -> None:
    limits = config["visualization"]["fixed_color_limits"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    panels = [
        (height_map, "Building height (m)", "viridis", limits["building_height_m"]),
        (gt, "Ground truth / signed PL (dB)", "viridis", limits["ground_truth_path_loss_db"]),
        (legacy_prior, "PDF-compatible auto/CD", "viridis", limits["physics_prior_db"]),
        (local_prior, "Local-segment CD", "viridis", limits["physics_prior_db"]),
        (legacy_diff, "PDF-compatible signed diffraction", "viridis", limits["diffraction_loss_db"]),
        (local_diff, "Local-segment signed diffraction", "viridis", limits["diffraction_loss_db"]),
        (local_prior - legacy_prior, "Local - PDF prior (dB)", "coolwarm", (-100.0, 100.0)),
        (corner_count, "Local CD corner count / unresolved", "magma", (0.0, max(3.0, float(np.nanmax(corner_count)) + 1.0)),),
    ]
    for ax, (array, title, cmap, clim) in zip(axes.flat, panels):
        image = ax.imshow(np.ma.masked_invalid(array), cmap=cmap, vmin=clim[0], vmax=clim[1], interpolation="nearest")
        ax.plot(tx_col, tx_row, marker="+", color="red", markersize=8, markeredgewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("RadioMapSeer: PDF-compatible OWR-CD vs local-segment OWR-CD", fontsize=14)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-id", default="0")
    parser.add_argument("--tx-id", default="0")
    parser.add_argument("--legacy-scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-corner-depth", type=int, default=None)
    args = parser.parse_args()

    config = load_json(Path(args.config))
    scene = load_scene(args.data_root, args.scene_id, config)
    tx = scene.tx_records[int(args.tx_id)]
    frequency_hz = float(config["wireless"]["frequency_hz"])
    resolution_m = float(config["scene"]["resolution_m_per_pixel"])
    wavelength_m = float(config["physics_prior"]["speed_of_light_m_per_s"]) / frequency_hz
    floor_db = float(config["dataset_labels"]["png_gray_mapping"]["pathgain_db_range"][0])
    fspl = free_space_path_loss_db(
        scene.height_map_m,
        tx,
        frequency_hz,
        float(config["scene"]["rx_height_m"]),
        resolution_m,
        float(config["physics_prior"].get("speed_of_light_m_per_s", 299792458.0)),
    )
    gt = load_ground_truth_path_gain_db(args.data_root, args.scene_id, args.tx_id, config).astype(np.float32)

    legacy_dir = Path(args.legacy_scene_dir)
    legacy_arrays = legacy_dir / "physics_arrays"
    legacy_prior = np.load(legacy_arrays / "physics_prior_db.npy").astype(np.float32)
    legacy_diff = np.load(legacy_arrays / "diffraction_loss_db.npy").astype(np.float32)
    legacy_los = np.load(legacy_arrays / "los_mask.npy").astype(bool)
    building_mask = scene.height_map_m > 0.0
    legacy_prior_eval = np.where(building_mask, floor_db, legacy_prior).astype(np.float32)

    labels = _building_component_labels(scene.height_map_m)
    polygons = _normalize_footprint_polygons(scene.polygons)
    if not polygons:
        polygons = _extract_footprint_polygons(scene.height_map_m)
    polygon_map = _associate_footprints_with_components(polygons, labels)

    height_px, width_px = scene.height_map_m.shape
    local_diff = np.full((height_px, width_px), np.nan, dtype=np.float32)
    local_corner_count = np.full((height_px, width_px), np.nan, dtype=np.float32)
    local_los = np.zeros((height_px, width_px), dtype=bool)
    unresolved = np.zeros((height_px, width_px), dtype=bool)
    rx_height_m = float(config["scene"]["rx_height_m"])
    for row in range(height_px):
        if row % 16 == 0:
            print(f"[local-cd] row={row}/{height_px}", flush=True)
        for col in range(width_px):
            solution = solve_owr_cd_local_segment_diffraction(
                scene.height_map_m,
                tx,
                row,
                col,
                rx_height_m,
                resolution_m,
                wavelength_m,
                polygon_map,
                labels,
                args.max_corner_depth,
            )
            if solution.is_los:
                local_los[row, col] = True
                local_diff[row, col] = 0.0
                local_corner_count[row, col] = 0.0
            elif solution.termination == "rx_visible" and solution.events:
                local_diff[row, col] = -float(solution.loss_db)
                local_corner_count[row, col] = float(len(solution.events))
            else:
                unresolved[row, col] = True
                local_corner_count[row, col] = float(len(solution.events))

    local_raw = (fspl + np.nan_to_num(local_diff, nan=0.0)).astype(np.float32)
    local_prior_eval = np.where(building_mask, floor_db, local_raw).astype(np.float32)
    local_prior_eval[unresolved & ~building_mask] = np.nan

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "legacy_pdf_prior_db.npy", legacy_prior_eval)
    np.save(output / "legacy_pdf_diffraction_loss_db.npy", legacy_diff)
    np.save(output / "local_segment_prior_db.npy", local_prior_eval)
    np.save(output / "local_segment_diffraction_loss_db.npy", local_diff)
    np.save(output / "local_segment_corner_count.npy", local_corner_count)
    np.save(output / "local_segment_unresolved_mask.npy", unresolved)
    np.save(output / "los_mask.npy", legacy_los)
    np.save(output / "building_mask.npy", building_mask)
    np.save(output / "ground_truth_pathgain_db.npy", gt)

    scopes = {
        "all": np.ones(gt.shape, dtype=bool),
        "nlos": ~legacy_los,
        "open_nlos": (~legacy_los) & (~building_mask),
        "los_open": legacy_los & (~building_mask),
    }
    rows = []
    for method, prediction in (("pdf-compatible-auto", legacy_prior_eval), ("local-segment-cd", local_prior_eval)):
        for scope, mask in scopes.items():
            rows.append(_metric(prediction, gt, mask, method, scope))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "reference_method": "PDF-compatible auto/CD implementation",
        "corrected_method": "geometry-first OWR-CD with local adjacent-segment UTD",
        "legacy_scene_dir": str(legacy_dir),
        "resolved_local_los_count": int(local_los.sum()),
        "resolved_local_corner_count": int(np.isfinite(local_diff).sum() - local_los.sum()),
        "unresolved_local_nlos_count": int(unresolved.sum()),
        "local_max_corner_count": int(np.nanmax(local_corner_count)),
        "local_min_diffraction_db": float(np.nanmin(local_diff)),
        "building_floor_db": floor_db,
        "formula": "prior = FSPL + signed diffraction; local UTD uses Tx-C1-C2, ..., Cn-Rx",
        "metrics": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(
        output / "cd_legacy_vs_local_segment",
        scene.height_map_m,
        gt,
        legacy_prior_eval,
        local_prior_eval,
        legacy_diff,
        local_diff,
        local_corner_count,
        unresolved.astype(np.float32),
        scene.height_map_m.shape[0] - 1 - tx.y_m,
        tx.x_m,
        config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
