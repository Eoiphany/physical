"""Inspect a small set of OWR-CD Rx paths without running a full map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radiomap_physics import (
    DiffractionEndpoint,
    _building_component_labels,
    _endpoint_from_tx,
    _first_blocking_component,
    load_json,
    load_scene,
    solve_diffraction,
)
from run_physics_prior import _serialize_solution, render_corner_recursive_debug


def _parse_pairs(value: str) -> list[tuple[int, int]]:
    pairs = []
    for item in value.split(";"):
        row, col = (int(part.strip()) for part in item.split(","))
        pairs.append((row, col))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect deterministic OWR-CD paths for selected Rx points.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-id", default="0")
    parser.add_argument("--tx-id", default="0")
    parser.add_argument("--rx-pairs", default="0,107;32,107;128,32")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_json(Path(args.config))
    scene = load_scene(args.data_root, args.scene_id, config)
    tx = scene.tx_records[int(args.tx_id)]
    rx_height_m = float(config["scene"]["rx_height_m"])
    resolution_m = float(config["scene"]["resolution_m_per_pixel"])
    wavelength_m = float(config["physics_prior"]["speed_of_light_m_per_s"]) / float(config["wireless"]["frequency_hz"])
    max_depth = int(config["physics_prior"].get("max_corner_depth", 3))
    labels = _building_component_labels(scene.height_map_m)
    tx_endpoint = _endpoint_from_tx(scene.height_map_m, tx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rx_row, rx_col in _parse_pairs(args.rx_pairs):
        target = DiffractionEndpoint(float(rx_row), float(rx_col), rx_height_m)
        first_component, first_progress = _first_blocking_component(labels, tx_endpoint, target, resolution_m, frozenset())
        solution = solve_diffraction(
            scene.height_map_m,
            tx,
            rx_row,
            rx_col,
            "owr-cd",
            rx_height_m,
            resolution_m,
            wavelength_m,
            footprint_polygons=scene.polygons,
            max_corner_depth=max_depth,
        )
        sample_stem = output_dir / f"rx_{rx_row}_{rx_col}" / "owr_cd_path"
        render_corner_recursive_debug(
            scene.height_map_m,
            tx,
            rx_row,
            rx_col,
            solution,
            scene.polygons,
            sample_stem,
            str(config.get("dataset", "Dataset")),
        )
        record = {
            "scene_id": args.scene_id,
            "tx_id": args.tx_id,
            "rx_row_col": [rx_row, rx_col],
            "first_blocking_building_id": first_component,
            "first_blocking_progress": first_progress,
            "max_corner_depth": max_depth,
            "solution": _serialize_solution(solution),
            "figure_png": str(sample_stem.with_suffix(".png")),
            "figure_pdf": str(sample_stem.with_suffix(".pdf")),
        }
        (sample_stem.parent / "path_debug.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        print(json.dumps({"rx": [rx_row, rx_col], "first_blocking_building": first_component, "termination": solution.termination, "corners": [[event.edge.row, event.edge.col, event.component_id] for event in solution.events]}, ensure_ascii=False), flush=True)
    (output_dir / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
