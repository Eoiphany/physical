"""Run an auditable geometry-only case study for the downloaded PPData5D data.

The repository's signed-dB metrics require a verified transmitter geometry and
RSS-to-dB calibration.  The checked sample NPZ files currently expose only a
building occupancy proxy (``inBldg_zyx[0]``) and terrain elevation, while the
PNG is documented as RSS and the Tx coordinate/absolute height mapping is not
encoded in the inspected files.  This script therefore runs the deterministic
geometry prior with explicit proxies and refuses to report a misleading dB
RMSE against the raw RSS grayscale image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radiomap_physics import TxRecord, compute_physics_maps, free_space_path_loss_db


SCENE_FOLDERS = {
    "T01": "01.Grassland",
    "T02": "02.Island",
    "T03": "03.Ocean",
    "T04": "04.Lake",
    "T05": "05.Suburban",
    "T06": "06.DenseUrban",
    "T07": "07.Rural",
    "T08": "08.OrdinaryUrban",
    "T09": "09.Desert",
    "T10": "10.Mountainous",
    "T11": "11.Forest",
}


def _parse_sample(sample: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"(?P<base>T\d+C\d+D\d{4}_n\d{2})_f(?P<f>\d{2})_z(?P<z>\d{2})", sample)
    if match is None:
        raise ValueError("sample must look like T06C0D0001_n02_f00_z00")
    return match.group("base"), match.group("f"), match.group("z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="External PPData5D directory")
    parser.add_argument("--sample", default="T06C0D0001_n02_f00_z00")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frequency-hz", type=float, default=150_000_000.0)
    parser.add_argument("--resolution-m", type=float, default=10.0)
    parser.add_argument("--building-height-proxy-m", type=float, default=20.0)
    parser.add_argument("--tx-height-proxy-m", type=float, default=1.5)
    parser.add_argument("--rx-height-proxy-m", type=float, default=1.5)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.data_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_id, frequency_id, rx_height_id = _parse_sample(args.sample)
    scene_folder = SCENE_FOLDERS[args.sample[:3]]
    npz_path = root / "npz" / f"{base_id}_bdtr.npz"
    png_path = root / "png" / scene_folder / f"{base_id}_f{frequency_id}_ss_z{rx_height_id}.png"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not png_path.exists():
        raise FileNotFoundError(png_path)

    arrays = np.load(npz_path, allow_pickle=False)
    if "inBldg_zyx" not in arrays or "terrain_yx" not in arrays:
        raise KeyError("PPData5D sample must contain inBldg_zyx and terrain_yx")
    building_stack = np.asarray(arrays["inBldg_zyx"])
    terrain = np.asarray(arrays["terrain_yx"], dtype=np.float64)
    if building_stack.ndim != 3 or building_stack.shape[1:] != terrain.shape:
        raise ValueError(f"unexpected PPData5D shapes: building={building_stack.shape}, terrain={terrain.shape}")
    building_mask = building_stack[0] > 0
    rss_gray = np.asarray(Image.open(png_path).convert("L"), dtype=np.uint8)
    if rss_gray.shape != terrain.shape:
        raise ValueError(f"PNG shape {rss_gray.shape} does not match NPZ {terrain.shape}")

    # This is intentionally a proxy: the checked NPZ has no Tx coordinate.
    tx_row, tx_col = np.unravel_index(int(np.argmax(rss_gray)), rss_gray.shape)
    terrain_at_tx = float(terrain[tx_row, tx_col])
    height_map = np.where(building_mask, terrain + float(args.building_height_proxy_m), 0.0).astype(np.float32)
    tx = TxRecord(
        x_m=float(tx_col * args.resolution_m),
        y_m=float((terrain.shape[0] - 1 - tx_row) * args.resolution_m),
        z_m=terrain_at_tx + float(args.tx_height_proxy_m),
        rooftop_height_m=None,
    )
    rx_height_abs = terrain_at_tx + float(args.rx_height_proxy_m)
    c_mps = 299_792_458.0
    fspl = free_space_path_loss_db(height_map, tx, args.frequency_hz, rx_height_abs, args.resolution_m, c_mps)
    maps = compute_physics_maps(
        height_map,
        tx,
        args.frequency_hz,
        rx_height_abs,
        args.resolution_m,
        10.0,
        c_mps,
        diffraction_mode="auto",
        fspl_db=fspl,
        footprint_polygons=[],
        max_corner_depth=None,
    )

    shutil.copy2(png_path, output / "ground_truth_rss_gray.png")
    shutil.copy2(npz_path, output / "environment_bdtr.npz")
    array_dir = output / "physics_arrays"
    array_dir.mkdir(exist_ok=True)
    for name, value in {
        "terrain_yx": terrain,
        "building_mask": building_mask,
        "fspl_db": maps.fspl_db,
        "diffraction_loss_db": maps.diffraction_loss_db,
        "physics_prior_db": maps.physics_prior_db,
        "los_mask": maps.los_mask,
        "resolved_method_map": maps.resolved_method_map,
        "fallback_to_rd_mask": maps.fallback_to_rd_mask,
    }.items():
        np.save(array_dir / f"{name}.npy", value)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        (terrain, "Terrain altitude (observed, m)", "terrain"),
        (building_mask, "Building occupancy proxy", "gray"),
        (rss_gray, "Observed PNG RSS grayscale", "magma"),
        (maps.los_mask, "LOS mask from explicit geometry proxy", "viridis"),
        (maps.diffraction_loss_db, "Auto diffraction loss (signed dB)", "coolwarm"),
        (maps.physics_prior_db, "FSPL + auto prior (proxy geometry)", "viridis"),
    ]
    for ax, (value, title, cmap) in zip(axes.flat, panels):
        image = ax.imshow(value, cmap=cmap, origin="upper", interpolation="nearest")
        ax.plot(tx_col, tx_row, "+", color="red", markersize=10, markeredgewidth=2)
        ax.set_title(title)
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"PPData5D provisional case study: {args.sample}")
    fig.savefig(output / "case_study.png", dpi=220)
    fig.savefig(output / "case_study.pdf")
    plt.close(fig)

    summary = {
        "dataset": "PPData5D",
        "sample": args.sample,
        "frequency_id": f"f{frequency_id}",
        "frequency_hz": args.frequency_hz,
        "source_files": {
            "npz": f"npz/{base_id}_bdtr.npz",
            "rss_png": f"png/{scene_folder}/{base_id}_f{frequency_id}_ss_z{rx_height_id}.png",
        },
        "observed_npz_keys": sorted(arrays.files),
        "observed_shapes": {key: list(np.asarray(arrays[key]).shape) for key in arrays.files},
        "proxy_geometry": {
            "tx_source": "argmax of observed RSS grayscale; not verified transmitter metadata",
            "tx_image_row_col": [int(tx_row), int(tx_col)],
            "tx_xyz_m_proxy": [tx.x_m, tx.y_m, tx.z_m],
            "rx_height_m_proxy_absolute": rx_height_abs,
            "building_top_proxy": "terrain_yx + building_height_proxy_m on inBldg_zyx[0]",
            "building_height_proxy_m": args.building_height_proxy_m,
            "terrain_in_free_space_los": False,
        },
        "auto_method_counts": {
            str(method): int(np.sum(maps.resolved_method_map == method))
            for method in np.unique(maps.resolved_method_map)
        },
        "los_ratio": float(np.mean(maps.los_mask)),
        "fallback_to_rd_ratio": float(np.mean(maps.fallback_to_rd_mask)),
        "metric_status": "not_reported",
        "metric_limitation": "The inspected PPData5D files do not verify Tx position/absolute z or RSS-to-dB calibration; raw RSS grayscale is not compared as signed pathloss.",
        "reproducibility_note": "This output is a provisional geometry case study, not a validated cross-dataset dB benchmark.",
    }
    (output / "case_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
