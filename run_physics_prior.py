"""
运行跨数据集 Physics Prior并生成数组、五种diffraction method对比图、递归剖面和实验摘要。

输入:
  --data-root: RadioMap3DSeer根目录，默认../dataset；
  --config: 独立无线配置JSON；
  --scene-ids: 逗号分隔scene id，默认0,1,2；
  --tx-ids: 逗号分隔每个scene使用的Tx id，默认0；
  --diffraction-method: none、single、owr-rd、owr-cd、deygout或auto；未指定时读取配置的default_diffraction_method；
  --compare-all-modes: 同一次运行none/single/owr-rd/deygout/owr-cd，几何与FSPL只共享一份；
  --output-dir: 结果目录，默认runs/radiomap3dseer_physics_prior/3.5GHz_1m；
  --debug-rx: 可选row,col；不提供时为每个样本选择nu_max最大的Rx生成剖面。

uv run run_physics_prior.py --data-root G:/paper/radiomap3dseer --config configs/radiomap3dseer.json --scene-ids 0,1,2 --tx-ids 0 --compare-all-modes --output-dir runs/radiomap3dseer_physics_prior/3.5GHz_1m

输出:
  单mode时每个scene/Tx目录保存physics_arrays/*.npy、comparison.png/.pdf、propagation_profile.png/.pdf、
  metadata.json；--compare-all-modes时额外保存modes/<mode>/physics_arrays、comparison_all_modes和
  recursive_debug_all_modes。总目录保存summary.json。命令在macOS/Linux使用逗号参数，Windows CMD可使用同一
  单行命令。脚本默认不训练模型、不写外部数据集；指定输出目录内的同名实验结果会被本次运行更新。

逻辑:
  读取polygon绝对建筑高度和antenna绝对Tx高度，先用三维距离向量化计算FSPL，再逐Rx按mode
  重新建立区间LOS；OWR-RD只递归rooftop edge→Rx，OWR-CD只递归footprint corner→Rx，Deygout同时递归A→E*和E*→B。递归子区间排除已选
  8邻域建筑component，防止同一栅格屋顶重复计数。Ground Truth仅用于
  图示和误差统计，按配置把gain PNG映射为论文约定的负signed PL/pathgain。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from radiomap_physics import (
    DIFFRACTION_METHODS,
    DIFFRACTION_MODES,
    PhysicsMaps,
    load_ground_truth_path_loss_db,
    load_ground_truth_path_gain_db,
    load_ground_truth_path_loss_magnitude_db,
    load_json,
    load_scene,
    solve_diffraction,
    select_diffraction_method,
    propagation_profile,
    compute_physics_maps,
    free_space_path_loss_db,
)


def _configure_fonts() -> tuple[str, str]:
    """设置中文宋体、西文Times New Roman；缺失时返回可复核的回退字体。"""

    installed = {font.name for font in font_manager.fontManager.ttflist}
    chinese = "SimSun" if "SimSun" in installed else "DejaVu Sans"
    western = "Times New Roman" if "Times New Roman" in installed else "DejaVu Serif"
    plt.rcParams["font.family"] = [western]
    plt.rcParams["axes.unicode_minus"] = False
    return chinese, western


def _building_evaluated_prior(
    height_map_m: np.ndarray,
    physics_prior_db: np.ndarray,
    building_min_db: float,
) -> np.ndarray:
    """Apply the protocol's building-interior display/evaluation rule.

    The raw solver output remains ``FSPL + diffraction``.  Building pixels are
    replaced only in this derived evaluation map, so the physical invariant is
    still auditable from ``physics_prior_db`` and ``diffraction_loss_db``.
    """

    return np.where(np.asarray(height_map_m) > 0.0, float(building_min_db), physics_prior_db).astype(np.float32)


def _save_arrays(output_dir: Path, maps: PhysicsMaps, building_evaluated_prior_db: np.ndarray) -> None:
    """保存全部用户要求的物理中间量，使用稳定文件名和float32/int32/bool dtype。"""

    array_dir = output_dir / "physics_arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "fspl_db": maps.fspl_db,
        "fspl_paper_pl_db": maps.fspl_db,
        "fspl_loss_magnitude_db": -maps.fspl_db,
        "diffraction_loss_db": maps.diffraction_loss_db,
        "diffraction_loss_magnitude_db": -maps.diffraction_loss_db,
        "nu_max": maps.nu_max,
        "dominant_row": maps.dominant_row,
        "dominant_col": maps.dominant_col,
        "dominant_height_m": maps.dominant_height_m,
        "dominant_h_m": maps.dominant_h_m,
        "dominant_d1_m": maps.dominant_d1_m,
        "dominant_d2_m": maps.dominant_d2_m,
        "los_mask": maps.los_mask,
        "physics_prior_db": maps.physics_prior_db,
        "physics_prior_paper_pl_db": maps.physics_prior_db,
        "physics_prior_loss_magnitude_db": -maps.physics_prior_db,
        "physics_prior_evaluation_db": building_evaluated_prior_db,
        "physics_prior_evaluation_loss_magnitude_db": -building_evaluated_prior_db,
        "edge_count": maps.edge_count,
        "corner_count": maps.corner_count,
        "unresolved_nlos_mask": maps.unresolved_nlos_mask,
        "cd_validity_mask": maps.cd_validity_mask,
        "first_blocking_building_id": maps.first_blocking_building_id,
        "resolved_method_map": maps.resolved_method_map,
        "fallback_to_rd_mask": maps.fallback_to_rd_mask,
    }
    for name, array in arrays.items():
        np.save(array_dir / f"{name}.npy", array)


def _imshow(ax, array: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = "viridis") -> None:
    """用指定共享色阶绘制一个等比例地图子图。"""

    image = ax.imshow(array, cmap=cmap, aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x / col")
    ax.set_ylabel("y / row")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def render_comparison(
    height_map_m: np.ndarray,
    maps: PhysicsMaps,
    evaluated_prior_db: np.ndarray,
    ground_truth_path_loss_db: np.ndarray,
    tx_x: float,
    tx_row: float,
    color_limits: dict[str, list[float]],
    output_stem: Path,
    dataset_name: str,
) -> dict[str, list[float]]:
    """生成五联图；每类物理量使用配置中跨样本固定的色阶，不做单图min-max。"""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.7), constrained_layout=True)
    _imshow(axes[0], height_map_m, "Building Height (m)", *color_limits["building_height_m"])
    _imshow(axes[1], maps.fspl_db, "FSPL / signed PL (dB)", *color_limits["fspl_db"])
    _imshow(axes[2], maps.diffraction_loss_db, "Diffraction / signed loss (dB)", *color_limits["diffraction_loss_db"])
    _imshow(axes[3], evaluated_prior_db, "FSPL + Diffraction / evaluated signed PL (dB)", *color_limits["physics_prior_db"])
    _imshow(axes[4], ground_truth_path_loss_db, "Ground Truth / signed PL (dB)", *color_limits["ground_truth_path_loss_db"])
    for ax in axes:
        ax.plot(tx_x, tx_row, marker="+", color="red", markersize=8, markeredgewidth=1.5)
    fig.suptitle(f"{dataset_name} Deterministic Physics Prior", fontsize=14)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)
    return color_limits


def render_profile(
    profile,
    tx_x: float,
    tx_y_world: float,
    tx_z_m: float,
    rx_height_m: float,
    horizontal_distance_m: float,
    output_stem: Path,
) -> None:
    """生成包含建筑剖面、LOS、候选障碍和dominant E*标注的高分辨率图。"""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    ax.plot(profile.s_m, profile.building_height_m, color="#4C78A8", linewidth=1.8, label="Building Height Profile")
    ax.plot(profile.s_m, profile.los_height_m, color="#F58518", linestyle="--", linewidth=1.5, label="LOS Line")
    candidate = profile.candidate_mask
    ax.scatter(profile.s_m[candidate], profile.building_height_m[candidate], color="#E45756", s=16, label="candidate obstacles", zorder=3)
    ax.scatter([0.0], [tx_z_m], color="#54A24B", marker="o", s=55, label="Tx", zorder=5)
    ax.scatter([horizontal_distance_m], [rx_height_m], color="#FF9DA6", marker="o", s=55, label="Rx", zorder=5)
    ax.annotate("Tx", (0.0, tx_z_m), xytext=(5, 6), textcoords="offset points", fontsize=10)
    ax.annotate("Rx", (horizontal_distance_m, rx_height_m), xytext=(-20, 6), textcoords="offset points", fontsize=10)
    if profile.has_obstruction:
        index = profile.dominant_index
        ax.scatter([profile.s_m[index]], [profile.building_height_m[index]], color="#B279A2", marker="*", s=140, label="dominant E*", zorder=4)
        ax.annotate(
            "E*",
            (profile.s_m[index], profile.building_height_m[index]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Horizontal path distance s (m)")
    ax.set_ylabel("Absolute height (m)")
    ax.set_title(f"Tx world ({tx_x:.0f}, {tx_y_world:.0f}) → Rx (row={profile.rx_row}, col={profile.rx_col})")
    ax.legend(loc="upper left", fontsize=9)
    if profile.has_obstruction:
        status = "NLOS"
        detail = (
            f"E*=({profile.dominant_row},{profile.dominant_col})  H={profile.dominant_height_m:.3f} m\n"
            f"h={profile.dominant_h_m:.3f} m, d1={profile.dominant_d1_m:.3f} m, d2={profile.dominant_d2_m:.3f} m\n"
            f"ν_max={profile.nu_max:.5f}, J={profile.diffraction_loss_db:.3f} dB, signed L_diff={-profile.diffraction_loss_db:.3f} dB"
        )
    else:
        status = "LOS"
        detail = "No candidate obstacle with h > 0; L_diff=0 dB; ν_max=NaN"
    ax.text(
        0.99,
        0.98,
        f"{status}\n{detail}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def _endpoint_s_m(endpoint, tx_x: float, tx_row: float, resolution_m: float) -> float:
    """把递归节点端点转换为相对原始Tx的水平路径距离，单位m。"""

    return float(np.hypot(endpoint.col - tx_x, endpoint.row - tx_row) * resolution_m)


def _serialize_solution(solution) -> dict:
    """将递归解转换为可读JSON，保留每一级A/B/E*/h/d1/d2/nu/J。"""

    def endpoint_dict(endpoint) -> dict[str, float]:
        return {"row": endpoint.row, "col": endpoint.col, "z_m": endpoint.z_m}

    return {
        "mode": solution.mode,
        "loss_db": solution.loss_db,
        "signed_loss_db": -solution.loss_db,
        "loss_magnitude_db": solution.loss_db,
        "termination": solution.termination,
        "is_los": solution.is_los,
        "dispatch_method": solution.dispatch_method,
        "fallback_from": solution.fallback_from,
        "edge_count": len(solution.events),
        "corner_count": sum(event.event_type == "corner" for event in solution.events),
        "corner_diagnostics": solution.corner_diagnostics,
        "first_blocking_building_id": solution.first_blocking_component_id,
        "events": [
            {
                "depth": event.depth,
                "A": endpoint_dict(event.a),
                "B": endpoint_dict(event.b),
                "E_star": endpoint_dict(event.edge),
                "h_m": event.h_m,
                "d1_m": event.d1_m,
                "d2_m": event.d2_m,
                "nu": event.nu,
                "J_nu_db": event.loss_db,
                "building_component_id": event.component_id,
                "event_type": event.event_type,
                "wedge_angle_rad": event.wedge_angle_rad,
                "incident_angle_rad": event.incident_angle_rad,
                "diffraction_angle_rad": event.diffraction_angle_rad,
                "corner_model": event.corner_model,
                "incident_direction_rc": event.incident_direction_rc,
                "outgoing_direction_rc": event.outgoing_direction_rc,
            }
            for event in solution.events
        ],
    }


def render_recursive_debug(
    height_map_m: np.ndarray,
    tx,
    rx_row: int,
    rx_col: int,
    rx_height_m: float,
    resolution_m: float,
    frequency_hz: float,
    path_sampling_step_m: float,
    solutions: dict[str, object],
    output_stem: Path,
) -> None:
    """对同一Tx-Rx并排展示single、OWR-RD链和Deygout双向递归树。"""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    tx_row = height_map_m.shape[0] - 1 - tx.y_m
    wavelength_m = 299792458.0 / frequency_hz
    base_profile = propagation_profile(
        height_map_m,
        tx,
        rx_row,
        rx_col,
        rx_height_m,
        resolution_m,
        wavelength_m,
        path_sampling_step_m,
    )
    total_distance_m = float(np.hypot(rx_col - tx.x_m, rx_row - tx_row) * resolution_m)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True, sharex=True)
    names = [("single", "Single Dominant Rooftop Edge"), ("owr-rd", "OWR-RD (One-Way Recursive Rooftop Diffraction)"), ("deygout", "Deygout Multiple Rooftop Edge")]
    for ax, (mode, title) in zip(axes, names):
        ax.plot(base_profile.s_m, base_profile.building_height_m, color="#4C78A8", linewidth=1.5, label="Building Height")
        ax.plot(base_profile.s_m, base_profile.los_height_m, color="#F58518", linestyle="--", linewidth=1.2, label="Tx-Rx LOS")
        ax.scatter([0.0], [tx.z_m], color="#54A24B", s=40, label="Tx", zorder=5)
        ax.scatter([total_distance_m], [rx_height_m], color="#FF9DA6", s=40, label="Rx", zorder=5)
        solution = solutions[mode]
        max_depth = max((event.depth for event in solution.events), default=0)
        colors = plt.get_cmap("plasma")
        for event_index, event in enumerate(solution.events):
            edge_s = _endpoint_s_m(event.edge, tx.x_m, tx_row, resolution_m)
            color = colors(0.15 + 0.75 * (event.depth / max(1, max_depth)))
            label = "E*" if mode == "single" else (f"E{event_index + 1}*" if mode == "owr-rd" else f"depth {event.depth}")
            ax.scatter([edge_s], [event.edge.z_m], color=color, s=95 if mode == "single" else 65, marker="*" if mode == "single" else "o", zorder=6, label=label if event_index == 0 or mode != "deygout" else None)
            a_s = _endpoint_s_m(event.a, tx.x_m, tx_row, resolution_m)
            b_s = _endpoint_s_m(event.b, tx.x_m, tx_row, resolution_m)
            bar_y = -0.65 - 0.48 * event.depth
            ax.plot([a_s, b_s], [bar_y, bar_y], color=color, linewidth=3.0, alpha=0.85)
            ax.scatter([edge_s], [bar_y], color=color, s=22, zorder=7)
            if mode != "single":
                ax.annotate(label, (edge_s, event.edge.z_m), xytext=(3, 4), textcoords="offset points", fontsize=8)
        if solution.events:
            lines = [
                f"d{e.depth}: E*=({e.edge.row:.0f},{e.edge.col:.0f}) h={e.h_m:.2f} d1={e.d1_m:.2f} d2={e.d2_m:.2f} ν={e.nu:.2f} J={e.loss_db:.2f}"
                for e in solution.events
            ]
            detail = f"signed L_diff={-solution.loss_db:.2f} dB\n" + "\n".join(lines[:8])
            if len(lines) > 8:
                detail += f"\n... {len(lines) - 8} more edges"
        else:
            detail = "No valid h > 0 obstacle; L_diff=0 dB"
        ax.text(0.995, 0.98, detail, transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88})
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Height (m)")
        ax.set_ylim(-0.9 - 0.5 * max(1, max_depth), max(float(np.nanmax(base_profile.building_height_m)), tx.z_m) + 2.0)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Horizontal distance from original Tx (m)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=4)
    fig.suptitle(f"Recursive diffraction debug: Tx ({tx.x_m:.0f},{tx.y_m:.0f}) → Rx (row={rx_row}, col={rx_col})", fontsize=14)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def render_corner_recursive_debug(
    height_map_m: np.ndarray,
    tx,
    rx_row: int,
    rx_col: int,
    solution,
    footprint_polygons: list[dict] | None,
    output_stem: Path,
    dataset_name: str,
) -> None:
    """Render the OWR-CD Tx→corner chain over building footprints."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 8.0), constrained_layout=True)
    ax.imshow(height_map_m > 0.0, cmap="Greys", interpolation="nearest", origin="upper", aspect="equal")
    for item in footprint_polygons or []:
        coordinates = item.get("coordinates_xy", []) if isinstance(item, dict) else item
        if len(coordinates) < 3:
            continue
        polygon = np.asarray([(float(y), float(x)) for x, y in coordinates], dtype=np.float64)
        closed = np.vstack((polygon, polygon[0]))
        ax.plot(closed[:, 1], closed[:, 0], color="#4C78A8", linewidth=0.7, alpha=0.75)
    tx_row = height_map_m.shape[0] - 1 - tx.y_m
    ax.plot(tx.x_m, tx_row, marker="+", color="#D55E00", markersize=12, markeredgewidth=2.0, label="Tx")
    ax.plot(rx_col, rx_row, marker="o", color="#E45756", markersize=6, markeredgewidth=1.5, markerfacecolor="none", label="Rx")
    ax.plot([tx.x_m, rx_col], [tx_row, rx_row], color="#CC3311", linestyle=":", linewidth=1.4, label="original Tx-Rx LOS")
    for diagnostic in getattr(solution, "corner_diagnostics", []):
        for candidate in diagnostic.get("candidates", []):
            corner = candidate.get("corner", {})
            row = float(corner.get("row", np.nan))
            col = float(corner.get("col", np.nan))
            if not np.isfinite(row + col):
                continue
            if candidate.get("selected"):
                ax.scatter(col, row, color="#B279A2", marker="o", s=95, zorder=6, label="selected corner")
            elif candidate.get("silhouette_corner"):
                ax.scatter(col, row, facecolors="none", edgecolors="#F0E442", marker="o", s=75, linewidths=1.5, zorder=5, label="rejected silhouette corner")
            elif candidate.get("visible_from_current"):
                ax.scatter(col, row, color="#999999", marker="x", s=38, zorder=4, label="rejected visible vertex")
    path_points = [(float(tx.x_m), float(tx_row))]
    path_points.extend((float(event.edge.col), float(event.edge.row)) for event in solution.events)
    path_points.append((float(rx_col), float(rx_row)))
    for segment_index in range(len(path_points) - 1):
        start = path_points[segment_index]
        end = path_points[segment_index + 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], color="#F58518", linewidth=2.4, linestyle="-" if segment_index == 0 else "--", label="selected free-space path" if segment_index == 0 else None)
    for index, event in enumerate(solution.events):
        ax.scatter(event.edge.col, event.edge.row, color="#B279A2", marker="o", s=65, zorder=7)
        ax.annotate(f"C{index + 1} / B{event.component_id}", (event.edge.col, event.edge.row), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_title(f"{dataset_name} OWR-CD: Tx → corner chain → Rx")
    ax.set_xlabel("image col / x")
    ax.set_ylabel("image row")
    ax.legend(loc="upper right", fontsize=8)
    detail = "\n".join(
        f"C{index + 1}: d1={event.d1_m:.2f} m, d2={event.d2_m:.2f} m, "
        f"wedge={event.wedge_angle_rad:.3f} rad, loss={event.loss_db:.2f} dB"
        for index, event in enumerate(solution.events)
    ) or f"termination={solution.termination}"
    detail += f"\naccepted corners={len(solution.events)}; diagnostics={len(getattr(solution, 'corner_diagnostics', []))}"
    ax.text(0.01, 0.99, detail, transform=ax.transAxes, va="top", fontsize=8, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88})
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def render_owr_cd_diagnostics(
    height_map_m: np.ndarray,
    maps: PhysicsMaps,
    tx_x: float,
    tx_row: float,
    color_limits: dict[str, list[float]],
    output_stem: Path,
    dataset_name: str,
) -> None:
    """Render the six geometry/validity maps specific to strict OWR-CD."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    first_blocker = np.ma.masked_where(maps.first_blocking_building_id < 0, maps.first_blocking_building_id)
    panels = [
        ("LOS mask (1=LOS)", maps.los_mask.astype(np.float32), (0.0, 1.0), "viridis"),
        ("CD validity (1=valid)", maps.cd_validity_mask.astype(np.float32), (0.0, 1.0), "viridis"),
        ("Unresolved NLOS", maps.unresolved_nlos_mask.astype(np.float32), (0.0, 1.0), "magma"),
        ("Corner count", maps.corner_count, (0.0, max(3.0, float(np.max(maps.corner_count)))), "viridis"),
        ("OWR-CD diffraction / signed dB", maps.diffraction_loss_db, tuple(color_limits["diffraction_loss_db"]), "viridis"),
        ("First blocking building ID", first_blocker, (-1.0, max(1.0, float(np.max(maps.first_blocking_building_id)))), "tab20"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (title, array, limits, cmap) in zip(axes.flat, panels):
        image = ax.imshow(array, cmap=cmap, aspect="equal", interpolation="nearest", vmin=limits[0], vmax=limits[1])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")
        ax.plot(tx_x, tx_row, marker="+", color="red", markersize=8, markeredgewidth=1.5)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{dataset_name} OWR-CD geometry diagnostics", fontsize=15)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)

def render_mode_comparison(
    height_map_m: np.ndarray,
    ground_truth_path_loss_db: np.ndarray,
    fspl_db: np.ndarray,
    maps_by_mode: dict[str, PhysicsMaps],
    evaluated_prior_by_mode: dict[str, np.ndarray],
    tx_x: float,
    tx_row: float,
    color_limits: dict[str, list[float]],
    output_stem: Path,
    dataset_name: str,
) -> None:
    """生成Building/GT/FSPL及四种非零diffraction method的对比图。"""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Building Height (m)", height_map_m, "building_height_m"),
        ("Ground Truth / signed PL (dB)", ground_truth_path_loss_db, "ground_truth_path_loss_db"),
        ("FSPL / none / signed PL (dB)", fspl_db, "fspl_db"),
        ("Single Diffraction / signed (dB)", maps_by_mode["single"].diffraction_loss_db, "diffraction_loss_db"),
        ("FSPL + Single / evaluated signed PL (dB)", evaluated_prior_by_mode["single"], "physics_prior_db"),
        ("OWR-RD Diffraction / signed (dB)", maps_by_mode["owr-rd"].diffraction_loss_db, "diffraction_loss_db"),
        ("FSPL + OWR-RD / evaluated signed PL (dB)", evaluated_prior_by_mode["owr-rd"], "physics_prior_db"),
        ("Deygout Diffraction / signed (dB)", maps_by_mode["deygout"].diffraction_loss_db, "diffraction_loss_db"),
        ("FSPL + Deygout / evaluated signed PL (dB)", evaluated_prior_by_mode["deygout"], "physics_prior_db"),
        ("OWR-CD Diffraction / signed (dB)", maps_by_mode["owr-cd"].diffraction_loss_db, "diffraction_loss_db"),
        ("FSPL + OWR-CD / evaluated signed PL (dB)", evaluated_prior_by_mode["owr-cd"], "physics_prior_db"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(17, 13), constrained_layout=True)
    for ax, (title, array, limit_key) in zip(axes.flat, panels):
        _imshow(ax, array, title, *color_limits[limit_key])
        ax.plot(tx_x, tx_row, marker="+", color="red", markersize=7, markeredgewidth=1.4)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{dataset_name} Diffraction Mode Comparison", fontsize=15)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def _mode_metrics(
    maps: PhysicsMaps,
    ground_truth_path_loss_db: np.ndarray,
    evaluated_prior_db: np.ndarray,
) -> dict[str, float]:
    """计算单mode的runtime之外的diffraction统计和与GT的误差。"""

    error = evaluated_prior_db.astype(np.float64) - ground_truth_path_loss_db.astype(np.float64)
    los = maps.los_mask
    nlos = ~los
    ground_truth_min = float(np.min(ground_truth_path_loss_db))
    return {
        "mean_diffraction_loss_db": float(maps.diffraction_loss_db.mean()),
        "min_diffraction_loss_db": float(maps.diffraction_loss_db.min()),
        "max_abs_diffraction_loss_db": float(np.max(np.abs(maps.diffraction_loss_db))),
        "nlos_ratio": float((~maps.los_mask).mean()),
        "mean_edge_count": float(maps.edge_count.mean()),
        "max_edge_count": int(maps.edge_count.max()),
        "mae_vs_ground_truth_db": float(np.mean(np.abs(error))),
        "rmse_vs_ground_truth_db": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_vs_ground_truth_db": float(np.max(np.abs(error))),
        "bias_vs_ground_truth_db": float(np.mean(error)),
        "los_rmse_vs_ground_truth_db": float(np.sqrt(np.mean(error[los] ** 2))) if np.any(los) else float("nan"),
        "nlos_rmse_vs_ground_truth_db": float(np.sqrt(np.mean(error[nlos] ** 2))) if np.any(nlos) else float("nan"),
        "prediction_min_db": float(np.min(evaluated_prior_db)),
        "raw_prediction_min_db": float(np.min(maps.physics_prior_db)),
        "prediction_below_ground_truth_min_ratio": float(np.mean(evaluated_prior_db < ground_truth_min)),
        "corner_count_mean": float(np.mean(maps.corner_count)),
        "corner_count_max": int(np.max(maps.corner_count)),
        "cd_validity_ratio": float(np.mean(maps.cd_validity_mask)),
        "cd_validity_nlos_ratio": float(np.mean(maps.cd_validity_mask & nlos)),
        "unresolved_nlos_ratio": float(np.mean(maps.unresolved_nlos_mask)),
        "fallback_to_rd_ratio": float(np.mean(maps.fallback_to_rd_mask)),
        "resolved_method_counts": {
            str(method): int(np.sum(maps.resolved_method_map == method))
            for method in np.unique(maps.resolved_method_map)
        },
    }


def _pairwise_mode_metrics(maps_by_mode: dict[str, PhysicsMaps]) -> dict[str, dict[str, float]]:
    """比较同一张图上三组非none模式的diffraction/prior差异。"""

    comparisons = {}
    for left, right in (("single", "owr-rd"), ("owr-rd", "deygout"), ("single", "deygout"), ("owr-cd", "owr-rd"), ("owr-cd", "deygout")):
        diffraction_delta = maps_by_mode[left].diffraction_loss_db.astype(np.float64) - maps_by_mode[right].diffraction_loss_db.astype(np.float64)
        prior_delta = maps_by_mode[left].physics_prior_db.astype(np.float64) - maps_by_mode[right].physics_prior_db.astype(np.float64)
        key = f"{left}_vs_{right}"
        comparisons[key] = {
            "mean_abs_diffraction_delta_db": float(np.mean(np.abs(diffraction_delta))),
            "rmse_diffraction_delta_db": float(np.sqrt(np.mean(diffraction_delta**2))),
            "max_abs_diffraction_delta_db": float(np.max(np.abs(diffraction_delta))),
            "mean_abs_prior_delta_db": float(np.mean(np.abs(prior_delta))),
            "rmse_prior_delta_db": float(np.sqrt(np.mean(prior_delta**2))),
            "max_abs_prior_delta_db": float(np.max(np.abs(prior_delta))),
        }
    return comparisons


def _parse_ids(value: str) -> list[str]:
    """解析逗号分隔的非空字符串ID列表。"""

    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("At least one id is required.")
    return ids


def _pick_debug_rx(maps: PhysicsMaps, tx, resolution_m: float) -> tuple[int, int]:
    """优先选择递归edge最多、且距Tx至少32m的Rx，使默认剖面展示完整树。"""

    if not np.any(maps.edge_count > 0):
        return (0, 0)
    rows, cols = np.indices(maps.nu_max.shape)
    tx_row = maps.nu_max.shape[0] - 1 - tx.y_m
    distance_m = np.hypot(rows - tx_row, cols - tx.x_m) * resolution_m
    far_mask = (maps.edge_count > 0) & (distance_m >= 32.0)
    score = maps.edge_count.astype(np.float64) * 1000.0 + np.nan_to_num(maps.nu_max, nan=-np.inf)
    candidate_map = np.where(far_mask, score, -np.inf)
    if np.isfinite(candidate_map).any():
        row, col = np.unravel_index(np.argmax(candidate_map), candidate_map.shape)
    else:
        row, col = np.unravel_index(np.argmax(score), score.shape)
    return int(row), int(col)


def main() -> None:
    """解析命令、运行样本、写入可复现结果和运行时摘要。"""

    parser = argparse.ArgumentParser(description="Generate deterministic cross-dataset FSPL and selectable diffraction priors.")
    parser.add_argument("--data-root", default="../dataset", help="External dataset root; layout is selected by config data.format.")
    parser.add_argument("--config", default="configs/radiomap3dseer.json", help="Independent wireless configuration JSON.")
    parser.add_argument("--scene-ids", default="0,1,2", help="Comma-separated scene IDs.")
    parser.add_argument("--tx-ids", default="0", help="Comma-separated Tx IDs applied to every selected scene.")
    parser.add_argument(
        "--diffraction-method",
        "--diffraction-mode",
        dest="diffraction_method",
        choices=DIFFRACTION_METHODS,
        default=None,
        help="Diffraction solver: none, single, owr-rd, owr-cd, deygout, or auto.",
    )
    parser.add_argument("--compare-all-modes", action="store_true", help="Run none, single, owr-rd, deygout and owr-cd in one shared-geometry comparison.")
    parser.add_argument("--output-dir", default="runs/radiomap3dseer_physics_prior/3.5GHz_1m", help="Output experiment directory.")
    parser.add_argument("--debug-rx", default=None, help="Optional row,col used for every sample's propagation profile; default is max-nu Rx.")
    args = parser.parse_args()

    chinese_font, western_font = _configure_fonts()
    config = load_json(Path(args.config))
    configured_default = config.get("physics_prior", {}).get("default_diffraction_method", "single")
    diffraction_method = args.diffraction_method or configured_default
    if diffraction_method not in DIFFRACTION_METHODS:
        raise ValueError(f"Unsupported diffraction method: {diffraction_method}")
    scene_ids = _parse_ids(args.scene_ids)
    tx_ids = _parse_ids(args.tx_ids)
    modes = list(DIFFRACTION_MODES) if args.compare_all_modes else [diffraction_method]
    debug_rx = None
    if args.debug_rx:
        parts = [int(value.strip()) for value in args.debug_rx.split(",")]
        if len(parts) != 2:
            raise ValueError("--debug-rx must be row,col")
        debug_rx = (parts[0], parts[1])
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    frequency_hz = float(config["wireless"]["frequency_hz"])
    rx_height_m = float(config["scene"]["rx_height_m"])
    resolution_m = float(config["scene"]["resolution_m_per_pixel"])
    step_m = float(config["physics_prior"]["path_sampling_step_m"])
    c_mps = float(config["physics_prior"]["speed_of_light_m_per_s"])
    max_corner_depth_raw = config["physics_prior"].get("max_corner_depth")
    max_corner_depth = None if max_corner_depth_raw is None else int(max_corner_depth_raw)
    color_limits = config["visualization"]["fixed_color_limits"]
    building_min_db = float(config["dataset_labels"]["png_gray_mapping"]["pathgain_db_range"][0])

    print(f"[start] scenes={scene_ids} tx_ids={tx_ids} modes={modes} data_root={Path(args.data_root).resolve()}", flush=True)
    print(f"[config] f={frequency_hz/1e9:.3f} GHz, resolution={resolution_m:.3f} m, rx_height={rx_height_m:.3f} m, fonts={chinese_font}/{western_font}", flush=True)
    for scene_id in scene_ids:
        scene = load_scene(args.data_root, scene_id, config)
        for tx_id in tx_ids:
            index = int(tx_id)
            if not 0 <= index < len(scene.tx_records):
                raise IndexError(f"Tx id {tx_id} out of range for scene {scene_id}; count={len(scene.tx_records)}")
            tx = scene.tx_records[index]
            sample_dir = output_root / f"scene_{scene_id}_tx_{tx_id}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_started = time.perf_counter()
            fspl_db = free_space_path_loss_db(scene.height_map_m, tx, frequency_hz, rx_height_m, resolution_m, c_mps)
            ground_truth = load_ground_truth_path_loss_db(args.data_root, scene_id, tx_id, config)
            ground_truth_paper_pl = load_ground_truth_path_gain_db(args.data_root, scene_id, tx_id, config)
            ground_truth_magnitude = load_ground_truth_path_loss_magnitude_db(args.data_root, scene_id, tx_id, config)
            np.save(sample_dir / "ground_truth_pathgain_db.npy", ground_truth_paper_pl)
            np.save(sample_dir / "ground_truth_pathloss_magnitude_db.npy", ground_truth_magnitude)
            auto_method = select_diffraction_method(scene.height_map_m, tx, rx_height_m)
            maps_by_mode: dict[str, PhysicsMaps] = {}
            evaluated_prior_by_mode: dict[str, np.ndarray] = {}
            mode_records: dict[str, dict] = {}
            for mode in modes:
                mode_started = time.perf_counter()
                maps = compute_physics_maps(
                    scene.height_map_m,
                    tx,
                    frequency_hz,
                    rx_height_m,
                    resolution_m,
                    step_m,
                    c_mps,
                    diffraction_mode=mode,
                    fspl_db=fspl_db,
                    footprint_polygons=scene.polygons,
                    max_corner_depth=max_corner_depth,
                )
                maps_by_mode[mode] = maps
                evaluated_prior = _building_evaluated_prior(scene.height_map_m, maps.physics_prior_db, building_min_db)
                evaluated_prior_by_mode[mode] = evaluated_prior
                mode_dir = sample_dir / "modes" / mode if args.compare_all_modes else sample_dir
                _save_arrays(mode_dir, maps, evaluated_prior)
                mode_records[mode] = {
                    "runtime_seconds": time.perf_counter() - mode_started,
                    **_mode_metrics(maps, ground_truth, evaluated_prior),
                }
                (mode_dir / "mode_metadata.json").write_text(
                    json.dumps({"mode": mode, **mode_records[mode]}, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[mode] scene={scene_id} tx={tx_id} method={mode} resolved={maps.mode} runtime={mode_records[mode]['runtime_seconds']:.2f}s edges_max={mode_records[mode]['max_edge_count']}", flush=True)

            debug_reference_maps = maps_by_mode.get("owr-cd", maps_by_mode.get("deygout", maps_by_mode.get("single", maps_by_mode[modes[0]])))
            selected_rx = debug_rx if debug_rx is not None else _pick_debug_rx(debug_reference_maps, tx, resolution_m)
            solutions = {
                mode: solve_diffraction(
                    scene.height_map_m,
                    tx,
                    selected_rx[0],
                    selected_rx[1],
                    mode,
                    rx_height_m,
                    resolution_m,
                    c_mps / frequency_hz,
                    step_m,
                    None,
                    scene.polygons,
                    max_corner_depth,
                )
                for mode in (*DIFFRACTION_MODES, "auto")
            }
            tx_row = scene.height_map_m.shape[0] - 1 - tx.y_m
            horizontal_distance_m = float(np.hypot(selected_rx[0] - tx_row, selected_rx[1] - tx.x_m) * resolution_m)
            profile = propagation_profile(
                scene.height_map_m,
                tx,
                selected_rx[0],
                selected_rx[1],
                rx_height_m,
                resolution_m,
                c_mps / frequency_hz,
                step_m,
            )
            actual_color_limits = color_limits
            if args.compare_all_modes:
                render_mode_comparison(
                    scene.height_map_m,
                    ground_truth,
                    fspl_db,
                    maps_by_mode,
                    evaluated_prior_by_mode,
                    tx.x_m,
                    tx_row,
                    color_limits,
                    sample_dir / "comparison_all_modes",
                    str(config.get("dataset", "Dataset")),
                )
                render_recursive_debug(
                    scene.height_map_m,
                    tx,
                    selected_rx[0],
                    selected_rx[1],
                    rx_height_m,
                    resolution_m,
                    frequency_hz,
                    step_m,
                    solutions,
                    sample_dir / "recursive_debug_all_modes",
                )
                render_corner_recursive_debug(
                    scene.height_map_m,
                    tx,
                    selected_rx[0],
                    selected_rx[1],
                    solutions["owr-cd"],
                    scene.polygons,
                    sample_dir / "corner_recursive_debug_owr_cd",
                    str(config.get("dataset", "Dataset")),
                )
            else:
                maps = maps_by_mode[modes[0]]
                actual_color_limits = render_comparison(
                    scene.height_map_m,
                    maps,
                    evaluated_prior_by_mode[modes[0]],
                    ground_truth,
                    tx.x_m,
                    tx_row,
                    color_limits,
                    sample_dir / "comparison",
                    str(config.get("dataset", "Dataset")),
                )
                render_profile(
                    profile,
                    tx.x_m,
                    tx.y_m,
                    tx.z_m,
                    rx_height_m,
                    horizontal_distance_m,
                    sample_dir / "propagation_profile",
                )
                render_recursive_debug(
                    scene.height_map_m,
                    tx,
                    selected_rx[0],
                    selected_rx[1],
                    rx_height_m,
                    resolution_m,
                    frequency_hz,
                    step_m,
                    solutions,
                    sample_dir / "recursive_debug_all_modes",
                )
                render_corner_recursive_debug(
                    scene.height_map_m,
                    tx,
                    selected_rx[0],
                    selected_rx[1],
                    solutions["owr-cd"],
                    scene.polygons,
                    sample_dir / "corner_recursive_debug_owr_cd",
                    str(config.get("dataset", "Dataset")),
                )
            if "owr-cd" in maps_by_mode:
                render_owr_cd_diagnostics(
                    scene.height_map_m,
                    maps_by_mode["owr-cd"],
                    tx.x_m,
                    tx_row,
                    color_limits,
                    sample_dir / "owr_cd_diagnostics",
                    str(config.get("dataset", "Dataset")),
                )
            selected_method = "auto-per-nlos-rx" if diffraction_method == "auto" else maps_by_mode[modes[0]].mode
            selected_solution = solutions["auto"] if diffraction_method == "auto" else solutions[selected_method]
            mode_comparisons = _pairwise_mode_metrics(maps_by_mode) if args.compare_all_modes else {}
            (sample_dir / "recursive_debug.json").write_text(
                json.dumps(
                    {"rx_row_col": list(selected_rx), "solutions": {mode: _serialize_solution(solution) for mode, solution in solutions.items()}},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata = {
                "scene_id": scene_id,
                "tx_id": tx_id,
                "selected_diffraction_method": "compare-all" if args.compare_all_modes else diffraction_method,
                "resolved_diffraction_method": selected_method,
                "compare_all_modes": args.compare_all_modes,
                "executed_methods": modes,
                "auto_selected_method": "per-nlos-rx",
                "auto_height_method_hint": auto_method,
                "resolved_method_counts": {
                    str(method): int(np.sum(maps_by_mode[modes[0]].resolved_method_map == method))
                    for method in np.unique(maps_by_mode[modes[0]].resolved_method_map)
                },
                "auto_selection_rule": config.get("physics_prior", {}).get("auto_selection_rule"),
                "tx_xyz_m": [tx.x_m, tx.y_m, tx.z_m],
                "tx_rooftop_height_m": tx.rooftop_height_m,
                "tx_image_row_col": [scene.height_map_m.shape[0] - 1 - tx.y_m, tx.x_m],
                "tx_height_audit": config["scene"].get(
                    "tx_height_audit_note",
                    "antenna z - configured rooftop offset exactly matches a polygon rooftop height",
                ),
                "data_format": config.get("data", {}).get("format", "polygon_height_json"),
                "db_convention": {
                    "physics_arrays_fspl_diffraction_prior": "negative signed PL/pathgain dB",
                    "ground_truth_pathgain_and_paper_pl": "negative dB, PL=(P_Rx)_dB-(P_Tx)_dB",
                    "optional_magnitude_arrays": "loss_magnitude_db = -signed_db",
                },
                "building_interior_evaluation_rule": {
                    "mask": "height_map_m > 0",
                    "replacement_signed_pathgain_db": building_min_db,
                    "raw_solver_arrays_unchanged": True,
                    "evaluation_array": "physics_prior_evaluation_db.npy",
                },
                "rx_height_m": rx_height_m,
                "debug_rx_row_col": list(selected_rx),
                "debug_status": "NLOS" if not selected_solution.is_los else "LOS",
                "debug_solutions": {mode: _serialize_solution(solution) for mode, solution in solutions.items()},
                "mode_records": mode_records,
                "mode_comparisons": mode_comparisons,
                "map_stats": {
                    "building_height_min_m": float(scene.height_map_m[scene.height_map_m > 0].min()),
                    "building_height_max_m": float(scene.height_map_m.max()),
                    "nlos_ratio": float((~maps_by_mode[modes[0]].los_mask).mean()),
                    "fspl_min_db": float(fspl_db.min()),
                    "fspl_max_db": float(fspl_db.max()),
                    "selected_diffraction_min_db": float(maps_by_mode[modes[0]].diffraction_loss_db.min()),
                    "selected_diffraction_max_abs_db": float(np.max(np.abs(maps_by_mode[modes[0]].diffraction_loss_db))),
                    "selected_prior_max_db": float(maps_by_mode[modes[0]].physics_prior_db.max()),
                    "selected_evaluated_prior_min_db": float(evaluated_prior_by_mode[modes[0]].min()),
                    "ground_truth_path_loss_min_db": float(ground_truth.min()),
                    "ground_truth_path_loss_max_db": float(ground_truth.max()),
                },
                "fixed_color_limits": actual_color_limits,
                "runtime_seconds": time.perf_counter() - sample_started,
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(metadata)
            print(f"[done] scene={scene_id} tx={tx_id} runtime={metadata['runtime_seconds']:.2f}s debug_rx={selected_rx} status={metadata['debug_status']} output={sample_dir}", flush=True)
    summary = {
        "experiment": (
            f"{config.get('dataset', 'dataset').lower()}_diffraction_modes"
            if args.compare_all_modes
            else f"{config.get('dataset', 'dataset').lower()}_physics_prior"
        ),
        "config_path": str(Path(args.config).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "scene_ids": scene_ids,
        "tx_ids": tx_ids,
        "selected_diffraction_method": "compare-all" if args.compare_all_modes else diffraction_method,
        "compare_all_modes": args.compare_all_modes,
        "executed_methods": modes,
        "fonts": {"chinese": chinese_font, "western": western_font},
        "total_runtime_seconds": time.perf_counter() - started,
        "samples": records,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] runtime={summary['total_runtime_seconds']:.2f}s output={output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
