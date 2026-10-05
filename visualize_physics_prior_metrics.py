"""
用途:
    读取已经生成的跨数据集 Physics Prior结果，只构造并评估
  P_NLoS = FSPL + I_NLoS * L_diff；建筑内部固定使用配置中的最低signed pathgain。

输入:
  --experiment-root: run_physics_prior.py --compare-all-modes生成的实验目录；
  --data-root: 外部RadioMap3DSeer数据集根目录，仅用于读取建筑高度图和Tx位置；
  --config: 独立物理配置JSON，用于读取固定标签范围和地图尺寸；
  --scene-ids: 逗号分隔的scene id，默认0,1,2；
  --tx-ids: 逗号分隔的Tx id，默认0；
  --output-dir: 指标CSV、JSON、PNG、PDF的输出目录。

输出:
  metrics_summary.csv/json: 每个scene/Tx、每种方法、全图和NLoS区域的MSE、RMSE、MAE、
    NMSE、PSNR、SSIM和R^2；RMSE、MAE同时提供dB域数值，NMSE和PSNR采用固定标签范围归一化域；
  physics_prior_NLoS_comparison.png/pdf: 一张总图，包含建筑高度、GT、FSPL、
  FSPL+NLoS-Single、FSPL+NLoS-OWR-RD、FSPL+NLoS-Deygout、FSPL+NLoS-OWR-CD和聚合指标。
  figure_manifest.json: 总图面板和指标定义，便于后续数据驱动实验复用。

运行示例(macOS/Linux):
  uv run visualize_physics_prior_metrics.py \
    --experiment-root runs/radiomap3dseer_physics_prior/3.5GHz_1m \
    --data-root /path/to/RadioMap3DSeer \
    --config configs/radiomap3dseer.json \
    --scene-ids 0,1,2 --tx-ids 0 \
    --output-dir runs/radiomap3dseer_physics_prior/3.5GHz_1m/metrics

运行示例(Windows CMD):
  uv run visualize_physics_prior_metrics.py ^
    --experiment-root runs\\radiomap3dseer_physics_prior\\3.5GHz_1m ^
    --data-root  G:/paper/radiomap3dseer ^
    --config configs\\radiomap3dseer.json ^
    --scene-ids 0,1,2 --tx-ids 0 ^
    --output-dir runs\\radiomap3dseer_physics_prior\\3.5GHz_1m\\metrics

指标语义:
  P_NLoS = FSPL + I_NLoS * L_diff，只在共同几何NLoS mask中加入diffraction；
  scope=all/nlos分别表示全图指标和NLoS子区域指标。SSIM只在完整地图上计算。
  建筑高度图大于0的像素始终设置为配置中的最低signed pathgain，当前为-162 dB；
  该操作是数据集建筑内部不可接收区域的固定后处理，不改变原始diffraction solver数组。

逻辑:
  物理结果统一读取signed PL/pathgain dB数组。所有方法共享同一Ground Truth、FSPL和
  各方法读取各自的几何LOS mask；none模式的输出作为纯FSPL，single、OWR-RD、Deygout和OWR-CD读取各自physics_prior，
  统一显式构造P_NLoS。
  图像使用配置中的固定pathgain范围，不做逐图min-max归一化。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from radiomap_physics import load_json, load_scene


MODES = ("none", "single", "owr-rd", "deygout", "owr-cd")
METHOD_LABELS = {
    "none": "FSPL",
    "single": "FSPL+NLoS-Single",
    "owr-rd": "FSPL+NLoS-OWR-RD",
    "deygout": "FSPL+NLoS-Deygout",
    "owr-cd": "FSPL+NLoS-OWR-CD",
}
METHOD_COLORS = {
    "none": "#666666",
    "single": "#E69F00",
    "owr-rd": "#0072B2",
    "deygout": "#009E73",
    "owr-cd": "#D55E00",
}


def _configure_fonts() -> tuple[str, str]:
    """配置西文Times New Roman；中文字体若存在则记录，否则使用可复核回退字体。"""

    installed = {font.name for font in font_manager.fontManager.ttflist}
    chinese = "SimSun" if "SimSun" in installed else "DejaVu Sans"
    western = "Times New Roman" if "Times New Roman" in installed else "DejaVu Serif"
    plt.rcParams["font.family"] = [western]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    return chinese, western


def _parse_ids(value: str) -> list[str]:
    """解析逗号分隔scene/Tx id，拒绝空列表。"""

    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one scene or Tx id is required.")
    return values


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """构造与surrogate.utils中SSIM相同参数的二维高斯窗口。"""

    coordinates = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel_1d = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return kernel_1d[:, None] @ kernel_1d[None, :]


def _convolve_zero_padded(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """使用零填充滑动窗口实现二维卷积，边界行为对齐surrogate的conv2d。"""

    pad = kernel.shape[0] // 2
    padded = np.pad(array, ((pad, pad), (pad, pad)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def _ssim_numpy(pred_norm: np.ndarray, target_norm: np.ndarray) -> float:
    """计算全图SSIM；输入已按固定pathgain范围归一化到[0,1]。"""

    kernel = _gaussian_kernel()
    mu_pred = _convolve_zero_padded(pred_norm, kernel)
    mu_target = _convolve_zero_padded(target_norm, kernel)
    sigma_pred = _convolve_zero_padded(pred_norm * pred_norm, kernel) - mu_pred**2
    sigma_target = _convolve_zero_padded(target_norm * target_norm, kernel) - mu_target**2
    covariance = _convolve_zero_padded(pred_norm * target_norm, kernel) - mu_pred * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * covariance + c2)
    denominator = (mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2)
    return float(np.mean(numerator / np.maximum(denominator, np.finfo(np.float64).eps)))


def _normalize(array: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """按跨样本固定dB范围归一化并截断到[0,1]，不执行逐图min-max。"""

    return np.clip((array.astype(np.float64) - db_min) / (db_max - db_min), 0.0, 1.0)


def _apply_building_interior_min_pathgain(
    predictions: dict[str, np.ndarray],
    building_mask: np.ndarray,
    db_min: float,
) -> dict[str, np.ndarray]:
    """将建筑内部Rx像素固定为配置最低signed pathgain；原始物理数组不被覆盖。"""

    return {
        mode: np.where(building_mask, db_min, array).astype(np.float64)
        for mode, array in predictions.items()
    }


def _metric_row(
    prediction_db: np.ndarray,
    target_db: np.ndarray,
    mask: np.ndarray,
    db_min: float,
    db_max: float,
    scope: str,
    sample_name: str,
    method: str,
) -> dict[str, Any]:
    """计算一个样本、一个P_NLoS方法和一个评价区域的统一指标行。"""

    prediction_values = prediction_db[mask].astype(np.float64)
    target_values = target_db[mask].astype(np.float64)
    if prediction_values.size == 0:
        return {
            "sample": sample_name,
            "method": method,
            "scope": scope,
            "pixel_count": 0,
            "mse_db2": float("nan"),
            "rmse_db": float("nan"),
            "mae_db": float("nan"),
            "mse_norm": float("nan"),
            "rmse_norm": float("nan"),
            "mae_norm": float("nan"),
            "nmse": float("nan"),
            "psnr_db": float("nan"),
            "ssim": float("nan"),
            "r2": float("nan"),
            "bias_db": float("nan"),
            "prediction_min_db": float("nan"),
            "prediction_below_ground_truth_min_ratio": float("nan"),
        }
    error_db = prediction_values - target_values
    prediction_norm = _normalize(prediction_values, db_min, db_max)
    target_norm = _normalize(target_values, db_min, db_max)
    error_norm = prediction_norm - target_norm
    mse_db2 = float(np.mean(error_db**2))
    mse_norm = float(np.mean(error_norm**2))
    target_energy_norm = float(np.sum(target_norm**2))
    target_mean_db = float(np.mean(target_values))
    ss_tot_db2 = float(np.sum((target_values - target_mean_db) ** 2))
    ss_res_db2 = float(np.sum(error_db**2))
    r2 = 0.0 if ss_tot_db2 == 0.0 else 1.0 - ss_res_db2 / ss_tot_db2
    ssim = float("nan")
    if scope == "all":
        ssim = _ssim_numpy(_normalize(prediction_db, db_min, db_max), _normalize(target_db, db_min, db_max))
    return {
        "sample": sample_name,
        "method": method,
        "scope": scope,
        "pixel_count": int(mask.sum()),
        "mse_db2": mse_db2,
        "rmse_db": float(np.sqrt(mse_db2)),
        "mae_db": float(np.mean(np.abs(error_db))),
        "mse_norm": mse_norm,
        "rmse_norm": float(np.sqrt(mse_norm)),
        "mae_norm": float(np.mean(np.abs(error_norm))),
        "nmse": float(np.sum(error_norm**2) / max(target_energy_norm, 1e-12)),
        "psnr_db": float("inf") if mse_norm == 0.0 else float(10.0 * np.log10(1.0 / mse_norm)),
        "ssim": ssim,
        "r2": float(r2),
        "bias_db": float(np.mean(error_db)),
        "prediction_min_db": float(np.min(prediction_values)),
        "prediction_below_ground_truth_min_ratio": float(np.mean(prediction_values < np.min(target_db))),
    }


def _load_sample(
    experiment_root: Path,
    data_root: Path,
    config: dict[str, Any],
    scene_id: str,
    tx_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, int, int]:
    """读取同一scene/Tx的GT、五种预测、各自LOS mask、高度图和Tx图像坐标。"""

    sample_dir = experiment_root / f"scene_{scene_id}_tx_{tx_id}"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing experiment sample directory: {sample_dir}")
    ground_truth = np.load(sample_dir / "ground_truth_pathgain_db.npy").astype(np.float64)
    predictions: dict[str, np.ndarray] = {}
    los_masks: dict[str, np.ndarray] = {}
    for mode in MODES:
        array_dir = sample_dir / "modes" / mode / "physics_arrays"
        prediction_name = "fspl_paper_pl_db.npy" if mode == "none" else "physics_prior_paper_pl_db.npy"
        predictions[mode] = np.load(array_dir / prediction_name).astype(np.float64)
        los_masks[mode] = np.load(array_dir / "los_mask.npy").astype(bool)
    scene = load_scene(data_root, scene_id, config)
    tx = scene.tx_records[int(tx_id)]
    tx_row = int(scene.height_map_m.shape[0] - 1 - tx.y_m)
    return predictions, los_masks, ground_truth, scene.height_map_m, tx_row, int(tx.x_m)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """将指标行写为可被后续数据驱动实验直接读取的CSV。"""

    fields = [
        "sample", "method", "scope", "pixel_count", "mse_db2", "rmse_db", "mae_db",
        "mse_norm", "rmse_norm", "mae_norm", "nmse", "psnr_db", "ssim", "r2",
        "bias_db", "prediction_min_db", "prediction_below_ground_truth_min_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """对每个P_NLoS方法和scope做macro-sample平均，图表使用该聚合口径。"""

    metric_names = [
        "mse_db2", "rmse_db", "mae_db", "mse_norm", "rmse_norm", "mae_norm", "nmse", "psnr_db", "ssim", "r2",
        "bias_db", "prediction_min_db", "prediction_below_ground_truth_min_ratio",
    ]
    aggregate: dict[tuple[str, str], dict[str, float]] = {}
    for method in MODES:
        for scope in ("all", "nlos"):
            selected = [
                row for row in rows
                if row["method"] == METHOD_LABELS[method]
                and row["scope"] == scope
            ]
            aggregate[(method, scope)] = {}
            for metric in metric_names:
                values = [row[metric] for row in selected if np.isfinite(row[metric])]
                aggregate[(method, scope)][metric] = float(np.mean(values)) if values else float("nan")
    return aggregate


def _imshow(ax, array: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = "viridis") -> None:
    """使用固定vmin/vmax、等比例坐标和统一色图绘制地图面板。"""

    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#eeeeee")
    image = ax.imshow(array, cmap=color_map, aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def _plot_metric_panel(ax, aggregate: dict[tuple[str, str], dict[str, float]], metric: str, title: str, ylabel: str) -> None:
    """绘制P_NLoS方法在全图上的聚合指标柱状图。"""

    x = np.arange(len(MODES), dtype=float)
    values = [aggregate[(mode, "all")][metric] for mode in MODES]
    bars = ax.bar(x, values, width=0.62, color=[METHOD_COLORS[mode] for mode in MODES], alpha=0.82)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x, [METHOD_LABELS[mode] for mode in MODES], rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="y", labelsize=8)
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        text = f"{value:.2f}" if abs(value) < 100 else f"{value:.1e}"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6, rotation=90)


def _render_figure(
    *,
    height_map: np.ndarray,
    ground_truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    nlos_predictions: dict[str, np.ndarray],
    tx_row: int,
    tx_col: int,
    aggregate: dict[tuple[str, str], dict[str, float]],
    db_min: float,
    db_max: float,
    output_stem: Path,
    figure_title: str,
) -> None:
    """生成唯一P_NLoS主线地图和全图聚合指标总图。"""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(24, 12))
    grid = fig.add_gridspec(2, 7, height_ratios=(1.0, 0.72), hspace=0.30, wspace=0.18)
    map_panels = [
        (height_map, "Building height (m)", 0.0, 19.8, "viridis"),
        (ground_truth, "Ground truth (dB)", db_min, db_max, "viridis"),
        (predictions["none"], "FSPL (dB)", db_min, db_max, "viridis"),
        (nlos_predictions["single"], "FSPL+NLoS-Single (dB)", db_min, db_max, "viridis"),
        (nlos_predictions["owr-rd"], "FSPL+NLoS-OWR-RD (dB)", db_min, db_max, "viridis"),
        (nlos_predictions["deygout"], "FSPL+NLoS-Deygout (dB)", db_min, db_max, "viridis"),
        (nlos_predictions["owr-cd"], "FSPL+NLoS-OWR-CD (dB)", db_min, db_max, "viridis"),
    ]
    for col_index, (array, title, vmin, vmax, cmap) in enumerate(map_panels):
        ax = fig.add_subplot(grid[0, col_index])
        _imshow(ax, array, title, vmin, vmax, cmap)
        if col_index >= 1:
            ax.plot(tx_col, tx_row, marker="+", color="#D55E00", markersize=8, markeredgewidth=1.5)
    metric_specs = [
        ("rmse_db", "RMSE (dB)", "dB"),
        ("mae_db", "MAE (dB)", "dB"),
        ("r2", r"$R^2$", "score"),
        ("nmse", "NMSE", "normalized"),
        ("psnr_db", "PSNR", "dB"),
        ("ssim", "SSIM (full map)", "score"),
    ]
    for index, (metric, title, ylabel) in enumerate(metric_specs):
        ax = fig.add_subplot(grid[1, index])
        _plot_metric_panel(ax, aggregate, metric, title, ylabel)
    fig.suptitle(figure_title, fontsize=16)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.925, bottom=0.08, hspace=0.34, wspace=0.26)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    """解析参数、构造唯一P_NLoS主线、计算两种scope指标并生成可追溯图表。"""

    parser = argparse.ArgumentParser(description="Evaluate the P_NLoS Physics Prior maps and metrics.")
    parser.add_argument("--experiment-root", required=True, help="Existing compare-all-modes output directory.")
    parser.add_argument("--data-root", required=True, help="External RadioMap3DSeer dataset root.")
    parser.add_argument("--config", default="configs/radiomap3dseer.json", help="Independent dataset configuration JSON.")
    parser.add_argument("--scene-ids", default="0,1,2", help="Comma-separated scene IDs.")
    parser.add_argument("--tx-ids", default="0", help="Comma-separated Tx IDs.")
    parser.add_argument("--output-dir", required=True, help="Directory for metric tables and figures.")
    args = parser.parse_args()

    chinese_font, western_font = _configure_fonts()
    experiment_root = Path(args.experiment_root)
    data_root = Path(args.data_root)
    config = load_json(Path(args.config))
    scene_ids = _parse_ids(args.scene_ids)
    tx_ids = _parse_ids(args.tx_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_min, db_max = map(float, config["dataset_labels"]["simulation_pathgain_db_range"])
    rows: list[dict[str, Any]] = []
    representative: tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, int, int, str] | None = None
    warnings: list[str] = []
    for scene_id in scene_ids:
        for tx_id in tx_ids:
            raw_predictions, los_masks, ground_truth, height_map, tx_row, tx_col = _load_sample(
                experiment_root, data_root, config, scene_id, tx_id
            )
            sample_name = f"scene_{scene_id}_tx_{tx_id}"
            all_mask = np.ones_like(height_map, dtype=bool)
            building_mask = height_map > 0.0
            predictions = _apply_building_interior_min_pathgain(
                raw_predictions,
                building_mask,
                db_min,
            )
            nlos_predictions = {
                mode: predictions["none"] + np.where(~los_masks[mode], predictions[mode] - predictions["none"], 0.0)
                for mode in MODES
            }
            if representative is None:
                representative = (predictions, nlos_predictions, ground_truth, los_masks["owr-cd"], height_map, tx_row, tx_col, sample_name)
            for mode in MODES:
                application_prediction = nlos_predictions[mode]
                rows.append(_metric_row(application_prediction, ground_truth, all_mask, db_min, db_max, "all", sample_name, METHOD_LABELS[mode]))
                nlos_mask = ~los_masks[mode]
                rows.append(_metric_row(application_prediction, ground_truth, nlos_mask, db_min, db_max, "nlos", sample_name, METHOD_LABELS[mode]))
                if nlos_mask.mean() == 0.0:
                    warnings.append(f"{sample_name}/{mode}: no NLoS pixels")
    if representative is None:
        raise ValueError("No representative sample was loaded.")
    aggregate_by_label: dict[tuple[str, str], dict[str, float]] = {}
    raw_aggregate = _aggregate_rows(rows)
    for key, value in raw_aggregate.items():
        aggregate_by_label[(METHOD_LABELS[key[0]], key[1])] = value
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "metrics_summary.csv")
    summary = {
        "dataset": config.get("dataset", "unknown"),
        "application_definitions": {
            "nlos_region": "P_NLoS = FSPL + I_NLoS * L_diff",
        },
        "building_interior_rule": "height_map_m > 0 pixels are fixed to the configuration minimum signed pathgain",
        "building_min_signed_pathgain_db": db_min,
        "scope_definitions": {"all": "all pixels", "nlos": "pixels where the method-specific geometry LOS mask is false"},
        "metric_units": {
            "rmse_db": "dB signed PL/pathgain domain",
            "mae_db": "dB signed PL/pathgain domain",
            "mse_norm": "fixed-range normalized domain",
            "nmse": "normalized target energy denominator, matching surrogate metric convention",
            "psnr_db": "fixed-range normalized domain",
            "ssim": "full-map only; NLoS masked SSIM is not defined for this report",
        },
        "fixed_pathgain_range_db": [db_min, db_max],
        "aggregation": "macro mean across selected scene/Tx samples; raw rows remain in metrics_summary.csv",
        "fonts": {"chinese": chinese_font, "western": western_font},
        "warnings": warnings,
        "aggregate_metrics": {
            f"{method}_{scope}": values
            for (method, scope), values in aggregate_by_label.items()
        },
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    representative_predictions, representative_nlos_predictions, representative_gt, representative_nlos, representative_height, representative_tx_row, representative_tx_col, representative_name = representative
    _render_figure(
        height_map=representative_height,
        ground_truth=representative_gt,
        predictions=representative_predictions,
        nlos_predictions=representative_nlos_predictions,
        tx_row=representative_tx_row,
        tx_col=representative_tx_col,
        aggregate=raw_aggregate,
        db_min=db_min,
        db_max=db_max,
        output_stem=output_dir / "physics_prior_NLoS_comparison",
        figure_title=(
            f"{config.get('dataset', 'Dataset')} Physics Prior: "
            f"P_NLoS = FSPL + I_NLoS * L_diff ({representative_name})"
        ),
    )
    figure_manifest = {
        "main_figure": "physics_prior_NLoS_comparison.png",
        "main_figure_pdf": "physics_prior_NLoS_comparison.pdf",
        "representative_sample": representative_name,
        "map_panels": ["Building height", "Ground truth", "FSPL", "FSPL+NLoS-Single", "FSPL+NLoS-OWR-RD", "FSPL+NLoS-Deygout", "FSPL+NLoS-OWR-CD"],
        "metric_panels": ["RMSE", "MAE", "R^2", "NMSE", "PSNR", "SSIM"],
        "metric_definition": "All map and NLoS-subregion metrics use P_NLoS; no direct all-pixel diffraction application is evaluated.",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(figure_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] samples={len(scene_ids) * len(tx_ids)} rows={len(rows)} output={output_dir.resolve()}")
    print(f"[figure] {output_dir / 'physics_prior_NLoS_comparison.png'}")
    print(f"[metrics] {output_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
