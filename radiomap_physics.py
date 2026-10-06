"""
RadioMap3DSeer确定性Physics Prior核心实现。

输入:
  - RadioMap3DSeer根目录，内部包含polygon/buildings、antenna和gain；
  - scene_id与tx_id；
  - configs/radiomap3dseer.json中的频率、分辨率、Tx/Rx高度和标签范围。

输出:
  - 256x256的signed FSPL、可选signed diffraction loss（none/single/OWR-RD/OWR-CD/Deygout）、
    nu_max、dominant obstruction、LOS/NLOS与signed FSPL+diffraction数组；
  - 可按任意Tx-Rx重建传播高度剖面，并返回候选障碍、E*、h、d1、d2与nu_max。

直接执行:
  本文件无命令行入口。请在本目录运行:
  uv run run_physics_prior.py --data-root ../dataset --scene-ids 0,1,2 --tx-ids 0

参数与单位:
  height_map: [H,W]，建筑屋顶绝对高度，单位m；0表示非建筑。
  tx: x、y为地图米制坐标，z为绝对高度，单位m。
  rx_height_m: Rx绝对高度，单位m。
  frequency_hz: 载波频率，单位Hz。
  path_sampling_step_m: Tx-Rx水平路径的最大采样间隔，默认1m。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TxRecord:
    """一个发射机记录；x/y为米制地图坐标，z为绝对高度，全部单位为m。"""

    x_m: float
    y_m: float
    z_m: float
    rooftop_height_m: float | None = None


def _data_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the optional data-layout section, preserving the old 3D schema."""

    return config.get("data", {})


def _format_data_template(template: str, scene_id: str, tx_id: str | None = None) -> str:
    """Format a repository-relative dataset template without accepting absolute paths in JSON."""

    values = {"scene_id": scene_id, "tx_id": "" if tx_id is None else tx_id}
    return template.format(**values)


def _load_image_array(path: Path) -> np.ndarray:
    """Load a grayscale image as a 2D uint8 array."""

    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _load_zip_image_array(zip_path: Path, member_name: str) -> np.ndarray:
    """Load one grayscale image from a ZIP archive without extracting the archive."""

    with zipfile.ZipFile(zip_path) as archive:
        try:
            payload = archive.read(member_name)
        except KeyError as exc:
            raise FileNotFoundError(f"Missing ZIP member {member_name!r} in {zip_path}") from exc
    return np.asarray(Image.open(io.BytesIO(payload)).convert("L"), dtype=np.uint8)


def _gray_to_signed_db(gray: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Decode a configured grayscale label interval into signed PL/pathgain dB."""

    mapping = config["dataset_labels"]["png_gray_mapping"]
    low, high = map(float, mapping["pathgain_db_range"])
    return (low + gray.astype(np.float32) / 255.0 * (high - low)).astype(np.float32)


@dataclass
class SceneData:
    """一个scene的建筑高度栅格、polygon记录和Tx列表。"""

    scene_id: str
    height_map_m: np.ndarray
    polygons: list[dict[str, Any]]
    tx_records: list[TxRecord]
    # Optional finite Rx locations used to generate an interpolated pmap.
    # This is evaluation metadata only and never changes the physics solver.
    rx_observation_mask: np.ndarray | None = None


@dataclass
class PropagationProfile:
    """单个Tx-Rx传播剖面及dominant knife-edge中间量。"""

    rx_row: int
    rx_col: int
    s_m: np.ndarray
    building_height_m: np.ndarray
    los_height_m: np.ndarray
    candidate_mask: np.ndarray
    candidate_h_m: np.ndarray
    candidate_d1_m: np.ndarray
    candidate_d2_m: np.ndarray
    candidate_nu: np.ndarray
    dominant_index: int
    dominant_row: int
    dominant_col: int
    dominant_height_m: float
    dominant_h_m: float
    dominant_d1_m: float
    dominant_d2_m: float
    nu_max: float
    diffraction_loss_db: float
    has_obstruction: bool


@dataclass
class PhysicsMaps:
    """一个Tx对应的所有像素物理量；数组形状均为[H,W]。"""

    fspl_db: np.ndarray
    diffraction_loss_db: np.ndarray
    nu_max: np.ndarray
    dominant_row: np.ndarray
    dominant_col: np.ndarray
    dominant_height_m: np.ndarray
    dominant_h_m: np.ndarray
    dominant_d1_m: np.ndarray
    dominant_d2_m: np.ndarray
    los_mask: np.ndarray
    physics_prior_db: np.ndarray
    edge_count: np.ndarray
    corner_count: np.ndarray
    unresolved_nlos_mask: np.ndarray
    cd_validity_mask: np.ndarray
    first_blocking_building_id: np.ndarray
    resolved_method_map: np.ndarray
    fallback_to_rd_mask: np.ndarray
    mode: str = "single"
    # Pixels where the diffraction solver was intentionally evaluated.  This
    # is a sparse Rx mask for image-mask datasets, or all True for the normal
    # dense-grid mode.  Unobserved pixels are not zero-diffraction evidence.
    observation_mask: np.ndarray | None = None


@dataclass(frozen=True)
class DiffractionEndpoint:
    """递归区间端点；row/col为图像坐标，z_m为该端点的绝对高度。"""

    row: float
    col: float
    z_m: float


@dataclass
class IntervalProfile:
    """任意A-B区间重新建立LOS后得到的候选障碍数组。"""

    a: DiffractionEndpoint
    b: DiffractionEndpoint
    rows: np.ndarray
    cols: np.ndarray
    s_m: np.ndarray
    building_height_m: np.ndarray
    los_height_m: np.ndarray
    h_m: np.ndarray
    d1_m: np.ndarray
    d2_m: np.ndarray
    nu: np.ndarray
    candidate_mask: np.ndarray


@dataclass
class DiffractionEvent:
    """一个递归节点选出的dominant edge及其重新计算的物理量。"""

    depth: int
    a: DiffractionEndpoint
    b: DiffractionEndpoint
    edge: DiffractionEndpoint
    h_m: float
    d1_m: float
    d2_m: float
    nu: float
    loss_db: float
    component_id: int | None = None
    event_type: str = "rooftop"
    wedge_angle_rad: float | None = None
    incident_angle_rad: float | None = None
    diffraction_angle_rad: float | None = None
    corner_model: str | None = None
    incident_direction_rc: tuple[float, float] | None = None
    outgoing_direction_rc: tuple[float, float] | None = None


@dataclass
class DiffractionSolution:
    """一个Rx在指定diffraction mode下的损耗和完整递归记录。"""

    mode: str
    loss_db: float
    events: list[DiffractionEvent]
    termination: str
    is_los: bool = True
    corner_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    first_blocking_component_id: int | None = None
    dispatch_method: str | None = None
    fallback_from: str | None = None


def load_json(path: Path) -> Any:
    """读取UTF-8 JSON；path必须是当前实验的数据或配置文件。"""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rasterize_polygons(
    polygons_raw: list[Any],
    map_size_px: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """将[x,y] polygon和绝对屋顶高度栅格化为[H,W]米制高度图。

    输入polygon坐标单位为m且在本数据集内与1m像素网格对齐；输出对重叠建筑取较高屋顶，
    以避免JSON顺序影响物理先验。Pillow只用于生成建筑占用mask，不参与高度量化。
    """

    height_px, width_px = map_size_px
    height_map = np.zeros((height_px, width_px), dtype=np.float32)
    normalized: list[dict[str, Any]] = []
    for item in polygons_raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each building polygon entry must be [coordinates, [height]].")
        coords_raw, height_raw = item
        if not coords_raw or not isinstance(coords_raw[0], list):
            raise ValueError("Building polygon coordinates are empty or malformed.")
        height_m = float(height_raw[0] if isinstance(height_raw, list) else height_raw)
        # 数据的polygon坐标以地图底边为y=0；Pillow/NumPy图像以顶部为row=0，必须翻转y。
        coords = [(int(round(float(x))), height_px - 1 - int(round(float(y)))) for x, y in coords_raw]
        mask_image = Image.new("1", (width_px, height_px), 0)
        ImageDraw.Draw(mask_image).polygon(coords, fill=1)
        mask = np.asarray(mask_image, dtype=bool)
        height_map[mask] = np.maximum(height_map[mask], height_m)
        normalized.append({"coordinates_xy": coords, "height_m": height_m})
    return height_map, normalized


def _rasterize_mask_polygons(
    polygons_raw: list[Any],
    map_size_px: tuple[int, int],
    building_height_m: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rasterize 2D polygon-only data with an explicitly configured proxy height."""

    height_px, width_px = map_size_px
    height_map = np.zeros((height_px, width_px), dtype=np.float32)
    normalized: list[dict[str, Any]] = []
    for item in polygons_raw:
        coords_raw = item[0] if isinstance(item, list) and len(item) == 2 and isinstance(item[0], list) and item[0] and isinstance(item[0][0], list) else item
        if not coords_raw or not isinstance(coords_raw[0], list):
            raise ValueError("Building polygon coordinates are empty or malformed.")
        coords = [(int(round(float(x))), height_px - 1 - int(round(float(y)))) for x, y in coords_raw]
        mask_image = Image.new("1", (width_px, height_px), 0)
        ImageDraw.Draw(mask_image).polygon(coords, fill=1)
        height_map[np.asarray(mask_image, dtype=bool)] = float(building_height_m)
        normalized.append({"coordinates_xy": coords, "height_m": float(building_height_m)})
    return height_map, normalized


def _read_rm_directory_entries(root: Path, scene_id: str, config: dict[str, Any]) -> list[tuple[Path, float, float]]:
    """Read UrbanRadio3D labels from an extracted directory.

    UrbanRadio3D names files as ``scene_Xx_Yy.png``, but the image coordinate
    convention is ``row=255-X, col=Y``.  Therefore the physical map
    coordinates used by this repository are ``x=Y`` and ``y=X``.  Keeping
    this conversion here prevents the filename convention from leaking into
    the propagation solver.
    """

    data = _data_config(config)
    label_dir = root / str(data["label_directory_template"]).format(rx_height_level=data["rx_height_level"])
    pattern = re.compile(
        rf"^{re.escape(str(scene_id))}_X(?P<file_x>-?\d+)_Y(?P<file_y>-?\d+)\.png$"
    )
    entries: list[tuple[Path, float, float]] = []
    for path in label_dir.glob(f"{scene_id}_X*_Y*.png"):
        match = pattern.match(path.name)
        if match:
            file_x = float(match.group("file_x"))
            file_y = float(match.group("file_y"))
            entries.append((path, file_y, file_x))
    entries.sort(key=lambda item: (item[2], item[1], item[0].name))
    if not entries:
        raise FileNotFoundError(f"No UrbanRadio3D labels for scene={scene_id} in {label_dir}")
    return entries


def _read_rm_zip_entries(root: Path, scene_id: str, config: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Return sorted (ZIP member, x, y) transmitter entries for legacy archives."""

    data = _data_config(config)
    archive_path = root / str(data["label_archive_template"]).format(rx_height_level=data["rx_height_level"])
    member_prefix = str(data.get("label_member_prefix_template", "h{rx_height_level}/")).format(
        rx_height_level=data["rx_height_level"]
    )
    pattern = re.compile(
        rf"^{re.escape(member_prefix)}{re.escape(str(scene_id))}_X(?P<file_x>-?\d+)_Y(?P<file_y>-?\d+)\.png$"
    )
    with zipfile.ZipFile(archive_path) as archive:
        entries: list[tuple[str, float, float]] = []
        for name in archive.namelist():
            match = pattern.match(name)
            if match:
                # Keep legacy support consistent with the extracted layout:
                # filename X is the image-row axis and filename Y is image-col.
                entries.append((name, float(match.group("file_y")), float(match.group("file_x"))))
    entries.sort(key=lambda item: (item[2], item[1], item[0]))
    if not entries:
        raise FileNotFoundError(f"No UrbanRadio3D labels for scene={scene_id} in {archive_path}")
    return entries


def load_scene(data_root: str | Path, scene_id: str, config: dict[str, Any]) -> SceneData:
    """读取一个scene的polygon绝对高度与antenna JSON，并核对尺寸。

    antenna JSON使用底部原点[x,y,z]；代码保留TxRecord中的世界坐标，访问图像数组时转换为
    [row=(H-1)-y,col=x]。Tx的z同时保留为数据集记录，运行时会检查它与polygon屋顶+3m的一致性。
    """

    root = Path(data_root)
    height_px, width_px = map(int, config["scene"]["map_size_px"])
    data = _data_config(config)
    data_format = data.get("format", "polygon_height_json")

    if data_format == "polygon_height_json":
        polygon_path = root / str(data.get("polygon_path_template", "polygon/buildings/{scene_id}.json")).format(scene_id=scene_id)
        antenna_path = root / str(data.get("antenna_path_template", "antenna/{scene_id}.json")).format(scene_id=scene_id)
        if not polygon_path.exists() or not antenna_path.exists():
            raise FileNotFoundError(f"Missing scene files: {polygon_path}, {antenna_path}")
        height_map, polygons = _rasterize_polygons(load_json(polygon_path), (height_px, width_px))
        tx_records: list[TxRecord] = []
        tx_offset_m = float(config["scene"]["tx_height_offset_above_rooftop_m"])
        polygon_heights = np.asarray([float(item["height_m"]) for item in polygons], dtype=np.float64)
        for raw in load_json(antenna_path):
            if len(raw) < 3:
                raise ValueError(f"Malformed Tx record in {antenna_path}: {raw}")
            tx_z_m = float(raw[2])
            rooftop_height_m = tx_z_m - tx_offset_m
            nearest_error_m = float(np.min(np.abs(polygon_heights - rooftop_height_m)))
            if nearest_error_m > 1e-3:
                raise ValueError(
                    f"Tx height audit failed for scene={scene_id}, tx={raw}: "
                    f"z-offset={rooftop_height_m:.6f} m is not a polygon height (error={nearest_error_m:.6f} m)."
                )
            tx_records.append(TxRecord(float(raw[0]), float(raw[1]), tx_z_m, rooftop_height_m))
        return SceneData(str(scene_id), height_map, polygons, tx_records)

    if data_format == "polygon_mask_json":
        polygon_path = root / _format_data_template(str(data["polygon_path_template"]), str(scene_id))
        antenna_path = root / _format_data_template(str(data["antenna_path_template"]), str(scene_id))
        if not polygon_path.exists() or not antenna_path.exists():
            raise FileNotFoundError(f"Missing scene files: {polygon_path}, {antenna_path}")
        proxy_height = float(config["scene"]["building_height_m"][1])
        height_map, polygons = _rasterize_mask_polygons(load_json(polygon_path), (height_px, width_px), proxy_height)
        tx_height_m = float(config["scene"]["tx_height_m"])
        tx_records = [TxRecord(float(raw[0]), float(raw[1]), tx_height_m, proxy_height) for raw in load_json(antenna_path)]
        return SceneData(str(scene_id), height_map, polygons, tx_records)

    if data_format == "image_mask":
        map_path = root / _format_data_template(str(data["building_map_template"]), str(scene_id))
        tx_path = root / _format_data_template(str(data["tx_map_template"]), str(scene_id))
        if not map_path.exists() or not tx_path.exists():
            raise FileNotFoundError(f"Missing image-mask scene files: {map_path}, {tx_path}")
        building_gray = _load_image_array(map_path)
        tx_gray = _load_image_array(tx_path)
        if building_gray.shape != (height_px, width_px) or tx_gray.shape != (height_px, width_px):
            raise ValueError(f"Image-mask scene {scene_id} has unexpected shape: map={building_gray.shape}, tx={tx_gray.shape}")
        building_height_m = float(config["scene"]["building_height_m"][1])
        threshold = float(data.get("building_mask_threshold", 1))
        height_map = np.where(building_gray >= threshold, building_height_m, 0.0).astype(np.float32)
        tx_pixels = np.argwhere(tx_gray >= float(data.get("tx_mask_threshold", 1)))
        if tx_pixels.size == 0:
            raise ValueError(f"No Tx pixel found in {tx_path}")
        tx_row, tx_col = tx_pixels.astype(np.float64).mean(axis=0)
        tx_height_m = float(config["scene"]["tx_height_m"])
        tx_records = [TxRecord(float(tx_col), float(height_px - 1 - tx_row), tx_height_m, building_height_m)]
        rx_mask = None
        rx_template = data.get("rx_map_template")
        if rx_template:
            rx_path = root / _format_data_template(str(rx_template), str(scene_id))
            if rx_path.exists():
                rx_gray = _load_image_array(rx_path)
                if rx_gray.shape != (height_px, width_px):
                    raise ValueError(f"Image-mask Rx scene {scene_id} has unexpected shape: {rx_gray.shape}")
                rx_mask = rx_gray >= float(data.get("rx_mask_threshold", 1))
        return SceneData(str(scene_id), height_map, [], tx_records, rx_mask)

    if data_format == "urbanradio3d_directory":
        polygon_path = root / _format_data_template(str(data["building_path_template"]), str(scene_id))
        if not polygon_path.exists():
            raise FileNotFoundError(f"Missing UrbanRadio3D building file: {polygon_path}")
        height_map, polygons = _rasterize_polygons(load_json(polygon_path), (height_px, width_px))
        entries = _read_rm_directory_entries(root, str(scene_id), config)
        tx_height_m = float(config["scene"]["tx_height_m"])
        tx_records = [TxRecord(x, y, tx_height_m, None) for _, x, y in entries]
        return SceneData(str(scene_id), height_map, polygons, tx_records)

    if data_format == "rm_data_zip":
        building_archive = root / str(data["building_archive"])
        building_member = str(data["building_member_template"]).format(scene_id=scene_id)
        building_gray = _load_zip_image_array(building_archive, building_member)
        if building_gray.shape != (height_px, width_px):
            raise ValueError(f"UrbanRadio3D building map {building_member} has shape {building_gray.shape}")
        height_low, height_high = map(float, config["scene"]["building_height_m"])
        height_map = (height_low + building_gray.astype(np.float32) / 255.0 * (height_high - height_low)).astype(np.float32)
        height_map[building_gray == 0] = 0.0
        entries = _read_rm_zip_entries(root, str(scene_id), config)
        tx_height_m = float(config["scene"]["tx_height_m"])
        tx_records = [TxRecord(x, y, tx_height_m, None) for _, x, y in entries]
        return SceneData(str(scene_id), height_map, [], tx_records)

    raise ValueError(f"Unsupported data.format={data_format!r}")


def load_ground_truth_path_gain_db(data_root: str | Path, scene_id: str, tx_id: str, config: dict[str, Any]) -> np.ndarray:
    """读取gain PNG并按论文/数据集约定返回负值path-gain/PL，单位dB。

    论文定义 ``PL = (P_Rx)_dB - (P_Tx)_dB``，所以该数据集的标签范围是负值；
    灰度255代表较高path gain，对应映射区间的高端 ``-75 dB``。
    """

    root = Path(data_root)
    data = _data_config(config)
    data_format = data.get("format", "polygon_height_json")
    if data_format == "urbanradio3d_directory":
        entries = _read_rm_directory_entries(root, str(scene_id), config)
        index = int(tx_id)
        if not 0 <= index < len(entries):
            raise IndexError(f"Tx id {tx_id} out of range for UrbanRadio3D scene={scene_id}; count={len(entries)}")
        return _gray_to_signed_db(_load_image_array(entries[index][0]), config)
    if data_format == "rm_data_zip":
        entries = _read_rm_zip_entries(root, str(scene_id), config)
        index = int(tx_id)
        if not 0 <= index < len(entries):
            raise IndexError(f"Tx id {tx_id} out of range for UrbanRadio3D scene={scene_id}; count={len(entries)}")
        archive_path = root / str(data["label_archive_template"]).format(rx_height_level=data["rx_height_level"])
        gray = _load_zip_image_array(archive_path, entries[index][0]).astype(np.float32)
        return _gray_to_signed_db(gray, config)
    if data_format == "image_mask":
        path = root / _format_data_template(str(data["ground_truth_template"]), str(scene_id), str(tx_id))
    else:
        path = root / _format_data_template(
            str(data.get("ground_truth_template", "gain/{scene_id}_{tx_id}.png")), str(scene_id), str(tx_id)
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return _gray_to_signed_db(_load_image_array(path), config)


def load_ground_truth_path_loss_db(data_root: str | Path, scene_id: str, tx_id: str, config: dict[str, Any]) -> np.ndarray:
    """读取ground truth并返回论文约定的负值PL/pathgain，单位dB。"""

    return load_ground_truth_path_gain_db(data_root, scene_id, tx_id, config)


def load_ground_truth_path_loss_magnitude_db(
    data_root: str | Path,
    scene_id: str,
    tx_id: str,
    config: dict[str, Any],
) -> np.ndarray:
    """读取ground truth并返回正的损耗幅度，作为论文负值PL的绝对值。"""

    return -load_ground_truth_path_loss_db(data_root, scene_id, tx_id, config)


def loss_magnitude_to_paper_pl_db(loss_magnitude_db: np.ndarray | float) -> np.ndarray | float:
    """将正损耗幅度转换为论文约定的负值PL/pathgain。"""

    converted = -np.asarray(loss_magnitude_db)
    return float(converted) if converted.ndim == 0 else converted


def free_space_path_loss_db(
    height_map_m: np.ndarray,
    tx: TxRecord,
    frequency_hz: float,
    rx_height_m: float | np.ndarray,
    resolution_m: float,
    speed_of_light_m_per_s: float = 299792458.0,
) -> np.ndarray:
    """按三维Tx-Rx欧氏距离计算每个像素的signed FSPL，返回负dB的[H,W]数组。

    height_map_m只用于检查输入尺寸。rx_height_m可以是所有像素共享的绝对高度，
    也可以是与地图同形状的逐像素绝对Rx高度（例如PPData5D的相对terrain高度层）。
    距离为sqrt(dx²+dy²+dz²)。
    """

    height_map_m = np.asarray(height_map_m, dtype=np.float64)
    rows, cols = np.indices(height_map_m.shape, dtype=np.float64)
    tx_row = float(height_map_m.shape[0] - 1) - tx.y_m
    dx_m = (cols - tx.x_m) * resolution_m
    dy_m = (rows - tx_row) * resolution_m
    rx_height_array = np.asarray(rx_height_m, dtype=np.float64)
    if rx_height_array.ndim == 0:
        dz_m = float(rx_height_array) - tx.z_m
    elif rx_height_array.shape == height_map_m.shape:
        dz_m = rx_height_array - tx.z_m
    else:
        raise ValueError(
            f"rx_height_m must be scalar or shape {height_map_m.shape}, got {rx_height_array.shape}"
        )
    distance_m = np.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m)
    # At a 2D proxy Tx/Rx pixel both heights may be equal, making d=0.  The
    # continuous FSPL equation is undefined there; use one declared pixel as
    # the deterministic near-field floor so every output map remains finite.
    distance_m = np.maximum(distance_m, float(resolution_m))
    wavelength_m = speed_of_light_m_per_s / float(frequency_hz)
    with np.errstate(divide="ignore"):
        fspl_loss_magnitude_db = 20.0 * np.log10(4.0 * np.pi * distance_m / wavelength_m)
    return (-fspl_loss_magnitude_db).astype(np.float32)


def free_space_path_loss_magnitude_db(
    height_map_m: np.ndarray,
    tx: TxRecord,
    frequency_hz: float,
    rx_height_m: float | np.ndarray,
    resolution_m: float,
    speed_of_light_m_per_s: float = 299792458.0,
) -> np.ndarray:
    """返回FSPL损耗幅度的正值版本；主PhysicsMaps使用论文负值约定。"""

    return -free_space_path_loss_db(
        height_map_m,
        tx,
        frequency_hz,
        rx_height_m,
        resolution_m,
        speed_of_light_m_per_s,
    )


def _knife_edge_loss_db(nu: float) -> float:
    """按用户给定的单刀刃分段公式计算一个候选的diffraction loss，单位dB。"""

    if not np.isfinite(nu) or nu <= -0.78:
        return 0.0
    return float(6.9 + 20.0 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1.0) + nu - 0.1))


DIFFRACTION_MODES = ("none", "single", "owr-rd", "deygout", "owr-cd")
DIFFRACTION_METHODS = (*DIFFRACTION_MODES, "auto")


def knife_edge_loss_db(nu: float) -> float:
    """公开统一J(nu)函数，供四种mode和测试共同调用。"""

    return _knife_edge_loss_db(float(nu))


def validate_diffraction_mode(mode: str) -> str:
    """校验命令行diffraction mode，避免静默落回另一种物理模型。"""

    if mode not in DIFFRACTION_METHODS:
        raise ValueError(f"Unsupported diffraction mode {mode!r}; choose one of {DIFFRACTION_METHODS}.")
    return mode


def _endpoint_from_tx(height_map_m: np.ndarray, tx: TxRecord) -> DiffractionEndpoint:
    """把底部原点Tx世界坐标转换为图像row/col端点，保留绝对Tx高度。"""

    return DiffractionEndpoint(float(height_map_m.shape[0] - 1) - tx.y_m, tx.x_m, tx.z_m)


def _building_component_labels(height_map_m: np.ndarray) -> np.ndarray:
    """为连续建筑屋顶分配8邻域连通分量ID。

    single仍使用完整候选profile；递归子区间使用此标签避免把同一栋平坦屋顶
    的相邻栅格反复当作新刀刃，否则一个物理建筑会产生与其像素数量成正比的
    非物理损耗。
    """

    occupied = np.asarray(height_map_m) > 0.0
    labels = np.full(occupied.shape, -1, dtype=np.int32)
    height_px, width_px = occupied.shape
    component_id = 0
    for row in range(height_px):
        for col in range(width_px):
            if not occupied[row, col] or labels[row, col] >= 0:
                continue
            labels[row, col] = component_id
            stack = [(row, col)]
            while stack:
                current_row, current_col = stack.pop()
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        if delta_row == 0 and delta_col == 0:
                            continue
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if (
                            0 <= next_row < height_px
                            and 0 <= next_col < width_px
                            and occupied[next_row, next_col]
                            and labels[next_row, next_col] < 0
                        ):
                            labels[next_row, next_col] = component_id
                            stack.append((next_row, next_col))
            component_id += 1
    return labels


def _rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Deterministically simplify an ordered 2-D contour with Ramer-Douglas-Peucker."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 3:
        return points
    start = points[0]
    end = points[-1]
    segment = end - start
    segment_norm = float(np.linalg.norm(segment))
    if segment_norm == 0.0:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        # Compute the 2-D scalar cross product explicitly. NumPy 2.0
        # removed the implicit 2-D vector cross-product result from
        # ``np.cross``; contour coordinates are two-dimensional here.
        offsets = points - start
        cross_z = segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]
        distances = np.abs(cross_z) / segment_norm
    index = int(np.argmax(distances))
    if float(distances[index]) <= epsilon:
        return np.vstack((start, end))
    left = _rdp_simplify(points[: index + 1], epsilon)
    right = _rdp_simplify(points[index:], epsilon)
    return np.vstack((left[:-1], right))


def _raster_component_polygon(height_map_m: np.ndarray, component_labels: np.ndarray, component_id: int) -> np.ndarray:
    """Extract and simplify one raster component's outer boundary into polygon corners.

    The contour is built from exposed pixel-cell edges, not by treating every
    occupied pixel as a corner.  Coordinates are image ``(row, col)`` in pixel
    units, with half-pixel cell boundaries.
    """

    rows, cols = np.where(component_labels == component_id)
    occupied = {(int(row), int(col)) for row, col in zip(rows, cols)}
    directed_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row, col in occupied:
        if (row - 1, col) not in occupied:
            directed_edges.append(((row - 0.5, col - 0.5), (row - 0.5, col + 0.5)))
        if (row, col + 1) not in occupied:
            directed_edges.append(((row - 0.5, col + 0.5), (row + 0.5, col + 0.5)))
        if (row + 1, col) not in occupied:
            directed_edges.append(((row + 0.5, col + 0.5), (row + 0.5, col - 0.5)))
        if (row, col - 1) not in occupied:
            directed_edges.append(((row + 0.5, col - 0.5), (row - 0.5, col - 0.5)))
    if not directed_edges:
        return np.empty((0, 2), dtype=np.float64)
    outgoing: dict[tuple[float, float], list[int]] = {}
    for edge_index, (start, _end) in enumerate(directed_edges):
        outgoing.setdefault(start, []).append(edge_index)
    for edge_indices in outgoing.values():
        edge_indices.sort(key=lambda index: directed_edges[index][1])
    unused = set(range(len(directed_edges)))
    loops: list[np.ndarray] = []
    while unused:
        first = min(unused, key=lambda index: (directed_edges[index][0], directed_edges[index][1]))
        start = directed_edges[first][0]
        current_edge = first
        loop: list[tuple[float, float]] = []
        while current_edge in unused:
            unused.remove(current_edge)
            current_start, current_end = directed_edges[current_edge]
            loop.append(current_start)
            if current_end == start:
                break
            choices = [index for index in outgoing.get(current_end, []) if index in unused]
            if not choices:
                break
            current_edge = choices[0]
        if len(loop) >= 3:
            closed = np.asarray(loop + [loop[0]], dtype=np.float64)
            simplified = _rdp_simplify(closed, epsilon=0.75)
            if len(simplified) > 1 and np.allclose(simplified[0], simplified[-1]):
                simplified = simplified[:-1]
            if len(simplified) >= 3:
                loops.append(simplified)
    if not loops:
        return np.empty((0, 2), dtype=np.float64)
    return max(loops, key=lambda polygon: abs(_polygon_signed_area(polygon)))


def _extract_footprint_polygons(height_map_m: np.ndarray) -> list[np.ndarray]:
    """Extract one simplified footprint polygon per raster building component."""

    labels = _building_component_labels(height_map_m)
    return [
        polygon
        for component_id in range(int(labels.max()) + 1)
        for polygon in [_raster_component_polygon(height_map_m, labels, component_id)]
        if len(polygon) >= 3
    ]


def _polygon_signed_area(polygon: np.ndarray) -> float:
    """Return signed area for an image-coordinate ``(row, col)`` polygon."""

    polygon = np.asarray(polygon, dtype=np.float64)
    rows = polygon[:, 0]
    cols = polygon[:, 1]
    return float(0.5 * np.sum(cols * np.roll(rows, -1) - np.roll(cols, -1) * rows))


def _normalize_footprint_polygons(footprint_polygons: list[Any] | None) -> list[np.ndarray]:
    """Normalize scene polygons to image-coordinate ``(row, col)`` corner arrays."""

    if not footprint_polygons:
        return []
    normalized: list[np.ndarray] = []
    for item in footprint_polygons:
        if isinstance(item, dict):
            coordinates = item.get("coordinates_xy", [])
            # Rasterized scene polygons store (image-col, image-row).
            polygon = np.asarray([(float(y), float(x)) for x, y in coordinates], dtype=np.float64)
        else:
            polygon = np.asarray(item, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            continue
        if np.allclose(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        normalized.append(polygon)
    return normalized


def _polygon_contains_point(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test in image coordinates."""

    row, col = map(float, point)
    inside = False
    for index in range(len(polygon)):
        row_a, col_a = polygon[index]
        row_b, col_b = polygon[(index + 1) % len(polygon)]
        crosses = (row_a > row) != (row_b > row)
        if crosses:
            crossing_col = (col_b - col_a) * (row - row_a) / (row_b - row_a) + col_a
            if col < crossing_col:
                inside = not inside
    return inside


def _associate_footprints_with_components(
    footprint_polygons: list[np.ndarray],
    component_labels: np.ndarray,
) -> dict[int, list[np.ndarray]]:
    """Associate vector/raster footprints with the raster component they occupy."""

    height_px, width_px = component_labels.shape
    associated: dict[int, list[np.ndarray]] = {}
    for polygon in footprint_polygons:
        samples = [polygon.mean(axis=0), *polygon]
        labels: list[int] = []
        for row, col in samples:
            row_index = int(np.clip(round(float(row)), 0, height_px - 1))
            col_index = int(np.clip(round(float(col)), 0, width_px - 1))
            label = int(component_labels[row_index, col_index])
            if label >= 0:
                labels.append(label)
        if labels:
            component_id = max(set(labels), key=labels.count)
            associated.setdefault(component_id, []).append(polygon)
    return associated


def _build_interval_profile(
    height_map_m: np.ndarray,
    a: DiffractionEndpoint,
    b: DiffractionEndpoint,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    path_sampling_step_m: float,
    component_labels: np.ndarray | None = None,
    excluded_components: frozenset[int] = frozenset(),
) -> IntervalProfile:
    """为任意A-B递归区间建立新的reference LOS并重新计算所有h/d1/d2/nu。

    端点A/B从候选集合中排除；每次递归调用本函数都会重新以当前A/B的绝对高度生成LOS，
    因此OWR-RD和Deygout不会复用父区间的nu。b.z_m已在调用处传入；rx_height_m仅保留接口
    语义，供顶层Rx端点校验，递归子区间以b.z_m为准。
    """

    height_map_m = np.asarray(height_map_m, dtype=np.float64)
    height_px, width_px = height_map_m.shape
    delta_row = b.row - a.row
    delta_col = b.col - a.col
    horizontal_distance_m = np.hypot(delta_row, delta_col) * resolution_m
    empty = np.empty(0, dtype=np.float64)
    if horizontal_distance_m == 0.0:
        return IntervalProfile(a, b, np.empty(0, np.int64), np.empty(0, np.int64), empty, empty, empty, empty, empty, empty, empty, np.empty(0, bool))
    n_steps = max(1, int(np.ceil(horizontal_distance_m / path_sampling_step_m)))
    if n_steps <= 1:
        return IntervalProfile(a, b, np.empty(0, np.int64), np.empty(0, np.int64), empty, empty, empty, empty, empty, empty, empty, np.empty(0, bool))
    t = np.arange(1, n_steps, dtype=np.float64) / n_steps
    rows = np.rint(a.row + t * delta_row).astype(np.int64)
    cols = np.rint(a.col + t * delta_col).astype(np.int64)
    keep = np.concatenate(([True], (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])))
    rows, cols = rows[keep], cols[keep]
    valid_grid = (rows >= 0) & (rows < height_px) & (cols >= 0) & (cols < width_px)
    rows, cols = rows[valid_grid], cols[valid_grid]
    s_m = np.hypot((cols - a.col) * resolution_m, (rows - a.row) * resolution_m)
    building_height_m = height_map_m[rows, cols]
    los_height_m = a.z_m + np.clip(s_m / horizontal_distance_m, 0.0, 1.0) * (b.z_m - a.z_m)
    h_m = building_height_m - los_height_m
    d1_m = np.sqrt(s_m**2 + (building_height_m - a.z_m) ** 2)
    remaining_horizontal_m = np.hypot((b.col - cols) * resolution_m, (b.row - rows) * resolution_m)
    d2_m = np.sqrt(remaining_horizontal_m**2 + (building_height_m - b.z_m) ** 2)
    candidate_mask = (building_height_m > 0.0) & (h_m > 0.0) & (d1_m > 0.0) & (d2_m > 0.0)
    if component_labels is not None and excluded_components:
        candidate_components = component_labels[rows, cols]
        candidate_mask &= ~np.isin(candidate_components, np.fromiter(excluded_components, dtype=np.int32))
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = h_m * np.sqrt(2.0 * (d1_m + d2_m) / (wavelength_m * d1_m * d2_m))
    nu = np.where(candidate_mask, nu, np.nan)
    return IntervalProfile(a, b, rows, cols, s_m, building_height_m, los_height_m, h_m, d1_m, d2_m, nu, candidate_mask)


def _dominant_event(profile: IntervalProfile, depth: int) -> DiffractionEvent | None:
    """从当前区间profile选择唯一最大nu的E*，并把该节点的J(nu)写入事件。"""

    valid_indices = np.flatnonzero(np.isfinite(profile.nu))
    if valid_indices.size == 0:
        return None
    index = int(valid_indices[np.argmax(profile.nu[valid_indices])])
    edge = DiffractionEndpoint(
        float(profile.rows[index]),
        float(profile.cols[index]),
        float(profile.building_height_m[index]),
    )
    nu = float(profile.nu[index])
    return DiffractionEvent(
        depth=depth,
        a=profile.a,
        b=profile.b,
        edge=edge,
        h_m=float(profile.h_m[index]),
        d1_m=float(profile.d1_m[index]),
        d2_m=float(profile.d2_m[index]),
        nu=nu,
        loss_db=knife_edge_loss_db(nu),
        component_id=None,
    )


def _set_event_component_id(event: DiffractionEvent, component_labels: np.ndarray) -> DiffractionEvent:
    """把选中的edge映射回建筑连通分量，供后续递归子区间去重。"""

    row = int(round(event.edge.row))
    col = int(round(event.edge.col))
    component_id = int(component_labels[row, col])
    return DiffractionEvent(
        depth=event.depth,
        a=event.a,
        b=event.b,
        edge=event.edge,
        h_m=event.h_m,
        d1_m=event.d1_m,
        d2_m=event.d2_m,
        nu=event.nu,
        loss_db=event.loss_db,
        component_id=None if component_id < 0 else component_id,
        event_type=event.event_type,
        wedge_angle_rad=event.wedge_angle_rad,
        incident_angle_rad=event.incident_angle_rad,
        diffraction_angle_rad=event.diffraction_angle_rad,
        corner_model=event.corner_model,
        incident_direction_rc=event.incident_direction_rc,
        outgoing_direction_rc=event.outgoing_direction_rc,
    )


def select_diffraction_method(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_height_m: float,
    reference_building_height_m: float | None = None,
) -> str:
    """Select OWR-CD for low street-level links and OWR-RD otherwise.

    The rule uses only geometry.  If a root obstruction height is supplied,
    it is the local reference; otherwise the maximum occupied building height
    is used for backwards-compatible standalone calls. A link is street-level
    when Tx/Rx are close in height and both are below half that reference
    building height. An explicit rooftop height audit on a Tx takes precedence
    and selects OWR-RD. No GT or learned parameter participates in this dispatcher.
    """

    occupied = np.asarray(height_map_m, dtype=np.float64)
    positive = occupied[occupied > 0.0]
    if positive.size == 0:
        return "owr-rd"
    building_height = float(np.max(positive)) if reference_building_height_m is None else float(reference_building_height_m)
    if building_height <= 0.0:
        return "owr-rd"
    if tx.rooftop_height_m is not None and tx.z_m >= tx.rooftop_height_m:
        return "owr-rd"
    height_gap = abs(float(tx.z_m) - float(rx_height_m))
    low_enough = max(float(tx.z_m), float(rx_height_m)) <= 0.5 * building_height
    height_similar = height_gap <= max(2.0, 0.15 * building_height)
    return "owr-cd" if low_enough and height_similar else "owr-rd"


def _first_blocking_component(
    component_labels: np.ndarray,
    a: DiffractionEndpoint,
    b: DiffractionEndpoint,
    resolution_m: float,
    excluded_components: frozenset[int],
) -> tuple[int | None, float | None]:
    """Find the first raster building component intersected by an open segment."""

    distance_m = float(np.hypot(b.row - a.row, b.col - a.col) * resolution_m)
    if distance_m <= 0.0:
        return None, None
    n_steps = max(2, int(math.ceil(distance_m / max(resolution_m, 1e-6))))
    height_px, width_px = component_labels.shape
    for step in range(1, n_steps):
        t = step / n_steps
        row = int(round(a.row + t * (b.row - a.row)))
        col = int(round(a.col + t * (b.col - a.col)))
        if not (0 <= row < height_px and 0 <= col < width_px):
            continue
        component_id = int(component_labels[row, col])
        if component_id >= 0 and component_id not in excluded_components:
            return component_id, float(t)
    return None, None


def _segment_entry_parameter(a: np.ndarray, b: np.ndarray, polygon: np.ndarray) -> float | None:
    """Return the first open-segment parameter inside a polygon, if any.

    This is an exact polygon-interior test for the horizontal footprint.  It
    deliberately treats a segment that only touches a corner or follows an
    edge as non-blocking; wall-following is rejected separately.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    direction = b - a
    if float(np.linalg.norm(direction)) <= 1e-9:
        return None
    parameters = [0.0, 1.0]
    for index in range(len(polygon)):
        edge_start = np.asarray(polygon[index], dtype=np.float64)
        edge_end = np.asarray(polygon[(index + 1) % len(polygon)], dtype=np.float64)
        edge_direction = edge_end - edge_start
        denominator = _cross2(direction, edge_direction)
        relative = edge_start - a
        if abs(denominator) <= 1e-10:
            continue
        t = _cross2(relative, edge_direction) / denominator
        u = _cross2(relative, direction) / denominator
        if -1e-10 <= t <= 1.0 + 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
            parameters.append(float(np.clip(t, 0.0, 1.0)))
    parameters = sorted(set(round(value, 12) for value in parameters))
    for left, right in zip(parameters[:-1], parameters[1:]):
        if right - left <= 1e-10:
            continue
        midpoint = a + 0.5 * (left + right) * direction
        if _polygon_contains_point(midpoint, polygon):
            return float(left)
    return None


def _first_blocking_polygon(
    component_to_polygons: dict[int, list[np.ndarray]],
    current: np.ndarray,
    target: np.ndarray,
    component_polygon_bboxes: dict[int, list[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> tuple[int | None, float | None]:
    """Find the first actual footprint interior hit along CurrentPoint -> Rx.

    OWR-CD must identify the first blocking building geometrically, rather than
    sampling the raster and accidentally treating a boundary pixel as a new
    blocker after a corner.  Ties are resolved by component ID.
    """

    best_component: int | None = None
    best_parameter: float | None = None
    for component_id in sorted(component_to_polygons):
        for polygon_index, polygon in enumerate(component_to_polygons[component_id]):
            if component_polygon_bboxes is not None:
                polygon_min, polygon_max = component_polygon_bboxes[component_id][polygon_index]
                segment_min = np.minimum(current, target) - 1e-9
                segment_max = np.maximum(current, target) + 1e-9
                if not (np.all(segment_max >= polygon_min) and np.all(segment_min <= polygon_max)):
                    continue
            entry = _segment_entry_parameter(current, target, polygon)
            if entry is None:
                continue
            if best_parameter is None or (entry, component_id) < (best_parameter, int(best_component)):
                best_component = int(component_id)
                best_parameter = float(entry)
    return best_component, best_parameter

def _corner_angles(polygon: np.ndarray, corner_index: int) -> tuple[float, float, float]:
    """Return interior wedge, incident angle and outgoing edge angle at a corner."""

    corner = polygon[corner_index]
    previous = polygon[(corner_index - 1) % len(polygon)] - corner
    following = polygon[(corner_index + 1) % len(polygon)] - corner
    previous_norm = float(np.linalg.norm(previous))
    following_norm = float(np.linalg.norm(following))
    if previous_norm == 0.0 or following_norm == 0.0:
        return math.pi, 0.0, 0.0
    cosine = float(np.clip(np.dot(previous, following) / (previous_norm * following_norm), -1.0, 1.0))
    interior_angle = float(math.acos(cosine))
    return interior_angle, previous_norm, following_norm


def _canonical_wedge_diffraction_loss_db(
    wedge_angle_rad: float,
    incident_angle_rad: float,
    diffraction_angle_rad: float,
    d1_m: float,
    d2_m: float,
    wavelength_m: float,
) -> float:
    """Canonical perfectly-conducting-wedge UTD magnitude approximation.

    This is a material-independent high-frequency wedge approximation.  The
    UTD coefficient uses the wedge index ``n=beta/pi``, wavelength/wavenumber,
    both path distances, and the angular separation between incident and
    outgoing rays.  No material reflection coefficient is invented.
    """

    beta = float(np.clip(wedge_angle_rad, 1e-3, 2.0 * math.pi - 1e-3))
    n = max(beta / math.pi, 1e-3)
    k = 2.0 * math.pi / max(float(wavelength_m), 1e-12)
    delta = float(np.clip(abs(diffraction_angle_rad), 1e-6, math.pi - 1e-6))
    denominator = max(2.0 * n, 1e-6)

    def cot(angle: float) -> float:
        sine = math.sin(angle)
        return math.cos(angle) / (sine if abs(sine) > 1e-6 else math.copysign(1e-6, sine or 1.0))

    cot_sum = cot((math.pi + delta) / denominator) + cot((math.pi - delta) / denominator)
    d1 = max(float(d1_m), 0.5 * wavelength_m)
    d2 = max(float(d2_m), 0.5 * wavelength_m)
    direct_distance = d1 + d2
    coefficient_magnitude = abs(cot_sum) / (2.0 * n * math.sqrt(2.0 * math.pi * k))
    spreading_ratio = coefficient_magnitude * math.sqrt(direct_distance / (d1 * d2))
    # Incident angle is included as a deterministic grazing regularizer.  It
    # tends to zero for a ray tangent to an edge and prevents a singular gain.
    angular_factor = max(0.05, abs(math.sin(float(incident_angle_rad))))
    field_ratio = max(1e-9, spreading_ratio * angular_factor)
    return float(max(0.0, -20.0 * math.log10(field_ratio)))


def _corner_event(
    current: DiffractionEndpoint,
    target: DiffractionEndpoint,
    polygon: np.ndarray,
    corner_index: int,
    component_id: int,
    depth: int,
    resolution_m: float,
    wavelength_m: float,
) -> DiffractionEvent:
    """Build one UTD corner event from a footprint vertex and two path rays."""

    corner = np.asarray(polygon[corner_index], dtype=np.float64)
    corner_endpoint = DiffractionEndpoint(float(corner[0]), float(corner[1]), current.z_m)
    incoming_from_corner = np.asarray([corner[0] - current.row, corner[1] - current.col], dtype=np.float64)
    outgoing_from_corner = np.asarray([target.row - corner[0], target.col - corner[1]], dtype=np.float64)
    incoming_norm = float(np.linalg.norm(incoming_from_corner))
    outgoing_norm = float(np.linalg.norm(outgoing_from_corner))
    cosine = float(np.clip(np.dot(incoming_from_corner, outgoing_from_corner) / max(incoming_norm * outgoing_norm, 1e-12), -1.0, 1.0))
    diffraction_angle = float(math.acos(cosine))
    interior_angle, previous_edge_length, following_edge_length = _corner_angles(polygon, corner_index)
    wedge_angle = float(2.0 * math.pi - interior_angle)
    edge_direction = polygon[(corner_index + 1) % len(polygon)] - corner
    edge_norm = float(np.linalg.norm(edge_direction))
    incident_angle = float(math.acos(np.clip(np.dot(incoming_from_corner, edge_direction) / max(incoming_norm * edge_norm, 1e-12), -1.0, 1.0)))
    d1_m = float(np.hypot(current.row - corner[0], current.col - corner[1]) * resolution_m)
    d2_m = float(np.hypot(target.row - corner[0], target.col - corner[1]) * resolution_m)
    loss_db = _canonical_wedge_diffraction_loss_db(
        wedge_angle,
        incident_angle,
        diffraction_angle,
        d1_m,
        d2_m,
        wavelength_m,
    )
    return DiffractionEvent(
        depth=depth,
        a=current,
        b=target,
        edge=corner_endpoint,
        h_m=0.0,
        d1_m=d1_m,
        d2_m=d2_m,
        nu=diffraction_angle,
        loss_db=loss_db,
        component_id=component_id,
        event_type="corner",
        wedge_angle_rad=wedge_angle,
        incident_angle_rad=incident_angle,
        diffraction_angle_rad=diffraction_angle,
        corner_model="canonical_utd_wedge",
        incident_direction_rc=tuple((incoming_from_corner / max(incoming_norm, 1e-12)).tolist()),
        outgoing_direction_rc=tuple((outgoing_from_corner / max(outgoing_norm, 1e-12)).tolist()),
    )


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2-D scalar cross product in image (row, col) coordinates."""

    return float(a[0] * b[1] - a[1] * b[0])


def _segment_overlaps_polygon_edge(a: np.ndarray, b: np.ndarray, polygon: np.ndarray) -> bool:
    """Return true when an open segment follows a polygon edge for a length."""

    direction = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= 1e-9:
        return False
    for index in range(len(polygon)):
        edge_start = np.asarray(polygon[index], dtype=np.float64)
        edge_direction = np.asarray(polygon[(index + 1) % len(polygon)], dtype=np.float64) - edge_start
        edge_length = float(np.linalg.norm(edge_direction))
        if edge_length <= 1e-9:
            continue
        if abs(_cross2(direction, edge_direction)) > 1e-8 * segment_length * edge_length:
            continue
        if abs(_cross2(edge_start - np.asarray(a, dtype=np.float64), direction)) > 1e-8 * segment_length * edge_length:
            continue
        t0 = float(np.dot(edge_start - a, direction) / np.dot(direction, direction))
        t1 = float(np.dot(np.asarray(polygon[(index + 1) % len(polygon)], dtype=np.float64) - a, direction) / np.dot(direction, direction))
        overlap = min(1.0, max(t0, t1)) - max(0.0, min(t0, t1))
        if overlap > 1e-3:
            return True
    return False


def _segment_intersects_polygon_interior(a: np.ndarray, b: np.ndarray, polygon: np.ndarray) -> bool:
    """Exact open-segment interior test using edge intersection intervals."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    direction = b - a
    if float(np.linalg.norm(direction)) <= 1e-9:
        return False
    parameters = [0.0, 1.0]
    for index in range(len(polygon)):
        edge_start = np.asarray(polygon[index], dtype=np.float64)
        edge_direction = np.asarray(polygon[(index + 1) % len(polygon)], dtype=np.float64) - edge_start
        denominator = _cross2(direction, edge_direction)
        relative = edge_start - a
        if abs(denominator) <= 1e-10:
            continue
        t = _cross2(relative, edge_direction) / denominator
        u = _cross2(relative, direction) / denominator
        if 1e-10 < t < 1.0 - 1e-10 and 1e-10 < u < 1.0 - 1e-10:
            parameters.append(float(t))
    parameters = sorted(set(round(value, 12) for value in parameters))
    for left, right in zip(parameters[:-1], parameters[1:]):
        if right - left <= 1e-10:
            continue
        midpoint = a + 0.5 * (left + right) * direction
        if _polygon_contains_point(midpoint, polygon):
            return True
    return False


def _segment_enters_polygon(a: np.ndarray, b: np.ndarray, polygon: np.ndarray) -> bool:
    """Return whether an open segment enters polygon interior."""

    return _segment_intersects_polygon_interior(a, b, polygon)


def _segment_bbox_may_hit(a: np.ndarray, b: np.ndarray, polygon: np.ndarray) -> bool:
    """Cheap deterministic bounding-box rejection before polygon geometry."""

    segment_min = np.minimum(a, b) - 1e-9
    segment_max = np.maximum(a, b) + 1e-9
    polygon_min = np.min(polygon, axis=0)
    polygon_max = np.max(polygon, axis=0)
    return bool(np.all(segment_max >= polygon_min) and np.all(segment_min <= polygon_max))


def _segment_hits_any_polygon(
    a: np.ndarray,
    b: np.ndarray,
    polygons: list[np.ndarray],
    polygon_bboxes: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> bool:
    """Check a candidate free-space segment against every known footprint."""

    segment_min = np.minimum(a, b) - 1e-9
    segment_max = np.maximum(a, b) + 1e-9
    for index, polygon in enumerate(polygons):
        if polygon_bboxes is None:
            bbox_hit = _segment_bbox_may_hit(a, b, polygon)
        else:
            polygon_min, polygon_max = polygon_bboxes[index]
            bbox_hit = bool(np.all(segment_max >= polygon_min) and np.all(segment_min <= polygon_max))
        if bbox_hit and _segment_enters_polygon(a, b, polygon):
            return True
    return False


def _polygon_edge_index(polygons: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten footprint edges and bounding boxes for repeated CD segment checks."""

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for polygon in polygons:
        starts.extend(np.asarray(polygon[index], dtype=np.float64) for index in range(len(polygon)))
        ends.extend(np.asarray(polygon[(index + 1) % len(polygon)], dtype=np.float64) for index in range(len(polygon)))
    if not starts:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty, empty, empty
    start_array = np.asarray(starts, dtype=np.float64)
    end_array = np.asarray(ends, dtype=np.float64)
    return start_array, end_array, np.minimum(start_array, end_array), np.maximum(start_array, end_array)


def _segment_overlaps_any_polygon_edge(
    a: np.ndarray,
    b: np.ndarray,
    polygons: list[np.ndarray],
    edge_index: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Reject long wall-following segments against any footprint.

    CD checks this predicate for every candidate corner.  The optional flattened
    edge index keeps the exact collinearity/overlap test but avoids re-iterating
    Python polygon/edge objects for every query.
    """

    if edge_index is None:
        return any(
            _segment_bbox_may_hit(a, b, polygon) and _segment_overlaps_polygon_edge(a, b, polygon)
            for polygon in polygons
        )
    starts, ends, edge_min, edge_max = edge_index
    if len(starts) == 0:
        return False
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    direction = b - a
    segment_length = float(np.linalg.norm(direction))
    if segment_length <= 1e-12:
        return False
    segment_min = np.minimum(a, b) - 1e-9
    segment_max = np.maximum(a, b) + 1e-9
    bbox_mask = np.all(edge_max >= segment_min, axis=1) & np.all(edge_min <= segment_max, axis=1)
    if not np.any(bbox_mask):
        return False
    candidate_starts = starts[bbox_mask]
    candidate_ends = ends[bbox_mask]
    edge_direction = candidate_ends - candidate_starts
    edge_length = np.linalg.norm(edge_direction, axis=1)
    nonzero = edge_length > 1e-12
    if not np.any(nonzero):
        return False
    candidate_starts = candidate_starts[nonzero]
    edge_direction = edge_direction[nonzero]
    edge_length = edge_length[nonzero]
    cross_direction = direction[0] * edge_direction[:, 1] - direction[1] * edge_direction[:, 0]
    parallel = np.abs(cross_direction) <= 1e-8 * segment_length * edge_length
    if not np.any(parallel):
        return False
    candidate_starts = candidate_starts[parallel]
    edge_direction = edge_direction[parallel]
    segment_squared = float(np.dot(direction, direction))
    start_relative = candidate_starts - a
    collinear = np.abs(direction[0] * start_relative[:, 1] - direction[1] * start_relative[:, 0]) <= 1e-8 * segment_length**2
    if not np.any(collinear):
        return False
    projections_start = np.sum(start_relative[collinear] * direction, axis=1) / segment_squared
    projections_end = projections_start + np.sum(edge_direction[collinear] * direction, axis=1) / segment_squared
    return bool(np.any((np.maximum(projections_start, projections_end) > 1e-8) & (np.minimum(projections_start, projections_end) < 1.0 - 1e-8)))


def _turn_angle(current: np.ndarray, corner: np.ndarray, target: np.ndarray) -> float:
    """Return the deterministic direction change at a candidate corner."""

    incoming = np.asarray(corner, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    outgoing = np.asarray(target, dtype=np.float64) - np.asarray(corner, dtype=np.float64)
    denominator = max(float(np.linalg.norm(incoming) * np.linalg.norm(outgoing)), 1e-12)
    cosine = float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0))
    return float(math.acos(cosine))


def _silhouette_corner_indices(current: np.ndarray, polygon: np.ndarray) -> tuple[set[int], set[int]]:
    """Return visible vertices and the two deterministic tangent corners.

    Visible vertices are tested against the polygon interior and boundary.
    The silhouette pair bounds the smallest circular angular interval seen
    from CurrentPoint; intermediate vertices are not propagation nodes.
    """

    visible: list[int] = []
    for index, corner in enumerate(polygon):
        corner_array = np.asarray(corner, dtype=np.float64)
        if _segment_overlaps_polygon_edge(current, corner_array, polygon):
            continue
        if not _segment_enters_polygon(current, corner_array, polygon):
            visible.append(index)
    if len(visible) <= 2:
        return set(visible), set(visible)

    angles = {
        index: float(math.atan2(float(polygon[index][0] - current[0]), float(polygon[index][1] - current[1])) % (2.0 * math.pi))
        for index in visible
    }
    ordered = sorted(visible, key=lambda index: (angles[index], index))
    gaps: list[tuple[float, int]] = []
    for position, left_index in enumerate(ordered):
        right_index = ordered[(position + 1) % len(ordered)]
        left_angle = angles[left_index]
        right_angle = angles[right_index] if position + 1 < len(ordered) else angles[right_index] + 2.0 * math.pi
        gaps.append((right_angle - left_angle, position))
    _largest_gap, gap_position = max(gaps, key=lambda item: (item[0], -ordered[item[1]]))
    left_tangent = ordered[(gap_position + 1) % len(ordered)]
    right_tangent = ordered[gap_position]
    return set(visible), {left_tangent, right_tangent}

def _solve_owr_cd_diffraction(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_row: int,
    rx_col: int,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    footprint_polygons: list[Any] | dict[int, list[np.ndarray]] | None,
    component_labels: np.ndarray | None = None,
    max_corner_depth: int | None = None,
    local_segment_losses: bool = False,
) -> DiffractionSolution:
    """Strict one-way recursive corner diffraction.

    Each recursion examines only the first footprint that blocks the current
    point-to-Rx segment. It tests that building's tangent/silhouette corners
    independently; it never constructs a global corner graph and never
    advances from one corner to another on the same building.
    """

    if component_labels is None:
        component_labels = _building_component_labels(height_map_m)
    if isinstance(footprint_polygons, dict):
        component_to_polygons = footprint_polygons
    else:
        polygons = _normalize_footprint_polygons(footprint_polygons)
        if not polygons:
            polygons = _extract_footprint_polygons(height_map_m)
        component_to_polygons = _associate_footprints_with_components(polygons, component_labels)

    current = _endpoint_from_tx(height_map_m, tx)
    target = DiffractionEndpoint(float(rx_row), float(rx_col), float(rx_height_m))
    events: list[DiffractionEvent] = []
    selected_nodes: list[dict[str, Any]] = []
    visited_corners: set[tuple[int, int, int]] = set()
    visited_components: set[int] = set()
    diagnostics: list[dict[str, Any]] = []
    all_polygons = [polygon for polygons in component_to_polygons.values() for polygon in polygons]
    all_polygon_bboxes = [
        (np.min(polygon, axis=0), np.max(polygon, axis=0))
        for polygon in all_polygons
    ]
    component_polygon_bboxes = {
        component_id: [
            (np.min(polygon, axis=0), np.max(polygon, axis=0))
            for polygon in polygons
        ]
        for component_id, polygons in component_to_polygons.items()
    }
    all_polygon_edges = _polygon_edge_index(all_polygons)
    first_blocking_component_id: int | None = None

    def endpoint_dict(endpoint: DiffractionEndpoint) -> dict[str, float]:
        return {"row": endpoint.row, "col": endpoint.col, "z_m": endpoint.z_m}

    def finish(termination: str, is_los: bool) -> DiffractionSolution:
        finalized_events = events
        if local_segment_losses and selected_nodes:
            # Geometry is searched first.  Only after the complete one-way
            # corner chain is known do we evaluate UTD on adjacent free-space
            # segments: Tx-C1-C2, C1-C2-C3, ..., Cn-Rx.
            finalized_events = []
            for index, node in enumerate(selected_nodes):
                local_target = (
                    selected_nodes[index + 1]["edge"]
                    if index + 1 < len(selected_nodes)
                    else target
                )
                event = _corner_event(
                    node["current"],
                    local_target,
                    node["polygon"],
                    node["corner_index"],
                    node["component_id"],
                    node["depth"],
                    resolution_m,
                    wavelength_m,
                )
                finalized_events.append(event)
                record = node["record"]
                record.update(
                    {
                        "utd_target": endpoint_dict(local_target),
                        "local_outgoing_distance_px": float(
                            np.hypot(
                                local_target.row - event.edge.row,
                                local_target.col - event.edge.col,
                            )
                        ),
                        "utd_loss_db": event.loss_db,
                        "incident_direction_rc": event.incident_direction_rc,
                        "outgoing_direction_rc": event.outgoing_direction_rc,
                        "wedge_angle_rad": event.wedge_angle_rad,
                    }
                )
        return DiffractionSolution(
            mode="owr-cd",
            loss_db=float(sum(event.loss_db for event in finalized_events)),
            events=finalized_events,
            termination=termination,
            is_los=is_los,
            corner_diagnostics=diagnostics,
            first_blocking_component_id=first_blocking_component_id,
        )

    # When no explicit cap is supplied, the visited-building guard bounds the
    # recursion: a connected building can contribute at most one corner event.
    # An explicit cap remains available for synthetic tests and diagnostics.
    component_count = max(0, int(component_labels.max()) + 1)
    iteration_limit = component_count + 1 if max_corner_depth is None else max_corner_depth + 1
    for depth in range(iteration_limit):
        current_xy = np.asarray([current.row, current.col], dtype=np.float64)
        target_xy = np.asarray([target.row, target.col], dtype=np.float64)
        component_id, progress = _first_blocking_polygon(
            component_to_polygons,
            current_xy,
            target_xy,
            component_polygon_bboxes,
        )
        if component_id is None:
            has_corner_path = bool(events) or bool(selected_nodes)
            return finish("rx_visible" if has_corner_path else "los", not has_corner_path)
        if first_blocking_component_id is None:
            first_blocking_component_id = component_id
        if max_corner_depth is not None and depth >= max_corner_depth:
            return finish("max_corner_depth", False)
        if component_id in visited_components:
            return finish("same_building_reentry", False)
        blocking_polygons = component_to_polygons.get(component_id, [])
        if not blocking_polygons:
            return finish("missing_blocking_footprint", False)

        visible_indices: set[int] = set()
        silhouette_indices: set[int] = set()
        for polygon_index, polygon in enumerate(blocking_polygons):
            visible, silhouette = _silhouette_corner_indices(current_xy, polygon)
            visible_indices.update((polygon_index << 16) | index for index in visible)
            silhouette_indices.update((polygon_index << 16) | index for index in silhouette)

        candidate_records: list[dict[str, Any]] = []
        candidates: list[
            tuple[
                tuple[float, float, float, float, float],
                DiffractionEvent | None,
                dict[str, Any],
            ]
        ] = []
        forward_vector = target_xy - current_xy
        forward_norm_sq = float(np.dot(forward_vector, forward_vector))
        for polygon_index, polygon in enumerate(blocking_polygons):
            for corner_index, corner in enumerate(polygon):
                corner_code = (polygon_index << 16) | corner_index
                corner_array = np.asarray(corner, dtype=np.float64)
                corner_key = (
                    component_id,
                    int(round(float(corner[0]) * 2.0)),
                    int(round(float(corner[1]) * 2.0)),
                )
                record: dict[str, Any] = {
                    "building_id": component_id,
                    "building_component_id": component_id,
                    "polygon_index": polygon_index,
                    "corner_index": corner_index,
                    "corner": {"row": float(corner[0]), "col": float(corner[1])},
                    "visible_from_current": corner_code in visible_indices,
                    "silhouette_corner": corner_code in silhouette_indices,
                    "accepted": False,
                    "reject_reason": None,
                }
                if corner_key in visited_corners:
                    record["reject_reason"] = "visited_corner"
                elif corner_code not in silhouette_indices:
                    record["reject_reason"] = "not_silhouette_tangent"
                else:
                    incoming_vector = corner_array - current_xy
                    current_to_corner = float(np.linalg.norm(incoming_vector))
                    corner_to_target_vector = target_xy - corner_array
                    corner_to_target = float(np.linalg.norm(corner_to_target_vector))
                    route_progress = float(np.dot(incoming_vector, forward_vector) / max(forward_norm_sq, 1e-12))
                    record.update({
                        "incoming_distance_px": current_to_corner,
                        "outgoing_distance_px": corner_to_target,
                        "forward_projection": route_progress,
                    })
                    if current_to_corner <= 0.25 or corner_to_target <= 0.25 or route_progress <= 1e-6:
                        record["reject_reason"] = "not_forward_progress"
                    elif _segment_overlaps_any_polygon_edge(current_xy, corner_array, all_polygons, all_polygon_edges):
                        record["reject_reason"] = "incoming_wall_overlap"
                    elif _segment_hits_any_polygon(current_xy, corner_array, all_polygons, all_polygon_bboxes):
                        record["reject_reason"] = "incoming_blocked"
                    elif _segment_overlaps_any_polygon_edge(corner_array, target_xy, all_polygons, all_polygon_edges):
                        record["reject_reason"] = "outgoing_wall_overlap"
                    elif _segment_enters_polygon(corner_array, target_xy, polygon):
                        record["reject_reason"] = "outgoing_reenters_blocking_building"
                    else:
                        outgoing_unit = corner_to_target_vector / max(corner_to_target, 1e-12)
                        probe = corner_array + min(0.25, 0.1 * max(resolution_m, 1e-6)) * outgoing_unit
                        if _polygon_contains_point(probe, polygon):
                            record["reject_reason"] = "outgoing_does_not_leave_blocking_building"
                        else:
                            event = None
                            if not local_segment_losses:
                                event = _corner_event(
                                    current, target, polygon, corner_index, component_id,
                                    depth, resolution_m, wavelength_m,
                                )
                            turn_angle = _turn_angle(current_xy, corner_array, target_xy)
                            record.update({"turn_angle_rad": turn_angle, "accepted": True})
                            if event is not None:
                                record.update({
                                    "utd_loss_db": event.loss_db,
                                    "incident_direction_rc": event.incident_direction_rc,
                                    "outgoing_direction_rc": event.outgoing_direction_rc,
                                    "wedge_angle_rad": event.wedge_angle_rad,
                                })
                            score_tail = (
                                event.loss_db if event is not None else 0.0
                            )
                            score = (
                                current_to_corner + corner_to_target,
                                turn_angle,
                                score_tail,
                                float(corner[0]),
                                float(corner[1]),
                            )
                            candidates.append((score, event, record))
                candidate_records.append(record)

        if not candidates:
            diagnostics.append({
                "depth": depth,
                "current": endpoint_dict(current),
                "rx": endpoint_dict(target),
                "blocking_building_id": component_id,
                "blocking_progress": progress,
                "candidates": candidate_records,
                "termination": "no_valid_corner",
            })
            return finish("no_valid_corner", False)

        candidates.sort(key=lambda item: item[0])
        _score, event, selected_record = candidates[0]
        selected_record["selected"] = True
        diagnostics.append({
            "depth": depth,
            "current": endpoint_dict(current),
            "rx": endpoint_dict(target),
            "blocking_building_id": component_id,
            "blocking_progress": progress,
            "candidates": candidate_records,
            "selected_corner": selected_record["corner"],
            "selected_building_id": component_id,
        })
        selected_corner = selected_record["corner"]
        selected_endpoint = DiffractionEndpoint(
            float(selected_corner["row"]),
            float(selected_corner["col"]),
            current.z_m,
        )
        corner_key = (
            component_id,
            int(round(selected_endpoint.row * 2.0)),
            int(round(selected_endpoint.col * 2.0)),
        )
        visited_corners.add(corner_key)
        visited_components.add(component_id)
        selected_nodes.append(
            {
                "current": current,
                "edge": selected_endpoint,
                "polygon": blocking_polygons[int(selected_record["polygon_index"])],
                "corner_index": int(selected_record["corner_index"]),
                "component_id": component_id,
                "depth": depth,
                "record": selected_record,
            }
        )
        if event is not None:
            events.append(event)
        current = selected_endpoint

    return finish("max_corner_depth" if max_corner_depth is not None else "max_connected_buildings", False)


def solve_owr_cd_local_segment_diffraction(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_row: int,
    rx_col: int,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    footprint_polygons: list[Any] | dict[int, list[np.ndarray]] | None,
    component_labels: np.ndarray | None = None,
    max_corner_depth: int | None = None,
) -> DiffractionSolution:
    """Evaluate OWR-CD UTD on the already selected adjacent-corner chain.

    The legacy PDF-compatible solver evaluates each corner against the final
    Rx while searching the chain.  This diagnostic/reference entry point keeps
    the same one-way geometry rules, but defers UTD until the chain is known:
    ``Tx-C1-C2``, ``C1-C2-C3``, ..., ``Cn-Rx``.  It is intentionally separate
    from the public mode dispatcher until the comparison has been reviewed.
    """

    return _solve_owr_cd_diffraction(
        height_map_m,
        tx,
        rx_row,
        rx_col,
        rx_height_m,
        resolution_m,
        wavelength_m,
        footprint_polygons,
        component_labels,
        max_corner_depth,
        local_segment_losses=True,
    )


def _solve_single_diffraction(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_row: int,
    rx_col: int,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    path_sampling_step_m: float,
) -> DiffractionSolution:
    """保留当前Single Dominant Knife-Edge语义：完整Tx-Rx profile只选择一个E*。"""

    a = _endpoint_from_tx(height_map_m, tx)
    b = DiffractionEndpoint(float(rx_row), float(rx_col), float(rx_height_m))
    profile = _build_interval_profile(height_map_m, a, b, rx_height_m, resolution_m, wavelength_m, path_sampling_step_m)
    event = _dominant_event(profile, 0)
    return DiffractionSolution(
        "single",
        0.0 if event is None else event.loss_db,
        [] if event is None else [event],
        "no_obstacle" if event is None else "single_edge",
        event is None,
    )


def _solve_owr_rd_diffraction(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_row: int,
    rx_col: int,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    path_sampling_step_m: float,
    component_labels: np.ndarray | None = None,
) -> DiffractionSolution:
    """实现OWR-RD：每次只递归rooftop edge->Rx，且重建当前区间LOS。"""

    a = _endpoint_from_tx(height_map_m, tx)
    b = DiffractionEndpoint(float(rx_row), float(rx_col), float(rx_height_m))
    events: list[DiffractionEvent] = []
    max_edges = height_map_m.shape[0] + height_map_m.shape[1]
    if component_labels is None:
        component_labels = _building_component_labels(height_map_m)

    def recurse(
        current_a: DiffractionEndpoint,
        current_b: DiffractionEndpoint,
        depth: int,
        excluded_components: frozenset[int],
    ) -> None:
        if depth >= max_edges:
            return
        profile = _build_interval_profile(
            height_map_m,
            current_a,
            current_b,
            rx_height_m,
            resolution_m,
            wavelength_m,
            path_sampling_step_m,
            component_labels,
            excluded_components,
        )
        event = _dominant_event(profile, depth)
        if event is None:
            return
        event = _set_event_component_id(event, component_labels)
        events.append(event)
        # 研究定义的单向递归：不执行A->E*，只将当前E*作为下一层起点继续到B。
        next_excluded = excluded_components if event.component_id is None else excluded_components | {event.component_id}
        recurse(event.edge, current_b, depth + 1, next_excluded)

    recurse(a, b, 0, frozenset())
    return DiffractionSolution(
        "owr-rd",
        float(sum(event.loss_db for event in events)),
        events,
        "no_obstacle" if not events else "no_obstacle_or_max_edges",
        not bool(events),
    )


def _solve_deygout_diffraction(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_row: int,
    rx_col: int,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    path_sampling_step_m: float,
    component_labels: np.ndarray | None = None,
) -> DiffractionSolution:
    """实现标准Deygout树：每个主刀刃后同时递归A->E*与E*->B。

    这里采用Deygout基本构造的加和规则：每个递归选中的edge贡献J(nu)，总损耗为所有递归
    节点贡献之和；不加入地球曲率、反射或ITU针对一般地形的经验修正，因为当前profile是
    局部建筑二维剖面，且用户指定了统一J(nu)。
    """

    a = _endpoint_from_tx(height_map_m, tx)
    b = DiffractionEndpoint(float(rx_row), float(rx_col), float(rx_height_m))
    events: list[DiffractionEvent] = []
    max_depth = 64
    if component_labels is None:
        component_labels = _building_component_labels(height_map_m)

    def recurse(
        current_a: DiffractionEndpoint,
        current_b: DiffractionEndpoint,
        depth: int,
        excluded_components: frozenset[int],
    ) -> None:
        if depth >= max_depth:
            return
        profile = _build_interval_profile(
            height_map_m,
            current_a,
            current_b,
            rx_height_m,
            resolution_m,
            wavelength_m,
            path_sampling_step_m,
            component_labels,
            excluded_components,
        )
        event = _dominant_event(profile, depth)
        if event is None:
            return
        event = _set_event_component_id(event, component_labels)
        events.append(event)
        # 标准Deygout的双向递归：父节点的nu不下传，两个子区间各自重新建LOS并计算nu。
        next_excluded = excluded_components if event.component_id is None else excluded_components | {event.component_id}
        recurse(current_a, event.edge, depth + 1, next_excluded)
        recurse(event.edge, current_b, depth + 1, next_excluded)

    recurse(a, b, 0, frozenset())
    return DiffractionSolution(
        "deygout",
        float(sum(event.loss_db for event in events)),
        events,
        "no_obstacle_or_max_depth" if events else "no_obstacle",
        not bool(events),
    )


def _solve_auto_diffraction(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_row: int,
    rx_col: int,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    path_sampling_step_m: float,
    component_labels: np.ndarray | None = None,
    footprint_polygons: list[Any] | dict[int, list[np.ndarray]] | None = None,
    max_corner_depth: int | None = None,
    root_solution: DiffractionSolution | None = None,
) -> DiffractionSolution:
    """按单个Rx的几何状态选择CD/RD，并保证NLoS不静默退化为零绕射。

    先用完整Tx-Rx rooftop profile判定根区间是否被遮挡。LOS像素直接返回
    ``los-fspl``，不会执行corner/rooftop diffraction。NLoS像素按Tx/Rx与
    建筑高度关系选择OWR-CD或OWR-RD；若严格OWR-CD找不到可连接Rx的合法
    corner chain，则使用同一根区间的OWR-RD作为几何回退。若极端离散几何仍
    让RD没有事件，最后保留single root edge，避免NLoS被错误记录为零损耗。
    回退来源写入``fallback_from``，供数组和JSON审计。
    """

    root = root_solution or _solve_single_diffraction(
        height_map_m,
        tx,
        rx_row,
        rx_col,
        rx_height_m,
        resolution_m,
        wavelength_m,
        path_sampling_step_m,
    )
    if root.is_los:
        root.dispatch_method = "los-fspl"
        return root

    selected_method = select_diffraction_method(
        height_map_m,
        tx,
        rx_height_m,
        reference_building_height_m=float(root.events[0].edge.z_m) if root.events else None,
    )
    if selected_method == "owr-cd":
        corner = _solve_owr_cd_diffraction(
            height_map_m,
            tx,
            rx_row,
            rx_col,
            rx_height_m,
            resolution_m,
            wavelength_m,
            footprint_polygons,
            component_labels,
            max_corner_depth,
        )
        if corner.termination == "rx_visible" and corner.events:
            corner.dispatch_method = "owr-cd"
            return corner
        fallback = _solve_owr_rd_diffraction(
            height_map_m,
            tx,
            rx_row,
            rx_col,
            rx_height_m,
            resolution_m,
            wavelength_m,
            path_sampling_step_m,
            component_labels,
        )
        if fallback.events:
            fallback.dispatch_method = "owr-rd"
            fallback.fallback_from = "owr-cd"
            return fallback
        root.dispatch_method = "single-fallback"
        root.fallback_from = "owr-cd"
        return root

    rooftop = _solve_owr_rd_diffraction(
        height_map_m,
        tx,
        rx_row,
        rx_col,
        rx_height_m,
        resolution_m,
        wavelength_m,
        path_sampling_step_m,
        component_labels,
    )
    if rooftop.events:
        rooftop.dispatch_method = "owr-rd"
        return rooftop
    root.dispatch_method = "single-fallback"
    root.fallback_from = "owr-rd"
    return root


def solve_diffraction(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_row: int,
    rx_col: int,
    mode: str,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    path_sampling_step_m: float = 1.0,
    component_labels: np.ndarray | None = None,
    footprint_polygons: list[Any] | dict[int, list[np.ndarray]] | None = None,
    max_corner_depth: int | None = None,
    root_solution: DiffractionSolution | None = None,
) -> DiffractionSolution:
    """统一入口：none/single/OWR-RD/Deygout/OWR-CD/auto共享同一几何。"""

    mode = validate_diffraction_mode(mode)
    if mode == "auto":
        return _solve_auto_diffraction(
            height_map_m,
            tx,
            rx_row,
            rx_col,
            rx_height_m,
            resolution_m,
            wavelength_m,
            path_sampling_step_m,
            component_labels,
            footprint_polygons,
            max_corner_depth,
            root_solution,
        )
    if mode == "none":
        return DiffractionSolution("none", 0.0, [], "disabled")
    if mode == "single":
        return _solve_single_diffraction(height_map_m, tx, rx_row, rx_col, rx_height_m, resolution_m, wavelength_m, path_sampling_step_m)
    if mode == "owr-rd":
        return _solve_owr_rd_diffraction(height_map_m, tx, rx_row, rx_col, rx_height_m, resolution_m, wavelength_m, path_sampling_step_m, component_labels)
    if mode == "deygout":
        return _solve_deygout_diffraction(height_map_m, tx, rx_row, rx_col, rx_height_m, resolution_m, wavelength_m, path_sampling_step_m, component_labels)
    return _solve_owr_cd_diffraction(
        height_map_m,
        tx,
        rx_row,
        rx_col,
        rx_height_m,
        resolution_m,
        wavelength_m,
        footprint_polygons,
        component_labels,
        max_corner_depth,
    )


def propagation_profile(
    height_map_m: np.ndarray,
    tx: TxRecord,
    rx_row: int,
    rx_col: int,
    rx_height_m: float,
    resolution_m: float,
    wavelength_m: float,
    path_sampling_step_m: float = 1.0,
) -> PropagationProfile:
    """为一个Rx建立Tx-Rx高度剖面，并只保留nu最大的dominant obstruction。

    路径采样规则是：沿连续Tx-Rx线按不超过path_sampling_step_m的间距采样，映射到最近的整数
    行列像素并去重；Tx/Rx端点不作为障碍候选。候选点E=(x,y,H_i)，d1/d2是从Tx/Rx到E的
    三维欧氏距离，垂直分量分别为H_i-z_t和H_i-z_r。
    """

    height_map_m = np.asarray(height_map_m, dtype=np.float64)
    height_px, width_px = height_map_m.shape
    if not (0 <= rx_row < height_px and 0 <= rx_col < width_px):
        raise IndexError(f"Rx ({rx_row}, {rx_col}) is outside {height_map_m.shape}.")
    # TxRecord.y_m是底部原点世界坐标；数组row从顶部开始，二者只在此处转换。
    tx_row = float(height_px - 1) - tx.y_m
    tx_col = float(tx.x_m)
    delta_row = float(rx_row) - tx_row
    delta_col = float(rx_col) - tx_col
    horizontal_distance_m = np.hypot(delta_row, delta_col) * resolution_m
    if horizontal_distance_m == 0.0:
        empty = np.empty(0, dtype=np.float64)
        return PropagationProfile(
            rx_row, rx_col, empty, empty, empty, np.empty(0, bool), empty, empty, empty, empty,
            -1, -1, -1, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, False,
        )

    n_steps = max(1, int(np.ceil(horizontal_distance_m / path_sampling_step_m)))
    if n_steps <= 1:
        empty = np.empty(0, dtype=np.float64)
        return PropagationProfile(
            rx_row, rx_col, empty, empty, empty, np.empty(0, bool), empty, empty, empty, empty,
            -1, -1, -1, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, False,
        )
    t = np.arange(1, n_steps, dtype=np.float64) / n_steps
    rows = np.rint(tx_row + t * delta_row).astype(np.int64)
    cols = np.rint(tx_col + t * delta_col).astype(np.int64)
    keep = np.concatenate(([True], (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])))
    rows, cols, t = rows[keep], cols[keep], t[keep]
    valid_grid = (rows >= 0) & (rows < height_px) & (cols >= 0) & (cols < width_px)
    rows, cols, t = rows[valid_grid], cols[valid_grid], t[valid_grid]

    s_m = np.hypot((cols - tx_col) * resolution_m, (rows - tx_row) * resolution_m)
    building_height_m = height_map_m[rows, cols]
    los_height_m = tx.z_m + np.clip(s_m / horizontal_distance_m, 0.0, 1.0) * (rx_height_m - tx.z_m)
    candidate_h_m = building_height_m - los_height_m
    candidate_mask = (building_height_m > 0.0) & (candidate_h_m > 0.0)
    horizontal_d1_m = s_m
    horizontal_d2_m = np.hypot((rx_col - cols) * resolution_m, (rx_row - rows) * resolution_m)
    candidate_d1_m = np.sqrt(horizontal_d1_m**2 + (building_height_m - tx.z_m) ** 2)
    candidate_d2_m = np.sqrt(horizontal_d2_m**2 + (building_height_m - rx_height_m) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate_nu = candidate_h_m * np.sqrt(
            2.0 * (candidate_d1_m + candidate_d2_m)
            / (wavelength_m * candidate_d1_m * candidate_d2_m)
        )
    candidate_nu = np.where(candidate_mask, candidate_nu, np.nan)
    if not candidate_mask.any():
        return PropagationProfile(
            rx_row, rx_col, s_m, building_height_m, los_height_m, candidate_mask, candidate_h_m,
            candidate_d1_m, candidate_d2_m, candidate_nu, -1, -1, -1, np.nan, np.nan, np.nan,
            np.nan, np.nan, 0.0, False,
        )

    candidate_indices = np.flatnonzero(np.isfinite(candidate_nu))
    dominant_index = int(candidate_indices[np.argmax(candidate_nu[candidate_indices])])
    nu_max = float(candidate_nu[dominant_index])
    dominant_row = int(rows[dominant_index])
    dominant_col = int(cols[dominant_index])
    return PropagationProfile(
        rx_row, rx_col, s_m, building_height_m, los_height_m, candidate_mask, candidate_h_m,
        candidate_d1_m, candidate_d2_m, candidate_nu, dominant_index, dominant_row, dominant_col,
        float(building_height_m[dominant_index]), float(candidate_h_m[dominant_index]),
        float(candidate_d1_m[dominant_index]), float(candidate_d2_m[dominant_index]), nu_max,
        _knife_edge_loss_db(nu_max), True,
    )


def compute_physics_maps(
    height_map_m: np.ndarray,
    tx: TxRecord,
    frequency_hz: float,
    rx_height_m: float | np.ndarray,
    resolution_m: float,
    path_sampling_step_m: float = 1.0,
    speed_of_light_m_per_s: float = 299792458.0,
    diffraction_mode: str = "single",
    fspl_db: np.ndarray | None = None,
    footprint_polygons: list[Any] | None = None,
    max_corner_depth: int | None = None,
    observation_mask: np.ndarray | None = None,
) -> PhysicsMaps:
    """计算一个Tx覆盖全图的确定性FSPL与指定diffraction mode结果。

    每个Rx先用同一份root rooftop profile判定LOS/NLoS：LOS像素只保留FSPL，
    不执行绕射；NLoS像素才进入指定的single、OWR-RD、OWR-CD或Deygout求解。
    ``auto``在每个NLoS Rx上按Tx/Rx/建筑相对高度选择OWR-CD或OWR-RD；严格
    OWR-CD无法建立到Rx的合法corner链时，自动回退OWR-RD，避免把真实NLoS
    静默写成零绕射。fspl_db可传入已计算的FSPL，保证各对比method共享完全相同
    的FSPL而不重复计算。

    ``rx_height_m`` may be a scalar absolute Rx height or a per-pixel absolute
    height grid.  The latter is required when the dataset defines Rx height
    relative to a varying terrain surface.

    ``resolved_method_map``记录每个像素的``los-fspl``、实际``owr-cd``/
    ``owr-rd``或显式mode；``fallback_to_rd_mask``记录CD到RD的几何回退。
    """

    diffraction_mode = validate_diffraction_mode(diffraction_mode)
    resolved_mode = diffraction_mode
    height_map_m = np.asarray(height_map_m, dtype=np.float32)
    rx_height_array = np.asarray(rx_height_m, dtype=np.float64)
    if rx_height_array.ndim == 0:
        rx_height_grid: np.ndarray | None = None
    elif rx_height_array.shape == height_map_m.shape:
        rx_height_grid = rx_height_array
    else:
        raise ValueError(
            f"rx_height_m must be scalar or shape {height_map_m.shape}, got {rx_height_array.shape}"
        )
    if fspl_db is None:
        fspl_db = free_space_path_loss_db(height_map_m, tx, frequency_hz, rx_height_m, resolution_m, speed_of_light_m_per_s)
    else:
        fspl_db = np.asarray(fspl_db, dtype=np.float32)
        if fspl_db.shape != height_map_m.shape:
            raise ValueError(f"fspl_db shape {fspl_db.shape} does not match height_map {height_map_m.shape}")
    wavelength_m = speed_of_light_m_per_s / float(frequency_hz)
    shape = height_map_m.shape
    if observation_mask is None:
        observation_mask = np.ones(shape, dtype=bool)
    else:
        observation_mask = np.asarray(observation_mask, dtype=bool)
        if observation_mask.shape != shape:
            raise ValueError(f"observation_mask shape {observation_mask.shape} does not match height_map {shape}")
    diffraction = np.zeros(shape, dtype=np.float32)
    nu_max = np.full(shape, np.nan, dtype=np.float32)
    dominant_row = np.full(shape, -1, dtype=np.int32)
    dominant_col = np.full(shape, -1, dtype=np.int32)
    dominant_height = np.full(shape, np.nan, dtype=np.float32)
    dominant_h = np.full(shape, np.nan, dtype=np.float32)
    dominant_d1 = np.full(shape, np.nan, dtype=np.float32)
    dominant_d2 = np.full(shape, np.nan, dtype=np.float32)
    los_mask = np.ones(shape, dtype=bool)
    edge_count = np.zeros(shape, dtype=np.int32)
    corner_count = np.zeros(shape, dtype=np.int32)
    unresolved_nlos_mask = np.zeros(shape, dtype=bool)
    cd_validity_mask = np.zeros(shape, dtype=bool)
    first_blocking_building_id = np.full(shape, -1, dtype=np.int32)
    component_labels = _building_component_labels(height_map_m) if resolved_mode in ("auto", "owr-rd", "deygout", "owr-cd") else None
    corner_component_polygons: dict[int, list[np.ndarray]] | None = None
    if resolved_mode in ("auto", "owr-cd"):
        normalized_polygons = _normalize_footprint_polygons(footprint_polygons)
        if not normalized_polygons:
            normalized_polygons = _extract_footprint_polygons(height_map_m)
        corner_component_polygons = _associate_footprints_with_components(normalized_polygons, component_labels)

    resolved_method_map = np.full(shape, "uninitialized", dtype="<U16")
    fallback_to_rd_mask = np.zeros(shape, dtype=bool)

    for rx_row in range(shape[0]):
        for rx_col in range(shape[1]):
            rx_height_at_pixel = (
                float(rx_height_grid[rx_row, rx_col])
                if rx_height_grid is not None
                else float(rx_height_array)
            )
            root_solution = solve_diffraction(
                height_map_m,
                tx,
                rx_row,
                rx_col,
                "single",
                rx_height_at_pixel,
                resolution_m,
                wavelength_m,
                path_sampling_step_m,
            )
            is_los = bool(root_solution.is_los)
            los_mask[rx_row, rx_col] = is_los
            if is_los:
                # LOS is always a complete, cheap physical result.  A sparse
                # Rx mask may limit the expensive NLOS CD/RD solve, but it
                # must never replace an LOS pixel with an interpolated value.
                resolved_method_map[rx_row, rx_col] = "los-fspl"
                if resolved_mode == "owr-cd":
                    cd_validity_mask[rx_row, rx_col] = True
                continue
            if not observation_mask[rx_row, rx_col]:
                # This is an NLOS pixel for which only the diffraction field
                # is completed from measured/simulated Rx locations later.
                # It is not a zero-diffraction claim.
                resolved_method_map[rx_row, rx_col] = "interpolated-rx"
                continue

            if resolved_mode == "none":
                resolved_method_map[rx_row, rx_col] = "none"
                continue
            if resolved_mode == "single":
                solution = root_solution
            else:
                solution = solve_diffraction(
                    height_map_m,
                    tx,
                    rx_row,
                    rx_col,
                    resolved_mode,
                    rx_height_at_pixel,
                    resolution_m,
                    wavelength_m,
                    path_sampling_step_m,
                    component_labels,
                    corner_component_polygons if resolved_mode in ("auto", "owr-cd") else footprint_polygons,
                    max_corner_depth,
                    root_solution if resolved_mode == "auto" else None,
                )

            actual_method = solution.dispatch_method or solution.mode
            resolved_method_map[rx_row, rx_col] = actual_method
            fallback_to_rd_mask[rx_row, rx_col] = solution.fallback_from == "owr-cd"
            edge_count[rx_row, rx_col] = len(solution.events)
            corner_count[rx_row, rx_col] = sum(event.event_type == "corner" for event in solution.events)
            if resolved_mode in ("owr-cd", "auto") and solution.mode == "owr-cd":
                first_blocking_building_id[rx_row, rx_col] = -1 if solution.first_blocking_component_id is None else int(solution.first_blocking_component_id)
                cd_validity_mask[rx_row, rx_col] = bool(solution.termination == "rx_visible")
            unresolved_nlos_mask[rx_row, rx_col] = not bool(solution.events)
            if resolved_mode == "owr-cd" and solution.termination != "rx_visible":
                unresolved_nlos_mask[rx_row, rx_col] = True
            if not solution.events or (resolved_mode == "owr-cd" and solution.termination != "rx_visible"):
                continue
            primary = solution.events[0]
            # J(nu)在solver内部是正的损耗贡献；PhysicsMaps按论文PL符号保存为负值。
            diffraction[rx_row, rx_col] = -solution.loss_db
            nu_max[rx_row, rx_col] = primary.nu
            dominant_row[rx_row, rx_col] = int(primary.edge.row)
            dominant_col[rx_row, rx_col] = int(primary.edge.col)
            dominant_height[rx_row, rx_col] = primary.edge.z_m
            dominant_h[rx_row, rx_col] = primary.h_m
            dominant_d1[rx_row, rx_col] = primary.d1_m
            dominant_d2[rx_row, rx_col] = primary.d2_m
    return PhysicsMaps(
        fspl_db=fspl_db,
        diffraction_loss_db=diffraction,
        nu_max=nu_max,
        dominant_row=dominant_row,
        dominant_col=dominant_col,
        dominant_height_m=dominant_height,
        dominant_h_m=dominant_h,
        dominant_d1_m=dominant_d1,
        dominant_d2_m=dominant_d2,
        los_mask=los_mask,
        physics_prior_db=(fspl_db + diffraction).astype(np.float32),
        edge_count=edge_count,
        corner_count=corner_count,
        unresolved_nlos_mask=unresolved_nlos_mask,
        cd_validity_mask=cd_validity_mask,
        first_blocking_building_id=first_blocking_building_id,
        resolved_method_map=resolved_method_map,
        fallback_to_rd_mask=fallback_to_rd_mask,
        mode=resolved_mode,
        observation_mask=observation_mask,
    )
