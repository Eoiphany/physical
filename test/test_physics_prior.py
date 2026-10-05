"""
Physics Prior最小单元测试。

命令:
  uv run pytest -q
测试覆盖:
  1. FSPL三维距离与闭式公式一致；
  2. 无障碍时diffraction为0且LOS为True；
  3. 单个高于LOS的障碍只产生一个dominant E*，且d1/d2包含垂直分量；
  4. knife-edge分段公式的nu阈值行为。
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radiomap_physics import (
    TxRecord,
    compute_physics_maps,
    free_space_path_loss_db,
    knife_edge_loss_db,
    loss_magnitude_to_paper_pl_db,
    propagation_profile,
    solve_diffraction,
    select_diffraction_method,
)


def test_fspl_uses_3d_distance() -> None:
    """验证FSPL使用Tx/Rx三维距离，而不是只使用XY距离。"""

    height = np.zeros((1, 2), dtype=np.float32)
    tx = TxRecord(x_m=0.0, y_m=0.0, z_m=4.0)
    frequency = 3.5e9
    rx_height = 1.5
    result = free_space_path_loss_db(height, tx, frequency, rx_height, 1.0)
    distance = np.sqrt(1.0**2 + (rx_height - tx.z_m) ** 2)
    wavelength = 299792458.0 / frequency
    expected = 20.0 * np.log10(4.0 * np.pi * distance / wavelength)
    np.testing.assert_allclose(result[0, 1], -expected, rtol=1e-6, atol=1e-6)
    assert result[0, 1] < 0.0


def test_no_obstacle_is_los_and_zero_diffraction() -> None:
    """验证没有高于LOS的建筑候选时L_diff=0、nu为空并保持FSPL。"""

    height = np.zeros((7, 7), dtype=np.float32)
    tx = TxRecord(x_m=0.0, y_m=0.0, z_m=10.0)
    maps = compute_physics_maps(height, tx, 3.5e9, 1.5, 1.0)
    assert np.all(maps.diffraction_loss_db == 0.0)
    assert np.all(maps.los_mask)
    assert np.isnan(maps.nu_max).all()
    np.testing.assert_allclose(maps.physics_prior_db, maps.fspl_db)


def test_dominant_obstruction_and_3d_d1_d2() -> None:
    """验证单刀刃只保留最大nu，并显式采用障碍屋顶的三维d1/d2。"""

    height = np.zeros((9, 9), dtype=np.float32)
    height[4, 4] = 12.0
    tx = TxRecord(x_m=0.0, y_m=8.0, z_m=2.0)
    profile = propagation_profile(height, tx, 8, 8, 1.5, 1.0, 299792458.0 / 3.5e9)
    assert profile.has_obstruction
    assert (profile.dominant_row, profile.dominant_col) == (4, 4)
    expected_d1 = np.sqrt((4.0**2 + 4.0**2) + (12.0 - 2.0) ** 2)
    expected_d2 = np.sqrt((4.0**2 + 4.0**2) + (12.0 - 1.5) ** 2)
    np.testing.assert_allclose(profile.dominant_d1_m, expected_d1, atol=1e-6)
    np.testing.assert_allclose(profile.dominant_d2_m, expected_d2, atol=1e-6)
    assert profile.diffraction_loss_db > 0.0


def test_profile_obstacle_above_los_is_the_only_candidate() -> None:
    """验证建筑低于LOS时不会被误判为有效遮挡。"""

    height = np.zeros((5, 5), dtype=np.float32)
    height[2, 2] = 3.0
    tx = TxRecord(x_m=0.0, y_m=0.0, z_m=10.0)
    profile = propagation_profile(height, tx, 4, 4, 1.5, 1.0, 299792458.0 / 3.5e9)
    assert not profile.has_obstruction
    assert profile.diffraction_loss_db == 0.0


def test_all_modes_are_zero_without_obstacles() -> None:
    """验证none/single/OWR-RD/OWR-CD/Deygout在无障碍profile上都不产生损耗。"""

    height = np.zeros((9, 9), dtype=np.float32)
    tx = TxRecord(x_m=0.0, y_m=8.0, z_m=2.0)
    for mode in ("none", "single", "owr-rd", "owr-cd", "deygout"):
        solution = solve_diffraction(height, tx, 8, 8, mode, 1.5, 1.0, 299792458.0 / 3.5e9)
        assert solution.loss_db == 0.0
        assert solution.events == []


def test_one_obstacle_modes_agree() -> None:
    """验证只有一个有效刀刃时三种diffraction solver给出同一J(nu)。"""

    height = np.zeros((9, 9), dtype=np.float32)
    height[4, 4] = 12.0
    tx = TxRecord(x_m=0.0, y_m=8.0, z_m=2.0)
    solutions = {
        mode: solve_diffraction(height, tx, 8, 8, mode, 1.5, 1.0, 299792458.0 / 3.5e9)
        for mode in ("single", "owr-rd", "deygout")
    }
    np.testing.assert_allclose([solutions[mode].loss_db for mode in solutions], solutions["single"].loss_db, atol=1e-6)
    assert all(len(solution.events) == 1 for solution in solutions.values())


def test_multiple_obstacles_distinguish_owr_rd_and_deygout_recursion() -> None:
    """验证OWR-RD只沿E*->B生成链，而Deygout在主刀刃两侧生成递归树。"""

    height = np.zeros((15, 15), dtype=np.float32)
    for (row, col), value in [((3, 3), 12.0), ((7, 7), 18.0), ((11, 11), 12.0)]:
        height[row, col] = value
    tx = TxRecord(x_m=0.0, y_m=14.0, z_m=2.0)
    single = solve_diffraction(height, tx, 14, 14, "single", 1.5, 1.0, 299792458.0 / 3.5e9)
    owr_rd = solve_diffraction(height, tx, 14, 14, "owr-rd", 1.5, 1.0, 299792458.0 / 3.5e9)
    deygout = solve_diffraction(height, tx, 14, 14, "deygout", 1.5, 1.0, 299792458.0 / 3.5e9)
    assert len(single.events) == 1
    assert len(owr_rd.events) > len(single.events)
    assert owr_rd.events[1].a == owr_rd.events[0].edge
    assert owr_rd.events[1].b.row == 14.0 and owr_rd.events[1].b.col == 14.0
    assert deygout.events[0].depth == 0
    depth_one = [event for event in deygout.events if event.depth == 1]
    assert len(depth_one) == 2
    assert any(event.b == deygout.events[0].edge for event in depth_one)
    assert any(event.a == deygout.events[0].edge for event in depth_one)
    assert deygout.loss_db > owr_rd.loss_db > single.loss_db


def test_unified_knife_edge_function_threshold() -> None:
    """验证rooftop diffraction modes共用用户指定的J(nu)阈值与正值公式。"""

    assert knife_edge_loss_db(-0.78) == 0.0
    assert knife_edge_loss_db(0.0) > 0.0
    assert knife_edge_loss_db(1.0) > 0.0


def test_paper_signed_pl_is_negative_of_loss_magnitude() -> None:
    """验证论文负值PL/pathgain与正损耗幅度只差一个符号。"""

    positive_loss = np.asarray([75.0, 100.0, 162.0], dtype=np.float32)
    np.testing.assert_allclose(loss_magnitude_to_paper_pl_db(positive_loss), -positive_loss)


def test_recursive_modes_do_not_repeat_one_connected_building() -> None:
    """验证连续屋顶不会被递归子区间重复累计为多把刀刃。"""

    height = np.zeros((17, 17), dtype=np.float32)
    height[4:7, 4:8] = 16.0
    height[9, 9] = 13.0
    height[12, 12] = 12.0
    tx = TxRecord(x_m=0.0, y_m=16.0, z_m=2.0)
    for mode in ("owr-rd", "deygout"):
        solution = solve_diffraction(height, tx, 16, 16, mode, 1.5, 1.0, 299792458.0 / 3.5e9)
        component_ids = [event.component_id for event in solution.events]
        assert len(component_ids) == len(set(component_ids))


def test_auto_dispatch_uses_height_relationship() -> None:
    """低Tx/Rx和高建筑选择OWR-CD，屋顶以上Tx选择OWR-RD。"""

    height = np.zeros((9, 9), dtype=np.float32)
    height[3:6, 3:6] = 25.0
    street_tx = TxRecord(x_m=0.0, y_m=8.0, z_m=1.5)
    rooftop_tx = TxRecord(x_m=0.0, y_m=8.0, z_m=26.0)
    assert select_diffraction_method(height, street_tx, 1.5) == "owr-cd"
    assert select_diffraction_method(height, rooftop_tx, 1.5) == "owr-rd"


def test_owr_cd_rejects_same_building_wall_walk() -> None:
    """验证OWR-CD从真实footprint候选corner生成单向、去重且可推进的递归链。"""

    height = np.zeros((15, 15), dtype=np.float32)
    height[5:9, 4:12] = 25.0
    tx = TxRecord(x_m=1.0, y_m=7.0, z_m=1.5)
    polygons = [{"coordinates_xy": [[4.0, 5.0], [12.0, 5.0], [12.0, 9.0], [4.0, 9.0]]}]
    solution = solve_diffraction(
        height, tx, 7, 13, "owr-cd", 1.5, 1.0, 299792458.0 / 3.5e9,
        footprint_polygons=polygons,
    )
    assert solution.events == []
    assert not solution.is_los
    assert solution.termination == "no_valid_corner"
    reasons = {candidate["reject_reason"] for step in solution.corner_diagnostics for candidate in step["candidates"]}
    assert "outgoing_reenters_blocking_building" in reasons


def test_owr_cd_accepts_one_valid_silhouette_corner() -> None:
    """A legal single-corner detour must restore Rx visibility."""

    height = np.zeros((15, 15), dtype=np.float32)
    height[4, 4] = 25.0
    height[5, 4:7] = 25.0
    height[6, 4:9] = 25.0
    height[7, 4:7] = 25.0
    height[8, 4] = 25.0
    tx = TxRecord(x_m=1.0, y_m=5.0, z_m=1.5)
    polygon = {"coordinates_xy": [[4.0, 4.0], [8.0, 6.0], [4.0, 8.0]]}
    solution = solve_diffraction(
        height, tx, 3, 12, "owr-cd", 1.5, 1.0, 299792458.0 / 3.5e9,
        footprint_polygons=[polygon], max_corner_depth=3,
    )
    assert len(solution.events) == 1
    assert solution.termination == "rx_visible"
    assert solution.events[0].event_type == "corner"
    assert solution.events[0].corner_model == "canonical_utd_wedge"
    assert solution.events[0].component_id == 0
    assert any(candidate.get("selected") for step in solution.corner_diagnostics for candidate in step["candidates"])


def test_owr_cd_case_a_los_has_no_diffraction():
    """Case A: a direct free-space segment is LOS, not an implicit corner path."""

    height = np.zeros((15, 15), dtype=np.float32)
    solution = solve_diffraction(
        height, TxRecord(x_m=1.0, y_m=13.0, z_m=1.5), 1, 13,
        "owr-cd", 1.5, 1.0, 299792458.0 / 3.5e9,
        footprint_polygons=[],
    )
    assert solution.is_los
    assert solution.termination == "los"
    assert solution.events == []
    assert solution.first_blocking_component_id is None


def test_owr_cd_case_c_recurses_only_to_next_building():
    """Case C: a legal chain contains different blocking buildings only."""

    height = np.zeros((50, 50), dtype=np.float32)
    height[28:34, 15:22] = 25.0
    height[28:34, 23:32] = 25.0
    polygons = [
        {"coordinates_xy": [[15.0, 28.0], [22.0, 28.0], [22.0, 34.0], [15.0, 34.0]]},
        {"coordinates_xy": [[23.0, 28.0], [32.0, 28.0], [32.0, 34.0], [23.0, 34.0]]},
    ]
    solution = solve_diffraction(
        height, TxRecord(x_m=2.0, y_m=4.0, z_m=1.5), 18, 40,
        "owr-cd", 1.5, 1.0, 299792458.0 / 3.5e9,
        footprint_polygons=polygons,
    )
    assert solution.termination == "rx_visible"
    assert [event.component_id for event in solution.events] == [0, 1]
    assert len(set(event.component_id for event in solution.events)) == len(solution.events)


def test_owr_cd_case_d_rejects_one_silhouette_and_accepts_another():
    """Case D: candidates on the same blocker are tested independently."""

    height = np.zeros((35, 35), dtype=np.float32)
    height[17:22, 15:23] = 25.0
    polygon = {"coordinates_xy": [[15.0, 17.0], [23.0, 17.0], [23.0, 22.0], [15.0, 22.0]]}
    solution = solve_diffraction(
        height, TxRecord(x_m=2.0, y_m=9.0, z_m=1.5), 17, 33,
        "owr-cd", 1.5, 1.0, 299792458.0 / 3.5e9,
        footprint_polygons=[polygon],
    )
    candidates = solution.corner_diagnostics[0]["candidates"]
    assert any(candidate.get("silhouette_corner") and candidate.get("reject_reason") for candidate in candidates)
    assert any(candidate.get("selected") for candidate in candidates)
    assert solution.termination == "rx_visible"

if __name__ == "__main__":
    # 在无pytest的离线环境中仍可直接用本项目约定的uv run命令执行最小测试集。
    tests = [
        test_fspl_uses_3d_distance,
        test_no_obstacle_is_los_and_zero_diffraction,
        test_dominant_obstruction_and_3d_d1_d2,
        test_profile_obstacle_above_los_is_the_only_candidate,
        test_all_modes_are_zero_without_obstacles,
        test_one_obstacle_modes_agree,
        test_multiple_obstacles_distinguish_owr_rd_and_deygout_recursion,
        test_unified_knife_edge_function_threshold,
        test_paper_signed_pl_is_negative_of_loss_magnitude,
        test_recursive_modes_do_not_repeat_one_connected_building,
        test_auto_dispatch_uses_height_relationship,
        test_owr_cd_rejects_same_building_wall_walk,
        test_owr_cd_case_a_los_has_no_diffraction,
        test_owr_cd_case_c_recurses_only_to_next_building,
        test_owr_cd_case_d_rejects_one_silhouette_and_accepts_another,
        test_owr_cd_accepts_one_valid_silhouette_corner,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
