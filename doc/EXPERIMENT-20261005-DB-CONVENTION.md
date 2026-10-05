# Experiment log: dB sign convention audit

日期：2026-10-05  
实验类型：符号约定修正与兼容性验证

## 目的与结论

核对用户指出的论文负值 Pathloss 与当前 Physics Prior dB 符号是否一致。

论文原文定义：`PL = (P_Rx)_dB - (P_Tx)_dB`，所以 RadioMap3DSeer PNG 标签是负值，范围为
`[-162, -75] dB`。本轮按用户约定，PhysicsMaps 和图直接采用该负值 signed PL；FSPL
公式的正衰减幅度只作为中间解释，最终写入地图时使用负值：

```text
paper_signed_PL_db = -positive_loss_magnitude_db
```

本轮按用户最终约定，将主 PhysicsMaps、Ground Truth 和图统一改为论文负值 signed PL；正损耗幅度仅作为可选 `*_loss_magnitude_db.npy` 数组保存。

## 修改内容

- 新增 `load_ground_truth_path_gain_db()`：返回论文原始负值 PL/pathgain。
- `load_ground_truth_path_loss_db()` 现在返回论文负值 signed PL；另有
  `load_ground_truth_path_loss_magnitude_db()` 返回正损耗幅度。
- 新增 `loss_magnitude_to_paper_pl_db()` 及符号单元测试。
- 每个 sample 新增：
  - `ground_truth_pathgain_db.npy`
  - `ground_truth_pathloss_magnitude_db.npy`
- 每个 mode 新增：
  - `physics_arrays/fspl_paper_pl_db.npy`
  - `physics_arrays/physics_prior_paper_pl_db.npy`
- 图标题明确标注 `signed PL`，与论文保持一致。

## 验证命令与结果

```bash
MPLCONFIGDIR=/private/tmp/mplconfig UV_CACHE_DIR=/private/tmp/radiomap_uv_cache \
uv run --no-project --python ../sionna_osm/.venv/bin/python python -m py_compile \
  radiomap_physics.py run_physics_prior.py test/test_physics_prior.py

MPLCONFIGDIR=/private/tmp/mplconfig UV_CACHE_DIR=/private/tmp/radiomap_uv_cache \
uv run --no-project --python ../sionna_osm/.venv/bin/python python test/test_physics_prior.py
```

结果：**10/10 PASS**。新增测试验证 `[75,100,162]` 的正损耗幅度与论文约定的
`[-75,-100,-162] dB` 符号关系；最终主数组直接为负值。

真实输出目录保持为：
`runs/radiomap3dseer_diffraction_modes/3.5GHz_1m/`。
