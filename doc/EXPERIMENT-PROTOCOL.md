# Cross-dataset Physics Prior Experiment Protocol

本协议用于另一台电脑、另一个数据集和另一个 Codex 继续执行相同的物理先验实验。仓库不携带数据集，所有路径由命令行 `--data-root` 提供。

## 1. Repository contract

固定入口：

- 物理实现：`radiomap_physics.py`；
- CLI：`run_physics_prior.py`；
- 数据集配置：`configs/<dataset>.json`；
- 仓库运行上下文：根目录 `AGENTS.md`；
- 实验协议：本文件；
- 实验日志：`doc/EXPERIMENT-*.md`；
- 测试：`test/test_physics_prior.py`。

不允许为每个数据集复制一套 FSPL/diffraction solver。新数据集只能新增配置和必要的通用 loader 适配。

## 2. Dataset configuration

JSON 只记录跨机器可复现内容：地图尺寸、分辨率、坐标约定、频率、功率、Tx/Rx 高度、标签范围、采样步长、目录模板和物理 mode。禁止写入本机 dataset absolute path、用户目录、临时目录、凭据或 API key。`--data-root` 是唯一数据集位置入口。

当前配置至少应包含：

- `dataset` 和 `schema_version`；
- `scene`：地图像素尺寸、米制尺寸、分辨率、坐标原点、建筑高度范围、Tx/Rx 高度语义；
- `wireless`：frequency、Tx power、bandwidth、noise、antenna；
- `dataset_labels`：signed PL/pathgain 范围、阈值和 PNG/标签映射；
- `physics_prior`：光速、采样间隔、支持的 diffraction mode、`d1/d2` 定义；
- `visualization.fixed_color_limits`：跨 scene/Tx 固定色阶。

## 3. Signed dB convention

论文式 signed PL：

```text
PL = P_Rx(dB) - P_Tx(dB)       # negative
```

主数组和图使用负值：

```text
fspl_db                  <= 0
diffraction_loss_db      <= 0
physics_prior_db = fspl_db + diffraction_loss_db
```

Knife-edge `J(nu)` 在 solver 事件记录中仍是正的损耗贡献；加入最终 signed map 时表现为负 diffraction contribution。需要正数时只能读取或生成明确命名的 `*_loss_magnitude_db` 数组。

## 4. Required run

```bash
python run_physics_prior.py \
  --data-root /path/to/DATASET \
  --config configs/<dataset>.json \
  --scene-ids <scene_ids> \
  --tx-ids <tx_ids> \
  --compare-all-modes \
  --output-dir runs/physics_prior/<dataset>/<experiment_id>
```

四个 mode 必须共享同一份建筑 geometry、Tx/Rx geometry、frequency、resolution 和 FSPL。输出至少包括每个 mode 的 `physics_arrays/*.npy`、`ground_truth_pathgain_db.npy`、`comparison_all_modes.png/.pdf`、`recursive_debug.json`、`recursive_debug_all_modes.png/.pdf`、`metadata.json` 和 `summary.json`。

物理先验效果评估使用统一的二次分析入口：

```bash
python visualize_physics_prior_metrics.py \
  --experiment-root runs/physics_prior/<dataset>/<experiment_id> \
  --data-root /path/to/DATASET \
  --config configs/<dataset>.json \
  --scene-ids <scene_ids> --tx-ids <tx_ids> \
  --output-dir runs/physics_prior/<dataset>/<experiment_id>/metrics
```

该入口只构造`P_NLoS = FSPL + I_NLoS·L_diff`，不再评估直接全区域叠加的旧分支。建筑高度图大于0的像素固定为配置最低signed Pathloss。输出给出P_NLoS全图和共同几何NLoS mask区域的指标。RMSE/MAE保留signed dB域，MSE、NMSE和PSNR按配置固定标签范围归一化，R²按signed dB误差计算，SSIM只报告完整地图；不对不规则NLoS mask人为填充SSIM。

## 5. Acceptance checks

每个实验日志必须明确：

1. 目的和假设；
2. 相对上一版本的真实代码/配置修改；
3. 数据集配置和外部 `--data-root`，但不把绝对路径写入 Git 文件；
4. 运行命令、Python 环境和随机性说明；
5. `none/single/owr-rd/owr-cd/deygout` 的 runtime 和 signed dB 统计；
6. 人工 profile 测试结果；
7. 可视化和原始数值输出路径；
8. 物理模型限制和未验证事项；
9. `git status`、`git diff` 和用户授权后的提交信息。

## 6. Cross-machine handoff

新电脑上的 Codex 应先阅读：

```text
AGENTS.md
doc/EXPERIMENT-PROTOCOL.md
configs/<dataset>.json
doc/EXPERIMENT-*.md
```

然后检查：

```bash
python test/test_physics_prior.py
python -m py_compile radiomap_physics.py run_physics_prior.py test/test_physics_prior.py
```

接着由用户提供该电脑上的 `--data-root`，先运行小规模 scene/Tx smoke test，最后运行正式对比。
