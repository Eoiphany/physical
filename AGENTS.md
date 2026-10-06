# Physics Prior Repository Guide

本文件是另一个 Codex 或研究者在新电脑上接手本仓库时的运行入口。仓库只保存代码、跨机器物理配置、协议、实验日志和可追溯输出；数据集本体不进入仓库，数据集位置永远通过 `--data-root` 传入。

## Scope

本仓库实现统一的确定性 RadioMap Physics Prior。所有数据集必须复用同一套 `radiomap_physics.py` 和 `run_physics_prior.py`，数据集差异通过 `configs/*.json` 表达，不复制一份新的物理实现。

当前支持的 diffraction mode：

```text
none       pure signed FSPL baseline
single     Single Dominant Knife-Edge
owr-rd     OWR-RD / One-Way Recursive Rooftop Diffraction
owr-cd     OWR-CD / One-Way Recursive Corner Diffraction
deygout   standard two-sided Deygout recursive construction
auto      per-NLoS dispatch: OWR-CD or OWR-RD with explicit fallback diagnostics
```

OWR-RD 不能称为 Deygout。每个递归区间都必须重新计算 reference LOS、`h`、`d1`、`d2` 和 `nu`；OWR-RD 只递归 `E* -> B`，Deygout 同时递归 `A -> E*` 和 `E* -> B`。
默认 `auto` 先判定每个Rx的root LOS：LOS只使用FSPL；NLoS按局部主阻挡屋顶与Tx/Rx相对高度选择OWR-CD或OWR-RD。OWR-CD不能建立合法corner链时自动回退OWR-RD，并保存 `resolved_method_map.npy` 与 `fallback_to_rd_mask.npy`。配置中的 `max_corner_depth: null` 表示不按corner数量截断，递归由已访问建筑物集合和Rx可见性终止；仅人工测试或诊断时允许设置有限整数上限。

## Data is external

不要在 JSON、Python 默认配置、实验日志或提交命令中写入本机数据集绝对路径。运行时显式传入：

```bash
python run_physics_prior.py --data-root /path/to/DATASET --config configs/radiomap3dseer.json
```

Windows CMD 示例：

```cmd
python run_physics_prior.py ^
  --data-root %DATA_ROOT% ^
  --config configs\radiomap3dseer.json
```

`--data-root` 指向的数据集目录至少应提供：

```text
DATASET/
├── polygon/buildings/<scene_id>.json
├── antenna/<scene_id>.json
└── gain/<scene_id>_<tx_id>.png
```

如果新数据集目录结构不同，应在对应配置 JSON 中记录相对目录模板并扩展统一 loader；不要把绝对路径写入配置。

## Configuration contract

每个数据集配置 JSON 至少应包含：

- `dataset` 和 `schema_version`；
- `scene`：地图像素尺寸、米制尺寸、分辨率、坐标原点、建筑高度范围、Tx/Rx 高度语义；
- `wireless`：frequency、Tx power、bandwidth、noise、antenna；
- `dataset_labels`：signed PL/pathgain 范围、阈值、PNG/标签映射；
- `physics_prior`：光速、采样间隔、支持的 diffraction mode、`d1/d2` 定义；
- `visualization.fixed_color_limits`：跨 scene/Tx 固定色阶，禁止逐图 min-max。

跨数据集 loader 由 JSON 的 `data.format` 选择：`polygon_height_json`（3D 高度 polygon）、
`polygon_mask_json`（2D footprint polygon + 配置高度）、`image_mask`（map/Tx/pmap 图像）或
`urbanradio3d_directory`（解压后的 UrbanRadio3D 目录）。二维 proxy 高度、标签 dB 解码或缺失的 Tx z 必须在
`source.evidence_boundary`、`scene.*_source`/`tx_height_audit_note` 和实验日志中明确标注，不能伪装成已验证的三维几何。

当前主输出遵循论文 signed PL convention：

```text
PL = P_Rx(dB) - P_Tx(dB)                 # negative
fspl_db, diffraction_loss_db, prior_db   # signed negative maps
```

递归 solver 内部的 `J(nu)` 保留为正的损耗贡献；最终 map 和图使用负值。可选的正损耗幅度数组使用 `*_loss_magnitude_db.npy` 命名。

## Standard commands

在本目录运行：

```bash
python -m py_compile radiomap_physics.py run_physics_prior.py test/test_physics_prior.py
python test/test_physics_prior.py

python visualize_physics_prior_metrics.py \
  --experiment-root runs/radiomap3dseer_diffraction_modes/3.5GHz_1m \
  --data-root /path/to/DATASET \
  --config configs/radiomap3dseer.json \
  --scene-ids 0,1,2 --tx-ids 0 \
  --output-dir runs/radiomap3dseer_diffraction_modes/3.5GHz_1m/metrics

python run_physics_prior.py \
  --data-root /path/to/DATASET \
  --config configs/radiomap3dseer.json \
  --scene-ids 0,1,2 \
  --tx-ids 0 \
  --compare-all-modes \
  --output-dir runs/physics_prior/<dataset>/<experiment_id>
```

单 mode：

```bash
python run_physics_prior.py \
  --data-root /path/to/DATASET \
  --config configs/radiomap3dseer.json \
  --scene-ids 0 \
  --tx-ids 0 \
  --diffraction-mode single \
  --output-dir runs/physics_prior/<dataset>/<experiment_id>
```

如果项目环境使用 `uv`，可在以上命令前加项目已有的 `uv run --no-project --python <python>`；不要为适应某台电脑把路径写死到仓库。

## Verification protocol

每次新增数据集配置或修改物理实现，必须：

1. 运行 10 个现有人工/符号测试；
2. 检查 `prior_db == fspl_db + diffraction_loss_db`；
3. 检查主 map 的 signed dB 符号、GT 标签范围和有限值比例；
4. 至少运行一个 scene/Tx 的四 mode 对比；
5. 生成 comparison PNG/PDF、recursive debug JSON 和实验日志；
6. 运行 `visualize_physics_prior_metrics.py`，只构造并评估 `P_NLoS = FSPL + I_NLoS·L_diff`；建筑高度图大于0的像素固定为配置最低signed Pathloss，不保留直接全区域叠加分支；报告全图和共同几何NLoS mask区域的 MSE、RMSE、MAE、NMSE、PSNR、SSIM、R²；NLoS SSIM不直接计算；
7. 记录数据集配置、命令、runtime、mode统计、输出目录和已知物理限制。

协议细节见 `doc/EXPERIMENT-PROTOCOL.md`。已有实验日志只代表对应版本和配置，不得用新实验结果静默覆盖旧日志。

## Git authorization

默认只检查和修改工作树，不自动 commit/push。用户说“提交”时默认指最终执行 `git push`，不是只做本地 `git commit`；只有用户明确要求“提交”“推送”或“push”时，才为本次相关修改执行 `git add`、`git commit` 和 `git push`。若用户明确说“只本地 commit”，则不得 push。push 前在本目录运行：

```bash
git status --short -- .
git diff --check -- .
git diff -- .
git log -1 --oneline
```

只提交本次逻辑相关文件，不使用无范围的 `git add .`；完成本地 commit 后，再按用户授权执行 `git push origin <branch>`。提交信息使用 Conventional Commits，例如 `feat: add dataset configuration`、`exp: run signed physics prior comparison`、`docs: clarify push authorization semantics`。
