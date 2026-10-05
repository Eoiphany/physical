# Experiment log: RadioMap3DSeer diffraction modes

日期：2026-10-04  
实验目录：`runs/radiomap3dseer_diffraction_modes/3.5GHz_1m`

## 目的与假设

目的：在完全相同的 RadioMap3DSeer 建筑栅格、Tx/Rx 几何、频率、分辨率和 FSPL 上，对比
`none`、`single`、OWR-KE `oneway` 和标准双向递归 `deygout`。

假设：single 是纯单主刀刃 baseline；OWR-KE 沿 Tx→Rx 只继续递归 `E*→B`；Deygout
同时递归 `A→E*` 与 `E*→B`，每个子区间重新计算 LOS、h、d1、d2、nu。多刀刃模式的
diffraction loss 应该高于或等于 single，但不应因同一栋平坦屋顶的像素数量而无界增长。

## 数据、物理模型和配置

- 数据：本地 `../dataset`，scene `0,1,2`，每个使用 Tx `0`。
- 来源论文：Yapar et al., *Dataset of Pathloss and ToA Radio Maps With Localization Application*；论文文件由运行环境外部提供。
- 配置文件：`configs/radiomap3dseer.json`。
- 地图：256×256 m、256×256 pixel、1 m/pixel；建筑高度 6.6–19.8 m。
- 频率：3.5 GHz；Tx power 23 dBm；Rx 高度 1.5 m；Tx 为屋顶绝对高度 +3 m；各向同性天线。
- FSPL：使用 Tx/Rx 三维欧氏距离；正衰减幅度为 `20log10(4πdf/c)`，最终 PhysicsMaps 按论文 signed PL 约定保存为负值。
- 候选障碍：沿水平 Tx→Rx 线按不超过 1 m 采样，端点不作为候选。
- 距离定义：根区间中 `d1=||Tx-E||_2`、`d2=||E-Rx||_2`；递归区间 A→B 中
  `d1=||A-E||_2`、`d2=||E-B||_2`，均包含绝对高度差。
- 统一刀刃函数：用户给定的 `J(nu)`，阈值 `nu<=-0.78` 时为 0。
- 递归栅格 guard：选中 edge 后，子区间排除该 edge 所属的 8-connected 建筑 component，
  只防止同一栋连续屋顶重复计数；single 不使用该 guard。

## 相比上一版本的修改

1. 保留已有 FSPL 和 Single Dominant Knife-Edge 实现。
2. 新增统一 CLI：`--diffraction-mode none|single|oneway|deygout`。
3. 新增 OWR-KE 单向递归和双向 Deygout 递归；每个递归节点保存 A、B、E*、depth、h、d1、d2、nu、J(nu)。
4. 四模式共享一次 FSPL；新增 edge count、pairwise map difference 和完整递归 JSON。
5. 新增人工 profile 测试：无障碍、单刀刃、多障碍单向链/双向树、统一 J 阈值、连续屋顶去重。
6. 新增固定色阶九联图和三行 propagation/recursive debug 图；同类图使用统一 vmin/vmax。

## 运行命令

```bash
MPLCONFIGDIR=/private/tmp/mplconfig UV_CACHE_DIR=/private/tmp/radiomap_uv_cache \
uv run --no-project --python ../sionna_osm/.venv/bin/python python run_physics_prior.py \
  --compare-all-modes --scene-ids 0,1,2 --tx-ids 0 \
  --output-dir runs/radiomap3dseer_diffraction_modes/3.5GHz_1m
```

## 实验结果

最终 signed PL 版本完整运行总耗时 **52.90 s**，每个样本分别为 16.50 s、17.07 s、19.32 s。mode runtime
和 map 统计如下；MAE/RMSE 是相对于本地 gain PNG 按仓库线性灰度映射得到的负值 signed PL，
不是对 WinProp IRT 的物理等价性声明。

| scene | mode | runtime (s) | mean signed Ldiff (dB) | min signed Ldiff (dB) | mean/max edges | MAE (dB) | RMSE (dB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | none | 0.001 | 0.000 | 0.000 | 0.000/0 | 60.846 | 63.004 |
| 0 | single | 2.27 | -26.591 | -43.836 | 0.799/1 | 34.349 | 36.067 |
| 0 | oneway | 4.36 | -27.732 | -84.600 | 0.841/3 | 33.292 | 35.307 |
| 0 | deygout | 8.01 | -31.831 | -100.365 | 1.065/4 | 29.381 | 32.150 |
| 1 | none | 0.001 | 0.000 | 0.000 | 0.000/0 | 63.906 | 65.775 |
| 1 | single | 2.33 | -32.000 | -43.956 | 0.926/1 | 31.975 | 34.010 |
| 1 | oneway | 4.69 | -33.527 | -83.715 | 0.981/3 | 30.594 | 33.065 |
| 1 | deygout | 8.44 | -36.989 | -99.558 | 1.171/4 | 27.274 | 30.248 |
| 2 | none | 0.002 | 0.000 | 0.000 | 0.000/0 | 63.860 | 65.724 |
| 2 | single | 2.35 | -31.687 | -43.534 | 0.918/1 | 32.284 | 34.320 |
| 2 | oneway | 4.97 | -33.303 | -103.435 | 0.981/4 | 30.834 | 33.334 |
| 2 | deygout | 10.31 | -40.409 | -110.262 | 1.399/4 | 24.171 | 27.730 |

OWR-KE 与 single 的平均绝对 diffraction-map 差为 1.141、1.527、1.616 dB（scene 0/1/2）。
Deygout 与 OWR-KE 的对应差为 4.099、3.462、7.106 dB；single 与 Deygout 的对应差为
5.241、4.989、8.722 dB。默认 debug Rx 已改为优先选择 Deygout edge count 最大的位置，
最终三个 profile 分别为 `(row,col)=(75,83),(252,228),(215,106)`；scene 0 图中 single 有
1 个 edge、OWR-KE 有 2 个 edge、Deygout 有 4 个递归节点。

## 测试与结论

- `py_compile`：通过。
- 离线直接测试：**9/9 PASS**。
- 人工 profile 验证：无障碍三种 diffraction solver 均为 0；单刀刃三种结果一致；多障碍
  OWR-KE 只沿 forward chain，Deygout 在 depth 1 产生左右分支；连续屋顶不重复计数。
- 结论：四种 mode、共享 FSPL、递归节点记录和真实数据可视化均已落地。Deygout 在本次
  三个 Tx0 样本上取得最低 GT MAE，但这只是本地 PNG 映射下的小样本比较，不足以证明其
  对真实传播的普适优势。

## 尚存问题

1. 这是确定性 FSPL+建筑 knife-edge prior，不包含反射、散射、材质、天线方向性和 IRT 多径。
2. 1 m 最近像素路径是离散几何近似；改变 `path_sampling_step_m` 会改变候选集合。
3. Deygout 实现采用用户指定的统一 J(nu) 与本地 2-D 建筑 profile，没有加入 ITU-R P.526 的
   地球曲率、地形/经验修正；文档中明确区分了 OWR-KE 与 Deygout。
4. GT 数值依赖仓库对 gain PNG 的线性灰度映射，后续若拿到发布者的精确后处理，应重新核验误差。
