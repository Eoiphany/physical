# Experiment log: P_NLoS-only Physics Prior

## Purpose and hypothesis

目的：固定使用唯一物理先验主线`P_NLoS=FSPL+I_NLoS·L_diff`，并在全图和NLoS子区域上评估其效果，建立与后续surrogate数据驱动模型一致的指标输出格式。

假设：绕射只应作用于几何NLoS区域；在RadioMap3DSeer中，建筑像素是不可接收区域，因此固定为配置最低signed Pathloss。加入Single、OWR-KE和Deygout绕射项后，RMSE、MAE、NMSE下降，PSNR和R²上升。

## Change from previous result

- 不重新计算FSPL或diffraction，直接复用同一批`none/single/oneway/deygout`物理数组。
- `visualize_physics_prior_metrics.py`删除直接全区域叠加分支，只构造`P_NLoS=FSPL+I_NLoS·L_diff`。
- 删除`--building-interior-policy`参数；建筑高度图大于0的像素始终置为配置最低signed Pathloss `-162 dB`。
- 总图只保留建筑高度、GT、FSPL、`FSPL+NLoS-Single`、`FSPL+NLoS-OWR-KE`、`FSPL+NLoS-Deygout`和指标面板。
- NLoS使用共同几何LOS mask的补集；`none`模式的mask不参与定义，因为纯FSPL模式不计算遮挡。

## Configuration and command

数据集：RadioMap3DSeer，3个scene、每个scene的Tx 0，1 m分辨率，3.5 GHz。

外部数据路径只通过命令行传入，未写入配置或本实验日志：

```bash
python visualize_physics_prior_metrics.py \
  --experiment-root runs/radiomap3dseer_diffraction_modes/3.5GHz_1m \
  --data-root /path/to/RadioMap3DSeer \
  --config configs/radiomap3dseer.json \
  --scene-ids 0,1,2 --tx-ids 0 \
  --output-dir runs/radiomap3dseer_diffraction_modes/3.5GHz_1m/metrics
```

指标口径：RMSE/MAE为signed dB域；MSE、NMSE、PSNR按固定`[-162,-75] dB`范围归一化；R²按signed dB误差计算；SSIM只对完整地图计算，NLoS子区域SSIM为N/A。

## Building-interior rule and evidence

以`height_map_m > 0`定义建筑像素，并读取同一批RadioMap3DSeer GT标签后，GT建筑像素满足`GT <= -160 dB`的比例为：scene 0 `91.65%`、scene 1 `88.54%`、scene 2 `88.17%`。因此在本数据集的标签定义下，将建筑内部设置为配置下限`-162 dB`是有数据依据的，且能避免FSPL在建筑内部产生不符合标签语义的高接收功率预测。该规则固定属于当前RadioMap3DSeer物理先验主线，不再提供可切换的旧策略。

## Aggregate results

以下数值是3个scene/Tx的macro-sample mean，数值越小越好：

| Scope | Method | RMSE (dB) | MAE (dB) | R² | NMSE | PSNR (dB) | SSIM |
|---|---|---:|---:|---:|---:|---:|---:|
| All / P_NLoS | FSPL | 48.4605 | 39.3794 | -6.5584 | 4.2112 | 5.0926 | 0.2617 |
| All / P_NLoS-Single | FSPL+NLoS-Single | 26.8044 | 21.1515 | -1.3105 | 1.2792 | 10.2629 | 0.3488 |
| All / P_NLoS-OWR-KE | FSPL+NLoS-OWR-KE | 26.1994 | 20.4728 | -1.2082 | 1.2216 | 10.4659 | 0.3409 |
| All / P_NLoS-Deygout | FSPL+NLoS-Deygout | **23.7601** | **18.0439** | **-0.8215** | **1.0025** | **11.3392** | **0.3649** |
| NLoS / P_NLoS | FSPL | 49.4415 | 39.5051 | -10.1525 | 6.5787 | 4.9167 | N/A |
| NLoS / P_NLoS-Single | FSPL+NLoS-Single | 24.4240 | 18.8091 | -1.7256 | 1.6102 | 11.0421 | N/A |
| NLoS / P_NLoS-OWR-KE | FSPL+NLoS-OWR-KE | 23.6814 | 18.0462 | -1.5635 | 1.5143 | 11.3114 | N/A |
| NLoS / P_NLoS-Deygout | FSPL+NLoS-Deygout | **20.6121** | **15.3048** | **-0.9504** | **1.1484** | **12.5261** | N/A |

## Fit conclusion

按全图RMSE、MAE、NMSE、PSNR和R²，当前建筑后处理后的顺序为：

```text
Deygout > OWR-KE > Single > FSPL
```

这里的`>`表示拟合效果更好，而不是数值更大。所有结果都来自唯一的`P_NLoS=FSPL+I_NLoS·L_diff`，没有直接全区域叠加版本。建筑内部固定为最低Pathloss后，绕射先验仍显著优于纯FSPL。

SSIM的全图顺序为`Deygout > Single > OWR-KE > FSPL`，因此SSIM与点对点误差指标对OWR-KE和Single的局部排序不完全一致。后续论文主表应以RMSE、MAE和R²为主，SSIM作为空间结构补充指标，不应只依赖SSIM下结论。

## Outputs

- 主图PNG：`runs/radiomap3dseer_diffraction_modes/3.5GHz_1m/metrics/physics_prior_NLoS_comparison.png`
- 主图PDF：`runs/radiomap3dseer_diffraction_modes/3.5GHz_1m/metrics/physics_prior_NLoS_comparison.pdf`
- 原始指标CSV：`runs/radiomap3dseer_diffraction_modes/3.5GHz_1m/metrics/metrics_summary.csv`
- 指标汇总JSON：`runs/radiomap3dseer_diffraction_modes/3.5GHz_1m/metrics/metrics_summary.json`
- 图面板说明：`runs/radiomap3dseer_diffraction_modes/3.5GHz_1m/metrics/figure_manifest.json`

## Validation and limitations

- `test/test_physics_prior.py`：10/10通过。
- 指标与绘图重算runtime：约`1.40 s`（3个scene/Tx、24行指标、PNG/PDF/CSV/JSON）。
- 12个scene/Tx/mode数组检查通过：`prior_db == fspl_db + diffraction_loss_db`；`none`模式满足`prior_db == fspl_db`。
- PNG尺寸：7200×3600；PDF已生成。
- 唯一P_NLoS主线使用同一批signed PL数组与共同几何mask，四种方法共享GT、频率、分辨率和FSPL。
- 当前GT仍是本地gain PNG按配置范围线性映射得到的代理标签；未核验发布者的完整后处理，因此结果是当前标签映射下的物理先验比较，不等同于对WinProp IRT的最终精度结论。
- 当前只比较3个scene/Tx，尚不足以判断跨场景统计稳定性。
