# Experiment: Per-NLoS FSPL/CD/RD dispatch

## Purpose and hypothesis

验证新的主线物理先验：LOS像素只使用FSPL；NLoS像素根据局部主阻挡屋顶与Tx/Rx的相对高度选择OWR-CD或OWR-RD。严格OWR-CD无法建立合法corner到Rx的链时，必须回退OWR-RD，不能把NLoS写成零绕射。

## Baseline and change

- Baseline commit: `db6371e` (`docs: clarify signed corner diffraction diagnostics`)
- Current commit: pending
- Previous behavior: `auto`按一个Tx整张图选择单一方法，严格CD失败的NLoS可能保留零diffraction。
- Current behavior: root LOS逐Rx判定；LOS直接FSPL；NLoS逐Rx dispatch；CD失败回退RD；保存`resolved_method_map.npy`和`fallback_to_rd_mask.npy`。
- Corner recursion: 所有配置的`max_corner_depth`改为`null`；默认由已访问建筑component和Rx可见性终止，不按固定corner数量截断。整数仍可用于人工测试。

## Configuration and command

Configuration: `configs/radiomap3dseer.json`

- frequency: 3.5 GHz
- resolution: 1 m/pixel
- Rx height: 1.5 m
- Tx height: polygon rooftop absolute height + 3 m
- signed output convention: FSPL/diffraction/prior are negative dB
- data root: external, supplied only through `--data-root`

```bash
MPLCONFIGDIR=/tmp/physics_prior_mpl python run_physics_prior.py \
  --data-root /path/to/DATASET \
  --config configs/radiomap3dseer.json \
  --scene-ids 0 --tx-ids 0 \
  --diffraction-method auto \
  --output-dir runs/physics_prior/radiomap3dseer/auto_nlos_dispatch
```

## Results

RadioMap3DSeer scene 0 / Tx 0, run on the current workstation:

| quantity | result |
|---|---:|
| runtime | 12.28 s |
| LOS pixels | 13,198 / 65,536 (20.14%) |
| NLoS pixels | 52,338 / 65,536 (79.86%) |
| per-pixel dispatch | `los-fspl`: 13,198; `owr-rd`: 52,338 |
| OWR-CD dispatch | 0; Tx is above audited rooftop height, so RD is selected |
| unresolved NLoS | 0 |
| NLoS pixels with nonzero diffraction | 52,338 / 52,338 |
| `max_edge_count` | 3 |
| `prior - (fspl + diffraction)` max abs | 0 dB |
| full-map RMSE vs label | 35.3066 dB |
| full-map MAE vs label | 33.2922 dB |
| full-map bias vs label | +33.1137 dB |
| NLoS RMSE vs label | 33.2249 dB |

The label metrics are diagnostic only; this experiment tests the physical dispatch and numerical invariants, not a claim of accuracy improvement.

## Verification

The direct artificial-profile test suite completed 19/19 tests. It covers 3-D FSPL, no-obstacle LOS, single edge, OWR-RD chain, Deygout tree, local height dispatch, LOS-only FSPL, and CD-to-RD fallback. The real output also passed finite-value, signed-map, and prior identity checks.

The strict OWR-CD full 256x256 stress run was intentionally stopped after approximately 73 s without completion. Single-Rx OWR-CD tests complete and pass, but all-pixel CD requires further spatial acceleration before treating its runtime as production-ready. The default RadioMap3DSeer auto path is unaffected for the current rooftop Tx records because it selects OWR-RD.

## Outputs

- `runs/physics_prior/radiomap3dseer/auto_nlos_dispatch/scene_0_tx_0/comparison.png`
- `runs/physics_prior/radiomap3dseer/auto_nlos_dispatch/scene_0_tx_0/comparison.pdf`
- `runs/physics_prior/radiomap3dseer/auto_nlos_dispatch/scene_0_tx_0/propagation_profile.png`
- `runs/physics_prior/radiomap3dseer/auto_nlos_dispatch/scene_0_tx_0/recursive_debug.json`
- `runs/physics_prior/radiomap3dseer/auto_nlos_dispatch/scene_0_tx_0/physics_arrays/`

## Limitations

OWR-CD uses strict 2-D footprint visibility and a material-independent canonical UTD wedge approximation. OWR-RD uses rooftop knife-edge events and the project-defined one-way recursive combination. Neither model includes material-specific reflection, diffuse scattering, waveguide modes, or a learned calibration term.
