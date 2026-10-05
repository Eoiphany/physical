# Cross-dataset physics-prior configurations

本目录的唯一物理实现仍是 `radiomap_physics.py`，不同数据集只通过 `configs/*.json` 的 `data.format` 和数据模板切换。`--data-root` 必须由命令行传入，JSON 不保存本机数据集绝对路径。

## Local dataset mapping

| Config | External `--data-root` | Local layout | Geometry status | Basic smoke sample |
|---|---|---|---|---|
| `radiomap3dseer.json` | `G:/paper/radiomap3dseer` | `polygon/buildings`, `antenna`, `gain` | 3D polygon heights and Tx z are authoritative | scenes `0,1,2`, Tx `0` |
| `radiomapseer.json` | `G:/paper/radiomapseer` | `polygon/buildings_complete`, `antenna`, `gain/IRT4` | 2D footprints; fixed 25 m building height and 1.5 m Tx/Rx are dataset parameters | scene `0`, Tx `0` |
| `usc.json` | `G:/paper/usc` | `map`, `Tx`, `pmap` | 2D image-only proxy: fixed 10 m building and 1.5 m Tx/Rx | scene `1`, Tx `0` |
| `boston.json` | `G:/paper/Boston` | `map`, `Tx`, `pmap` | 2D image-only proxy: fixed 10 m building and 1.5 m Tx/Rx | scene `1`, Tx `0` |
| `ucla.json` | `G:/paper/UCLA` | `UCLA_crop/map`, `UCLA_crop/Tx`, `UCLA_crop/pmap` | 2D image-only proxy: fixed 10 m building and 1.5 m Tx/Rx | scene `1`, Tx `0` |
| `rm_data.json` | `G:/paper/RM_data` | `Building_Infomation/buildings/*.json`, `PathLoss_h*_train/h*/*.png` | building heights come from extracted polygon JSON; UrbanRadio3D filename axes are mapped as `x=Y,y=X`; Tx z and label dB interval remain explicit proxies | scene `0`, Tx index `0` |

The 2D RadioMapSeer parameters follow the local reproduction note: 5.9 GHz, 23 dBm, 10 MHz, 1.5 m Tx/Rx, 25 m buildings, and `-127 + gray/255 * 80 dB` for the IRT4 labels. The USC/Boston/UCLA pmap interval follows the local surrogate configuration (`-254..0 dB`) and is recorded as a target-dB convention rather than a newly recovered Wireless Insite calibration.

RM_data is read in-place from the extracted directory. Its label names encode `scene_Xx_Yy`; the actual image convention is `row=255-X,col=Y`, so the physics coordinates are `x=Y,y=X`. `--tx-ids` is the deterministic sorted index of matching labels for the selected receiver-height layer. Because the dataset has no wireless parameter manifest, the configured Tx height and grayscale dB interval are marked as proxies in both the JSON and run metadata.

## Smoke commands

From the repository root, with `DATA_ROOT` set to the external dataset directory:

```powershell
& .\.venv\Scripts\python.exe run_physics_prior.py `
  --data-root $env:DATA_ROOT `
  --config configs\radiomapseer.json `
  --scene-ids 0 --tx-ids 0 --compare-all-modes `
  --output-dir runs\cross_dataset_physics_prior\radiomapseer\basic_5.9GHz_1m_irt4
```

Replace the data root/config/output directory for the other rows. For RM_data, the `h1` layer is selected by `configs/rm_data.json`; change `data.rx_height_level` only when a different receiver-height archive is intended.

The output is the same five-method structure as the RadioMap3DSeer reproduction: `modes/{none,single,owr-rd,deygout,owr-cd}/physics_arrays`, `recursive_debug.json`, `comparison_all_modes.png/.pdf`, `metadata.json`, and `summary.json`.
