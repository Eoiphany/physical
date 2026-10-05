# Cross-dataset physics-prior smoke reproduction — 2026-10-05

## Purpose

验证统一 `radiomap_physics.py` / `run_physics_prior.py` 是否可以在本机已有的 USC、Boston、UCLA、RadioMapSeer 和 UrbanRadio3D ZIP 数据上完成最小四 mode 物理先验流程，同时不复制 solver、不把数据集绝对路径写入仓库配置或日志。

## Configuration changes

- Added `configs/usc.json`, `configs/boston.json`, `configs/ucla.json`, `configs/radiomapseer.json`, and `configs/rm_data.json`.
- Added config-driven loader formats: `image_mask`, `polygon_mask_json`, and extracted `urbanradio3d_directory`; the original `polygon_height_json` path remains unchanged.
- Added a finite near-field floor of one declared pixel in FSPL. This is needed when a 2D proxy places Tx and Rx at the same pixel and height, where the continuous FSPL expression is undefined.
- Added `doc/CROSS-DATASET-CONFIGS.md` with layout, evidence boundaries, and command templates.

## Input and proxy boundaries

- RadioMapSeer 2D uses the locally documented 5.9 GHz / 23 dBm / 10 MHz / 1.5 m Tx-Rx / 25 m building setup and IRT4 labels decoded from `-127..-47 dB`.
- USC, Boston, and UCLA expose only 2D `map/Tx/pmap` images in this workspace. Their 10 m building height, 1.5 m Tx/Rx height, 2.5 GHz, and `-254..0 dB` label interval are explicitly marked as proxies or local target conventions.
- RM_data is read from the extracted `Building_Infomation/buildings/*.json` and `PathLoss_h1_train/h1/*.png` directories. Building heights come from the polygon JSON; the local dataset has no wireless manifest, so the 20 m Tx height and `-160..-40 dB` label interval are explicit proxies. Its filename axes are mapped as image `row=255-X,col=Y`, i.e. physics `x=Y,y=X`.

## Commands

For each dataset, the runtime command had the form:

```powershell
& .\.venv\Scripts\python.exe run_physics_prior.py `
  --data-root $env:DATA_ROOT `
  --config configs\<dataset>.json `
  --scene-ids <scene> --tx-ids 0 --compare-all-modes `
  --output-dir runs\cross_dataset_physics_prior\<dataset>\<experiment>
```

The selected smoke samples were scene `1` / Tx `0` for USC, Boston, and UCLA, and scene `0` / Tx `0` for RadioMapSeer and RM_data. The metrics command was then run for each output using `visualize_physics_prior_metrics.py`.

## Results

| Dataset | Output | Four-mode runtime | Metrics |
|---|---|---:|---|
| USC | `runs/cross_dataset_physics_prior/usc/basic_2.5GHz_0.86m` | 34.53 s | PASS |
| Boston | `runs/cross_dataset_physics_prior/boston/basic_2.5GHz_0.86m` | 35.73 s | PASS |
| UCLA | `runs/cross_dataset_physics_prior/ucla/basic_2.5GHz_0.86m` | 16.07 s | PASS |
| RadioMapSeer | `runs/cross_dataset_physics_prior/radiomapseer/basic_5.9GHz_1m_irt4` | 30.68 s | PASS |
| UrbanRadio3D ZIP | `runs/cross_dataset_physics_prior/rm_data/basic_h1_3.5GHz_1m` | 29.26 s | PASS |

Every output contains the four mode arrays, recursive debug JSON/figure, comparison PNG/PDF, per-sample metadata, summary, and metrics PNG/PDF/CSV/JSON. A post-run checker verified finite values, `fspl_db <= 0`, `diffraction_loss_db <= 0`, `none: prior_db == fspl_db`, and `prior_db == fspl_db + diffraction_loss_db`; the maximum absolute floating-point identity error was `7.63e-6 dB`.

## Verification

- `python test/test_physics_prior.py`: 10/10 PASS.
- `python -m py_compile radiomap_physics.py run_physics_prior.py visualize_physics_prior_metrics.py test/test_physics_prior.py`: PASS.
- JSON parsing for all six configs: PASS.
- No dataset absolute path, user path, credential, or API key was written into `configs/*.json`.

## Known limitations

These are smoke reproductions of the deterministic prior, not claims of exact ray-tracing reproduction for the 2D proxy datasets or the current UrbanRadio3D archive decoding. For publishable cross-dataset accuracy, the original wireless parameter tables, Tx z semantics, building PNG encoding, and label dB conversion for each source must be independently verified.
