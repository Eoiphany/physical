# PPData5D selected provisional case

- External root placeholder: `<PPDATA5D_ROOT>`
- Sample: `T06C0D0001_n02_f00_z00`
- Bundled real RSS image: `real_rss_gain.png`
- Bundled environment: `environment_bdtr.npz`
- Generated figure: `case_study.png` / `case_study.pdf`

Run from the repository root:

```powershell
python case_studies/ppdata5d_case_study.py `
  --data-root <PPDATA5D_ROOT> `
  --sample T06C0D0001_n02_f00_z00 `
  --output-dir runs/cross_dataset_physics_prior/ppdata5d/T06C0D0001_n02_f00_z00_proxy
```

This case is intentionally marked provisional.  The inspected NPZ contains
`inBldg_zyx` and `terrain_yx`, and the PNG is RSS.  Tx position/absolute height
and RSS-to-dB calibration were not verified, so no signed-dB RMSE is reported.
The displayed Physics Prior uses an explicit 20 m building-height proxy and
argmax-RSS Tx proxy; see `case_summary.json`.
