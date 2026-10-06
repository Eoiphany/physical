# Reproducible dataset case studies

These directories contain small, selected case bundles rather than complete
datasets.  The repository follows `AGENTS.md`: full datasets remain external
and are supplied with `--data-root`.  Each bundle includes the original gain
image for the selected sample, the minimum geometry/input files needed to
identify the case, generated diagnostics, and a README with the external
source layout.

Active datasets in this revision are:

- `radiomapseer/`: 2D footprint case, street-level Tx/Rx; `auto` is selected
  per receiver and normally attempts OWR-CD first.
- `urbanradio3d/`: extracted UrbanRadio3D directory case; the Tx height and
  label dB conversion remain the explicit proxies recorded by its config.
- `usc/`: image-only case with a fixed-height building/Tx proxy; Boston and
  UCLA are intentionally excluded from the active comparison because their
  input/label quality was not considered reliable for this study.
- `ppdata5d/`: provisional PPData5D geometry case.  Its inspected NPZ exposes
  occupancy and terrain, while Tx metadata and RSS-to-dB calibration were not
  verified, so it reports geometry diagnostics but no signed-dB RMSE.

Run the PPData5D case from the repository root with:

```powershell
python case_studies/ppdata5d_case_study.py `
  --data-root <PPDATA5D_ROOT> `
  --sample T06C0D0001_n02_f00_z00 `
  --output-dir runs/cross_dataset_physics_prior/ppdata5d/T06C0D0001_n02_f00_z00_proxy
```

For the standard loader-backed cases, use the commands recorded in each
case's `README.md`; replace `<DATA_ROOT>` with a local dataset directory.

`physics_prior_db.npy` is the raw deterministic solver output and preserves
`prior_db = fspl_db + diffraction_loss_db`.  The protocol-level display and
metric map is `physics_prior_evaluation_db.npy`, where every building pixel
(`height_map_m > 0`) is set to the configuration's minimum signed pathgain.
