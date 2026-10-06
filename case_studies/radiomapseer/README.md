# RadioMapSeer selected case

- External root placeholder: `<RADIOMAPSEER_ROOT>`
- Config: `configs/radiomapseer.json`
- Case: `scene_id=0`, `tx_id=0`, IRT4 gain image
- Bundled truth: `real_gain_0_0.png`
- Bundled geometry excerpt: `buildings_0.json`

Re-run the case with:

```powershell
python run_physics_prior.py `
  --data-root <RADIOMAPSEER_ROOT> `
  --config configs/radiomapseer.json `
  --scene-ids 0 --tx-ids 0 --diffraction-method auto `
  --output-dir runs/case_studies/radiomapseer_scene0_tx0
```

The copied comparison is historical output from the previous cross-dataset
run.  New runner outputs distinguish raw `physics_prior_db.npy` from the
building-masked `physics_prior_evaluation_db.npy`.
