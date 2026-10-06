# USC selected case

- External root placeholder: `<USC_ROOT>`
- Config: `configs/usc.json`
- Case: image id `1`
- Bundled files: `real_gain_1.png`, `building_mask_1.png`, `tx_mask_1.png`

The local USC package is image-only.  The fixed building and Tx heights are
explicit proxies, and its README records that pmap targets include bilinear
interpolation.  It is retained as an active proxy case; Boston and UCLA are
not included in the active study.

```powershell
python run_physics_prior.py `
  --data-root <USC_ROOT> `
  --config configs/usc.json `
  --scene-ids 1 --tx-ids 0 --diffraction-method auto `
  --output-dir runs/case_studies/usc_scene1_tx0
```
