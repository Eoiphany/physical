# UrbanRadio3D selected case

- External root placeholder: `<URBANRADIO3D_ROOT>` (extracted directory only)
- Config: `configs/rm_data.json`
- Case: `scene_id=0`, first sorted train label `0_X10_Y77.png`
- Bundled truth: `real_gain_0_X10_Y77.png`
- Bundled geometry excerpt: `buildings_0.json`

The filename convention is preserved by the loader: filename `X` is the image
row axis and `Y` is the image column axis, so the physics Tx is `(x=Y,y=X)`.
The config documents the explicit Tx-height and gray-to-dB proxy assumptions
because the extracted directory does not include a wireless parameter file.

```powershell
python run_physics_prior.py `
  --data-root <URBANRADIO3D_ROOT> `
  --config configs/rm_data.json `
  --scene-ids 0 --tx-ids 0 --diffraction-method auto `
  --output-dir runs/case_studies/urbanradio3d_scene0_tx0
```
