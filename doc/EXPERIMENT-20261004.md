# Experiment log: deterministic RadioMap3DSeer Physics Prior

## Purpose / hypothesis

Hypothesis: a deterministic 3D FSPL baseline plus one dominant single knife-edge diffraction term can be generated from the RadioMap3DSeer polygon heights and transmitter metadata without using a learned model or ray tracer. The run is intended to validate implementation consistency and produce inspectable physical intermediates, not to claim agreement with WinProp IRT.

## Change from the previous version

- Added the standalone `physical/` subproject and independent RadioMap3DSeer JSON configuration.
- Added 3D FSPL and single dominant knife-edge diffraction with no multi-edge loss summation.
- Corrected the dataset coordinate convention after inspecting local antenna PNGs: JSON y is bottom-origin and image row is `255-y`.
- Added Tx height auditing against polygon rooftop heights because some Tx points lie on/just outside a rasterized building boundary.
- Added fixed, cross-sample visualization limits instead of per-image normalization.
- Added raw physical arrays, debug profiles, tests and documentation.

## Data and configuration

- Dataset root: supplied externally at runtime through `--data-root`; no dataset path is committed.
- Selected scenes: `0,1,2`.
- Selected Tx: `0` in each scene.
- Map: 256 x 256 pixels at 1 m/pixel.
- Frequency: 3.5 GHz; Tx power: 23 dBm; Rx height: 1.5 m; Tx: rooftop + 3 m.
- Building heights: 6.6-19.8 m from polygon JSON.
- Path sampler: 1 m maximum spacing; candidate E is the sampled pixel rooftop.
- Ground-truth comparison: repository-established gray-to-path-gain mapping `[-162,-75] dB`, then sign-inverted to positive path loss.

## Commands

Actual offline validation command:

```text
MPLCONFIGDIR=/private/tmp/mplconfig UV_CACHE_DIR=/private/tmp/radiomap_uv_cache uv run --no-project --python ../sionna_osm/.venv/bin/python python run_physics_prior.py --data-root ../dataset --config configs/radiomap3dseer.json --scene-ids 0,1,2 --tx-ids 0 --output-dir runs/radiomap3dseer_physics_prior/3.5GHz_1m
```

Tests and compile checks:

```text
MPLCONFIGDIR=/private/tmp/mplconfig UV_CACHE_DIR=/private/tmp/radiomap_uv_cache uv run --no-project --python ../sionna_osm/.venv/bin/python python test/test_physics_prior.py
MPLCONFIGDIR=/private/tmp/mplconfig UV_CACHE_DIR=/private/tmp/radiomap_uv_cache uv run --no-project --python ../sionna_osm/.venv/bin/python python -m py_compile radiomap_physics.py run_physics_prior.py test/test_physics_prior.py
```

## Results

| Scene/Tx | Runtime (s) | Debug Rx (row,col) | NLOS ratio | FSPL range (dB) | Diffraction max (dB) | Prior max (dB) | GT positive path-loss range (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0/0 | 3.32 | (171,207) | 0.7986 | 69.71-91.35 | 43.84 | 128.75 | 80.46-162.00 |
| 1/0 | 3.09 | (130,181) | 0.9256 | 69.81-89.50 | 43.96 | 128.10 | 82.85-162.00 |
| 2/0 | 3.08 | (93,156) | 0.9179 | 68.66-90.72 | 43.53 | 128.26 | 77.05-162.00 |

Total runtime was 9.51 s. The selected default debug profiles are NLOS and include candidate obstacles, dominant E*, h, d1, d2 and νmax. The array invariant `physics_prior_db == fspl_db + diffraction_loss_db` and the analytic test cases were checked; the invariant check is repeated in the final validation command/report.

## Conclusion

The deterministic prior is implemented and reproducibly runs on real RadioMap3DSeer data. The numerical and visual outputs are internally consistent with the requested formulas and coordinate conversion. Agreement with the WinProp ground-truth maps remains unestablished because the local PNG post-processing and full ray-tracing physics are outside this prior; this is an explicit limitation, not a failed implementation claim.

## Traceability

- Baseline experiment: no prior implementation existed in `physical/`.
- Baseline/current commit: pending user-controlled Git commit.
- Dataset version: local RadioMap3DSeer directory inspected on 2026-10-04.
- Config: `configs/radiomap3dseer.json`.
