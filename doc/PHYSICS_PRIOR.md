# RadioMap3DSeer Deterministic Physics Prior

## Scope and inputs

This subproject implements the requested deterministic prior for the local RadioMap3DSeer layout. It reads:

- `../dataset/polygon/buildings/<scene_id>.json`: building polygons and absolute rooftop heights in metres;
- `../dataset/antenna/<scene_id>.json`: 80 transmitter records `[x, y, z]` in the dataset's bottom-origin map coordinates;
- `../dataset/gain/<scene_id>_<tx_id>.png`: the repository's 8-bit ground-truth gain map;
- `configs/radiomap3dseer.json`: the independent wireless and visualization configuration.

The implementation is self-contained in `radiomap_physics.py` and the executable entry point is `run_physics_prior.py`. It does not modify the neighbouring training or surrogate projects.

## Verified RadioMap3DSeer configuration

The values below were transcribed from Table I, Section II, Section IV and Appendix A of the supplied Yapar et al. 2024 PDF, then checked against the local file structure.

| Field | Value |
|---|---:|
| Map | 256 x 256 pixels, 256 x 256 m |
| Resolution | 1 m/pixel |
| Building heights | 6.6-19.8 m |
| Rx height | 1.5 m |
| Tx height | rooftop + 3 m |
| Carrier frequency | 3.5 GHz |
| Tx power | 23 dBm |
| Bandwidth | 20 MHz |
| Noise PSD / figure | -174 dBm/Hz / 20 dB |
| Antenna | isotropic |
| 3D simulation | IRT, at most 2 interactions, 10 m building tile, no cars |
| Simulation path-gain range | -162 to -75 dB |
| 3D dataset threshold | -104 dB |
| Analytic threshold | -111 dB |
| Derived RSS range | -139 to -52 dBm for 23 dBm Tx power |

The source-of-truth values are stored in JSON so another dataset can add an independent configuration with the same schema. The ground-truth PNG mapping is recorded as the existing repository convention `gray/255 -> [-162,-75] dB path-gain`; this is also the signed convention used by the main physics arrays and figures. Optional `*_loss_magnitude_db.npy` arrays expose positive magnitudes for numerical inspection. The PDF/local files do not independently expose any additional publisher post-processing, so this is explicitly marked as an evidence boundary rather than presented as a newly inferred physical fact.

## Coordinate and height handling

The local JSON coordinates use bottom-origin `(x, y)`, while NumPy image arrays use top-origin `[row, col]`. The implementation uses:

```text
col = x
row = (H - 1) - y
```

Polygon heights are rasterized after this y flip. Overlapping polygons use the higher rooftop height. Tx `z` is not inferred from an empty boundary pixel: for every local Tx, `z - 3 m` is audited against the exact polygon-height set. All 80 Tx records in each tested scene pass this audit.

## Algorithm

For each Rx pixel, `Rx=(x_r,y_r,1.5 m)` and `Tx=(x_t,y_t,z_t)` are represented in the same metre coordinate system.

1. FSPL uses the three-dimensional distance

   `d = sqrt((x_r-x_t)^2 + (y_r-y_t)^2 + (z_r-z_t)^2)`

   and the positive attenuation magnitude `L_FSPL_mag = 20 log10(4 pi d f / c)`.
   The stored signed map is `PL_FSPL = -L_FSPL_mag`.

2. The horizontal Tx-Rx line is sampled at no more than 1 m spacing, mapped to nearest grid pixels, and consecutive duplicate pixels are removed. Tx/Rx endpoints are not obstacle candidates.

3. For every sampled building height `H_i`, the LOS height is interpolated using the actual sampled horizontal distance `s_i`. Only `h_i = H_i - z_LOS,i > 0` is retained.

4. For every retained candidate `E_i=(x_i,y_i,H_i)`,

   `d1_i = ||Tx-E_i||_2` and `d2_i = ||E_i-Rx||_2`.

   Thus d1/d2 are 3D Euclidean propagation distances and include the vertical offsets to the candidate rooftop.

5. The implementation computes `nu_i`, keeps only `nu_max` and its `E*`, and applies the requested piecewise single knife-edge formula. It never sums losses from several buildings and does not run Deygout recursion.

6. `los_mask=True` means no valid `h_i>0` candidate was found. For this case signed
   `L_diff=0` and the stored `nu_max`/dominant quantities are NaN or -1 sentinels. Otherwise
   `physics_prior = FSPL + L_diff` in the paper's negative signed convention.

## Outputs

For each scene/Tx, the run stores:

- `physics_arrays/fspl_db.npy`
- `physics_arrays/fspl_paper_pl_db.npy` (same negative signed convention as `fspl_db.npy`)
- `physics_arrays/fspl_loss_magnitude_db.npy` (optional positive magnitude)
- `physics_arrays/diffraction_loss_db.npy`
- `physics_arrays/nu_max.npy`
- `physics_arrays/dominant_{row,col,height_m,h_m,d1_m,d2_m}.npy`
- `physics_arrays/los_mask.npy`
- `physics_arrays/physics_prior_db.npy`
- `physics_arrays/physics_prior_paper_pl_db.npy` (same negative signed convention as `physics_prior_db.npy`)
- `physics_arrays/physics_prior_loss_magnitude_db.npy` (optional positive magnitude)
- `physics_arrays/diffraction_loss_magnitude_db.npy` (optional positive magnitude)
- `ground_truth_pathgain_db.npy` (negative paper signed PL/pathgain)
- `ground_truth_pathloss_magnitude_db.npy` (positive magnitude used for comparison)
- `comparison.png` and `comparison.pdf`
- `propagation_profile.png` and `propagation_profile.pdf`
- `metadata.json`

The profile CLI option `--debug-rx row,col` accepts any image-array Rx coordinate. Without it, the script chooses the largest-ν NLOS Rx at least 32 m from the Tx so the default profile is informative rather than a one-building near-field slice.

## Run commands

From this directory on macOS/Linux:

```text
uv run run_physics_prior.py --data-root ../dataset --config configs/radiomap3dseer.json --scene-ids 0,1,2 --tx-ids 0 --output-dir runs/radiomap3dseer_physics_prior/3.5GHz_1m
```

Windows CMD uses the same single-line command. A multiline CMD form is:

```text
uv run run_physics_prior.py ^
  --data-root ..\dataset ^
  --config configs\radiomap3dseer.json ^
  --scene-ids 0,1,2 ^
  --tx-ids 0 ^
  --output-dir runs\radiomap3dseer_physics_prior\3.5GHz_1m
```

For an arbitrary debug Rx:

```text
uv run run_physics_prior.py --data-root ../dataset --scene-ids 0 --tx-ids 7 --debug-rx 120,180
```

## Verification status

The following were executed on 2026-10-04:

- `uv run --no-project --python ../sionna_osm/.venv/bin/python python test/test_physics_prior.py`: 9/9 PASS;
- `uv run --no-project --python ../sionna_osm/.venv/bin/python python -m py_compile radiomap_physics.py run_physics_prior.py test/test_physics_prior.py`: PASS;
- 3 scenes x 1 Tx were executed in all four modes in 55.07 s total, with per-sample runtimes 17.32 s, 18.09 s and 19.65 s;
- all selected scenes had 80 Tx records and passed the `z - 3 m` polygon-height audit;
- all generated PDFs were one-page files and were rendered with Poppler for visual inspection.

The normal `uv run pytest -q` command was not usable in the sandbox because it attempted to fetch packages from PyPI while DNS/network access was unavailable. The same test functions were directly executed using the local environment that contains NumPy, Pillow and Matplotlib; pytest itself was therefore not independently exercised here.

## Known limitations

- The prior is intentionally FSPL plus one of the deterministic knife-edge modes. It does not represent reflection, scattering, material loss, antenna patterns beyond the isotropic configuration, or multipath.
- The 1 m nearest-pixel path sampler is deterministic, but it is a grid approximation of the continuous line; changing the declared sampling step changes the discrete obstruction candidates.
- The local gain PNG is used as a comparison target, not as a claim that the analytic prior should reproduce WinProp IRT values. A quantitative accuracy claim needs an agreed publisher-label decoding and a larger scene/Tx evaluation.

## Diffraction mode extension

### dB sign convention

The supplied paper writes `PL=(P_Rx)_dB-(P_Tx)_dB`, so its reported PL/pathgain values are
negative, such as `-162` to `-75 dB`. The main FSPL, diffraction and Physics Prior maps now use
this signed negative convention directly: FSPL is the negative of the usual positive attenuation
magnitude, diffraction map contributions are non-positive, and `physics_prior_db=FSPL+diffraction`.
The recursive solver still records the positive `J(nu)` contribution internally because that is the
knife-edge formula; the final map contribution is signed negative. Optional `*_loss_magnitude_db`
arrays expose positive magnitudes only when needed for numerical inspection.

The current CLI keeps the original single-edge implementation and adds one unified selector:

```text
--diffraction-method none|single|owr-rd|owr-cd|deygout|auto
```

The five methods share the same rasterized building geometry, Tx/Rx coordinates, 3-D FSPL array,
frequency, wavelength, path sampling step and piecewise knife-edge function `J(nu)=0` for
`nu<=-0.78`, otherwise the formula specified in the experiment request.

| Mode | Solver | Recursive intervals | Combination |
|---|---|---|---|
| `none` | no diffraction | none | `L_prior=L_FSPL` |
| `single` | Single Dominant Knife-Edge | none | one root `E*=argmax(nu)` |
| `owr-rd` | **OWR-RD, One-Way Recursive Rooftop Diffraction** | only `E* -> B` | sum of selected forward `J(nu)` contributions |
| `owr-cd` | **OWR-CD, One-Way Recursive Corner Diffraction** | footprint corner chain toward Rx | canonical material-independent UTD wedge contributions |
| `deygout` | standard two-sided Deygout construction | both `A -> E*` and `E* -> B` | sum of selected recursive `J(nu)` contributions |

With `--diffraction-method auto`, LOS pixels use FSPL only. Each NLoS pixel is
dispatched independently using the configured Tx/Rx/building-height rule, with
the dominant root blocking roof as the local height reference: low,
height-similar street links use OWR-CD and other links use OWR-RD. A strict CD
corner chain that cannot legally reach the Rx is not recorded as zero loss; it
falls back to OWR-RD on the same geometry. Only a pathological case in which
the root NLoS profile also yields no rooftop event uses the root single edge as
an explicitly recorded `single-fallback`. The arrays
`resolved_method_map.npy` and `fallback_to_rd_mask.npy` make this dispatch
auditable per pixel.

All checked-in dataset configs use `max_corner_depth: null`. In that setting the
corner recursion is not limited by an arbitrary corner count; the visited
building-component guard and Rx visibility provide the finite geometric stop
condition. A finite integer is retained only as an explicit synthetic-test or
debug override.

Image-mask datasets may additionally set
`physics_prior.sampling_mask: "rx_observation"` or pass
`--rx-only-physics-prior`. The solver then evaluates diffraction only at the
dataset's `Rx/{scene_id}.png` mask and writes `physics_prior_observed_db.npy`
plus `physics_observation_mask.npy`. Pixels outside that mask are marked
`interpolated-rx` in the solver audit map, then completed by deterministic
USC-style two-dimensional interpolation. The USC Rx mask is irregular rather
than a rectangular lattice, so the runner uses piecewise-linear interpolation
on the observed Rx triangulation (the rectangular four-corner case is the
bilinear special case), with nearest-sample boundary completion outside the
convex hull. Crucially, only the NLOS diffraction field is completed: every
LOS pixel is restored to its exact FSPL value, while NLOS pixels use
`FSPL + interpolated(CD/RD diffraction)`. These pixels are not evidence of
zero diffraction.
The complete finite map is saved as `physics_prior_interpolated_db.npy` and
can be used directly or as a downstream data-driven model input.

For every recursive interval the implementation rebuilds the interval reference LOS and recomputes
`h`, 3-D `d1`, 3-D `d2` and `nu`; parent values are never reused. For a recursive child, `d1`
is the 3-D Euclidean distance from the current child start `A` to `E`, and `d2` is the 3-D
Euclidean distance from `E` to the current child end `B`. At the root these are Tx-to-E and
E-to-Rx. The endpoint height is the absolute rooftop/receiver height carried by that interval.

OWR-RD is a project-specific simplified one-way recursive approximation. It must not be called
Deygout: its left branch is intentionally not evaluated. The Deygout mode follows the classical
predominant-edge construction and recurses on both sides. The historical basis is J. Deygout,
“Multiple knife-edge diffraction of microwaves,” IEEE Transactions on Antennas and Propagation,
14(4), 480–489, 1966, DOI [10.1109/TAP.1966.1138719](https://doi.org/10.1109/TAP.1966.1138719).
The ITU-R P.526 recommendation describes the general multiple-knife-edge method based on the
Deygout construction; see [ITU-R P.526-5](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.526-5-199708-S%21%21PDF-E.pdf)
and the current [P.526 recommendation page](https://www.itu.int/rec/R-REC-P.526/en). This implementation
uses the user-specified unified `J(nu)` and local 2-D building profile, so it does not claim to
implement every ITU terrain/earth-curvature/empirical correction.

A practical raster guard is applied only after a recursive edge is selected: the selected 8-connected
building component is excluded from child intervals. This prevents a flat roof represented by many
pixels from becoming many artificial knife edges. The root `single` profile still evaluates all
sampled candidates exactly as before. Each event stores depth, A, B, E*, component ID, `h`, `d1`,
`d2`, `nu` and `J(nu)` in `recursive_debug.json`.

Use all modes in one shared-geometry run:

```text
uv run --no-project --python ../sionna_osm/.venv/bin/python python run_physics_prior.py \
  --data-root ../dataset --config configs/radiomap3dseer.json \
  --scene-ids 0,1,2 --tx-ids 0 --compare-all-modes \
  --output-dir runs/radiomap3dseer_diffraction_modes/3.5GHz_1m
```

The comparison run writes `modes/{none,single,owr-rd,deygout,owr-cd}/physics_arrays`, a fixed-color-scale
`comparison_all_modes.png/.pdf`, `recursive_debug_all_modes.png/.pdf`, and per-sample
`metadata.json`, `recursive_debug.json` and pairwise mode statistics. The recursive debug figure
contains the original building profile and Tx-Rx LOS, the single dominant edge, the OWR-RD chain,
and the Deygout depth-marked tree; the JSON is the authoritative full record when the tree has more
labels than can fit in a paper figure.
