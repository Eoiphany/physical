# OWR-CD strict algorithm re-audit — 2026-10-05

## Scope

This experiment re-audits and reimplements OWR-CD against the strict definition:
`CurrentPoint -> first blocking building -> silhouette/tangent corner -> re-check LOS -> next building only`.
No GT, learned gate, visibility graph, Dijkstra, reflection, scattering, or fitted electromagnetic parameter is used.

## Violations found in the previous implementation

1. First blocking was selected from raster samples. Boundary-pixel aliasing could make a corner out-ray appear to re-enter the previous building or select a blocker inconsistent with the footprint geometry.
2. Silhouette selection used a broad visible-vertex pair heuristic without an explicit circular tangent interval. It was not sufficiently explicit about the two current-point tangent boundaries.
3. Candidate progress incorrectly required `|C-Rx| < |CurrentPoint-Rx|`, rejecting legal detours whose outgoing segment is farther from Rx than the original straight-line distance.
4. The corner direction convention was reversed: `corner -> CurrentPoint` was treated as the incident propagation direction. Near-straight `Tx -> C -> Rx` paths therefore received an almost-pi diffraction angle and could produce an erroneous zero-loss corner.
5. The recursion stopped at `max_corner_depth` without rechecking whether the last selected corner already made Rx visible.
6. Full-map output did not expose CD validity or the first blocking building as arrays.

OWR-RD code and its recursive interval semantics were not changed.

## Implemented algorithm

- `_segment_entry_parameter` performs an exact open-segment polygon-interior intersection test.
- `_first_blocking_polygon` scans only the current `CurrentPoint -> Rx` segment and returns the earliest footprint hit, with deterministic building-ID tie breaking.
- `_silhouette_corner_indices` keeps current-point-visible vertices and selects the endpoints of the smallest circular angular interval as tangent/silhouette corners.
- Each candidate is tested independently for incoming blockage, wall overlap, outgoing re-entry, outward departure, and forward projection.
- A candidate is scored deterministically by route distance, turn angle, canonical UTD loss, and row/column tie breakers.
- `visited_components` forbids `B1 -> C1 -> C2(B1)` recursion. Only a different first blocker can produce the next event.
- After the third corner, the solver performs one additional LOS check; it returns `rx_visible` when the chain is complete, otherwise `max_corner_depth` or `no_valid_corner`.
- Canonical material-independent wedge UTD is evaluated at accepted vertical-edge projections. The incident vector is `CurrentPoint -> C`; the outgoing vector is `C -> Rx`.

The solver does not construct a global corner graph and does not follow a building boundary.

## Debug validation

The offline test suite has 16 passing tests, including:

- Case A: direct LOS, zero CD events;
- Case B: one valid `Tx -> C1(B1) -> Rx` chain;
- Case C: `Tx -> C1(B1) -> C2(B2) -> Rx` with distinct building IDs;
- Case D: one silhouette candidate rejected and another independently selected on the same blocker;
- Case E: all candidates invalid, returning `no_valid_corner` without wall walking;
- automatic OWR-CD/OWR-RD dispatch;
- signed-map identity and existing RD/Deygout regression tests.

## RadioMapSeer run

Output directory:
`runs/cross_dataset_physics_prior/radiomapseer/strict_owr_cd_reaudit_20261005_v2`

The complete run compared `none`, `single`, `owr-rd`, `deygout`, and `owr-cd` on scene 0 / Tx 0. Metrics and figures are under `metrics/`; recursive paths are under `typical_paths/`.

| Method | Scope | RMSE (dB) | MAE (dB) | Bias (dB) | Prediction min (dB) |
|---|---:|---:|---:|---:|---:|
| FSPL | all | 13.2781 | 9.3177 | 7.6871 | -127.0000 |
| Single | all | 23.2602 | 17.8341 | -17.3024 | -138.0550 |
| OWR-RD | all | 28.5090 | 20.8408 | -20.3111 | -212.4693 |
| Deygout | all | 39.8073 | 29.4642 | -28.9347 | -219.3511 |
| OWR-CD | all | 47.4113 | 30.3673 | -22.8900 | -290.3361 |
| OWR-CD | NLOS | 51.5316 | 35.6015 | -27.4272 | -290.3361 |

The previous strict implementation output `strict_owr_cd_final_v2` had OWR-CD all RMSE 19.5080 dB and NLOS RMSE 20.9415 dB. The increase is intentionally not hidden or tuned away: the corrected direction convention and stricter geometry produce substantially larger canonical wedge losses for some near-tangent paths. This is a physical-model limitation to investigate separately, not a reason to use GT fitting.

For the corrected full map:

- LOS ratio: 15.6937%; NLOS ratio: 84.3063%;
- CD validity, including LOS: 54.5258%;
- valid NLOS ratio: 38.8321% of all pixels;
- unresolved NLOS ratio: 45.4742% of all pixels, or 53.8865% of NLOS pixels;
- first-blocking-building coverage: 84.3063%;
- mean corner count: 1.2209; maximum: 3;
- `prior_db == fspl_db + diffraction_loss_db`: verified;
- `cd_validity_mask.npy`, `first_blocking_building_id.npy`, `corner_count.npy`, `diffraction_loss_db.npy`, `los_mask.npy`, and `unresolved_nlos_mask.npy` are saved under `modes/owr-cd/physics_arrays/`.

Generated figures:

- `scene_0_tx_0/comparison_all_modes.png/.pdf`;
- `scene_0_tx_0/owr_cd_diagnostics.png/.pdf`;
- `scene_0_tx_0/corner_recursive_debug_owr_cd.png/.pdf`;
- `typical_paths/rx_0_107/owr_cd_path.png/.pdf`;
- `typical_paths/rx_32_107/owr_cd_path.png/.pdf`;
- `typical_paths/rx_128_32/owr_cd_path.png/.pdf`;
- `metrics/physics_prior_NLoS_comparison.png/.pdf` and `metrics_summary.csv`.

The typical path records include the original LOS, first blocker, silhouette candidates, reject reasons, selected building/corner, directions, wedge angle, loss, and termination.

## RadioMap3DSeer regression

Run output:
`runs/cross_dataset_physics_prior/radiomap3dseer/strict_owr_cd_reaudit_20261005_auto`

The dispatcher resolved `auto -> owr-rd`. Compared with the previous OWR-RD arrays, the following values had zero maximum absolute difference: `diffraction_loss_db`, `physics_prior_db`, `edge_count`, `los_mask`, `dominant_row`, and `dominant_col`. Floating arrays containing NaNs were numerically identical with `equal_nan=True` and had zero finite-value difference.

## Reproduction command

```text
.venv/Scripts/python.exe run_physics_prior.py ^
  --data-root <RADIOMAPSEER_ROOT> ^
  --config configs/radiomapseer.json ^
  --scene-ids 0 --tx-ids 0 --compare-all-modes ^
  --output-dir runs/cross_dataset_physics_prior/radiomapseer/<experiment_id>
```

Data remains external; no machine-specific dataset path is part of the repository configuration.