# Strict OWR-CD geometry validation and RadioMapSeer rerun

## Geometry contract

OWR-CD is now a local recursive path construction, not a map-wide corner graph:

1. Start at Tx and test the current-point to Rx segment.
2. Take the first raster blocking component on that segment.
3. Keep only that component's visible tangent/silhouette vertices.
4. Reject a corner if the incoming segment is blocked, overlaps a wall, does
   not progress toward Rx, or if the outgoing segment re-enters the same
   building or follows its wall.
5. Rank legal corners deterministically by total geometric route length, turn
   angle, UTD loss, and row/column tie-breakers.
6. Re-test the new corner to Rx segment. If the same component is encountered
   again, return `same_building_reentry`; do not walk to another corner of that
   building.

The solver does not build a Dijkstra graph and does not use GT or learned
parameters. A complete path may contain corners from successive blocking
buildings only. The default `max_corner_depth` is 3. If a complete route is
not established, the Rx is `unresolved NLOS`; partial corner loss is excluded
from the physics map and the pixel is recorded in `unresolved_nlos_mask.npy`.

Raster-only footprints are traced from exposed cell edges and simplified into
polygons. UTD is evaluated only after a legal geometric corner has been
selected, using the existing canonical material-independent wedge model.

## Typical Rx inspection before full-map execution

Command:

```text
python inspect_owr_cd_paths.py --data-root G:/paper/radiomapseer --config configs/radiomapseer.json --scene-id 0 --tx-id 0 --rx-pairs "0,107;32,107;128,32" --output-dir runs/cross_dataset_physics_prior/radiomapseer/strict_owr_cd_typical_paths_v2
```

Observed paths:

| Rx | first blocker | accepted path | termination |
|---|---:|---|---|
| (0,107) | B34 | Tx -> C(90,107)/B34 -> C(54,104)/B23 | unresolved: B13 outgoing rays re-enter B13 |
| (32,107) | B34 | Tx -> C(90,107)/B34 -> C(54,104)/B23 | unresolved: B13 outgoing rays re-enter B13 |
| (128,32) | B37 | Tx -> C(147,48)/B37 -> Rx | rx_visible |

The diagnostic JSON contains every visible/non-visible candidate and rejection
reason. The PNG/PDF overlays the original Tx-Rx LOS, all footprint outlines,
rejected silhouette candidates, selected corners, building IDs, and the final
free-space path. No same-building corner chain is accepted.

## Full RadioMapSeer run

Output:

```text
runs/cross_dataset_physics_prior/radiomapseer/strict_owr_cd_final_v2
```

The comparison metrics use the repository's P_NLoS report rule, including the
fixed building-interior signed pathgain value:

| method | RMSE (dB) | MAE (dB) | Bias (dB) |
|---|---:|---:|---:|
| FSPL | 13.278 | 9.318 | 7.687 |
| Single | 23.260 | 17.834 | -17.302 |
| OWR-RD | 28.509 | 20.841 | -20.311 |
| Deygout | 39.807 | 29.464 | -28.935 |
| OWR-CD | 19.508 | 12.975 | 1.241 |

OWR-CD NLoS RMSE is `20.941 dB`, NLoS MAE is `14.906 dB`, and NLoS Bias is
`1.198 dB`. Its reported minimum is `-206.101 dB`, and `9.43%` of all pixels
are below the GT minimum (`11.08%` within the method-specific NLoS region).

The raw solver record, before the report-only building-interior replacement,
is `RMSE=26.768 dB`, `MAE=21.689 dB`, `Bias=11.503 dB`, `LOS RMSE=8.117 dB`,
`NLoS RMSE=28.822 dB`, minimum `-206.101 dB`, and below-GT-minimum ratio
`9.52%`. `71.29%` of map pixels are explicitly unresolved NLoS and have zero
OWR-CD diffraction contribution rather than an invented continuation.

Figures and tables:

```text
runs/cross_dataset_physics_prior/radiomapseer/strict_owr_cd_final_v2/metrics/physics_prior_NLoS_comparison.png
runs/cross_dataset_physics_prior/radiomapseer/strict_owr_cd_final_v2/metrics/metrics_summary.csv
runs/cross_dataset_physics_prior/radiomapseer/strict_owr_cd_final_v2/scene_0_tx_0/corner_recursive_debug_owr_cd.png
```

## OWR-RD regression

RadioMap3DSeer `auto` still resolves to `owr-rd`. For scene 0 / Tx 0, the
final auto arrays and the previous `modes/oneway` arrays have maximum absolute
differences of zero for diffraction loss, physics prior, edge count, and LOS
mask. OWR-RD therefore remains an independent rooftop-profile solver.
