# OWR-RD / OWR-CD deterministic Physics Prior experiment

## Implementation

The former one-way recursive solver is named `owr-rd` (One-Way Recursive
Rooftop Diffraction) without changing its interval recursion.  `owr-cd`
(One-Way Recursive Corner Diffraction) was added for street-level links.

OWR-CD first tests the current-point to Rx segment against connected raster
building components.  It then selects the first blocking component and walks
through deterministic footprint vertices that are visible from the current
point, have positive forward progress, and have not appeared in
`visited_corners`.  A finite rectangular footprint may therefore contribute
two or more boundary corners before the route becomes visible.  The recursion
stops at Rx visibility, no valid corner, no progress, or `max_corner_depth`.

Vector polygon footprints are used when supplied by the loader.  For image-only
maps, exposed raster cell edges are traced into connected-component polygons
and simplified with RDP; individual building pixels are not treated as corners.

Each corner uses a deterministic, material-independent canonical UTD wedge
approximation.  The wedge angle, incident angle, diffraction angle, wavelength,
and metric d1/d2 distances enter the coefficient.  No unknown material
reflection parameter and no GT-fitted weight are used.

The dispatcher is geometry-only:

```text
max(Tx_z, Rx_z) <= 0.5 * max(building_height)
and abs(Tx_z - Rx_z) <= max(2 m, 0.15 * max(building_height)) -> owr-cd
otherwise -> owr-rd
```

An audited Tx height at or above its rooftop height takes the `owr-rd` branch.
The CLI is `--diffraction-method none|single|owr-rd|owr-cd|deygout|auto`.

## RadioMapSeer

Command:

```text
python run_physics_prior.py --data-root G:/paper/radiomapseer --config configs/radiomapseer.json --scene-ids 0 --tx-ids 0 --compare-all-modes --output-dir runs/cross_dataset_physics_prior/radiomapseer/basic_5.9GHz_1m_irt4_owr_cd
```

The representative OWR-CD recursion is saved as
`scene_0_tx_0/corner_recursive_debug_owr_cd.png/.pdf`, with footprint outlines,
Tx/Rx, and the `Tx -> C1 -> ... -> C7 -> Rx` route.  The run also writes
`corner_count.npy`, LOS/NLOS masks, signed diffraction/prior maps, and the
five-method comparison figure.

| method | full-map RMSE (dB) | full-map MAE (dB) | Bias (dB) | NLoS RMSE (dB) |
|---|---:|---:|---:|---:|
| FSPL | 13.278 | 9.318 | 7.687 | N/A (none has no NLoS mask) |
| Single | 23.260 | 17.834 | -17.302 | 25.044 |
| OWR-RD | 28.509 | 20.841 | -20.311 | 30.767 |
| Deygout | 39.807 | 29.464 | -28.935 | 43.058 |
| OWR-CD | 41.353 | 26.071 | -24.813 | 44.738 |

OWR-CD prediction minimum is `-360.149 dB`; `41.43%` of all pixels are below
the GT minimum.  This is an honest negative result for the current canonical
UTD approximation on this dataset: it captures a corner-route mechanism, but
its material-independent field coefficient is too punitive here and does not
beat FSPL.

Metrics and the expanded seven-map figure are in
`runs/cross_dataset_physics_prior/radiomapseer/basic_5.9GHz_1m_irt4_owr_cd/metrics`.

## RadioMap3DSeer regression

For scene 0 / Tx 0, `--diffraction-method auto` resolved to `owr-rd`.
Auto and explicit `owr-rd` have zero maximum absolute difference for
`diffraction_loss_db`, `physics_prior_db`, and `edge_count`; their LOS masks
are also identical.  Comparing explicit `owr-rd` to the previous repository
output under `modes/oneway` gives the same zero-difference result for all four
checked arrays.  The new result therefore changes the public name and adds the
dispatcher without changing the rooftop solver.

Outputs:

```text
runs/cross_dataset_physics_prior/radiomap3dseer/3.5GHz_1m_auto_owr_rd
runs/cross_dataset_physics_prior/radiomap3dseer/3.5GHz_1m_explicit_owr_rd
```
