# Wireless configuration summary

The table below separates verified/configured values from proxy values.  For
the 2-D packages, a height is not recovered from the image; it is an explicit
configuration assumption.

| Dataset / case | Tx height | Rx height | Building height used by prior | Frequency | Resolution | Rx mask / selection expectation |
|---|---:|---:|---:|---:|---:|---|
| RadioMapSeer scene 0 / tx 0 | 1.5 m absolute proxy | 1.5 m absolute proxy | 25 m fixed 2-D proxy | 5.9 GHz | 1 m/px | no finite mask / OWR-CD first on NLoS |
| RadioMap3DSeer scene 0 | 20.1187–22.3359 m actual; antenna z is rooftop + 3 m | 1.5 m configured | 6.6–19.8 m polygon rooftops | 3.5 GHz | 1 m/px | no finite mask / OWR-RD |
| UrbanRadio3D scene 0 / tx 0 | 20 m fixed proxy; label archive has no Tx z | 1 m configured by `h1` | 6.6–19.8 m polygon heights | 3.5 GHz | 1 m/px | no finite mask / OWR-RD |
| USC image 1 | 1.5 m fixed proxy; Tx PNG has no z | 1.5 m fixed proxy | 10 m fixed 2-D proxy | 2.5 GHz | 0.86 m/px | 1,216 / 65,536 = 1.86% / OWR-CD first on NLoS |
| Boston scene 1 / tx 0 | 1.5 m fixed proxy | 1.5 m fixed proxy | 10 m fixed 2-D proxy | 2.5 GHz | 0.86 m/px | 480 / 65,536 = 0.73% / OWR-CD first on NLoS |
| UCLA scene 1 / tx 0 | 1.5 m fixed proxy | 1.5 m fixed proxy | 10 m fixed 2-D proxy | 2.5 GHz | 0.86 m/px | 352 / 65,536 = 0.54% / OWR-CD first on NLoS |
| PPData5D selected `z00` | Tx coordinate/z not encoded in inspected NPZ; current case uses RSS-argmax and 1.5 m proxy | 1.5 m above terrain in dataset semantics; current 2-D proxy uses 1.5 m flat-ground height | Current case uses 20 m flat-ground proxy on `inBldg_zyx[0]`; terrain is not yet included in LOS | 150 MHz (`f00`) | 10 m/px | OWR-CD first on NLoS |

PPData5D also defines `z01=30 m` and `z02=200 m` receiver planes above local
terrain.  The current solver accepts one fixed absolute Rx height, so those
planes must not be passed as if they were the same 2-D flat-ground problem.
The PPData5D PNG is RSS/grayscale, not a verified signed pathloss map; it must
be calibrated before dB metrics are meaningful.

For the current signed convention:

```text
FSPL_db = -20 log10(4 pi d / lambda)
diffraction_loss_db <= 0
prior_db = FSPL_db + diffraction_loss_db
```

The sign mismatch between positive RSS display values and negative signed
pathgain/pathloss values affects comparison and color mapping, but it does not
change the FSPL/CD/RD calculation itself.  The earlier PPData5D case also had
a separate Tx coordinate-unit bug, now corrected: Tx x/y stay pixel indices;
the solver applies the 10 m resolution once.

Boston/UCLA/USC now load their `Rx/{scene_id}.png` masks.  Their full pmap
remains useful for visual comparison, but full-image RMSE mixes direct
Wireless Insite samples with interpolated pixels.  The runner also writes
Rx-masked MAE/RMSE/Bias and LOS/NLOS RMSE for the measured subset.

For PPData5D, the inspected local package/config has no Tx power and no
PNG-to-dBm calibration.  Therefore its current case study reports no absolute
dB error metric.  Once the missing calibration is provided, the physical
conversion is `pathgain_db = RSS_dBm - PTx_dBm` and
`pathloss_magnitude_db = PTx_dBm - RSS_dBm` for unit-gain isotropic antennas.
