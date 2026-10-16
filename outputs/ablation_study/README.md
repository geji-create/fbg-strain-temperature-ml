# FBG temperature–strain ablation study

Model seeds: 20261013, 20261014, 20261015.
All results use the same 70/15/15 condition-grouped split. Paired solver spectra from one condition never cross splits.

## Test-set results

| Variant | Temperature MAE (°C) | Strain MAE (µε) | Temperature R² | Strain R² |
|---|---:|---:|---:|---:|
| Six features (MLP) | 5.061 ± 0.194 | 40.643 ± 1.469 | 0.5405 ± 0.0237 | 0.9588 ± 0.0016 |
| Full spectrum (GAP CNN) | 2.925 ± 0.299 | 13.420 ± 1.007 | 0.8097 ± 0.0291 | 0.9956 ± 0.0006 |
| Peak-centred spectrum | 8.312 ± 0.021 | 23.866 ± 4.033 | -0.0088 ± 0.0039 | 0.9875 ± 0.0038 |
| Full spectrum + position | 1.321 ± 0.123 | 20.241 ± 3.464 | 0.9694 ± 0.0056 | 0.9898 ± 0.0028 |
| Uniform strain + position | 7.419 ± 0.031 | 63.751 ± 0.725 | 0.1579 ± 0.0064 | 0.9150 ± 0.0014 |

## Controlled conclusions

- Full spectrum + position reduces temperature MAE by 73.9% and strain MAE by 50.2% relative to the six-feature MLP.
- Replacing the position-aware head with global average pooling changes temperature MAE from 1.321 to 2.925 °C, quantifying the value of absolute wavelength position.
- Explicit peak centering changes temperature MAE from 1.321 to 8.312 °C; this is a second, independent check of position information.
- With device nuisance and operating conditions held fixed, non-uniform strain reduces temperature MAE by 82.2% and strain MAE by 68.3% relative to uniform strain.
- These are synthetic-data ablations. They establish mechanism and internal validity, but do not replace experimental strain validation.
