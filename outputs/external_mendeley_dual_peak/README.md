# External dual-peak FBG dataset analysis

Source: Serif Ali Sadik, *Optical Spectrum Dataset for Dual-Peak Temperature Measurements*, Mendeley Data v2, DOI: 10.17632/fymrd8js55.2. License: CC BY 4.0.

This dataset is treated as an independent external benchmark, not as a large deep-learning training set. It contains 17 OSA spectra (1001 wavelength points each), with a sensing FBG at 2–50 °C and a reference FBG held at 20 °C.

- Sensing-FBG sensitivity: 9.285 pm/°C
- Linear fit R²: 0.9763
- Leave-one-out temperature MAE: 2.034 °C
- Leave-one-out temperature RMSE: 2.494 °C
- Reference-peak standard deviation: 12.917 pm

Processed files are under `data/external_mendeley_dual_peak/processed`; publication figures are `fig06_external_dual_peak_dataset` and `fig07_external_temperature_benchmark`.
