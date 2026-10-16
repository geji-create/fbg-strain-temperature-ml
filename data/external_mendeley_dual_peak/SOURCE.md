# Third-party FBG dataset source

- Dataset: *Optical Spectrum Dataset for Dual-Peak Temperature Measurements*
- Author: Şerif Ali Sadık
- Repository: Mendeley Data, version 2
- DOI: [10.17632/fymrd8js55.2](https://doi.org/10.17632/fymrd8js55.2)
- Published: 19 December 2025
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Downloaded: 15 October 2026
- Original ZIP SHA-256: `7AA0038E3D8954400D5062F2F797CA36410CDCE8D86D383C056734CC7ECE09DE`

The archive contains 17 spectra measured with an Anritsu MS9710B optical spectrum analyser. The sensing FBG was varied from 2 °C to 50 °C, while the reference FBG was held at 20 °C. Each spectrum has 1001 wavelength samples from 1548.68 nm to 1550.68 nm. Raw files are preserved without modification under `raw/`.

Use `scripts/process_external_dual_peak_dataset.py` to rebuild the standardized NPZ/CSV files, analysis summary, and publication figures. Any redistributed or published derivative must retain attribution to the original author and dataset DOI.
