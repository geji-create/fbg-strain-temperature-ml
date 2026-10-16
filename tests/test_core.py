from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from analyze_real_data import refined_peak
from fbg_repro.config import FeatureConfig, SimulationConfig
from fbg_repro.features import FEATURE_COLUMNS, extract_six_features
from fbg_repro.simulation import (
    bragg_wavelength_nm,
    generate_spectra,
    uniform_fbg_reflectivity,
    wavelength_axis_nm,
)
from fbg_repro.cnn import SpectrumCNN


class TestCorePipeline(unittest.TestCase):
    def test_bragg_wavelength_increases_with_temperature_and_strain(self) -> None:
        config = SimulationConfig()
        reference = bragg_wavelength_nm(0.0, 20.0, config)
        self.assertGreater(bragg_wavelength_nm(100.0, 20.0, config), reference)
        self.assertGreater(bragg_wavelength_nm(0.0, 30.0, config), reference)

    def test_reflectivity_is_finite_and_bounded(self) -> None:
        config = SimulationConfig()
        wavelengths = wavelength_axis_nm(config)
        center = bragg_wavelength_nm(300.0, 25.0, config)
        reflectivity = uniform_fbg_reflectivity(
            wavelengths, center, 300.0, 25.0, config
        )
        self.assertTrue(np.isfinite(reflectivity).all())
        self.assertGreater(reflectivity.max(), 0.5)
        self.assertGreaterEqual(reflectivity.min(), 0.0)
        self.assertLessEqual(reflectivity.max(), 1.0)

    def test_small_dataset_feature_extraction(self) -> None:
        config = SimulationConfig(
            strain_levels_ue=[0.0, 300.0],
            temperature_count=2,
            replicates_per_condition=2,
        )
        spectra = generate_spectra(config)
        features = extract_six_features(spectra, FeatureConfig())
        self.assertEqual(len(features), 8)
        self.assertFalse(features[FEATURE_COLUMNS].isna().any().any())

    def test_cnn_output_shape(self) -> None:
        import torch

        model = SpectrumCNN()
        output = model(torch.randn(4, 1, 461))
        self.assertEqual(tuple(output.shape), (4, 2))

        multichannel_model = SpectrumCNN(input_channels=4)
        multichannel_output = multichannel_model(torch.randn(4, 4, 461))
        self.assertEqual(tuple(multichannel_output.shape), (4, 2))

    def test_subsample_peak_refinement(self) -> None:
        wavelengths = np.array([1549.0, 1550.0, 1551.0])
        signal = -((wavelengths - 1550.25) ** 2) + 2.0
        peak_nm, _ = refined_peak(wavelengths, signal)
        self.assertAlmostEqual(peak_nm, 1550.25, places=8)


if __name__ == "__main__":
    unittest.main()

