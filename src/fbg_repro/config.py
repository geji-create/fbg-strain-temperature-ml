from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimulationConfig:
    reference_bragg_nm: float = 1548.097
    reference_temperature_c: float = 20.0
    effective_index: float = 1.447
    grating_length_m: float = 0.01
    index_modulation: float = 1.3e-4
    photoelastic_coefficient: float = 0.22
    thermal_expansion_per_c: float = 5.5e-7
    thermo_optic_per_c: float = 7.0e-6
    wavelength_min_nm: float = 1547.5
    wavelength_max_nm: float = 1549.8
    wavelength_step_nm: float = 0.005
    strain_levels_ue: list[float] = field(
        default_factory=lambda: [float(x) for x in range(0, 1000, 100)]
    )
    temperature_min_f: float = 46.0
    temperature_max_f: float = 105.0
    temperature_count: int = 7
    replicates_per_condition: int = 50
    strain_jitter_std_ue: float = 2.0
    temperature_jitter_std_c: float = 0.08
    wavelength_offset_std_nm: float = 0.002
    power_noise_std_db: float = 0.12
    power_scale_std_db: float = 0.25
    baseline_slope_std_db_per_nm: float = 0.08
    ripple_amplitude_std_db: float = 0.05
    instrument_resolution_fwhm_nm: float = 0.01
    coupling_temperature_coefficient_per_c: float = 0.0
    coupling_strain_coefficient_per_ue: float = 0.0
    random_seed: int = 20261012


@dataclass
class FeatureConfig:
    savgol_window: int = 9
    savgol_order: int = 2
    sidelobe_near_nm: float = 0.06
    sidelobe_far_nm: float = 0.35


@dataclass
class ModelConfig:
    n_estimators: int = 150
    boosting_learning_rate: float = 0.01
    random_seed: int = 42


@dataclass
class ProjectConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            simulation=SimulationConfig(**raw.get("simulation", {})),
            features=FeatureConfig(**raw.get("features", {})),
            models=ModelConfig(**raw.get("models", {})),
        )

