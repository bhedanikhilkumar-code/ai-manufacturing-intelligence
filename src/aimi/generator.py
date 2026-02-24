from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    n_batches: int = 200
    time_steps: int = 96
    seed: int = 42


class SyntheticBatchGenerator:
    """Generate synthetic manufacturing batches with energy profiles and outcomes."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _energy_profile(self, load: float, process_temp: float, reliability: float) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, self.config.time_steps)
        base = load * (1 + 0.15 * np.sin(2 * t) + 0.08 * np.sin(6 * t))
        transient = self.rng.normal(0, (1.2 - reliability) * 2.0, size=self.config.time_steps)
        temp_factor = (process_temp - 60) * 0.05
        profile = np.clip(base + transient + temp_factor, 5, None)
        return profile

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        for batch_id in range(self.config.n_batches):
            machine_age = self.rng.integers(1, 15)
            maintenance_score = self.rng.uniform(0.6, 1.0)
            operator_skill = self.rng.uniform(0.5, 1.0)
            setpoint_temp = self.rng.normal(68, 6)
            pressure = self.rng.normal(5.0, 0.8)
            feed_rate = self.rng.normal(110, 15)
            cycle_time = np.clip(self.rng.normal(50, 7), 25, 90)
            load = np.clip(feed_rate / 3 + pressure * 3, 20, 80)
            reliability = np.clip(0.45 * maintenance_score + 0.55 * (1 - machine_age / 20), 0.2, 0.98)

            energy_profile = self._energy_profile(load=load, process_temp=setpoint_temp, reliability=reliability)
            energy_mean = float(np.mean(energy_profile))
            carbon_intensity = self.rng.uniform(0.28, 0.72)
            emissions_factor = self.rng.uniform(0.85, 1.15)
            energy_total = float(np.sum(energy_profile) * 0.25)
            carbon_kg = energy_total * carbon_intensity * emissions_factor

            quality = np.clip(
                75 + 18 * operator_skill + 6 * maintenance_score - 0.3 * abs(setpoint_temp - 70) - 0.08 * energy_mean,
                50,
                100,
            )
            yield_pct = np.clip(80 + 10 * reliability + 0.06 * feed_rate - 0.02 * cycle_time - 0.004 * carbon_kg, 40, 100)
            performance = np.clip(70 + 12 * reliability + 0.04 * feed_rate - 0.15 * pressure + self.rng.normal(0, 1.5), 35, 100)

            rows.append(
                {
                    "batch_id": batch_id,
                    "machine_age_years": machine_age,
                    "maintenance_score": maintenance_score,
                    "operator_skill": operator_skill,
                    "setpoint_temp": setpoint_temp,
                    "pressure": pressure,
                    "feed_rate": feed_rate,
                    "cycle_time": cycle_time,
                    "carbon_intensity": carbon_intensity,
                    "emissions_factor": emissions_factor,
                    "energy_total_kwh": energy_total,
                    "carbon_kg": carbon_kg,
                    "quality": quality,
                    "yield": yield_pct,
                    "performance": performance,
                }
            )
            profile_rows.append({"batch_id": batch_id, "energy_profile": ",".join(np.round(energy_profile, 3).astype(str))})

        return pd.DataFrame(rows), pd.DataFrame(profile_rows)


def parse_profile(profile_text: str) -> np.ndarray:
    return np.array([float(x) for x in profile_text.split(",")], dtype=float)
