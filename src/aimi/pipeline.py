from __future__ import annotations

import numpy as np
import pandas as pd

from aimi.energy_intelligence import EnergyPatternIntelligence
from aimi.generator import parse_profile

REQUIRED_COLUMNS = {
    "machine_age_years",
    "maintenance_score",
    "operator_skill",
    "setpoint_temp",
    "pressure",
    "feed_rate",
    "cycle_time",
    "carbon_intensity",
    "emissions_factor",
    "energy_total_kwh",
    "quality",
    "yield",
    "performance",
}


class DataPipeline:
    def clean(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        df = batch_df.copy()
        numeric_cols = [c for c in df.columns if c != "batch_id"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return df.fillna(df.median(numeric_only=True))

    def validate(self, batch_df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(batch_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

    def feature_engineer(self, batch_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
        df = batch_df.merge(profile_df, on="batch_id", how="inner")
        profiles = [parse_profile(p) for p in df["energy_profile"]]
        epi = EnergyPatternIntelligence()
        sig = pd.DataFrame([epi.signature_features(p) for p in profiles])

        df = pd.concat([df.reset_index(drop=True), sig], axis=1)
        df["energy_per_output"] = df["energy_total_kwh"] / np.maximum(df["yield"], 1)
        df["temp_pressure_interaction"] = df["setpoint_temp"] * df["pressure"]
        return df
