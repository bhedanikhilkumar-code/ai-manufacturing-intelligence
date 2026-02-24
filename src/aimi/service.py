from __future__ import annotations

import numpy as np
import pandas as pd

from aimi.energy_intelligence import EnergyPatternIntelligence
from aimi.generator import GeneratorConfig, SyntheticBatchGenerator, parse_profile
from aimi.modeling import ManufacturingModel, ModelArtifacts
from aimi.optimization import GoldenSignatureStore, ParetoOptimizer, adaptive_carbon_target, maybe_update_signature
from aimi.pipeline import DataPipeline


class ServiceContainer:
    def __init__(self) -> None:
        self.pipeline = DataPipeline()
        self.generator = SyntheticBatchGenerator(GeneratorConfig(n_batches=250))
        self.batch_df, self.profile_df = self.generator.generate()
        self.train_df = self.pipeline.feature_engineer(self.pipeline.clean(self.batch_df), self.profile_df)
        self.model = ManufacturingModel()
        self.artifacts: ModelArtifacts = self.model.train(self.train_df)
        self.epi = EnergyPatternIntelligence()
        self.epi.fit([parse_profile(p) for p in self.profile_df["energy_profile"]])
        self.optimizer = ParetoOptimizer()
        self.store = GoldenSignatureStore()

    def predict(self, row: dict) -> dict:
        df = pd.DataFrame([row])
        pred_df = self.model.predict(self.artifacts, df)
        return pred_df.iloc[0].to_dict()

    def optimize(self, n_samples: int = 40) -> dict:
        sample = self.train_df.sample(min(n_samples, len(self.train_df)), random_state=42).copy()
        sample["carbon_kg"] = sample["energy_total_kwh"] * sample["carbon_intensity"] * sample["emissions_factor"]
        pareto = self.optimizer.optimize(sample)
        target = adaptive_carbon_target(sample)
        return {
            "carbon_target": target,
            "pareto_candidates": pareto.head(10).to_dict(orient="records"),
        }

    def golden(self, profile: list[float], metrics: dict, accept: bool) -> dict:
        return maybe_update_signature(self.store, np.array(profile, dtype=float), metrics, accept)
