from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["quality", "yield", "performance", "energy_total_kwh"]


@dataclass
class ModelArtifacts:
    model: Pipeline
    features: list[str]
    mae: dict[str, float]


class ManufacturingModel:
    def train(self, df: pd.DataFrame) -> ModelArtifacts:
        features = [c for c in df.columns if c not in TARGETS and c not in {"batch_id", "energy_profile", "carbon_kg"}]
        X = df[features]
        y = df[TARGETS]

        pre = ColumnTransformer([("num", StandardScaler(), features)], remainder="drop")
        model = Pipeline(
            steps=[
                ("pre", pre),
                ("reg", MultiOutputRegressor(RandomForestRegressor(n_estimators=180, random_state=42))),
            ]
        )
        model.fit(X, y)
        preds = model.predict(X)
        mae = {t: float(mean_absolute_error(y[t], preds[:, i])) for i, t in enumerate(TARGETS)}
        return ModelArtifacts(model=model, features=features, mae=mae)

    def predict(self, artifacts: ModelArtifacts, rows: pd.DataFrame) -> pd.DataFrame:
        preds = artifacts.model.predict(rows[artifacts.features])
        return pd.DataFrame(preds, columns=TARGETS)
