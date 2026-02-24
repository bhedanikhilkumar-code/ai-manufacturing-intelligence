from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aimi.energy_intelligence import EnergyPatternIntelligence


@dataclass
class ConstraintConfig:
    min_quality: float = 85.0
    min_yield: float = 88.0
    min_performance: float = 84.0
    max_energy_kwh: float = 1600.0


class GoldenSignatureStore:
    def __init__(self, db_path: str = "data/golden_signature.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signatures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER,
                    accepted INTEGER,
                    metrics_json TEXT,
                    signature_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def latest(self) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT version, accepted, metrics_json, signature_json, created_at FROM signatures ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {
            "version": row[0],
            "accepted": bool(row[1]),
            "metrics": json.loads(row[2]),
            "signature": json.loads(row[3]),
            "created_at": row[4],
        }

    def add(self, signature: dict, metrics: dict, accepted: bool) -> dict:
        current = self.latest()
        version = 1 if current is None else current["version"] + 1
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO signatures (version, accepted, metrics_json, signature_json) VALUES (?, ?, ?, ?)",
                (version, int(accepted), json.dumps(metrics), json.dumps(signature)),
            )
        return {"version": version, "accepted": accepted, "metrics": metrics, "signature": signature}


class ParetoOptimizer:
    def __init__(self, constraints: ConstraintConfig | None = None) -> None:
        self.constraints = constraints or ConstraintConfig()

    def pareto_front(self, candidates: pd.DataFrame) -> pd.DataFrame:
        objs = candidates[["quality", "yield", "performance", "energy_total_kwh", "carbon_kg"]].copy()
        objs["energy_total_kwh"] *= -1
        objs["carbon_kg"] *= -1

        keep = np.ones(len(objs), dtype=bool)
        vals = objs.values
        for i in range(len(vals)):
            if not keep[i]:
                continue
            dominates = np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1)
            if dominates.any():
                keep[i] = False
        return candidates[keep].sort_values(by=["carbon_kg", "energy_total_kwh"], ascending=True)

    def optimize(self, candidates: pd.DataFrame) -> pd.DataFrame:
        c = self.constraints
        constrained = candidates[
            (candidates["quality"] >= c.min_quality)
            & (candidates["yield"] >= c.min_yield)
            & (candidates["performance"] >= c.min_performance)
            & (candidates["energy_total_kwh"] <= c.max_energy_kwh)
        ]
        if constrained.empty:
            constrained = candidates
        return self.pareto_front(constrained)


def adaptive_carbon_target(df: pd.DataFrame, constraint_factor: float = 0.95) -> float:
    baseline = float(df["carbon_kg"].quantile(0.4))
    best_quality = float(df[df["quality"] >= df["quality"].quantile(0.75)]["carbon_kg"].median())
    return min(baseline, best_quality) * constraint_factor


def maybe_update_signature(store: GoldenSignatureStore, profile: np.ndarray, metrics: dict, accept: bool) -> dict:
    signature = EnergyPatternIntelligence.signature_features(profile)
    latest = store.latest()
    if latest and accept:
        if metrics.get("quality", 0) > latest["metrics"].get("quality", 0) and metrics.get("carbon_kg", 1e9) < latest["metrics"].get("carbon_kg", 1e9):
            return store.add(signature=signature, metrics=metrics, accepted=True)
        return {"updated": False, "reason": "No superior outcome vs latest accepted signature."}
    return store.add(signature=signature, metrics=metrics, accepted=accept)
