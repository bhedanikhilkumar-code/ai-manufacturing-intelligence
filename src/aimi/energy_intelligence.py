from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest


class EnergyPatternIntelligence:
    def __init__(self, random_state: int = 42) -> None:
        self.model = IsolationForest(contamination=0.1, random_state=random_state)
        self._is_fit = False

    @staticmethod
    def signature_features(profile: np.ndarray) -> dict[str, float]:
        arr = np.asarray(profile, dtype=float)
        fft_mag = np.abs(np.fft.rfft(arr))
        freqs = np.fft.rfftfreq(len(arr), d=1)

        def band(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            return float(fft_mag[mask].sum())

        return {
            "rms": float(np.sqrt(np.mean(arr**2))),
            "peak": float(arr.max()),
            "peak_to_avg": float(arr.max() / max(arr.mean(), 1e-6)),
            "shape_skew": float(skew(arr)),
            "shape_kurtosis": float(kurtosis(arr)),
            "crest_factor": float(arr.max() / max(np.sqrt(np.mean(arr**2)), 1e-6)),
            "fft_low": band(0.0, 0.08),
            "fft_mid": band(0.08, 0.2),
            "fft_high": band(0.2, 0.5),
        }

    def fit(self, profiles: list[np.ndarray]) -> pd.DataFrame:
        features = pd.DataFrame([self.signature_features(p) for p in profiles])
        self.model.fit(features)
        self._is_fit = True
        return features

    def score(self, profiles: list[np.ndarray]) -> np.ndarray:
        features = pd.DataFrame([self.signature_features(p) for p in profiles])
        if not self._is_fit:
            self.fit(profiles)
        return -self.model.score_samples(features)
