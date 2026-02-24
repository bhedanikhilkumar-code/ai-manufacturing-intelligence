import pandas as pd

from aimi.generator import GeneratorConfig, SyntheticBatchGenerator
from aimi.pipeline import DataPipeline


def test_generator_outputs_expected_shapes():
    bdf, pdf = SyntheticBatchGenerator(GeneratorConfig(n_batches=10, seed=1)).generate()
    assert len(bdf) == 10
    assert len(pdf) == 10
    assert {"quality", "yield", "performance", "energy_total_kwh"}.issubset(bdf.columns)


def test_pipeline_feature_engineering_adds_signature_features():
    bdf, pdf = SyntheticBatchGenerator(GeneratorConfig(n_batches=8, seed=2)).generate()
    pipe = DataPipeline()
    clean = pipe.clean(bdf)
    pipe.validate(clean)
    feat = pipe.feature_engineer(clean, pdf)
    assert "rms" in feat.columns
    assert "fft_mid" in feat.columns
    assert not feat.isna().any().any()
