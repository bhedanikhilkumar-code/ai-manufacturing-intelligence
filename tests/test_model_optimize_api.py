from fastapi.testclient import TestClient

from aimi.api import app, svc
from aimi.modeling import TARGETS
from aimi.optimization import GoldenSignatureStore, adaptive_carbon_target


def test_model_prediction_targets_present():
    row = svc.train_df.iloc[0][svc.artifacts.features].to_dict()
    pred = svc.predict(row)
    for t in TARGETS:
        assert t in pred


def test_adaptive_carbon_target_positive():
    target = adaptive_carbon_target(svc.train_df)
    assert target > 0


def test_golden_store_roundtrip(tmp_path):
    store = GoldenSignatureStore(str(tmp_path / "g.db"))
    stored = store.add(signature={"rms": 1.2}, metrics={"quality": 90, "carbon_kg": 50}, accepted=True)
    assert stored["version"] == 1
    assert store.latest()["metrics"]["quality"] == 90


def test_api_health_and_optimize():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    resp = client.post("/optimize", json={"n_samples": 12})
    assert resp.status_code == 200
    body = resp.json()
    assert "carbon_target" in body
    assert isinstance(body["pareto_candidates"], list)
