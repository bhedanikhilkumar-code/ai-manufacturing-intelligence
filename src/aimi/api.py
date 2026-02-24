from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from aimi.service import ServiceContainer

app = FastAPI(title="AI-Driven Manufacturing Intelligence")
svc = ServiceContainer()


class PredictRequest(BaseModel):
    row: dict = Field(..., description="Single fully-featured row aligned to model features")


class OptimizeRequest(BaseModel):
    n_samples: int = 40


class GoldenRequest(BaseModel):
    profile: list[float]
    metrics: dict
    accept: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    return svc.predict(req.row)


@app.post("/optimize")
def optimize(req: OptimizeRequest) -> dict:
    return svc.optimize(req.n_samples)


@app.get("/golden-signature")
def get_golden() -> dict:
    return {"data": svc.store.latest()}


@app.post("/golden-signature")
def post_golden(req: GoldenRequest) -> dict:
    return svc.golden(req.profile, req.metrics, req.accept)
