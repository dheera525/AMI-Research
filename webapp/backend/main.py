"""
AMI Predictor API
=================
FastAPI backend that loads the final.py serving bundle and exposes:
  GET  /api/             - health
  GET  /api/metrics      - per-model test metrics + leaderboard
  GET  /api/features     - feature list + ranges (for building the form)
  POST /api/predict      - accept a feature dict, return AMI probability for
                           every served model + the winning model's verdict
The static Next.js export is mounted at "/" so the whole app ships in one
container (used by the Hugging Face Spaces Docker build).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load bundle + metrics (try a few locations so the same image works locally
# and on Hugging Face Spaces)
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
CANDIDATE_ASSET_DIRS = [
    Path(os.environ.get("AMI_ASSETS_DIR", "")),
    HERE / "final_assets",
    HERE.parent / "final_assets",
    HERE.parent.parent / "final_assets",
]

ASSETS_DIR: Path | None = None
for c in CANDIDATE_ASSET_DIRS:
    if c and (c / "bundle.joblib").exists():
        ASSETS_DIR = c
        break

if ASSETS_DIR is None:
    raise RuntimeError(
        "Could not find final_assets/bundle.joblib. Run `python final.py` "
        "first or set AMI_ASSETS_DIR."
    )

BUNDLE = joblib.load(ASSETS_DIR / "bundle.joblib")
METRICS = json.loads((ASSETS_DIR / "metrics.json").read_text())

SCALER = BUNDLE["scaler"]
FEATURES: List[str] = BUNDLE["features"]
MODELS: Dict = BUNDLE["models"]
WINNER_NAME: str = BUNDLE["winner_name"]
FEATURE_RANGES: Dict = BUNDLE["feature_ranges"]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AMI Predictor API", version="1.0.0")

allowed = os.environ.get("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed.split(",")] if allowed != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    # Free-form so the frontend can send the raw column names (some contain
    # spaces / slashes like "NEU/LY" and "PDW ").
    features: Dict[str, float] = Field(..., description="Feature name -> value")


class ModelPrediction(BaseModel):
    model: str
    probability: float
    prediction: int


class PredictResponse(BaseModel):
    winner: str
    probability: float
    prediction: int
    verdict: str
    all_models: List[ModelPrediction]


@app.get("/api/")
def root():
    return {
        "status": "ok",
        "winner": WINNER_NAME,
        "n_features": len(FEATURES),
        "n_models": len(MODELS),
    }


@app.get("/api/metrics")
def get_metrics():
    return METRICS


@app.get("/api/features")
def get_features():
    return {
        "features": FEATURES,
        "ranges": FEATURE_RANGES,
    }


def _vectorize(payload: Dict[str, float]) -> np.ndarray:
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}",
        )
    row = np.array([[float(payload[f]) for f in FEATURES]])
    return SCALER.transform(row)


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = _vectorize(req.features)

    per_model: List[ModelPrediction] = []
    for name, model in MODELS.items():
        try:
            pred = int(model.predict(x)[0])
        except Exception:
            continue
        if hasattr(model, "predict_proba"):
            try:
                proba = float(model.predict_proba(x)[0, 1])
            except Exception:
                proba = float(pred)
        else:
            proba = float(pred)
        per_model.append(ModelPrediction(model=name, probability=proba, prediction=pred))

    winner = next(
        (p for p in per_model if p.model == WINNER_NAME),
        per_model[0] if per_model else None,
    )
    if winner is None:
        raise HTTPException(status_code=500, detail="No models produced a prediction")

    verdict = (
        f"AMI likely (probability {winner.probability:.1%})"
        if winner.prediction == 1
        else f"Control / no AMI (AMI probability {winner.probability:.1%})"
    )

    return PredictResponse(
        winner=winner.model,
        probability=winner.probability,
        prediction=winner.prediction,
        verdict=verdict,
        all_models=per_model,
    )


# ---------------------------------------------------------------------------
# Static frontend (Next.js export). Mounted last so /api/* routes take
# precedence. The directory only exists inside the Docker image.
# ---------------------------------------------------------------------------
STATIC_DIR = HERE / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
