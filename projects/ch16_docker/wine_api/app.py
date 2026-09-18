"""FastAPI service for the Chapter 15 champion wine-quality model.

Run locally:   uvicorn app:app --port 8000
In Docker:     see Dockerfile (listens on 8000)
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Wine quality API", version="1.0")


class Wine(BaseModel):
    fixed_acidity: float = Field(..., alias="fixed acidity", examples=[7.4])
    volatile_acidity: float = Field(..., alias="volatile acidity", examples=[0.7])
    citric_acid: float = Field(..., alias="citric acid", examples=[0.0])
    residual_sugar: float = Field(..., alias="residual sugar", examples=[1.9])
    chlorides: float = Field(..., examples=[0.076])
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide", examples=[11.0])
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide", examples=[34.0])
    density: float = Field(..., examples=[0.9978])
    pH: float = Field(..., examples=[3.51])
    sulphates: float = Field(..., examples=[0.56])
    alcohol: float = Field(..., examples=[9.4])

    model_config = {"populate_by_name": True}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
def predict(wines: list[Wine]):
    frame = pd.DataFrame([w.model_dump(by_alias=True) for w in wines])
    preds = model.predict(frame)
    return {"predictions": [round(float(p), 3) for p in preds]}
