# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import json

app = FastAPI(title="Fall Detection API")

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

class SensorData(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    sma: float
    smg: float
    resultant: float

@app.get("/")
def root():
    return {"message": "Fall Detection API is running 🩺"}

@app.get("/health")
def health():
    with open("model/metrics.json") as f:
        metrics = json.load(f)
    return {"status": "ok", "model_accuracy": metrics["accuracy"]}

@app.post("/predict")
def predict(data: SensorData):
    features = pd.DataFrame(
        [[data.ax, data.ay, data.az,
          data.gx, data.gy, data.gz,
          data.sma, data.smg, data.resultant]],
        columns=["ax","ay","az","gx","gy","gz","sma","smg","resultant"]
    )
    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0][prediction])
    return {
        "prediction": prediction,
        "label": "FALL DETECTED 🚨" if prediction == 1 else "Normal Activity ✅",
        "confidence": round(confidence, 4)
    }