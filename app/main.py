from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib, json, os, numpy as np

# Load artifacts at startup
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_order.json")

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)

# Globals
model = None
feature_order = None
metrics_cache: Dict[str, Any] = {}

@app.on_event("startup")
async def load_artifacts():
    global model, feature_order, metrics_cache
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run training first.")
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_order = json.load(f)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_cache = json.load(f)
    else:
        metrics_cache = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(payload: PatientInput):
    if model is None or feature_order is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Assemble features in correct order
    x = np.array([[
        getattr(payload, name) for name in feature_order
    ]], dtype=float)

    # Predict
    try:
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)[0]
            pred = int(np.argmax(proba))
            confidence = float(np.max(proba))
        else:
            # Some models (e.g., SVC without probability) may not have predict_proba
            pred = int(model.predict(x)[0])
            confidence = 0.5  # default fallback
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result_text = "Diabetic" if pred == 1 else "Not Diabetic"

    return {
        "prediction": pred,
        "result": result_text,
        "confidence": round(confidence, 4),
    }

@app.get("/metrics")
async def get_metrics():
    if not metrics_cache:
        raise HTTPException(status_code=404, detail="Metrics not available. Train and save metrics first.")
    return metrics_cache