from fastapi import FastAPI, Request
import pandas as pd
import mlflow.sklearn
from pathlib import Path

import logging
import time

from api.schema import HeartDiseaseInput, PredictionOutput
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API Requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API Request Latency",
    ["endpoint"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Prediction API")

MODEL_PATH = Path(__file__).resolve().parents[1] / "model_artifact"
model = mlflow.sklearn.load_model(MODEL_PATH)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"method={request.method} "
        f"path={request.url.path} "
        f"status_code={response.status_code} "
        f"latency={duration:.4f}s"
    )
    return response

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: HeartDiseaseInput):
    REQUEST_COUNT.labels("POST", "/predict").inc()

    with REQUEST_LATENCY.labels("/predict").time():
        df = pd.DataFrame([input_data.model_dump()])
        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

