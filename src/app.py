from fastapi import FastAPI

from model import predict_risk
from schemas import HealthInput, HealthPrediction

app = FastAPI(
    title="Health Index API Demo",
    description="Minimal demo API for health risk prediction",
    version="0.1.0",
)


@app.get("/health", summary="Health check")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=HealthPrediction, summary="Predict health risk")
def predict_health_risk(payload: HealthInput) -> HealthPrediction:
    """
    Predict health risk based on basic input features.

    The underlying model is a simple logistic regression
    trained on a small synthetic dataset.
    """
    prob, label = predict_risk(
        age=payload.age,
        bmi=payload.bmi,
        smoker=payload.smoker,
        exercise_mins=payload.exercise_mins,
        systolic_bp=payload.systolic_bp,
    )

    return HealthPrediction(
        risk_prob=prob,
        risk_label=label,
        model_version="0.1.0",
    )
