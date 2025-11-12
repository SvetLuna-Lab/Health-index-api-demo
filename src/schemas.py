from pydantic import BaseModel, Field


class HealthInput(BaseModel):
    age: float = Field(..., ge=0, description="Age in years")
    bmi: float = Field(..., ge=0, description="Body Mass Index")
    smoker: int = Field(..., ge=0, le=1, description="0 = non-smoker, 1 = smoker")
    exercise_mins: float = Field(..., ge=0, description="Average exercise minutes per day")
    systolic_bp: float = Field(..., ge=0, description="Systolic blood pressure (mmHg)")


class HealthPrediction(BaseModel):
    risk_prob: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of high risk")
    risk_label: int = Field(..., ge=0, le=1, description="0 = low risk, 1 = high risk")
    model_version: str = Field(..., description="Model version identifier")
