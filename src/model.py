import os
from functools import lru_cache
from typing import Tuple

import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "health_model.joblib")


@lru_cache(maxsize=1)
def load_model():
    """
    Load and cache the trained model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run `python src/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def make_feature_vector(
    age: float,
    bmi: float,
    smoker: int,
    exercise_mins: float,
    systolic_bp: float,
) -> np.ndarray:
    """
    Build feature vector in the same order used during training.
    """
    return np.array([[age, bmi, smoker, exercise_mins, systolic_bp]], dtype=float)


def predict_risk(
    age: float,
    bmi: float,
    smoker: int,
    exercise_mins: float,
    systolic_bp: float,
) -> Tuple[float, int]:
    """
    Predict health risk probability and label.

    Returns:
        (risk_prob, risk_label)
        risk_prob  in [0.0, 1.0]
        risk_label in {0, 1}
    """
    model = load_model()
    X = make_feature_vector(age, bmi, smoker, exercise_mins, systolic_bp)
    prob = float(model.predict_proba(X)[0][1])
    label = int(prob >= 0.5)
    return prob, label
