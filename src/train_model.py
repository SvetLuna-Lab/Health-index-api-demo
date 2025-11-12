import csv
import os
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "health_samples.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "health_model.joblib")


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load health samples from CSV into feature matrix X and target vector y.
    Features:
        age, bmi, smoker, exercise_mins, systolic_bp
    Target:
        risk (0 or 1)
    """
    X: List[List[float]] = []
    y: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = float(row["age"])
            bmi = float(row["bmi"])
            smoker = float(row["smoker"])
            exercise_mins = float(row["exercise_mins"])
            systolic_bp = float(row["systolic_bp"])
            risk = int(row["risk"])

            X.append([age, bmi, smoker, exercise_mins, systolic_bp])
            y.append(risk)

    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_and_save_model() -> None:
    """
    Train a simple logistic regression model and save it to models/ directory.
    """
    X, y = load_data(DATA_PATH)

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
