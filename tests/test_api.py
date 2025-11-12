import os
import unittest

from fastapi.testclient import TestClient

from app import app
from model import PROJECT_ROOT

client = TestClient(app)


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure model file exists before running API tests
        model_path = os.path.join(PROJECT_ROOT, "models", "health_model.joblib")
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file not found at {model_path}. "
                f"Run `python src/train_model.py` before executing API tests."
            )

    def test_health_endpoint(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_predict_endpoint(self):
        payload = {
            "age": 52,
            "bmi": 29.8,
            "smoker": 1,
            "exercise_mins": 5,
            "systolic_bp": 150
        }
        resp = client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("risk_prob", data)
        self.assertIn("risk_label", data)
        self.assertIn("model_version", data)


if __name__ == "__main__":
    unittest.main()
