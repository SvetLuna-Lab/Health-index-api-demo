import os
import unittest

from model import PROJECT_ROOT, predict_risk


class TestHealthModel(unittest.TestCase):
    def test_predict_risk_output_range(self):
        # Simple sanity check: probability is in [0, 1] and label is 0 or 1.
        prob, label = predict_risk(
            age=50,
            bmi=30.0,
            smoker=1,
            exercise_mins=10,
            systolic_bp=150,
        )

        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertIn(label, (0, 1))

    def test_model_file_exists(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "health_model.joblib")
        self.assertTrue(os.path.exists(model_path), "Model file should exist after training")


if __name__ == "__main__":
    unittest.main()
