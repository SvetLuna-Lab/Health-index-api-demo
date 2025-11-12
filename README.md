# Health-index-api-demo

Small demo project that exposes a simple **health risk model** via a FastAPI endpoint.

The goal of this repository is to show how to:

- train a very simple ML model (logistic regression) from a tiny CSV dataset,
- wrap the model in a minimal **FastAPI** application,
- provide a `/predict` endpoint that returns a health risk score and label.

This is a lightweight example of turning a basic ML model into an API.

---

## Repository structure

```text
health-index-api-demo/
├─ data/
│  └─ health_samples.csv      # small synthetic health dataset
├─ models/
│  └─ health_model.joblib     # created after running train_model.py
├─ src/
│  ├─ __init__.py
│  ├─ train_model.py          # train and save logistic regression model
│  ├─ model.py                # load model and run predictions
│  ├─ schemas.py              # Pydantic models for input and output
│  └─ app.py                  # FastAPI app with /health and /predict
├─ tests/
│  ├─ __init__.py
│  ├─ test_health_model.py    # tests for the model wrapper
│  └─ test_api.py             # tests for the FastAPI endpoints
├─ README.md
├─ requirements.txt
└─ .gitignore



Requirements

Python 3.10+

Dependencies from requirements.txt:


fastapi
uvicorn
scikit-learn
joblib
numpy



Install them (ideally in a virtual environment):

pip install -r requirements.txt



Data

The data/health_samples.csv file contains a tiny synthetic dataset with the following columns:

age – age in years

bmi – body mass index

smoker – 0 = non-smoker, 1 = smoker

exercise_mins – average exercise minutes per day

systolic_bp – systolic blood pressure (mmHg)

risk – target label: 0 = low risk, 1 = high risk

This dataset is only for demonstration and is not medically meaningful.



Model training

The training script src/train_model.py:

loads the CSV from data/health_samples.csv,

trains a LogisticRegression classifier (scikit-learn),

saves the model to models/health_model.joblib.

Run it once before using the API or running tests:


python src/train_model.py


After this, the models/ directory should contain:

models/
└─ health_model.joblib


If the file is missing, both the API and tests will fail when trying to load the model.



FastAPI application

The FastAPI app is defined in src/app.py and provides two endpoints:

GET /health – simple health check.

Example:

curl http://127.0.0.1:8000/health


Response:

{
  "status": "ok"
}



POST /predict – predict health risk based on input features.

Input JSON schema (HealthInput):


{
  "age": 52,
  "bmi": 29.8,
  "smoker": 1,
  "exercise_mins": 5,
  "systolic_bp": 150
}



Output JSON schema (HealthPrediction):

{
  "risk_prob": 0.87,
  "risk_label": 1,
  "model_version": "0.1.0"
}


risk_prob – predicted probability of high risk (0.0–1.0)

risk_label – 0 = low risk, 1 = high risk

model_version – simple string identifier for the model



Running the API

Install dependencies:

pip install -r requirements.txt



Train the model (creates models/health_model.joblib):

python src/train_model.py



Start the FastAPI app with Uvicorn:

uvicorn src.app:app --reload



Open the interactive docs in your browser:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc

From there you can call /predict directly via the web UI.



Tests

Make sure the model file exists first:

python src/train_model.py



Then run tests from the project root:

python -m unittest discover -s tests



This will execute:

tests/test_health_model.py – basic checks for the model wrapper (probability range, label, model file presence).

tests/test_api.py – basic checks for the FastAPI endpoints (/health and /predict).



Extending the demo

Possible next steps:

Add more features and a larger, more realistic dataset.

Try different models (e.g., random forest, gradient boosting).

Add additional validation rules and business logic around the prediction.

Containerize the service with Docker and run it in a minimal environment.

Integrate authentication or API keys for protected usage.

Add logging and simple request/response monitoring.

