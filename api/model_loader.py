import joblib
import os


def load_model():
    model_path = "credit_scoring_project/output/model.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(" modèle introuvable")

    model = joblib.load(model_path)

    return {
        "model": model,
        "scaler": None,
        "threshold": 0.5,
        "columns": [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "DAYS_BIRTH"
        ]
    }