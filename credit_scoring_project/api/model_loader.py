import os
import joblib

def load_model():
    model_path = "output/model.pkl"

    if not os.path.exists(model_path):
        return {
            "model": None,
            "scaler": None,
            "threshold": 0.5,
            "columns": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]
        }

    return joblib.load(model_path)