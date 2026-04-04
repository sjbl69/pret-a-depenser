from fastapi import FastAPI, HTTPException, Request
import pandas as pd
import time
import json
import logging
import os

from credit_scoring_project.api.model_loader import load_model


# ==============================
# 1. Initialisation API (IMPORTANT)
# ==============================
app = FastAPI()


# ==============================
# 2. Logging
# ==============================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/api_logs.json",
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger()


# ==============================
# 3. Chargement modèle
# ==============================
bundle = load_model()

model = bundle.get("model")
scaler = bundle.get("scaler")
threshold = bundle.get("threshold", 0.5)
columns = bundle.get("columns")


# ==============================
# 4. Route test
# ==============================
@app.get("/")
def home():
    return {"message": "API Credit Scoring active"}


# ==============================
# 5. Route prédiction (VALIDATION + LOGGING)
# ==============================
@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()

    try:
        # 🔥 récupérer données brutes
        data = await request.json()

        # =========================
        # VALIDATION (POUR TESTS)
        # =========================
        required_fields = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]

        # champs manquants
        if not all(field in data for field in required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # mauvais type
        for field in required_fields:
            if not isinstance(data[field], (int, float)):
                raise HTTPException(status_code=400, detail=f"Invalid type for {field}")

        data_dict = data

        # =========================
        # PREPROCESSING
        # =========================
        df = pd.DataFrame([0] * len(columns)).T
        df.columns = columns

        for key, value in data_dict.items():
            if key in df.columns:
                df[key] = value

        # scaler
        if scaler is not None:
            df_scaled = scaler.transform(df.values)
        else:
            df_scaled = df.values

        # =========================
        # PREDICTION
        # =========================
        proba = model.predict_proba(df_scaled)[0][1]
        prediction = int(proba >= threshold)

        execution_time = time.time() - start_time

        # =========================
        # LOG SUCCESS
        # =========================
        log = {
            "inputs": data_dict,
            "prediction": prediction,
            "probability": float(proba),
            "execution_time": execution_time,
            "status": "success"
        }

        logger.info(json.dumps(log))

        return {
            "prediction": prediction,
            "probability": float(proba)
        }

    except HTTPException:
        raise

    except Exception as e:
        execution_time = time.time() - start_time

        log = {
            "inputs": data if 'data' in locals() else {},
            "error": str(e),
            "execution_time": execution_time,
            "status": "error"
        }

        logger.error(json.dumps(log))

        raise HTTPException(status_code=400, detail=str(e))