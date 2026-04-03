from fastapi import FastAPI, HTTPException
from credit_scoring_project.api.model_loader import load_model
import pandas as pd

import logging
import json
import time
import os


# 1. Configuration logging

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/api_logs.json",
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger()


#  2. Initialisation API

app = FastAPI()


# 3. Chargement modèle

bundle = load_model()

model = bundle.get("model")
scaler = bundle.get("scaler")  
threshold = bundle.get("threshold", 0.5)
columns = bundle.get("columns")


# 4. Route test

@app.get("/")
def home():
    return {"message": "API Credit Scoring active"}


# 5. Route prédiction avec logging

@app.post("/predict")
def predict(data: dict):
    start_time = time.time()

    try:
        # Vérification sécurité
        if model is None:
            raise ValueError("Modèle non chargé")

        if columns is None:
            raise ValueError("Colonnes non définies")

        # créer dataframe vide
        df = pd.DataFrame([0] * len(columns)).T
        df.columns = columns

        # injecter inputs utilisateur
        for key, value in data.items():
            if key in df.columns:
                df[key] = value

        # appliquer scaler si présent
        if scaler is not None:
            df_scaled = scaler.transform(df.values)
        else:
            df_scaled = df.values

        # prédiction
        proba = model.predict_proba(df_scaled)[0][1]
        prediction = int(proba >= threshold)

        execution_time = time.time() - start_time

        # log succès
        log = {
            "inputs": data,
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

    except Exception as e:
        execution_time = time.time() - start_time

        # log erreur
        log = {
            "inputs": data,
            "error": str(e),
            "execution_time": execution_time,
            "status": "error"
        }

        logger.error(json.dumps(log))

        raise HTTPException(status_code=400, detail=str(e))