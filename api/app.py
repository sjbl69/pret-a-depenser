from fastapi import FastAPI, HTTPException
from credit_scoring_project.api.model_loader import load_model
import pandas as pd

# 1. créer l'API
app = FastAPI()

# 2. charger le modèle une seule fois
bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]
columns = bundle["columns"] 


# 3. route simple test
@app.get("/")
def home():
    return {"message": "API Credit Scoring active"}


# 4. route prédiction 
@app.post("/predict")
def predict(data: dict):
    try:

        # créer une ligne vide avec les bonnes colonnes
        df = pd.DataFrame([0]*len(columns)).T
        df.columns = columns

        # injecter les données utilisateur
        for key, value in data.items():
            if key in df.columns:
                df[key] = value

        #  appliquer le scaler
        df_scaled = scaler.transform(df.values)

        # prédiction
        proba = model.predict_proba(df_scaled)[0][1]
        prediction = int(proba >= threshold)

        return {
            "prediction": prediction,
            "probability": float(proba)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))