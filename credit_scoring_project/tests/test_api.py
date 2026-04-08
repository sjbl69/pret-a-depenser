from unittest.mock import patch
from fastapi.testclient import TestClient


# Fake model pour simuler le comportement du modèle
class FakeModel:
    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return [[0.7, 0.3]]


# Patch du loader pour éviter de charger un vrai modèle
with patch("credit_scoring_project.api.model_loader.load_model") as mock_model:
    mock_model.return_value = {
        "model": FakeModel(),
        "scaler": None,
        "threshold": 0.5,
        "columns": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]
    }

    from credit_scoring_project.api.app import app


client = TestClient(app)


#  Test route home
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


#  Test prédiction valide (STRICT)
def test_predict_valid():
    data = {
        "AMT_INCOME_TOTAL": 50000,
        "AMT_CREDIT": 100000,
        "DAYS_BIRTH": -12000
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert isinstance(result["prediction"], int)
    assert isinstance(result["probability"], float)


#  Type invalide
def test_predict_invalid_type():
    response = client.post("/predict", json={
        "AMT_INCOME_TOTAL": "invalid"
    })

    assert response.status_code == 400


#  Champs manquants
def test_predict_missing_fields():
    response = client.post("/predict", json={})

    assert response.status_code == 400



def test_predict_edge_case():
    data = {
        "AMT_INCOME_TOTAL": 0,
        "AMT_CREDIT": 0,
        "DAYS_BIRTH": -1
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "probability" in result