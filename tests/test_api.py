from fastapi.testclient import TestClient
from credit_scoring_project.api.app import app

client = TestClient(app)


# test route home
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API Credit Scoring active"}


# test prédiction valide
def test_predict_valid():
    response = client.post("/predict", json={
        "AMT_INCOME_TOTAL": 50000,
        "AMT_CREDIT": 100000,
        "DAYS_BIRTH": -12000
    })

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "probability" in data


#  test données invalides
def test_predict_invalid():
    response = client.post("/predict", json={
        "AMT_INCOME_TOTAL": "invalid"
    })

    assert response.status_code == 400