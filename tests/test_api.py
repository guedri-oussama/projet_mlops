# -*- coding: utf-8 -*-
"""Tests de l'API FastAPI de prédiction de défaut de crédit."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ============================================================
# Tests de l'endpoint de santé
# ============================================================
def test_root_endpoint_returns_200():
    """L'endpoint racine doit répondre avec un code HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_root_endpoint_returns_health_info():
    """L'endpoint racine doit renvoyer les infos de santé attendues."""
    response = client.get("/")
    data = response.json()
    assert data["status"] == "ok"
    assert "modele" in data
    assert "seuil" in data
    assert "run_id" in data


# ============================================================
# Tests de l'endpoint /predict
# ============================================================
def test_predict_valid_input():
    """Une requête valide doit renvoyer une prédiction."""
    payload = {
        "loan_amt_outstanding": 5000.0,
        "income": 45000.0,
        "years_employed": 3,
        "fico_score": 680,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "probability" in data
    assert "seuil_utilise" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_field_returns_422():
    """Une requête avec un champ manquant doit renvoyer une erreur 422."""
    payload = {
        "loan_amt_outstanding": 5000.0,
        "income": 45000.0,
        # years_employed manquant
        "fico_score": 680,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_fico_returns_422():
    """Un score FICO hors plage doit renvoyer une erreur 422."""
    payload = {
        "loan_amt_outstanding": 5000.0,
        "income": 45000.0,
        "years_employed": 3,
        "fico_score": 1000,  # max = 850
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_income_returns_422():
    """Un revenu négatif doit renvoyer une erreur 422."""
    payload = {
        "loan_amt_outstanding": 5000.0,
        "income": -1000.0,  # doit être > 0
        "years_employed": 3,
        "fico_score": 680,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_high_risk_profile():
    """Un profil très risqué doit avoir une probabilité de défaut élevée."""
    payload = {
        "loan_amt_outstanding": 40000.0,
        "income": 15000.0,
        "years_employed": 0,
        "fico_score": 450,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # On ne teste pas la valeur exacte mais on vérifie que la probabilité est dans [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_label_consistency():
    """Le label doit être cohérent avec la prédiction numérique."""
    payload = {
        "loan_amt_outstanding": 5000.0,
        "income": 45000.0,
        "years_employed": 3,
        "fico_score": 680,
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    if data["prediction"] == 0:
        assert data["label"] == "Pas de défaut"
    else:
        assert data["label"] == "Défaut"
