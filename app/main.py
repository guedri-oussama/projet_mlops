# -*- coding: utf-8 -*-
"""Application FastAPI pour la prédiction de défaut de crédit."""

import numpy as np
from fastapi import FastAPI, HTTPException

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    RUN_ID,
    SEUIL,
)
from .model_loader import load_model
from .schemas import HealthResponse, LoanRequest, LoanResponse

# ============================================================
# Initialisation de l'application
# ============================================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Chargement du modèle au démarrage (une seule fois, conservé en mémoire)
model = load_model()


# ============================================================
# Endpoints
# ============================================================
@app.get("/", response_model=HealthResponse)
def root():
    """Endpoint de santé : vérifie que l'API est opérationnelle."""
    return HealthResponse(
        status="ok",
        message="APP de prédiction de défaut de crédit opérationnelle",
        modele="XGBoost (Pipeline MLflow)",
        run_id=RUN_ID,
        seuil=SEUIL,
    )


@app.post("/predict", response_model=LoanResponse)
def predict(request: LoanRequest):
    """
    Prédit le risque de défaut pour un client.

    Le modèle utilisé est un pipeline complet (StandardScaler + XGBoost)
    chargé depuis MLflow au démarrage de l'application.

    Parameters
    ----------
    request : LoanRequest
        Données du client : montant du prêt, revenu, années d'emploi, score FICO.

    Returns
    -------
    LoanResponse
        Prédiction (0 ou 1), libellé, probabilité et seuil appliqué.
    """
    try:
        # Construction de l'array dans l'ordre attendu par le pipeline
        X = np.array(
            [
                [
                    request.loan_amt_outstanding,
                    request.income,
                    request.years_employed,
                    request.fico_score,
                ]
            ]
        )

        # Probabilité de défaut (classe 1)
        proba = float(model.predict_proba(X)[0, 1])

        # Application du seuil de classification
        prediction = int(proba >= SEUIL)

        return LoanResponse(
            prediction=prediction,
            label="Défaut" if prediction == 1 else "Pas de défaut",
            probability=round(proba, 4),
            seuil_utilise=SEUIL,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {exc}")
