# -*- coding: utf-8 -*-
"""Schémas Pydantic pour la validation des entrées/sorties de l'API."""

from pydantic import BaseModel, Field


class LoanRequest(BaseModel):
    """Données d'entrée d'un client pour prédire le risque de défaut."""

    loan_amt_outstanding: float = Field(
        ..., gt=0, description="Montant du prêt restant dû (en unité monétaire)"
    )
    income: float = Field(..., gt=0, description="Revenu annuel du client")
    years_employed: int = Field(..., ge=0, le=50, description="Nombre d'années d'emploi (0-50)")
    fico_score: int = Field(..., ge=300, le=850, description="Score FICO de solvabilité (300-850)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "loan_amt_outstanding": 5000.0,
                "income": 45000.0,
                "years_employed": 3,
                "fico_score": 680,
            }
        }
    }


class LoanResponse(BaseModel):
    """Résultat de la prédiction renvoyé par l'API."""

    prediction: int = Field(..., description="0 = pas de défaut, 1 = défaut")
    label: str = Field(..., description="Libellé textuel de la prédiction")
    probability: float = Field(..., description="Probabilité de défaut (0.0 à 1.0)")
    seuil_utilise: float = Field(..., description="Seuil de classification appliqué")


class HealthResponse(BaseModel):
    """Réponse de l'endpoint de santé."""

    status: str
    message: str
    modele: str
    run_id: str
    seuil: float
