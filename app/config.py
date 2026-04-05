# -*- coding: utf-8 -*-
"""Configuration de l'application FastAPI."""

from pathlib import Path

# Chemin absolu vers le fichier pickle du modèle (export depuis MLflow)
# Le fichier est généré par export_model.py à la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "app" / "model.pkl"

# Run ID MLflow d'origine (conservé pour traçabilité)
# XGBoost personnalisé : max_depth=3, n_estimators=10, scale_pos_weight=5.4
RUN_ID = "1afd98d2e5044dbd86463f29133456bc"

# Seuil de classification
# Modifier cette valeur pour ajuster le compromis recall/precision
SEUIL = 0.5

# Features attendues (dans l'ordre utilisé lors de l'entraînement)
FEATURES = ["loan_amt_outstanding", "income", "years_employed", "fico_score"]

# Métadonnées API
API_TITLE = "API de prédiction de défaut de crédit"
API_DESCRIPTION = "Projet MLOps - DU Data Analytics PS1 - Modèle XGBoost"
API_VERSION = "1.0.0"
