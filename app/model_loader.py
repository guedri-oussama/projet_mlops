# -*- coding: utf-8 -*-
"""Chargement du modèle depuis le fichier pickle autonome."""

import joblib

from .config import MODEL_PATH, RUN_ID


def load_model():
    """
    Charge le pipeline complet (StandardScaler + XGBoost) depuis app/model.pkl.

    Le modèle a été préalablement exporté depuis MLflow via export_model.py.
    Il est chargé une seule fois au démarrage de l'application et conservé en mémoire.
    Cette approche ne dépend pas de MLflow à l'exécution, ce qui allège l'image Docker.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline complet prêt à l'emploi (scaler + modèle).
    """
    print("[model_loader] Chargement du modèle")
    print(f"[model_loader] Fichier  : {MODEL_PATH}")
    print(f"[model_loader] Run ID   : {RUN_ID} (traçabilité)")

    model = joblib.load(MODEL_PATH)
    print("[model_loader] Modèle chargé avec succès")
    return model
