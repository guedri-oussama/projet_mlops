import joblib
import numpy as np


MODEL_PATH = "model.joblib"
THRESHOLD_PATH = "threshold.joblib"


def load_artifacts():
    """
    Charge le modèle et le seuil sauvegardés.
    """
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold


def prepare_input(
    loan_amt_outstanding: float,
    income: float,
    years_employed: float,
    fico_score: float,
) -> np.ndarray:
    """
    Prépare les données d'entrée au format attendu par le modèle.
    """
    data = np.array([[loan_amt_outstanding, income, years_employed, fico_score]])
    return data


def predict_default(
    loan_amt_outstanding: float,
    income: float,
    years_employed: float,
    fico_score: float,
) -> dict:
    """
    Fait une prédiction à partir des 4 variables d'entrée.
    Retourne un dictionnaire avec la probabilité, le seuil et la classe prédite.
    """
    model, threshold = load_artifacts()
    input_data = prepare_input(
        loan_amt_outstanding=loan_amt_outstanding,
        income=income,
        years_employed=years_employed,
        fico_score=fico_score,
    )

    probability_default = float(model.predict_proba(input_data)[0, 1])
    prediction = 1 if probability_default >= threshold else 0

    return {
        "prediction": prediction,
        "probability_default": probability_default,
        "threshold": float(threshold),
    }