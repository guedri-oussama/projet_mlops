# -*- coding: utf-8 -*-
"""Interface Streamlit pour la prédiction de défaut de crédit.

Cette application fournit une interface utilisateur visuelle qui appelle
l'API FastAPI (endpoint /predict) et affiche le résultat de manière engageante.

Lancement :
    streamlit run app/streamlit_app.py

Prérequis : le serveur FastAPI doit être démarré sur http://localhost:8000
"""

import requests
import streamlit as st

# ============================================================
# Configuration de la page
# ============================================================
st.set_page_config(
    page_title="Prédiction de défaut de crédit",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ============================================================
# Style CSS personnalisé
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #1F4E79;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .result-ok {
        background: linear-gradient(135deg, #d4f4dd, #a8e6b8);
        color: #1a5d2a;
    }
    .result-ko {
        background: linear-gradient(135deg, #ffd4d4, #f5a8a8);
        color: #6d1f1f;
    }
    .metric-box {
        background: #f2f7fb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E75B6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# En-tête
# ============================================================
st.markdown(
    '<div class="main-title">💳 Prédiction de défaut de crédit</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Projet MLOps — DU Data Analytics PS1 — Modèle XGBoost</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Vérification de l'API
# ============================================================
def check_api():
    """Vérifie que l'API FastAPI est accessible."""
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        if response.status_code == 200:
            return True, response.json()
    except requests.exceptions.RequestException:
        pass
    return False, None


api_ok, api_info = check_api()

if not api_ok:
    st.error(
        "⚠️ L'API FastAPI n'est pas accessible sur http://localhost:8000\n\n"
        "Lance le serveur avec :\n\n"
        "```bash\nuvicorn app.main:app --reload --port 8000\n```"
    )
    st.stop()


# ============================================================
# Sidebar : informations sur le modèle
# ============================================================
with st.sidebar:
    st.header("ℹ️ Informations sur le modèle")
    st.markdown(f"**Modèle :** {api_info['modele']}")
    st.markdown(f"**Seuil de décision :** `{api_info['seuil']}`")
    st.markdown(f"**Statut API :** ✅ {api_info['status']}")
    st.markdown("---")
    st.caption(f"Run ID MLflow :\n`{api_info['run_id']}`")
    st.markdown("---")
    st.markdown(
        "### Comment ça marche ?\n"
        "1. Saisissez les informations du client\n"
        "2. Cliquez sur **Prédire**\n"
        "3. L'API renvoie la probabilité de défaut\n"
        "4. Le seuil détermine la décision finale"
    )


# ============================================================
# Formulaire de saisie
# ============================================================
st.subheader("📋 Données du client")

col1, col2 = st.columns(2)

with col1:
    loan_amt = st.number_input(
        "💰 Montant du prêt restant dû",
        min_value=0.0,
        max_value=50000.0,
        value=5000.0,
        step=500.0,
        help="Montant actuellement dû par le client",
    )
    years_employed = st.slider(
        "💼 Années d'emploi",
        min_value=0,
        max_value=20,
        value=3,
        help="Nombre d'années dans l'emploi actuel",
    )

with col2:
    income = st.number_input(
        "💵 Revenu annuel",
        min_value=0.0,
        max_value=200000.0,
        value=45000.0,
        step=1000.0,
        help="Revenu annuel brut du client",
    )
    fico_score = st.slider(
        "📊 Score FICO",
        min_value=300,
        max_value=850,
        value=680,
        help="Score de solvabilité FICO (300-850). Plus c'est élevé, meilleur est le profil.",
    )

# Indicateur visuel du score FICO
if fico_score >= 800:
    fico_label = "🟢 Excellent"
elif fico_score >= 740:
    fico_label = "🟢 Très bon"
elif fico_score >= 670:
    fico_label = "🟡 Bon"
elif fico_score >= 580:
    fico_label = "🟠 Moyen"
else:
    fico_label = "🔴 Faible"

st.caption(f"Catégorie du score FICO saisi : **{fico_label}**")

st.markdown("")


# ============================================================
# Bouton de prédiction
# ============================================================
if st.button("🔮 Prédire le risque de défaut", type="primary", use_container_width=True):
    payload = {
        "loan_amt_outstanding": loan_amt,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score,
    }

    with st.spinner("Analyse en cours..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur API : {e.response.status_code} — {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.stop()

    # ========================================================
    # Affichage du résultat
    # ========================================================
    st.markdown("---")
    st.subheader("🎯 Résultat de la prédiction")

    proba = result["probability"]
    prediction = result["prediction"]
    label = result["label"]
    seuil = result["seuil_utilise"]

    # Carte de résultat colorée
    if prediction == 0:
        st.markdown(
            f"""
            <div class="result-card result-ok">
                <h2>✅ {label}</h2>
                <p style="font-size: 1.1rem; margin: 0;">Le client présente un profil à faible risque</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-ko">
                <h2>⚠️ {label}</h2>
                <p style="font-size: 1.1rem; margin: 0;">Le client présente un risque de défaut élevé</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Métriques détaillées
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            label="Probabilité de défaut",
            value=f"{proba:.1%}",
            delta=f"{(proba - seuil)*100:+.1f} pts vs seuil" if proba != seuil else None,
            delta_color="inverse",
        )
    with col_b:
        st.metric(label="Seuil de décision", value=f"{seuil:.4f}")
    with col_c:
        st.metric(label="Prédiction", value="Défaut" if prediction == 1 else "Safe")

    # Barre de probabilité visuelle
    st.markdown("**Échelle de probabilité**")
    st.progress(proba)

    col_left, col_mid, col_right = st.columns([1, 1, 1])
    with col_left:
        st.caption("0% — Sans risque")
    with col_mid:
        st.caption(f"↑ Seuil : {seuil:.2%}")
    with col_right:
        st.caption("100% — Défaut certain")

    # Explication de la décision
    st.markdown("---")
    st.subheader("📖 Comment lire ce résultat")

    explanation = (
        f"Le modèle XGBoost a calculé une probabilité de défaut de **{proba:.2%}** "
        f"pour ce client. Comme cette probabilité est "
    )
    if proba >= seuil:
        explanation += (
            f"**supérieure ou égale** au seuil de **{seuil:.2%}**, "
            f"le modèle prédit un **défaut de paiement** (classe 1)."
        )
    else:
        explanation += (
            f"**inférieure** au seuil de **{seuil:.2%}**, "
            f"le modèle prédit **pas de défaut** (classe 0)."
        )

    st.info(explanation)

    with st.expander("🔍 Détails techniques"):
        st.json(result)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Application construite avec Streamlit + FastAPI + MLflow • "
    "Projet MLOps DU Data Analytics PS1"
)
