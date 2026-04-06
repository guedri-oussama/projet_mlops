# ============================================================
# Image Docker pour l'API FastAPI + Frontend Streamlit
# Prédiction de défaut de crédit (modèle XGBoost)
# ============================================================
# Cette image contient :
#   - FastAPI (uvicorn)    sur le port 8000 (backend API REST)
#   - Streamlit            sur le port 8501 (frontend visuel)
# Les deux services sont lancés en parallèle via start.sh
# Streamlit appelle l'API FastAPI via http://localhost:8000 (même conteneur)
# ============================================================
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Projet MLOps DU Data Analytics PS1"
LABEL description="API FastAPI + Streamlit pour la prédiction de défaut de crédit"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires à XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python (en premier pour bénéficier du cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif (FastAPI + Streamlit + modèle pickle)
COPY app/ ./app/

# Copie et permissions du script de démarrage combiné
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# Exposition des deux ports
EXPOSE 8000 8501

# Vérification de santé du conteneur (via FastAPI)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Commande de démarrage : lance FastAPI puis Streamlit en parallèle
CMD ["./start.sh"]
