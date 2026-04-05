#!/bin/bash
# ============================================================
# Script de démarrage combiné : FastAPI + Streamlit
# ============================================================
# Lance les deux services dans le même conteneur :
#   - FastAPI (uvicorn) sur le port 8000 (backend API)
#   - Streamlit sur le port 8501 (frontend visuel)
#
# Streamlit appelle l'API FastAPI via http://localhost:8000
# (les deux services tournent dans le même conteneur, donc "localhost"
# fonctionne correctement entre eux).
# ============================================================


set -e

# Fonction de nettoyage
cleanup() {
    echo "[start.sh] Arrêt des services..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT
# 1. Démarrage de FastAPI
echo "[start.sh] Démarrage de FastAPI sur le port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# 2. Attente de FastAPI
echo "[start.sh] Attente de FastAPI..."
for i in {1..30}; do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" 2>/dev/null; then
        echo "[start.sh] FastAPI est prêt."
        break
    fi
    sleep 1
done

# 3. Démarrage de Streamlit avec TOUTES les options Cloud
echo "[start.sh] Démarrage de Streamlit sur le port 8501..."
streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.enableWebsocketCompression false \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!

wait -n $FASTAPI_PID $STREAMLIT_PID
cleanup