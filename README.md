# Projet MLOps : Prédiction de Défaut de Crédit

Projet de Machine Learning supervisée visant à prédire le défaut de paiement d'un emprunteur (variable binaire `default`) à partir de ses caractéristiques financières.

Réalisé dans le cadre du **SDA Data Analytics 25/26**.

Ce projet couvre l'intégralité d'un cycle MLOps : de l'analyse exploratoire du dataset, à l'entraînement et au suivi des modèles avec MLflow, à l'exposition du modèle retenu via une API FastAPI, jusqu'au déploiement automatisé sur Docker Hub et AWS via GitHub Actions.

---

## 🚀 Aperçu du Projet

* **Modèle** : XGBoost (optimisé via GridSearchCV & MLflow).
* **Backend** : FastAPI (Inférence temps réel).
* **Frontend** : Streamlit (Interface utilisateur interactive).
* **Conteneurisation** : Docker (Image multi-services).
* **CI/CD** : GitHub Actions (Linting, Tests, Push Docker Hub).
* **Cloud** : AWS ECS (Fargate) + Application Load Balancer (ALB).
---

## Sommaire

1. [Dataset](#dataset)
2. [Structure du projet](#structure-du-projet)
3. [Notebook et pipeline ML](#notebook-et-pipeline-ml)
4. [Stratégie de sélection et métriques](#stratégie-de-sélection-et-métriques)
5. [MLflow - Tracking des expériences](#mlflow---tracking-des-expériences)
6. [Choix du modèle pour la production](#choix-du-modèle-pour-la-production)
7. [Optimisation du seuil de classification](#optimisation-du-seuil-de-classification)
8. [Export du modèle pour la production](#export-du-modèle-pour-la-production)
9. [API FastAPI](#api-fastapi)
10. [Interface Streamlit](#interface-streamlit)
11. [Image Docker intégrée FastAPI + Streamlit](#image-docker-intégrée-fastapi--streamlit)
12. [Pipeline CI/CD GitHub Actions](#pipeline-cicd-github-actions)
13. [Déploiement complet pas à pas](#déploiement-complet-pas-à-pas)
14. [Problèmes identifiés et corrections](#problèmes-identifiés-et-corrections)
15. [Déploiement Cloud sur AWS](#ECS-Fargate)
16. [Prérequis et exécution](#prérequis-et-exécution)

---

## Dataset

| Fichier | Description |
|---|---|
| `Loan_Data.csv` | 10 000 observations, 8 variables |

### Variables

| Variable | Type | Description | Statut |
|---|---|---|---|
| `customer_id` | int | Identifiant unique du client | Supprimée |
| `credit_lines_outstanding` | int (0-5) | Nombre de lignes de crédit en cours | Supprimée (data leaker) |
| `loan_amt_outstanding` | float | Montant du prêt restant dû | **Feature retenue** |
| `total_debt_outstanding` | float | Dette totale en cours | Supprimée (multicolinéarité) |
| `income` | float | Revenu annuel | **Feature retenue** |
| `years_employed` | int (0-10) | Années d'emploi | **Feature retenue** |
| `fico_score` | int (408-850) | Score de solvabilité FICO | **Feature retenue** |
| `default` | int (0/1) | **Variable cible** (0 = pas de défaut, 1 = défaut) | Cible |

---

## Structure du Projet

```
ProjetMLOPS/
|-- README.md                          # Ce fichier
|-- GUIDE_MLFLOW.md                    # Guide d'utilisation de MLflow
|-- Rapport_Etonnement_MLOps.docx      # Rapport d'étonnement
|-- Guide_Pedagogique_MLOps_Complet.docx  # Guide pédagogique détaillé (47 sections)
|-- Loan_Data.csv                      # Jeu de données
|-- projetmlops_MLFlow.ipynb           # Notebook de modélisation avec tracking MLflow
|-- export_model.py                    # Script d'export du modèle MLflow vers pickle
|-- mlruns/                            # Dossier de tracking MLflow (local)
|-- app/                               # API FastAPI + frontend Streamlit
|   |-- __init__.py
|   |-- config.py                      # RUN_ID, seuil, chemins
|   |-- schemas.py                     # Schémas Pydantic (validation)
|   |-- model_loader.py                # Chargement du modèle via joblib
|   |-- main.py                        # Endpoints FastAPI
|   |-- streamlit_app.py               # Interface utilisateur visuelle
|   |-- model.pkl                      # Modèle exporté (autonome, sans MLflow)
|-- tests/                             # Tests pytest de l'API
|   |-- __init__.py
|   |-- test_api.py                    # 8 tests de l'endpoint /predict
|-- .github/workflows/
|   |-- ci.yml                         # Pipeline CI/CD (Black, Flake8, pytest, Docker)
|-- Dockerfile                         # Image FastAPI + Streamlit
|-- start.sh                           # Script de démarrage combiné des 2 services
|-- .dockerignore                      # Exclusions du contexte Docker
|-- .gitignore                         # Exclusions Git
|-- .flake8                            # Config PEP 8 (compatible Black)
|-- pyproject.toml                     # Config Black + isort
|-- requirements.txt                   # Dépendances production (image Docker)
|-- requirements-dev.txt               # Dépendances dev + CI (Black, Flake8, pytest)
|-- github_ready/                      # Copie minimale des fichiers pour push GitHub
```

---

## Notebook et pipeline ML

### Le notebook `projetmlops_MLFlow.ipynb`

Notebook complet intégrant l'ensemble du pipeline de modélisation avec tracking MLflow. Chaque ajout MLflow est marqué par le commentaire `# [MLF0]` dans le code.

### Sommaire du notebook

| Section | Contenu |
|---|---|
| **1.** Installation des dépendances | pip install |
| **2.** Imports et configuration | Imports Python + MLflow |
| **3.** Chargement et exploration (EDA) | Données manquantes, types, corrélation, distributions, boxplots |
| **4.** Investigation Data Leakage | Crosstab credit_lines_outstanding vs default, FICO score |
| **5.** Analyse de multicolinéarité (VIF) | VIF, suppression total_debt_outstanding |
| **6.** Préparation des données | Sélection features, split 80/20, conversion numpy |
| **7.** Modélisation et GridSearchCV | Définition modèles, entraînement, classement des configurations |
| **8.** Analyse post-entraînement | Importance des variables, courbes d'apprentissage |
| **9.** Test de robustesse avec SMOTE | Génération synthétique, ré-entraînement, comparaison, courbes ROC |
| **10.** Optimisation du seuil de classification | Courbes Précision-Rappel, tableau des seuils, comparaison |
| **11.** Test multi-seuils | Grille de seuils (0.45, 0.40, 0.35) sur tous les modèles |
| **12.** Sauvegarde d'un modèle XGBoost personnalisé | Configuration retenue pour l'API FastAPI |

### Pipeline de traitement

```
Loan_Data.csv
      |
      v
 Chargement & Exploration (EDA)
      |
      v
 Investigation Data Leakage
   -> credit_lines_outstanding = leaker (supprimé)
      |
      v
 Analyse VIF (multicolinéarité)
   -> total_debt_outstanding supprimé
      |
      v
 Sélection des features (4 retenues)
      |
      v
 Train / Test Split (80/20, stratifié)
      |
      v
 Conversion numpy (compatibilité pandas 2.x / joblib)
      |
      v
 Pipeline(StandardScaler + Modèle) x GridSearchCV(5 folds)
   -> Logistic Regression (L1)
   -> Random Forest
   -> XGBoost
   -> MLP (réseau de neurones)
      |
      v
 [MLflow] Enregistrement params, métriques, modèles, artifacts
      |
      v
 SMOTE + Ré-entraînement + Comparaison
      |
      v
 Optimisation du seuil (F1 max par modèle)
      |
      v
 Test multi-seuils (0.45, 0.40, 0.35)
      |
      v
 Sauvegarde XGBoost personnalisé -> modèle retenu
```

### Modèles et hyperparamètres testés

| Modèle | Hyperparamètres | Combinaisons |
|---|---|---|
| Logistic Regression (L1) | C : [0.01, 0.1, 1, 5] | 4 |
| Random Forest | n_estimators : [50, 100, 200], max_depth : [3, 5, 7] | 9 |
| XGBoost | n_estimators, max_depth, scale_pos_weight | 9 |
| MLP Classifier | hidden_layer_sizes : [(32,20), (80,40,24)] | 2 |

**Total** : 24 configurations × 5 folds × 2 campagnes (avec/sans SMOTE) = **240 entraînements**.

---

## Stratégie de sélection et métriques

Le projet utilise **deux critères distincts** à deux étapes différentes du pipeline :

| Étape | Critère | Objectif |
|---|---|---|
| **GridSearchCV** (choix des hyperparamètres) | **Recall** (`scoring="recall"`) | Sélectionner le modèle qui détecte le plus de défauts |
| **Seuil optimal** (choix du seuil de décision) | **F1-score** (max) | Ajuster le compromis recall/précision après entraînement |

### Métriques d'évaluation

| Métrique | Définition | Importance |
|---|---|---|
| **Recall** | TP / (TP + FN) | **Critère GridSearchCV** (détecter les défauts) |
| F1-score | Moyenne harmonique P/R | **Critère d'optimisation du seuil** (compromis) |
| Précision | TP / (TP + FP) | Limiter les faux positifs |
| AUC-ROC | Aire sous la courbe ROC | Discrimination globale |
| Accuracy | (TP + TN) / Total | Informative (trompeuse si déséquilibre) |

---

## MLflow - Tracking des expériences

### Principe

Le notebook intègre MLflow pour tracer automatiquement chaque entraînement. Aucun serveur distant n'est nécessaire : tout est stocké localement dans le dossier `mlruns/`.

### Runs enregistrés

| Run | Tag smote | Contenu |
|---|---|---|
| Logistic Regression | false | Params + métriques + modèle + confusion matrix |
| Random Forest | false | Params + métriques + modèle + confusion matrix |
| XGBoost | false | Params + métriques + modèle + confusion matrix |
| MLP Classifier | false | Params + métriques + modèle + confusion matrix |
| Logistic Regression_SMOTE | true | Params + métriques (train, test, gaps) + modèle + deltas |
| Random Forest_SMOTE | true | Params + métriques + modèle + deltas |
| XGBoost_SMOTE | true | Params + métriques + modèle + deltas |
| MLP Classifier_SMOTE | true | Params + métriques + modèle + deltas |
| Feature_Importances | - | Graphique feature importance |
| Learning_Curves | - | Graphique learning curves |
| SMOTE_Comparison_ConfusionMatrix | - | Matrices de confusion comparatives |
| ROC_Curves_Comparison | - | Courbes ROC sans/avec SMOTE |
| Threshold_Optimization | - | Seuil optimal, métriques recalculées, graphique comparatif |
| Multi_Threshold_Grid | - | Run parent + 24 runs enfants (8 modèles × 3 seuils) |
| **XGBoost_custom_d3_n10** | false | **Modèle XGBoost personnalisé retenu pour l'API FastAPI** |

### Visualisation

```bash
cd ProjetMLOPS
mlflow ui
```

Ouvrir **http://localhost:5000** dans le navigateur.

---

## Choix du modèle pour la production

Le modèle retenu pour la production n'est **pas issu directement de GridSearchCV** mais a été entraîné séparément dans la section 12 du notebook, avec une configuration volontairement simple et légère :

| Paramètre | Valeur | Justification |
|---|---|---|
| `max_depth` | 3 | Modèle peu profond, évite l'overfitting |
| `n_estimators` | 10 | Seulement 10 arbres — modèle très léger |
| `scale_pos_weight` | 5.4 | Compense le déséquilibre 81.5%/18.5% |
| `random_state` | 42 | Reproductibilité |

**Run ID MLflow** : `1afd98d2e5044dbd86463f29133456bc`
**Taille du pickle** : 16 KB seulement
**Avantages** : image Docker légère, inférence ultra rapide, comportement prévisible.

---

## Optimisation du seuil de classification

Par défaut, `predict()` utilise un seuil de 0.5. Pour un dataset déséquilibré (18.5% de défauts), ce seuil n'est pas optimal.

### Seuil optimal F1 par modèle

Le notebook calcule le seuil qui maximise le F1-score pour chaque modèle via la courbe Précision-Rappel.

### Test multi-seuils

La section 11 du notebook teste chaque modèle sur une grille de seuils prédéfinis (0.45, 0.40, 0.35) et produit :

1. **Tableaux comparatifs** des métriques (Accuracy, Recall, Précision, F1) par combinaison modèle × seuil
2. **Graphiques** Recall et F1 vs Seuil pour chaque modèle
3. **Tableau du meilleur seuil** par modèle (F1 max parmi les 3 seuils testés)
4. **24 runs enfants MLflow** (8 modèles × 3 seuils) pour comparaison

### Seuil utilisé par l'API FastAPI

Le fichier `app/config.py` définit `SEUIL = 0.6276` pour le modèle XGBoost retenu. Ce seuil peut être modifié sans ré-entraîner le modèle.

---

## Export du modèle pour la production

Bien que MLflow soit utilisé pour le tracking pendant l'expérimentation, il est **retiré de l'image Docker** pour alléger le déploiement. Le modèle est exporté sous forme d'un fichier pickle autonome via `export_model.py` :

```python
import joblib
import mlflow.sklearn

mlflow.set_tracking_uri("mlruns")
model = mlflow.sklearn.load_model("runs:/1afd98d2e5044dbd86463f29133456bc/model")
joblib.dump(model, "app/model.pkl")
```

L'API charge ensuite directement le pickle via `joblib.load("app/model.pkl")`, sans dépendance à MLflow. Gains :

| Mesure | Avec MLflow dans l'image | Sans MLflow (joblib) |
|---|---|---|
| Taille des dépendances | ~500 MB | ~200 MB |
| Temps d'installation pip | ~3-4 min | ~1 min |
| Démarrage du conteneur | ~3-4 s | ~1 s |

---

## API FastAPI

### Rôle

Expose le modèle XGBoost comme un endpoint HTTP `/predict` accessible par n'importe quel client (navigateur, Postman, script, autre API).

### Architecture du dossier `app/`

```
app/
|-- config.py           # RUN_ID, SEUIL, chemin vers model.pkl
|-- schemas.py          # LoanRequest, LoanResponse (Pydantic)
|-- model_loader.py     # Chargement du pipeline via joblib
|-- main.py             # FastAPI + endpoints GET / et POST /predict
|-- streamlit_app.py    # Interface utilisateur visuelle
|-- model.pkl           # Pipeline sérialisé (16 KB)
```

### Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Health check + infos sur le modèle |
| `/predict` | POST | Prédiction avec 4 features en entrée |
| `/docs` | GET | Documentation Swagger interactive (auto-générée) |
| `/redoc` | GET | Documentation ReDoc |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"loan_amt_outstanding": 5000, "income": 45000, "years_employed": 3, "fico_score": 680}'
```

### Réponse

```json
{
  "prediction": 0,
  "label": "Pas de défaut",
  "probability": 0.6157,
  "seuil_utilise": 0.6276
}
```

### Tests automatisés

Le dossier `tests/` contient **8 tests pytest** qui vérifient :
- L'endpoint de santé (code 200, contenu JSON)
- Une prédiction valide
- Les erreurs de validation (champ manquant, FICO hors plage, revenu négatif)
- La cohérence entre la prédiction numérique et le label

---

## Interface Streamlit

### Rôle

Interface web visuelle qui **appelle l'API FastAPI via HTTP** (`requests.post`). Elle ne charge pas le modèle elle-même.

### Architecture client-serveur dans le conteneur

```
Utilisateur (navigateur)
      |
      | formulaire + bouton Prédire
      v
Streamlit (port 8501)
      |
      | requests.post('http://localhost:8000/predict', json={...})
      v
FastAPI (port 8000)
      |
      | Pydantic valide -> model.predict_proba -> renvoie JSON
      v
Streamlit affiche le résultat visuellement
```

### Fonctionnalités

- Formulaire avec 4 champs (montant, revenu, années emploi, FICO)
- Indicateur visuel du score FICO (Excellent / Bon / Moyen / Faible)
- Sidebar avec infos du modèle et run_id
- Carte de résultat colorée (vert = pas de défaut, rouge = défaut)
- Barre de probabilité vs seuil
- Explication textuelle de la décision
- Détails techniques dépliables

---

## Image Docker intégrée FastAPI + Streamlit

### Architecture du conteneur

L'image embarque **les deux services** dans un seul conteneur, lancés en parallèle via `start.sh` :

```
+-------------------------------------------+
|  Image Docker unique                      |
|                                           |
|  +-----------------+  +----------------+  |
|  | FastAPI         |  | Streamlit      |  |
|  | uvicorn         |<-| (frontend UI)  |  |
|  | port 8000       |  | port 8501      |  |
|  | (backend API)   |  | requests.post  |  |
|  +-----------------+  +----------------+  |
|         ^                    ^            |
|         |                    |            |
|         +------ start.sh ----+            |
|                                           |
|  app/model.pkl (16 KB)                    |
+-------------------------------------------+
            |                    |
            v                    v
   docker run -p 8000:8000 -p 8501:8501 <image>
            |                    |
            v                    v
    http://localhost:8000  http://localhost:8501
    (API + /docs)          (Interface visuelle)
```

### Le script `start.sh`

Lance `uvicorn` en arrière-plan, attend qu'il soit prêt, puis démarre `streamlit`. Gère proprement les signaux SIGTERM/SIGINT pour arrêter les deux services ensemble.

### Dépendances (requirements.txt)

```
fastapi>=0.100
uvicorn[standard]>=0.23
pydantic>=2.0
joblib>=1.3
scikit-learn>=1.2
xgboost>=1.7
numpy>=1.23
streamlit>=1.30
requests>=2.31
```

**Note** : MLflow n'est **pas** inclus — le modèle est chargé via joblib directement depuis le pickle.

### Lancement local

```bash
docker build -t projetmlops-api .
docker run -p 8000:8000 -p 8501:8501 projetmlops-api
```

Puis ouvrir :
- `http://localhost:8000/docs` pour Swagger (API FastAPI)
- `http://localhost:8501` pour l'interface Streamlit

---

## Pipeline CI/CD GitHub Actions

Le fichier `.github/workflows/ci.yml` définit un pipeline à **3 jobs séquentiels** qui s'exécutent à chaque push :

### Job 1 — Lint (qualité du code)

- Installe Black et Flake8
- `black --check app/ tests/` : vérifie le formatage
- `flake8 app/ tests/` : vérifie le respect de PEP 8

### Job 2 — Test (pytest)

- Installe toutes les dépendances (`requirements-dev.txt`)
- Exécute les 8 tests pytest
- Dépend du job Lint (ne tourne que si Lint passe)

### Job 3 — Docker (build + push)

- Se connecte à Docker Hub (via les secrets GitHub)
- Construit l'image Docker depuis le Dockerfile
- Publie l'image sur Docker Hub
- Dépend du job Test
- **Condition** : ne se déclenche que sur les branches `main`, `master` ou `dev`

```yaml
if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master' || github.ref == 'refs/heads/dev')
```

### Les 2 secrets GitHub requis

| Nom | Valeur | Où le trouver |
|---|---|---|
| `DOCKERHUB_USERNAME` | Nom d'utilisateur Docker Hub | hub.docker.com profil |
| `DOCKERHUB_TOKEN` | Token d'accès personnel | hub.docker.com/settings/security |

**Important** : les secrets sont **spécifiques à chaque repository GitHub**. Il faut les reconfigurer si on crée un nouveau repo ou si on fork.

---

## Déploiement complet pas à pas

### Étape 1 — Entraîner et sélectionner le modèle

1. Ouvrir `projetmlops_MLFlow.ipynb` dans Jupyter
2. Exécuter toutes les cellules (section 1 à 12)
3. Le modèle XGBoost personnalisé (section 12) est tracé dans MLflow avec un `run_id`
4. Noter le `run_id` pour l'étape suivante

### Étape 2 — Exporter le modèle

```bash
python export_model.py
```

Cela crée `app/model.pkl` (16 KB, autonome, sans MLflow).

### Étape 3 — Tester localement

```bash
# Lancer l'API en local
uvicorn app.main:app --reload --port 8000

# Dans un autre terminal, tester
curl http://localhost:8000/
pytest tests/ -v
```

### Étape 4 — Vérifier la qualité du code

```bash
python -m black --check app/ tests/
python -m flake8 app/ tests/
```

### Étape 5 — Build Docker local (optionnel)

```bash
docker build -t projetmlops-api .
docker run -p 8000:8000 -p 8501:8501 projetmlops-api
```

Ouvrir http://localhost:8000/docs et http://localhost:8501 pour vérifier.

### Étape 6 — Créer un token Docker Hub

1. Aller sur https://hub.docker.com/settings/security
2. **New Access Token** → permissions **Read, Write, Delete**
3. **Copier le token immédiatement** (non récupérable ensuite)

### Étape 7 — Créer le repo GitHub

Via github.com/new ou directement via GitHub Desktop (Publish repository).

### Étape 8 — Configurer les secrets GitHub

Sur le repo GitHub : **Settings** → **Secrets and variables** → **Actions** :
- `DOCKERHUB_USERNAME` = nom Docker Hub
- `DOCKERHUB_TOKEN` = token copié à l'étape 6

### Étape 9 — Pusher le code

Via GitHub Desktop (recommandé) ou Git CLI :

```bash
git add .
git commit -m "Initial commit - MLOps API"
git push
```

### Étape 10 — Suivre la CI/CD

Dans l'onglet **Actions** du repo GitHub, le workflow se déclenche automatiquement. Les 3 jobs s'enchaînent :

1. **Lint** (~30 s)
2. **Test** (~1 min)
3. **Docker** (~4-6 min) → image publiée sur Docker Hub

### Étape 11 — Utiliser l'image déployée

Depuis n'importe quelle machine avec Docker :

```bash
docker pull <ton-username>/projetmlops-api:latest
docker run -p 8000:8000 -p 8501:8501 <ton-username>/projetmlops-api
```

---

## Problèmes identifiés et corrections

### Data Leakage
- `credit_lines_outstanding` encodait quasi-directement la cible (modalité 5 = 99.8% de défaut)
- **Correction** : variable supprimée des features

### Scaling prématuré
- Le code original appliquait `StandardScaler` sur tout le dataset avant le split
- **Correction** : scaling intégré dans le Pipeline (appliqué uniquement sur le train à chaque fold CV)

### Déséquilibre des classes
- 81.5% non-défaut vs 18.5% défaut
- **Correction** : SMOTE + `class_weight="balanced"` + `scale_pos_weight`

### Compatibilité pandas 2.x / joblib
- `future.infer_string` peut activer `StringDtype`, incompatible avec la sérialisation joblib
- **Correction** : désactivation de l'option + conversion en arrays numpy avant GridSearchCV

### Erreurs CI rencontrées au premier push
- **Black** : code mal formaté → `python -m black app/ tests/` en local avant le push
- **Flake8 F541** : f-strings sans placeholder → `f"..."` remplacé par `"..."`
- **Warning Node.js 20** : ajout de `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"` dans le workflow

### Bug : incohérence `.gitignore` / `.dockerignore`
- `streamlit_app.py` retiré de `.dockerignore` mais oublié dans `.gitignore`
- Le fichier n'arrivait jamais sur GitHub → l'image Docker ne le contenait pas → `start.sh` plantait
- **Correction** : retrait de `app/streamlit_app.py` dans les DEUX fichiers

### Job Docker skipped sur une branche non standard
- La condition initiale n'incluait que `main` et `master`
- **Correction** : ajout de `dev` dans la condition `if:` du job Docker

### Secrets Docker Hub
- Les secrets sont spécifiques à chaque repository GitHub
- Ne pas oublier de les reconfigurer si on crée un nouveau repo

---

## Prérequis et exécution

### Installation des dépendances

```bash
pip install -r requirements-dev.txt
```

| Package | Version minimale | Usage |
|---|---|---|
| pandas | >= 2.0 | Manipulation de données |
| numpy | >= 1.23 | Calcul numérique |
| scikit-learn | >= 1.2 | Pipeline, GridSearchCV, métriques |
| xgboost | >= 1.7 | Gradient boosting |
| imbalanced-learn | >= 0.10 | SMOTE |
| mlflow | >= 2.0 | Tracking des expériences |
| fastapi | >= 0.100 | API backend |
| uvicorn[standard] | >= 0.23 | Serveur ASGI |
| pydantic | >= 2.0 | Validation des entrées |
| joblib | >= 1.3 | Chargement du modèle pickle |
| streamlit | >= 1.30 | Interface utilisateur |
| pytest | >= 7.0 | Tests unitaires |
| black | >= 24.0 | Formatage |
| flake8 | >= 6.0 | Linting PEP 8 |

### Exécution du notebook

1. Ouvrir `projetmlops_MLFlow.ipynb` dans Jupyter Notebook ou JupyterLab
2. Exécuter toutes les cellules séquentiellement (Kernel > Restart & Run All)
3. Les résultats s'affichent et sont enregistrés dans `mlruns/`
4. Lancer `mlflow ui` pour visualiser

### Lancement de l'API en local (sans Docker)

```bash
uvicorn app.main:app --reload --port 8000
# API sur http://localhost:8000/docs
```

### Lancement de Streamlit en local

```bash
streamlit run app/streamlit_app.py
# Interface sur http://localhost:8501
```

### Lancement complet avec Docker

```bash
docker build -t projetmlops-api .
docker run -p 8000:8000 -p 8501:8501 projetmlops-api
```

---

## Auteurs

GUEDRI Oussama - DRUI Bernard - BANAIAS Patrice
