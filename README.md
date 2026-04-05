# Projet MLOps : Prédiction de Défaut de Crédit

Projet de Machine Learning supervisée visant à prédire le défaut de paiement d'un emprunteur (variable binaire `default`) à partir de ses caractéristiques financières.

Réalisé dans le cadre du **DU Data Analytics - PS1**.

---

## Objectif

Construire et comparer plusieurs modèles de classification pour identifier les emprunteurs à risque de défaut, en respectant les bonnes pratiques MLOps : reproductibilité, absence de data leakage, validation rigoureuse, interprétabilité et tracking des expériences avec MLflow.

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
|-- Rapport_Etonnement_MLOps.docx      # Rapport d'étonnement du projet
|-- Loan_Data.csv                      # Jeu de données
|-- projetmlops_MLFlow.ipynb             # Notebook version MLflow (tracking des expériences)
|-- Projet_MLOPS.docx                  # Cahier des charges / énoncé
|-- Guide_Explicatif_Notebook.docx     # Explication détaillée de chaque cellule
|-- Schema_Traitements_Pipeline.docx   # Schémas du pipeline, GridSearchCV, SMOTE
|-- Guide_Pedagogique_Complet.docx     # Guide complet pour débutants en Python/ML
|-- Note_Integration_MLflow.docx       # Note d'intégration MLflow
|-- mlruns/                            # Dossier de tracking MLflow (généré à l'exécution)
```

---

## Notebook `projetmlops_MLFlow.ipynb`

Notebook complet intégrant l'ensemble du pipeline de modélisation avec tracking MLflow. Chaque ajout MLflow est marqué par le commentaire `# [MLF0]` dans le code. Le notebook enregistre automatiquement :
- Les hyperparamètres optimaux de chaque modèle
- Les métriques de performance (train, test, validation croisée)
- Les indicateurs d'overfitting (écarts train/test)
- Les modèles sérialisés (pipelines complets StandardScaler + modèle)
- Les graphiques (matrices de confusion, feature importance, learning curves, courbes ROC)

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
| **12.** Sauvegarde d'un modèle XGBoost personnalisé | Entraînement et enregistrement dans MLflow d'une configuration XGBoost hors GridSearchCV (`max_depth=3`, `n_estimators=10`, `scale_pos_weight=5.4`) — modèle retenu pour l'API FastAPI |

---

## Pipeline de traitement

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
 Évaluation : train vs test, matrices de confusion, classification report
      |
      v
 Feature Importance + Learning Curves
      |
      v
 SMOTE (rééquilibrage de la classe minoritaire)
      |
      v
 Ré-entraînement + Comparaison Sans/Avec SMOTE
      |
      v
 [MLflow] Enregistrement des modèles SMOTE + deltas de performance
      |
      v
 Courbes ROC comparatives
      |
      v
 Optimisation du seuil de classification
   -> Courbes Précision-Rappel vs Seuil (par modèle, sans/avec SMOTE)
   -> Seuil optimal maximisant le F1-score
   -> Comparaison matrice de confusion seuil 0.5 vs seuil optimal
      |
      v
 Test multi-seuils (0.45, 0.40, 0.35)
   -> Tableau comparatif par modèle et par seuil
   -> Graphiques Recall et F1 vs Seuil
      |
      v
 Sauvegarde XGBoost personnalisé (hors GridSearchCV)
   -> max_depth=3, n_estimators=10, scale_pos_weight=5.4
   -> Entraînement + log MLflow complet (params, métriques, modèle pickle)
   -> Modèle retenu pour l'API FastAPI
```

---

## Modèles et Hyperparamètres

| Modèle | Hyperparamètres testés | Configs |
|---|---|---|
| Logistic Regression (L1) | C : [0.01, 0.1, 1, 5] | 4 |
| Random Forest | n_estimators : [50, 100, 200], max_depth : [3, 5, 7] | 9 |
| XGBoost | n_estimators : [50, 100, 200], max_depth : [3, 5, 7], scale_pos_weight : [4] | 9 |
| MLP Classifier | hidden_layer_sizes : [(32,20), (80,40,24)] | 2 |

**Total** : 24 configurations x 5 folds = **120 entraînements** par campagne (x2 avec SMOTE).

---

## Stratégie de sélection

Le projet utilise **deux critères distincts** à deux étapes différentes :

| Étape | Critère | Objectif |
|---|---|---|
| **GridSearchCV** (choix des hyperparamètres) | **Recall** (`scoring="recall"`) | Sélectionner le modèle qui détecte le plus de défauts |
| **Seuil optimal** (choix du seuil de décision) | **F1-score** (max) | Ajuster le compromis recall/précision après entraînement |

Cette approche en deux temps est cohérente avec le contexte métier du risque de crédit : on veut d'abord un modèle **sensible** (recall élevé pour ne pas rater de défauts), puis on calibre le seuil pour limiter les faux positifs (F1 comme compromis).

---

## Métriques d'évaluation

| Métrique | Définition | Importance |
|---|---|---|
| **Recall** | TP / (TP + FN) | **Critère de sélection GridSearchCV** (détecter les défauts) |
| F1-score | Moyenne harmonique P/R | **Critère d'optimisation du seuil** (compromis) |
| Précision | TP / (TP + FP) | Limiter les faux positifs |
| AUC-ROC | Aire sous la courbe ROC | Discrimination globale |
| Accuracy | (TP + TN) / Total | Informative (trompeuse si déséquilibre) |

---

## MLflow - Tracking des expériences

### Principe
Le notebook `projetmlops_MLFlow.ipynb` intègre MLflow pour tracer automatiquement chaque entraînement. Aucun serveur distant n'est nécessaire : tout est stocké localement dans le dossier `mlruns/`.

### Runs enregistrés

| Run | Tag smote | Contenu |
|---|---|---|
| Logistic Regression | false | Params + métriques + modèle + confusion matrix |
| Random Forest | false | Params + métriques + modèle + confusion matrix |
| XGBoost | false | Params + métriques + modèle + confusion matrix |
| MLP Classifier | false | Params + métriques + modèle + confusion matrix |
| Logistic Regression_SMOTE | true | Params + métriques (train, test, gaps) + modèle + deltas |
| Random Forest_SMOTE | true | Params + métriques (train, test, gaps) + modèle + deltas |
| XGBoost_SMOTE | true | Params + métriques (train, test, gaps) + modèle + deltas |
| MLP Classifier_SMOTE | true | Params + métriques (train, test, gaps) + modèle + deltas |
| Feature_Importances | - | Graphique feature importance |
| Learning_Curves | - | Graphique learning curves |
| SMOTE_Comparison_ConfusionMatrix | - | Matrices de confusion comparatives |
| ROC_Curves_Comparison | - | Courbes ROC sans/avec SMOTE |
| Threshold_Optimization | - | Seuil optimal, métriques recalculées, graphique comparatif |
| Multi_Threshold_Grid | - | Run parent + 24 runs enfants (8 modèles × 3 seuils), graphique comparatif |
| XGBoost_custom_d3_n10 | false | Modèle XGBoost personnalisé (section 12) retenu pour l'API FastAPI |

### Métriques tracées par run

| Métrique | Description |
|---|---|
| `best_cv_recall` | Meilleur recall en validation croisée (5 folds) |
| `train_accuracy`, `train_recall`, `train_f1` | Performance sur le jeu d'entraînement |
| `test_accuracy`, `test_recall`, `test_f1`, `test_precision` | Performance sur le jeu de test |
| `test_auc_roc` | Aire sous la courbe ROC |
| `gap_accuracy`, `gap_recall` | Écarts train/test (indicateurs d'overfitting, sans ET avec SMOTE) |
| `delta_*` (runs SMOTE) | Écart de performance avec/sans SMOTE |
| `optimal_threshold` | Seuil de classification optimal (maximise le F1) |
| `threshold_recall`, `threshold_precision`, `threshold_f1` | Métriques recalculées au seuil optimal |

### Runs enfants multi-seuils

Le run `Multi_Threshold_Grid` contient **24 runs enfants** (nested runs), un par combinaison modèle × seuil (le seuil 0.50 n'est pas dupliqué car il est déjà enregistré dans les 8 runs de modèles). Chaque run enfant enregistre :

| Paramètre / Métrique | Description |
|---|---|
| `variante` | Sans SMOTE ou Avec SMOTE |
| `modele` | Nom du modèle (Logistic Regression, Random Forest, XGBoost, MLP) |
| `seuil` | Seuil de classification testé (0.45, 0.40, 0.35) |
| `test_accuracy`, `test_recall`, `test_precision`, `test_f1` | Métriques au seuil donné |

Dans MLflow UI, déplier le run parent `Multi_Threshold_Grid` pour accéder aux 32 combinaisons, les trier et les comparer.

### Visualisation des résultats

```bash
cd ProjetMLOPS
mlflow ui
```

Ouvrir **http://localhost:5000** dans le navigateur.

Pour plus de détails, consulter le fichier `GUIDE_MLFLOW.md`.

---

## Optimisation du seuil de classification

Par défaut, `predict()` utilise un seuil de 0.5 : si la probabilité prédite de défaut est >= 0.5, le modèle prédit un défaut. Pour un dataset déséquilibré (18.5% de défauts), ce seuil n'est pas optimal.

### Seuil optimal (F1-score)

Le notebook calcule le seuil optimal maximisant le F1-score pour chaque modèle :

1. **Courbes Précision-Rappel vs Seuil** pour chaque modèle (sans et avec SMOTE)
2. **Courbes Précision vs Rappel** avec le point F1 maximal
3. **Tableau récapitulatif** du seuil optimal, de la précision, du recall et du F1 pour chaque modèle
4. **Comparaison visuelle** : matrice de confusion avec seuil 0.5 vs seuil optimal

### Test multi-seuils

La section 11 du notebook teste chaque modèle (sans et avec SMOTE) sur une grille de seuils prédéfinis (0.45, 0.40, 0.35) et produit :

1. **Tableaux comparatifs** des métriques (Accuracy, Recall, Précision, F1) pour chaque combinaison modèle × seuil
2. **Graphiques** Recall vs Seuil et F1 vs Seuil pour chaque modèle
3. **Tableau du meilleur seuil** par modèle (F1 max parmi les 4 seuils testés)
4. **24 runs enfants MLflow** (8 modèles × 3 seuils) pour comparaison dans l'interface MLflow (le seuil 0.50 est déjà couvert par les runs de modèles)

Un seuil plus bas que 0.5 augmente le recall (davantage de défauts détectés) au prix d'une baisse de précision (plus de faux positifs).

---





### Déséquilibre des classes
- 81.5% non-défaut vs 18.5% défaut
- **Correction** : SMOTE sur le train set + class_weight="balanced" + scale_pos_weight



### Compatibilité pandas 2.x / joblib
- Depuis pandas 2.x, l'option `future.infer_string` peut activer le type `StringDtype` pour les noms de colonnes, incompatible avec la sérialisation joblib utilisée par `GridSearchCV(n_jobs=-1)`
- **Correction** : désactivation de `future.infer_string` + conversion des DataFrames en arrays numpy avant l'entraînement

---

## Prérequis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn statsmodels scipy mlflow
```

| Package | Version minimale | Usage |
|---|---|---|
| pandas | >= 2.0 | Manipulation de données |
| numpy | >= 1.23 | Calcul numérique |
| matplotlib | >= 3.6 | Visualisation |
| seaborn | >= 0.12 | Visualisation statistique |
| scikit-learn | >= 1.2 | Modèles, Pipeline, GridSearchCV, métriques |
| xgboost | >= 1.7 | Gradient boosting |
| imbalanced-learn | >= 0.10 | SMOTE |
| statsmodels | >= 0.13 | VIF, tests statistiques |
| mlflow | >= 2.0 | Tracking des expériences |

---

## Exécution

1. Ouvrir `projetmlops_MLFlow.ipynb` dans Jupyter Notebook ou JupyterLab
2. Exécuter toutes les cellules séquentiellement (Kernel > Restart & Run All)
3. Les résultats s'affichent dans le notebook ET sont enregistrés dans `mlruns/`
4. Lancer `mlflow ui` dans le terminal pour visualiser et comparer les runs

---

## Auteur

Projet réalisé dans le cadre du DU Data Analytics - PS1 (MLOps).
