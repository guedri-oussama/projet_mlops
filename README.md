# Projet MLOps : Prediction de Defaut de Credit

Projet de Machine Learning supervisee visant a predire le defaut de paiement d'un emprunteur (variable binaire `default`) a partir de ses caracteristiques financieres.

Realise dans le cadre du **DU Data Analytics - PS1**.

---

## Objectif

Construire et comparer plusieurs modeles de classification pour identifier les emprunteurs a risque de defaut, en respectant les bonnes pratiques MLOps : reproductibilite, absence de data leakage, validation rigoureuse, interpretabilite et tracking des experiences avec MLflow.

---

## Dataset

| Fichier | Description |
|---|---|
| `Loan_Data.csv` | 10 000 observations, 8 variables |

### Variables

| Variable | Type | Description | Statut |
|---|---|---|---|
| `customer_id` | int | Identifiant unique du client | Supprimee |
| `credit_lines_outstanding` | int (0-5) | Nombre de lignes de credit en cours | Supprimee (data leaker) |
| `loan_amt_outstanding` | float | Montant du pret restant du | **Feature retenue** |
| `total_debt_outstanding` | float | Dette totale en cours | Supprimee (multicolinearite) |
| `income` | float | Revenu annuel | **Feature retenue** |
| `years_employed` | int (0-10) | Annees d'emploi | **Feature retenue** |
| `fico_score` | int (408-850) | Score de solvabilite FICO | **Feature retenue** |
| `default` | int (0/1) | **Variable cible** (0 = pas de defaut, 1 = defaut) | Cible |

---

## Structure du Projet

```
ProjetMLOPS/
|-- README.md                          # Ce fichier
|-- GUIDE_MLFLOW.md                    # Guide d'utilisation de MLflow
|-- Loan_Data.csv                      # Jeu de donnees
|-- projetmlops.ipynb                  # Notebook principal (sans MLflow)
|-- projetmlops_MLF0.ipynb             # Notebook version MLflow (tracking des experiences)
|-- Projet_MLOPS.docx                  # Cahier des charges / enonce
|-- Guide_Explicatif_Notebook.docx     # Explication detaillee de chaque cellule
|-- Schema_Traitements_Pipeline.docx   # Schemas du pipeline, GridSearchCV, SMOTE
|-- Guide_Pedagogique_Complet.docx     # Guide complet pour debutants en Python/ML
|-- mlruns/                            # Dossier de tracking MLflow (genere a l'execution)
```

---

## Notebooks

### `projetmlops.ipynb` - Version originale
Notebook de reference contenant l'ensemble du pipeline de modelisation, de l'exploration des donnees jusqu'a la comparaison des modeles avec SMOTE. Ne contient pas de tracking MLflow.

### `projetmlops_MLF0.ipynb` - Version MLflow
Copie du notebook original enrichie avec l'integration MLflow. Chaque ajout est marque par le commentaire `# [MLF0]`. Cette version enregistre automatiquement :
- Les hyperparametres optimaux de chaque modele
- Les metriques de performance (train, test, validation croisee)
- Les indicateurs d'overfitting (ecarts train/test)
- Les modeles serialises (pipelines complets)
- Les graphiques (matrices de confusion, feature importance, learning curves, courbes ROC)

---

## Pipeline du Notebook

```
Loan_Data.csv
      |
      v
 Chargement & Exploration (EDA)
      |
      v
 Investigation Data Leakage
   -> credit_lines_outstanding = leaker (supprime)
      |
      v
 Analyse VIF (multicolinearite)
   -> total_debt_outstanding supprime
      |
      v
 Selection des features (4 retenues)
      |
      v
 Train / Test Split (80/20, stratifie)
      |
      v
 Conversion numpy (compatibilite pandas 2.x / joblib)
      |
      v
 Pipeline(StandardScaler + Modele) x GridSearchCV(5 folds)
   -> Logistic Regression (L1)
   -> Random Forest
   -> XGBoost
   -> MLP (reseau de neurones)
      |
      v
 [MLflow] Enregistrement params, metriques, modeles, artifacts
      |
      v
 Evaluation : train vs test, matrices de confusion, classification report
      |
      v
 Feature Importance + Learning Curves
      |
      v
 SMOTE (reequilibrage de la classe minoritaire)
      |
      v
 Re-entrainement + Comparaison Sans/Avec SMOTE
      |
      v
 [MLflow] Enregistrement des modeles SMOTE + deltas de performance
      |
      v
 Courbes ROC comparatives
      |
      v
 Optimisation du seuil de classification
   -> Courbes Precision-Recall vs Seuil (par modele, sans/avec SMOTE)
   -> Seuil optimal maximisant le F1-score
   -> Comparaison matrice de confusion seuil 0.5 vs seuil optimal
```

---

## Modeles et Hyperparametres

| Modele | Hyperparametres testes | Configs |
|---|---|---|
| Logistic Regression (L1) | C : [0.01, 0.1, 1, 5] | 4 |
| Random Forest | n_estimators : [50, 100, 200], max_depth : [3, 5, 7] | 9 |
| XGBoost | n_estimators : [50, 100, 200], max_depth : [3, 5, 7], scale_pos_weight : [4] | 9 |
| MLP Classifier | hidden_layer_sizes : [(32,20), (80,40,24)] | 2 |

**Total** : 24 configurations x 5 folds = **120 entrainements** par campagne (x2 avec SMOTE).

---

## Metriques d'Evaluation

| Metrique | Definition | Importance |
|---|---|---|
| **Recall** | TP / (TP + FN) | **Metrique principale** (detecter les defauts) |
| Precision | TP / (TP + FP) | Limiter les faux positifs |
| F1-score | Moyenne harmonique P/R | Compromis |
| AUC-ROC | Aire sous la courbe ROC | Discrimination globale |
| Accuracy | (TP + TN) / Total | Informative (trompeuse si desequilibre) |

---

## MLflow - Tracking des Experiences

### Principe
La version MLF0 du notebook integre MLflow pour tracer automatiquement chaque entrainement. Aucun serveur distant n'est necessaire : tout est stocke localement dans le dossier `mlruns/`.

### Runs enregistres

| Run | Tag smote | Contenu |
|---|---|---|
| Logistic Regression | false | Params + metriques + modele + confusion matrix |
| Random Forest | false | Params + metriques + modele + confusion matrix |
| XGBoost | false | Params + metriques + modele + confusion matrix |
| MLP Classifier | false | Params + metriques + modele + confusion matrix |
| Logistic Regression_SMOTE | true | Params + metriques + modele + deltas |
| Random Forest_SMOTE | true | Params + metriques + modele + deltas |
| XGBoost_SMOTE | true | Params + metriques + modele + deltas |
| MLP Classifier_SMOTE | true | Params + metriques + modele + deltas |
| Feature_Importances | - | Graphique feature importance |
| Learning_Curves | - | Graphique learning curves |
| SMOTE_Comparison_ConfusionMatrix | - | Matrices de confusion comparatives |
| ROC_Curves_Comparison | - | Courbes ROC sans/avec SMOTE |
| Threshold_Optimization | - | Seuil optimal, metriques recalculees, graphique comparatif |

### Metriques tracees par run

| Metrique | Description |
|---|---|
| `best_cv_recall` | Meilleur recall en validation croisee (5 folds) |
| `train_accuracy`, `train_recall`, `train_f1` | Performance sur le jeu d'entrainement |
| `test_accuracy`, `test_recall`, `test_f1`, `test_precision` | Performance sur le jeu de test |
| `test_auc_roc` | Aire sous la courbe ROC |
| `gap_accuracy`, `gap_recall` | Ecarts train/test (indicateurs d'overfitting) |
| `delta_*` (runs SMOTE) | Ecart de performance avec/sans SMOTE |
| `optimal_threshold` | Seuil de classification optimal (maximise le F1) |
| `threshold_recall`, `threshold_precision`, `threshold_f1` | Metriques recalculees au seuil optimal |

### Visualisation des resultats

```bash
cd ProjetMLOPS
mlflow ui
```

Ouvrir **http://localhost:5000** dans le navigateur.

Pour plus de details, consulter le fichier `GUIDE_MLFLOW.md`.

---

## Optimisation du Seuil de Classification

Par defaut, `predict()` utilise un seuil de 0.5 : si la probabilite predite de defaut est >= 0.5, le modele predit un defaut. Pour un dataset desequilibre (18.5% de defauts), ce seuil n'est pas optimal.

La derniere cellule du notebook realise :

1. **Courbes Precision-Recall vs Seuil** pour chaque modele (sans et avec SMOTE), permettant de visualiser comment precision et recall evoluent quand on fait varier le seuil de 0 a 1
2. **Courbes Precision vs Recall** avec une etoile au point ou le F1-score est maximal
3. **Tableau recapitulatif** du seuil optimal, de la precision, du recall et du F1 pour chaque modele
4. **Comparaison visuelle** : matrice de confusion avec seuil 0.5 vs seuil optimal sur le meilleur modele

Un seuil plus bas que 0.5 augmente le recall (davantage de defauts detectes) au prix d'une baisse de precision (plus de faux positifs). Le seuil optimal F1 offre le meilleur compromis entre les deux.

---

## Problemes Identifies et Corrections

### Data Leakage
- `credit_lines_outstanding` encodait quasi-directement la cible (modalite 5 = 99.8% de defaut)
- **Correction** : variable supprimee des features

### Scaling Premature
- Le code original appliquait `StandardScaler` sur tout le dataset (y compris la cible) avant le split
- **Correction** : scaling integre dans le Pipeline (applique uniquement sur le train a chaque fold CV)

### Desequilibre des Classes
- 81.5% non-defaut vs 18.5% defaut
- **Correction** : SMOTE sur le train set + class_weight="balanced" + scale_pos_weight

### Donnees Synthetiques Irrealistes
- Le code original utilisait `make_classification` (donnees aleatoires sans lien avec les features reelles)
- **Correction** : remplacement par SMOTE qui genere des echantillons par interpolation entre vrais voisins

### Compatibilite pandas 2.x / joblib
- Depuis pandas 2.x, l'option `future.infer_string` peut activer le type `StringDtype` pour les noms de colonnes, incompatible avec la serialisation joblib utilisee par `GridSearchCV(n_jobs=-1)`
- **Correction** : desactivation de `future.infer_string` + conversion des DataFrames en arrays numpy avant l'entrainement

---

## Prerequis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn statsmodels scipy mlflow
```

| Package | Version minimale | Usage |
|---|---|---|
| pandas | >= 2.0 | Manipulation de donnees |
| numpy | >= 1.23 | Calcul numerique |
| matplotlib | >= 3.6 | Visualisation |
| seaborn | >= 0.12 | Visualisation statistique |
| scikit-learn | >= 1.2 | Modeles, Pipeline, GridSearchCV, metriques |
| xgboost | >= 1.7 | Gradient boosting |
| imbalanced-learn | >= 0.10 | SMOTE |
| statsmodels | >= 0.13 | VIF, tests statistiques |
| mlflow | >= 2.0 | Tracking des experiences |

---

## Execution

### Notebook sans MLflow
1. Ouvrir `projetmlops.ipynb` dans Jupyter Notebook ou JupyterLab
2. Executer toutes les cellules sequentiellement (Kernel > Restart & Run All)
3. Les resultats s'affichent dans le notebook

### Notebook avec MLflow
1. Ouvrir `projetmlops_MLF0.ipynb` dans Jupyter Notebook ou JupyterLab
2. Executer toutes les cellules sequentiellement
3. Les resultats s'affichent dans le notebook ET sont enregistres dans `mlruns/`
4. Lancer `mlflow ui` dans le terminal pour visualiser et comparer les runs

---

## Auteur

Projet realise dans le cadre du DU Data Analytics - PS1 (MLOps).
