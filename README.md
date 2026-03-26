# Projet MLOps : Prediction de Defaut de Credit

Projet de Machine Learning supervisee visant a predire le defaut de paiement d'un emprunteur (variable binaire `default`) a partir de ses caracteristiques financieres.

Realise dans le cadre du **DU Data Analytics - PS1**.

---

## Objectif

Construire et comparer plusieurs modeles de classification pour identifier les emprunteurs a risque de defaut, en respectant les bonnes pratiques MLOps : reproductibilite, absence de data leakage, validation rigoureuse et interpretabilite.

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
|-- Loan_Data.csv                      # Jeu de donnees
|-- projetmlops.ipynb                  # Notebook principal
|-- Projet_MLOPS.docx                  # Cahier des charges / enonce
|-- Guide_Explicatif_Notebook.docx     # Explication detaillee de chaque cellule
|-- Schema_Traitements_Pipeline.docx   # Schemas du pipeline, GridSearchCV, SMOTE
|-- Guide_Pedagogique_Complet.docx     # Guide complet pour debutants en Python/ML
|-- mlflow.db                          # Base MLflow (tracking des experiences)
```

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
 Pipeline(StandardScaler + Modele) x GridSearchCV(5 folds)
   -> Logistic Regression (L1)
   -> Random Forest
   -> XGBoost
   -> MLP (reseau de neurones)
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
 Courbes ROC comparatives
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

---

## Prerequis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn statsmodels scipy
```

| Package | Version minimale | Usage |
|---|---|---|
| pandas | >= 1.5 | Manipulation de donnees |
| numpy | >= 1.23 | Calcul numerique |
| matplotlib | >= 3.6 | Visualisation |
| seaborn | >= 0.12 | Visualisation statistique |
| scikit-learn | >= 1.2 | Modeles, Pipeline, GridSearchCV, metriques |
| xgboost | >= 1.7 | Gradient boosting |
| imbalanced-learn | >= 0.10 | SMOTE |
| statsmodels | >= 0.13 | VIF, tests statistiques |

---

## Execution

1. Ouvrir `projetmlops.ipynb` dans Jupyter Notebook ou JupyterLab
2. Executer toutes les cellules sequentiellement (Kernel > Restart & Run All)
3. Les resultats (metriques, graphiques, comparaisons) s'affichent dans le notebook

---


---

## Auteur

Projet realise dans le cadre du DU Data Analytics - PS1 (MLOps).
