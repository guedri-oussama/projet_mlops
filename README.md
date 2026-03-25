# Projet MLOps - Prediction de defaut de paiement

Projet MLOps du DU Data Analytics visant a predire le defaut de paiement d'emprunteurs a partir de donnees de credit.

## Objectif

Construire et comparer plusieurs modeles de classification binaire pour predire si un emprunteur fera defaut (`default = 1`) ou non (`default = 0`).

## Donnees

Le dataset `Loan_data.csv` contient **10 000 observations** et **8 variables** :

| Variable | Type | Description |
|---|---|---|
| `customer_id` | int | Identifiant unique du client |
| `credit_lines_outstanding` | int | Nombre de lignes de credit ouvertes (0-5) |
| `loan_amt_outstanding` | float | Montant du pret en cours |
| `total_debt_outstanding` | float | Dette totale en cours |
| `income` | float | Revenu annuel |
| `years_employed` | int | Nombre d'annees d'emploi (0-10) |
| `fico_score` | int | Score de credit FICO (408-850) |
| `default` | int | Variable cible (0 = pas de defaut, 1 = defaut) |

La variable cible est desequilibree : environ **18,5 %** de defauts.

## Structure du notebook

Le notebook `projetmlops.ipynb` suit les etapes suivantes :

1. **Chargement et exploration des donnees** - Shape, types, valeurs manquantes, statistiques descriptives
2. **Analyse exploratoire (EDA)** - Matrice de correlation, distributions, boxplots, pairplot
3. **Analyse de multicolinearite (VIF)** - Calcul du Variance Inflation Factor sur les variables explicatives
4. **Preparation des donnees** - Suppression de `customer_id`, split train/test (80/20)
5. **Modelisation avec GridSearchCV** - Comparaison de 4 modeles via pipeline (StandardScaler + modele) :
   - Regression Logistique (L1, parametre C)
   - Random Forest (n_estimators, max_depth)
   - XGBoost (n_estimators, max_depth)
   - MLP Classifier (hidden_layer_sizes)
6. **Evaluation** - Validation croisee 5-fold sur le score F1, rapport de classification, matrice de confusion

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost statsmodels scipy
```

## Utilisation

```bash
jupyter notebook projetmlops.ipynb
```

## Stack technique

- Python 3.x
- scikit-learn (pipelines, GridSearchCV, metriques)
- XGBoost
- statsmodels (VIF)
- matplotlib / seaborn (visualisations)

## Licence

Voir le fichier [LICENSE](LICENSE).
