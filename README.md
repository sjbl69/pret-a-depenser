# Projet Prêt à Dépenser – Credit Scoring

## Objectif

L’objectif de ce projet est de prédire le risque de défaut de paiement d’un client à partir de ses données financières et personnelles.

Il s’agit d’un problème de classification binaire :

* 0 → Client fiable
* 1 → Client en défaut de paiement

---

## 📁 Structure du projet

```bash
credit_scoring_project/
│
├── data/
│   ├── application_train.csv
│   ├── bureau.csv
│   └── previous_application.csv
│
├── mlruns/
│
├── output/
│   ├── mlflow_screenshots/
│   ├── cost_vs_threshold.png
│   └── dataset_final.csv
│
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── mlflow_training.py
│
├── analysis/
│   └── analyze_logs.py
│
├── docs/
│   └── performance_optimiziation.md
│
├── main.py
├── mlflow.db
├── requirements.txt
└── README.md
```

---

## Lancer le projet

```bash
python main.py
```

Le point d’entrée principal du projet est `main.py`.

---

## Pipeline du projet

Le pipeline complet est orchestré dans `main.py` :

1. Préparation des données (`data_preparation.py`)
2. Entraînement des modèles (`model_training.py`)
3. Tracking des expériences avec MLflow (`mlflow_training.py`)

---

## Préparation des données

### Gestion des valeurs manquantes

* Suppression des colonnes avec plus de 70% de valeurs manquantes  
* Imputation par médiane  

---

### Feature Engineering

Création de variables métier :

* CREDIT_INCOME_RATIO  
* ANNUITY_INCOME_RATIO  
* CREDIT_ANNUITY_RATIO  

---

### Encodage

Utilisation de One-Hot Encoding (`pd.get_dummies`)

---

## Modélisation

### Modèles utilisés

* Logistic Regression  
* Random Forest  

---

### Gestion du déséquilibre

```python
class_weight = "balanced"
```

---

### Validation

* Train / Test split (80 / 20)
* Cross-validation

---

## Métriques

### Technique

* AUC (ROC-AUC)

### Métier

* FN = 10  
* FP = 1  

---

## Optimisation du seuil

Le seuil de décision est optimisé afin de minimiser le coût métier.

---

## MLflow

MLflow permet de :

* tracker les expériences
* enregistrer les métriques
* sauvegarder les modèles

---

## Résultats

Captures disponibles dans :

output/mlflow_screenshots/

### Performances

* AUC ≈ 0.74  
* CV AUC ≈ 0.74  

---

## Monitoring et stockage des données de production

Les logs sont stockés en JSON dans :

logs/api_logs.json

Chaque requête contient :

- inputs
- prediction
- probability
- execution_time
- status

---

## Exemple de logs

```json
{
  "inputs": {
    "AMT_INCOME_TOTAL": 50000,
    "AMT_CREDIT": 100000,
    "DAYS_BIRTH": -12000
  },
  "prediction": 0,
  "probability": 0.23,
  "execution_time": 0.04,
  "status": "success"
}
```

---

## Script d’analyse des logs et data drift

Le script est disponible ici :

analysis/analyze_logs.py

Il permet de :

- calculer le taux d’erreur
- analyser les temps de réponse
- détecter le data drift

### Lancer l’analyse

```bash
python analysis/analyze_logs.py
```

---

## Optimisation des performances

Un rapport détaillé est disponible ici :

docs/performance_optimiziation.md

---

## Tests API

Les tests vérifient :

- cas valide → 200  
- cas invalide → 400  
- données manquantes → 400  

Cela garantit une CI fiable et un comportement correct de l’API.

---

## Auteur

Selma — Ingénieure en Intelligence Artificielle  

Projet réalisé dans le cadre de la formation OpenClassrooms
