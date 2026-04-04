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
├── main.py
├── mlflow.db
├── requirements.txt
└── README.md
```

---

##  Lancer le projet

python main.py

Le point d’entrée principal du projet est main.py.

---

##  Pipeline du projet

Le pipeline complet est orchestré dans main.py :

1. Préparation des données (data_preparation.py)
2. Entraînement des modèles (model_training.py)
3. Tracking des expériences avec MLflow (mlflow_training.py)

---

##  Préparation des données

### Gestion des valeurs manquantes

* Suppression des colonnes avec plus de 70% de valeurs manquantes
  → ces variables contiennent trop peu d’information exploitable

* Imputation par médiane
  → robuste aux valeurs extrêmes (outliers), fréquentes dans les données financières

---

### Feature Engineering

Création de variables métier :

* CREDIT_INCOME_RATIO → niveau d’endettement
* ANNUITY_INCOME_RATIO → charge mensuelle
* CREDIT_ANNUITY_RATIO → structure du prêt

Ces variables traduisent la capacité de remboursement du client.

---

### Encodage

Utilisation de One-Hot Encoding (pd.get_dummies)
→ transformation des variables catégorielles en variables numériques
→ évite d’introduire un ordre artificiel

---

##  Modélisation

### Modèles utilisés

Plusieurs modèles de classification ont été testés afin de comparer leurs performances :

* Régression logistique (LogisticRegression)
* Random Forest (RandomForestClassifier)

---

### Gestion du déséquilibre

Le dataset est fortement déséquilibré (~8% de défaut).

→ Utilisation de :

class_weight = "balanced"

---

### Validation

* Train / Test split (80 / 20) avec stratification
* Validation croisée (cross-validation)

---

##  Métriques

### Technique

* AUC (ROC-AUC)
  → adaptée aux datasets déséquilibrés

---

### Métier

Fonction de coût personnalisée :

* Faux négatif (FN) = 10
* Faux positif (FP) = 1

L’objectif est de minimiser le coût pour la banque.

---

###  Optimisation du seuil

Le seuil de décision est optimisé afin de minimiser le coût métier, car le seuil standard de 0.5 n’est pas toujours optimal.

---

##  MLflow

MLflow est utilisé pour :

* tracker les expériences
* enregistrer les métriques :

  * AUC
  * CV_AUC
  * coût métier
  * seuil optimal
* sauvegarder les modèles

Plusieurs runs ont été réalisés afin de comparer les performances des différents modèles.

---

##  Résultats

Les captures d’écran de MLflow sont disponibles dans :

output/mlflow_screenshots/

##  Visualisation MLflow

###  Suivi des modèles

<img width="1920" height="1080" alt="mlflow_runs png" src="https://github.com/user-attachments/assets/597a4ab6-b830-4df3-b1e7-337c9634d74f" />


###  Comparaison des expériences

<img width="1920" height="1080" alt="mlflow_metrics png" src="https://github.com/user-attachments/assets/44550d82-1379-44e5-ab7d-ad9dbd76fe49" />

### Performances (ordre de grandeur)

* AUC ≈ 0.74
* CV AUC ≈ 0.74

Les modèles ont été comparés en utilisant à la fois des métriques techniques et un score métier, permettant de sélectionner le modèle le plus pertinent pour le contexte bancaire.

---


##  Monitoring et stockage des données de production

Dans le cadre du déploiement du modèle, une solution de monitoring a été mise en place afin de suivre le comportement de l’API en production.

Les données générées par l’API sont stockées sous forme de logs structurés en JSON dans le fichier :

logs/api_logs.json

Chaque requête enregistrée contient :

- les données d’entrée (inputs)
- la prédiction du modèle
- la probabilité associée
- le temps d’exécution
- le statut de la requête (succès ou erreur)

Cette approche permet de conserver un historique des prédictions et d’analyser le comportement du modèle en production.

---

##  Exemple de logs

Exemple d’une entrée de log :

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

![Uploading Capture d'écran 2026-04-04 161912.png…]()

##  Auteur

Selma — Ingénieure en Intelligence Artificielle

Projet réalisé dans le cadre de la formation OpenClassrooms

