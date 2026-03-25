# Projet Prêt à Dépenser – Credit Scoring

## Objectif

L’objectif de ce projet est de prédire le risque de défaut de paiement d’un client à partir de ses données financières et personnelles.

Il s’agit d’un problème de classification binaire :

* 0 → Client fiable
* 1 → Client en défaut de paiement

---

## 📁 Structure du projet

credit_scoring_project/
│
├── data/
│   ├── application_train.csv
│   ├── bureau.csv
│   └── previous_application.csv
│
├── output/
│   ├── figures/
│   └── mlflow_screenshots/
│
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── mlflow_training.py
│
├── main.py
├── requirements.txt
└── README.md

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
* LightGBM (LGBMClassifier) *(si disponible)*

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

### Performances (ordre de grandeur)

* AUC ≈ 0.74
* CV AUC ≈ 0.74

Les modèles ont été comparés en utilisant à la fois des métriques techniques et un score métier, permettant de sélectionner le modèle le plus pertinent pour le contexte bancaire.

---

##  Auteur

Selma — Ingénieure en Intelligence Artificielle

Projet réalisé dans le cadre de la formation OpenClassrooms

