import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ======================================================
# 1. METRIQUE METIER
# ======================================================
def business_score(y_true, y_pred):
    """
    Fonction de coût métier spécifique au crédit.

    Dans un contexte bancaire :
    - Faux négatif (FN) = accepter un mauvais client → perte financière importante
    - Faux positif (FP) = refuser un bon client → manque à gagner

    On pénalise donc beaucoup plus les FN que les FP.

    Formule utilisée :
    coût = 10 * FN + 1 * FP

    Objectif : minimiser ce coût plutôt que maximiser uniquement l'AUC
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return 10 * fn + 1 * fp


# ======================================================
# 2. OPTIMISATION DU SEUIL
# ======================================================
def find_best_threshold(y_true, y_proba):
    """
    Recherche du seuil optimal de décision.

    Par défaut, les modèles utilisent un seuil de 0.5,
    mais ce seuil n'est pas forcément adapté dans un contexte métier.

    Ici, on teste plusieurs seuils entre 0.1 et 0.9 afin de :
    → minimiser le coût métier (et non seulement maximiser la performance technique)

    Cela permet d'adapter le modèle aux enjeux business.
    """
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        cost = business_score(y_true, y_pred)

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold, best_cost


# ======================================================
# 3. TRAIN MODEL
# ======================================================
def train_model(application):
    print("Training model...")

    # --------------------------------------------------
    # SEPARATION FEATURES / TARGET
    # --------------------------------------------------
    # TARGET = variable à prédire (défaut de paiement)
    # X = variables explicatives
    X = application.drop(columns=['TARGET'])
    y = application['TARGET']

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------
    # stratify=y → conserve le déséquilibre réel (~8% défaut)
    # important pour éviter un biais d’apprentissage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # SCALING
    # --------------------------------------------------
    # StandardScaler :
    # → centre les données (moyenne = 0)
    # → réduit les écarts (variance = 1)
    #
    # IMPORTANT :
    # La régression logistique est sensible à l’échelle des variables
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --------------------------------------------------
    # MODELE
    # --------------------------------------------------
    # Logistic Regression :
    # → modèle simple, interprétable et baseline solide
    #
    # class_weight='balanced' :
    # → corrige le déséquilibre de classes (91% vs 8%)
    model = LogisticRegression(
        max_iter=5000,
        solver='liblinear',
        class_weight='balanced'
    )

    # --------------------------------------------------
    # VALIDATION CROISEE
    # --------------------------------------------------
    # Permet de vérifier la robustesse du modèle
    # sur plusieurs sous-échantillons des données
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='roc_auc'
    )

    cv_auc = cv_scores.mean()
    print(f"CV AUC: {cv_auc:.4f}")

    # --------------------------------------------------
    # ENTRAINEMENT FINAL
    # --------------------------------------------------
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------
    # On récupère les probabilités de défaut
    y_proba = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # METRIQUE TECHNIQUE
    # --------------------------------------------------
    # AUC (Area Under Curve) :
    # → adaptée aux datasets déséquilibrés
    # → mesure la capacité du modèle à séparer les classes
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.4f}")

    # --------------------------------------------------
    # METRIQUE METIER
    # --------------------------------------------------
    # On optimise le seuil en fonction du coût métier
    best_threshold, best_cost = find_best_threshold(y_test, y_proba)

    print(f"Best threshold: {best_threshold}")
    print(f"Business cost: {best_cost}")

    # --------------------------------------------------
    # RETURN
    # --------------------------------------------------
    return model, auc, best_threshold, best_cost, cv_auc