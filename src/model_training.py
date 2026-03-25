import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



# METRIQUE METIER

def business_score(y_true, y_pred):
    # coût = 10 * FN + 1 * FP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 10 * fn + 1 * fp


# OPTIMISATION DU SEUIL

def find_best_threshold(y_true, y_proba):
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        cost = business_score(y_true, y_pred)

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold, best_cost


# OPTIMISATION LOGISTIC REGRESSION

def optimize_logistic_regression(X_train, y_train):

    print("Optimizing Logistic Regression...")

    model = LogisticRegression(
        class_weight='balanced',
        solver='liblinear'
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [1000, 5000]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid.best_estimator_


# COURBE COUT VS SEUIL

def plot_cost_vs_threshold(y_true, y_proba):

    thresholds = np.arange(0.1, 0.9, 0.05)
    costs = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cost = business_score(y_true, y_pred)
        costs.append(cost)

    plt.plot(thresholds, costs)
    plt.xlabel("Threshold")
    plt.ylabel("Business Cost")
    plt.title("Cost vs Threshold")
    plt.savefig("output/cost_vs_threshold.png")
    plt.close()


# TRAIN MODEL

def train_model(application):

    print("Training model...")

    X = application.drop(columns=['TARGET'])
    y = application['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = optimize_logistic_regression(X_train, y_train)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='roc_auc'
    )

    cv_auc = cv_scores.mean()
    print("CV AUC:", cv_auc)

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print("AUC:", auc)

    best_threshold, best_cost = find_best_threshold(y_test, y_proba)

    print("Best threshold:", best_threshold)
    print("Business cost:", best_cost)

    plot_cost_vs_threshold(y_test, y_proba)

    return model, auc, best_threshold, best_cost, cv_auc