import mlflow
import mlflow.sklearn

import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# LightGBM 

try:
    from lightgbm import LGBMClassifier
    USE_LGBM = True
except:
    USE_LGBM = False


# ===============================
# METRIQUE METIER
# ===============================
def business_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 10 * fn + 1 * fp


# ===============================
# SEUIL OPTIMAL
# ===============================
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


# ===============================
# TRAIN AVEC MLFLOW
# ===============================
def train_with_mlflow(application):

    print("Training with MLflow...")

    # ===================
    # FEATURES / TARGET
    # ===================
    X = application.drop(columns=['TARGET'])
    y = application['TARGET']

    # ===================
    # SPLIT
    # ===================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===================
    # SCALING
    # ===================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===================
    # EXPERIMENT
    # ===================
    mlflow.set_experiment("credit_scoring")

    # ===================
    # MODELES A TESTER
    # ===================
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            solver='liblinear',
            class_weight='balanced'
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
    }

    if USE_LGBM:
        models["LightGBM"] = LGBMClassifier(
            class_weight='balanced',
            random_state=42
        )

    # ===================
    # LOOP MODELES
    # ===================
    for name, model in models.items():

        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):

            # ===================
            # CROSS VALIDATION
            # ===================
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring='roc_auc'
            )

            cv_auc = cv_scores.mean()
            print(f"CV AUC: {cv_auc:.4f}")

            # ===================
            # TRAIN
            # ===================
            model.fit(X_train, y_train)

            # ===================
            # PREDICTIONS
            # ===================
            y_proba = model.predict_proba(X_test)[:, 1]

            # ===================
            # AUC
            # ===================
            auc = roc_auc_score(y_test, y_proba)
            print(f"AUC: {auc:.4f}")

            # ===================
            # METRIQUE METIER
            # ===================
            best_threshold, best_cost = find_best_threshold(y_test, y_proba)

            print(f"Best threshold: {best_threshold}")
            print(f"Business cost: {best_cost}")

            # ===================
            # LOGGING MLFLOW
            # ===================

            # PARAMS
            mlflow.log_param("model", name)

            # METRICS
            mlflow.log_metric("AUC", auc)
            mlflow.log_metric("CV_AUC", cv_auc)
            mlflow.log_metric("business_cost", best_cost)
            mlflow.log_metric("threshold", best_threshold)

            # MODELE
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} logged in MLflow ✅")

    print("\nAll models trained and logged!")