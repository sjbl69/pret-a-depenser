from src.data_preparation import prepare_dataset
from src.model_training import train_model
from src.mlflow_training import train_with_mlflow


def main():
    print("===================================")
    print(" PROJET PRET A DEPENSER")
    print("===================================\n")

    # =========================
    # ETAPE 1 : DATA PREPARATION
    # =========================
    print("ETAPE 1 - PREPARATION DES DONNEES")

    application = prepare_dataset()

    print("\nETAPE 1 TERMINÉE\n")

    # =========================
    # ETAPE 2 : MODEL TRAINING
    # =========================
    print("ETAPE 2 - ENTRAINEMENT MODELE")

    model, auc, threshold, cost, cv_auc = train_model(application)

    print("\nETAPE 2 TERMINÉE\n")

    # =========================
    # ETAPE 3 : MLFLOW
    # =========================
    print("ETAPE 3 - MLFLOW")

    train_with_mlflow(application)

    print("\nETAPE 3 TERMINÉE")


if __name__ == "__main__":
    main()