from src.data_preparation import prepare_dataset
from src.model_training import train_model
from src.mlflow_training import train_with_mlflow

import time
import numpy as np


def simulate_inference(model, X_sample):
    """
    Simule des appels API pour mesurer la latence
    """
    latencies = []

    for i in range(100):  # simulation 100 requêtes
        start = time.time()
        model.predict(X_sample)
        end = time.time()

        latencies.append(end - start)

    return latencies


def main():
    print("===================================")
    print(" PROJET PRET A DEPENSER")
    print("===================================\n")

    # ETAPE 1 : DATA PREPARATION
    
    print("ETAPE 1 - PREPARATION DES DONNEES")

    application = prepare_dataset()

    print("\nETAPE 1 TERMINÉE\n")

    # ETAPE 2 : MODEL TRAINING
    
    print("ETAPE 2 - ENTRAINEMENT MODELE")

    model, auc, threshold, cost, cv_auc = train_model(application)

    print("\nETAPE 2 TERMINÉE\n")

    # ETAPE 3 : MLFLOW

    print("ETAPE 3 - MLFLOW")

    train_with_mlflow(application)

    print("\nETAPE 3 TERMINÉE\n")

    # ================================
    # ETAPE 4 : ANALYSE PERFORMANCE
    # ================================

    print("ETAPE 4 - ANALYSE PERFORMANCE")

    # échantillon pour simulation
    X_sample = application.drop(columns=["TARGET"]).sample(1)

    latencies = simulate_inference(model, X_sample)

    print("\n--- RESULTATS PERFORMANCE ---")
    print(f"Latence moyenne : {np.mean(latencies):.4f}s")
    print(f"Latence max : {np.max(latencies):.4f}s")
    print(f"Latence min : {np.min(latencies):.4f}s")

    print("\nETAPE 4 TERMINÉE")


if __name__ == "__main__":
    main()