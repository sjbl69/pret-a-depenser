import pandas as pd
import os

def load_data():
    print("Chargement des données...")

    data_path = "credit_scoring_project/data/application_train.csv"

    print("Chemin utilisé :", data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Fichier introuvable : {data_path}")

    df = pd.read_csv(data_path)

    print("Shape:", df.shape)

    return df