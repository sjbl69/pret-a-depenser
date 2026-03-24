import pandas as pd


# ===============================
# LOAD DATA
# ===============================
def load_data():
    print("Chargement des données...")

    df = pd.read_csv("data/application_train.csv")

    print("Shape:", df.shape)

    return df


# ===============================
# CLEAN DATA
# ===============================
def clean_data(df):

    print("Nettoyage des données...")

# Suppression des colonnes avec plus de 70% de valeurs manquantes
# Justification : ces variables contiennent trop peu d'information exploitable

    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.7].index

    df = df.drop(columns=cols_to_drop)

# Imputation par la médiane
# Justification : méthode robuste aux valeurs extrêmes (outliers)

    df = df.fillna(df.median(numeric_only=True))

    return df


# ===============================
# FEATURE ENGINEERING
# ===============================

# Feature engineering
# Ces variables représentent la capacité de remboursement du client

def feature_engineering(df):

    print("Feature engineering...")

    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    return df


# ===============================
# ENCODING
# ===============================

# Encodage des variables catégorielles
# Justification : les modèles ML nécessitent des données numériques

def encode_data(df):

    print("Encodage des variables...")

    df = pd.get_dummies(df, drop_first=True)

    return df


# ===============================
# MAIN FUNCTION
# ===============================
def prepare_dataset():

    df = load_data()

    df = clean_data(df)

    df = feature_engineering(df)

    df = encode_data(df)  # 🔥 IMPORTANT

    print("Dataset prêt !")

    return df