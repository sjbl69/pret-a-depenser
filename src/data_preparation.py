import pandas as pd


# ETAPE 1 - CHARGEMENT DONNEES

def load_data():
    print("Chargement des données...")

    """
    J’ai exploré les données brutes
    -> chargement du dataset
    -> vérification du nombre de lignes/colonnes
    """

    df = pd.read_csv("data/application_train.csv")

    print("Shape:", df.shape)

    return df


# ETAPE 1 - NETTOYAGE DONNEES

def clean_data(df):

    print("Nettoyage des données...")

    """
    J’ai identifié et traité les valeurs manquantes
    """

    # Suppression colonnes avec trop de valeurs manquantes
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.7].index

    df = df.drop(columns=cols_to_drop)

    """
    J’ai vérifié et supprimé les doublons
    """
    df = df.drop_duplicates()

    """
    J’ai imputé les valeurs manquantes
    (méthode robuste : médiane)
    """
    df = df.fillna(df.median(numeric_only=True))

    return df


# ETAPE 1 - FEATURE ENGINEERING

def feature_engineering(df):

    print("Feature engineering...")

    """
    J’ai construit des features pertinentes
    basées sur des ratios financiers (logique métier)
    """

    # capacité de remboursement
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # poids du crédit vs revenus
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    # charge du crédit
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    return df


# ETAPE 1 - ENCODING

def encode_data(df):

    print("Encodage des variables...")

    """
    Transformation des variables catégorielles en variables numériques
    nécessaire pour les modèles ML
    """

    df = pd.get_dummies(df, drop_first=True)

    return df


# PIPELINE COMPLET

def prepare_dataset():

    """
    Pipeline complet de préparation des données :
    - exploration
    - nettoyage
    - feature engineering
    - encodage
    """

    df = load_data()

    df = clean_data(df)

    df = feature_engineering(df)

    df = encode_data(df)

    print("Dataset prêt !")

    return df