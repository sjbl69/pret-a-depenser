import pandas as pd
import json
import os


# 1. Charger les logs

LOG_PATH = "logs/api_logs.json"

if not os.path.exists(LOG_PATH):
    raise FileNotFoundError(" Aucun fichier de logs trouvé")

logs = []

with open(LOG_PATH, "r") as f:
    for line in f:
        try:
            logs.append(json.loads(line))
        except:
            continue

df = pd.DataFrame(logs)

print("\n Nombre total de requêtes :", len(df))


# 2. Succès / erreurs

df_success = df[df["status"] == "success"]
df_error = df[df["status"] == "error"]

print(" Succès :", len(df_success))
print(" Erreurs :", len(df_error))

error_rate = len(df_error) / len(df)
print("Taux d'erreur :", round(error_rate, 4))


# 3. Performance API

print("\n PERFORMANCE API")

if len(df_success) > 0:
    print("Temps moyen :", round(df_success["execution_time"].mean(), 4))
    print("Temps max :", round(df_success["execution_time"].max(), 4))


# 4. Inputs

inputs_df = pd.json_normalize(df_success["inputs"])

print("\n Données production :")
print(inputs_df.head())


#  5. Dataset référence

REFERENCE_PATH = "credit_scoring_project/output/dataset_final.csv"

if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(" Dataset de référence introuvable")

reference_df = pd.read_csv(REFERENCE_PATH)


# =6. Colonnes communes

common_cols = list(set(reference_df.columns).intersection(set(inputs_df.columns)))

reference_df = reference_df[common_cols]
current_df = inputs_df[common_cols]


# 7. DETECTION DRIFT SIMPLE

print("\n DETECTION DRIFT (simple)")

drift_detected = False

for col in common_cols:
    ref_mean = reference_df[col].mean()
    prod_mean = current_df[col].mean()

    diff = abs(ref_mean - prod_mean)

    print(f"{col} → diff = {round(diff, 4)}")

    if diff > 0.1:
        print(f" Drift détecté sur {col}")
        drift_detected = True


# 8. Conclusions

print("\n ALERTES")

if error_rate > 0.1:
    print(" Taux d'erreur élevé")

if len(df_success) > 0 and df_success["execution_time"].mean() > 1:
    print(" Latence élevée")

if drift_detected:
    print(" Data drift détecté")

print("\n Analyse terminée")