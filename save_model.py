import joblib
from src.model_training import train_model
from src.data_preparation import prepare_dataset
from sklearn.preprocessing import StandardScaler

print("Loading data...")

application = prepare_dataset()

print("Training model...")

model, auc, threshold, cost, cv_auc = train_model(application)

X = application.drop(columns=['TARGET'])

columns = X.columns.tolist()

scaler = StandardScaler()
scaler.fit(X)

joblib.dump({
    "model": model,
    "scaler": scaler,
    "threshold": threshold,
    "columns": columns   # LA CLÉ
}, "output/model.pkl")

print(" Modèle sauvegardé avec colonnes")

# sauvegarde dataset final pour analyse drift
application.to_csv("credit_scoring_project/output/dataset_final.csv", index=False)
print("Dataset sauvegardé pour drift")