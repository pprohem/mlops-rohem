from google.cloud import bigquery
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

PROJECT_ID = os.environ["PROJECT_ID"]
DATASET = os.environ["DATASET"]
TABLE = os.environ["TABLE"]
MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "/tmp")

print("MODEL_DIR =", MODEL_DIR)
print("PWD =", os.getcwd())
print("FILES IN MODEL_DIR (before):", os.listdir(os.environ.get("MODEL_DIR", "/")))

client = bigquery.Client(project=PROJECT_ID)

df = client.query(
    f"SELECT * FROM `{PROJECT_ID}.{DATASET}.{TABLE}`"
).to_dataframe()

if df.empty:
    raise ValueError("A tabela está vazia ou não existe")

# Selecionar apenas colunas numéricas
df = df.select_dtypes(include=['number'])

if "preco" not in df.columns:
    raise ValueError("Coluna 'preco' não encontrada nos dados")

X = df.drop("preco", axis=1)
y = df["preco"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = model.score(X_test, y_test)

print(f"MAE={mae}")
print(f"R2={r2}")

joblib.dump(model, f"{MODEL_DIR}/model.joblib")

metrics = {
    "mae": mae,
    "r2": r2
}

with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f)

print("FILES IN MODEL_DIR (after):", os.listdir(MODEL_DIR))