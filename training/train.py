from google.cloud import bigquery, storage
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

# Salvar localmente primeiro
local_model_path = "/tmp/model.joblib"
local_metrics_path = "/tmp/metrics.json"

joblib.dump(model, local_model_path)

metrics = {
    "mae": mae,
    "r2": r2
}

with open(local_metrics_path, "w") as f:
    json.dump(metrics, f)

# Upload para GCS se MODEL_DIR for um caminho gs://
if MODEL_DIR.startswith("gs://"):
    # Extrair bucket e path do MODEL_DIR
    gcs_path = MODEL_DIR.replace("gs://", "")
    bucket_name = gcs_path.split("/")[0]
    blob_prefix = "/".join(gcs_path.split("/")[1:])
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Upload model
    model_blob = bucket.blob(f"{blob_prefix}/model.joblib")
    model_blob.upload_from_filename(local_model_path)
    print(f"Model uploaded to {MODEL_DIR}/model.joblib")
    
    # Upload metrics
    metrics_blob = bucket.blob(f"{blob_prefix}/metrics.json")
    metrics_blob.upload_from_filename(local_metrics_path)
    print(f"Metrics uploaded to {MODEL_DIR}/metrics.json")
else:
    # Salvar direto no diretório local
    joblib.dump(model, f"{MODEL_DIR}/model.joblib")
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"Files saved to {MODEL_DIR}")