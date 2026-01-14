from google.cloud import bigquery
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

PROJECT_ID = os.environ["PROJECT_ID"]
DATASET = os.environ["DATASET"]
TABLE = os.environ["TABLE"]
MODEL_DIR = os.environ["MODEL_DIR"]

client = bigquery.Client(project=PROJECT_ID)

df = client.query(
    f"SELECT * FROM `{PROJECT_ID}.{DATASET}.{TABLE}`"
).to_dataframe()

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = model.score(X_test, y_test)

print(f"MAE={mae}")
print(f"R2={r2}")

joblib.dump(model, f"{MODEL_DIR}/model.joblib")
