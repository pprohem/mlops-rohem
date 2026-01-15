import argparse
import json
import os

import pandas as pd
import pandas_gbq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import numpy as np


def read_bigquery_table(dataset_uri: str) -> pd.DataFrame:
    """
    dataset_uri esperado:
    bq://PROJECT.DATASET.TABLE
    """
    if not dataset_uri.startswith("bq://"):
        raise ValueError("dataset_uri deve começar com bq://")

    _, ref = dataset_uri.split("bq://")
    project, dataset, table = ref.split(".")

    query = f"SELECT * FROM `{project}.{dataset}.{table}`"

    return pandas_gbq.read_gbq(query, project_id=project)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    dataset_uri = args.dataset_uri
    model_dir = args.model_dir

    print(f"Lendo dataset: {dataset_uri}")
    print(f"Diretório de saída do modelo: {model_dir}")

    # ======================================================
    # Leitura do dataset
    # ======================================================
    df = read_bigquery_table(dataset_uri)

    print(f"Linhas carregadas: {len(df)}")
    print(df.head())

    # ======================================================
    # Treino mínimo
    # ======================================================
    # Ajuste os nomes conforme sua tabela real
    if "preco" not in df.columns:
        raise ValueError("Coluna alvo 'preco' não encontrada no dataset")

    X = df.drop(columns=["preco"])
    y = df["preco"]

    model = LinearRegression()
    model.fit(X, y)

    print("Treino concluído com sucesso")

    # ======================================================
    # Avaliação simples (opcional no treino)
    # ======================================================
    y_pred = model.predict(X)

    metrics = {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "num_samples": int(len(df)),
    }

    print("Métricas de treino:")
    print(metrics)

    # ======================================================
    # Persistência dos artefatos (GCS via filesystem)
    # ======================================================
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    metrics_path = os.path.join(model_dir, "metrics.json")

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Artefatos salvos com sucesso:")
    print(f"- Modelo:   {model_path}")
    print(f"- Métricas: {metrics_path}")


if __name__ == "__main__":
    main()
