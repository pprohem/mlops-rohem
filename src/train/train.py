import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression

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

    return pd.read_gbq(query, project_id=project)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-uri", required=True)
    args = parser.parse_args()

    print(f"Lendo dataset: {args.dataset_uri}")
    df = read_bigquery_table(args.dataset_uri)

    print(f"Linhas carregadas: {len(df)}")
    print(df.head())

    # ======= TREINO MINIMO =======
    # Ajuste os nomes conforme sua tabela real
    X = df.drop(columns=["preco"])
    y = df["preco"]

    model = LinearRegression()
    model.fit(X, y)

    print("Treino concluído com sucesso")

if __name__ == "__main__":
    main()
