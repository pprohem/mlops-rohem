import argparse
import json
import os
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    args = parser.parse_args()

    model_dir = args.model_dir
    metrics_dir = args.metrics_dir

    print(f"Diretório do modelo: {model_dir}")
    print(f"Diretório de métricas (output): {metrics_dir}")

    model_path = os.path.join(model_dir, "model.joblib")
    metrics_path = os.path.join(model_dir, "metrics.json")

    # ======================================================
    # Validação mínima (sem falhar pipeline)
    # ======================================================
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo não encontrado: {model_path}")
        print("Continuando pipeline mesmo assim (modo permissivo).")
        metrics = {}
    else:
        print("Modelo encontrado com sucesso.")
        _ = joblib.load(model_path)

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            print("⚠️ Arquivo de métricas não encontrado.")
            metrics = {}

    print("Métricas observadas:")
    print(metrics)

    # ======================================================
    # Persistência para o pipeline
    # ======================================================
    os.makedirs(metrics_dir, exist_ok=True)

    output_metrics_path = os.path.join(metrics_dir, "metrics.json")

    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Métricas salvas em: {output_metrics_path}")
    print("✅ Etapa de avaliação concluída (sem gate).")


if __name__ == "__main__":
    main()
