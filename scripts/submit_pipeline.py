# scripts/submit_pipeline.py
import argparse
from google.cloud import aiplatform
from kfp import compiler
from pipelines.train_pipeline import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--image-uri", required=True)
args = parser.parse_args()

PROJECT_ID = "mlops-rohem"
REGION = "us-central1"
PIPELINE_ROOT = "gs://mlops-rohem-pipelines/pipelines"

# 1️⃣ Compila a pipeline
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="pipeline.json",
)

# 2️⃣ Inicializa o Vertex
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)

# 3️⃣ Submete a pipeline
job = aiplatform.PipelineJob(
    display_name="cd4ml-train-pipeline",
    template_path="pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "image": args.image_uri
    },
)

job.submit()
