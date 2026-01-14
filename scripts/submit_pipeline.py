from google.cloud import aiplatform
from kfp.v2 import compiler

from pipelines.train_pipeline import pipeline

# ======================================================
# CONFIGURAÇÕES (bootstrap / depois vão para CI)
# ======================================================
PROJECT_ID = "mlops-rohem"
REGION = "us-central1"

# Use A IMAGEM REAL que o Cloud Build gerou
IMAGE_URI = "us-central1-docker.pkg.dev/mlops-rohem/mlops/train:6ae29a5d3b91"

PIPELINE_ROOT = "gs://mlops-cd4ml-trial/pipelines"
PIPELINE_JSON = "pipeline.json"

# ======================================================
# 1) COMPILA A PIPELINE (DSL → JSON)
# ======================================================
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=PIPELINE_JSON,
)

# ======================================================
# 2) INICIALIZA O VERTEX
# ======================================================
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)

# ======================================================
# 3) SUBMETE A PIPELINE
# ======================================================
job = aiplatform.PipelineJob(
    display_name="cd4ml-initial-pipeline",
    template_path=PIPELINE_JSON,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project_id": PROJECT_ID,
        "region": REGION,
        "image": IMAGE_URI,
    },
)

job.run()
