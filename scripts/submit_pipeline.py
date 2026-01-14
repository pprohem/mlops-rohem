from google.cloud import aiplatform
from pipelines.train_pipeline import pipeline

aiplatform.init(
    project="mlops-rohem",
    location="us-central1"
)

aiplatform.PipelineJob(
    display_name="cd4ml-pipeline-run",
    template_path="pipeline.json",
    pipeline_root="gs://mlops-cd4ml-trial/pipelines",
    parameter_values={
        "project_id": "mlops-rohem",
        "region": "us-central1",
        "image": "IMAGE_URI"
    }
).run()
