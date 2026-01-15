# pipelines/train_pipeline.py
from kfp import dsl
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

@dsl.pipeline(name="cd4ml-train-pipeline")
def pipeline(image: str):
    CustomTrainingJobOp(
        project="mlops-rohem",
        location="us-central1",
        display_name="train-model",
        base_output_directory="gs://mlops-cd4ml-trial/training",  # 👈 AQUI
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "e2-standard-4",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": image,
            },
        }],
    )
