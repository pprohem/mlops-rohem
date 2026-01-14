# pipelines/train_pipeline.py
from kfp import dsl
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

@dsl.pipeline(name="cd4ml-train-pipeline")
def pipeline(image: str):
    CustomTrainingJobOp(
        display_name="train-model",
        container_uri=image,
        machine_type="e2-standard-4",
        replica_count=1,
    )
