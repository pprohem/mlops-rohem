from kfp.v2 import dsl
from google.cloud import aiplatform

@dsl.pipeline(name="cd4ml-train-pipeline")
def pipeline(project_id: str, region: str, image: str):

    aiplatform.CustomContainerTrainingJobRunOp(
        display_name="train-model",
        container_uri=image,
        replica_count=1,
        machine_type="n1-standard-2",
        environment_variables={
            "PROJECT_ID": project_id,
            "DATASET": "mlops_trial",
            "TABLE": "casas_rw",
            "MODEL_DIR": "/gcs/models"
        }
    )
