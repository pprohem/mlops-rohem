FROM python:3.10

WORKDIR /app
COPY train.py .

RUN pip install pandas scikit-learn joblib google-cloud-bigquery

ENV PROJECT_ID=mlops-rohem
ENV DATASET=mlops_trial
ENV TABLE=casas_rw
ENV MODEL_DIR=/gcs/mlops-rohem-pipelines/models

ENTRYPOINT ["python", "train.py"]
