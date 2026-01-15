FROM python:3.10

WORKDIR /app
COPY train.py .

RUN pip install pandas scikit-learn joblib google-cloud-bigquery db-dtypes

ENV PROJECT_ID=mlops-rohem
ENV DATASET=mlops_trial
ENV TABLE=casas_rw

RUN mkdir -p /tmp

ENTRYPOINT ["python", "train.py"]
