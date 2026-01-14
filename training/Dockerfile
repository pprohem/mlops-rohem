FROM python:3.10

WORKDIR /app
COPY train.py .

RUN pip install pandas scikit-learn joblib google-cloud-bigquery

ENTRYPOINT ["python", "train.py"]
