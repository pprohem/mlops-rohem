FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia TODO o código necessário
COPY src/train/train.py ./train.py
COPY src/train/evaluate.py ./evaluate.py

