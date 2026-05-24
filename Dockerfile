FROM python:3.12-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py    .
COPY src/        ./src/
RUN pip install -e .

COPY api/        ./api/
COPY models/     ./models/
COPY run.py      .
COPY configs/    ./configs/
COPY config.yml  .
COPY locustfile.py .
COPY drift_detection.py .
COPY simulate_drift.py .
COPY logs/       ./logs/
COPY retraining/ ./app/retraining/
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
