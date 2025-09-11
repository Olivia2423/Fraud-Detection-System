# Real-time Fraud Detection (Python, Kafka/Redpanda, FastAPI, MLflow)

End-to-end, containerized system:

- Synthetic transaction stream → Kafka (Redpanda)
- Model training w/ MLflow tracking
- FastAPI microservice for real-time scoring
- Streaming consumer that calls the API & emits alerts
- Tests + CI (GitHub Actions)
- Docker Compose for one-command local run

## Quickstart

```bash
docker compose up --build
# MLflow UI: http://localhost:5000
# API docs:  http://localhost:8000/docs
```

To (re)train the model:
```bash
docker compose run --rm trainer
```

## Design choices
- Redpanda for local Kafka-compatible broker.
- RandomForest + sklearn pipeline; easy to serialize & serve.
- MLflow for metrics & artifacts; API reads a local artifact file.

## Testing
```bash
pytest
```

## API
- `POST /predict` → `{transaction_id, fraud_probability, is_fraud}`

## Environment variables
See `src/fraud/config.py` for defaults.
