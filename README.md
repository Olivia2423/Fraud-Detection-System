## Project Showcase

### Containerized services and live scoring

The Docker Compose environment runs Redpanda, MLflow, the transaction generator, model trainer, FastAPI service, and streaming consumer. The logs show the API continuously processing `POST /predict` requests.

![Docker services and live scoring](https://github.com/user-attachments/assets/ca9d70d2-edc8-418a-8194-ea10790076aa)

### Interactive scoring API

FastAPI generates interactive OpenAPI documentation for the `POST /predict` endpoint.

![FastAPI fraud scoring API](https://github.com/user-attachments/assets/e42d94cd-c088-47ae-b3e2-b4cc59268bec)

### Transaction request and prediction response

The API validates transaction features and returns the transaction identifier, fraud probability, and binary fraud classification.

#### Example request

![Fraud prediction request](https://github.com/user-attachments/assets/4890551e-21f0-4fb1-99f2-c016cb628940)

#### Example response

<!-- Drag Screenshot 2025-09-10 231840 here and delete this comment -->

### MLflow experiment tracking

Training runs are tracked in MLflow, including evaluation metrics, artifacts, source information, and the registered `fraud_model` version.

![MLflow experiment and registered model](https://github.com/user-attachments/assets/07806c9a-804d-421d-b4d5-7950d84c8a05)
