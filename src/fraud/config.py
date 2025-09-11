import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "/app/artifacts/model.joblib")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "redpanda:9092")
KAFKA_TOPIC_TX = os.getenv("KAFKA_TOPIC_TX", "transactions")
KAFKA_TOPIC_SCORED = os.getenv("KAFKA_TOPIC_SCORED", "transactions_scored")
API_URL = os.getenv("API_URL", "http://api:8000/predict")
