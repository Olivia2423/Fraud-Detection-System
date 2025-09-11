import os
from fraud.data import synth_transactions
from fraud.model import train_and_log
from fraud.config import MLFLOW_TRACKING_URI, MODEL_NAME

if __name__ == "__main__":
    n = int(os.getenv("TRAIN_N", "20000"))
    df = synth_transactions(n=n, with_labels=True)
    info = train_and_log(df, MLFLOW_TRACKING_URI, MODEL_NAME)
    print("Trained and logged:", info)
