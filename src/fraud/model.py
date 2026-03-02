import mlflow
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from .features import build_pipeline
from .config import MODEL_LOCAL_PATH

def train_and_log(train_df: pd.DataFrame, mlflow_tracking_uri: str, model_name: str):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    X = train_df.drop(columns=["is_fraud"])
    y = train_df["is_fraud"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=7, n_jobs=-1, class_weight="balanced_subsample"
    )
    pipe = build_pipeline(clf)

    with mlflow.start_run(run_name="rf_baseline") as run:
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, proba)
        ap = average_precision_score(yte, proba)

        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("avg_precision", ap)
        mlflow.sklearn.log_model(pipe, "model", registered_model_name=model_name)

        joblib.dump(pipe, MODEL_LOCAL_PATH)

        info = {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model"
        }
    return info
