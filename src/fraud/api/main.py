from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os
from fraud.features import ensure_dataframe
from fraud.config import MODEL_LOCAL_PATH

app = FastAPI(title="Fraud Scoring API")

class TxIn(BaseModel):
    transaction_id: str
    user_id: int
    merchant_id: int
    amount: float
    timestamp: float
    device_id: int
    channel: str
    lat: float
    lon: float
    country: str
    merchant_category: str
    entry_mode: str
    card_present: bool
    hours_since_last_txn: float
    num_txn_24h: int
    is_foreign: bool
    high_risk_country: bool

_model = None

@app.on_event("startup")
def _load():
    global _model
    path = os.getenv("MODEL_LOCAL_PATH", MODEL_LOCAL_PATH)
    _model = joblib.load(path)

@app.post("/predict")
def predict(tx: TxIn):
    df = ensure_dataframe(tx.model_dump())
    p = float(_model.predict_proba(df)[0,1])
    return {"transaction_id": tx.transaction_id, "fraud_probability": p, "is_fraud": p >= 0.5}
