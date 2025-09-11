from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os
from fraud.features import ensure_dataframe
from fraud.config import MODEL_LOCAL_PATH

_model = None  # cached model


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


def _model_path() -> str:
    # env var set by the test; fall back to config
    return os.getenv("MODEL_LOCAL_PATH", MODEL_LOCAL_PATH)


def get_model():
    """Lazy-load the model so requests work even if startup didn’t run."""
    global _model
    if _model is None:
        path = _model_path()
        _model = joblib.load(path)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the model if possible; don't crash if MODEL_LOCAL_PATH isn't set
    try:
        get_model()
    except Exception:
        pass
    yield


app = FastAPI(title="Fraud Scoring API", lifespan=lifespan)


@app.post("/predict")
def predict(tx: TxIn):
    df = ensure_dataframe(tx.model_dump())
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")
    p = float(model.predict_proba(df)[0, 1])
    return {
        "transaction_id": tx.transaction_id,
        "fraud_probability": p,
        "is_fraud": p >= 0.5,
    }
