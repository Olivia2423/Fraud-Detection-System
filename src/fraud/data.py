from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, UTC
import numpy as np
import pandas as pd
import uuid

class Transaction(BaseModel):
    transaction_id: str
    user_id: int
    merchant_id: int
    amount: float
    timestamp: float  # epoch seconds
    device_id: int
    channel: str      # "POS","WEB","APP"
    lat: float
    lon: float
    country: str
    merchant_category: str  # "grocery","electronics","fashion","fuel","travel"
    entry_mode: str         # "chip","swipe","manual","online"
    card_present: bool
    hours_since_last_txn: float
    num_txn_24h: int
    is_foreign: bool
    high_risk_country: bool
    # for training only
    is_fraud: Optional[int] = Field(default=None)

CHANNELS = ["POS","WEB","APP"]
COUNTRIES = ["US","CA","GB","DE","FR","IN","CN","NG","BR","MX"]
MCCS = ["grocery","electronics","fashion","fuel","travel"]
ENTRY = ["chip","swipe","manual","online"]
HIGH_RISK = {"NG","RU","UA","CN"}  # simplistic
RNG = np.random.default_rng(7)

def _fraud_score_row(r):
    score = 0.0
    score += 0.7 * (r["amount"] > 300)
    score += 0.5 * (r["is_foreign"])
    score += 0.6 * (r["high_risk_country"])
    score += 0.4 * (r["entry_mode"] in ["manual","online"])
    score += 0.3 * (r["num_txn_24h"] > 10)
    score += 0.2 * (r["hours_since_last_txn"] < 0.1)
    score += 0.2 * (r["merchant_category"] in ["electronics","travel"])
    return score

def synth_transactions(n=10000, with_labels=True, start_ts=None) -> pd.DataFrame:
    now = datetime.now(UTC).timestamp() if start_ts is None else start_ts
    df = pd.DataFrame({
        "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
        "user_id": RNG.integers(1, 5000, size=n),
        "merchant_id": RNG.integers(1, 1000, size=n),
        "amount": RNG.gamma(shape=2.0, scale=60.0, size=n).clip(1, 5000),
        "timestamp": now + RNG.uniform(0, 3600*24, size=n),  # within 24h
        "device_id": RNG.integers(1, 3000, size=n),
        "channel": RNG.choice(CHANNELS, size=n, p=[0.6,0.25,0.15]),
        "lat": RNG.uniform(-60, 60, size=n),
        "lon": RNG.uniform(-120, 120, size=n),
        "country": RNG.choice(COUNTRIES, size=n, p=[.45,.25,.08,.05,.05,.04,.03,.02,.02,.01]),
        "merchant_category": RNG.choice(MCCS, size=n, p=[.35,.2,.2,.15,.1]),
        "entry_mode": RNG.choice(ENTRY, size=n, p=[.5,.25,.05,.2]),
        "card_present": RNG.choice([True, False], size=n, p=[.7,.3]),
        "hours_since_last_txn": RNG.exponential(scale=8.0, size=n).clip(0, 72),
        "num_txn_24h": RNG.poisson(3, size=n).clip(0, 40),
    })
    df["is_foreign"] = df["country"].ne("US")
    df["high_risk_country"] = df["country"].isin(HIGH_RISK)
    if with_labels:
        base = df.apply(_fraud_score_row, axis=1).values
        prob = 1 / (1 + np.exp(- (base - 1.2)))  # ~5–15% fraud rate
        df["is_fraud"] = RNG.binomial(1, prob)
    return df
