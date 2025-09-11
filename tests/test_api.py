import joblib
from fastapi.testclient import TestClient
from fraud.api.main import app
from fraud.data import synth_transactions
from fraud.config import MODEL_LOCAL_PATH

def test_api_predict(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier
    from fraud.features import build_pipeline
    df = synth_transactions(n=200, with_labels=True)
    X, y = df.drop(columns=["is_fraud"]), df["is_fraud"]
    pipe = build_pipeline(RandomForestClassifier(n_estimators=50, random_state=0))
    pipe.fit(X, y)
    path = tmp_path/"model.joblib"
    joblib.dump(pipe, path)
    monkeypatch.setenv("MODEL_LOCAL_PATH", str(path))

    client = TestClient(app)
    tx = synth_transactions(n=1, with_labels=False).to_dict(orient="records")[0]
    resp = client.post("/predict", json=tx)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
