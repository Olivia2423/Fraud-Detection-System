from fraud.data import synth_transactions
from fraud.features import make_preprocessor
import numpy as np

def test_preprocessor_shapes():
    df = synth_transactions(n=50, with_labels=True)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].values
    pre = make_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert np.isfinite(Xt).all()
    assert set(np.unique(y)).issubset({0,1})
