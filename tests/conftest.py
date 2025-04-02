import pytest
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use("Agg")

@pytest.fixture
def dummy_data():
    X, y = make_blobs(n_samples=50, n_features=4, centers=3, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y