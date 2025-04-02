import pandas as pd
import pytest
from sklearn.datasets import make_classification
from clustering_utils.importance import compute_permutation_importance, plot_importance_boxplot

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y)

def test_compute_permutation_importance_basic(classification_data):
    X, y = classification_data
    result_df = compute_permutation_importance(X, y, n_repeats=5, n_splits=3, scoring="accuracy")
    
    assert isinstance(result_df, pd.DataFrame)
    assert set(["feature", "accuracy", "fold"]).issubset(result_df.columns)
    assert result_df["feature"].nunique() == X.shape[1]

def test_compute_permutation_importance_custom_metric(classification_data):
    X, y = classification_data
    result_df = compute_permutation_importance(X, y, scoring="f1_macro")
    
    assert "f1_macro" in result_df.columns

def test_plot_importance_boxplot_creates_plot(classification_data, tmp_path):
    X, y = classification_data
    importances_df = compute_permutation_importance(X, y, n_repeats=3, n_splits=2)
    output_path = tmp_path / "importance_plot.png"

    # Should not raise error and create a file
    plot_importance_boxplot(importances_df, metric="accuracy", top_n=5, save_path=output_path)
    
    assert output_path.exists()