import pandas as pd
import pytest
import os
from sklearn.cluster import KMeans
from clustering_utils.reporting import (
    export_cluster_summary, export_cluster_counts, export_pca_components,
    export_cluster_centroids, export_top_cluster_reports, export_single_cluster_report
)

# Create dummy dataset
def create_dummy_clustered_df():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 20, 30, 40],
        "cluster": [0, 0, 1, 1]
    })
    return df

def test_export_cluster_summary(tmp_path):
    df = create_dummy_clustered_df()
    out_file = tmp_path / "summary.csv"
    export_cluster_summary(df, "cluster", out_file)
    assert out_file.exists()
    summary = pd.read_csv(out_file)
    assert not summary.empty

def test_export_cluster_counts(tmp_path):
    df = create_dummy_clustered_df()
    out_file = tmp_path / "counts.csv"
    export_cluster_counts(df, "cluster", out_file)
    assert out_file.exists()
    counts = pd.read_csv(out_file)
    assert "count" in counts.columns

def test_export_pca_components(tmp_path):
    df = create_dummy_clustered_df()
    out_file = tmp_path / "pca.csv"
    export_pca_components(df, "cluster", out_file, n_components=2)
    assert out_file.exists()
    df_pca = pd.read_csv(out_file)
    assert "PC1" in df_pca.columns

def test_export_cluster_centroids(tmp_path):
    X = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [10, 20, 30, 40]
    })
    model = KMeans(n_clusters=2, random_state=0).fit(X)
    out_file = tmp_path / "centroids.csv"
    export_cluster_centroids(model, X.columns, out_file)
    assert out_file.exists()
    centroids = pd.read_csv(out_file)
    assert not centroids.empty

def test_export_single_cluster_report(tmp_path):
    X = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [10, 20, 30, 40]
    })
    model_name = "KMeans"
    model_params = {"n_clusters": 2}
    export_single_cluster_report(
        X=X,
        model_name=model_name,
        model_params=model_params,
        output_dir=tmp_path,
        feature_names=["x", "y"]
    )
    assert any((tmp_path / f).is_dir() for f in os.listdir(tmp_path))

def test_export_top_cluster_reports(tmp_path):
    from sklearn.datasets import make_blobs
    from clustering_utils.benchmark import run_full_benchmark

    X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
    X_df = pd.DataFrame(X, columns=["f1", "f2"])
    results = run_full_benchmark(X_df, algorithms=["KMeans"], top_n=1, verbose=False)
    
    export_top_cluster_reports(
        X=X_df,
        df_results=results,
        output_dir=tmp_path,
        top_n=1,
        feature_names=["f1", "f2"]
    )
    assert any((tmp_path / f).is_dir() for f in os.listdir(tmp_path))