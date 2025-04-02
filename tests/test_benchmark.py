import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from clustering_utils.benchmark import (
    build_search_space, evaluate_model, run_full_benchmark, benchmark_clustering_algorithms
)

def test_build_search_space():
    space = build_search_space(
        algorithms=["KMeans", "Agglomerative"],
        cluster_range=range(2, 5),
        spectral_affinities=["rbf"],
        dbscan_eps_values=[0.3],
        hdbscan_min_cluster_sizes=[5]
    )
    assert any(a == "KMeans" for a, _, _ in space)
    assert any(a == "Agglomerative" for a, _, _ in space)

def test_evaluate_model():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = KMeans(n_clusters=3, random_state=42)
    result = evaluate_model(X, "KMeans", model, {"n_clusters": 3})
    assert isinstance(result[2], float)
    assert isinstance(result[3], int)
    assert isinstance(result[4], float)

def test_run_full_benchmark():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    results = run_full_benchmark(X, ["KMeans", "Agglomerative"], range(2, 5), top_n=2)
    assert not results.empty

def test_benchmark_clustering_algorithms():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    results = benchmark_clustering_algorithms(X, ["KMeans", "Agglomerative"], range(2, 5))
    assert "Algorithm" in results.columns

def test_return_best():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    _, best_model, best_labels = run_full_benchmark(X, ["KMeans"], range(2, 5), top_n=1, return_best=True)
    assert best_model is not None
    assert len(best_labels) == len(X)