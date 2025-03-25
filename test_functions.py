import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
from clustering_utils import prepare_scaled_df
from clustering_utils.benchmark import (
    build_search_space,
    evaluate_model,
    run_full_benchmark,
    benchmark_clustering_algorithms
)
from clustering_utils.reporting import (
    export_cluster_summary,
    export_cluster_counts,
    export_pca_components,
    export_cluster_centroids,
    export_top_cluster_reports,
    export_single_cluster_report
)
from clustering_utils.visualization import (
    plot_pca_projection,
    plot_tsne_projection,
    plot_umap_projection,
    plot_silhouette,
    plot_cluster_heatmap,
    plot_cluster_distribution,
    plot_elbow_method,
    generate_all_cluster_plots,
    generate_top_cluster_visuals
)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def test_prepare_scaled_df_minmax():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [10, 20, 30],
        "c": [100, 200, 300]
    })
    scaled = prepare_scaled_df(df, cols_to_scale=["a", "b"], cols_to_keep=["c"], scaler_type="minmax", minmax_range=(0, 1))

    assert scaled.shape == (3, 3)
    assert scaled["a"].min() == 0.0
    assert scaled["a"].max() == 1.0
    assert scaled["c"].equals(df["c"])

def test_build_search_space():
    # Example of minimal inputs
    search_space = build_search_space(
        algorithms=["KMeans", "Agglomerative"],
        cluster_range=range(2, 5),
        spectral_affinities=["rbf"],
        dbscan_eps_values=[0.3],
        hdbscan_min_cluster_sizes=[5]
    )

    assert len(search_space) > 0
    # Check for KMeans configuration
    assert any(config[0] == "KMeans" for config in search_space)
    # Check for AgglomerativeClustering configuration
    assert any(config[0] == "Agglomerative" for config in search_space)
    # Check for the parameter inclusion
    assert any(config[2] == {"n_clusters": 2} for config in search_space)

def test_evaluate_model():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = KMeans(n_clusters=3, random_state=42)
    name = "KMeans"
    params = {"n_clusters": 3}
    
    result = evaluate_model(X, name, model, params)
    
    assert len(result) == 5
    assert isinstance(result[2], float)  # Silhouette score
    assert isinstance(result[3], int)    # Number of clusters
    assert isinstance(result[4], float)  # Execution time

def test_run_full_benchmark():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    
    # Run the full benchmark
    df_results = run_full_benchmark(
        X=X,
        algorithms=["KMeans", "Agglomerative"],
        cluster_range=range(2, 5),
        top_n=2
    )
    
    # Check if DataFrame is returned and is not empty
    assert isinstance(df_results, pd.DataFrame)
    assert not df_results.empty
    assert "Algorithm" in df_results.columns
    assert "SilhouetteScore" in df_results.columns
    assert "NumClusters" in df_results.columns
    assert "ExecutionTime" in df_results.columns

def test_benchmark_clustering_algorithms():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    
    # Run benchmarking for KMeans and Agglomerative with cluster range 2 to 4
    df_results = benchmark_clustering_algorithms(
        X=X,
        algorithms=["KMeans", "Agglomerative"],
        cluster_range=range(2, 5),
        verbose=False
    )
    
    # Ensure the results contain the necessary columns
    assert "Algorithm" in df_results.columns
    assert "SilhouetteScore" in df_results.columns
    assert "NumClusters" in df_results.columns
    assert "ExecutionTime" in df_results.columns
    # Check if results are sorted by SilhouetteScore
    assert df_results["SilhouetteScore"].iloc[0] >= df_results["SilhouetteScore"].iloc[-1]

def test_return_best():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    
    # Run the full benchmark with return_best=True
    _, best_model, best_labels = run_full_benchmark(
        X=X,
        algorithms=["KMeans", "Agglomerative"],
        cluster_range=range(2, 5),
        top_n=2,
        return_best=True
    )
    
    # Ensure the best model is returned
    assert best_model is not None
    assert len(best_labels) == len(X)

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

def create_dummy_data(n_samples=50, n_features=4, centers=3):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y

def test_plot_pca_projection(tmp_path):
    X, y = create_dummy_data()
    path = tmp_path / "pca.png"
    plot_pca_projection(X, y, save_path=path)
    assert path.exists()

def test_plot_tsne_projection(tmp_path):
    X, y = create_dummy_data()
    plot_tsne_projection(X, y, save_dir=tmp_path, perplexities=[5, 10])

    for p in [5, 10]:
        filepath = tmp_path / f"tsne_projection_perp_{p}.png"
        assert filepath.exists(), f"Expected file {filepath} not found."

def test_plot_umap_projection(tmp_path):
    X, y = create_dummy_data()
    path = tmp_path / "umap.png"
    plot_umap_projection(X, y, save_path=path)
    assert path.exists()

def test_plot_silhouette(tmp_path):
    X, y = create_dummy_data()
    path = tmp_path / "silhouette.png"
    plot_silhouette(X, y, save_path=path)
    assert path.exists()

def test_plot_cluster_heatmap(tmp_path):
    X, y = create_dummy_data()
    path = tmp_path / "heatmap.png"
    plot_cluster_heatmap(X, y, save_path=path)
    assert path.exists()

def test_plot_cluster_distribution(tmp_path):
    X, y = create_dummy_data()
    path = tmp_path / "distribution.png"
    plot_cluster_distribution(X, y, save_path=path)
    assert path.exists()

def test_plot_elbow_method(tmp_path):
    X, _ = create_dummy_data()
    path = tmp_path / "elbow.png"
    plot_elbow_method(X, max_k=5, save_path=path)
    assert path.exists()

def test_generate_all_cluster_plots(tmp_path):
    X, y = create_dummy_data()
    save_dir = tmp_path / "all_plots"
    save_dir.mkdir()
    generate_all_cluster_plots(X, y, model_name="KMeans", params={"n_clusters": 3}, save_dir=save_dir, tsne_perplexities=[10])
    assert len(list(save_dir.glob("*.png"))) >= 5

def test_generate_top_cluster_visuals(tmp_path):
    from clustering_utils.benchmark import run_full_benchmark
    X, _ = create_dummy_data()
    df = pd.DataFrame(X)
    results = run_full_benchmark(df, algorithms=["KMeans"], top_n=1, verbose=False)
    generate_top_cluster_visuals(df, results, top_n=1, output_root=tmp_path)
    output_dirs = list(tmp_path.glob("1_KMeans_*"))
    assert output_dirs
    assert len(list(output_dirs[0].glob("*.png"))) >= 5