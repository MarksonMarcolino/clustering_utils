import pytest
from clustering_utils.visualization import *
import pandas as pd

def test_plot_pca_projection(dummy_data, tmp_path):
    X, y = dummy_data
    path = tmp_path / "pca.png"
    plot_pca_projection(X, y, save_path=path)
    assert path.exists()

def test_plot_tsne_projection(dummy_data, tmp_path):
    X, y = dummy_data
    plot_tsne_projection(X, y, save_dir=tmp_path, perplexities=[5, 10])
    for p in [5, 10]:
        filepath = tmp_path / f"tsne_projection_perp_{p}.png"
        assert filepath.exists(), f"Expected file {filepath} not found."

def test_plot_umap_projection(dummy_data, tmp_path):
    X, y = dummy_data
    path = tmp_path / "umap.png"
    plot_umap_projection(X, y, save_path=path)
    assert path.exists()

def test_plot_silhouette(dummy_data, tmp_path):
    X, y = dummy_data
    path = tmp_path / "silhouette.png"
    plot_silhouette(X, y, save_path=path)
    assert path.exists()

def test_plot_cluster_heatmap(dummy_data, tmp_path):
    X, y = dummy_data
    path = tmp_path / "heatmap.png"
    plot_cluster_heatmap(X, y, save_path=path)
    assert path.exists()

def test_plot_cluster_distribution(dummy_data, tmp_path):
    X, y = dummy_data
    path = tmp_path / "distribution.png"
    plot_cluster_distribution(X, y, save_path=path)
    assert path.exists()

def test_plot_elbow_method(dummy_data, tmp_path):
    X, _ = dummy_data
    path = tmp_path / "elbow.png"
    plot_elbow_method(X, max_k=5, save_path=path)
    assert path.exists()

def test_plot_cluster_radar(dummy_data, tmp_path):
    X, y = dummy_data
    path = tmp_path / "radar_chart.png"
    plot_cluster_radar(X, y, save_path=path)
    assert path.exists()

def test_plot_cluster_bar_compare(dummy_data, tmp_path):
    X, y = dummy_data
    df = pd.DataFrame(X)
    df.columns = [f"var_{i}" for i in range(df.shape[1])]
    path = tmp_path / "bar_compare.png"
    plot_cluster_bar_compare(df, y, key_var="var_0", save_path=path)
    assert path.exists()

def test_generate_all_cluster_plots(dummy_data, tmp_path):
    X, y = dummy_data
    save_dir = tmp_path / "all_plots"
    save_dir.mkdir()
    generate_all_cluster_plots(
        X, y,
        model_name="KMeans",
        params={"n_clusters": 3},
        save_dir=save_dir,
        tsne_perplexities=[10],
        radar_max_features=5,
        bar_compare_key_var="feature_0",
        bar_top_n=5
    )
    # At least 7 plots including the new ones
    assert len(list(save_dir.glob("*.png"))) >= 7

def test_generate_top_cluster_visuals(dummy_data, tmp_path):
    from clustering_utils.benchmark import run_full_benchmark
    X, _ = dummy_data
    df = pd.DataFrame(X)
    df.columns = [f"feature_{i}" for i in range(df.shape[1])]
    results = run_full_benchmark(df, algorithms=["KMeans"], top_n=1, verbose=False)
    generate_top_cluster_visuals(
        df,
        results,
        top_n=1,
        output_root=tmp_path,
        radar_max_features=5,
        bar_compare_key_var="feature_0",
        bar_top_n=5
    )
    output_dirs = list(tmp_path.glob("1_KMeans_*"))
    assert output_dirs
    assert len(list(output_dirs[0].glob("*.png"))) >= 7