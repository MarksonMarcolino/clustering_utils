import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import umap.umap_ as umap
from .benchmark import build_search_space
from concurrent.futures import ThreadPoolExecutor, as_completed


def plot_pca_projection(X, labels, save_path):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette='tab10', s=40)
    plt.title("PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if len(np.unique(labels)) > 1:
        plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_tsne_projection(X, labels, save_dir, perplexities=[30], max_iter=1000, learning_rate=200):
    for perp in perplexities:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=max_iter, learning_rate=learning_rate)
        embedding = tsne.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='tab10', s=40)
        plt.title(f"t-SNE Projection (perplexity={perp})")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        if len(np.unique(labels)) > 1:
            plt.legend(title="Cluster")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"tsne_projection_perp_{perp}.png")
        plt.savefig(save_path)
        plt.close()


def plot_umap_projection(X, labels, save_path):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
    embedding = reducer.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='tab10', s=40)
    plt.title("UMAP Projection")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    if len(np.unique(labels)) > 1:
        plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_silhouette(X, labels, save_path):
    score = silhouette_score(X, labels)
    values = silhouette_samples(X, labels)
    
    plt.figure(figsize=(8, 6))
    y_lower = 10
    for i in np.unique(labels):
        ith_cluster_silhouette_values = values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.tab10(i % 10)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=score, color="red", linestyle="--")
    plt.title(f"Silhouette plot (avg = {score:.2f})")
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cluster_heatmap(X, labels, save_path):
    df = pd.DataFrame(X)
    df["cluster"] = labels
    means = df.groupby("cluster").mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(means, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Cluster Mean Feature Heatmap")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cluster_distribution(X, labels, save_path):
    df = pd.DataFrame(X)
    df["cluster"] = labels
    all_features = df.columns.drop("cluster")
    num_features = len(all_features)
    rows = (num_features // 2) + (num_features % 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(all_features):
        sns.boxplot(x=df["cluster"], y=df[col], ax=axes[i])
        axes[i].set_title(f"Distribution of {col} by Cluster")

    if num_features % 2 != 0:
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_elbow_method(X, max_k, save_path):
    distortions = []
    K = range(1, max_k+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method for KMeans")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_all_cluster_plots(X, labels, model_name, params, save_dir, tsne_perplexities=[30]):
    os.makedirs(save_dir, exist_ok=True)

    def threaded_plot(func, *args):
        try:
            func(*args)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(threaded_plot, plot_pca_projection, X, labels, os.path.join(save_dir, "pca_projection.png"))
        executor.submit(threaded_plot, plot_silhouette, X, labels, os.path.join(save_dir, "silhouette_plot.png"))
        executor.submit(threaded_plot, plot_cluster_heatmap, X, labels, os.path.join(save_dir, "cluster_heatmap.png"))
        executor.submit(threaded_plot, plot_cluster_distribution, X, labels, os.path.join(save_dir, "cluster_distribution.png"))

    if model_name == "KMeans":
        plot_elbow_method(X, max_k=10, save_path=os.path.join(save_dir, "elbow_method.png"))

    # Run heavy plots serially
    plot_tsne_projection(X, labels, save_dir, perplexities=tsne_perplexities)
    plot_umap_projection(X, labels, os.path.join(save_dir, "umap_projection.png"))


def generate_top_cluster_visuals(X, df_results, top_n=5, output_root="visuals", tsne_perplexities=[30]):
    from .benchmark import build_search_space

    def process_visual(i, row):
        model_name = row["Algorithm"]
        model_params = row["Params"]
        print(f"▶ Generating visualizations for Top {i+1}: {model_name} with {model_params}")

        search_space = build_search_space(
            algorithms=[model_name],
            cluster_range=range(2, 20),
            spectral_affinities=[model_params.get("affinity", "rbf")],
            dbscan_eps_values=[model_params.get("eps", 0.5)],
            hdbscan_min_cluster_sizes=[model_params.get("min_cluster_size", 5)]
        )
        model = next((m for name, m, p in search_space if name == model_name and p == model_params), None)

        if model is None:
            print(f"⚠️ Could not find model for {model_name} {model_params}")
            return

        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X)
        else:
            model.fit(X)
            labels = model.predict(X)

        param_str = "_".join([f"{k}={v}" for k, v in model_params.items()]) if model_params else "default"
        output_folder = os.path.join(output_root, f"{i+1}_{model_name}_{param_str}")

        generate_all_cluster_plots(X, labels, model_name, model_params, output_folder, tsne_perplexities=tsne_perplexities)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_visual, i, df_results.iloc[i]) for i in range(top_n)]
        for future in as_completed(futures):
            future.result()