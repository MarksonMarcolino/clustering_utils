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
    """
    Plots a 2D PCA projection of high-dimensional data, colored by cluster labels.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input data to reduce using PCA.

    labels : array-like
        Cluster labels for each data point, used for coloring the points.

    save_path : str
        The file path where the PNG image will be saved.

    Returns
    -------
    None
        The plot is saved to the specified path and not displayed.
    """
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
    """
    Generates and saves t-SNE projections for multiple perplexity values.

    For each specified perplexity value, the function computes a 2D t-SNE embedding
    of the input data and saves a PNG plot colored by cluster labels.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The high-dimensional input data to reduce using t-SNE.

    labels : array-like
        Cluster labels for each data point, used to color the scatter plots.

    save_dir : str
        Directory path where each t-SNE plot image will be saved.

    perplexities : list[int], optional
        List of perplexity values to use for t-SNE (default is [30]).

    max_iter : int, optional
        Maximum number of iterations for t-SNE optimization (default is 1000).

    learning_rate : float, optional
        Learning rate for t-SNE optimization (default is 200).

    Returns
    -------
    None
        Each t-SNE projection is saved as a separate PNG file in the specified directory.
    """
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
    """
    Generates and saves a 2D UMAP projection of the input data colored by cluster labels.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The high-dimensional input data to reduce using UMAP.

    labels : array-like
        Cluster labels for each data point, used for coloring in the scatter plot.

    save_path : str
        Path where the resulting UMAP PNG image will be saved.

    Returns
    -------
    None
        The UMAP plot is saved to the specified file path.
    """
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
    """
    Plots and saves the silhouette scores for each sample in the dataset.

    The silhouette plot helps visualize how well each data point fits within its cluster.
    Higher silhouette values indicate better cluster cohesion and separation.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input data used for clustering.

    labels : array-like
        Cluster labels assigned to each sample.

    save_path : str
        File path where the silhouette plot image (PNG) will be saved.

    Returns
    -------
    None
        The silhouette plot is saved to the specified path.
    """
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
    """
    Generates and saves a heatmap of feature averages per cluster.

    The heatmap displays the mean value of each feature for each cluster, 
    helping to visualize differences in cluster centroids across variables.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input data used to compute cluster statistics.

    labels : array-like
        Cluster labels assigned to each sample.

    save_path : str
        File path where the heatmap image (PNG) will be saved.

    Returns
    -------
    None
        The heatmap is saved to the specified path.
    """
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
    """
    Creates and saves a grid of boxplots showing the distribution of each feature by cluster.

    Each subplot represents one feature, with boxplots showing how its values are distributed
    across different clusters. This helps visualize how clusters differ in terms of individual variables.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input dataset containing numeric features.

    labels : array-like
        Cluster labels assigned to each data point.

    save_path : str
        The file path where the combined boxplot image (PNG) will be saved.

    Returns
    -------
    None
        The figure is saved to the specified path as a single PNG image.
    """
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
    """
    Generates and saves an elbow plot to help determine the optimal number of clusters using KMeans.

    The elbow plot shows the inertia (sum of squared distances) for different values of K.
    The 'elbow' point on the curve typically indicates a good number of clusters.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input dataset to cluster.

    max_k : int
        The maximum number of clusters (K) to test. The function evaluates K from 1 to max_k.

    save_path : str
        File path where the elbow plot (PNG) will be saved.

    Returns
    -------
    None
        The plot is saved to the specified path.
    """
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
    """
    Generates and saves a full set of cluster visualizations for a given model and its labels.

    This function creates multiple visual representations of the clustering results, including:
    - PCA projection
    - t-SNE projections for each specified perplexity
    - UMAP projection
    - Silhouette plot
    - Cluster feature heatmap
    - Boxplot grid showing feature distributions by cluster
    - Elbow method plot (only illustrative; not used to fit model)

    All plots are saved as separate PNG files in the specified directory.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input dataset with numeric features.

    labels : array-like
        Cluster labels assigned to each sample.

    model_name : str
        Name of the clustering algorithm used (e.g., "KMeans", "DBSCAN").

    params : dict
        Dictionary of parameters used to configure the model. Used for display or filenames.

    save_dir : str
        Directory path where all plot images will be saved.

    tsne_perplexities : list[int], optional
        List of perplexity values to use for t-SNE projections (default is [30]).

    Returns
    -------
    None
        Saves multiple plot images to the specified directory.
    """
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
    """
    Generates and saves visualizations for the top N clustering results from benchmarking.

    For each of the top N models (based on silhouette score), this function:
    - Rebuilds the clustering model using its parameters
    - Fits the model to the dataset
    - Assigns cluster labels
    - Creates and saves a set of plots to visualize clustering behavior

    The following plots are generated for each top model:
    - PCA projection
    - t-SNE projections (for each perplexity value)
    - UMAP projection
    - Silhouette plot
    - Cluster heatmap
    - Feature distribution boxplots
    - Elbow method (illustrative only)

    All visualizations are saved in a subfolder named after the model and parameters
    inside the provided output directory.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The scaled dataset to be clustered and visualized.

    df_results : pd.DataFrame
        Benchmarking results DataFrame containing columns: "Algorithm", "Params", etc.

    top_n : int, optional (default=5)
        Number of top-performing clustering configurations to visualize.

    output_root : str, optional (default="visuals")
        Root directory where visualization folders for each top model will be created.

    tsne_perplexities : list[int], optional
        List of perplexity values to use for t-SNE plots (default is [30]).

    Returns
    -------
    None
        Saves all generated visualizations to the specified output directory.
    """
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
            print(f"Could not find model for {model_name} {model_params}")
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