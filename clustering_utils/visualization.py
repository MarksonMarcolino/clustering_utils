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
from concurrent.futures import ThreadPoolExecutor, as_completed
from .benchmark import build_search_space

def sanitize_X_for_visualization(X, verbose=False):
    """
    Attempts to convert all columns to numeric. Drops columns that cannot be converted.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe with possible mixed types.

    verbose : bool, default=False
        If True, prints which columns were dropped.

    Returns
    -------
    pd.DataFrame
        A sanitized dataframe with only numeric columns.
    """
    if not isinstance(X, pd.DataFrame):
        return X  # return as is if it's already a numpy array or not a DataFrame

    # Try to convert all columns to numeric where possible
    X_converted = X.copy()
    for col in X_converted.columns:
        X_converted[col] = pd.to_numeric(X_converted[col], errors="coerce")

    # Drop columns that are entirely NaN after conversion
    non_numeric_cols = X_converted.columns[X_converted.isna().all()].tolist()
    if non_numeric_cols and verbose:
        print(f"[Sanitize] Dropping non-numeric columns after conversion: {non_numeric_cols}")

    return X_converted.drop(columns=non_numeric_cols)

def plot_pca_projection(X, labels, save_path, verbose=False):
    """
    Plot a 2D PCA projection of high-dimensional data, colored by cluster labels.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Input data to reduce using PCA.

    labels : array-like
        Cluster labels for each data point.

    save_path : str
        File path where the PCA image will be saved.

    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    if verbose:
        print("[PCA] Generating PCA projection...")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    elif not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame or numpy ndarray.")

    if len(labels) != len(X):
        raise ValueError("Length of labels does not match number of samples in X.")

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=components[:, 0],
        y=components[:, 1],
        hue=labels,
        palette='tab10',
        s=40,
        legend="full" if len(np.unique(labels)) <= 20 else False
    )
    plt.title("PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if len(np.unique(labels)) <= 20:
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if verbose:
        print(f"[PCA] Saved to {save_path}")

def plot_tsne_projection(X, labels, save_dir, perplexities=[30], max_iter=1000, learning_rate=200, verbose=False):
    """
    Generate and save t-SNE projections for multiple perplexity values.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        High-dimensional input data.
    labels : array-like
        Cluster labels for each data point.
    save_dir : str
        Directory to save t-SNE plots.
    perplexities : list of int, optional
        List of perplexity values. Default is [30].
    max_iter : int, optional
        Maximum iterations for t-SNE. Default is 1000.
    learning_rate : float, optional
        Learning rate for t-SNE. Default is 200.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    for perp in perplexities:
        if verbose:
            print(f"[t-SNE] Generating t-SNE projection (perplexity={perp})...")

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

        if verbose:
            print(f"[t-SNE] Saved to {save_path}")

def plot_umap_projection(X, labels, save_path, verbose=False):
    """
    Generate and save a 2D UMAP projection colored by cluster labels.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Input data to reduce using UMAP.
    labels : array-like
        Cluster labels for each data point.
    save_path : str
        File path to save UMAP plot.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    if verbose:
        print("[UMAP] Generating UMAP projection...")

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

    if verbose:
        print(f"[UMAP] Saved to {save_path}")

def plot_silhouette(X, labels, save_path, verbose=False):
    """
    Generate and save a silhouette score plot for each sample.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Data used for clustering.
    labels : array-like
        Cluster labels for each sample.
    save_path : str
        File path to save the silhouette plot.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    if verbose:
        print("[Silhouette] Calculating silhouette scores...")

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

    if verbose:
        print(f"[Silhouette] Saved to {save_path}")

...

def plot_cluster_heatmap(X, labels, save_path, max_features=50, verbose=False):
    """
    Generate and save a heatmap of feature averages per cluster.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Input data to compute cluster statistics.

    labels : array-like
        Cluster labels assigned to each sample.

    save_path : str
        File path to save the heatmap image.

    max_features : int, optional
        Maximum number of features to display. Default is 50.

    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    if verbose:
        print("[Heatmap] Generating cluster heatmap...")

    df = pd.DataFrame(X)
    df["cluster"] = labels
    means = df.groupby("cluster").mean()

    if means.shape[1] > max_features:
        selected_cols = means.std().sort_values(ascending=False).head(max_features).index
        means = means[selected_cols]

    if means.empty or means.shape[1] == 0:
        if verbose:
            print("[Heatmap] Skipping: no valid features to plot.")
        return

    # Ajusta dinamicamente o tamanho da figura
    n_features = means.shape[1]
    n_clusters = means.shape[0]
    fig_width = max(12, n_features * 0.5)  # largura proporcional às features
    fig_height = max(6, n_clusters * 0.5)  # altura proporcional aos clusters

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(means, cmap="viridis", annot=True, fmt=".2f", cbar=True)
    plt.title("Cluster Mean Feature Heatmap")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if verbose:
        print(f"[Heatmap] Saved to {save_path}")

def plot_cluster_distribution(X, labels, save_path, verbose=False):
    """
    Generate and save a grid of boxplots showing the distribution of each feature by cluster.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Input dataset with numeric features.
    labels : array-like
        Cluster labels assigned to each data point.
    save_path : str
        File path where the boxplot image will be saved.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    if verbose:
        print("[Boxplot] Generating cluster distribution boxplots...")

    df = pd.DataFrame(X)
    invalid_cols = [col for col in df.columns if isinstance(df[col].iloc[0], (np.ndarray, list))]

    if invalid_cols:
        raise ValueError(f"[Boxplot] Columns {invalid_cols} contain arrays/lists instead of scalar values.")

    df["cluster"] = labels
    all_features = df.columns.drop("cluster")
    num_features = len(all_features)
    rows = (num_features // 2) + (num_features % 2)

    fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(all_features):
        try:
            sns.boxplot(x=df["cluster"], y=df[col], ax=axes[i])
            axes[i].set_title(f"Distribution of {col} by Cluster")
        except Exception as e:
            print(f"[Boxplot] Error plotting feature {col}: {e}")
            axes[i].axis('off')

    if num_features % 2 != 0:
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if verbose:
        print(f"[Boxplot] Saved to {save_path}")

def plot_elbow_method(X, max_k, save_path, verbose=False):
    """
    Generate and save an elbow plot to determine the optimal number of clusters.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Input dataset to cluster.
    max_k : int
        Maximum number of clusters to test.
    save_path : str
        File path to save the elbow plot image.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    if verbose:
        print("[Elbow] Generating elbow plot...")

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

    if verbose:
        print(f"[Elbow] Saved to {save_path}")

def plot_cluster_radar(X, labels, save_path, max_features=8, verbose=False):
    """
    Generate a radar chart to compare average cluster profiles.

    Parameters
    ----------
    X : pandas.DataFrame
        Normalized feature data (e.g., Z-score, MinMax).
    labels : array-like
        Cluster labels.
    save_path : str
        Path to save the radar chart.
    max_features : int
        Maximum number of features to display (with highest variance across clusters).
    verbose : bool
        If True, prints progress information.

    Returns
    -------
    None
    """
    if verbose:
        print("[Radar] Generating radar chart...")

    df = pd.DataFrame(X).copy()
    df["cluster"] = labels
    mean_profiles = df.groupby("cluster").mean()

    # Select features with highest variance across clusters
    top_features = mean_profiles.std().sort_values(ascending=False).head(max_features).index
    mean_profiles = mean_profiles[top_features]

    categories = list(mean_profiles.columns)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # complete the circle

    # Color palette
    colors = sns.color_palette("tab10", n_colors=len(mean_profiles))

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    for idx, (cluster, row) in enumerate(mean_profiles.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {cluster}", linewidth=2.5, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Y-axis ticks (auto-scaled)
    min_val = mean_profiles.min().min()
    max_val = mean_profiles.max().max()
    y_range = max(abs(min_val), abs(max_val))
    ax.set_ylim(-y_range * 1.1, y_range * 1.1)

    y_ticks = [-1, 0, 1] if min_val < 0 else [0, 0.5, 1.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(v) for v in y_ticks], color="gray", fontsize=9)

    # Center line at 0 for z-score clarity
    if min_val < 0:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Title and legend
    ax.set_title("Radar Chart - Cluster Profiles", size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), frameon=False)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if verbose:
        print(f"[Radar] Saved to {save_path}")

def plot_cluster_linechart(X, labels, save_path, max_features=8, verbose=False):
    """
    Generate a line chart to compare average cluster profiles by feature.

    Parameters
    ----------
    X : pandas.DataFrame
        Normalized feature data (e.g., Z-score, MinMax).
    labels : array-like
        Cluster labels.
    save_path : str
        Path to save the line chart.
    max_features : int
        Max number of features to show (based on highest std across clusters).
    verbose : bool
        If True, prints progress information.
    """
    if verbose:
        print("[Linechart] Generating line chart...")

    df = pd.DataFrame(X).copy()
    df["cluster"] = labels
    mean_profiles = df.groupby("cluster").mean()

    
    top_features = mean_profiles.std().sort_values(ascending=False).head(max_features).index
    mean_profiles = mean_profiles[top_features]

    
    plt.figure(figsize=(12, 6))
    for cluster in mean_profiles.index:
        plt.plot(top_features, mean_profiles.loc[cluster], label=f"Cluster {cluster}", linewidth=2)

    plt.title("Line Chart - Cluster Profiles", fontsize=14)
    plt.xlabel("Features")
    plt.ylabel("Mean Value (Normalized)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if verbose:
        print(f"[Linechart] Saved to {save_path}")

def plot_cluster_bar_compare(X, labels, key_var, save_path, top_n=10, verbose=False):
    """
    Plot a horizontal bar chart comparing clusters based on variables most correlated with a target variable.

    Parameters
    ----------
    X : pandas.DataFrame
        Normalized feature data.
    labels : array-like
        Cluster labels.
    key_var : str
        Name of the variable to use for correlation-based comparison.
    save_path : str
        Path to save the bar chart.
    top_n : int
        Number of most correlated variables to display.
    verbose : bool
        If True, prints progress information.

    Returns
    -------
    None
    """
    if verbose:
        print(f"[Bar] Comparing clusters based on '{key_var}' correlations...")

    df = pd.DataFrame(X).copy()
    if key_var not in df.columns:
        raise ValueError(f"'{key_var}' is not in the dataframe columns.")

    df["cluster"] = labels
    correlations = df.corr()[key_var].drop("cluster").abs().sort_values(ascending=False)
    top_features = correlations.head(top_n).index

    means = df.groupby("cluster")[top_features].mean().T

    fig, ax = plt.subplots(figsize=(12, top_n * 0.6 + 2))
    bars = means.plot(kind="barh", ax=ax, width=0.8)

    # Invert Y-axis to show the most correlated features at the top
    ax.invert_yaxis()

    # Add numerical labels to each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3)

    ax.set_title(f"Cluster Comparison Based on Top Correlated Features with '{key_var}'")
    ax.set_xlabel("Cluster Mean")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if verbose:
        print(f"[Bar] Saved to {save_path}")


def generate_all_cluster_plots(
    X, labels, model_name, params, save_dir,
    tsne_perplexities=[30], radar_max_features=8,
    bar_compare_key_var="media_freq_media_valida_pct", bar_top_n=10,
    verbose=False
):
    """
    Generate and save a complete set of visualizations for clustering results.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Dataset with numeric features.
    labels : array-like
        Cluster labels.
    model_name : str
        Name of the clustering algorithm.
    params : dict
        Parameters of the clustering model.
    save_dir : str
        Directory to save all plots.
    tsne_perplexities : list of int, optional
        Perplexity values for t-SNE. Default is [30].
    radar_max_features : int, optional
        Number of features to show in radar chart. Default is 8.
    bar_compare_key_var : str, optional
        Key variable to compute correlation for bar plot. Default is "TP_LOCALIZACAO".
    bar_top_n : int, optional
        Top correlated features to show in bar plot. Default is 10.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    os.makedirs(save_dir, exist_ok=True)
    X = sanitize_X_for_visualization(X, verbose=verbose)

    if verbose:
        print(f"[Viz] Generating plots for {model_name} with params {params} -> {save_dir}")
        print("[Viz] X shape:", X.shape)
        print("[Viz] Labels shape:", np.array(labels).shape)

    def threaded_plot(func, *args):
        try:
            func(*args, verbose=verbose)
        except Exception as e:
            print(f"[Viz] Error in {func.__name__}: {e}")

    # Standard visualizations
    plot_pca_projection(X, labels, os.path.join(save_dir, "pca_projection.png"), verbose=verbose)
    plot_silhouette(X, labels, os.path.join(save_dir, "silhouette_plot.png"), verbose=verbose)
    plot_cluster_heatmap(X, labels, os.path.join(save_dir, "cluster_heatmap.png"), verbose=verbose)
    plot_cluster_distribution(X, labels, os.path.join(save_dir, "cluster_distribution.png"), verbose=verbose)
    if model_name == "KMeans":
        plot_elbow_method(X, max_k=10, save_path=os.path.join(save_dir, "elbow_method.png"), verbose=verbose)
    plot_tsne_projection(X, labels, save_dir, perplexities=tsne_perplexities, verbose=verbose)
    plot_umap_projection(X, labels, os.path.join(save_dir, "umap_projection.png"), verbose=verbose)

    # New visualizations
    plot_cluster_radar(X, labels, os.path.join(save_dir, "radar_chart.png"), max_features=radar_max_features, verbose=verbose)
    plot_cluster_bar_compare(X, labels, key_var=bar_compare_key_var, save_path=os.path.join(save_dir, "bar_compare.png"), top_n=bar_top_n, verbose=verbose)
    plot_cluster_linechart(X, labels, os.path.join(save_dir, "line_chart.png"), max_features=radar_max_features, verbose=verbose)
def generate_top_cluster_visuals(
    X,
    df_results,
    top_n=5,
    output_root="visuals",
    tsne_perplexities=[30],
    radar_max_features=8,
    bar_compare_key_var="media_freq_media_valida_pct",
    bar_top_n=10,
    verbose=False
):
    """
    Generate and save visualizations for the top N clustering results.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        The scaled dataset to be clustered.
    df_results : pandas.DataFrame
        Benchmarking results containing "Algorithm" and "Params" columns.
    top_n : int, optional
        Number of top clustering results to visualize. Default is 5.
    output_root : str, optional
        Root directory to store output folders. Default is "visuals".
    tsne_perplexities : list of int, optional
        Perplexity values for t-SNE. Default is [30].
    radar_max_features : int, optional
        Number of features to display in radar chart. Default is 8.
    bar_compare_key_var : str, optional
        Variable name used for correlation in bar plot. Default is "TP_LOCALIZACAO".
    bar_top_n : int, optional
        Number of top correlated variables to show in bar plot. Default is 10.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    None
    """
    def process_visual(i, row):
        model_name = row["Algorithm"]
        model_params = row["Params"]
        if verbose:
            print(f"▶ Generating visualizations for Top {i+1}: {model_name} with {model_params}")

        search_space = build_search_space(
            algorithms=[model_name],
            cluster_range=range(2, 20),
            spectral_affinities=[model_params.get("affinity", "rbf")],
            dbscan_eps_values=[model_params.get("eps", 0.5)],
            dbscan_min_samples_values=[model_params.get("min_samples", 5)],
            hdbscan_min_cluster_sizes=[model_params.get("min_cluster_size", 5)]
        )
        model = next((m for name, m, p in search_space if name == model_name and p == model_params), None)

        if model is None:
            print(f"[Viz] Could not find model for {model_name} {model_params}")
            return

        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X)
        else:
            model.fit(X)
            labels = model.predict(X)

        param_str = "_".join([f"{k}={v}" for k, v in model_params.items()]) if model_params else "default"
        output_folder = os.path.join(output_root, f"{i+1}_{model_name}_{param_str}")

        generate_all_cluster_plots(
            X, labels, model_name, model_params, output_folder,
            tsne_perplexities=tsne_perplexities,
            radar_max_features=radar_max_features,
            bar_compare_key_var=bar_compare_key_var,
            bar_top_n=bar_top_n,
            verbose=verbose
        )

    for i in range(top_n):
        process_visual(i, df_results.iloc[i])