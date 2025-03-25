import os
from datetime import datetime
from sklearn.decomposition import PCA
from datetime import datetime
import pandas as pd

import os
from datetime import datetime
from sklearn.decomposition import PCA
import pandas as pd

def export_single_cluster_report(
    X,
    model_name,
    model_params,
    output_dir="cluster_reports",
    label_colname="cluster",
    feature_names=None
):
    """
    Generates and exports a full clustering report for a single model configuration.

    This includes:
    - Cluster labels (computed using the provided model configuration)
    - Cluster counts and proportions (saved as CSV)
    - Cluster-level mean and standard deviation (saved as CSV)
    - PCA-transformed components with cluster labels (saved as CSV)
    - Centroids (if the model supports them) (saved as CSV)

    Parameters:
    ----------
    X : pd.DataFrame
        Scaled dataset to be clustered.

    model_name : str
        Name of the clustering algorithm (e.g., "KMeans", "MeanShift", etc.).

    model_params : dict
        Dictionary of parameters used to configure the model.
        Must match an entry in `build_search_space`.

    output_dir : str, optional
        Root directory where the report will be saved (default is "cluster_reports").
        A timestamped subfolder will be created inside this directory.

    label_colname : str, optional
        Name of the column to store cluster labels in exported files (default is "cluster").

    feature_names : list[str], optional
        Names of features used in the dataset (for centroid reporting).
        If None, column names from `X` will be used.

    Returns:
    -------
    None
        Saves CSV files to disk in a structured directory.
    """
    from .benchmark import build_search_space
    from .reporting import (
        export_cluster_summary,
        export_cluster_counts,
        export_pca_components,
        export_cluster_centroids
    )

    print(f"Generating report for: {model_name} with {model_params}")

    # Rebuild model
    search_space = build_search_space(
        algorithms=[model_name],
        cluster_range=range(2, 20),
        spectral_affinities=[model_params.get("affinity", "rbf")],
        dbscan_eps_values=[model_params.get("eps", 0.5)],
        hdbscan_min_cluster_sizes=[model_params.get("min_cluster_size", 5)]
    )

    model = None
    for name, m, p in search_space:
        if name == model_name and p == model_params:
            model = m
            break

    if model is None:
        print(f"⚠ Could not match model for: {model_name} {model_params}")
        return

    # Fit and predict
    if hasattr(model, "fit_predict"):
        labels = model.fit_predict(X)
    else:
        model.fit(X)
        labels = model.predict(X)

    df_with_labels = X.copy()
    df_with_labels[label_colname] = labels

    # Create folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Format model parameters into a readable string
    param_str = "_".join([f"{k}={v}" for k, v in model_params.items()]) if model_params else "default"
    subfolder_name = f"{model_name}_{param_str}_{timestamp}"
    base_path = os.path.join(output_dir, subfolder_name)
    os.makedirs(base_path, exist_ok=True)

    # Save reports
    export_cluster_counts(df_with_labels, label_colname, os.path.join(base_path, "counts.csv"))
    export_cluster_summary(df_with_labels, label_colname, os.path.join(base_path, "summary.csv"))
    export_pca_components(df_with_labels, label_colname, os.path.join(base_path, "pca.csv"))

    if feature_names is None:
        feature_names = X.columns

    export_cluster_centroids(model, feature_names, os.path.join(base_path, "centroids.csv"))

    print(f" Report saved to: {base_path}")

def export_top_cluster_reports(
    X,
    df_results,
    top_n=5,
    output_dir="cluster_reports",
    label_colname="cluster",
    feature_names=None
):
    """
    Generates and exports clustering reports for the top N results from a benchmark.

    For each of the top N models (based on silhouette score or any ordering in df_results), the function:
    - Reconstructs the model from the benchmark parameters
    - Fits it on the provided dataset X
    - Generates and exports the following reports:
        - Cluster counts and proportions
        - Mean and standard deviation of each numeric feature per cluster
        - PCA components with cluster labels
        - Centroids (if the model supports them)

    Each model's report is saved in a dedicated timestamped subfolder.

    Parameters:
    ----------
    X : pd.DataFrame
        Scaled dataset to be clustered.

    df_results : pd.DataFrame
        Result DataFrame from `benchmark_clustering_algorithms` or `run_full_benchmark`.

    top_n : int, optional (default=5)
        Number of top models to generate reports for.

    output_dir : str, optional (default="cluster_reports")
        Root directory where the reports will be saved. Each model gets its own subfolder.

    label_colname : str, optional (default="cluster")
        Name of the column to store cluster labels in exported files.

    feature_names : list[str], optional
        List of feature names to include in the centroid exports.
        If None, uses the columns of `X`.

    Returns:
    -------
    None
        Saves CSV files for each top model's clustering report into the output directory.
    """

    from .benchmark import build_search_space  # to rebuild models
    from .reporting import (
        export_cluster_summary,
        export_cluster_counts,
        export_pca_components,
        export_cluster_centroids
    )

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"top_{top_n}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(top_n):
        row = df_results.iloc[i]
        algo, params = row["Algorithm"], row["Params"]

        print(f" Generating report for Top {i+1}: {algo} with {params}")

        # Rebuild model
        search_space = build_search_space(
            algorithms=[algo],
            cluster_range=range(2, 20),  # wide enough to match anything
            spectral_affinities=[params.get("affinity", "rbf")],
            dbscan_eps_values=[params.get("eps", 0.5)],
            hdbscan_min_cluster_sizes=[params.get("min_cluster_size", 5)]
        )
        model = None
        for name, m, p in search_space:
            if name == algo and p == params:
                model = m
                break
        if model is None:
            print(f"⚠ Could not match model for: {algo} {params}")
            continue

        # Fit and get labels
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X)
        else:
            model.fit(X)
            labels = model.predict(X)

        # Prepare DataFrame with labels
        df_with_labels = X.copy()
        df_with_labels[label_colname] = labels

        # Format parameters into a readable string
        param_str = "_".join([f"{k}={v}" for k, v in params.items()]) if params else "default"
        subfolder_name = f"{i+1}_{algo}_{param_str}"
        base_path = os.path.join(output_dir, subfolder_name)
        os.makedirs(base_path, exist_ok=True)

        # Ensure feature_names is set
        if feature_names is None:
            feature_names = X.columns

        # Export all reports
        export_cluster_counts(df_with_labels, label_colname, os.path.join(base_path, "counts.csv"))
        export_cluster_summary(df_with_labels, label_colname, os.path.join(base_path, "summary.csv"))
        export_pca_components(df_with_labels, label_colname, os.path.join(base_path, "pca.csv"))
        export_cluster_centroids(model, feature_names, os.path.join(base_path, "centroids.csv"))

    print(f"\n All reports saved to: {output_dir}")

def export_cluster_summary(df, labels_column, path):
    """
    Saves mean and std of numeric features per cluster to CSV.
    """
    numeric_cols = df.select_dtypes(include="number").columns.difference([labels_column])
    grouped = df.groupby(labels_column)[numeric_cols]
    summary = pd.concat([
        grouped.mean().add_suffix("_mean"),
        grouped.std().add_suffix("_std")
    ], axis=1)
    summary.to_csv(path, index=True)
    print(f"✔ Cluster summary exported to {path}")

def export_cluster_counts(df, labels_column, path):
    """
    Saves cluster counts and proportions to CSV.
    """
    counts = df[labels_column].value_counts().sort_index()
    proportions = (counts / len(df)).round(3)
    result = pd.DataFrame({"count": counts, "proportion": proportions})
    result.to_csv(path)
    print(f"✔ Cluster counts exported to {path}")


def export_pca_components(df, labels_column, path, n_components=2):
    """
    Performs PCA on numeric features and saves components + cluster label.
    """
    features = df.select_dtypes(include="number").drop(columns=[labels_column])
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features)

    df_pca = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
    df_pca[labels_column] = df[labels_column].values
    df_pca.to_csv(path, index=False)
    print(f"✔ PCA components exported to {path}")

def export_cluster_centroids(model, feature_names, path):
    """
    Saves cluster centroids to CSV. Assumes model has `.cluster_centers_`.
    """
    if hasattr(model, "cluster_centers_"):
        df_centroids = pd.DataFrame(model.cluster_centers_, columns=feature_names)
        df_centroids.to_csv(path, index_label="cluster")
        print(f"✔ Cluster centroids exported to {path}")
    else:
        print("⚠ Model does not support cluster_centers_ — skipping centroid export.")