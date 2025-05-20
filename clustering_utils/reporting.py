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

    Parameters
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

    Returns
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
    X_raw=None,
    label_colname="cluster",
    feature_names=None,
    dbscan_min_samples_values=[5],
    hdbscan_min_cluster_sizes=[5],
    spectral_affinities=["rbf"],
    cluster_range=range(2, 20),
    is_distance_matrix=False
):
    """
    Generates and exports clustering reports for the top N results from a benchmark.

    Parameters
    ----------
    X : array-like
        Data used to refit the models (can be feature matrix or precomputed distance matrix).

    df_results : pd.DataFrame
        Output from run_full_benchmark or benchmark_clustering_algorithms.

    top_n : int, optional
        Number of top models to report on.

    output_dir : str, optional
        Root directory where reports will be saved.

    X_raw : pd.DataFrame, optional
        DataFrame with named columns for exporting summary, counts, etc.
        If None, a DataFrame will be created from X using feature_names.

    label_colname : str, optional
        Name for the column storing cluster labels.

    feature_names : list[str], optional
        Used when X is a NumPy array. If None and X_raw is given, X_raw.columns will be used.

    dbscan_min_samples_values : list[int], optional
        Values for DBSCAN re-instantiation.

    hdbscan_min_cluster_sizes : list[int], optional
        Values for HDBSCAN re-instantiation.

    spectral_affinities : list[str], optional
        Values for SpectralClustering.

    cluster_range : range, optional
        Range of cluster values for rebuild.

    is_distance_matrix : bool, optional
        Whether X is a precomputed distance matrix.
    """
    from datetime import datetime
    import os
    import pandas as pd
    from .benchmark import build_search_space
    from .reporting import (
        export_cluster_summary,
        export_cluster_counts,
        export_pca_components,
        export_cluster_centroids
    )

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
            cluster_range=cluster_range,
            spectral_affinities=spectral_affinities,
            dbscan_eps_values=[params.get("eps", 0.5)],
            dbscan_min_samples_values=[params.get("min_samples", 5)],
            hdbscan_min_cluster_sizes=[params.get("min_cluster_size", 5)],
            is_distance_matrix=is_distance_matrix
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

        # Determine data to use for reporting
        if X_raw is not None:
            df_with_labels = X_raw.copy()
        else:
            df_with_labels = pd.DataFrame(X, columns=feature_names if feature_names else None)

        df_with_labels[label_colname] = labels

        # Format parameters
        param_str = "_".join([f"{k}={v}" for k, v in params.items()]) if params else "default"
        subfolder_name = f"{i+1}_{algo}_{param_str}"
        base_path = os.path.join(output_dir, subfolder_name)
        os.makedirs(base_path, exist_ok=True)

        # Determine final feature names
        if feature_names is None:
            feature_names = df_with_labels.drop(columns=[label_colname], errors="ignore").columns

        # Export reports
        export_cluster_counts(df_with_labels, label_colname, os.path.join(base_path, "counts.csv"))
        export_cluster_summary(df_with_labels, label_colname, os.path.join(base_path, "summary.csv"))
        export_pca_components(df_with_labels, label_colname, os.path.join(base_path, "pca.csv"))
        export_cluster_centroids(model, feature_names, os.path.join(base_path, "centroids.csv"))

    print(f"\n✅ All reports saved to: {output_dir}")

def export_cluster_summary(df, labels_column, path):
    """
    Computes and exports the mean and standard deviation of numeric features for each cluster.

    The output CSV will have features as rows and clusters as columns, with suffixes
    "_mean" and "_std" to indicate the aggregated metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric features and a column with cluster labels.

    labels_column : str
        Name of the column that contains cluster labels.

    path : str
        File path where the summary CSV file will be saved.

    Returns
    -------
    None
        The summary statistics are written to the specified file path.
    """
    import pandas as pd

    numeric_cols = df.select_dtypes(include="number").columns.difference([labels_column])
    grouped = df.groupby(labels_column)[numeric_cols]

    
    means = grouped.mean().T.add_suffix("_mean") 
    stds = grouped.std().T.add_suffix("_std")

    summary = pd.concat([means, stds])
    summary.to_csv(path, index=True)
    print(f"✔ Cluster summary exported to {path}")

def export_cluster_counts(df, labels_column, path):
    """
    Computes and exports the count and proportion of samples in each cluster.

    This function calculates how many samples belong to each cluster and what 
    proportion of the total each cluster represents. The result is saved as a CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column with cluster labels.

    labels_column : str
        Name of the column containing the cluster labels.

    path : str
        File path where the counts CSV file will be saved.

    Returns
    -------
    None
        The cluster count and proportion table is saved to the specified path.
    """
    counts = df[labels_column].value_counts().sort_index()
    proportions = (counts / len(df)).round(3)
    result = pd.DataFrame({"count": counts, "proportion": proportions})
    result.to_csv(path)
    print(f"✔ Cluster counts exported to {path}")


from sklearn.decomposition import PCA
import pandas as pd

def export_pca_components(df, labels_column, path, n_components=2):
    """
    Exports the weights (loadings) of each variable in the first principal components of a PCA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric features and cluster labels.

    labels_column : str
        Name of the column that contains cluster labels.

    path : str
        File path where the PCA variable weights will be saved.

    n_components : int, optional (default=2)
        Number of principal components to compute.

    Returns
    -------
    None
        The PCA loadings table is saved to the specified path.
    """
    # Select only numeric columns and exclude the cluster label
    numeric_df = df.select_dtypes(include="number")
    if labels_column not in numeric_df.columns:
        raise ValueError(f"The label column '{labels_column}' must be numeric and present in the dataframe.")
    
    features = numeric_df.drop(columns=[labels_column])
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(features)

    # Get loadings (components_.T are the weights of original variables in PCs)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Save to CSV
    loadings.to_csv(path)
    print(f"✔ PCA variable weights exported to {path}")

def export_cluster_centroids(model, feature_names, path):
    """
    Extracts and exports cluster centroids from a clustering model to a CSV file.

    This function checks whether the given model has a `cluster_centers_` attribute.
    If available, it saves the centroids as a DataFrame using the provided feature names
    as column headers and writes the result to the specified path.

    Parameters
    ----------
    model : object
        A fitted clustering model that exposes the `cluster_centers_` attribute 
        (e.g., KMeans, Birch, or MeanShift).

    feature_names : list of str
        List of feature names to use as column headers for the centroid values.

    path : str
        File path where the centroid data will be saved as a CSV file.

    Returns
    -------
    None
        The centroids are saved to the specified file path. If the model does not
        support centroids, a warning message is printed and nothing is saved.
    """
    if hasattr(model, "cluster_centers_"):
        df_centroids = pd.DataFrame(model.cluster_centers_, columns=feature_names)
        df_centroids.to_csv(path, index_label="cluster")
        print(f"✔ Cluster centroids exported to {path}")
    else:
        print("⚠ Model does not support cluster_centers_ — skipping centroid export.")