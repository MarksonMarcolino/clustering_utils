import pandas as pd
import time

from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch,
    OPTICS, MeanShift
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: for HDBSCAN
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

def run_full_benchmark(
    X,
    algorithms=None,
    cluster_range=range(2, 10),
    spectral_affinities=None,
    dbscan_eps_values=None,
    dbscan_min_samples_values=None,
    hdbscan_min_cluster_sizes=None,
    verbose=True,
    n_jobs=None,
    top_n=5,
    return_best=False
):
    """
    Executes a full clustering benchmark pipeline using multiple algorithms and parameter configurations.

    This function evaluates clustering performance (using silhouette score) across different algorithms
    and hyperparameter combinations. It returns the top N performing models and, optionally, refits
    and returns the best-performing model and its cluster labels.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input data to be clustered.

    algorithms : list of str, optional
        List of clustering algorithms to include. If None, uses all available:
        ["KMeans", "Agglomerative", "Spectral", "Birch", "DBSCAN", "OPTICS", "MeanShift", "GMM", "HDBSCAN"]

    cluster_range : range, optional
        Range of cluster values to try for algorithms that support a `n_clusters` parameter.

    spectral_affinities : list of str, optional
        List of affinities to test for SpectralClustering (e.g., ["rbf", "nearest_neighbors"]).

    dbscan_eps_values : list of float, optional
        Epsilon values to evaluate for DBSCAN.

    dbscan_min_samples_values : list of int, optional
        Values for the `min_samples` parameter in DBSCAN. Default is [5].

    hdbscan_min_cluster_sizes : list of int, optional
        Values to test for the `min_cluster_size` parameter in HDBSCAN.

    verbose : bool, default=True
        If True, prints progress and timing information during execution.

    n_jobs : int or None, optional
        Number of parallel threads to use. If None, uses all available cores.

    top_n : int, default=5
        Number of top-performing clustering results to return.

    return_best : bool, default=False
        If True, returns the best model and its predicted labels in addition to the results DataFrame.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame of clustering results with columns: 
        ["Algorithm", "Params", "SilhouetteScore", "NumClusters", "ExecutionTime"]

    tuple, optional
        If `return_best=True`, also returns (best_model, best_labels)
    """

    if algorithms is None:
        algorithms = ["KMeans", "Agglomerative", "Spectral", "Birch", "DBSCAN", "OPTICS", "MeanShift", "GMM", "HDBSCAN"]
    if spectral_affinities is None:
        spectral_affinities = ["rbf"]
    if dbscan_eps_values is None:
        dbscan_eps_values = [0.3, 0.5, 0.7, 1.0]
    if dbscan_min_samples_values is None:
        dbscan_min_samples_values = [5]
    if hdbscan_min_cluster_sizes is None:
        hdbscan_min_cluster_sizes = [5, 10]

    df_results = benchmark_clustering_algorithms(
        X=X,
        algorithms=algorithms,
        cluster_range=cluster_range,
        spectral_affinities=spectral_affinities,
        dbscan_eps_values=dbscan_eps_values,
        dbscan_min_samples_values=dbscan_min_samples_values,
        hdbscan_min_cluster_sizes=hdbscan_min_cluster_sizes,
        verbose=verbose,
        n_jobs=n_jobs
    )

    top_results = df_results.head(top_n)

    if not return_best:
        return top_results

    # Get best configuration and re-fit it
    best_algo, best_params = top_results.iloc[0][["Algorithm", "Params"]]

    # Rebuild best model
    search_space = build_search_space(
        algorithms=[best_algo],
        cluster_range=cluster_range,
        spectral_affinities=spectral_affinities,
        dbscan_eps_values=dbscan_eps_values,
        dbscan_min_samples_values=dbscan_min_samples_values,
        hdbscan_min_cluster_sizes=hdbscan_min_cluster_sizes
    )
    for name, model, params in search_space:
        if name == best_algo and params == best_params:
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(X)
            else:
                model.fit(X)
                labels = model.predict(X)
            return top_results, model, labels

    return top_results

def build_search_space(
    algorithms,
    cluster_range,
    spectral_affinities,
    dbscan_eps_values,
    dbscan_min_samples_values,
    hdbscan_min_cluster_sizes
):
    """
    Constructs the search space for clustering benchmarking.

    This function generates a list of algorithm configurations, each represented as a tuple
    of (algorithm_name, model_instance, parameter_dict). It dynamically configures clustering
    models such as KMeans, Spectral, DBSCAN, etc., based on the specified parameter ranges.

    Parameters
    ----------
    algorithms : list of str
        List of algorithm names to include.
        Supported values: "KMeans", "Agglomerative", "Spectral", "Birch",
        "DBSCAN", "OPTICS", "MeanShift", "GMM", "HDBSCAN"

    cluster_range : range
        Range of cluster counts for algorithms that accept a `n_clusters` or similar parameter.

    spectral_affinities : list of str
        Affinity strategies for SpectralClustering (e.g., "rbf", "nearest_neighbors").

    dbscan_eps_values : list of float
        Values of epsilon (`eps`) to use when configuring DBSCAN.

    dbscan_min_samples_values : list of int
        Values for the `min_samples` parameter in DBSCAN.

    hdbscan_min_cluster_sizes : list of int
        Values for `min_cluster_size` used when configuring HDBSCAN.

    Returns
    -------
    list of tuple
        A list of (algorithm_name, model_instance, parameter_dict) tuples,
        representing the full search space for benchmarking.
    """
    search_space = []

    for algo in algorithms:
        if algo == "KMeans":
            for k in cluster_range:
                search_space.append((
                    "KMeans",
                    KMeans(n_clusters=k, random_state=42),
                    {"n_clusters": k}
                ))

        elif algo == "Agglomerative":
            for k in cluster_range:
                search_space.append((
                    "Agglomerative",
                    AgglomerativeClustering(n_clusters=k),
                    {"n_clusters": k}
                ))

        elif algo == "Spectral":
            for affinity in spectral_affinities:
                for k in cluster_range:
                    search_space.append((
                        "Spectral",
                        SpectralClustering(n_clusters=k, affinity=affinity, assign_labels='discretize', random_state=42),
                        {"n_clusters": k, "affinity": affinity}
                    ))

        elif algo == "Birch":
            for k in cluster_range:
                search_space.append((
                    "Birch",
                    Birch(n_clusters=k),
                    {"n_clusters": k}
                ))

        elif algo == "DBSCAN":
            for eps in dbscan_eps_values:
                for min_samples in dbscan_min_samples_values:
                    search_space.append((
                      "DBSCAN",
                        DBSCAN(eps=eps, min_samples=min_samples),
                        {"eps": eps, "min_samples": min_samples}
                ))

        elif algo == "OPTICS":
            search_space.append((
                "OPTICS",
                OPTICS(min_samples=5),
                {"min_samples": 5}
            ))

        elif algo == "MeanShift":
            search_space.append((
                "MeanShift",
                MeanShift(),
                {}
            ))

        elif algo == "GMM":
            for k in cluster_range:
                search_space.append((
                    "GMM",
                    GaussianMixture(n_components=k, random_state=42),
                    {"n_components": k}
                ))

        elif algo == "HDBSCAN" and HDBSCAN_AVAILABLE:
            for min_size in hdbscan_min_cluster_sizes:
                search_space.append((
                    "HDBSCAN",
                    HDBSCAN(min_cluster_size=min_size),
                    {"min_cluster_size": min_size}
                ))

    return search_space


def evaluate_model(X, name, model, params, verbose=True):
    """
    Fits a clustering model, evaluates its performance using silhouette score, and tracks execution time.

    This function attempts to fit the provided model on the input data using either `fit_predict()` 
    or `fit()` followed by `predict()`. It computes the silhouette score for the resulting cluster 
    labels (if more than one cluster is found), and measures how long the operation took.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The input feature matrix to cluster.

    name : str
        The name of the clustering algorithm (e.g., "KMeans", "DBSCAN", "HDBSCAN").

    model : object
        The clustering model instance. Must implement either `fit_predict()` or both `fit()` and `predict()` methods.

    params : dict
        Dictionary of parameters used to configure the model. Used for display and logging.

    verbose : bool, default=True
        If True, prints progress messages, timing, and any errors encountered.

    Returns
    -------
    tuple
        A tuple containing:
        - algorithm name (str)
        - parameters used (dict)
        - silhouette score (float or None)
        - number of clusters (int)
        - execution time in seconds (float)

        If the model fails or results in a single cluster, silhouette score is set to None or -1 respectively.
    """
    if verbose:
        print(f"Running {name} with {params}...")
    start = time.time()
    try:
        if hasattr(model, "fit_predict"):
            labels = model.fit_predict(X)
        else:
            model.fit(X)
            labels = model.predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        score = silhouette_score(X, labels) if n_clusters > 1 else -1
        duration = time.time() - start

        if verbose:
            print(f"✔ Done {name} with {params} — Score: {score:.3f}, Clusters: {n_clusters}, Time: {duration:.2f}s")

        return (name, params, score, n_clusters, duration)

    except Exception as e:
        duration = time.time() - start
        if verbose:
            print(f"Error in {name} with {params}: {e} (after {duration:.2f}s)")
        return (name, params, None, 0, duration)

def benchmark_clustering_algorithms(
    X,
    algorithms=["KMeans", "Agglomerative", "Spectral", "Birch", "DBSCAN", "OPTICS", "MeanShift", "GMM", "HDBSCAN"],
    cluster_range=range(2, 10),
    spectral_affinities=["rbf"],
    dbscan_eps_values=[0.3, 0.5, 0.7, 1.0],
    dbscan_min_samples_values=[5],
    hdbscan_min_cluster_sizes=[5, 10],
    verbose=True,
    n_jobs=None
):
    """
    Benchmarks multiple clustering algorithms and hyperparameter configurations on a given dataset.

    For each algorithm and parameter setting, this function fits the model, calculates the 
    silhouette score (when more than one cluster is detected), counts the number of clusters,
    and measures execution time. The benchmark is parallelized using multithreading.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Scaled feature matrix to cluster.

    algorithms : list of str, optional
        Clustering algorithms to include in the benchmark.
        Supported: "KMeans", "Agglomerative", "Spectral", "Birch", "DBSCAN",
                   "OPTICS", "MeanShift", "GMM", "HDBSCAN"

    cluster_range : range, optional
        Range of cluster values to test (e.g., for KMeans, Agglomerative, Spectral, etc.).

    spectral_affinities : list of str, optional
        List of affinity metrics for SpectralClustering. Default is ["rbf"].

    dbscan_eps_values : list of float, optional
        List of epsilon values to evaluate for DBSCAN.

    dbscan_min_samples_values : list of int, optional
        Values for the `min_samples` parameter in DBSCAN. Default is [5].

    hdbscan_min_cluster_sizes : list of int, optional
        List of min_cluster_size values to try for HDBSCAN.

    verbose : bool, default=True
        If True, prints progress updates and logs model execution details.

    n_jobs : int or None, optional
        Number of parallel threads to use. If None, defaults to the number of CPU cores.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing one row per configuration with the following columns:
        - "Algorithm" : str
        - "Params" : dict
        - "SilhouetteScore" : float or None
        - "NumClusters" : int
        - "ExecutionTime" : float (seconds)
    """
    search_space = build_search_space(
        algorithms=algorithms,
        cluster_range=cluster_range,
        spectral_affinities=spectral_affinities,
        dbscan_eps_values=dbscan_eps_values,
        dbscan_min_samples_values=dbscan_min_samples_values,
        hdbscan_min_cluster_sizes=hdbscan_min_cluster_sizes
    )

    results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(evaluate_model, X, name, model, params, verbose)
            for name, model, params in search_space
        ]
        for future in as_completed(futures):
            results.append(future.result())

    df_results = pd.DataFrame(results, columns=["Algorithm", "Params", "SilhouetteScore", "NumClusters", "ExecutionTime"])
    df_results = df_results.sort_values(by="SilhouetteScore", ascending=False)

    return df_results