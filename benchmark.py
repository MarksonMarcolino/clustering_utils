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
    hdbscan_min_cluster_sizes=None,
    verbose=True,
    n_jobs=None,
    top_n=5,
    return_best=False
):
    """
    Full clustering benchmark pipeline using configured algorithms and parameters.
    Returns top N results and optionally the best model/labels.

    Parameters:
    - X: DataFrame or np.array to cluster
    - algorithms: list of algorithm names to include (optional)
    - return_best: if True, returns (df_results, best_model, best_labels)
    
    Returns:
    - DataFrame of top N results
    - (optional) best model + labels
    """

    if algorithms is None:
        algorithms = ["KMeans", "Agglomerative", "Spectral", "Birch", "DBSCAN", "OPTICS", "MeanShift", "GMM", "HDBSCAN"]
    if spectral_affinities is None:
        spectral_affinities = ["rbf"]
    if dbscan_eps_values is None:
        dbscan_eps_values = [0.3, 0.5, 0.7, 1.0]
    if hdbscan_min_cluster_sizes is None:
        hdbscan_min_cluster_sizes = [5, 10]

    df_results = benchmark_clustering_algorithms(
        X=X,
        algorithms=algorithms,
        cluster_range=cluster_range,
        spectral_affinities=spectral_affinities,
        dbscan_eps_values=dbscan_eps_values,
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
    hdbscan_min_cluster_sizes
):
    """
    Constructs a list of (name, model, params) tuples to be used in benchmarking.

    Parameters:
    ----------
    algorithms : list[str]
        List of algorithm names to include.

    cluster_range : range
        Range of cluster counts for cluster-based algorithms.

    spectral_affinities : list[str]
        Affinities to try for SpectralClustering.

    dbscan_eps_values : list[float]
        Epsilon values to test with DBSCAN.

    hdbscan_min_cluster_sizes : list[int]
        Values for min_cluster_size to test with HDBSCAN.

    Returns:
    -------
    list[tuple]
        List of (algorithm_name, model_instance, params_dict) for each config.
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
                search_space.append((
                    "DBSCAN",
                    DBSCAN(eps=eps, min_samples=5),
                    {"eps": eps}
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
    Fits a clustering model, evaluates silhouette score, and times execution.

    Parameters:
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix to fit the model on.

    name : str
        Algorithm name (e.g., "KMeans", "DBSCAN", etc.)

    model : sklearn or compatible estimator
        A model instance with `fit_predict()` or `fit()` + `predict()` methods.

    params : dict
        Parameters used to configure the model (for logging only).

    verbose : bool, optional
        If True, prints progress and results.

    Returns:
    -------
    tuple
        (AlgorithmName, Params, SilhouetteScore, NumClusters, ExecutionTime)
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
    hdbscan_min_cluster_sizes=[5, 10],
    verbose=True,
    n_jobs=None
):
    """
    Benchmarks a set of clustering algorithms across different configurations.

    For each algorithm and parameter combination, it fits the model on the provided data,
    computes silhouette score, and stores the number of clusters and execution time.

    Parameters:
    ----------
    X : pd.DataFrame or np.ndarray
        Scaled feature dataset to cluster.

    algorithms : list[str], optional
        List of algorithm names to benchmark.
        Options: "KMeans", "Agglomerative", "Spectral", "Birch", "DBSCAN", "OPTICS", "MeanShift", "GMM", "HDBSCAN"

    cluster_range : range, optional
        Range of cluster counts to test for applicable models (e.g., KMeans, Agglomerative).

    spectral_affinities : list[str], optional
        List of affinities for SpectralClustering ("rbf", "nearest_neighbors").

    dbscan_eps_values : list[float], optional
        List of epsilon values to try for DBSCAN.

    hdbscan_min_cluster_sizes : list[int], optional
        List of min_cluster_size values to try for HDBSCAN.

    verbose : bool, optional
        If True, prints progress logs during benchmarking.

    n_jobs : int or None, optional
        Number of threads to use. If None, defaults to the number of CPUs.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing:
        - Algorithm
        - Params
        - SilhouetteScore
        - NumClusters
        - ExecutionTime
    """
    search_space = build_search_space(
        algorithms=algorithms,
        cluster_range=cluster_range,
        spectral_affinities=spectral_affinities,
        dbscan_eps_values=dbscan_eps_values,
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