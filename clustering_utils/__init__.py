"""
clustering_utils

This module exposes high-level functions for clustering analysis,
including scaling, benchmarking, reporting, and visualization.

Functions are organized across submodules:
- `scaling`: preprocessing and feature normalization
- `benchmark`: clustering algorithm evaluation and comparison
- `reporting`: cluster statistics and CSV export
- `visualization`: plots and projections for cluster insights
"""

from .scaling import prepare_scaled_df
from .benchmark import benchmark_clustering_algorithms, run_full_benchmark
from .visualization import (
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
from .reporting import (
    export_top_cluster_reports,
    export_single_cluster_report
)

__all__ = [
    "prepare_scaled_df",
    "benchmark_clustering_algorithms",
    "run_full_benchmark",
    "plot_pca_projection",
    "plot_tsne_projection",
    "plot_umap_projection",
    "plot_silhouette",
    "plot_cluster_heatmap",
    "plot_cluster_distribution",
    "plot_elbow_method",
    "generate_all_cluster_plots",
    "generate_top_cluster_visuals",
    "export_top_cluster_reports",
    "export_single_cluster_report"
]