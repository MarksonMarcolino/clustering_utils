# clustering_utils

Modular and reusable clustering benchmarking and visualization toolkit.

This in-house Python package is designed to streamline the process of scaling, benchmarking, evaluating, and visualizing clustering results using multiple algorithms and metrics.

---

## Features

- Standard and MinMax scaling support  
- Benchmarking for multiple clustering algorithms:
  - KMeans, Agglomerative, Spectral, DBSCAN, OPTICS, Birch, MeanShift, GMM, HDBSCAN
- Automated ranking by silhouette score  
- Export reports:
  - Cluster counts
  - Feature summary (mean, std)
  - PCA components
  - Centroids
- Visualizations:
  - PCA, t-SNE, UMAP
  - Silhouette scores
  - Feature distribution heatmaps and boxplots

---

## Installation

### Local development

```bash
git clone git@github.com:MarksonMarcolino/clustering_utils.git
cd clustering_utils
pip install -e .
```

### From GitHub (optional)

```bash
pip install git+ssh://git@github.com/MarksonMarcolino/clustering_utils.git
```

---

## Example Usage

```python
from clustering_utils import prepare_scaled_df, run_full_benchmark
from clustering_utils.reporting import export_top_cluster_reports
from clustering_utils.visualization import generate_top_cluster_visuals

# Step 1: Scale
df_scaled = prepare_scaled_df(df, cols_to_scale, cols_to_keep, scaler="minmax", minmax_range=(0, 5))

# Step 2: Benchmark
results = run_full_benchmark(df_scaled, top_n=5)

# Step 3: Generate reports and visualizations
export_top_cluster_reports(df_scaled, results)
generate_top_cluster_visuals(df_scaled, results)
```

---

## Project Structure

```
clustering_utils/
├── benchmark.py
├── reporting.py
├── scaling.py
├── visualization.py
├── __init__.py
```

---

## Dependencies

- scikit-learn
- pandas
- matplotlib
- seaborn
- umap-learn
- hdbscan