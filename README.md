# clustering_utils

Modular and reusable clustering benchmarking and visualization toolkit.

This in-house Python package streamlines the process of scaling, benchmarking, evaluating, and visualizing clustering results using multiple algorithms and metrics.

---

## Features

- **Flexible scaling** using StandardScaler or MinMaxScaler  
- **Benchmarking** for a variety of clustering algorithms:
  - `KMeans`, `Agglomerative`, `Spectral`, `DBSCAN`, `OPTICS`, `Birch`, `MeanShift`, `GMM`, `HDBSCAN`
- **Automated ranking** by silhouette score  
- **Exportable reports**:
  - Cluster counts & proportions
  - Feature summary (mean & std per cluster)
  - PCA component projections
  - Cluster centroids
- **Visualizations**:
  - PCA, t-SNE, UMAP projections
  - Silhouette plots
  - Cluster heatmaps, boxplots, elbow method

---

## Installation

### For local development

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
from clustering_utils.scaling import prepare_scaled_df
from clustering_utils.benchmark import run_full_benchmark
from clustering_utils.reporting import export_top_cluster_reports
from clustering_utils.visualization import generate_top_cluster_visuals

# Step 1: Scale
df_scaled = prepare_scaled_df(df, cols_to_scale, cols_to_keep, scaler_type="minmax", minmax_range=(0, 5))

# Step 2: Benchmark
results = run_full_benchmark(df_scaled, top_n=5)

# Step 3: Reports and Visualizations
export_top_cluster_reports(df_scaled, results)
generate_top_cluster_visuals(df_scaled, results)
```

---

## Project Structure

```
clustering_utils/
├── __init__.py
├── benchmark.py
├── reporting.py
├── scaling.py
├── visualization.py
docs/
├── source/
│   ├── conf.py
│   ├── index.rst
│   └── ... auto-generated .rst files
tests/
├── test_functions.py
```

---

## Documentation

After installing dependencies, build the HTML docs:

```bash
cd docs
make html
```

Then open `docs/build/html/index.html` in your browser.

---

## Dependencies

- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`
- `umap-learn`
- `hdbscan`

For development and testing:

- `pytest`
- `sphinx`

---

## Author

Markson Rebelo Marcolino