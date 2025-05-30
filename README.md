# clustering_utils

Modular and reusable clustering benchmarking and visualization toolkit.

This in-house Python package streamlines the process of scaling, benchmarking, evaluating, and visualizing clustering results using multiple algorithms and metrics.

---

## Features

- **Flexible scaling** using StandardScaler or MinMaxScaler  
- **Benchmarking** for a variety of clustering algorithms:
  - `KMeans`, `Agglomerative`, `Spectral`, `DBSCAN`, `OPTICS`, `Birch`, `MeanShift`, `GMM`, `HDBSCAN`
- **Automated ranking** by silhouette score  
- **Permutation importance** using cross-validation with `RandomForestClassifier` or any scikit-learn compatible classifier
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
from clustering_utils.importance import compute_permutation_importance, plot_importance_boxplot

# Step 1: Scale
df_scaled = prepare_scaled_df(df, cols_to_scale, cols_to_keep, scaler_type="minmax", minmax_range=(0, 5))

# Step 2: Benchmark
results = run_full_benchmark(df_scaled, top_n=5)

# Step 3: Reports and Visualizations
export_top_cluster_reports(df_scaled, results)
generate_top_cluster_visuals(df_scaled, results)

# Step 4: Permutation Importance
importances_df = compute_permutation_importance(df_scaled, y, model=RandomForestClassifier())
plot_importance_boxplot(importances_df)
```

---

## Project Structure

```
clustering_utils/
├── __init__.py
├── benchmark.py
├── importance.py
├── reporting.py
├── scaling.py
├── visualization.py
docs/
├── source/
│   ├── conf.py
│   ├── index.rst
│   └── ... auto-generated .rst files
tests/
├── test_benchmark.py
├── test_reporting.py
├── test_scaling.py
├── test_visualization.py
├── test_importance.py
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

[![scikit-learn](https://img.shields.io/pypi/v/scikit-learn.svg?label=scikit-learn&color=blue)](https://pypi.org/project/scikit-learn/)
[![pandas](https://img.shields.io/pypi/v/pandas.svg?label=pandas&color=blue)](https://pypi.org/project/pandas/)
[![matplotlib](https://img.shields.io/pypi/v/matplotlib.svg?label=matplotlib&color=blue)](https://pypi.org/project/matplotlib/)
[![seaborn](https://img.shields.io/pypi/v/seaborn.svg?label=seaborn&color=blue)](https://pypi.org/project/seaborn/)
[![umap-learn](https://img.shields.io/pypi/v/umap-learn.svg?label=umap-learn&color=blue)](https://pypi.org/project/umap-learn/)
[![hdbscan](https://img.shields.io/pypi/v/hdbscan.svg?label=hdbscan&color=blue)](https://pypi.org/project/hdbscan/)
[![joblib](https://img.shields.io/pypi/v/joblib.svg?label=joblib&color=blue)](https://pypi.org/project/joblib/)

## Development & Testing

[![pytest](https://img.shields.io/badge/tested%20with-pytest-0a0a0a?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![sphinx](https://img.shields.io/badge/docs%20with-Sphinx-blue?logo=readthedocs&logoColor=white)](https://www.sphinx-doc.org/)

---

## Author

Markson Rebelo Marcolino
