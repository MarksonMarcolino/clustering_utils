# db_toolkit/importance.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def compute_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    n_repeats: int = 10,
    random_state: int = 42,
    n_splits: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Computes permutation importance using a specified classifier and returns importance distribution.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target labels (e.g., cluster assignments).

    model : sklearn estimator, optional
        The classification model to use. Must follow the scikit-learn API (i.e., implement `fit`, `predict`, etc).
        Defaults to `RandomForestClassifier`.

        You can pass any compatible classifier, such as:

        - `RandomForestClassifier`
        - `LogisticRegression`
        - `GradientBoostingClassifier`
        - `KNeighborsClassifier`
        - `SVC`
        - `DecisionTreeClassifier`
        - `XGBClassifier` (from xgboost)
        - `CatBoostClassifier` (from catboost)
        - `LightGBM` (from lightgbm)

        For a full list of classifiers, see:
        https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

    n_repeats : int
        Number of shuffles for each feature.

    random_state : int
        Random seed.

    n_splits : int
        Number of CV folds to estimate variation in importance.

    scoring : str
        Metric used to evaluate model performance.

        You can use any classification metric supported by scikit-learn, such as:

        - 'accuracy'
        - 'balanced_accuracy'
        - 'f1'
        - 'f1_macro'
        - 'f1_micro'
        - 'f1_weighted'
        - 'precision'
        - 'precision_macro'
        - 'precision_micro'
        - 'precision_weighted'
        - 'recall'
        - 'recall_macro'
        - 'recall_micro'
        - 'recall_weighted'
        - 'roc_auc'
        - 'roc_auc_ovr'
        - 'roc_auc_ovo'

        For a full list of supported metrics, see:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    n_jobs : int
        Number of parallel jobs to run. Use -1 to use all processors.

    Returns
    -------
    pd.DataFrame
        A DataFrame with permutation importances for each feature across CV folds.
    """
    if model is None:
        model = RandomForestClassifier(random_state=random_state)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def compute_fold(fold_idx, train_idx, test_idx):
        clf = clone(model)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])

        result = permutation_importance(
            clf,
            X.iloc[test_idx],
            y.iloc[test_idx],
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=n_jobs
        )

        fold_results = []
        for i, feature in enumerate(X.columns):
            for value in result.importances[i]:
                fold_results.append({
                    "feature": feature,
                    scoring: value,
                    "fold": fold_idx
                })
        return fold_results

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_fold)(fold_idx, train_idx, test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y))
    )

    all_importances = [item for fold in results for item in fold]
    return pd.DataFrame(all_importances)

def plot_importance_boxplot(
    importances_df: pd.DataFrame,
    metric: str = "accuracy",
    top_n: int = 20,
    ascending: bool = False,
    model_name: str = None,
    save_path: str = None
):
    """
    Plots a boxplot of permutation importances for each feature.

    Parameters
    ----------
    importances_df : pd.DataFrame
        Output from `compute_permutation_importance`.

    metric : str
        Name of the metric column (used on x-axis).

    top_n : int
        Number of top features to display based on importance median.

    ascending : bool
        Whether to sort features in ascending order (False = most important first).

    model_name : str, optional
        Name of the model to display in the plot title (e.g., 'RandomForest').

    save_path : str, optional
        Path to save the plot. If None, the plot will be displayed.

    Returns
    -------
    None
    """
    median_importance = (
        importances_df.groupby("feature")[metric]
        .median()
        .sort_values(ascending=ascending)
    )

    top_features = median_importance.head(top_n).index
    filtered_df = importances_df[importances_df["feature"].isin(top_features)]

    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    sns.boxplot(
        data=filtered_df,
        x=metric,
        y="feature",
        orient="h",
        showfliers=False
    )

    model_str = f" - {model_name}" if model_name else ""
    plt.title(f"Permutation Importance{model_str} - Top {top_n} Features ({metric})")
    plt.xlabel(metric.capitalize())
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

