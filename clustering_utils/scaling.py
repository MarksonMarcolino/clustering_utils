import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_scaled_df(
    df,
    cols_to_scale=None,
    cols_to_keep=None,
    dropna=True,
    return_scaler=False,
    verbose=True,
    scaler_type='standard',  # 'standard', 'minmax' or 'zscore'
    minmax_range=(0, 1)       # only used if scaler_type is 'minmax'
):
    """
    Scales selected columns of a DataFrame using StandardScaler, MinMaxScaler, or z-score.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features to be scaled.

    cols_to_scale : list of str, optional
        List of column names to scale. If None, all numeric columns are scaled.

    cols_to_keep : list of str, optional
        Columns to keep unscaled. These will be concatenated to the scaled output.

    dropna : bool, default=True
        Whether to drop rows with NaN values after scaling.

    return_scaler : bool, default=False
        If True, returns the fitted scaler object alongside the scaled DataFrame.

    verbose : bool, default=True
        If True, prints which columns are being scaled.

    scaler_type : {'standard', 'minmax', 'zscore'}, default='standard'
        Type of scaler to apply.

    minmax_range : tuple, default=(0, 1)
        Value range for MinMaxScaler. Ignored if `scaler_type` is not 'minmax'.

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame with scaled and kept columns.

    sklearn.preprocessing.Scaler or None
        The fitted scaler object, if `return_scaler=True`. None if 'zscore' is used.
    """
    df = df.copy()

    if cols_to_keep is None:
        cols_to_keep = []

    if cols_to_scale is None:
        cols_to_scale = df.select_dtypes(include='number').columns.tolist()

    cols_to_scale = [col for col in cols_to_scale if col not in cols_to_keep]

    if verbose:
        print(f"[Scaling] Columns to scale: {cols_to_scale}")
        print(f"[Scaling] Columns to keep: {cols_to_keep}")
        print(f"[Scaling] Using scaler: {scaler_type}")

    all_used_cols = cols_to_scale + cols_to_keep
    df = df[all_used_cols].copy()

    df_scaled = df.copy()

    if scaler_type == 'standard':
        scaler = StandardScaler()
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=minmax_range)
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    elif scaler_type == 'zscore':
        scaler = None
        df_scaled[cols_to_scale] = (df[cols_to_scale] - df[cols_to_scale].mean()) / df[cols_to_scale].std()
    else:
        raise ValueError(f"Invalid scaler_type: '{scaler_type}'. Use 'standard', 'minmax', or 'zscore'.")

    if dropna:
        df_scaled.dropna(inplace=True)

    return (df_scaled, scaler) if return_scaler else df_scaled