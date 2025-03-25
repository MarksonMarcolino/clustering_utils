import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_scaled_df(
    df,
    cols_to_scale=None,
    cols_to_keep=None,
    dropna=True,
    return_scaler=False,
    verbose=True,
    scaler_type='standard',  # 'standard' or 'minmax'
    minmax_range=(0, 1)       # only used if scaler_type is 'minmax'
):
    """
    Scales selected columns of a DataFrame using StandardScaler or MinMaxScaler.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - cols_to_scale (list[str], optional): Columns to scale. Defaults to all numeric.
    - cols_to_keep (list[str], optional): Columns to leave unscaled.
    - dropna (bool): Drop rows with NaNs AFTER scaling. Default is True.
    - return_scaler (bool): Return the fitted scaler object.
    - verbose (bool): Print which columns are being scaled.
    - scaler_type (str): 'standard' or 'minmax'
    - minmax_range (tuple): (min, max) range for MinMaxScaler.

    Returns:
    - pd.DataFrame: Scaled DataFrame
    - Scaler object (optional)
    """
    df = df.copy()

    if cols_to_keep is None:
        cols_to_keep = []

    if cols_to_scale is None:
        cols_to_scale = df.select_dtypes(include='number').columns.tolist()

    all_used_cols = cols_to_scale + cols_to_keep
    df = df[all_used_cols].copy()

    cols_to_scale = [col for col in cols_to_scale if col not in cols_to_keep]

    if verbose:
        print(f"[Scaling] Columns to scale: {cols_to_scale}")
        print(f"[Scaling] Using scaler: {scaler_type}")

    # Choose scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=minmax_range)
    else:
        raise ValueError(f"Invalid scaler_type: '{scaler_type}'. Use 'standard' or 'minmax'.")

    # Apply scaling
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Drop NA after scaling
    if dropna:
        df_scaled.dropna(inplace=True)

    return (df_scaled, scaler) if return_scaler else df_scaled