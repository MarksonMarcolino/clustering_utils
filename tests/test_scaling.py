import pandas as pd
from clustering_utils import prepare_scaled_df

def test_prepare_scaled_df_minmax():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [10, 20, 30],
        "c": [100, 200, 300]
    })
    scaled = prepare_scaled_df(df, cols_to_scale=["a", "b"], cols_to_keep=["c"], scaler_type="minmax", minmax_range=(0, 1))
    assert scaled.shape == (3, 3)
    assert scaled["a"].min() == 0.0
    assert scaled["a"].max() == 1.0
    assert scaled["c"].equals(df["c"])