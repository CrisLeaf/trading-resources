import pandas as pd
import numpy as np

def money_flow_index(
        df: pd.DataFrame,
        period: int = 14,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> pd.Series:
    """
    Calculates the Money Flow Index (MFI) for a given DataFrame.
    The MFI is a momentum indicator that uses both price and volume data to identify overbought or oversold conditions in an asset. It is similar to the RSI but incorporates volume, making it a volume-weighted RSI.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and volume data.
        period (int, optional): Period for calculating the MFI. Default is 14.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        volume_column (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.Series: A pandas Series containing the MFI values.
    """
    high, low, close, volume = df[high_column], df[low_column], df[close_column], df[volume_column]

    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume

    positive_mf = pd.Series(np.where(typical_price > typical_price.shift(1), raw_mf, 0), index=df.index)
    negative_mf = pd.Series(np.where(typical_price < typical_price.shift(1), raw_mf, 0), index=df.index)

    pos_mf_sum = positive_mf.rolling(window=period, min_periods=period).sum()
    neg_mf_sum = negative_mf.rolling(window=period, min_periods=period).sum()

    mfr = pos_mf_sum / neg_mf_sum
    mfi = 100 - (100 / (1 + mfr))

    return mfi
