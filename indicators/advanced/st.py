import pandas as pd
import numpy as np


def super_trend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the SuperTrend indicator for a given DataFrame.
    The SuperTrend is a trend-following indicator that uses the Average True Range (ATR) to determine dynamic support and resistance levels. It helps identify the current trend direction and potential entry or exit points.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the ATR. Default is 10.
        multiplier (float, optional): Multiplier for the ATR to set the band distance. Default is 3.0.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for SuperTrend, ST_direction, UpperBand, and LowerBand.
    """
    high, low, close = df[high_column], df[low_column], df[close_column]
    
    # Calculate ATR using EMA
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Classic Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Initialization
    st = np.zeros(len(df), dtype=float)
    direction = np.ones(len(df), dtype=int)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    # Iteration
    for i in range(1, len(df)):
        # Determine direction
        if close.iloc[i] > final_upper.iloc[i-1]:
            direction[i] = 1
        elif close.iloc[i] < final_lower.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        # Adjust final bands based on trend
        if direction[i] == 1:
            final_lower.iloc[i] = max(final_lower.iloc[i], final_lower.iloc[i-1])
            final_upper.iloc[i] = np.nan
            st[i] = final_lower.iloc[i]
        else:
            final_upper.iloc[i] = min(final_upper.iloc[i], final_upper.iloc[i-1])
            final_lower.iloc[i] = np.nan
            st[i] = final_upper.iloc[i]

    return pd.DataFrame({
        'SuperTrend': st,
        'ST_direction': direction,
        'UpperBand': final_upper,
        'LowerBand': final_lower
    }, index=df.index)
