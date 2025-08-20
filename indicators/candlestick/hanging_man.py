import pandas as pd
import numpy as np


def is_hanging_man(
        df: pd.DataFrame,
        open_column: str = 'Open',
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        lookback: int = 5
    ) -> pd.Series:
    """
    Identifies Hanging Man candlestick patterns in a given DataFrame.
    A Hanging Man is a bearish reversal candlestick pattern that appears after an uptrend. It is characterized by a small real body near the top of the candle, a long lower shadow, and little or no upper shadow. This pattern suggests potential weakness in the prevailing uptrend.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        open_column (str, optional): Name of the column containing open prices. Default is 'Open'.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        lookback (int, optional): Number of periods to look back for confirming the uptrend. Default is 5.

    Returns:
        pd.Series: A boolean Series indicating the presence of a Hanging Man pattern for each row.
    """
    open_, high, low, close = df[open_column], df[high_column], df[low_column], df[close_column]
    
    body = abs(close - open_)
    lower_shadow = np.minimum(open_, close) - low
    upper_shadow = high - np.maximum(open_, close)
    
    small_body = body <= (high - low) * 0.3
    long_lower_shadow = lower_shadow >= 2 * body
    small_upper_shadow = upper_shadow <= body
    
    trend_up = close.rolling(window=lookback, min_periods=lookback).mean().shift(1) < close
    
    hanging_man = small_body & long_lower_shadow & small_upper_shadow & trend_up
    
    return hanging_man
