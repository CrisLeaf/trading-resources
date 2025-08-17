import pandas as pd
import numpy as np


def is_hammer(
        df: pd.DataFrame,
        open_column: str = 'Open',
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.Series:
    """
    Identifies Hammer candlestick patterns in a given DataFrame.
    A Hammer is a bullish reversal candlestick pattern characterized by a small real body, a long lower shadow, and little or no upper shadow. It typically appears after a downtrend and signals a potential reversal to the upside.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        open_column (str, optional): Name of the column containing open prices. Default is 'Open'.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A boolean Series indicating the presence of a Hammer pattern for each row.
    """
    open_, high, low, close = df[open_column], df[high_column], df[low_column], df[close_column]
    
    body = abs(close - open_)
    lower_shadow = np.minimum(open_, close) - low
    upper_shadow = high - np.maximum(open_, close)
    
    small_body = body <= (high - low) * 0.3
    long_lower_shadow = lower_shadow >= 2 * body
    small_upper_shadow = upper_shadow <= body
    
    hammer = small_body & long_lower_shadow & small_upper_shadow
    
    return hammer
