import pandas as pd
import numpy as np


def is_marubozu(
        df: pd.DataFrame,
        open_column='Open',
        high_column='High',
        low_column='Low',
        close_column='Close',
        threshold=0.1
    ) -> pd.Series:
    """
    Identifies Marubozu candlestick patterns in a given DataFrame.
    A Marubozu is a candlestick pattern characterized by a long real body with little or no upper and lower shadows, indicating strong buying or selling pressure. The function uses a threshold to determine if the shadows are small enough relative to the total range to qualify as a Marubozu.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        open_column (str, optional): Name of the column containing open prices. Default is 'Open'.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        threshold (float, optional): Maximum ratio of the shadows to the total range to qualify as a Marubozu. Default is 0.1.

    Returns:
        pd.Series: A boolean Series indicating the presence of a Marubozu pattern for each row.
    """
    open_, high, low, close = df[open_column], df[high_column], df[low_column], df[close_column]

    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    range_ = high - low
    
    marubozu = (upper_shadow <= range_ * threshold) & (lower_shadow <= range_ * threshold)
    
    return marubozu
