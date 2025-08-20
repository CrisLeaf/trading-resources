import pandas as pd
import numpy as np


def ulcer_index(
        df: pd.DataFrame,
        column: str = 'Close',
        period: int = 14
    ) -> pd.Series:
    """
    Calculates the Ulcer Index (UI) for a given DataFrame.
    The Ulcer Index is a volatility indicator that measures the depth and duration of price drawdowns from recent highs over a specified period. It is used to assess downside risk and the volatility of an asset.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        column (str, optional): Name of the column containing price data. Default is 'Close'.
        period (int, optional): Period for calculating the Ulcer Index. Default is 14.

    Returns:
        pd.Series: A pandas Series containing the Ulcer Index values.
    """
    close = df[column]

    rolling_max = close.rolling(window=period, min_periods=period).max()
    drawdown_pct = ((close - rolling_max) / rolling_max) * 100

    ui = np.sqrt((drawdown_pct.pow(2)).rolling(window=period, min_periods=period).mean())
    
    return ui
