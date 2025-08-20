import pandas as pd

def average_true_range(
        df: pd.DataFrame,
        period: int = 14,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Average True Range (ATR) for a given DataFrame.
    The ATR is a volatility indicator that measures market volatility by decomposing the entire range of an asset price for a given period. It helps traders assess the degree of price movement or volatility.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the ATR. Default is 14.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the ATR values.
    """
    high, low, close = df[high_column], df[low_column], df[close_column]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1)
    tr = tr.max(axis=1)

    # ATR using EMA
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()

    return atr
