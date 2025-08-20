import pandas as pd


def williams_percentage_range(
        df: pd.DataFrame,
        period: int = 14,
        column_high: str = 'High',
        column_low: str = 'Low',
        column_close: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Williams %R indicator for a given DataFrame.
    Williams %R is a momentum oscillator that measures overbought and oversold levels, indicating potential reversal points in the price of an asset. It compares the current closing price to the highest high and lowest low over a specified period.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating Williams %R. Default is 14.
        column_high (str, optional): Name of the column containing high prices. Default is 'High'.
        column_low (str, optional): Name of the column containing low prices. Default is 'Low'.
        column_close (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the Williams %R values.
    """
    high, low, close = df[column_high], df[column_low], df[column_close]

    high_max = high.rolling(window=period, min_periods=period).max()
    low_min = low.rolling(window=period, min_periods=period).min()

    wr = -100 * (high_max - close) / (high_max - low_min)
    
    return wr
