import pandas as pd


def evening_star(
        df: pd.DataFrame,
        high_column: str = 'High',
        low_column: str = 'Low',
        open_column: str = 'Open',
        close_column: str = 'Close',
        period: int = 10
    ) -> pd.DataFrame:
    """
    Identifies Evening Star candlestick patterns in a given DataFrame.
    The Evening Star is a bearish reversal candlestick pattern that typically appears after an uptrend. It consists of three candles: a long bullish candle, a small-bodied candle (which can be bullish or bearish) that gaps up, and a long bearish candle that closes well into the body of the first candle. This pattern signals a potential reversal from an uptrend to a downtrend.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        open_column (str, optional): Name of the column containing open prices. Default is 'Open'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        period (int, optional): Number of periods to look back for confirming the highest high. Default is 10.

    Returns:
        pd.DataFrame: DataFrame containing the rows where an Evening Star pattern is detected.
    """
    open_, high, low, close = df[open_column], df[high_column], df[low_column], df[close_column]

    high_1, low_1, open_1, close_1 = high.shift(1), low.shift(1), open_.shift(1), close.shift(1)
    high_2, low_2, open_2, close_2 = high.shift(2), low.shift(2), open_.shift(2), close.shift(2)

    body_ratio_2 = (close_2 - open_2) / (high_2 - low_2)
    body_ratio_1 = abs(close_1 - open_1) / (high_1 - low_1)
    body_ratio_0 = (open_ - close) / (high - low)
    range_ratio_1 = (high_1 - low_1) / (high_2 - low_2)
    range_ratio_0 = (high - low) / (high_2 - low_2)
    inv_range_ratio = (high_2 - low_2) / (high - low)

    cond = (
        (body_ratio_2 >= 0.70) &
        (open_1 > high_2) &
        (body_ratio_1.between(0.10, 0.50)) &
        (range_ratio_1 <= 0.30) &
        (low_1 > high_2 - 0.30 * (high_2 - low_2)) &
        (body_ratio_0 >= 0.70) &
        (range_ratio_0 <= 1.0) &
        (inv_range_ratio <= 1.3) &
        (range_ratio_0 >= 0.70) &
        (high > low_1) &
        (open_ < open_1) &
        (open_ < close_1) &
        (high < high_2) &
        (close > open_2) &
        (high_1 == high_1.rolling(window=period, min_periods=period).max())
    )

    return df[cond]
