import pandas as pd


def three_outside_down(
        df: pd.DataFrame,
       high_column: str = 'High',
       low_column: str = 'Low',
       open_column: str = 'Open',
       close_column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Identifies Three Outside Down candlestick patterns in a given DataFrame.
    The Three Outside Down is a bearish reversal candlestick pattern that consists of three candles: a bullish candle, a bearish engulfing candle, and a third bearish candle that confirms the reversal. This pattern typically appears after an uptrend and signals a potential reversal to the downside.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        open_column (str, optional): Name of the column containing open prices. Default is 'Open'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame containing the rows where a Three Outside Down pattern is detected.
    """
    open_, high, low, close = df[open_column], df[high_column], df[low_column], df[close_column]

    high_1, low_1, open_1, close_1 = high.shift(1), low.shift(1), open_.shift(1), close.shift(1)
    high_2, low_2, open_2, close_2 = high.shift(2), low.shift(2), open_.shift(2), close.shift(2)

    body_ratio_2 = (close_2 - open_2) / (high_2 - low_2)
    body_ratio_1 = (open_1 - close_1) / (high_1 - low_1)
    body_ratio_0 = (open_ - close) / (high - low)
    relative_body_0_1 = (open_ - close) / (open_1 - close_1)
    range_ratio_1_2 = (high_1 - low_1) / (high_2 - low_2)
    mid_1 = (open_1 + close_1) / 2
    threshold_1 = open_1 - (open_1 - close_1) * 0.33

    cond = (
        (body_ratio_2 >= 0.50) &
        (low_2 > close_1) &
        (high_2 < open_1) &
        (body_ratio_1 >= 0.60) &
        (range_ratio_1_2 <= 2.5) &
        (open_ > close_1) &
        (close < close_1) &
        (body_ratio_0 >= 0.45) &
        (relative_body_0_1 < 1.0) &
        (open_ < mid_1) &
        (high < threshold_1) &
        (low < low_1)
    )

    return df[cond]
