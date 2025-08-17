import pandas as pd


def is_three_white_soldiers(
        df: pd.DataFrame,
        open_column: str = 'Open',
        close_column: str = 'Close'
    ) -> pd.Series:
    """
    Identifies Three White Soldiers candlestick patterns in a given DataFrame.
    The Three White Soldiers is a bullish reversal candlestick pattern that consists of three consecutive long-bodied bullish candles, each closing higher than the previous one. This pattern typically appears after a downtrend and signals a potential reversal to the upside.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        open_column (str, optional): Name of the column containing open prices. Default is 'Open'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A boolean Series indicating the presence of a Three White Soldiers pattern for each row.
    """
    open_, close = df[open_column], df[close_column]

    bullish_2 = close.shift(2) > open_.shift(2)
    bullish_1 = close.shift(1) > open_.shift(1)
    bullish_0 = close > open_

    close_up1 = close.shift(2) < close.shift(1)
    close_up2 = close.shift(1) < close

    open_up1 = open_.shift(2) < open_.shift(1)
    open_up2 = open_.shift(1) < open_

    return (bullish_2 & bullish_1 & bullish_0 & close_up1 & close_up2 & open_up1 & open_up2)
