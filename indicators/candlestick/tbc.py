import pandas as pd


def is_three_black_crows(
        df: pd.DataFrame,
        open_column: str = 'Open',
        close_column: str = 'Close'
    ) -> pd.Series:
    """
    Identifies Three Black Crows candlestick patterns in a given DataFrame.
    The Three Black Crows is a bearish reversal candlestick pattern that consists of three consecutive long-bodied bearish candles, each closing lower than the previous one. This pattern typically appears after an uptrend and signals a potential reversal to the downside.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        open_column (str, optional): Name of the column containing open prices. Default is 'Open'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A boolean Series indicating the presence of a Three Black Crows pattern for each row.
    """
    open_, close = df[open_column], df[close_column]

    bearish_2 = close.shift(2) < open_.shift(2)
    bearish_1 = close.shift(1) < open_.shift(1)
    bearish_0 = close < open_

    close_down1 = close.shift(2) > close.shift(1)
    close_down2 = close.shift(1) > close

    open_down1 = open_.shift(2) > open_.shift(1)
    open_down2 = open_.shift(1) > open_

    return (bearish_2 & bearish_1 & bearish_0 & close_down1 & close_down2 & open_down1 & open_down2)
