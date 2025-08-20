import pandas as pd


def is_doji(
        df: pd.DataFrame,
        open_column: str = 'Open',
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        threshold=0.1
    ) -> pd.Series:
    """
    Identifies Doji candlestick patterns in a given DataFrame.
    A Doji is a candlestick pattern that forms when the open and close prices are very close or equal, indicating indecision in the market. The function uses a threshold to determine if the real body is small enough relative to the total range to qualify as a Doji.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        open_column (str, optional): Name of the column containing open prices. Default is 'Open'.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        threshold (float, optional): Maximum ratio of the real body to the total range to qualify as a Doji. Default is 0.1.

    Returns:
        pd.Series: A boolean Series indicating the presence of a Doji pattern for each row.
    """
    open_, high, low, close = df[open_column], df[high_column], df[low_column], df[close_column]
    body = abs(close - open_)
    range_ = high - low
    
    doji = body <= (range_ * threshold)
    
    return doji
