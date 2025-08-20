import pandas as pd


def simple_moving_average(
        df: pd.DataFrame,
        period: int = 21,
        column: str = 'Close'
    ) -> pd.Series:
    """
    The Simple Moving Average (SMA) is a statistical calculation that computes the average of a selected range of prices, usually closing prices, by the number of periods in that range. It smooths out price data to identify trends over time.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        period (int): The period size for the SMA.
        column (str): The column name for which to calculate the SMA.

    Returns:
        pd.Series: A pandas Series containing the SMA values.
    """
    return df[column].rolling(window=period, min_periods=period).mean()
