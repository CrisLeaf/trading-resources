import pandas as pd


def exponential_moving_average(
        df: pd.DataFrame,
        period: int = 26,
        column: str = 'Close'
    ) -> pd.Series:
    """
    The Exponential Moving Average (EMA) is a type of moving average that gives more weight to recent data points, making it more responsive to new information compared to the Simple Moving Average (SMA). It is commonly used in time series analysis and financial markets to smooth out price data and identify trends more quickly.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        column (str): The column name for which to calculate the EMA.
        period (int): The period size for the EMA.

    Returns:
        pd.Series: A pandas Series containing the EMA values.

    """
    return df[column].ewm(span=period, min_periods=period, adjust=False).mean()
