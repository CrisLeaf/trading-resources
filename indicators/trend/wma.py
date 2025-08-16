import pandas as pd


def weighted_moving_average(
        df: pd.DataFrame,
        period: int = 9,
        column: str = 'Close'
    ) -> pd.Series:
    """
        Calculates the Weighted Moving Average (WMA) for a specified column in a pandas DataFrame.
        The WMA assigns more weight to recent data points, making it more responsive to new information compared to a simple moving average.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            period (int, optional): The number of periods to use for calculating the WMA. Default is 9.
            column (str, optional): The name of the column to calculate the WMA on. Default is 'Close'.
        Returns:
            pd.Series: A pandas Series containing the weighted moving average values.
    """
    weights = pd.Series(range(1, period + 1), index=range(period))

    return df[column].rolling(window=period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
