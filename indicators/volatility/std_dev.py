import pandas as pd

def standard_deviation(
        df: pd.DataFrame,
        column: str = 'Close',
        period: int = 14,
        ddof: int = 0
    ) -> pd.Series:
    """
    Calculates the rolling standard deviation for a given column in the DataFrame.
    Standard deviation is a statistical measure of volatility, indicating how much the values of a dataset deviate from the mean. In trading, it is commonly used to assess the volatility of asset prices over a specified period.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        column (str, optional): Name of the column for which to calculate the standard deviation. Default is 'Close'.
        period (int, optional): Period for calculating the rolling standard deviation. Default is 14.
        ddof (int, optional): Delta degrees of freedom for the calculation. Default is 0.

    Returns:
        pd.Series: A pandas Series containing the rolling standard deviation values.
    """
    close = df[column]

    return close.rolling(window=period, min_periods=period).std(ddof=ddof)
