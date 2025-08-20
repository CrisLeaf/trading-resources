import numpy as np
import pandas as pd


def historic_volatility(
        df: pd.DataFrame,
        period: int = 30,
        column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the historic volatility for a given DataFrame.
    Historic volatility is a statistical measure of the dispersion of returns for a given asset over a specified period. It is calculated as the standard deviation of the logarithmic returns, providing insight into the asset's past price fluctuations.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the rolling historic volatility. Default is 30.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the historic volatility values.
    """
    close = df[column]
    
    log_returns = np.log(close / close.shift(1))
    hist_vol = log_returns.rolling(window=period).std(ddof=0)
    
    return hist_vol
