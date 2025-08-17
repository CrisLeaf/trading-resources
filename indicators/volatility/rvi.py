import pandas as pd
import numpy as np


def relative_volatility_index(
        df: pd.DataFrame,
        std_dev_period: int = 14,
        smooth_period: int = 14,
        column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Relative Volatility Index (RVI) for a given DataFrame.
    The RVI is a volatility-based indicator that measures the direction of volatility. It is similar in concept to the Relative Strength Index (RSI), but uses standard deviation of price changes instead of price itself, helping to identify overbought and oversold conditions based on volatility.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        std_dev_period (int, optional): Period for calculating the rolling standard deviation. Default is 14.
        smooth_period (int, optional): Period for smoothing the volatility averages. Default is 14.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the RVI values.
    """
    close = df[column]
    
    returns = close.diff()
    volatility = returns.rolling(window=std_dev_period, min_periods=std_dev_period).std(ddof=0)

    vol_up = np.where(returns > 0, volatility, 0.0)
    vol_down = np.where(returns < 0, volatility, 0.0)

    smma_up = pd.Series(vol_up, index=df.index).ewm(span=smooth_period, adjust=False).mean()
    smma_down = pd.Series(vol_down, index=df.index).ewm(span=smooth_period, adjust=False).mean()

    rvi = 100 * smma_up / (smma_up + smma_down)
    
    return rvi
