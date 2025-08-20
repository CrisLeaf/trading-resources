import pandas as pd


def detrended_price_oscillator(
        df: pd.DataFrame,
        period: int = 20,
        centered: bool = True,
        column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Detrended Price Oscillator (DPO) for a given DataFrame.
    The DPO is a technical indicator used to remove the long-term trend from price data, making it easier to identify short-term cycles. It helps traders focus on shorter-term price movements by subtracting a shifted moving average from the price.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the moving average. Default is 20.
        centered (bool, optional): Whether to center the DPO by shifting the result. Default is True.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the DPO values.
    """
    close = df[column]
    
    sma = close.rolling(window=period, min_periods=period).mean()
    shift = int((period / 2) + 1)
    
    if centered:
        dpo = (close.shift(shift) - sma).shift(-shift)
    else:
        dpo = close - sma.shift(shift)

    return dpo
