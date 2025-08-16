import numpy as np
import pandas as pd


def cci_index(
        df: pd.DataFrame,
        period: int = 20,
        constant: float = 0.015,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        
    ) -> pd.Series:
    """
    Calculates the Commodity Channel Index (CCI) for a given DataFrame.
    The CCI is a technical indicator used to identify overbought and oversold levels in the price of an asset. It evaluates the direction and strength of the price trend, helping traders determine when to enter or exit a trade.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the CCI. Default is 20.
        constant (float, optional): Constant used in the CCI calculation (typically 0.015). Default is 0.015.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the CCI values.
    """
    
    high, low, close = df[high_column], df[low_column], df[close_column]
    tp = (high + low + close) / 3
    tp_rolling = tp.rolling(window=period, min_periods=period)
    mean_dev = tp_rolling.apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp_rolling.mean()) / (constant * mean_dev)
    cci.name = f'CCI_{period}'
    
    return cci
