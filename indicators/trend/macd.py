import pandas as pd

    
def macd_index(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD) and related indicators (MACD line, Signal line, Histogram) for a given DataFrame.
    The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. It consists of the MACD line (difference between fast and slow EMAs), the Signal line (EMA of the MACD line), and the Histogram (difference between MACD and Signal lines).
    
    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        fast_period (int, optional): Period for calculating the fast EMA. Default is 12.
        slow_period (int, optional): Period for calculating the slow EMA. Default is 26.
        signal_period (int, optional): Period for calculating the Signal line (EMA of MACD line). Default is 9.
        column (str, optional): Name of the column containing price data. Default is 'Close'.
    
    Returns:
        pd.DataFrame: DataFrame with additional columns for MACD line, Signal line, and Histogram.
    """
    ma_fast = df[column].ewm(span=fast_period, adjust=False).mean()
    ma_slow = df[column].ewm(span=slow_period, adjust=False).mean()
    
    macd_d = ma_fast - ma_slow
    signal = macd_d.ewm(span=signal_period, adjust=False).mean()
    
    return pd.DataFrame({
        'MACD': macd_d,
        'Signal': signal,
    })
