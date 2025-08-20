import pandas as pd


def ichimoku_cloud(
        df: pd.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        period_senkou_b: int = 52,
        displacement: int = 26,
        offset: bool = False,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Ichimoku Cloud indicator for a given DataFrame.
    The Ichimoku Cloud is a comprehensive technical indicator that defines support and resistance, identifies trend direction, gauges momentum, and provides trading signals. It consists of five lines: Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, and Chikou Span.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        tenkan_period (int, optional): Period for calculating the Tenkan-sen (Conversion Line). Default is 9.
        kijun_period (int, optional): Period for calculating the Kijun-sen (Base Line). Default is 26.
        period_senkou_b (int, optional): Period for calculating the Senkou Span B (Leading Span B). Default is 52.
        displacement (int, optional): Number of periods to shift the cloud forward. Default is 26.
        offset (bool, optional): Whether to offset the Senkou Span A and B. Default is False.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, and Chikou Span.
    """
    high, low, close = df[high_column], df[low_column], df[close_column]
    
    # Tenkan-sen
    rolling_min_tenkan = low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
    rolling_max_tenkan = high.rolling(window=tenkan_period, min_periods=tenkan_period).max()
    tenkan_sen = (rolling_min_tenkan + rolling_max_tenkan) / 2

    # Kijun-sen
    rolling_min_kijun = low.rolling(window=kijun_period, min_periods=kijun_period).min()
    rolling_max_kijun = high.rolling(window=kijun_period, min_periods=kijun_period).max()
    kijun_sen = (rolling_min_kijun + rolling_max_kijun) / 2
    
    # Senkow Span A - Cloud
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkow Span B - Cloud
    rolling_min_senkou = low.rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    rolling_max_senkou = high.rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    senkou_span_b = ((rolling_min_senkou + rolling_max_senkou) / 2)
    
    # Chikou Span
    chikou_span = close.shift(periods=-displacement)
    
    # Creeate DataFrame
    ic = pd.concat([tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span], axis=1)
    ic.columns = ['Tenkan-sen', 'Kijun-sen', 'Senkou Span A', 'Senkou Span B', 'Chikou Span']
    
    # Displacement
    if not offset:
        ic['Senkou Span A'] = ic['Senkou Span A'].shift(periods=kijun_period)
        ic['Senkou Span B'] = ic['Senkou Span B'].shift(periods=kijun_period)
    
    return ic
