import pandas as pd


def schaff_trend_cycle(
        df: pd.DataFrame,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle_period: int = 10,
        smoothing_period1: int = 10,
        smoothing_period2: int = 10,
        column: str = "Close"
    ) -> pd.Series:
    """
    Calculates the Schaff Trend Cycle (STC) for a given DataFrame.
    The STC is a trend-following indicator that combines the concepts of the MACD and the Stochastic Oscillator to identify market trends and potential turning points more quickly. It is designed to generate faster and more accurate signals than traditional trend indicators.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        fast_period (int, optional): Period for the fast EMA in the MACD calculation. Default is 23.
        slow_period (int, optional): Period for the slow EMA in the MACD calculation. Default is 50.
        cycle_period (int, optional): Period for the stochastic calculations. Default is 10.
        smoothing_period1 (int, optional): Smoothing period for the first EMA applied to the stochastic. Default is 10.
        smoothing_period2 (int, optional): Smoothing period for the second EMA applied to the stochastic. Default is 10.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the STC values.
    """
    close = df[column]
    
    # Calculate MACD
    ema_fast = close.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    # Calculate %K stochastic over MACD
    lowest_macd = macd.rolling(window=cycle_period, min_periods=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period, min_periods=cycle_period).max()
    stoch_k = 100 * (macd - lowest_macd) / (highest_macd - lowest_macd)
    
    # Calculate %D
    stoch_d = stoch_k.ewm(span=smoothing_period1, min_periods=smoothing_period1, adjust=False).mean()
    stoch_d_min = stoch_d.rolling(window=cycle_period, min_periods=cycle_period).min()
    stoch_d_max = stoch_d.rolling(window=cycle_period, min_periods=cycle_period).max()
    
    stoch_kd = 100 * (stoch_d - stoch_d_min) / (stoch_d_max - stoch_d_min)

    # Smooth using EMA
    stc = stoch_kd.ewm(span=smoothing_period2, min_periods=smoothing_period2, adjust=False).mean()

    return stc
