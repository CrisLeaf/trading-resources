import pandas as pd
import numpy as np


def chandelier_exit(
        df: pd.DataFrame,
        ce_period: int = 22,
        atr_period: int = 22,
        multiplier: float = 3.0,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Chandelier Exit indicator for a given DataFrame.
    The Chandelier Exit is a volatility-based trailing stop indicator that helps traders determine exit points for long and short positions. It uses the highest high or lowest low over a specified period, adjusted by a multiple of the Average True Range (ATR), to set dynamic stop levels.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        ce_period (int, optional): Period for calculating the highest high/lowest low for the Chandelier Exit. Default is 22.
        atr_period (int, optional): Period for calculating the Average True Range (ATR). Default is 22.
        multiplier (float, optional): Multiplier for the ATR to set the stop distance. Default is 3.0.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for Chandelier_Long, Chandelier_Short, Long_Stop, Short_Stop, Direction, CE_Buy, and CE_Sell.
    """
    high, low, close = df[high_column], df[low_column], df[close_column]
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Chandelier Exit
    chandelier_long = high.rolling(window=ce_period, min_periods=ce_period).max() - atr * multiplier
    chandelier_short = low.rolling(window=ce_period, min_periods=ce_period).min() + atr * multiplier
    
    # Add Signal
    long_stop_prev = chandelier_long.shift()
    short_stop_prev = chandelier_short.shift()

    # Update Stops
    long_stop = np.where(close.shift(periods=1) > long_stop_prev, np.maximum(chandelier_long, long_stop_prev), chandelier_long)
    short_stop = np.where(close.shift(periods=1) < short_stop_prev, np.minimum(chandelier_short, short_stop_prev), chandelier_short)
    
    # Direction
    direction = np.where(close > short_stop_prev, 1, np.where(close < long_stop_prev, -1, np.nan))
    direction = pd.Series(direction).ffill().fillna(1)
    

    # Buy/Sell Signals
    ce_buy = (direction == 1) & (direction.shift() == -1)
    ce_sell = (direction == -1) & (direction.shift() == 1)
    

    return pd.DataFrame({
        'Chandelier_Long': chandelier_long,
        'Chandelier_Short': chandelier_short,
        'Long_Stop': long_stop,
        'Short_Stop': short_stop,
        'Direction': direction.values,
        'CE_Buy': ce_buy.values,
        'CE_Sell': ce_sell.values
    })
