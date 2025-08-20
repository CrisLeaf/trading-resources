import numpy as np
import pandas as pd


def parabolic_sar(
        df: pd.DataFrame,
        acceleration: float = 0.02,
        max_step: float = 0.2,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Parabolic SAR (Stop and Reverse) indicator for a given DataFrame.
    The Parabolic SAR is a trend-following indicator used to determine potential reversal points in the price direction of an asset. It helps traders identify entry and exit points by plotting dots above or below the price, indicating the direction of the trend.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        acceleration (float, optional): Acceleration factor used in the SAR calculation. Default is 0.02.
        max_step (float, optional): Maximum value for the acceleration factor. Default is 0.2.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for PSAR, UpTrend, and DownTrend.
    """
    high, low, close = df[high_column].to_numpy().copy(), df[low_column].to_numpy().copy(), df[close_column].to_numpy().copy()
    psar_up, psar_down = np.repeat(np.nan, len(df)), np.repeat(np.nan, len(df))

    # Initialization
    up_trend = True
    up_trend_high = high[0]
    down_trend_low = low[0]
    acc_factor = acceleration

    # Iteration over prices to calculate PSAR
    for i in range(2, close.shape[0]):
        reversal = False
        max_high = high[i]
        min_low = low[i]
        
        # Uptrend
        if up_trend:
            close[i] = close[i-1] + (acc_factor * (up_trend_high - close[i-1]))
            
            if min_low < close[i]:
                reversal = True
                close[i] = up_trend_high
                down_trend_low = min_low
                acc_factor = acceleration
            
            else:
                if max_high > up_trend_high:
                    up_trend_high = max_high
                    acc_factor = min(acc_factor + acceleration, max_step)
                
                low1 = low[i-1]
                low2 = low[i-2]
                
                if low2 < close[i]:
                    close[i] = low2
                    
                elif low1 < close[i]:
                    close[i] = low1

        # Downtrend
        else:
            close[i] = close[i-1] - (acc_factor * (close[i-1] - down_trend_low))
            
            if max_high > close[i]:
                reversal = True
                close[i] = down_trend_low
                up_trend_high = max_high
                acc_factor = acceleration
                
            else:
                if min_low < down_trend_low:
                    down_trend_low = min_low
                    acc_factor = min(acc_factor + acceleration, max_step)

                high1 = high[i-1]
                high2 = high[i-2]

                if high2 > close[i]:
                    close[i] = high2
                    
                elif high1 > close[i]:
                    close[i] = high1
                    
        up_trend = up_trend != reversal
        
        if up_trend:
            psar_up[i] = close[i]
        else:
            psar_down[i] = close[i]

    return pd.DataFrame({
        'PSAR': close,
        'UpTrend': psar_up,
        'DownTrend': psar_down
    })
