import numpy as np
import pandas as pd


def on_balance_volume(
        df: pd.DataFrame,
        sma_period: int = None,
        close_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> pd.DataFrame:
    """

    """
    close, volume = df[close_column], df[volume_column]

    obv = np.sign(close.diff(periods=1)) * volume
    obv = obv.fillna(value=volume).cumsum()
    output_df = pd.DataFrame({'OBV': obv})

    if sma_period is not None and sma_period > 1:
        obv_sma = obv.rolling(window=sma_period, min_periods=sma_period).mean()
        output_df[f'OBV_SMA_{sma_period}'] = obv_sma

    return output_df
