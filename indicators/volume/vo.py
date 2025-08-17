import pandas as pd


def volume_oscillator(
        df: pd.DataFrame,
        fast_period: int = 5,
        slow_period: int = 20,
        signal_period: int = 9,
        column_volume: str = 'Volume'
    ) -> pd.DataFrame:
    """
    Calculates the Volume Oscillator (VO) for a given DataFrame.
    The Volume Oscillator measures the difference between two moving averages of volume, helping to identify changes in market activity and potential trend reversals. It consists of the Volume Oscillator line and a signal line.

    Args:
        df (pd.DataFrame): Input DataFrame containing volume data.
        fast_period (int, optional): Period for the fast exponential moving average (EMA) of volume. Default is 5.
        slow_period (int, optional): Period for the slow exponential moving average (EMA) of volume. Default is 20.
        signal_period (int, optional): Period for the signal line EMA. Default is 9.
        column_volume (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.DataFrame: DataFrame with columns for Volume Oscillator and VO Signal.
    """
    volume = df[column_volume]

    vol_fast = volume.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    vol_slow = volume.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()

    vo = (vol_fast - vol_slow) / vol_slow * 100
    vo_signal = vo.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    
    return pd.DataFrame({
        'Volume Oscillator': vo,
        'VO Signal': vo_signal
    })



import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from copy import deepcopy


ticker = 'AAPL'
df = yf.download(ticker, start='2021-01-01', end='2024-01-01', interval='1d')
df.columns = df.columns.droplevel(1)

vo_df = volume_oscillator(
    df=df,
    fast_period=12,
    slow_period=26,
    signal_period=9
)


subdf = df.iloc[-150: ]
subvo_df = vo_df.iloc[-150: ]

plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(subdf.index, subdf["Volume"], label="Volumen", color="coral")
plt.title("Volumen de: " + ticker)
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(subvo_df.index, subvo_df["Volume Oscillator"], label="Oscilador de Volumen", color="blue")
plt.plot(subvo_df.index, subvo_df["VO Signal"], label="Señal", color="red", linestyle="--")
plt.axhline(y=0, color="green", linestyle="--", linewidth=1.5)
plt.title("Oscilador de Volumen de: " + ticker)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()