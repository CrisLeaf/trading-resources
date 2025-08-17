import pandas as pd


def relative_strength_index(
        df: pd.DataFrame,
        period: int = 14,
        column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a given DataFrame.
    The RSI is a momentum oscillator that measures the speed and change of price movements. It is used to identify overbought or oversold conditions in the price of an asset, helping traders determine potential reversal points.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the RSI. Default is 14.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the RSI values.
    """
    delta = df[column].diff(periods=1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


import pandas as pd

import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

df = yf.download('NVDA', start='2021-01-01', end='2024-01-01', interval='1d')
df.columns = df.columns.droplevel(1)


rsi = relative_strength_index(df, period=14, column='Close')


subdf = df.iloc[-150: ]
subrsi = rsi.iloc[-150:]

over_buy_sell = [
    mpf.make_addplot([70] * len(subrsi), panel=2, color='gray', linestyle='--'),
    mpf.make_addplot([30] * len(subrsi), panel=2, color='gray', linestyle='--')
]

ap = [
      mpf.make_addplot(subrsi, panel=2, color='blue', ylabel='RSI')
] + over_buy_sell

mpf.plot(subdf,
    type='candle',
    style='yahoo',
    volume=True,
    addplot=ap,
    title='Índice de Fuerza Relativa (RSI)',
    ylabel='Precio del Activo',
    ylabel_lower='Volumen',
    figsize=(22, 10),
    figscale=3.0,
    warn_too_much_data=df.shape[0]
)

plt.show()