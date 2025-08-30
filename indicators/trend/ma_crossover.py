import numpy as np
import pandas as pd


def moving_average_crossover(
        df: pd.DataFrame,
        fast_period: int = 10,
        slow_period: int = 30,
        column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates a moving average crossover strategy on a given DataFrame.
    This function computes two simple moving averages (SMA) for the 'Close' price column using specified fast and slow
    periods. It generates trading signals based on the crossover of these moving averages.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least a 'Close' column with price data.
        fast_period (int, optional): Window size for the fast (short-term) moving average. Default is 10.
        slow_period (int, optional): Window size for the slow (long-term) moving average. Default is 30.
        column (str, optional): Name of the price column to use for calculations. Default is 'Close'.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the following additional columns:
            - 'SMA_fast': Fast period simple moving average of 'Close'.
            - 'SMA_slow': Slow period simple moving average of 'Close'.
            - 'Signal': Signal indicating a crossover event (1 for bullish crossover, -1 for bearish crossover, 0
                        otherwise).
    """
    price_col = df[column]
    
    sma_fast = price_col.rolling(window=fast_period, min_periods=fast_period).mean()
    sma_fast_shift = sma_fast.shift(1)
    sma_slow = price_col.rolling(window=slow_period, min_periods=slow_period).mean()
    sma_slow_shift = sma_slow.shift(1)
    
    crossover = np.where(
        ((sma_fast > sma_slow) & (sma_slow_shift > sma_fast_shift)),
        1,
        np.where(
            (sma_fast < sma_slow) & (sma_slow_shift < sma_fast_shift),
            -1,
            0
        )
    )
    mac = pd.concat(
        [sma_fast, sma_slow, pd.Series(crossover, index=df.index)], 
        axis=1
    )
    mac.columns = ["SMA_fast", "SMA_slow", "Signal"]
    
    # Direction
    direction = pd.Series(index=df.index, dtype=int)

    direction.iloc[0] = 0

    for i in range(1, len(df)):
        if mac['Signal'].iloc[i] == 1:
            direction.iloc[i] = 1
        elif mac['Signal'].iloc[i] == -1:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

    mac['Direction'] = direction
    
    return mac


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    mac = moving_average_crossover(df)
    
    df["Buy_Signal"] = ((mac["Direction"] == 1) & (mac["Direction"].shift(1) == -1)).astype(int)
    df["Sell_Signal"] = ((mac["Direction"] == -1) & (mac["Direction"].shift(1) == 1)).astype(int)

    # Plots
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        line=dict(color='skyblue', width=1),
        name='Close'
    ))
    fig.add_trace(go.Scatter(
        x=mac.index,
        y=mac['SMA_fast'],
        mode='lines',
        line=dict(color='orange', width=1.5),
        name='SMA Fast'
    ))
    fig.add_trace(go.Scatter(
        x=mac.index,
        y=mac['SMA_slow'],
        mode='lines',
        line=dict(color='red', width=1.5),
        name='SMA Slow'
    ))
    # Buy/Sell Signal
    fig.add_trace(go.Scatter(
        x=df.loc[df['Buy_Signal'] == 1].index,
        y=df.loc[df['Buy_Signal'] == 1]['Close'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='lime', size=12),
        name='Buy Signal'
    ))
    fig.add_trace(go.Scatter(
        x=df.loc[df['Sell_Signal'] == 1].index,
        y=df.loc[df['Sell_Signal'] == 1]['Close'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell Signal'
    ))
    fig.update_layout(
        template='plotly_dark',
        title='Signals Plot',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgb(20, 20, 20)',
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=600,
        width=1000
    )
    
    fig.show()