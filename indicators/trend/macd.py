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
    
    # Direction
    direction = pd.Series(index=df.index, dtype=int)
    direction.iloc[0] = 0

    for i in range(1, len(df)):
        if macd_d.iloc[i] > signal.iloc[i] and macd_d.iloc[i-1] <= signal.iloc[i-1]:
            direction.iloc[i] = 1
        elif macd_d.iloc[i] < signal.iloc[i] and macd_d.iloc[i-1] >= signal.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

    return pd.DataFrame({
        'MACD': macd_d,
        'Signal': signal,
        'Direction': direction
    })


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    macd = macd_index(df)
    
    df["Buy_Signal"] = ((macd["Direction"] == 1) & (macd["Direction"].shift(1) == -1)).astype(int)
    df["Sell_Signal"] = ((macd["Direction"] == -1) & (macd["Direction"].shift(1) == 1)).astype(int)

    # Plots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.25, 0.25]
    )
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        line=dict(color='skyblue', width=1),
        name='Close'
    ), row=1, col=1)
    
    # Signals
    fig.add_trace(go.Scatter(
        x=macd.index,
        y=macd['MACD'],
        mode='lines',
        line=dict(color='lime', width=1.5),
        name='MACD'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=macd.index,
        y=macd['Signal'],
        mode='lines',
        line=dict(color='red', width=1.5),
        name='Signal'
    ), row=2, col=1)
    
    colors = np.where(macd["MACD"] > macd["Signal"], "lime", "red")
    fig.add_trace(go.Bar(
        x=macd.index,
        y=macd["MACD"] - macd["Signal"],
        marker_color=colors,
        name="Macd - Signal"
    ), row=3, col=1)

    # Buy/Sell Signal
    fig.add_trace(go.Scatter(
        x=df.loc[df['Buy_Signal'] == 1].index,
        y=df.loc[df['Buy_Signal'] == 1]['Close'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='lime', size=12),
        name='Buy Signal'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.loc[df['Sell_Signal'] == 1].index,
        y=df.loc[df['Sell_Signal'] == 1]['Close'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell Signal'
    ), row=1, col=1)
    fig.update_layout(
        template='plotly_dark',
        title='Signals Plot',
        xaxis3_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgb(20, 20, 20)',
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=900,
        width=1000
    )
    
    fig.show()
