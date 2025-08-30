import pandas as pd
import numpy as np


def relative_volatility_index(
        df: pd.DataFrame,
        std_dev_period: int = 14,
        smooth_period: int = 14,
        column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Relative Volatility Index (RVI) for a given DataFrame.
    The RVI is a volatility-based indicator that measures the direction of volatility. It is similar in concept to
    the Relative Strength Index (RSI), but uses standard deviation of price changes instead of price itself, helping
    to identify overbought and oversold conditions based on volatility.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        std_dev_period (int, optional): Period for calculating the rolling standard deviation. Default is 14.
        smooth_period (int, optional): Period for smoothing the volatility averages. Default is 14.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the RVI values.
    """
    close = df[column]
    
    returns = close.diff()
    volatility = returns.rolling(window=std_dev_period, min_periods=std_dev_period).std(ddof=0)

    vol_up = np.where(returns > 0, volatility, 0.0)
    vol_down = np.where(returns < 0, volatility, 0.0)

    smma_up = pd.Series(vol_up, index=df.index).ewm(span=smooth_period, adjust=False).mean()
    smma_down = pd.Series(vol_down, index=df.index).ewm(span=smooth_period, adjust=False).mean()

    rvi = 100 * smma_up / (smma_up + smma_down)
    
    return rvi


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    rvi = relative_volatility_index(df)
    rvi_sma = rvi.rolling(window=14).mean()

    # Plots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        line=dict(color='skyblue', width=1),
        name='Close'
    ), row=1, col=1)

    # Signals
    fig.add_trace(
        go.Scatter(
            y=rvi,
            x=rvi.index,
            mode='lines',
            line=dict(color='grey'),
            name='RVI'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=rvi_sma,
            x=rvi_sma.index,
            mode='lines',
            line=dict(color='coral'),
            name='RVI SMA 14'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[70] * len(rvi),
            x=rvi.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Strong Bullish Trend'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[30] * len(rvi),
            x=rvi.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Strong Bearish Trend'
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        title='Signals Plot',
        xaxis2_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgb(20, 20, 20)',
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=900,
        width=1000
    )

    fig.show()
