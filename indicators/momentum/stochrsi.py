import pandas as pd


def stochastic_rsi(
        df: pd.DataFrame,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
        column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Stochastic Relative Strength Index (StochRSI) for a given DataFrame.
    The StochRSI is a momentum oscillator that applies the Stochastic formula to the Relative Strength Index (RSI)
    values, making it more sensitive and responsive to price changes. It is used to identify overbought and oversold
    conditions, as well as potential trend reversals.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        rsi_period (int, optional): Period for calculating the RSI. Default is 14.
        stoch_period (int, optional): Period for calculating the Stochastic RSI. Default is 14.
        smooth_k (int, optional): Smoothing period for the %K line. Default is 3.
        smooth_d (int, optional): Smoothing period for the %D line. Default is 3.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for StochRSI, StochRSI_K, and StochRSI_D.
    """
    close = df[column]
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # StochRSI
    min_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    max_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)

    # Smooth %K and %D
    stoch_rsi_k = stoch_rsi.rolling(window=smooth_k, min_periods=smooth_k).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    
    return pd.DataFrame({
        'StochRSI': stoch_rsi * 100,
        'StochRSI_K': stoch_rsi_k * 100,
        'StochRSI_D': stoch_rsi_d * 100
    })


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    stochrsi = stochastic_rsi(df)

    # Plots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3, 0.3]
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
            y=stochrsi['StochRSI'],
            x=stochrsi.index,
            mode='lines',
            line=dict(color='gray'),
            name='StochRSI'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=stochrsi['StochRSI_K'],
            x=stochrsi.index,
            mode='lines',
            line=dict(color='orange'),
            name='StochRSI K'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=stochrsi['StochRSI_D'],
            x=stochrsi.index,
            mode='lines',
            line=dict(color='violet'),
            name='StochRSI D'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[80] * len(stochrsi.index),
            x=stochrsi.index,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Overbought 80'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[20] * len(stochrsi.index),
            x=stochrsi.index,
            mode='lines',
            line=dict(color='lime', dash='dash'),
            name='Oversell 20'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[80] * len(stochrsi.index),
            x=stochrsi.index,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Overbought 80'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[20] * len(stochrsi.index),
            x=stochrsi.index,
            mode='lines',
            line=dict(color='lime', dash='dash'),
            name='Oversell 20'
        ),
        row=3, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        title='Signals Plot',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgb(20, 20, 20)',
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=1200,
        width=1000
    )

    fig.show()
