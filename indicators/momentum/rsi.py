import pandas as pd


def relative_strength_index(
        df: pd.DataFrame,
        period: int = 14,
        column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a given DataFrame.
    The RSI is a momentum oscillator that measures the speed and change of price movements. It is used to identify
    overbought or oversold conditions in the price of an asset, helping traders determine potential reversal points.

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


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    rsi = relative_strength_index(df)

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
            y=rsi,
            x=rsi.index,
            mode='lines',
            line=dict(color='white'),
            name='RSI'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[30] * len(rsi),
            x=rsi.index,
            mode='lines',
            line=dict(color='lime', dash='dash'),
            name='Oversell 30'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[70] * len(rsi),
            x=rsi.index,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Overbought 70'
        ),
        row=2, col=1
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
        height=900,
        width=1000
    )

    fig.show()
