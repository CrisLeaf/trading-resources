import pandas as pd


def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        k: float = 2.0,
        ddof: int = 0,
        column: str = "Close"
    ) -> pd.DataFrame:
    """
    Calculates the Bollinger Bands for a given DataFrame.
    Bollinger Bands are a volatility indicator consisting of a middle band (simple moving average), an upper band,
    and a lower band. The upper and lower bands are calculated as a specified number of standard deviations above and
    below the moving average, helping to identify overbought and oversold conditions.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the moving average and standard deviation. Default is 20.
        k (float, optional): Number of standard deviations to set the bands. Default is 2.0.
        ddof (int, optional): Delta degrees of freedom for the standard deviation calculation. Default is 0.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for Bollinger_Mid, Bollinger_Upper, and Bollinger_Lower.
    """
    close = df[column]

    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=ddof)

    upper_band = sma + (k * std)
    lower_band = sma - (k * std)

    return pd.DataFrame({
        "Bollinger_Mid": sma,
        "Bollinger_Upper": upper_band,
        "Bollinger_Lower": lower_band
    })


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    bb = bollinger_bands(df)

    # Plots
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            line=dict(color='skyblue', width=1),
            name='Close'
        )
    )

    # Signals
    fig.add_trace(
        go.Scatter(
            y=bb['Bollinger_Mid'],
            x=bb.index,
            mode='lines',
            line=dict(color='coral'),
            name='Bollinger Mid'
        )
    )
    fig.add_trace(
        go.Scatter(
            y=bb['Bollinger_Lower'],
            x=bb.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Bollinger Lower'
        )
    )
    fig.add_trace(
        go.Scatter(
            y=bb['Bollinger_Upper'],
            x=bb.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Bollinger Upper'
        )
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
        height=600,
        width=1000
    )

    fig.show()

