import pandas as pd

def average_true_range(
        df: pd.DataFrame,
        period: int = 14,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Average True Range (ATR) for a given DataFrame.
    The ATR is a volatility indicator that measures market volatility by decomposing the entire range of an asset price
    for a given period. It helps traders assess the degree of price movement or volatility.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the ATR. Default is 14.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the ATR values.
    """
    high, low, close = df[high_column], df[low_column], df[close_column]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1)
    tr = tr.max(axis=1)

    # ATR using EMA
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()

    return atr


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    atr = average_true_range(df)

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
            y=atr,
            x=atr.index,
            mode='lines',
            line=dict(color='skyblue'),
            name='ATR'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[atr.mean()] * len(atr),
            x=atr.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='ATR Mean'
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
