import pandas as pd


def williams_percentage_range(
        df: pd.DataFrame,
        period: int = 14,
        column_high: str = 'High',
        column_low: str = 'Low',
        column_close: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Williams %R indicator for a given DataFrame.
    Williams %R is a momentum oscillator that measures overbought and oversold levels, indicating potential reversal points in the price of an asset. It compares the current closing price to the highest high and lowest low over a specified period.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating Williams %R. Default is 14.
        column_high (str, optional): Name of the column containing high prices. Default is 'High'.
        column_low (str, optional): Name of the column containing low prices. Default is 'Low'.
        column_close (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the Williams %R values.
    """
    high, low, close = df[column_high], df[column_low], df[column_close]

    high_max = high.rolling(window=period, min_periods=period).max()
    low_min = low.rolling(window=period, min_periods=period).min()

    wr = -100 * (high_max - close) / (high_max - low_min)
    
    return wr


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    wr = williams_percentage_range(df)

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
            y=wr,
            x=wr.index,
            mode='lines',
            line=dict(color='white'),
            name='Williams %R'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[-20] * len(wr),
            x=wr.index,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Oversell -20'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[-80] * len(wr),
            x=wr.index,
            mode='lines',
            line=dict(color='lime', dash='dash'),
            name='Overbought -80'
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
