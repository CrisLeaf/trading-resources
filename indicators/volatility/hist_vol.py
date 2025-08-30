import numpy as np
import pandas as pd


def historic_volatility(
        df: pd.DataFrame,
        period: int = 30,
        column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the historic volatility for a given DataFrame.
    Historic volatility is a statistical measure of the dispersion of returns for a given asset over a specified
    period. It is calculated as the standard deviation of the logarithmic returns, providing insight into the asset's
    past price fluctuations.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the rolling historic volatility. Default is 30.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the historic volatility values.
    """
    close = df[column]
    
    log_returns = np.log(close / close.shift(1))
    hist_vol = log_returns.rolling(window=period).std(ddof=0)
    
    return hist_vol


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    hist_vol = historic_volatility(df)

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
            y=hist_vol,
            x=hist_vol.index,
            mode='lines',
            line=dict(color='skyblue'),
            name='Historic Volatility'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[hist_vol.mean()] * len(hist_vol),
            x=hist_vol.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Historic Volatility Mean'
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

