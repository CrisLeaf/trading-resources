import pandas as pd


def detrended_price_oscillator(
        df: pd.DataFrame,
        period: int = 20,
        centered: bool = True,
        column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Detrended Price Oscillator (DPO) for a given DataFrame.
    The DPO is a technical indicator used to remove the long-term trend from price data, making it easier to identify
    short-term cycles. It helps traders focus on shorter-term price movements by subtracting a shifted moving average
    from the price.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the moving average. Default is 20.
        centered (bool, optional): Whether to center the DPO by shifting the result. Default is True.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the DPO values.
    """
    close = df[column]
    
    sma = close.rolling(window=period, min_periods=period).mean()
    shift = int((period / 2) + 1)
    
    if centered:
        dpo = (close.shift(shift) - sma).shift(-shift)
    else:
        dpo = close - sma.shift(shift)

    return dpo


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    dpo = detrended_price_oscillator(df)

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
            y=dpo,
            x=dpo.index,
            mode='lines',
            line=dict(color='coral'),
            name='DPO'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[0] * len(dpo),
            x=dpo.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[-20] * len(dpo),
            x=dpo.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[20] * len(dpo),
            x=dpo.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            showlegend=False
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
