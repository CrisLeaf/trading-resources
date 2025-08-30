import pandas as pd


def chaikin_money_flow(
        df: pd.DataFrame,
        period: int = 20,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> pd.Series:
    """
    Calculates the Chaikin Money Flow (CMF) indicator for a given DataFrame.
    The CMF is a volume-based indicator that measures the accumulation and distribution of an asset over a specified
    period. It helps identify buying and selling pressure by combining price and volume data.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and volume data.
        period (int, optional): Period for calculating the CMF. Default is 20.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        volume_column (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.Series: A pandas Series containing the CMF values.
    """
    high, low, close, volume = df[high_column], df[low_column], df[close_column], df[volume_column]

    # Prevent division by zero
    denominator = high - low
    denominator = denominator.replace(0, 1e-10)

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / denominator

    # Money Flow Volume
    mfv = mfm * volume

    # CMF
    cmf_numerator = mfv.rolling(window=period, min_periods=period).sum()
    cmf_denominator = volume.rolling(window=period, min_periods=period).sum()
    cmf = cmf_numerator / cmf_denominator

    return cmf


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    df = yf.download('NVDA', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    cmf = chaikin_money_flow(df)

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

    cmf_pos = cmf.copy()
    cmf_pos[cmf_pos < 0] = 0
    fig.add_trace(
        go.Scatter(
            x=cmf_pos.index,
            y=cmf_pos,
            mode='lines',
            line=dict(color='lime'),
            name='CMF positive'
        ), row=2, col=1
    )
    cmf_neg = cmf.copy()
    cmf_neg[cmf_neg >= 0] = 0
    fig.add_trace(
        go.Scatter(
            x=cmf_neg.index,
            y=cmf_neg,
            mode='lines',
            line=dict(color='red'),
            name='CMF negative'
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[0] * len(cmf),
            x=cmf.index,
            mode='lines',
            line=dict(color='grey'),
            name='0 line'
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

