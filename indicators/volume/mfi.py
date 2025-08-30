import pandas as pd
import numpy as np

def money_flow_index(
        df: pd.DataFrame,
        period: int = 14,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> pd.Series:
    """
    Calculates the Money Flow Index (MFI) for a given DataFrame.
    The MFI is a momentum indicator that uses both price and volume data to identify overbought or oversold conditions
    in an asset. It is similar to the RSI but incorporates volume, making it a volume-weighted RSI.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and volume data.
        period (int, optional): Period for calculating the MFI. Default is 14.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        volume_column (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.Series: A pandas Series containing the MFI values.
    """
    high, low, close, volume = df[high_column], df[low_column], df[close_column], df[volume_column]

    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume

    positive_mf = pd.Series(np.where(typical_price > typical_price.shift(1), raw_mf, 0), index=df.index)
    negative_mf = pd.Series(np.where(typical_price < typical_price.shift(1), raw_mf, 0), index=df.index)

    pos_mf_sum = positive_mf.rolling(window=period, min_periods=period).sum()
    neg_mf_sum = negative_mf.rolling(window=period, min_periods=period).sum()

    mfr = pos_mf_sum / neg_mf_sum
    mfi = 100 - (100 / (1 + mfr))

    return mfi


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('NVDA', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    mfi = money_flow_index(df)

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
            y=mfi,
            x=mfi.index,
            mode='lines',
            line=dict(color='grey'),
            name='MFI'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[80] * len(mfi),
            x=mfi.index,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Overbought 80'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[20] * len(mfi),
            x=mfi.index,
            mode='lines',
            line=dict(color='lime', dash='dash'),
            name='Overbought 20'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[50] * len(mfi),
            x=mfi.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Neutral 50'
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
