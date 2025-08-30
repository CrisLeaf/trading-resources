import numpy as np
import pandas as pd


def on_balance_volume(
        df: pd.DataFrame,
        sma_period: int = None,
        close_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> pd.DataFrame:
    """
    Calculates the On-Balance Volume (OBV) for a given DataFrame.
    OBV is a volume-based indicator that measures buying and selling pressure as a cumulative total.
    It adds volume on up days and subtracts volume on down days, helping to identify momentum
    and potential trend reversals based on volume flow.

    Optionally, a simple moving average (SMA) of the OBV can be calculated to smooth the series.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and volume data.
        sma_period (int, optional): Period for calculating the OBV SMA. Default is None (no SMA).
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        volume_column (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.DataFrame: A DataFrame containing the OBV values and, if requested, the OBV SMA.
    """
    close, volume = df[close_column], df[volume_column]

    obv = np.sign(close.diff(periods=1)) * volume
    obv = obv.fillna(value=volume).cumsum()
    output_df = pd.DataFrame({'OBV': obv})

    if sma_period is not None and sma_period > 1:
        obv_sma = obv.rolling(window=sma_period, min_periods=sma_period).mean()
        output_df[f'OBV_SMA_{sma_period}'] = obv_sma

    return output_df


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('NVDA', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    obv = on_balance_volume(df, sma_period=20)

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
            y=obv['OBV'],
            x=obv.index,
            mode='lines',
            line=dict(color='skyblue'),
            name='OBV'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=obv['OBV_SMA_20'],
            x=obv.index,
            mode='lines',
            line=dict(color='green'),
            name='OBV_SMA_20'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[0] * len(obv),
            x=obv.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='0 line'
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
