import pandas as pd
import numpy as np


def ulcer_index(
        df: pd.DataFrame,
        column: str = 'Close',
        period: int = 14
    ) -> pd.Series:
    """
    Calculates the Ulcer Index (UI) for a given DataFrame.
    The Ulcer Index is a volatility indicator that measures the depth and duration of price drawdowns from recent
    highs over a specified period. It is used to assess downside risk and the volatility of an asset.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        column (str, optional): Name of the column containing price data. Default is 'Close'.
        period (int, optional): Period for calculating the Ulcer Index. Default is 14.

    Returns:
        pd.Series: A pandas Series containing the Ulcer Index values.
    """
    close = df[column]

    rolling_max = close.rolling(window=period, min_periods=period).max()
    drawdown_pct = ((close - rolling_max) / rolling_max) * 100

    ui = np.sqrt((drawdown_pct.pow(2)).rolling(window=period, min_periods=period).mean())
    
    return ui


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    ui = ulcer_index(df)

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
            y=ui,
            x=ui.index,
            mode='lines',
            line=dict(color='coral'),
            name='Ulcer Index'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[5] * len(ui),
            x=ui.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Risk Threshold'
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
