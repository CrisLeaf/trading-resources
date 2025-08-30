import numpy as np
import pandas as pd


def cci_index(
        df: pd.DataFrame,
        period: int = 20,
        constant: float = 0.015,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        
    ) -> pd.Series:
    """
    Calculates the Commodity Channel Index (CCI) for a given DataFrame.
    The CCI is a technical indicator used to identify overbought and oversold levels in the price of an asset. It
    evaluates the direction and strength of the price trend, helping traders determine when to enter or exit a trade.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the CCI. Default is 20.
        constant (float, optional): Constant used in the CCI calculation (typically 0.015). Default is 0.015.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the CCI values.
    """
    high, low, close = df[high_column], df[low_column], df[close_column]
    tp = (high + low + close) / 3
    tp_rolling = tp.rolling(window=period, min_periods=period)
    mean_dev = tp_rolling.apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp_rolling.mean()) / (constant * mean_dev)
    cci.name = f'CCI_{period}'
    
    return cci


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    cci = cci_index(df)

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
    fig.add_trace(go.Scatter(
        x=cci.index,
        y=cci,
        mode='lines',
        line=dict(color='white', width=1.5),
        name='CCI 20'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=cci.index,
        y=[100]*len(cci),
        mode='lines',
        name='Overbought',
        line=dict(color='lime', dash='dash')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=cci.index,
        y=[-100]*len(cci),
        mode='lines',
        name='Oversold',
        line=dict(color='red', dash='dash')
    ), row=2, col=1)

    
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