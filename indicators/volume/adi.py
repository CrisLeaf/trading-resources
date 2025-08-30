import pandas as pd


def accumulation_distribution_index(
        df: pd.DataFrame,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> pd.Series:
    """
    Calculates the Accumulation/Distribution Index (ADI) for a given DataFrame.
    The ADI is a volume-based indicator designed to measure the cumulative flow of money into and out of a security.
    It helps identify divergences between price and volume, signaling potential trend reversals or confirmations.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and volume data.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        volume_column (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.Series: A pandas Series containing the ADI values.
    """
    high, low, close, volume = df[high_column], df[low_column], df[close_column], df[volume_column]
    
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_volume = money_flow_multiplier * volume
    adi = money_flow_volume.cumsum()

    return adi


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('NVDA', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    adi = accumulation_distribution_index(df)

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
            y=adi,
            x=adi.index,
            mode='lines',
            line=dict(color='lime'),
            name='Accumulation Distribution Index'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[0] * len(adi),
            x=adi.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
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

