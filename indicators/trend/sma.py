import pandas as pd


def simple_moving_average(
        df: pd.DataFrame,
        period: int = 21,
        column: str = 'Close'
    ) -> pd.Series:
    """
    The Simple Moving Average (SMA) is a statistical calculation that computes the average of a selected range of prices, usually closing prices, by the number of periods in that range. It smooths out price data to identify trends over time.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        period (int): The period size for the SMA.
        column (str): The column name for which to calculate the SMA.

    Returns:
        pd.Series: A pandas Series containing the SMA values.
    """
    return df[column].rolling(window=period, min_periods=period).mean()


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    df['SMA_21'] = df['Close'].rolling(21).mean()

    # Plots
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        line=dict(color='skyblue', width=1),
        name='Close'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_21'],
        mode='lines',
        line=dict(color='orange', width=1.5),
        name='SMA 21'
    ))
    fig.update_layout(
        template='plotly_dark',
        title='Signals Plot',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgb(20, 20, 20)',
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=600,
        width=1000
    )
    
    fig.show()