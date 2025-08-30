import pandas as pd


def exponential_moving_average(
        df: pd.DataFrame,
        period: int = 26,
        column: str = 'Close'
    ) -> pd.Series:
    """
    The Exponential Moving Average (EMA) is a type of moving average that gives more weight to recent data points,
    making it more responsive to new information compared to the Simple Moving Average (SMA). It is commonly used in
    time series analysis and financial markets to smooth out price data and identify trends more quickly.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        column (str): The column name for which to calculate the EMA.
        period (int): The period size for the EMA.

    Returns:
        pd.Series: A pandas Series containing the EMA values.

    """
    return df[column].ewm(span=period, min_periods=period, adjust=False).mean()



if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    df['EMA_26'] = exponential_moving_average(df, period=26)

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
        y=df['EMA_26'],
        mode='lines',
        line=dict(color='orange', width=1.5),
        name='EMA 26'
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