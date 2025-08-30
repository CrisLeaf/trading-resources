import pandas as pd


def weighted_moving_average(
        df: pd.DataFrame,
        period: int = 9,
        column: str = 'Close'
    ) -> pd.Series:
    """
        Calculates the Weighted Moving Average (WMA) for a specified column in a pandas DataFrame.
        The WMA assigns more weight to recent data points, making it more responsive to new information compared to a
        simple moving average.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            period (int, optional): The number of periods to use for calculating the WMA. Default is 9.
            column (str, optional): The name of the column to calculate the WMA on. Default is 'Close'.
        Returns:
            pd.Series: A pandas Series containing the weighted moving average values.
    """
    weights = pd.Series(range(1, period + 1), index=range(period))

    return df[column].rolling(window=period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    df['WMA_9'] = weighted_moving_average(df, period=9)

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
        y=df['WMA_9'],
        mode='lines',
        line=dict(color='orange', width=1.5),
        name='WMA 9'
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