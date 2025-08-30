import pandas as pd

def standard_deviation(
        df: pd.DataFrame,
        column: str = 'Close',
        period: int = 14,
        ddof: int = 0
    ) -> pd.Series:
    """
    Calculates the rolling standard deviation for a given column in the DataFrame.
    Standard deviation is a statistical measure of volatility, indicating how much the values of a dataset deviate from
    the mean. In trading, it is commonly used to assess the volatility of asset prices over a specified period.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        column (str, optional): Name of the column for which to calculate the standard deviation. Default is 'Close'.
        period (int, optional): Period for calculating the rolling standard deviation. Default is 14.
        ddof (int, optional): Delta degrees of freedom for the calculation. Default is 0.

    Returns:
        pd.Series: A pandas Series containing the rolling standard deviation values.
    """
    close = df[column]

    return close.rolling(window=period, min_periods=period).std(ddof=ddof)


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    std_dev = standard_deviation(df)

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
            y=std_dev,
            x=std_dev.index,
            mode='lines',
            line=dict(color='skyblue'),
            name='Standard Deviation'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[std_dev.mean()] * len(std_dev),
            x=std_dev.index,
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='Standard Deviation Mean'
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
