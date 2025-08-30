import pandas as pd


def schaff_trend_cycle(
        df: pd.DataFrame,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle_period: int = 10,
        smoothing_period1: int = 10,
        smoothing_period2: int = 10,
        column: str = "Close"
    ) -> pd.Series:
    """
    Calculates the Schaff Trend Cycle (STC) for a given DataFrame.
    The STC is a trend-following indicator that combines the concepts of the MACD and the Stochastic Oscillator to
    identify market trends and potential turning points more quickly. It is designed to generate faster and more
    accurate signals than traditional trend indicators.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        fast_period (int, optional): Period for the fast EMA in the MACD calculation. Default is 23.
        slow_period (int, optional): Period for the slow EMA in the MACD calculation. Default is 50.
        cycle_period (int, optional): Period for the stochastic calculations. Default is 10.
        smoothing_period1 (int, optional): Smoothing period for the first EMA applied to the stochastic. Default is 10.
        smoothing_period2 (int, optional): Smoothing period for the second EMA applied to the stochastic. Default is 10.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the STC values.
    """
    close = df[column]
    
    # Calculate MACD
    ema_fast = close.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    # Calculate %K stochastic over MACD
    lowest_macd = macd.rolling(window=cycle_period, min_periods=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period, min_periods=cycle_period).max()
    stoch_k = 100 * (macd - lowest_macd) / (highest_macd - lowest_macd)
    
    # Calculate %D
    stoch_d = stoch_k.ewm(span=smoothing_period1, min_periods=smoothing_period1, adjust=False).mean()
    stoch_d_min = stoch_d.rolling(window=cycle_period, min_periods=cycle_period).min()
    stoch_d_max = stoch_d.rolling(window=cycle_period, min_periods=cycle_period).max()
    
    stoch_kd = 100 * (stoch_d - stoch_d_min) / (stoch_d_max - stoch_d_min)

    # Smooth using EMA
    stc = stoch_kd.ewm(span=smoothing_period2, min_periods=smoothing_period2, adjust=False).mean()

    return stc


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    stc = schaff_trend_cycle(df)

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
            y=stc,
            x=stc.index,
            mode='lines',
            line=dict(color='coral'),
            name='STC'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[75] * len(stc),
            x=stc.index,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Bearish Trend Cycle'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[25] * len(stc),
            x=stc.index,
            mode='lines',
            line=dict(color='lime', dash='dash'),
            name='Bullish Trend Cycle'
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
