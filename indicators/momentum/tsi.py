import pandas as pd


def true_strength_index(
        df: pd.DataFrame,
        ema1_period: int = 25,
        ema2_period: int = 13,
        signal_period: int = 7,
        column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the True Strength Index (TSI) for a given DataFrame.
    The TSI is a momentum oscillator used to identify the strength and direction of a trend. It applies double
    smoothing to price momentum, helping traders spot trend changes and potential buy or sell signals.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        ema1_period (int, optional): Period for the first exponential moving average (EMA). Default is 25.
        ema2_period (int, optional): Period for the second exponential moving average (EMA). Default is 13.
        signal_period (int, optional): Period for the signal line EMA. Default is 7.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for TSI, Signal, and Trend.
    """
    momentum = df[column].diff(periods=1)

    ema1_momentum = momentum.ewm(span=ema2_period, min_periods=ema2_period, adjust=False).mean()
    ema2_momentum = ema1_momentum.ewm(span=ema1_period, min_periods=ema1_period, adjust=False).mean()
    
    momentum_abs = abs(momentum)
    ema1_momentum_abs = momentum_abs.ewm(span=ema2_period, min_periods=ema2_period, adjust=False).mean()
    ema2_momentum_abs = ema1_momentum_abs.ewm(span=ema1_period, min_periods=ema1_period, adjust=False).mean()

    pre_tsi_df = 100 * (ema2_momentum / ema2_momentum_abs)
    signal_df = pre_tsi_df.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()

    tsi_df = pd.DataFrame({
        'TSI': pre_tsi_df,
        'Signal': signal_df
    })

    tsi_df['Trend'] = tsi_df['TSI'] > tsi_df['Signal']
    
    return tsi_df


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    tsi = true_strength_index(df)

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
        x=tsi.index,
        y=tsi['Signal'],
        mode='lines',
        name='Signal',
        line=dict(color='red', width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tsi.index,
        y=tsi['TSI'],
        mode='lines',
        name='TSI',
        line=dict(color='lime', width=1.5)
    ), row=2, col=1)

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
