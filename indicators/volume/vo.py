import pandas as pd


def volume_oscillator(
        df: pd.DataFrame,
        fast_period: int = 5,
        slow_period: int = 20,
        signal_period: int = 9,
        column_volume: str = 'Volume'
    ) -> pd.DataFrame:
    """
    Calculates the Volume Oscillator (VO) for a given DataFrame.
    The Volume Oscillator measures the difference between two moving averages of volume, helping to identify changes in
    market activity and potential trend reversals. It consists of the Volume Oscillator line and a signal line.

    Args:
        df (pd.DataFrame): Input DataFrame containing volume data.
        fast_period (int, optional): Period for the fast exponential moving average (EMA) of volume. Default is 5.
        slow_period (int, optional): Period for the slow exponential moving average (EMA) of volume. Default is 20.
        signal_period (int, optional): Period for the signal line EMA. Default is 9.
        column_volume (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.DataFrame: DataFrame with columns for Volume Oscillator and VO Signal.
    """
    volume = df[column_volume]

    vol_fast = volume.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    vol_slow = volume.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()


    vo = (vol_fast - vol_slow) / vol_slow * 100
    vo_signal = vo.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    
    return pd.DataFrame({
        'Volume Oscillator': vo,
        'VO Signal': vo_signal
    })


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('NVDA', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    vo = volume_oscillator(df)

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
            y=vo['Volume Oscillator'],
            x=vo.index,
            mode='lines',
            line=dict(color='skyblue'),
            name='Volume Oscillator'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=vo['VO Signal'],
            x=vo.index,
            mode='lines',
            line=dict(color='red'),
            name='VO Signal'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=[0] * len(vo),
            x=vo.index,
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
