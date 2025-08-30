import numpy as np
import pandas as pd


def squeeze_momentum(
        df: pd.DataFrame,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        momentum_period: int = 12,
        momentum_longitude: int = 6,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Squeeze Momentum indicator for a given DataFrame.
    The Squeeze Momentum indicator combines Bollinger Bands and Keltner Channels to identify periods of low volatility
    ("squeeze") and potential breakouts. It also calculates a momentum value to help determine the direction of the
    breakout.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        bb_period (int, optional): Period for calculating the Bollinger Bands. Default is 20.
        bb_std_dev (float, optional): Standard deviation multiplier for the Bollinger Bands. Default is 2.0.
        kc_period (int, optional): Period for calculating the Keltner Channel. Default is 20.
        kc_mult (float, optional): Multiplier for the Keltner Channel bands. Default is 1.5.
        momentum_period (int, optional): Period for calculating the momentum. Default is 12.
        momentum_longitude (int, optional): Period for smoothing the momentum. Default is 6.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        column_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for SQZ (momentum), SQZ_ON (squeeze on), SQZ_OFF (squeeze off), and
                      NO_SQZ (no squeeze).
    """
    high, low, close = df[high_column], df[low_column], df[close_column]
    
    # Bollinger Bands
    rolling = close.rolling(window=bb_period, min_periods=bb_period)
    bb_ma = rolling.mean()
    bb_std = rolling.std(ddof=0)
    bb_upper = bb_ma + bb_std_dev * bb_std
    bb_lower = bb_ma - bb_std_dev * bb_std

    # Keltner Channel
    ema = close.ewm(span=kc_period, min_periods=kc_period, adjust=False).mean()
    
    # TR
    high, low = df[high_column], df[low_column]
    prev_close = close.shift(1)
    tr = np.maximum.reduce([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ])
    tr = pd.Series(tr, index=df.index)
    tr_ema = tr.ewm(span=kc_period, min_periods=kc_period, adjust=True).mean()

    kc_upper = ema + kc_mult * tr_ema
    kc_lower = ema - kc_mult * tr_ema

    # Squeeze Momentum
    squeeze = close.diff(periods=momentum_period)
    squeeze = squeeze.rolling(window=momentum_longitude, min_periods=momentum_longitude).mean()
    
    # Conditions
    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
    no_sqz = ~(sqz_on | sqz_off)

    return pd.DataFrame({
        'SQZ': squeeze,
        'SQZ_ON': sqz_on,
        'SQZ_OFF': sqz_off,
        'NO_SQZ': no_sqz
    }, index=df.index)


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    sqz_m = squeeze_momentum(df)

    # Plots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3]
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            line=dict(color='skyblue', width=1),
            name='Close'
        ), row=1, col=1
    )

    # Signals
    colors = ['green' if val > 0 else 'red' for val in sqz_m['SQZ']]

    fig.add_trace(
        go.Bar(
            x=sqz_m.index,
            y=sqz_m['SQZ'],
            marker_color=colors,
            showlegend=False,
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        title='Signals Plot',
        xaxis2_title='Date',
        yaxis_title='Price',
        yaxis2_title='SQZ Momentum',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgb(20, 20, 20)',
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        height=900,
        width=1000
    )

    fig.show()

