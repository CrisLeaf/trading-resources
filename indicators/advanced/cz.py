import numpy as np
import pandas as pd


def chop_zone(
        df: pd.DataFrame,
        period: int = 30,
        ema_period: int = 34,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.Series:
    """
    Calculates the Chop Zone indicator for a given DataFrame.
    The Chop Zone is an advanced technical indicator designed to identify choppy or trending market conditions by
    analyzing the angle of the EMA relative to price action. It categorizes the market into different zones based on
    the calculated angle, helping traders assess the current market environment.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for smoothing the price range. Default is 30.
        ema_period (int, optional): Period for calculating the exponential moving average (EMA). Default is 34.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.Series: A pandas Series containing the Chop Zone values, indicating the market zone classification.
    """
    high, low, close = df[high_column], df[low_column], df[close_column]
    
    tp = (high + low + close) / 3
    price_smooth = close.rolling(window=period, min_periods=period)
    max_smooth = price_smooth.max()
    min_smooth = price_smooth.min()
    hl_range = 25 / (max_smooth - min_smooth) * min_smooth
    
    ema = close.ewm(span=ema_period, min_periods=ema_period, adjust=False).mean()
    x1_ema = 0
    x2_ema = 1
    y1_ema = 0
    y2_ema = (ema.shift(periods=1) - ema) / tp * hl_range
    c_ema = np.sqrt((x2_ema - x1_ema) ** 2 + (y2_ema - y1_ema) ** 2)
    angle_ema0 = round(np.rad2deg(np.arccos((x2_ema - x1_ema) / c_ema)), 2)
    angle_ema1 = np.where(y2_ema > 0, -angle_ema0, angle_ema0)[max(period, ema_period): ]
    
    chop_zone_conditions = [
        angle_ema1 >= 5,
        (angle_ema1 >= 3.57) & (angle_ema1 < 5),
        (angle_ema1 >= 2.14) & (angle_ema1 < 3.57),
        (angle_ema1 >= 0.71) & (angle_ema1 < 2.14),
        angle_ema1 <= -5,
        (angle_ema1 <= -3.57) & (angle_ema1 > -5),
        (angle_ema1 <= -2.14) & (angle_ema1 > -3.57),
        (angle_ema1 <= -0.71) & (angle_ema1 > -2.14)
    ]
    choices = [0, 1, 2, 3, 4, 5, 6, 7]
    default_value = 8
    chop_zone = np.select(chop_zone_conditions, choices, default=default_value)

    return pd.Series([np.nan] * max(period, ema_period) + list(chop_zone), index=df.index, name='Chop Zone')


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    cz = chop_zone(df)

    # Plots
    colors = {
        0: "#26C6DA",  # Turquoise - Strong Bullish Trend
        1: "#43A047",  # Dark Green - Moderate Bullish Trend
        2: "#A5D6A7",  # Light Green - Mild Bullish Trend
        3: "#009688",  # Teal - Weak Bullish Trend
        4: "#D50000",  # Dark Red - Strong Bearish Trend
        5: "#E91E63",  # Red - Moderate Bearish Trend
        6: "#FF6D00",  # Orange - Mild Bearish Trend
        7: "#FFB74D",  # Light Orange - Weak Bearish Trend
        8: "#FDD835"   # Yellow - Neutral / No Clear Direction
    }
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
    cz.fillna(8, inplace=True)
    for i, value in enumerate(cz):
        fig.add_trace(go.Bar(
            x=[cz.index[i]],
            y=[1],
            marker_color=colors[int(value)],
            showlegend=False,
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
