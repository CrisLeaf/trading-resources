import numpy as np
import pandas as pd


def super_trend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the SuperTrend indicator for a given DataFrame.
    The SuperTrend is a trend-following indicator that uses the Average True Range (ATR) to determine dynamic support
    and resistance levels. It helps identify the current trend direction and potential entry or exit points.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the ATR. Default is 10.
        multiplier (float, optional): Multiplier for the ATR to set the band distance. Default is 3.0.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for SuperTrend, ST_Direction, UpperBand, and LowerBand.
    """
    high, low, close = df[high_column], df[low_column], df[close_column]
    
    # Calculate ATR using EMA
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Classic Bands
    hl2 = ((high + low) / 2).shift(1)
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Initialization
    st = np.zeros(len(df), dtype=float)
    direction = np.ones(len(df), dtype=int)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    # Iteration
    for i in range(1, len(df)):
        # Determine direction
        if close.iloc[i] > final_upper.iloc[i-1]:
            direction[i] = 1
        elif close.iloc[i] < final_lower.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        # Adjust final bands based on trend
        if direction[i] == 1:
            final_lower.iloc[i] = max(final_lower.iloc[i], final_lower.iloc[i-1])
            final_upper.iloc[i] = np.nan
            st[i] = final_lower.iloc[i]
        else:
            final_upper.iloc[i] = min(final_upper.iloc[i], final_upper.iloc[i-1])
            final_lower.iloc[i] = np.nan
            st[i] = final_upper.iloc[i]

    return pd.DataFrame({
        'SuperTrend': st,
        'ST_Direction': direction,
        'UpperBand': final_upper,
        'LowerBand': final_lower
    }, index=df.index)


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    st = super_trend(df)

    # Plots
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        line=dict(color='skyblue', width=1),
        name='Close'
    ))
    fig.add_trace(
        go.Scatter(
            x=st.index,
            y=st['UpperBand'],
            mode='lines',
            line=dict(color='red', width=2),
            name='ST Upper Band'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=st.index,
            y=st['LowerBand'],
            mode='lines',
            line=dict(color='green', width=2),
            name='ST Lower Band'
        )
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
        height=600,
        width=1000
    )

    fig.show()
