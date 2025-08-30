import numpy as np
import pandas as pd

def nadaraya_watson_envelope(
        df: pd.DataFrame,
        bandwidth: float = 5.0,
        factor: float = 1.5,
        period: int = None,
        column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Nadaraya-Watson Envelope for a given DataFrame.
    The Nadaraya-Watson Envelope is an advanced smoothing technique that uses kernel regression to estimate a smoothed
    price series and constructs dynamic upper and lower bands based on the mean absolute error (MAE). It helps identify
    price deviations and potential reversal points.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        bandwidth (float, optional): Bandwidth parameter for the kernel regression. Default is 5.0.
        factor (float, optional): Multiplier for the MAE to set the envelope width. Default is 1.5.
        period (int, optional): Number of periods to use for the calculation. If None, uses the entire series. Default
        is None.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for NW_Smooth, Upper Band, Lower Band, and Bands Direction.
    """
    close = df[column]

    n = len(close) if period is None else period
    price_series = close[-n:].values
    idx = close.index[-n:]

    # Create weights matrix
    rows = np.arange(n)
    distances = rows[:, None] - rows[None, :]
    weights_matrix = np.exp(-0.5 * (distances / bandwidth)**2)
    weights_matrix /= weights_matrix.sum(axis=1)[:, None]
    
    # Nadaraya-Watson estimator
    estimator = weights_matrix @ price_series
    
    # Envelope (MAE * factor)
    mae = np.mean(np.abs(price_series - estimator)) * factor
    upper_band = estimator + mae
    lower_band = estimator - mae
    
    # Bands Direction
    price_shift = np.roll(price_series, 1)
    bands_dir = np.full_like(price_series, np.nan, dtype=float)
    bands_dir[(price_shift < (estimator - mae)) & (price_series > (estimator - mae))] = 1
    bands_dir[(price_shift > (estimator + mae)) & (price_series < (estimator + mae))] = -1
    bands_dir = np.roll(bands_dir, -1)  # shift forward
    
    return pd.DataFrame({
        'NW_Smooth': estimator,
        'Upper Band': upper_band,
        'Lower Band': lower_band,
        'Bands Direction': bands_dir
    }, index=idx)


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    nw_env = nadaraya_watson_envelope(df)

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
            x=nw_env.index,
            y=nw_env['Upper Band'],
            mode='lines',
            line=dict(color='red', width=2),
            name='NW Upper Band'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=nw_env.index,
            y=nw_env['Lower Band'],
            mode='lines',
            line=dict(color='green', width=2),
            name='NW Lower Band'
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
