import numpy as np
import pandas as pd


def nadaraya_watson_estimator(
        df: pd.DataFrame,
        column: str = 'Close',
        period: int = None,
        bandwidth: float = 5.0
    ) -> pd.DataFrame:
    """
    Calculates the Nadaraya-Watson Estimator for a given DataFrame.
    The Nadaraya-Watson Estimator is a non-parametric regression technique that uses kernel smoothing (typically
    Gaussian) to estimate a smoothed price series. It can help identify the underlying trend and turning points in
    the data by analyzing the direction of the estimator.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        column (str, optional): Name of the column containing price data. Default is 'Close'.
        period (int, optional): Number of periods to use for the calculation. If None, uses the entire series.
                                Default is None.
        bandwidth (float, optional): Bandwidth parameter for the kernel regression. Default is 5.0.

    Returns:
        pd.DataFrame: DataFrame with columns for NW_Estimator and Direction_Estimator.
    """
    close = df[column]
    
    n = len(close) if period is None else period
    price_series = close[-n:].values
    idx = close.index[-n:]

    # Weights Matrix (Gaussian Kernel)
    rows = np.arange(n)
    distances = rows[:, None] - rows[None, :]
    weights_matrix = np.exp(-0.5 * (distances / bandwidth)**2)
    weights_matrix /= weights_matrix.sum(axis=1)[:, None]

    # NW Estimator
    estimator = weights_matrix @ price_series
    nwe = pd.Series(estimator, index=idx, name='NW_Estimator')
    
    # Direction
    direction = nwe.diff()
    direction_shift = direction.shift(1)
    
    direction_estimator = np.where(
        (direction > 0) & (direction_shift < 0), 1,
        np.where(
            (direction < 0) & (direction_shift > 0), -1, np.nan
        )
    )

    return pd.DataFrame({
        "NW_Estimator": nwe,
        "Direction_Estimator": direction_estimator
    }, index=idx)


if __name__ == '__main__':
    import yfinance as yf
    import plotly.graph_objects as go

    df = yf.download('USDCLP=X', start='2024-01-01')
    df.columns = df.columns.droplevel(1)

    nw_est = nadaraya_watson_estimator(df)
    nw_est['Direction_Estimator'] = nw_est['Direction_Estimator'].ffill()
    nw_est['Direction_Estimator'] = nw_est['Direction_Estimator'].fillna(
        nw_est['Direction_Estimator'].dropna().iloc[0] * -1,
    )

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
            x=nw_est.index,
            y=nw_est['NW_Estimator'].where(nw_est['Direction_Estimator'] == 1),
            mode='lines',
            line=dict(color='green', width=2),
            name='NW Estimator Bullish'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=nw_est.index,
            y=nw_est['NW_Estimator'].where(nw_est['Direction_Estimator'] == -1),
            mode='lines',
            line=dict(color='red', width=2),
            name='NW Estimator Bearish'
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
