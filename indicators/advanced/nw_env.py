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
    The Nadaraya-Watson Envelope is an advanced smoothing technique that uses kernel regression to estimate a smoothed price series and constructs dynamic upper and lower bands based on the mean absolute error (MAE). It helps identify price deviations and potential reversal points.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        bandwidth (float, optional): Bandwidth parameter for the kernel regression. Default is 5.0.
        factor (float, optional): Multiplier for the MAE to set the envelope width. Default is 1.5.
        period (int, optional): Number of periods to use for the calculation. If None, uses the entire series. Default is None.
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
