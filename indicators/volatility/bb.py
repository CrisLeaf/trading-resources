import pandas as pd


def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        k: float = 2.0,
        ddof: int = 0,
        column: str = "Close"
    ) -> pd.DataFrame:
    """
    Calculates the Bollinger Bands for a given DataFrame.
    Bollinger Bands are a volatility indicator consisting of a middle band (simple moving average), an upper band, and a lower band. The upper and lower bands are calculated as a specified number of standard deviations above and below the moving average, helping to identify overbought and oversold conditions.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        period (int, optional): Period for calculating the moving average and standard deviation. Default is 20.
        k (float, optional): Number of standard deviations to set the bands. Default is 2.0.
        ddof (int, optional): Delta degrees of freedom for the standard deviation calculation. Default is 0.
        column (str, optional): Name of the column containing price data. Default is 'Close'.

    Returns:
        pd.DataFrame: DataFrame with columns for Bollinger_Mid, Bollinger_Upper, and Bollinger_Lower.
    """
    close = df[column]

    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=ddof)

    upper_band = sma + (k * std)
    lower_band = sma - (k * std)

    return pd.DataFrame({
        "Bollinger_Mid": sma,
        "Bollinger_Upper": upper_band,
        "Bollinger_Lower": lower_band
    })
