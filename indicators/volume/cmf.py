import pandas as pd


def chaikin_money_flow(
        df: pd.DataFrame,
        period: int = 20,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> pd.Series:
    """
    Calculates the Chaikin Money Flow (CMF) indicator for a given DataFrame.
    The CMF is a volume-based indicator that measures the accumulation and distribution of an asset over a specified period. It helps identify buying and selling pressure by combining price and volume data.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and volume data.
        period (int, optional): Period for calculating the CMF. Default is 20.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        volume_column (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.Series: A pandas Series containing the CMF values.
    """
    high, low, close, volume = df[high_column], df[low_column], df[close_column], df[volume_column]

    # Prevent division by zero
    denominator = high - low
    denominator = denominator.replace(0, 1e-10)

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / denominator

    # Money Flow Volume
    mfv = mfm * volume

    # CMF
    cmf_numerator = mfv.rolling(window=period, min_periods=period).sum()
    cmf_denominator = volume.rolling(window=period, min_periods=period).sum()
    cmf = cmf_numerator / cmf_denominator

    return cmf
