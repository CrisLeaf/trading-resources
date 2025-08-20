import pandas as pd


def accumulation_distribution_index(
        df: pd.DataFrame,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> pd.Series:
    """
    Calculates the Accumulation/Distribution Index (ADI) for a given DataFrame.
    The ADI is a volume-based indicator designed to measure the cumulative flow of money into and out of a security. It helps identify divergences between price and volume, signaling potential trend reversals or confirmations.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and volume data.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
        volume_column (str, optional): Name of the column containing volume data. Default is 'Volume'.

    Returns:
        pd.Series: A pandas Series containing the ADI values.
    """
    high, low, close, volume = df[high_column], df[low_column], df[close_column], df[volume_column]
    
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_volume = money_flow_multiplier * volume
    adi = money_flow_volume.cumsum()

    return adi
