import numpy as np
import pandas as pd

    
def directional_movement_index(
        df: pd.DataFrame,
        adx_period: int = 14,
        di_period: int = 14,
        high_column: str = 'High',
        low_column: str = 'Low',
        close_column: str = 'Close'
    ) -> pd.DataFrame:
    """
    Calculates the Directional Movement Index (DMI) and related indicators (ADX, +DI, -DI) for a given DataFrame.
    The DMI is a technical indicator used to assess the strength and direction of a trend in price data. It consists of the Positive Directional Indicator (+DI), Negative Directional Indicator (-DI), and the Average Directional Index (ADX).
    
    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        adx_period (int, optional): Period for calculating the Average Directional Index (ADX). Default is 14.
        di_period (int, optional): Period for calculating the Directional Indicators (+DI, -DI). Default is 14.
        high_column (str, optional): Name of the column containing high prices. Default is 'High'.
        low_column (str, optional): Name of the column containing low prices. Default is 'Low'.
        close_column (str, optional): Name of the column containing close prices. Default is 'Close'.
    
    Returns:
        pd.DataFrame: DataFrame with additional columns for +DI, -DI, and ADX.
    """
    # Calculate True Range
    high, low, close = df[high_column], df[low_column], df[close_column]
    h_minus_l = high - low
    prev_clo = close.shift(periods=1)
    h_minus_pc = abs(high - prev_clo)
    l_minus_pc = abs(low - prev_clo)
    tr = pd.Series(
        np.max([h_minus_l, h_minus_pc, l_minus_pc], axis=0),
        index=df.index,
        name='TR'
    )
    
    # Calculate Directional Movements (+DM and -MD)
    pre_pdm = high.diff().dropna()
    pre_mdm = low.diff(periods=-1).dropna()
    plus_dm = pre_pdm.where((pre_pdm > pre_mdm.values) & (pre_pdm > 0), 0)
    minus_dm = pre_mdm.where((pre_mdm > pre_pdm.values) & (pre_mdm > 0), 0)
    
    # Calculates initial values for smoothed sums of TR, +DM, and -DM
    trl = [np.nansum(tr[: adx_period + 1])]
    pdml = [plus_dm[: adx_period].sum()]
    mdml = [minus_dm[: adx_period].sum()]
    factor = 1 - 1/adx_period
    
    # Calculate smoothed sums of TR, +DM, and -DM using Wilder's method
    for i in range(0, int(df.shape[0] - adx_period - 1)):
        trl.append(trl[i] * factor + tr.iloc[adx_period + i + 1])
        pdml.append(pdml[i] * factor + plus_dm.iloc[adx_period + i])
        mdml.append(mdml[i] * factor + minus_dm.iloc[adx_period + i])

    # Calculate +DI and -DI
    pdi = np.array(pdml) / np.array(trl) * 100
    mdi = np.array(mdml) / np.array(trl) * 100
    
    # Calculate DX and ADX
    dx = np.abs(pdi - mdi) / (pdi + mdi) * 100
    adx = [dx[: adx_period].mean()]
    
    # Calculate ADX using di_period
    _ = [
        adx.append((adx[i] * (di_period - 1) + dx[di_period + i])/di_period)
        for i in range(int(len(dx) - di_period))
    ]
    adxi = pd.DataFrame(pdi, columns=['+DI'], index=df.index[-len(pdi): ])
    adxi['-DI'] = mdi
    adx = pd.DataFrame(adx, columns=['ADX'], index=df.index[-len(adx): ])
    
    return adx.merge(adxi, how='outer', left_index=True, right_index=True)
