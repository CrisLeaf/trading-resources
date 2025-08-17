import numpy as np
import pandas as pd


def turtle_trading_channels(
        df: pd.DataFrame,
        enter_period: int = 20,
        exit_period: int = 10,
        high_column: str = 'High',
        low_column: str = 'Low'
    ) -> pd.DataFrame:
    high, low = df[high_column], df[low_column]

    # Entry and exit levels
    enter_low = low.rolling(window=enter_period, min_periods=enter_period).min()
    enter_high = high.rolling(window=enter_period, min_periods=enter_period).max()
    exit_low = low.rolling(window=exit_period, min_periods=exit_period).min()
    exit_high = high.rolling(window=exit_period, min_periods=exit_period).max()
    
    # Break Signals
    buy_signal = (high == enter_high.shift(1)) | ((high > enter_high.shift(1)) & (high.shift(1) < enter_high.shift(2)))
    sell_signal = (low == enter_low.shift(1)) | ((low < enter_low.shift(1)) & (low.shift(1) > enter_low.shift(2)))
    buy_exit = (low == exit_low.shift(1)) | ((low < exit_low.shift(1)) & (low.shift(1) > exit_low.shift(2)))
    sell_exit = (high == exit_high.shift(1)) | ((high > exit_high.shift(1)) & (high.shift(1) < exit_high.shift(2)))
    
    # Displaced Signals
    buy_signal_shift = buy_signal.shift(1)
    sell_signal_shift = sell_signal.shift(1)
    buy_exit_shift = buy_exit.shift(1)
    sell_exit_shift = sell_exit.shift(1)
    
    # Bar counts
    def bar_count(signal):
        mask = signal.values.astype(bool)
        arr = np.arange(len(mask))
        last_idx = np.where(mask, arr, -1)
        last_signal = pd.Series(last_idx).where(pd.Series(mask)).ffill().fillna(-1).values

        return arr - last_signal

    # Counters
    o1 = bar_count(buy_signal)
    o2 = bar_count(sell_signal)
    o3 = bar_count(buy_exit)
    o4 = bar_count(sell_exit)
    e1 = bar_count(buy_signal_shift)
    e2 = bar_count(sell_signal_shift)
    e3 = bar_count(buy_exit_shift)
    e4 = bar_count(sell_exit_shift)
    
    signals_df = pd.DataFrame({
        'o1': o1, 'o2': o2, 'o3': o3, 'o4': o4,
        'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4
    }, index=df.index)
    
    # Trend and Exit Lines
    condition = o1 <= o2
    trend_line = np.where(condition, enter_low, enter_high)
    exit_line = np.where(condition, exit_low, exit_high)
    
    # Final Signals
    tdc = pd.DataFrame(index=df.index)
    tdc['Upper'] = enter_high
    tdc['Lower'] = enter_low
    tdc['Trend Line'] = trend_line
    tdc['Exit Line'] = exit_line
    tdc['Buy Signal'] = np.where(buy_signal & (o3 < np.roll(o1, 1)), enter_low, np.nan)
    tdc['Sell Signal'] = np.where(sell_signal & (o4 < np.roll(o2, 1)), enter_high, np.nan)
    tdc['Buy Exit'] = np.where(buy_exit & (o1 < np.roll(o3, 1)), exit_low, np.nan)
    tdc['Sell Exit'] = np.where(sell_exit & (o4 < np.roll(o2, 1)), exit_high, np.nan)

    tdc = pd.concat([tdc, signals_df], axis=1)

    return tdc
