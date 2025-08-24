import pandas as pd
import pandas_market_calendars as mcal
import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def ensure_ny_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp is None:
        return pd.Timestamp.now(tz="America/New_York")
    
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("America/New_York")
    
    return timestamp.tz_convert("America/New_York")

def is_market_open(
        calendar: mcal.MarketCalendar,
        timestamp: pd.Timestamp | None = None
    ) -> bool:
    ts = ensure_ny_timestamp(timestamp)

    schedule = calendar.schedule(start_date=ts.date(), end_date=ts.date())
    
    if schedule.empty:
        return False
    
    open_time = schedule.iloc[0]['market_open']
    close_time = schedule.iloc[0]['market_close']

    return open_time <= ts <= close_time
