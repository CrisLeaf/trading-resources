import yfinance as yf
from datetime import datetime, timedelta
import yaml

from strategies_manager import TradingStrategiesManager


TRADING_CONFIG_PATH = 'configurations/trading_config.yml'

with open(TRADING_CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)


def run_trading_system():
    timeframes = config['timeframes']
    download_period = config['download_period']
    strategies_keys = ['MACD_RSI_BB', 'SQZM_DMI']

    intraday_positions = []
    daily_positions = []

    for asset_type, instruments in config['assets'].items():
        for horizon, windows in timeframes.items():
            for interval in windows:
                df = yf.download(instruments, interval=interval, period=download_period[interval])
                df.columns = df.columns.swaplevel(0, 1)
                
                for instrument in instruments:
                    asset_data = df[instrument].dropna()
                
                    sub_strat_params = config['strategy_parameters'][horizon][interval]
                    
                    for key in sub_strat_params.keys():
                        sub_strat_params[key]['df'] = asset_data

                    manager = TradingStrategiesManager(strategy_params={
                        strategies_keys[0]: {
                            'data': asset_data,
                            'params': sub_strat_params['strategy1']['params']
                        },
                        strategies_keys[1]: {
                            'data': asset_data,
                            'params': sub_strat_params['strategy2']['params']
                        }
                    })

                    results = manager.run_all_strategies(allow_shorts=False)
                    
                    for key in strategies_keys:
                        position = {
                            'instrument': instrument,
                            'interval': interval,
                            'strategy': key,
                            'direction': manager.strategies[key].data['Direction'].iloc[-1]
                        }
                        if horizon == 'intraday':
                            intraday_positions.append(position)
                        else:
                            daily_positions.append(position)
    
    return intraday_positions, daily_positions


if __name__ == '__main__':
    intraday_positions, daily_positions = run_trading_system()

    # Here you can implement any logic to handle the collected positions
    # For example, you might want to print them, store them in a database, etc.
    print('Intraday Positions:')
    for pos in intraday_positions:
        print(pos)

    print('\nDaily Positions:')
    for pos in daily_positions:
        print(pos)