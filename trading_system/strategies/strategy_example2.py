import numpy as np
import pandas as pd
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
from typing import Dict, Any
import itertools
import random
from tqdm import tqdm
import mplfinance as mpf

from strategies.base_strategy import BaseStrategy
from indicators.advanced.sqz_m import squeeze_momentum
from indicators.trend.dmi import directional_movement_index


class SQZM_DMI_Strategy(BaseStrategy):
    """
    Strategy combining Squeeze Momentum and DMI indicators.
    
    Buy/Sell Signal Calculation
    ---------------------------
    This strategy generates buy and sell signals based on a combination of two technical indicators:

    1. **Squeeze Momentum (SQZ):**
    - Buy condition: SQZ value is greater than 0 (indicating bullish momentum).
    - Sell condition: SQZ value is less than 0 (indicating bearish momentum).

    2. **Directional Movement Index (DMI):**
    - Buy condition: +DI is greater than -DI (bullish trend).
    - Sell condition: +DI is less than -DI (bearish trend).

    **Buy Signal:**  
    Generated when both of the following are true:
    - SQZ > 0
    - +DI > -DI

    **Sell Signal:**  
    Generated when both of the following are true:
    - SQZ < 0
    - +DI < -DI
    """

    def __init__(
            self,
            data: pd.DataFrame,
            params: Dict[str, Any] = None,
            logger: Any = None,
            random_seed: int = None
        ):
        default_params = {    
            'sqzm_bb_period': 20,
            'sqzm_bb_std_dev': 2.0,
            'sqzm_kc_period': 20,
            'sqzm_kc_mult': 1.5,
            'sqzm_momentum_period': 12,
            'sqzm_momentum_longitude': 6,
            'sqzm_high_column': 'High',
            'sqzm_low_column': 'Low',
            'sqzm_close_column': 'Close',
            'dmi_adx_period': 14,
            'dmi_di_period': 14,
            'dmi_high_column': 'High',
            'dmi_low_column': 'Low',
            'dmi_close_column': 'Close'
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(data, default_params, logger, random_seed)

    def calculate_signals(self) -> Dict[str, Any]:
        """
        Calculate buy/sell signals based on Squeeze Momentum + DMI logic.
        Returns a DataFrame with columns: ['Buy_Signal', 'Sell_Signal']
        """
        sqzm = squeeze_momentum(
            df=self.data,
            bb_period=self.params['sqzm_bb_period'],
            bb_std_dev=self.params['sqzm_bb_std_dev'],
            kc_period=self.params['sqzm_kc_period'],
            kc_mult=self.params['sqzm_kc_mult'],
            momentum_period=self.params['sqzm_momentum_period'],
            momentum_longitude=self.params['sqzm_momentum_longitude'],
            high_column=self.params['sqzm_high_column'],
            low_column=self.params['sqzm_low_column'],
            close_column=self.params['sqzm_close_column']
        )
        dmi = directional_movement_index(
            df=self.data,
            adx_period=self.params['dmi_adx_period'],
            di_period=self.params['dmi_di_period'],
            high_column=self.params['dmi_high_column'],
            low_column=self.params['dmi_low_column'],
            close_column=self.params['dmi_close_column']
        )
        self.data['SQZ'] = sqzm['SQZ']
        self.data['SQZ_ON'] = sqzm['SQZ_ON']
        self.data['SQZ_OFF'] = sqzm['SQZ_OFF']
        self.data['NO_SQZ'] = sqzm['NO_SQZ']
        self.data['ADX'] = dmi['ADX']
        self.data['+DI'] = dmi['+DI']
        self.data['-DI'] = dmi['-DI']
        
        # Signals
        self.data['Buy_Signal'] = (
            (self.data['SQZ'] > 0) & 
            (self.data['+DI'] > self.data['-DI'])
        )
        self.data['Sell_Signal'] = (
            (self.data['SQZ'] < 0) & 
            (self.data['+DI'] < self.data['-DI'])
        )
        
        # Direction
        self.data['Direction'] = self.data['Buy_Signal'].astype(int) + self.data['Sell_Signal'].astype(int) * -1
        
        # Trend
        trend_map = {1: 'bullish', -1: 'bearish', 0: 'neutral'}
        self.data['Trend'] = self.data['Direction'].map(trend_map)
        
        # Detect current trend
        trend_map = {1: 'bullish', -1: 'bearish', 0: 'neutral'}
        current_trend = {'current_trend': trend_map.get(self.data['Direction'].iloc[-1])}

        return current_trend

    def evaluate_performance(self, allow_shorts: bool = False) -> Dict[str, Any]:
        """
        Placeholder: Evaluate performance of the strategy.
        Returns a dictionary with metrics like total return, Sharpe ratio, etc.
        """
        if 'Buy_Signal' not in self.data or 'Sell_Signal' not in self.data:
            raise ValueError("Signals not calculated yet. Run calculate_signals() first.")

        df = self.data.copy()
        initial_capital = 100_000.0
        commission = 0.001
        
        # Positions
        df['Position'] = 0
        
        for i in range(1, len(df)):
            if df['Buy_Signal'].iloc[i]:
                df.loc[df.index[i], 'Position'] = 1
            elif df['Sell_Signal'].iloc[i]:
                if allow_shorts:
                    df.loc[df.index[i], 'Position'] = -1
                else:
                    df.loc[df.index[i], 'Position'] = 0
            else:
                df.loc[df.index[i], 'Position'] = df['Position'].iloc[i-1]
        
        self.data['Position'] = df['Position']
        
        # Returns
        df['Trade_Size'] = df['Position'].diff().abs()
        df['Commission'] = commission * df['Trade_Size']
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return'] - df['Commission']
        df['Equity_Curve'] = (1 + df['Strategy_Return']).cumprod() * initial_capital

        self.data['Trade_Size'] = df['Trade_Size']
        self.data['Commission'] = df['Commission']
        self.data['Market_Return'] = df['Market_Return']
        self.data['Strategy_Return'] = df['Strategy_Return']
        self.data['Equity_Curve'] = df['Equity_Curve']

        # Metrics
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        trading_days = len(df)
        trading_days_per_year = trading_days / years
        mean_return = df['Strategy_Return'].mean()
        std_return = df['Strategy_Return'].std()
        
        total_return = df['Equity_Curve'].iloc[-1] / initial_capital - 1
        cagr = (df['Equity_Curve'].iloc[-1] / initial_capital) ** (1 / years) - 1
        sharpe = mean_return / std_return * np.sqrt(trading_days_per_year) if std_return != 0 else 0
        rolling_max = df['Equity_Curve'].cummax()
        drawdown = df['Equity_Curve'] / rolling_max - 1
        max_drawdown = drawdown.min()

        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Equity Curve': df['Equity_Curve'],
            'Final Equity': df['Equity_Curve'].iloc[-1],
        }

    def optimize(self, params_grid: Dict[str, Any], n_iter: int = 100) -> Dict[str, Any]:
        """
        Placeholder: Optimize strategy parameters using the provided grid.
        Returns the best parameters found.
        """
        keys = list(params_grid.keys())
        values = list(params_grid.values())
        all_combinations = list(itertools.product(*values))

        n_iter = min(n_iter, len(all_combinations))
        samples_combinations = random.sample(all_combinations, n_iter)
        
        best_score = float('-inf')
        best_params = self.params.copy()
        
        for comb in tqdm(samples_combinations, desc="Optimizing parameters"):
            test_params = dict(zip(keys, comb))
            self.set_params(test_params)
            self.calculate_signals()
            perf = self.evaluate_performance()
            cagr = perf.get('CAGR', 0)
            cagr = max(cagr, 0)
            sharpe = perf.get('Sharpe Ratio', 0)
            sharpe = max(sharpe, 0)
            max_drawdown = abs(perf.get('Max Drawdown', 0))
            score = (cagr * sharpe) / (1 + max_drawdown)
            
            if score > best_score:
                best_score = score
                best_params = self.params.copy()

        return best_params
    
    def save_to_excel(self, filename: str = "SQZM_DMI_Strategy_metrics.xlsx"):
        """
        Save the strategy DataFrame to an Excel file.
        """
        self.data.to_excel(filename)

    def plot(self, *args, **kwargs):
        """
        Placeholder: Plot price + indicators + signals.
        """
        plot_data = self.data.copy()[-kwargs['last_entries']: ]
        position_diff = plot_data['Position'].diff().fillna(0)
        position_diff = position_diff.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        buy_entries = plot_data['Close'].astype(float).copy()
        buy_entries[position_diff != 1] = float('nan')
        sell_entries = plot_data['Close'].astype(float).copy()
        sell_entries[position_diff != -1] = float('nan')
        
        apds = [
            mpf.make_addplot(
                buy_entries,
                type='scatter',
                markersize=120,
                marker='^',
                color='g'
            ),
            mpf.make_addplot(
                sell_entries,
                type='scatter',
                markersize=120,
                marker='v',
                color='r'
            ),
            mpf.make_addplot(plot_data['+DI'], panel=1, color='green', ylabel='DMI', label='+DI'),
            mpf.make_addplot(plot_data['-DI'], panel=1, color='red', label='-DI', secondary_y=False),
            mpf.make_addplot(plot_data['SQZ'], panel=2, color='blue', ylabel='SQZ Mom.', type='bar', label='SQZ')

        ]
        mpf.plot(
            plot_data,
            type='line',
            volume=False,
            addplot=apds,
            panel_ratios=(2, 1, 1),
            title='Candlesticks Chart with Squeeze Momentum and DMI',
            figratio=(20, 10),
            figscale=1.5,
        )

    
if __name__ == "__main__":
    import yfinance as yf
    import time

    # Obtener Datos
    df = yf.download('AAPL', start='2022-01-01', end='2025-08-01', interval='1d')
    df.columns = df.columns.droplevel(1)

    strategy = SQZM_DMI_Strategy(df)
    current_trend = strategy.calculate_signals()

    print(strategy.data['Direction'].value_counts())

    print(current_trend)
    
    # backtest
    performance_dict = strategy.evaluate_performance()

    print('Total Return:', round(performance_dict['Total Return'], 4))
    print('CAGR:', round(performance_dict['CAGR'], 4))
    print('Sharpe Ratio:', round(performance_dict['Sharpe Ratio'], 4))
    print('Max Drawdown:', round(performance_dict['Max Drawdown'], 4))
    print()
    
    # Optimize
    params_grid = {
        'sqzm_bb_period': np.arange(10, 31, 1).tolist(),
        'sqzm_bb_std_dev': [1.5, 2.0, 2.5],
        'sqzm_kc_period': np.arange(10, 31, 1).tolist(),
        'sqzm_kc_std_dev': [1.5, 2.0, 2.5],
        'sqzm_momentum_period': np.arange(5, 21, 1).tolist(),
        'sqzm_momentum_longitude': np.arange(3, 11, 1).tolist(),
        'dmi_adx_period': np.arange(10, 21, 1).tolist(),
        'dmi_di_period': np.arange(10, 21, 1).tolist()
    }
    
    best_params = strategy.optimize(params_grid, n_iter=10_000)

    print("Best Parameters:")
    print(best_params)

    # Best Params and Backtest
    strategy.set_params(best_params)
    strategy.calculate_signals()
    performance_dict = strategy.evaluate_performance()
    
    print()
    print('Total Return:', round(performance_dict['Total Return'], 4))
    print('CAGR:', round(performance_dict['CAGR'], 4))
    print('Sharpe Ratio:', round(performance_dict['Sharpe Ratio'], 4))
    print('Max Drawdown:', round(performance_dict['Max Drawdown'], 4))
    print()
    
    # Plot
    strategy.plot(last_entries=5000)
    
    strategy.save_to_excel()