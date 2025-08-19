import numpy as np
import pandas as pd
import sys
sys.path.append('../../indicators')
from typing import Dict, Any
import itertools
import random

from base_strategy import BaseStrategy
from trend.macd import macd_index
from momentum.rsi import relative_strength_index
from volatility.bb import bollinger_bands

from tqdm import tqdm


class MACD_RSI_BB_Strategy(BaseStrategy):
    """
    Strategy combining RSI, Bollinger Bands, and MACD indicators.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            params: Dict[str, Any] = None,
            logger: Any = None,
            random_seed: int = None
        ):
        default_params = {
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            'macd_column': 'Close',
            'rsi_period': 14,
            'rsi_column': 'Close',
            'bb_period': 20,
            'bb_k': 2.0,
            'bb_ddof': 0,
            'bb_column': 'Close',
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(data, default_params, logger, random_seed)

    def calculate_signals(self) -> Dict[str, Any]:
        """
        Calculate buy/sell signals based on RSI + Bollinger Bands + MACD logic.
        Returns a DataFrame with columns: ['Buy_Signal', 'Sell_Signal']
        """
        macd = macd_index(
            df=self.data,
            fast_period=self.params['macd_fast_period'],
            slow_period=self.params['macd_slow_period'],
            signal_period=self.params['macd_signal_period'],
            column=self.params['macd_column']
        )
        rsi = relative_strength_index(
            df=self.data,
            period=self.params['rsi_period'],
            column=self.params['rsi_column']
        )
        bollinger_bands_dict = bollinger_bands(
            df=self.data,
            period=self.params['bb_period'],
            k=self.params['bb_k'],
            ddof=self.params['bb_ddof'],
            column=self.params['bb_column']
        )
        self.data['MACD'] = macd['MACD']
        self.data['MACD_Signal'] = macd['Signal']
        self.data['RSI'] = rsi
        self.data['Bollinger_Mid'] = bollinger_bands_dict['Bollinger_Mid']
        
        # Signals
        self.data['Buy_Signal'] = (
            (self.data['MACD'] > self.data['MACD_Signal']) &
            (self.data['RSI'] < 50) &
            (self.data['Close'] < self.data['Bollinger_Mid'])
        )
        self.data['Sell_Signal'] = (
            (self.data['MACD'] < self.data['MACD_Signal']) &
            (self.data['RSI'] > 50) &
            (self.data['Close'] > self.data['Bollinger_Mid'])
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

    def evaluate_performance(self) -> Dict[str, Any]:
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
                df.loc[df.index[i], 'Position'] = -1
            else:
                df.loc[df.index[i], 'Position'] = df['Position'].iloc[i-1]
        
        # Returns
        df['Trade_Size'] = df['Position'].diff().abs()
        df['Commission'] = commission * df['Trade_Size']
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return'] - df['Commission']
        df['Equity_Curve'] = (1 + df['Strategy_Return']).cumprod() * initial_capital
        
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
        print('len(all_combinations):', len(all_combinations))

        n_iter = min(n_iter, len(all_combinations))
        samples_combinations = random.sample(all_combinations, n_iter)
        
        best_score = float('-inf')
        best_params = self.params.copy()
        
        for comb in tqdm(samples_combinations):
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

        print('best score:', best_score)

        return best_params

    def plot(self, *args, **kwargs):
        """
        Placeholder: Plot price + indicators + signals.
        """
        # initial_capital = 100_000.0
        # cagr = performance_dict['CAGR']

        # # Calcula los años transcurridos para cada punto
        # days = (df.index - df.index[0]).days
        # years = days / 365.25
        # equity_cagr = initial_capital * (1 + cagr) ** years

        # plt.figure(figsize=(10,6))
        # plt.plot(df.index, equity_cagr, label=f'Equity at CAGR {cagr:.2%}')
        # plt.plot(df.index, performance_dict['Equity Curve'], label='Real Equity')
        # plt.xlabel('Time')
        # plt.ylabel('Capital')
        # plt.title('Real Equity vs Ideal CAGR Equity')
        # plt.legend()
        # plt.show()
        pass
    
if __name__ == "__main__":
    import yfinance as yf
    import time

    # Obtener Datos
    df = yf.download('AAPL', start='2022-01-01', end='2025-01-01', interval='1d')
    df.columns = df.columns.droplevel(1)

    strategy = MACD_RSI_BB_Strategy(df)
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
        'macd_fast_period': np.arange(5, 21, 1).tolist(),
        'macd_slow_period': np.arange(21, 31, 1).tolist(),
        'macd_signal_period': np.arange(2, 18, 1).tolist(),
        'rsi_period': np.arange(7, 22, 1).tolist(),
        'bb_period': np.arange(16, 26, 1).tolist(),
        'bb_k': [2.0],
        'bb_ddof': [0],
    }
    
    best_params = strategy.optimize(params_grid, n_iter=1_000)

    print("Best Parameters:")
    print(best_params)

    strategy.set_params(best_params)
    strategy.calculate_signals()
    performance_dict = strategy.evaluate_performance()
    
    print()
    print('Total Return:', round(performance_dict['Total Return'], 4))
    print('CAGR:', round(performance_dict['CAGR'], 4))
    print('Sharpe Ratio:', round(performance_dict['Sharpe Ratio'], 4))
    print('Max Drawdown:', round(performance_dict['Max Drawdown'], 4))
    print()