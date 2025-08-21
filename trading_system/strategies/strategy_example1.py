import numpy as np
import pandas as pd
import sys
sys.path.append('../../indicators')
from typing import Dict, Any
import itertools
import random
from tqdm import tqdm
import mplfinance as mpf

from base_strategy import BaseStrategy
from trend.macd import macd_index
from momentum.rsi import relative_strength_index
from volatility.bb import bollinger_bands


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
    
    def save_to_excel(self, filename: str = "strategy_data.xlsx"):
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
            mpf.make_addplot(plot_data['Bollinger_Mid'], color='blue'),
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
            mpf.make_addplot(plot_data['MACD'], panel=1, color='green', ylabel='MACD', label='MACD'),
            mpf.make_addplot(plot_data['MACD_Signal'], panel=1, color='red'),
            mpf.make_addplot(plot_data['RSI'], panel=2, color='purple', ylabel='RSI'),
            mpf.make_addplot(
                [50]*len(plot_data), panel=2, color='gray', secondary_y=False, linestyle='dashed'
            )
        ]

        mpf.plot(
            plot_data,
            type='candle',
            volume=False,
            addplot=apds,
            panel_ratios=(3, 1, 1),
            title='Candlesticks Chart with MACD, RSI and Bollinger Bands',
            figratio=(20, 10),
            figscale=1.5,
        )

    
if __name__ == "__main__":
    import yfinance as yf
    import time

    # Obtener Datos
    df = yf.download('AAPL', start='2022-01-01', end='2025-08-01', interval='1d')
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
        'macd_fast_period': np.arange(4, 24, 1).tolist(),
        'macd_slow_period': np.arange(18, 34, 1).tolist(),
        'macd_signal_period': np.arange(2, 22, 1).tolist(),
        'rsi_period': np.arange(5, 25, 1).tolist(),
        'bb_period': np.arange(12, 28, 1).tolist(),
        'bb_k': [2.0],
        'bb_ddof': [0],
    }
    
    best_params = strategy.optimize(params_grid, n_iter=1_000)

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
    
    strategy.save_to_excel("strategy_data.xlsx")