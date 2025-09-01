import os
import sys
import itertools
import random
from typing import Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

# from strategies.base_strategy import BaseStrategy
from base_strategy import BaseStrategy
from indicators.trend.ema import exponential_moving_average
from indicators.momentum.rsi import relative_strength_index
from indicators.momentum.stochrsi import stochastic_rsi
from indicators.volatility.atr import average_true_range


class TrendReversalATRStrategy(BaseStrategy):
    """
    **This implementation only works for Long Trading**
    Implements a trend reversal trading strategy using technical indicators.

    This class is designed to execute a trading strategy that analyzes trends in a given market dataset.
    Key indicators used include Exponential Moving Averages (EMA), Relative Strength Index (RSI),
    Stochastic RSI, and Average True Range (ATR). It provides methods to calculate trading signals,
    evaluate performance metrics, and optimize strategy parameters for better results.
    """
    def __init__(
            self,
            data: pd.DataFrame,
            params: Dict[str, Any] = None,
            logger: Any = None,
            random_seed: int = None
        ):
        default_params = {
            'ema_short_period': 50,
            'ema_long_period': 200,
            'rsi_period': 14,
            'stochrsi_rsi_period': 14,
            'stochrsi_stoch_period': 14,
            'stochrsi_smooth_k': 3,
            'stochrsi_smooth_d': 3,
            'atr_period': 14,
            'high_column': 'High',
            'low_column': 'Low',
            'close_column': 'Close'
        }
        self.optimized_params = None
        
        if params:
            default_params.update(params)
            
        super().__init__(data, default_params, logger, random_seed)

    def calculate_signals(self) -> Dict[str, Any]:
        """
        Calculates trading signals and trends based on technical indicators applied to market
        data.
        This method incorporates various technical indicators such as EMA (Exponential Moving
        Averages), RSI (Relative Strength Index), Stochastic RSI, and ATR (Average True Range)
        to generate buy and sell signals. It integrates stop-loss and take-profit computations
        to finalize the trading logic and determine current market trends.
        """
        ema_short = exponential_moving_average(
            df=self.data,
            period=self.params['ema_short_period'],
            column=self.params['close_column']
        )
        ema_long = exponential_moving_average(
            df=self.data,
            period=self.params['ema_long_period'],
            column=self.params['close_column']
        )
        rsi = relative_strength_index(
            df=self.data,
            period=self.params['rsi_period'],
            column=self.params['close_column']
        )
        stochrsi = stochastic_rsi(
            df=self.data,
            rsi_period=self.params['stochrsi_rsi_period'],
            stoch_period=self.params['stochrsi_stoch_period'],
            smooth_k=self.params['stochrsi_smooth_k'],
            smooth_d=self.params['stochrsi_smooth_d'],
            column=self.params['close_column']
        )
        atr = average_true_range(
            df=self.data,
            period=self.params['atr_period'],
            high_column=self.params['high_column'],
            low_column=self.params['low_column'],
            close_column=self.params['close_column']
        )
        self.data['EMA_Short'] = ema_short
        self.data['EMA_Long'] = ema_long
        self.data['RSI'] = rsi
        self.data['StochRSI'] = stochrsi['StochRSI']
        self.data['StochRSI_K'] = stochrsi['StochRSI_K']
        self.data['StochRSI_D'] = stochrsi['StochRSI_D']
        self.data['ATR'] = atr
        self.data['Last_Buy_price'] = np.nan

        self.data['Buy_Signal'] = (
            (self.data['EMA_Short'] > self.data['EMA_Long']) &
            # (self.data['RSI'] < 50) &
            # (self.data['StochRSI_K'] < 20) &
            (self.data['StochRSI_K'] > self.data['StochRSI_D'])
        )
        self.data['Buy_Signal'] = np.random.rand(len(self.data)) <= 0.5
        self.data.loc[self.data['Buy_Signal'], 'Last_Buy_price'] = self.data['Close']
        self.data['Last_Buy_price'] = self.data['Last_Buy_price'].ffill()

        self.data['Stop_Loss'] = self.data['Last_Buy_price'] - 1.5 * self.data['ATR']
        self.data['Take_Profit'] = self.data['Last_Buy_price'] + 2 * self.data['ATR']

        self.data['Sell_Signal'] = (
            (self.data['Close'] <= self.data['Stop_Loss']) |
            (self.data['Close'] >= self.data['Take_Profit'])
        )

        # Direction
        self.data['Direction'] = self.data['Buy_Signal'].astype(int) + self.data['Sell_Signal'].astype(int) * -1

        # Detect current trend
        trend_map = {1: 'bullish', -1: 'bearish', 0: 'neutral'}
        self.data['Trend'] = self.data['Direction'].map(trend_map)
        current_trend_dict = {'current_trend': trend_map.get(self.data['Direction'].iloc[-1])}

        return current_trend_dict

    def evaluate_performance(self, allow_shorts: bool = False) -> Dict[str, Any]:
        """
        Evaluates the performance of the trading strategy based on generated buy and sell signals.
        Calculates positions, strategy returns, equity curve, and key performance metrics such as total return,
        CAGR, Sharpe ratio, and maximum drawdown. Supports optional short positions.
        """
        if 'Buy_Signal' not in self.data or 'Sell_Signal' not in self.data:
            raise ValueError("Signals not calculated yet. Run calculate_signals() first.")

        df = self.data.copy()
        initial_capital = 100_000.0
        commission = 0.00518
        
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
        Optimize strategy parameters by sampling combinations from the given grid.
        Evaluates each parameter set based on performance metrics and returns the set with the highest score.
        The score for each parameter combination is calculated as:

            score = (CAGR * Sharpe) / (1 + MaxDrawdown)

        Where:
            - CAGR is the compound annual growth rate of the strategy
            - Sharpe is the Sharpe ratio of the strategy
            - MaxDrawdown is the maximum drawdown of the strategy
        """
        keys = list(params_grid.keys())
        values = list(params_grid.values())
        all_combinations = list(itertools.product(*values))
        print('len all combinations:', len(all_combinations))

        n_iter = min(n_iter, len(all_combinations))
        samples_combinations = random.sample(all_combinations, n_iter)
        
        best_score = float('-inf')
        best_params = self.params.copy()
        
        for comb in tqdm(samples_combinations, desc='Optimizing parameters'):
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

        self.optimized_params = best_params

        return best_params
    
    def save_to_excel(self, filename: str = "TrendReversalATR_metrics.xlsx"):
        """
        Saves data and optimized parameters to an Excel file.
        The method writes the strategy's data and its optimized parameters to an Excel file using two
        separate sheets. The output format is suitable for further analysis or reporting.
        """
        params_df = pd.DataFrame([self.optimized_params]).T.reset_index()
        params_df.columns = ['Parameter', 'Value']

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.data.to_excel(writer, sheet_name='Data')
            params_df.to_excel(writer, sheet_name='Parameters', index=False)

    def plot(self, *args, **kwargs):
        """
        Plots price data with buy and sell signals along with an equity curve.
        This method creates a two-row visualization where:
        1. The top row includes the price data with buy and sell entries marked
           using specific symbols.
        2. The bottom row represents the equity curve over time.

        Parameters:
        args
            Positional arguments passed to the function. This is unused in the current implementation.
        kwargs
            Keyword arguments where `last_entries` is expected to specify the
            number of most recent data points to include in the plot.
        """
        plot_data = self.data.copy()[-kwargs['last_entries']: ]
        position_diff = plot_data['Position'].diff().fillna(0)
        position_diff = position_diff.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        buy_entries = plot_data['Close'].astype(float).copy()
        buy_entries[position_diff != 1] = float('nan')
        sell_entries = plot_data['Close'].astype(float).copy()
        sell_entries[position_diff != -1] = float('nan')

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.7, 0.3]
        )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Close'],
                mode='lines',
                line=dict(color='skyblue', width=1),
                name='Close'
            ), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=buy_entries.index,
                y=buy_entries,
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=18),
                name='Buy Signal'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sell_entries.index,
                y=sell_entries,
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=18),
                name='Sell Signal'
            )
        )

        # Equity Curve
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Equity_Curve'],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False,
            ), row=2, col=1
        )

        # Layout
        fig.update_layout(
            template='plotly_dark',
            title='Signals Plot',
            xaxis2_title='Date',
            yaxis_title='Price',
            yaxis2_title='Equity Curve',
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgb(20, 20, 20)',
            paper_bgcolor='rgb(20, 20, 20)',
            font=dict(color='white'),
            height=900,
            width=1000
        )

        fig.show()

    
if __name__ == "__main__":
    import yfinance as yf
    import numpy as np

    # Obtener Datos
    df = yf.download('USDCLP=X', start='2020-01-01', interval='1d')
    df.columns = df.columns.droplevel(1)

    strategy = TrendReversalATRStrategy(df)
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
        'ema_short_period': list(range(20, 101, 20)),
        'ema_long_period': list(range(100, 301, 50)),
        # 'rsi_period': [7, 14, 21],
        'stochrsi_rsi_period': [7, 14, 21],
        'stochrsi_stoch_period': [7, 14, 21],
        'stochrsi_smooth_k': [2, 3, 4],
        'stochrsi_smooth_d': [2, 3, 4],
        'atr_period': list(range(4, 21, 2)),
        # 'atr_period': [7, 14, 21],
    }

    best_params = strategy.optimize(params_grid, n_iter=1_000)

    print("Best Parameters:")
    print(best_params)
    print(pd.DataFrame([best_params]))

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
    strategy.plot(last_entries=252 * 10)

    # Write Excel
    strategy.save_to_excel()
