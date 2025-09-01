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
from indicators.advanced.ch_exit import chandelier_exit
from indicators.advanced.sqz_m import squeeze_momentum


class SqueezeChandelierStrategy(BaseStrategy):
    """
    **This implementation only works for Long Trading**

    Strategy combining Squeeze Momentum and Chandelier Exit.
    This strategy uses a volatility-based entry approach with Squeeze Momentum, paired with dynamic exit
    management through the Chandelier Exit indicator:

    - Squeeze Momentum: identifies periods of low volatility ("squeeze") that often precede strong price
                        moves. A potential buy setup occurs when the squeeze releases and momentum
                        turns bullish (momentum histogram > 0). A potential sell setup occurs when
                        the squeeze releases and momentum turns bearish (momentum histogram < 0).
    - Chandelier Exit: provides a volatility-adjusted trailing stop based on ATR. It defines the exit
                       point dynamically as price moves, allowing profitable trends to run while
                       protecting against sharp reversals.
    - Combined Signals: entries are determined by Squeeze Momentum, while exits are managed by
                        Chandelier Exit:
        - Buy_Signal: triggered when squeeze releases to the upside (momentum > 0).
                      The trade is held until the price closes below the Chandelier Exit stop level.
        - Sell_Signal: triggered when squeeze releases to the downside (momentum < 0).
                       The trade is held until Price closes above the Chandelier Exit stop level.

    This approach leverages volatility compression/expansion for precise entries (Squeeze Momentum)
    and a robust trailing exit mechanism (Chandelier Exit) to maximize trend-following potential
    while minimizing risk from sudden reversals.
    """
    def __init__(
            self,
            data: pd.DataFrame,
            params: Dict[str, Any] = None,
            logger: Any = None,
            random_seed: int = None
        ):
        default_params = {
            'sqz_bb_period': 20,
            'sqz_bb_std_dev': 2.0,
            'sqz_kc_period': 20,
            'sqz_kc_mult': 1.5,
            'sqz_momentum_period': 12,
            'sqz_momentum_longitude': 6,
            'sqz_high_column': 'High',
            'sqz_low_column': 'Low',
            'sqz_close_column': 'Close',
            'ch_ce_period': 22,
            'ch_atr_period': 22,
            'ch_multiplier': 3.0,
            'ch_high_column': 'High',
            'ch_low_column': 'Low',
            'ch_close_column': 'Close'
        }
        self.optimized_params = None
        
        if params:
            default_params.update(params)
            
        super().__init__(data, default_params, logger, random_seed)

    def calculate_signals(self) -> Dict[str, Any]:
        """
        Calculates trading signals and detects market trend based on several technical indicators.
        This method processes data using Squeeze Momentum and Chandelier Exit indicators to evaluate
        market signals and determine the current trend. The results are added to the data attribute,
        and the detected trend is returned.

        Raises:
            KeyError: If any required columns or parameters are missing from the input data or parameters dictionary.
            IndexError: If the method is unable to access the last value in the data attribute due to insufficient rows.

        Returns:
            Dict[str, Any]: A dictionary containing the current market trend with the key 'current_trend'.
        """
        sqz = squeeze_momentum(
            df=self.data,
            bb_period=self.params['sqz_bb_period'],
            bb_std_dev=self.params['sqz_bb_std_dev'],
            kc_period=self.params['sqz_kc_period'],
            kc_mult=self.params['sqz_kc_mult'],
            momentum_period=self.params['sqz_momentum_period'],
            momentum_longitude=self.params['sqz_momentum_longitude'],
            high_column=self.params['sqz_high_column'],
            low_column=self.params['sqz_low_column'],
            close_column=self.params['sqz_close_column']
        )
        ch = chandelier_exit(
            df=self.data,
            ce_period=self.params['ch_ce_period'],
            atr_period=self.params['ch_atr_period'],
            multiplier=self.params['ch_multiplier'],
            high_column=self.params['ch_high_column'],
            low_column=self.params['ch_low_column'],
            close_column=self.params['ch_close_column']
        )
        self.data['SQZ'] = sqz['SQZ']
        self.data['SQZ_ON'] = sqz['SQZ_ON']
        self.data['SQZ_OFF'] = sqz['SQZ_OFF']
        self.data['NO_SQZ'] = sqz['NO_SQZ']
        self.data['Chandelier_Long'] = ch['Chandelier_Long']
        self.data['Chandelier_Short'] = ch['Chandelier_Short']
        self.data['Long_Stop'] = ch['Long_Stop']
        self.data['Short_Stop'] = ch['Short_Stop']
        self.data['Direction'] = ch['Direction']
        self.data['CE_Buy'] = ch['CE_Buy']
        self.data['CE_Sell'] = ch['CE_Sell']

        # Signals
        self.data['Buy_Signal'] = ((self.data['SQZ_OFF'] == 1) & (self.data['SQZ'] > 0))
        self.data['Sell_Signal'] = (self.data['Close'] < self.data['Long_Stop']) | (self.data['CE_Sell'] == 1)

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
    
    def save_to_excel(self, filename: str = "SqueezeChandelierStrategy_metrics.xlsx"):
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

    strategy = SqueezeChandelierStrategy(df)
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
        'sqz_bb_period': [10, 15, 20, 25, 30],
        'sqz_bb_std_dev': [1.0, 1.5, 2.0, 2.5, 3.0],
        'sqz_kc_period': [10, 15, 20, 25, 30],
        'sqz_kc_mult': [0.5, 1.0, 1.5, 2.0, 2.5],
        'sqz_momentum_period': [6, 12, 18],
        'sqz_momentum_longitude': [3, 6, 9],
        'ch_ce_period': [14, 18, 22, 26, 30],
        'ch_atr_period': [14, 18, 22, 26, 30],
        'ch_multiplier': [1.0, 2.0, 3.0, 4.0, 5.0]
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
