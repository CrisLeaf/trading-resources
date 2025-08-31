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
from indicators.advanced.st import super_trend
from indicators.trend.ema import exponential_moving_average


class SuperTrendEMAStrategy(BaseStrategy):
    """
    Strategy combining SuperTrend and EMA Crossovers.
    This strategy uses a trend-following approach based on the SuperTrend indicator and confirmation from
    EMA crossovers:

    - SuperTrend: determines the prevailing market trend. A buy signal occurs when the trend switches to bullish
                  (ST_Direction == 1), and a sell signal when it switches to bearish (ST_Direction == -1).
    - EMA Crossovers: uses two EMAs (short and long periods) to confirm trend changes. A bullish crossover occurs
                      when EMA_Short crosses above EMA_Long, and a bearish crossover when EMA_Short crosses
                      below EMA_Long.
    - Combined Signals: a trade is triggered only when both SuperTrend and EMA conditions align:
        - Buy_Signal: SuperTrend turns bullish AND short EMA crosses above long EMA.
        - Sell_Signal: SuperTrend turns bearish AND short EMA crosses below long EMA.

    This approach helps filter out false signals by requiring agreement between a trend direction (SuperTrend)
    and momentum (EMA cross).
    """
    def __init__(
            self,
            data: pd.DataFrame,
            params: Dict[str, Any] = None,
            logger: Any = None,
            random_seed: int = None
        ):
        default_params = {
            'st_period': 10,
            'st_multiplier': 3.0,
            'st_high_column': 'High',
            'st_low_column': 'Low',
            'st_close_column': 'Close',
            'ema_short_period': 10,
            'ema_short_column': 'Close',
            'ema_long_period': 50,
            'ema_long_column': 'Close'
        }
        self.optimized_params = None
        
        if params:
            default_params.update(params)
            
        super().__init__(data, default_params, logger, random_seed)

    def calculate_signals(self) -> Dict[str, Any]:
        """
        Calculates trading signals based on the combination of SuperTrend and EMA crossovers.
        This function computes the SuperTrend indicator and short/long EMAs, then generates buy and sell signals when
        both conditions are met. It also determines the current market direction (bullish, bearish, or neutral) based
        on the latest signals and returns it as a dictionary.
        """
        st = super_trend(
            df=self.data,
            period=self.params['st_period'],
            multiplier=self.params['st_multiplier'],
            high_column=self.params['st_high_column'],
            low_column=self.params['st_low_column'],
            close_column=self.params['st_close_column'],
        )
        ema_short = exponential_moving_average(
            df=self.data,
            period=self.params['ema_short_period'],
            column=self.params['ema_short_column'],
        )
        ema_long = exponential_moving_average(
            df=self.data,
            period=self.params['ema_long_period'],
            column=self.params['ema_long_column'],
        )
        self.data['SuperTrend'] = st['SuperTrend']
        self.data['ST_Direction'] = st['ST_Direction']
        self.data['UpperBand'] = st['UpperBand']
        self.data['LowerBand'] = st['LowerBand']
        self.data['EMA_Short'] = ema_short
        self.data['EMA_Long'] = ema_long

        # Signals
        self.data['ST_Buy'] = (self.data['ST_Direction'] == 1) & (self.data['ST_Direction'].shift(1) != 1)
        self.data['ST_Sell'] = (self.data['ST_Direction'] == -1) & (self.data['ST_Direction'].shift(1) != -1)
        self.data['EMA_Buy'] = (
                (self.data['EMA_Short'] > self.data['EMA_Long']) &
                (self.data['EMA_Short'].shift(1) <= self.data['EMA_Long'].shift(1))
        )
        self.data['EMA_Sell'] = (
                (self.data['EMA_Short'] < self.data['EMA_Long']) &
                (self.data['EMA_Short'].shift(1) >= self.data['EMA_Long'].shift(1))
        )

        self.data['Buy_Signal'] = self.data['ST_Buy'] & self.data['EMA_Buy']
        self.data['Sell_Signal'] = self.data['ST_Sell'] & self.data['EMA_Sell']

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
    
    def save_to_excel(self, filename: str = "SuperTrendEMAStrategy_metrics.xlsx"):
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
        Plot price, SuperTrend bands, buy/sell signals, and the strategy equity curve.
        Visualizes the main price series with SuperTrend upper/lower bands, marks trade entries and exits,
        and displays the corresponding equity curve on a separate panel.
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

        # Signals
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['UpperBand'],
                mode='lines',
                line=dict(color='red', width=2),
                name='ST Upper Band'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['LowerBand'],
                mode='lines',
                line=dict(color='green', width=2),
                name='ST Lower Band'
            )
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

    # Obtener Datos
    df = yf.download('USDCLP=X', start='2024-01-01', interval='1d')
    df.columns = df.columns.droplevel(1)

    strategy = SuperTrendEMAStrategy(df)
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
        'st_period': np.arange(2, 41).tolist(),
        'st_multiplier': [round(x, 1) for x in np.arange(1.0, 8.0, 0.1).tolist()],
        'ema_short_period': np.arange(2, 41).tolist(),
        'ema_long_period': np.arange(20, 61).tolist()
    }
    
    best_params = strategy.optimize(params_grid, n_iter=50_000)

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
    strategy.plot(last_entries=252 * 5)

    # Write Excel
    strategy.save_to_excel()
