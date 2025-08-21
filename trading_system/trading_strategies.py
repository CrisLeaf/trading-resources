from strategies.strategy_example1 import MACD_RSI_BB_Strategy


class TradingStrategiesManager:
    """
    Manager class that integrates and executes multiple trading strategies.
    """

    def __init__(self, strategy_params: dict) -> None:
        """
        Constructor.

        Parameters
        ----------
        strategy_params : dict
            Dictionary containing parameters for each trading strategy.
        """
        self.strategies = {
            'MACD_RSI_BB': MACD_RSI_BB_Strategy(**strategy_params.get('MACD_RSI_BB', {})),
            # Other strategies can be added here
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def run_strategy(self, strategy_name: str, allow_shorts: bool) -> dict:
        """
        Run a specific trading strategy.

        Parameters
        ----------
        strategy_name : str
            The name of the strategy to execute (must match a key in self.strategies).

        Returns
        -------
        dict
            Results of the strategy execution.
        """
        strategy = self.strategies.get(strategy_name)
        
        if strategy is None:
            raise ValueError(f"Strategy '{strategy_name}' not found.")
        
        _ = strategy.calculate_signals()
        performance_dict = strategy.evaluate_performance(allow_shorts=allow_shorts)

        return performance_dict

    def run_all_strategies(self, allow_shorts: bool, verbose: bool = False) -> dict:
        """
        Run all available trading strategies.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the strategy being executed.

        Returns
        -------
        dict
            Dictionary containing the results of all strategies.
        """
        results = {}
        
        for name, _ in self.strategies.items():
            if verbose:
                print(f"Running {name}...")
            
            results[name] = self.run_strategy(name, allow_shorts=allow_shorts)
            
        return results


if __name__ == "__main__":
    import yfinance as yf
    import time

    df = yf.download("AAPL", start="2022-01-01", end="2025-01-01")
    df.columns = df.columns.droplevel(1)

    manager = TradingStrategiesManager(strategy_params={
        "MACD_RSI_BB": {"data": df}
    })

    performance_dict = manager.run_strategy("MACD_RSI_BB", allow_shorts=False)
    print(performance_dict)

    print()
    all_performances = manager.run_all_strategies(allow_shorts=False)
    print(all_performances)