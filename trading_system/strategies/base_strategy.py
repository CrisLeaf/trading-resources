from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    This template defines the fundamental structure that all trading strategies should follow.
    It provides essential methods and attributes for data management, parameter handling, logging, reproducibility, and results storage.
    Concrete strategies must inherit from this class and implement the defined abstract methods.

    Attributes:
        data (pd.DataFrame): DataFrame containing market data (OHLCV).
        params (dict): Dictionary of strategy-specific parameters.
        logger (Any, optional): Logger object for tracking events or messages.
        random_seed (int, optional): Seed for reproducibility in random processes.
        results (Any): Stores the results of the strategy after execution.

    Abstract methods to implement:
        calculate_signals(): Calculate buy/sell signals according to the strategy logic.
        evaluate_performance(): Evaluate the strategy's performance and return relevant metrics.
        optimize(params_grid): Optimize strategy parameters to maximize performance.
        plot(*args, **kwargs): Plot signals and results on the price chart.

    Utility methods:
        get_params(): Return a copy of the current strategy parameters.
        set_params(params): Update the strategy parameters.

    Notes:
        - It is recommended to implement additional validations in subclasses according to each strategy's logic.
        - The logger is optional but useful for debugging and tracking.
        - Reproducibility is important for testing and optimization, so setting the random seed is recommended if needed.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            params: Optional[Dict[str, Any]] = None,
            logger: Optional[Any] = None,
            random_seed: Optional[int] = None
        ):
        """
        Initialize the strategy with market data and optional parameters.

        Parameters:
        - data: pandas DataFrame containing OHLCV data (Open, High, Low, Close, Volume).
        - params: dictionary with strategy-specific parameters.
        - logger: optional logger for tracking events.
        - random_seed: optional random seed for reproducibility.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        
        self.data = data.copy()
        self.params = params.copy() if params else {}
        self.logger = logger
        self.random_seed = random_seed
        self.results = None

    @abstractmethod
    def calculate_signals(self) -> pd.DataFrame:
        """
        Calculate buy and sell signals based on the strategy logic.
        Should return a DataFrame with signals (e.g., buy/sell columns).
        """
        pass

    @abstractmethod
    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Evaluate the performance of the strategy.
        Should return a dictionary with metrics such as total return, Sharpe ratio, etc.
        """
        pass

    @abstractmethod
    def optimize(self, params_grid: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the strategy parameters to maximize performance.
        
        Parameters:
        - params_grid: dictionary with parameter ranges to test.
        
        Should return the best parameters found.
        """
        pass
    
    @abstractmethod
    def plot(self, *args, **kwargs):
        """
        Plot the strategy signals and results on top of the price chart.
        Can be customized in child classes.
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Return the current parameters of the strategy."""
        return self.params

    def set_params(self, params: Dict[str, Any]):
        """Update the strategy parameters."""
        self.params.update(params)
    