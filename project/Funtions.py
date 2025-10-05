from libraries import *
from dataclasses import dataclass


@dataclass
class Position:
    """
    Represents a position in the portfolio.

    Attributes:
    n_shares : float
        Number of shares or units in the position.
    price : float
        Entry price of the position.
    sl : float
        Stop loss price.
    tp : float
        Take profit price.
    profit : float, optional
        Realized profit or loss of the position (default: None).
    exit_price : float, optional
        Exit price of the position (default: None).
    """
    n_shares: float
    price: float
    sl: float
    tp: float
    profit: float = None
    exit_price: float = None


@dataclass
class BacktestingCapCOM:
    """
    Backtesting configuration with initial capital and commission.

    Attributes:
    initial_capital : float
        Initial capital for backtesting (default: 1_000_000).
    COM : float
        Commission per trade in percentage (default: 0.125 / 100).
    """
    initial_capital: float = 1_000_000
    COM: float = 0.125 / 100


@dataclass
class OptunaOpt:
    """
    Optuna optimization configuration.

    Attributes:
    direction : str
        Optimization direction ('maximize' or 'minimize').
    n_trials : int
        Number of optimization trials.
    n_jobs : int
        Number of parallel jobs (-1 uses all cores).
    n_splits : int
        Number of cross-validation splits.
    show_progress_bar : bool
        Show Optuna progress bar.
    """
    direction: str = 'maximize'
    n_trials: int = 5
    n_jobs: int = -1
    n_splits: int = 5
    show_progress_bar: bool = True


def dateset_split(data: pd.DataFrame, train: float, test: float, validation: float):
    """
    Splits a DataFrame into training, testing, and validation sets.

    Parameters:
    data : pd.DataFrame
        Complete dataset to split.
    train : float
        Fraction of data to use for training.
    test : float
        Fraction of data to use for testing.
    validation : float
        Fraction of data to use for validation.

    Returns:
    tuple
        train_data, test_data, validation_data
    """
    n = len(data)
    train_size = int(n * train)
    test_size = train_size + int(n * test)

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:test_size]
    validation_data = data.iloc[test_size:]

    return train_data, test_data, validation_data


def get_portfolio_value(cash: float, long_ops: list[Position], short_ops: list[Position], current_price: float, n_shares: float) -> float:
    """
    Calculates the total portfolio value at a given moment.

    Parameters:
    cash : float
        Cash available in the portfolio.
    long_ops : list[Position]
        List of active long positions.
    short_ops : list[Position]
        List of active short positions.
    current_price : float
        Current price of the asset.
    n_shares : float
        Number of shares per position.

    Returns:
    float
        Total portfolio value including long and short positions.
    """
    value = cash
    for pos in long_ops:
        value += current_price * pos.n_shares
    for pos in short_ops:
        value += (pos.price * pos.n_shares) + \
            (pos.price - current_price) * pos.n_shares
    return value
