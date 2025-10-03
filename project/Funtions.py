from libraries import *
from dataclasses import dataclass
from typing import Optional



@dataclass
class Position:
    ticker: str
    n_shares: float
    price: float
    sl: float
    tp: float
    time: str
    side: str  


@property
class BacktestingCapCOM:
    initial_capital = 1_000_000
    COM = 0.125 / 100  # Comisión por operación (0.125%)


class OptunaOpt:
    direction='maximize'
    n_trials: int =50
    n_jobs: int=-1
    n_jobs: int =1
    show_progress_bar: bool =True
    n_splits: int =5    




def dateset_split(data: pd.DataFrame, train: float, test: float, validation: float):
    """
    Split dataset into train, test, and validation sets.

    Args:
        data (pd.DataFrame): Input dataset.
        train (float): Fraction for training set.
        test (float): Fraction for test set.
        validation (float): Fraction for validation set.

    Returns:
        tuple: (train_data, test_data, validation_data)
    """
    n = len(data)
    train_size = int(n * train)
    test_size = train_size + int(n * test)

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:test_size]
    validation_data = data.iloc[test_size:]

    return train_data, test_data, validation_data


def get_portfolio_value(cash: float, long_ops: list[Position], short_ops: list[Position], current_price: float, n_shares: float) -> float:
    value = cash

    # Valor de las posiciones largas
    for pos in long_ops:
        value += current_price * pos.n_shares 

    # Valor de las posiciones cortas
    for pos in short_ops:
        value += (pos.price * pos.n_shares) + (pos.price - current_price) * pos.n_shares 

    return value



