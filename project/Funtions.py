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

class BacktestingCapCOM:
    initial_capital = 1_000_000
    COM = 0.125 / 100  # Comisión por operación (0.125%)

class OptunaOpt:
    direction='maximize'
    n_trials=1
    n_jobs=-1


def dateset_split(data, train, test, validation):
    """
    Split dataset into train, test, validation sets.
    train + test + validation must equal 1.
    """
    n = len(data)
    train_end = int(n * train)
    test_end = train_end + int(n * test)

    train = data.iloc[:train_end]
    test = data.iloc[train_end:test_end]
    validation = data.iloc[test_end:]

    return train, test, validation


def get_portfolio_value(cash: float, long_ops: list, short_ops: list, current_price: float, n_shares: float, COM: float) -> float:
    value = cash

    # Valor de las posiciones largas
    for pos in long_ops:
        value += current_price * pos.n_shares * (1 - COM)

    # Valor de las posiciones cortas
    for pos in short_ops:
        pnl = (pos.price - current_price) * pos.n_shares  # ganancia o pérdida
        value += pnl - (current_price * pos.n_shares * COM)

    return value



