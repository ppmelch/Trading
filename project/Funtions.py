from libraries import *
from dataclasses import dataclass
from typing import Optional



@dataclass
class Position:
    ticker: str
    n_shares: int
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
    n_trials=50
    n_jobs=-1


def dateset_split(data:pd.DataFrame, train:float,test:float ,validation:float): 
    data.copy()
    n = len(data)


def get_portfolio_value(cash: float, long_ops: list, short_ops: list, current_price: float, n_shares: int, COM: float) -> float:
    value = cash

    # Valor de las posiciones largas
    for pos in long_ops:
        value += current_price * pos.n_shares * (1 - COM)

    # Valor de las posiciones cortas
    for pos in short_ops:
        pnl = (pos.price - current_price) * pos.n_shares  # ganancia o pérdida
        value += pnl - (current_price * pos.n_shares * COM)

    return value

