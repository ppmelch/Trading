from libraries import *
from dataclasses import dataclass


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

