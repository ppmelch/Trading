from libraries import *
from funtions import Position, BacktestingCapCOM , OptunaOpt , get_portfolio_value
from Objetive import hyperparams
from dataclasses import dataclass
from Indicadores import get_rsi, get_momentum, get_volatility


def backtest(data, capital, trial) -> float:

    # --- Hiperparámetros ---
    data = data.copy()  

    params = hyperparams(trial)

    rsi_window = params["rsi_window"]
    rsi_lower = params["rsi_lower"]
    rsi_upper = params["rsi_upper"]
    momentum_window = params["momentum_window"]
    momentum_threshold = params["momentum_threshold"]
    volatility_window = params["volatility_window"]
    volatility_threshold = params["volatility_threshold"]
    stop_loss = params["stop_loss"]
    take_profit = params["take_profit"]
    n_shares = params["n_shares"]

    # --- Señales ---
    buy_rsi, sell_rsi = get_rsi(data, rsi_window, rsi_upper, rsi_lower)
    buy_momentum, sell_momentum = get_momentum(data, momentum_window, momentum_threshold)
    buy_volatility, sell_volatility = get_volatility(data, volatility_window, volatility_threshold)

    data["buy_signal"]  = buy_rsi & buy_momentum & buy_volatility
    data["sell_signal"] = sell_rsi & sell_momentum & sell_volatility

    # --- Backtest ---
    cash = float(capital)
    active_long_positions: list[Position] = []
    active_short_positions: list[Position] = []

    COM = BacktestingCapCOM.COM
    max_positions_per_side = 1

    portfolio_values = [BacktestingCapCOM.initial_capital]

    for i, row in data.iterrows():
        price = row.Close

        # --- Cierre LONG ---
        for pos in active_long_positions.copy():
            if price >= pos.tp or price <= pos.sl:
                cash += price * pos.n_shares * (1 - COM)
                active_long_positions.remove(pos)

        # --- Cierre SHORT ---
        for pos in active_short_positions.copy():
            if price <= pos.tp or price >= pos.sl:
                cash -= price * pos.n_shares * (1 + COM)
                active_short_positions.remove(pos)

        # --- Entrada LONG ---
        if row.buy_signal:
            if len(active_long_positions) < max_positions_per_side:
                cost = price * n_shares * (1 + COM)
                if cash >= cost:
                    cash -= cost
                    pos = Position(
                        ticker="BTCUSDT", n_shares=n_shares, price=price,
                        sl=price * (1 - stop_loss), tp=price * (1 + take_profit),
                        time=getattr(row, "Datetime", None), side="long"
                    )
                    active_long_positions.append(pos)

        # --- Entrada SHORT ---
        if row.sell_signal:
            if len(active_short_positions) < max_positions_per_side:
                entry_cash = price * n_shares * (1 - COM)
                cash += entry_cash
                pos = Position(
                    ticker="BTCUSDT", n_shares=n_shares, price=price,
                    sl=price * (1 + stop_loss), tp=price * (1 - take_profit),
                    time=getattr(row, "Datetime", None), side="short"
                )
                active_short_positions.append(pos)

        # --- Valor del portafolio ---
        unrealized_long  = sum((price - p.price) * p.n_shares for p in active_long_positions)
        unrealized_short = sum((p.price - price) * p.n_shares for p in active_short_positions)
        total_value = cash + unrealized_long + unrealized_short
        portfolio_values.append(total_value)

    # --- Cierre al último precio ---
    final_price = data.iloc[-1].Close
    for pos in active_long_positions:
        cash += final_price * pos.n_shares * (1 - COM)
    for pos in active_short_positions:
        cash -= final_price * pos.n_shares * (1 + COM)

    final_portfolio_value = cash
    retorno_relativo = float(final_portfolio_value / BacktestingCapCOM.initial_capital - 1)

    return retorno_relativo, portfolio_values


# Hacer solo una función de backtest , aplicando la optimización de hiperparámetros con Optuna
