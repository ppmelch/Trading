from libraries import *
from funtions import Position, BacktestingCapCOM, OptunaOpt, get_portfolio_value
from Objetive import hyperparams
from Indicadores import Indicadores, get_rsi, get_momentum, get_volatility


def backtest(data, sl, tp, n_shares, trial) -> float:
    # --- Copia del dataset ---
    data = data.copy()

    # --- Capital inicial ---
    capital = BacktestingCapCOM.initial_capital
    cash = float(capital)

    # --- Hiperparámetros ---
    params = hyperparams(trial)
    rsi_window = params["rsi_window"]
    rsi_lower = params["rsi_lower"]
    rsi_upper = params["rsi_upper"]
    momentum_window = params["momentum_window"]
    momentum_threshold = params["momentum_threshold"]
    volatility_window = params["volatility_window"]
    volatility_threshold = params["volatility_threshold"]
    sl = params["stop_loss"]
    tp = params["take_profit"]
    n_shares = params["n_shares"]

    COM = BacktestingCapCOM.COM

    # --- Señales ---
    buy_rsi, sell_rsi = Indicadores.get_rsi(data, rsi_window, rsi_upper, rsi_lower)
    buy_momentum, sell_momentum = Indicadores.get_momentum(
        data, momentum_window, momentum_threshold)
    buy_volatility, sell_volatility = Indicadores.get_volatility(
        data, volatility_window, volatility_threshold)

    data["buy_signal"] = buy_rsi & buy_momentum & buy_volatility
    data["sell_signal"] = sell_rsi & sell_momentum & sell_volatility

    # --- Posiciones activas ---
    active_long_positions: list[Position] = []
    active_short_positions: list[Position] = []

    # --- Historial del portafolio ---
    port_hist = []
    portafolio_value = cash

    # --- Iterar el DataFrame ---
    for i, row in data.iterrows():

        # === CIERRE DE POSICIONES ===
        for pos in active_long_positions.copy():
            if (pos.sl > row.Close) or (pos.tp < row.Close):
                cash += row.Close * pos.n_shares * (1 - COM)
                active_long_positions.remove(pos)

        for pos in active_short_positions.copy():
            if (pos.sl < row.Close) or (pos.tp > row.Close):
                cash += (pos.price * pos.n_shares) + \
                    (pos.price - row.Close) * n_shares * (1 - COM)
                active_short_positions.remove(pos)

        # === APERTURA DE POSICIONES ===
        if row.sell_signal:  # Entrada SHORT
            cost = row.Close * n_shares * (1 + COM)
            if cash >= cost:
                cash -= cost
                active_short_positions.append(Position(
                    price=row.Close,
                    n_shares=n_shares,
                    sl=row.Close * (1 + sl),
                    tp=row.Close * (1 - tp),
                ))

        if row.buy_signal:  # Entrada LONG
            cost = row.Close * n_shares * (1 + COM)
            if cash >= cost:
                cash -= cost
                active_long_positions.append(Position(
                    price=row.Close,
                    n_shares=n_shares,
                    sl=row.Close * (1 - sl),
                    tp=row.Close * (1 + tp),
                ))

        # === VALOR DEL PORTAFOLIO ===
        portafolio_value = cash

        for pos in active_long_positions:
            portafolio_value += row.Close * pos.n_shares

        for pos in active_short_positions:
            portafolio_value += (pos.price * pos.n_shares) + \
                (pos.price * n_shares - row.Close * n_shares)

        port_hist.append(portafolio_value)

    # --- Retorno final ---
    return port_hist
