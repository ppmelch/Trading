from libraries import *
from funtions import Position, BacktestingCapCOM, OptunaOpt, get_portfolio_value
from Objetive import hyperparams
from Indicadores import Indicadores
from metrics import Metrics

def backtest(data, sl, tp, n_shares, trial) -> tuple[list, dict, float]:
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
    buy_momentum, sell_momentum = Indicadores.get_momentum(data, momentum_window, momentum_threshold)
    buy_volatility, sell_volatility = Indicadores.get_volatility(data, volatility_window, volatility_threshold)

    data["buy_signal"] = buy_rsi & buy_momentum & buy_volatility
    data["sell_signal"] = sell_rsi & sell_momentum & sell_volatility

    # --- Posiciones activas ---
    active_long_positions: list[Position] = []
    active_short_positions: list[Position] = []
    closed_long_positions: list[Position] = []
    closed_short_positions: list[Position] = []

    # --- Historial del portafolio ---
    port_hist = [cash]

    # --- Iterar el DataFrame ---
    for _, row in data.iterrows():
        price = row.Close

        # === CIERRE DE POSICIONES ===
        for pos in active_long_positions.copy():
            if (pos.sl > price) or (pos.tp < price):
                cash += price * pos.n_shares * (1 - COM)
                pos.exit_price = price
                pos.profit = (price - pos.price) * pos.n_shares
                closed_long_positions.append(pos)
                active_long_positions.remove(pos)

        for pos in active_short_positions.copy():
            if (pos.sl < price) or (pos.tp > price):
                pnl = (pos.price - price) * pos.n_shares * (1 - COM)
                cash += (pos.price * pos.n_shares) + pnl
                pos.exit_price = price
                pos.profit = (pos.price - price) * pos.n_shares
                closed_short_positions.append(pos)
                active_short_positions.remove(pos)

        # === APERTURA DE POSICIONES ===
        if row.sell_signal:  # SHORT
            cost = price * n_shares * (1 + COM)
            if cash >= cost:
                cash -= cost
                active_short_positions.append(Position(
                    price=price,
                    n_shares=n_shares,
                    sl=price * (1 + sl),
                    tp=price * (1 - tp),
                ))

        if row.buy_signal:  # LONG
            cost = price * n_shares * (1 + COM)
            if cash >= cost:
                cash -= cost
                active_long_positions.append(Position(
                    price=price,
                    n_shares=n_shares,
                    sl=price * (1 - sl),
                    tp=price * (1 + tp),
                ))

        # === VALOR DEL PORTAFOLIO ===
        portafolio_value = cash
        for pos in active_long_positions:
            portafolio_value += price * pos.n_shares
        for pos in active_short_positions:
            portafolio_value += (pos.price * pos.n_shares) + (pos.price - price) * pos.n_shares

        port_hist.append(portafolio_value)

    # === MÉTRICAS (al final) ===
    df = pd.DataFrame({'PortValue': port_hist})
    df['Returns'] = df.PortValue.pct_change()
    df.dropna(inplace=True)

    metrics = {
        'Sharpe': Metrics(df.Returns).sharpe,
        'Sortino': Metrics(df.Returns).sortino,
        'Win Rate': Metrics(df.Returns).win_rate,
        'Max Drawdown': Metrics(df.Returns).max_drawdown,
        'Calmar': Metrics(df.Returns).calmar,
        'Win Rate on Long Positions': Metrics(closed_long_positions).win_rate
    }

    return port_hist, metrics, cash

