from libraries import *
from funtions import Position, BacktestingCapCOM
from Objetive import hyperparams
from Indicadores import Indicadores
from metrics import Metrics


def backtest(data, trial) -> tuple[list, dict, float]:
    """
    Executes a backtest on the provided dataset using specified hyperparameters.

    Parameters:
    -----------
    data : pd.DataFrame
        Historical price data, must include a 'Close' column.
    trial : optuna.trial.Trial
        Optuna trial object to suggest hyperparameters.

    Returns:
    --------
    port_hist : list[float]
        History of portfolio values at each timestep.
    metrics : dict
        Dictionary of performance metrics including Sharpe, Sortino, Win Rate, Max Drawdown, Calmar, 
        and Win Rate on Long Positions.
    cash : float
        Final available cash at the end of the backtest.

    Notes:
    ------
    The function:
    1. Initializes capital and commission.
    2. Retrieves hyperparameters from Optuna trial.
    3. Generates buy/sell signals based on RSI, momentum, and volatility indicators.
    4. Iterates over the data to open and close positions according to signals.
    5. Computes portfolio value at each step.
    6. Calculates final metrics based on closed trades and portfolio returns.
    """
    data = data.copy()

    # --- Initial capital ---
    capital = BacktestingCapCOM.initial_capital
    cash = float(capital)

    # --- Hyperparameters ---
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

    # --- Generate signals ---
    buy_rsi, sell_rsi = Indicadores.get_rsi(
        data, rsi_window, rsi_upper, rsi_lower)
    buy_momentum, sell_momentum = Indicadores.get_momentum(
        data, momentum_window, momentum_threshold)
    buy_volatility, sell_volatility = Indicadores.get_volatility(
        data, volatility_window, volatility_threshold)

    data["buy_signal"] = buy_rsi & buy_momentum & buy_volatility
    data["sell_signal"] = sell_rsi & sell_momentum & sell_volatility

    # --- DEBUG: number of signals ---
    print("Active BUY signals:", data["buy_signal"].sum())
    print("Active SELL signals:", data["sell_signal"].sum())

    # --- Active and closed positions ---
    active_long_positions: list[Position] = []
    active_short_positions: list[Position] = []
    closed_long_positions: list[Position] = []
    closed_short_positions: list[Position] = []

    # --- Portfolio history ---
    port_hist = [cash]

    # --- Iterate over dataset ---
    for idx, row in data.iterrows():
        price = row.Close

        # === CLOSE POSITIONS ===
        for pos in active_long_positions.copy():
            if (pos.sl >= price) or (pos.tp <= price):
                cash += price * pos.n_shares * (1 - COM)
                pos.exit_price = price
                pos.profit = (price - pos.price) * pos.n_shares
                closed_long_positions.append(pos)
                active_long_positions.remove(pos)

        for pos in active_short_positions.copy():
            if (pos.sl <= price) or (pos.tp >= price):
                pnl = (pos.price - price) * pos.n_shares * (1 - COM)
                cash += (pos.price * pos.n_shares) + pnl
                pos.exit_price = price
                pos.profit = (pos.price - price) * pos.n_shares
                closed_short_positions.append(pos)
                active_short_positions.remove(pos)

        # === OPEN POSITIONS ===
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

        # === PORTFOLIO VALUE ===
        port_value = cash
        for pos in active_long_positions:
            port_value += price * pos.n_shares
        for pos in active_short_positions:
            port_value += (pos.price * pos.n_shares) + \
                (pos.price - price) * pos.n_shares
        port_hist.append(port_value)

    # === METRICS (final) ===
    df = pd.DataFrame({'PortValue': port_hist})
    df['Returns'] = df.PortValue.pct_change().fillna(0)

    # Extract profits from closed positions
    long_profits = [
        pos.profit for pos in closed_long_positions if pos.profit is not None]
    short_profits = [
        pos.profit for pos in closed_short_positions if pos.profit is not None]

    metrics = {
        'Sharpe': Metrics(df.Returns).sharpe,
        'Sortino': Metrics(df.Returns).sortino,
        'Win Rate': sum(p > 0 for p in long_profits + short_profits) / max(1, len(long_profits + short_profits)),
        'Max Drawdown': Metrics(df.Returns).max_drawdown,
        'Calmar': Metrics(df.Returns).calmar,
        'Win Rate on Long Positions': sum(p > 0 for p in long_profits) / max(1, len(long_profits))
    }

    # --- DEBUG: closed positions ---
    print("Closed LONG positions:", len(closed_long_positions))
    print("Closed SHORT positions:", len(closed_short_positions))

    return port_hist, metrics, cash
