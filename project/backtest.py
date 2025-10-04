from libraries import *
from metrics import Metrics
from Objetive import hyperparams
from Indicadores import Indicadores
from funtions import Position, BacktestingCapCOM



def backtest(data, trial_or_params) -> tuple[list, dict, float]:
    """
    Executes a backtest using RSI, Momentum, and Volatility strategies.

    The function can receive either:
        - An Optuna trial object (to generate hyperparameters using `hyperparams`)
        - A dictionary of parameters (for testing or final evaluation)

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data containing at least a 'Close' column.
    trial_or_params : optuna.trial.Trial or dict
        Optuna trial object or a dictionary of strategy parameters.

    Returns
    -------
    tuple
        - port_value (list): Portfolio value over time.
        - metrics_dict (dict): Dictionary of performance metrics (currently only 'Calmar').
        - final_cash (float): Final cash value after closing all positions.
    """
    data = data.copy().reset_index(drop=True)

    # Determine parameters
    if isinstance(trial_or_params, dict):
        params = trial_or_params
    else:
        params = hyperparams(trial_or_params)

    rsi_window = params["rsi_window"]
    rsi_lower = params["rsi_lower"]
    rsi_upper = params["rsi_upper"]

    momentum_window = params["momentum_window"]
    momentum_threshold = params["momentum_threshold"]

    volatility_window = params["volatility_window"]
    volatility_threshold = params["volatility_threshold"]

    stop_loss = params["stop_loss"]
    take_profit = params["take_profit"]
    capital_pct_exp = params.get("capital_pct_exp", 0.1)  # default 10%

    # --- Commission and initial capital ---
    COM = BacktestingCapCOM.COM
    cash = BacktestingCapCOM.initial_capital

    # --- Signals ---
    buy_rsi, sell_rsi = Indicadores.get_rsi(
        data, rsi_window, rsi_upper, rsi_lower)
    buy_momentum, sell_momentum = Indicadores.get_momentum(
        data, momentum_window, momentum_threshold)
    buy_volatility, sell_volatility = Indicadores.get_volatility(
        data, volatility_window, volatility_threshold)

    historic = data.copy()
    historic["buy_signal"] = (buy_rsi.astype(
        int) + buy_momentum.astype(int) + buy_volatility.astype(int)) >= 2
    historic["sell_signal"] = (sell_rsi.astype(
        int) + sell_momentum.astype(int) + sell_volatility.astype(int)) >= 2
    historic = historic.dropna().reset_index(drop=True)

    active_long_positions, active_short_positions, port_value = [], [], []

    for i, row in historic.iterrows():
        price = row.Close
        n_shares = (cash * capital_pct_exp) / price

        # Close LONG positions
        for pos in active_long_positions.copy():
            if price >= pos.tp or price <= pos.sl:
                cash += price * pos.n_shares * (1 - COM)
                active_long_positions.remove(pos)

        # Close SHORT positions
        for pos in active_short_positions.copy():
            if price <= pos.tp or price >= pos.sl:
                pnl = (pos.price - price) * pos.n_shares * (1 - COM)
                cash += (pos.price * pos.n_shares * (1 + COM)) + pnl
                active_short_positions.remove(pos)

        # Open LONG positions
        if row.buy_signal and cash > price * n_shares * (1 + COM):
            cash -= price * n_shares * (1 + COM)
            active_long_positions.append(Position(price=price, n_shares=n_shares,
                                                  sl=price*(1-stop_loss), tp=price*(1+take_profit)))

        # Open SHORT positions
        if row.sell_signal and cash > price * n_shares * (1 + COM):
            cash -= price * n_shares * (1 + COM)
            active_short_positions.append(Position(price=price, n_shares=n_shares,
                                                   sl=price*(1+stop_loss), tp=price*(1-take_profit)))

        # Portfolio value
        port_value.append(
            cash
            + sum(price * pos.n_shares for pos in active_long_positions)
            + sum((pos.price * pos.n_shares) + (pos.price - price)
                  * pos.n_shares for pos in active_short_positions)
        )

    # Close remaining positions
    for pos in active_long_positions:
        cash += price * pos.n_shares * (1 - COM)
    for pos in active_short_positions:
        pnl = (pos.price - price) * pos.n_shares * (1 - COM)
        cash += (pos.price * pos.n_shares * (1 + COM)) + pnl

    # Compute metrics
    metrics_dict = {"Calmar": Metrics(pd.DataFrame({"PortValue": port_value}).PortValue).calmar,
                    "Sharpe": Metrics(pd.DataFrame({"PortValue": port_value}).PortValue).sharpe,
                    "Sortino": Metrics(pd.DataFrame({"PortValue": port_value}).PortValue).sortino,
                    "Maximum Drawdown": Metrics(pd.DataFrame({"PortValue": port_value}).PortValue).max_drawdown,
                    "Win_Rate": Metrics(pd.DataFrame({"PortValue": port_value}).PortValue).win_rate}


    return port_value, metrics_dict, cash
