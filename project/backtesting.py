from libraries import *
from metrics import Metrics
from hyperparams import hyperparams
from indicators import Indicadores
from functions import Position, BacktestingCapCOM, get_portfolio_value


def backtest(data: pd.DataFrame, trial_or_params, initial_cash: float = None) -> tuple[list, dict, float]:
    """
    Executes a backtest using RSI, Momentum, and Volatility strategies.

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data containing at least a 'Close' column.
    trial_or_params : optuna.trial.Trial or dict
        Optuna trial object or dictionary of parameters.
    initial_cash : float, optional
        Starting capital for the backtest. Defaults to BacktestingCapCOM.initial_capital.

    Returns
    -------
    tuple
        - port_value (list): Portfolio value over time.
        - metrics_dict (dict): Performance metrics.
        - final_cash (float): Final cash after closing all positions.
    """
    data = data.copy().reset_index(drop=True)

    # --- Parameters ---
    params = trial_or_params if isinstance(
        trial_or_params, dict) else hyperparams(trial_or_params)

    rsi_window = params["rsi_window"]
    rsi_lower = params["rsi_lower"]
    rsi_upper = params["rsi_upper"]
    momentum_window = params["momentum_window"]
    momentum_threshold = params["momentum_threshold"]
    volatility_window = params["volatility_window"]
    volatility_threshold = params["volatility_threshold"]
    stop_loss = params["stop_loss"]
    take_profit = params["take_profit"]
    capital_pct_exp = params["capital_pct_exp"]

    # --- Commission & Capital ---
    COM = BacktestingCapCOM.COM
    cash = BacktestingCapCOM.initial_capital if initial_cash is None else initial_cash

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

    # --- Tracking ---
    active_long_positions, active_short_positions, port_value = [], [], [cash]
    closed_positions = []

    # --- Backtest Loop ---
    for i, row in historic.iterrows():
        price = row.Close
        n_shares = (cash * capital_pct_exp) / price

        # --- Close LONG positions ---
        for pos in active_long_positions.copy():
            if price >= pos.tp or price <= pos.sl:
                cash += price * pos.n_shares * (1 - COM)
                pos.profit = (price - pos.price) * pos.n_shares
                closed_positions.append(pos)
                active_long_positions.remove(pos)

        # --- Close SHORT positions ---
        for pos in active_short_positions.copy():
            if price <= pos.tp or price >= pos.sl:
                pnl = (pos.price - price) * pos.n_shares * (1 - COM)
                cash += (pos.price * pos.n_shares) * (1 + COM) + pnl
                pos.profit = pnl
                closed_positions.append(pos)
                active_short_positions.remove(pos)

                # --- Open LONG positions ---
        if row.buy_signal and not active_long_positions and not active_short_positions:
            if cash > price * n_shares * (1 + COM):
                cash -= price * n_shares * (1 + COM)
                active_long_positions.append(Position(
                    price=price, n_shares=n_shares,
                    sl=price*(1-stop_loss), tp=price*(1+take_profit)
                ))

        # --- Open SHORT positions ---
        if row.sell_signal and not active_short_positions and not active_long_positions:
            if cash > price * n_shares * (1 + COM):
                cash -= price * n_shares * (1 + COM)
                active_short_positions.append(Position(
                    price=price, n_shares=n_shares,
                    sl=price*(1+stop_loss), tp=price*(1-take_profit)
                ))

        # --- Portfolio value ---
        port_value.append(get_portfolio_value(
            cash, active_long_positions, active_short_positions, price, n_shares
        ))

    # --- Close remaining positions ---
    for pos in active_long_positions:
        cash += price * pos.n_shares * (1 - COM)
        pos.profit = (price - pos.price) * pos.n_shares
        closed_positions.append(pos)

    for pos in active_short_positions:
        pnl = (pos.price - price) * pos.n_shares * (1 - COM)
        cash += (pos.price * pos.n_shares) * (1 + COM) + pnl
        pos.profit = pnl
        closed_positions.append(pos)

    # --- Metrics ---
    port_series = pd.Series(port_value).replace(0, np.nan).dropna()
    metrics_obj = Metrics(port_series)

    final_value = port_value[-1]
    initial_value = port_value[0]
    profit = final_value - initial_value

    metrics_dict = {
        "Calmar": metrics_obj.calmar,
        "Sharpe": metrics_obj.sharpe,
        "Sortino": metrics_obj.sortino,
        "Maximum Drawdown": metrics_obj.max_drawdown,
        "Win Rate": Metrics.win_rate(closed_positions),
        "Total Return (%)": (final_value - initial_value) / initial_value * 100,
        "Profit ($)": f"${profit:,.2f}",
        "Final Capital ($)": f"${cash:,.2f}"
    }

    return port_value, metrics_dict, cash
