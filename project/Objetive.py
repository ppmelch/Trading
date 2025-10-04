from libraries import *

# Indicators and Hyperparameters
# -- RSI: Measures overbought and oversold momentum
# -- Momentum: Measures the speed of price change
# -- Volatility: Measures price variation over time


def hyperparams(trial) -> dict:
    """
    Define the hyperparameter search space for the backtesting strategy.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object used to suggest hyperparameter values.

    Returns:
        dict: Dictionary containing hyperparameters and their suggested values.

    Hyperparameters:
        rsi_window (int): Lookback period for RSI calculation (5-90).
        rsi_lower (int): RSI threshold for generating buy signals (5-35).
        rsi_upper (int): RSI threshold for generating sell signals (65-95).
        momentum_window (int): Lookback period for momentum calculation (5-50).
        momentum_threshold (float): Threshold for generating momentum buy/sell signals (0.1-2.0).
        volatility_window (int): Lookback period for volatility calculation (5-50).
        volatility_threshold (float): Threshold for generating volatility buy/sell signals (0.01-0.5).
        stop_loss (float): Stop-loss ratio for positions (0.01-0.15).
        take_profit (float): Take-profit ratio for positions (0.01-0.2).
        n_shares (int): Number of shares to trade per position (1-80).
    """
    return {
        "rsi_window": trial.suggest_int("rsi_window", 5, 90),

        "rsi_lower": trial.suggest_int("rsi_lower", 5, 35),

        "rsi_upper": trial.suggest_int("rsi_upper", 65, 95),

        "momentum_window": trial.suggest_int("momentum_window", 5, 50),

        "momentum_threshold": trial.suggest_float("momentum_threshold", 0.1, 2.0),

        "volatility_window": trial.suggest_int("volatility_window", 5, 50),

        "volatility_threshold": trial.suggest_float("volatility_threshold", 0.01, 0.5),

        "stop_loss": trial.suggest_float("stop_loss", 0.01, 0.15),

        "take_profit": trial.suggest_float("take_profit", 0.01, 0.2),

        "n_shares": trial.suggest_int("n_shares", 1, 80),
    }
