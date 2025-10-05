from libraries import *

# Indicators and Hyperparameters
# -- RSI: Measures overbought and oversold momentum
# -- Momentum: Measures the speed of price change
# -- Volatility: Measures price variation over time

def hyperparams(trial) -> dict:
    """
    Define the hyperparameter search space for the backtesting strategy (corrected ranges).

    Args:
        trial (optuna.trial.Trial): An Optuna trial object used to suggest hyperparameter values.

    Returns:
        dict: Dictionary containing hyperparameters and their suggested values.
    """
    return {
        
        "rsi_window": trial.suggest_int("rsi_window", 14, 20),
        "rsi_lower": trial.suggest_int("rsi_lower", 30, 40),
        "rsi_upper": trial.suggest_int("rsi_upper", 60, 70),
        "momentum_window": trial.suggest_int("momentum_window", 10, 20),
        "momentum_threshold": trial.suggest_float("momentum_threshold", 0.05, 0.15),
        "volatility_window": trial.suggest_int("volatility_window", 10, 15),
        "volatility_threshold": trial.suggest_float("volatility_threshold", 0.01, 0.02),
        "stop_loss": trial.suggest_float("stop_loss", 0.01, 0.02),
        "take_profit": trial.suggest_float("take_profit", 0.02, 0.04),
        "capital_pct_exp": trial.suggest_float("capital_pct_exp", 0.05, 0.15)

    }