from libraries import *

# Indicators and Hyperparameters
# -- RSI: Measures overbought and oversold momentum
# -- Momentum: Measures the speed of price change
# -- Volatility: Measures price variation over time

def hyperparams(trial) -> dict:
    """
    Define the hyperparameter search space for the backtesting strategy (updated for volatility filter).

    Args:
        trial (optuna.trial.Trial): An Optuna trial object used to suggest hyperparameter values.

    Returns:
        dict: Dictionary containing hyperparameters and their suggested values.
    """
    return {
        # --- RSI ---
        "rsi_window": trial.suggest_int("rsi_window", 11, 25),
        "rsi_lower": trial.suggest_int("rsi_lower", 25, 35),
        "rsi_upper": trial.suggest_int("rsi_upper", 60, 80),

        # --- Momentum ---
        "momentum_window": trial.suggest_int("momentum_window", 10, 22),
        "momentum_threshold": trial.suggest_float("momentum_threshold", 0.02, 0.1),

        # --- Volatility filter ---
        "volatility_window": trial.suggest_int("volatility_window", 25, 35),  
        "volatility_quantile": trial.suggest_float("volatility_quantile", 0.6, 0.70),  

        # --- Risk management ---
        "stop_loss": trial.suggest_float("stop_loss", 0.02, 0.03),
        "take_profit": trial.suggest_float("take_profit", 0.05, 0.10),
        "capital_pct_exp": trial.suggest_float("capital_pct_exp", 0.05, 0.20),
    }
