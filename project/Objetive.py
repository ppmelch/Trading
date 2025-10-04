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
        # RSI
        "rsi_window": trial.suggest_int("rsi_window", 14, 30),  # Smoother RSI, less noise
        "rsi_lower": trial.suggest_int("rsi_lower", 30, 40),     # More realistic oversold
        "rsi_upper": trial.suggest_int("rsi_upper", 60, 70),     # More realistic overbought

        # Momentum
        "momentum_window": trial.suggest_int("momentum_window", 10, 30),  
        "momentum_threshold": trial.suggest_float("momentum_threshold", 0.05, 0.5),  # Sensible threshold

        # Volatility
        "volatility_window": trial.suggest_int("volatility_window", 10, 30),  
        "volatility_threshold": trial.suggest_float("volatility_threshold", 0.01, 0.2),  

        # Risk management
        "stop_loss": trial.suggest_float("stop_loss", 0.02, 0.1),    # 2%–10% loss limit
        "take_profit": trial.suggest_float("take_profit", 0.03, 0.04), # 3%–4% profit target

        # Position sizing
        "n_shares": trial.suggest_int("n_shares", 1, 15),  # Moderate number of shares
    }
