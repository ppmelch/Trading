from libraries import *
from backtest import backtest
from project.metrics import Metrics 
from funtions import BacktestingCapCOM

# Indicadores e Hyperparametros
# -- RSI -- Es el momentum de sobrecompra y sobreventa
# -- Momentum -- Mide la velocidad del cambio del precio
# -- Volatilidad -- Mide la variación del precio en el tiempo

# --- Hiperparámetros ---

def hyperparams(trial) -> dict:

    return {
        "rsi_window": trial.suggest_int("rsi_window", 5, 50),

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

