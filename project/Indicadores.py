
from libraries import *

def indicators(data, capital, trial) -> float:

    data= data.copy()
    # Indicadores e Hyperparametros
    # -- RSI -- Es el momentum de sobrecompra y sobreventa
    # -- Momentum -- Mide la velocidad del cambio del precio
    # -- Volatilidad -- Mide la variación del precio en el tiempo

    # --- Hiperparámetros ---
    rsi_window = trial.suggest_int('rsi_window', 5, 50)
    rsi_lower = trial.suggest_int('rsi_lower', 5, 35)
    rsi_upper = trial.suggest_int('rsi_upper', 65, 95)

    momentum_window = trial.suggest_int('momentum_window', 5, 50)
    momentum_threshold = trial.suggest_float('momentum_threshold', 0, 2)

    volatility_window = trial.suggest_int('volatility_window', 5, 30)
    volatility_threshold = trial.suggest_float('volatility_threshold', 1, 4)

    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.05)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.15)
    n_shares = trial.suggest_int('n_shares', 50, 500)

    # --- Indicadores ---
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
    data['Momentum'] = ta.momentum.MomentumIndicator(data['Close'], window=momentum_window).momentum()
    data['Volatility'] = ta.volatility.BollingerBands(data['Close'], window=volatility_window).bollinger_wband()

    # --- Señales ---
    data['buy_signal'] = ((data['RSI'] < rsi_lower) & 
                        (data['Momentum'] > momentum_threshold) &
                        (data['Volatility'] < volatility_threshold))

    data['sell_signal'] = ((data['RSI'] > rsi_upper) &
                        (data['Momentum'] < -momentum_threshold) &
                        (data['Volatility'] > volatility_threshold))
    
    # --- Backtest ---
    
    


