
from libraries import *
from Objetive import hyperparams
from dataclasses import dataclass

# Indicadores e Hyperparametros
# -- RSI -- Es el momentum de sobrecompra y sobreventa
# -- Momentum -- Mide la velocidad del cambio del precio
# -- Volatilidad -- Mide la variación del precio en el tiempo

# --- Función Indicadores ---


@dataclass
class Indicadores:
    # --- RSI ---
    @staticmethod
    def get_rsi(data: pd.DataFrame, windows: int, rsi_upper: int, rsi_lower: int) -> pd.Series:

        data['RSI'] = ta.momentum.RSIIndicator(
            data['Close'], window=windows).rsi()

        buy_signal = ((data['RSI'] < rsi_lower)).astype(int)

        sell_signal = ((data['RSI'] > rsi_upper)).astype(int)

        return buy_signal, sell_signal

    # --- Momentum ---

    @staticmethod
    def get_momentum(data: pd.DataFrame, windows: int, threshold: float) -> pd.Series:

        data['Momentum'] = ta.momentum.ROCIndicator(
            data['Close'], window=windows).roc()

        buy_signal = ((data['Momentum'] > threshold)).astype(int)

        sell_signal = ((data['Momentum'] < -threshold)).astype(int)

        return buy_signal, sell_signal

    # --- Volatility ---
    @staticmethod
    def get_volatility(data: pd.DataFrame, windows: int, volatility_threshold: float) -> pd.Series:

        data['Volatility'] = ta.volatility.BollingerBands(
            data['Close'], window=windows).bollinger_wband()

        buy_signal = ((data['Volatility'] < volatility_threshold)).astype(int)

        sell_signal = ((data['Volatility'] > volatility_threshold)).astype(int)

        return buy_signal, sell_signal
