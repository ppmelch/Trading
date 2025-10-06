from libraries import *
from dataclasses import dataclass


@dataclass
class Indicadores:
    """
    Class containing static methods for technical indicators used in backtesting.
    Each method returns buy and sell signals based on the indicator.
    """

    @staticmethod
    def get_rsi(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
        df = data.copy()
        macd = ta.trend.MACD(df['Close'], window_slow=slow, window_fast=fast, window_sign=signal)
        df['MACD_diff'] = macd.macd_diff()  # MACD - Signal
        buy_signal = (df['MACD_diff'] > 0).astype(int)
        sell_signal = (df['MACD_diff'] < 0).astype(int)
        return buy_signal.fillna(0), sell_signal.fillna(0)


    @staticmethod
    def get_momentum(data: pd.DataFrame, windows: int, threshold: float)-> tuple[pd.Series, pd.Series]:
        """
        Calculate momentum (Rate of Change) and generate buy/sell signals.

        Args:
            data (pd.DataFrame): DataFrame containing 'Close' prices.
            windows (int): Lookback period for momentum calculation.
            threshold (float): Threshold for generating buy/sell signals.

        Returns:
            tuple(pd.Series, pd.Series): Buy and sell signals (1 = signal, 0 = no signal).
        """
        windows = min(windows, len(data)-1)
        data['Momentum'] = ta.momentum.ROCIndicator(
            data['Close'], window=windows).roc()
        buy_signal = ((data['Momentum'] > threshold)).astype(int).fillna(0)
        sell_signal = ((data['Momentum'] < -threshold)).astype(int).fillna(0)
        return buy_signal, sell_signal

    @staticmethod
    def get_volatility(data: pd.DataFrame, windows: int, volatility_threshold: float)-> tuple[pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands-based volatility and generate buy/sell signals.

        Args:
            data (pd.DataFrame): DataFrame containing 'Close' prices.
            windows (int): Lookback period for volatility calculation.
            volatility_threshold (float): Threshold for generating buy/sell signals.

        Returns:
            tuple(pd.Series, pd.Series): Buy and sell signals (1 = signal, 0 = no signal).
        """
        windows = min(windows, len(data)-1)
        data['Volatility'] = ta.volatility.BollingerBands(
            data['Close'], window=windows).bollinger_wband()
        buy_signal = ((data['Volatility'] < volatility_threshold)).astype(
            int).fillna(0)
        sell_signal = ((data['Volatility'] > volatility_threshold)).astype(
            int).fillna(0)
        return buy_signal, sell_signal
