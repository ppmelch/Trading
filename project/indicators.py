from libraries import *
from dataclasses import dataclass


@dataclass
class Indicadores:
    """
    Class containing static methods for technical indicators used in backtesting.
    Each method returns buy and sell signals based on the indicator.
    """

    @staticmethod
    def get_rsi(data: pd.DataFrame, windows: int, rsi_upper: int, rsi_lower: int):
        """
        Calculate RSI and generate buy/sell signals.

        Args:
            data (pd.DataFrame): DataFrame containing 'Close' prices.
            windows (int): Lookback period for RSI calculation.
            rsi_upper (int): RSI threshold above which to generate sell signal.
            rsi_lower (int): RSI threshold below which to generate buy signal.

        Returns:
            tuple(pd.Series, pd.Series): Buy and sell signals (1 = signal, 0 = no signal).
        """
        windows = min(windows, len(data)-1)
        data['RSI'] = ta.momentum.RSIIndicator(
            data['Close'], window=windows).rsi()
        buy_signal = ((data['RSI'] < rsi_lower)).astype(int).fillna(0)
        sell_signal = ((data['RSI'] > rsi_upper)).astype(int).fillna(0)
        return buy_signal, sell_signal

    @staticmethod
    def get_momentum(data: pd.DataFrame, windows: int, threshold: float):
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
    def get_volatility(data: pd.DataFrame, windows: int, volatility_threshold: float):
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
