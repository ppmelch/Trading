from libraries import *


class Metrics:
    """
    Wrapper around an equity curve (portfolio values) 
    to compute performance metrics as attributes.
    """

    def __init__(self, data: pd.Series):
        self.data = data
        self.returns = data.pct_change().dropna()

    @property
    def sharpe(self) -> float:
        """Sharpe ratio (annualized, 252 trading days)."""
        mean_ret = self.returns.mean()
        std_ret = self.returns.std()
        return (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0

    @property
    def sortino(self) -> float:
        """Sortino ratio (annualized)."""
        mean_ret = self.returns.mean()
        downside_std = self.returns[self.returns < 0].std()
        return (mean_ret / downside_std) * np.sqrt(252) if downside_std != 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown (fraction, e.g., -0.25 = -25%)."""
        rolling_max = self.data.cummax()
        drawdowns = (self.data / rolling_max - 1)
        return drawdowns.min()

    @property
    def calmar(self) -> float:
        """Calmar ratio = annual return / max drawdown."""
        annual_return = self.returns.mean() * 252
        mdd = self.max_drawdown
        return annual_return / abs(mdd) if mdd != 0 else 0.0

    @property
    def win_rate(self) -> float:
        """
        Placeholder: needs trade data.
        For now, counts positive return days vs total days.
        """
        wins = (self.returns > 0).sum()
        total = len(self.returns)
        return wins / total if total > 0 else np.nan
    
