from libraries import *

class Metrics:
    """
    Wrapper around an equity curve (portfolio values) 
    to compute performance metrics as attributes.
    """

    def __init__(self, data: pd.Series):
        """
        Initialize the class with the portfolio equity curve.
        
        Parameters:
        - data: Series of portfolio values.
        
        Attributes:
        - returns: Daily percentage returns, computed with pct_change().
        """
        self.data = data
        self.returns = data.pct_change().dropna()

    @property
    def sharpe(self) -> float:
        """
        Compute the annualized Sharpe Ratio.
        
        Formula:
            (mean of returns / standard deviation of returns) * sqrt(252)
        
        Measures risk-adjusted return using total volatility.
        If volatility is zero, returns 0.0.
        """
        mean_ret = self.returns.mean()
        std_ret = self.returns.std()
        return (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0.0

    @property
    def sortino(self) -> float:
        """
        Compute the annualized Sortino Ratio.
        
        Similar to Sharpe but only penalizes downside volatility.
        
        Formula:
            (mean of returns / standard deviation of negative returns) * sqrt(252)
        
        Useful when focusing only on downside risk.
        """
        mean_ret = self.returns.mean()
        downside_std = self.returns[self.returns < 0].std()
        return (mean_ret / downside_std) * np.sqrt(252) if downside_std != 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """
        Compute the Maximum Drawdown (MDD).
        
        Measures the largest peak-to-trough decline in the equity curve.
        
        Formula:
            (current_value / rolling_max - 1)
        
        Returns the worst drawdown as a negative fraction 
        (e.g., -0.25 = -25%).
        """
        rolling_max = self.data.cummax()
        drawdowns = (self.data / rolling_max - 1)
        return drawdowns.min()

    @property
    def calmar(self) -> float:
        """
        Compute the Calmar Ratio.
        
        Formula:
            (annualized return / absolute max drawdown)
        
        Measures return relative to the worst drawdown.
        If MDD = 0, returns 0.0.
        """
        annual_return = self.returns.mean() * 252
        mdd = self.max_drawdown
        return annual_return / abs(mdd) if mdd != 0 else 0.0

    @property
    def win_rate(closed_positions) -> float:
        """
        Compute the Win Rate (success ratio).
        
        Parameters:
        - closed_positions: List of closed trades with a 'profit' attribute.
        
        Formula:
            (number of winning trades / total trades)
        
        If there are no trades, returns 0.
        """
        if not closed_positions:
            return 0
        n_wins = sum(1 for pos in closed_positions if pos.profit)
        return n_wins / len(closed_positions)



