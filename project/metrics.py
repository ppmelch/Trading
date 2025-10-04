from libraries import *


class Metrics:
    """
    Computes common trading performance metrics for a series of returns or portfolio values.

    Attributes:
    -----------
    data : pd.Series
        Price or portfolio value series.
    returns : pd.Series
        Computed returns from the input series.
    """

    def __init__(self, data: pd.Series):
        """
        Initializes the Metrics object.

        Parameters:
        -----------
        data : pd.Series
            Historical price or portfolio data.
        """
        self.data = data
        self.returns = data.pct_change().dropna() if not data.empty else pd.Series(dtype=float)

    @property
    def sharpe(self) -> float:
        """
        Calculates the annualized Sharpe ratio.

        Returns:
        --------
        float
            Sharpe ratio, 0 if not computable.
        """
        if self.returns.empty:
            return 0.0
        mean_ret = self.returns.mean()
        std_ret = self.returns.std()
        if np.isnan(mean_ret) or np.isnan(std_ret) or std_ret == 0:
            return 0.0
        return (mean_ret / std_ret) * np.sqrt(252)

    @property
    def sortino(self) -> float:
        """
        Calculates the annualized Sortino ratio.

        Returns:
        --------
        float
            Sortino ratio, 0 if not computable.
        """
        if self.returns.empty:
            return 0.0
        mean_ret = self.returns.mean()
        downside_std = self.returns[self.returns < 0].std()
        if np.isnan(mean_ret) or np.isnan(downside_std) or downside_std == 0:
            return 0.0
        return (mean_ret / downside_std) * np.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        """
        Computes the maximum drawdown from the data series.

        Returns:
        --------
        float
            Maximum drawdown as a negative percentage, 0 if not computable.
        """
        if self.data.empty:
            return 0.0
        rolling_max = self.data.cummax()
        drawdowns = (self.data / rolling_max - 1)
        if drawdowns.isnull().all() or drawdowns.empty:
            return 0.0
        return float(drawdowns.min()) if not np.isnan(drawdowns.min()) else 0.0

    @property
    def calmar(self) -> float:
        """
        Calculates the Calmar ratio: annualized return divided by max drawdown.

        Returns:
        --------
        float
            Calmar ratio, 0 if not computable or max drawdown is zero.
        """
        if self.returns.empty:
            return 0.0
        annual_return = self.returns.mean() * 252
        mdd = self.max_drawdown
        if np.isnan(annual_return) or mdd == 0:
            return 0.0
        return annual_return / abs(mdd)

    @staticmethod
    def win_rate(closed_positions) -> float:
        """
        Calculates the win rate for closed positions.

        Parameters:
        -----------
        closed_positions : list
            List of Position objects with 'profit' attribute.

        Returns:
        --------
        float
            Proportion of positions that were profitable, 0 if list is empty.
        """
        if not closed_positions:
            return 0.0
        n_wins = sum(1 for pos in closed_positions if pos.profit is not None and pos.profit > 0)
        return n_wins / len(closed_positions) if len(closed_positions) > 0 else 0.0
