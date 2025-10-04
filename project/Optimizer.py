from sklearn.model_selection import TimeSeriesSplit
from backtest import backtest
from libraries import *
from Indicadores import Indicadores
from funtions import OptunaOpt, BacktestingCapCOM
from metrics import Metrics
from Objetive import hyperparams


def optimize(trial, train) -> float:
    """
    Perform a simple K-fold like optimization on chunks of the training data.

    Args:
        trial: Optuna trial object used to suggest hyperparameters.
        train (pd.DataFrame): Training dataset.

    Returns:
        float: Average Calmar ratio across data splits.
    """
    data = train.copy()
    n_splits = 5
    len_data = len(data)
    calmars = []

    for i in range(n_splits):
        size = len_data // n_splits
        start_idx = i * size
        end_idx = (i+1) * size
        chunk = data.iloc[start_idx:end_idx]
        portfolio_value = backtest(
            chunk, Indicadores.stop_loss, Indicadores.take_profit, Indicadores.n_shares, trial
        )
        calmar = Metrics.calmar(portfolio_value)
        calmars.append(calmar)

    return sum(calmars) / n_splits


def CV(trial, data: pd.DataFrame, n_splits: int, metric: str) -> float:
    """
    Cross-validation for a time series dataset using backtest metrics.

    Args:
        trial: Optuna trial object.
        data (pd.DataFrame): Full dataset.
        n_splits (int): Number of time series splits.
        metric (str): Metric name to evaluate (e.g., 'Calmar').

    Returns:
        float: Average metric across splits.
    """
    splits = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for _, test_idx in splits.split(data):
        test_data = data.iloc[test_idx].copy().reset_index(drop=True)
        # backtest handles hyperparameters
        _, metrics_dict, _ = backtest(test_data, trial)
        scores.append(metrics_dict[metric])

    return np.mean(scores)


def optimize_hyperparams(data: pd.DataFrame, backtest_config: BacktestingCapCOM,
                         optuna_config: OptunaOpt, metric: str) -> optuna.study.Study:
    """
    Perform hyperparameter optimization using Optuna.

    Args:
        data (pd.DataFrame): Training dataset.
        backtest_config (BacktestingCapCOM): Backtesting configuration.
        optuna_config (OptunaOpt): Optuna configuration.
        metric (str): Metric to optimize (e.g., 'Calmar').

    Returns:
        optuna.study.Study: Optuna study object with optimization results.
    """
    def objective(trial):
        port_hist, metrics_dict, cash = backtest(data.copy(), trial)
        return metrics_dict['Calmar']  # Ensure key matches metrics dictionary

    study = optuna.create_study(direction=optuna_config.direction)
    study.optimize(
        objective,
        n_trials=optuna_config.n_trials,
        n_jobs=optuna_config.n_jobs,
        show_progress_bar=optuna_config.show_progress_bar
    )
    return study


def run_optimization(data, backtest_config, n_splits, metric):
    """
    Wrapper function to run hyperparameter optimization connecting hyperparams and backtest.

    Args:
        data (pd.DataFrame): Full dataset.
        backtest_config (BacktestingCapCOM): Backtesting configuration.
        n_splits (int): Number of splits for cross-validation.
        metric (str): Metric name for evaluation.

    Returns:
        dict: Best hyperparameters from the optimization function.
    """
    return hyperparams(data.copy(), backtest, backtest_config, n_splits, metric)
