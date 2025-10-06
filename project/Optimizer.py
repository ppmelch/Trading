from libraries import *
from backtesting import backtest
from functions import OptunaOpt, BacktestingCapCOM
from sklearn.model_selection import TimeSeriesSplit


def optimize(trial, train: pd.DataFrame) -> float:
    """
    Performs a simple K-fold like optimization on chunks of the training data.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object used to suggest hyperparameters.
    train : pd.DataFrame
        Training dataset.

    Returns
    -------
    float
        Average Calmar ratio across data splits.
    """
    data = train.copy()
    n_splits = OptunaOpt.n_splits
    len_data = len(data)
    calmars = []

    for i in range(n_splits):
        size = len_data // n_splits
        start_idx = i * size
        end_idx = (i + 1) * size
        chunk = data.iloc[start_idx:end_idx].reset_index(drop=True)

        port_value, metrics_dict, _ = backtest(chunk, trial)
        calmars.append(metrics_dict.get('Calmar', 0.0))

    return sum(calmars) / n_splits


def CV(trial, data: pd.DataFrame, n_splits: int, metric: str) -> float:
    """
    Cross-validation for a time series dataset using backtest metrics.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object.
    data : pd.DataFrame
        Full dataset with price information.
    n_splits : int
        Number of time series splits.
    metric : str
        Metric name to evaluate (e.g., 'Calmar').

    Returns
    -------
    float
        Average metric across splits.
    """
    splits = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for _, test_idx in splits.split(data):
        test_data = data.iloc[test_idx].copy().reset_index(drop=True)
        _, metrics_dict, _ = backtest(test_data, trial)
        scores.append(metrics_dict.get(metric, 0.0))

    return np.mean(scores)


def optimize_hyperparams(data: pd.DataFrame,
                         backtest_config: BacktestingCapCOM,
                         optuna_config: OptunaOpt,
                         metric: str) -> optuna.study.Study:
    """
    Performs hyperparameter optimization using Optuna.

    Parameters
    ----------
    data : pd.DataFrame
        Training dataset.
    backtest_config : BacktestingCapCOM
        Backtesting configuration.
    optuna_config : OptunaOpt
        Optuna configuration (number of trials, direction, etc.).
    metric : str
        Metric to optimize (e.g., 'Calmar').

    Returns
    -------
    optuna.study.Study
        Optuna study object with optimization results.
    """
    def objective(trial) -> float:
        port_value, metrics_dict, _ = backtest(data.copy(), trial)
        return metrics_dict.get(metric, 0.0)

    study = optuna.create_study(direction=optuna_config.direction)
    study.optimize(
        objective,
        n_trials=optuna_config.n_trials,
        n_jobs=optuna_config.n_jobs,
        show_progress_bar=optuna_config.show_progress_bar
    )
    return study
