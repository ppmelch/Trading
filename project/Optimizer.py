from backtest import backtest
from libraries import *
from metrics import get_calmar
from Indicadores import Indicadores
from funtions import OptunaOpt, BacktestingCapCOM
from metrics import Metrics


def optimize(trial, train) -> float:
    data = train.copy()

    n_splits = 5
    len_data = len(data)
    calmars = []
    for i in range(n_splits):
        size = len_data // n_splits
        start_idx = i * size
        end_idx = (i+1)*size
        chunk = data.iloc[start_idx:end_idx]
        portafolio_value = backtest(
            chunk, Indicadores.stop_loss, Indicadores.take_profit, Indicadores.n_shares, trial)
        portafolio_value
        calmar = get_calmar(portafolio_value)
        calmars.append(calmar)

    return sum(calmars) / n_splits


def CV(trial, data: pd.DataFrame, backtest_func, n_splits: int, metric: str) -> dict:

    params = {
        "rsi_window": trial.suggest_int("rsi_window", 5, 50),

        "rsi_lower": trial.suggest_int("rsi_lower", 5, 35),

        "rsi_upper": trial.suggest_int("rsi_upper", 65, 95),

        "momentum_window": trial.suggest_int("momentum_window", 5, 50),

        "momentum_threshold": trial.suggest_float("momentum_threshold", 0.1, 2.0),

        "volatility_window": trial.suggest_int("volatility_window", 5, 50),

        "volatility_threshold": trial.suggest_float("volatility_threshold", 0.01, 0.5),

        "stop_loss": trial.suggest_float("stop_loss", 0.01, 0.1),

        "take_profit": trial.suggest_float("take_profit", 0.01, 0.2),

        "n_shares": trial.suggest_int("n_shares", 1, 80),
    }

    splits = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for _, test_idx in splits.split(data):
        test_data = data.iloc[test_idx].reset_index(drop=True)

        equity = backtest_func(test_data, params)
        metrics = Metrics(equity)

        scores.append(getattr(metrics, metric))

    mean_score = np.mean(scores)

    return mean_score


def optimize_hyperparams(data: pd.DataFrame, backtest_config: BacktestingCapCOM, optuna_config: OptunaOpt, metric: str) -> optuna.study.Study:

    def objective(trial):
        return optimize(trial, data, backtest_config, optuna_config.n_splits, metric)

    study = optuna.create_study(direction=optuna_config.direction)
    study.optimize(
        objective,
        n_trials=optuna_config.n_trials,
        n_jobs=optuna_config.n_jobs,
        show_progress_bar=optuna_config.show_progress_bar
    )

    return study
