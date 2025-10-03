from backtest import backtest
from libraries import *
from metrics import get_calmar
from Indicadores import Indicadores
from funtions import OptunaOpt, BacktestingCapCOM


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


def optimize_hyperparams(
    data: pd.DataFrame,
    backtest_config: BacktestingCapCOM,
    optuna_config: OptunaOpt,
    metric: str
) -> optuna.study.Study:

    def objective(trial):
        return optimize(trial, data)  # solo 2 args

    study = optuna.create_study(direction=optuna_config.direction)
    study.optimize(
        objective,
        n_trials=optuna_config.n_trials,
        n_jobs=optuna_config.n_jobs,
        show_progress_bar=optuna_config.show_progress_bar
    )

    return study
