from libraries import *
from funtions import dateset_split, BacktestingCapCOM, OptunaOpt
from metrics import Metrics
from Optimizer import optimize_hyperparams
from backtest import backtest
from Indicadores import Indicadores
import os

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "Binance_BTCUSDT_1h.csv")

data = pd.read_csv(file_path).dropna()
data['Date'] = pd.to_datetime(data['Date'], format='mixed')

train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

metrics = Metrics(data.Close)
print("Data Sharpe after load:", metrics.sharpe)


def main():
    # --- Configuraciones ---
    backtest_config = BacktestingCapCOM()
    optimizacion_config = OptunaOpt()

    # --- Optimización ---
    study = optimize_hyperparams(
        train, backtest_config, optimizacion_config, metric="Calmar")
    best_params = study.best_params
    best_value = study.best_value
    print("Mejores parámetros:", best_params)
    print("Mejor valor de métrica:", best_value)


if __name__ == "__main__":
    main()
