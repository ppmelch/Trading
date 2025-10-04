from libraries import *
from funtions import dateset_split, BacktestingCapCOM, OptunaOpt
from metrics import Metrics
from Optimizer import optimize_hyperparams
from backtest import backtest
from Indicadores import Indicadores
from plot import plot_portfolio
import os

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "Binance_BTCUSDT_1h.csv")

data = pd.read_csv(file_path).dropna()
data['Date'] = pd.to_datetime(data['Date'], format='mixed')

train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

def main():
    # --- Configuraciones ---
    backtest_config = BacktestingCapCOM()
    optimizacion_config = OptunaOpt()

    # --- TRAIN ---
    study = optimize_hyperparams(
        train, backtest_config, optimizacion_config, metric="Calmar")
    best_params = study.best_trial.params
    best_value = study.best_value
    print("Mejores parámetros TRAIN:", best_params)
    print("Mejor valor de métrica TRAIN:", best_value)

    # --- PLOT  TRAIN ---
    port_value, metrics_dict, final_cash = backtest(train, best_params)
    print("Métricas TRAIN:", metrics_dict)
    plot_portfolio(port_value, final_cash)

    # --- TEST ---
    study = optimize_hyperparams(
        test, backtest_config, optimizacion_config, metric="Calmar")
    best_params = study.best_trial.params
    best_value = study.best_value
    print("Mejores parámetros TEST:", best_params)
    print("Mejor valor de métrica TEST:", best_value)

    # --- PLOT TEST ---
    port_value, metrics_dict, final_cash = backtest(test, best_params)
    print("Métricas TEST:", metrics_dict)

    # --- VALIDATION ---

    # --- VALIDATION ---
    study = optimize_hyperparams(
        validation, backtest_config, optimizacion_config, metric="Calmar")
    best_params = study.best_trial.params
    best_value = study.best_value
    print("Mejores parámetros VALIDATION:", best_params)
    print("Mejor valor de métrica VALIDATION:", best_value)

    # --- PLOT VALIDATION ---
    port_value, metrics_dict, final_cash = backtest(validation, best_params)
    print("Métricas VALIDATION:", metrics_dict)
    plot_portfolio(port_value, final_cash)

if __name__ == "__main__":
    main()
