from libraries import *
from backtest import backtest
from Optimizer import optimize_hyperparams
from funtions import dateset_split, BacktestingCapCOM, OptunaOpt
from plot import plot_portfolio, plot_test_validation, print_best_hyperparams, print_metricas


base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "Binance_BTCUSDT_1h.csv")

data = pd.read_csv(file_path).dropna()
data['Date'] = pd.to_datetime(data['Date'], format='mixed')

train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

train_dates = pd.concat([train['Date'], test['Date'].iloc[:1]]).tolist()
test_dates = pd.concat(
    [train['Date'].iloc[-1:], test['Date'].iloc[-1:]]).tolist()
val_dates = pd.concat(
    [test['Date'].iloc[-1:], validation['Date'].iloc[-1:]]).tolist()

optimization_metric = "Calmar"  # 'Sharpe', 'Sortino', 'Calmar'


def main():
    # --- CONFIG ---
    backtest_config = BacktestingCapCOM()
    optimizacion_config = OptunaOpt()

    # --- OPTUNA TRAIN ---
    study = optimize_hyperparams(
        train, backtest_config, optimizacion_config, metric=optimization_metric)
    best_params = study.best_trial
    print(f"--- Best {optimization_metric}: {study.best_trial.value:.4f} ---")
    print_best_hyperparams(best_params.params)

    # --- BACKTEST TRAIN ---
    port_value_train, metrics_train, final_cash_train = backtest(
        train, best_params)
    print_metricas(metrics_train, name="TRAIN")
    plot_portfolio(port_value_train, final_cash_train, name="TRAIN")

    # --- BACKTEST TEST ---
    port_value_test, metrics_test, final_cash_test = backtest(
        test, best_params)
    print_metricas(metrics_test , name="TEST")
    plot_portfolio(port_value_test, final_cash_test, name="TEST")

    # --- BACKTEST VALIDATION ---

    port_value_val, metrics_val, final_cash_val = backtest(
        validation, best_params, initial_cash=final_cash_test)
    print_metricas(metrics_val, name="VALIDATION")
    plot_portfolio(port_value_val, final_cash_val, name="VALIDATION")

    # ---  TEST + VALIDATION  ---
    plot_test_validation(port_value_test, port_value_val)


if __name__ == "__main__":
    main()
