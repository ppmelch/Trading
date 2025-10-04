from libraries import *
from backtest import backtest
from metrics import Metrics
from Optimizer import optimize_hyperparams
from plot import plot_portfolio, plot_test_validation
from funtions import dateset_split, BacktestingCapCOM, OptunaOpt



base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "Binance_BTCUSDT_1h.csv")

data = pd.read_csv(file_path).dropna()
data['Date'] = pd.to_datetime(data['Date'], format='mixed')

train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

train_dates = pd.concat([train['Date'], test['Date'].iloc[:1]]).tolist()
test_dates = pd.concat([test['Date'], validation['Date'].iloc[-1:]]).tolist()
val_dates = pd.concat([validation['Date'], data['Date'].iloc[-1:]]).tolist()

optimization_metric = "Sortino" # 'Sharpe', 'Sortino', 'Calmar'

def main():
    # --- Configuraciones ---
    backtest_config = BacktestingCapCOM()
    optimizacion_config = OptunaOpt()
    
    # --- OPTUNA TRAIN ---
    study = optimize_hyperparams(
        train, backtest_config, optimizacion_config, metric= optimization_metric)
    best_params = study.best_trial.params
    best_value = study.best_value
    print("Mejores parámetros TRAIN:", best_params)
    print("Mejor valor de métrica TRAIN:", best_value)

    # --- PLOT TRAIN ---
    port_value, metrics_dict, final_cash_train = backtest(train, best_params)
    print("Métricas TRAIN:", metrics_dict)
    # TRAIN
    plot_portfolio(port_value, final_cash_train, name="TRAIN")


    # --- OPTIMIZACIÓN + TEST ---
    port_value_test, metrics_dict, final_cash_test = backtest(test, best_params)
    print("Métricas TEST:", metrics_dict)
    plot_portfolio(port_value_test, final_cash_test, name="TEST")

    # --- VALIDATION ---
    port_value_val, metrics_dict_val, final_cash_val = backtest(validation, best_params)
    print("Métricas VALIDATION:", metrics_dict_val)
    plot_portfolio(port_value_val, final_cash_val, name="VALIDATION")

    # --- PLOT TEST + VALIDATION ---
    port_value_test, port_value_val, final_cash_test, final_cash_val = plot_test_validation(
        test, validation, backtest, best_params
    )


if __name__ == "__main__":
    main()
