from libraries import *
from backtesting import backtest
from optimizer import optimize_hyperparams
from functions import dateset_split, BacktestingCapCOM, OptunaOpt
from plot import plot_portfolio, plot_test_validation, print_best_hyperparams, print_metricas, tables

# --- Cargar datos ---
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "Binance_BTCUSDT_1h.csv")

data = pd.read_csv(file_path).copy().dropna()
data = data.iloc[::-1].reset_index(drop=True) 

# --- Split de datos ---
train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

# Fechas de TRAIN (puedes usarlas afuera si necesitas)
train_dates = train['Date'].reset_index(drop=True)

# Métrica de optimización
optimization_metric = "Calmar"  # 'Sharpe', 'Sortino', 'Calmar'

def main():
    # --- CONFIG ---
    backtest_config = BacktestingCapCOM()
    optimizacion_config = OptunaOpt()

    # --- OPTUNA TRAIN ---
    study = optimize_hyperparams(
        train, backtest_config, optimizacion_config, optimization_metric
    )
    best_params = study.best_trial
    print(f"--- Best {optimization_metric}: {study.best_trial.value:.4f} ---")
    print_best_hyperparams(best_params.params)

    # --- BACKTEST TRAIN ---
    port_value_train, metrics_train, final_cash_train = backtest(train, best_params)
    print_metricas(metrics_train, name="TRAIN")
    plot_portfolio(port_value_train, final_cash_train, name="TRAIN")

    # --- BACKTEST TEST ---
    port_value_test, metrics_test, final_cash_test = backtest(test, best_params)
    print_metricas(metrics_test, name="TEST")
    plot_portfolio(port_value_test, final_cash_test, name="TEST")

    # --- BACKTEST VALIDATION ---
    port_value_val, metrics_val, final_cash_val = backtest(
        validation, best_params, initial_cash=final_cash_test
    )
    print_metricas(metrics_val, name="VALIDATION")
    plot_portfolio(port_value_val, final_cash_val, name="VALIDATION")

    # --- Alinear fechas TEST + VALIDATION ---
    test_dates_aligned = test['Date'].iloc[-len(port_value_test):].reset_index(drop=True)
    val_dates_aligned = validation['Date'].iloc[-len(port_value_val):].reset_index(drop=True)
    test_val_dates_aligned = pd.concat([test_dates_aligned, val_dates_aligned]).reset_index(drop=True)

    # --- Tablas ---
    monthly_df, quarterly_df, annual_df = tables(
        port_value_test, port_value_val, test_val_dates_aligned, name="TEST + VALIDATION"
    )

    # --- Graficar TEST + VALIDATION ---
    plot_test_validation(port_value_test, port_value_val)

if __name__ == "__main__":
    main()


