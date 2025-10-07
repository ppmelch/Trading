"""
Main trading backtesting and optimization script.

This script executes the complete workflow for developing and testing
a trading strategy using Optuna for hyperparameter optimization and
custom backtesting logic.

Modules imported:
    - libraries: common dependencies (pandas, numpy, etc.)
    - backtesting: core backtest function
    - optimizer: Optuna optimization handler
    - functions: dataset splitting and helper classes
    - plot: visualization and reporting utilities

The process includes:
    1. Loading and preprocessing market data.
    2. Splitting data into train, test, and validation sets.
    3. Hyperparameter optimization on the training set.
    4. Backtesting on train, test, and validation datasets.
    5. Generating metrics, tables, and plots for analysis.
"""

from libraries import *
from backtesting import backtest
from optimizer import optimize_hyperparams
from functions import dateset_split, BacktestingCapCOM, OptunaOpt
from visualization import (plot_portfolio, plot_test_validation,
                  print_best_hyperparams, print_metricas, tables)

# --- Load Data ---
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "Binance_BTCUSDT_1h.csv")

# Read CSV, drop missing values, reverse to chronological order
data = pd.read_csv(file_path).copy().dropna()
data = data.iloc[::-1].reset_index(drop=True)

# --- Dataset Split ---
train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

# --- Alinear fechas con cada portafolio ---
# NOTA: asumimos que cada fila del dataset tiene 1 valor de portafolio
train_dates = train["Date"].reset_index(drop=True)
test_dates = test["Date"].reset_index(drop=True)
valid_dates = validation["Date"].reset_index(drop=True)

# Concatenar TEST + VALIDATION para tablas y plots combinados
test_val_dates_aligned = pd.concat([test_dates, valid_dates]).reset_index(drop=True)

# Optimization metric to guide Optuna
optimization_metric = "Calmar"  # Options: 'Sharpe', 'Sortino', 'Calmar'


def main():
    """
    Executes the full trading strategy pipeline:
    optimization, backtesting, and performance visualization.

    Steps:
        1. Configure backtesting and optimization parameters.
        2. Run Optuna optimization on the training dataset.
        3. Retrieve and print the best hyperparameters.
        4. Execute backtests on train, test, and validation splits.
        5. Generate performance metrics, portfolio plots, and result tables.
    """

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
    port_value_train, metrics_train, final_cash_train = backtest(
        train, best_params)
    print_metricas(metrics_train, name="TRAIN")
    plot_portfolio(port_value_train, final_cash_train, name="TRAIN")

    # --- BACKTEST TEST ---
    port_value_test, metrics_test, final_cash_test = backtest(
        test, best_params)
    print_metricas(metrics_test, name="TEST")
    plot_portfolio(port_value_test, final_cash_test, name="TEST")

    # --- BACKTEST VALIDATION ---
    port_value_val, metrics_val, final_cash_val = backtest(
        validation, best_params, initial_cash=final_cash_test
    )
    print_metricas(metrics_val, name="VALIDATION")
    plot_portfolio(port_value_val, final_cash_val, name="VALIDATION")

    # --- TABLES ---
    monthly_df, quarterly_df, annual_df = tables(
        port_value_test, port_value_val, test_val_dates_aligned, name="TEST+VALIDATION"
    )


    # --- TEST + VALIDATION PLOT ---
    plot_test_validation(port_value_test, port_value_val)


if __name__ == "__main__":
    main()
