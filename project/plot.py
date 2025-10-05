from libraries import *


def plot_portfolio(port_value, final_cash, name="Portfolio"):
    """
    Plots the portfolio value over time.
    Parameters
    ----------
    port_value : list, np.ndarray, or pd.Series
        Portfolio values over time.
    final_cash : float
        Final value of the portfolio.
    name : str, optional
        Name of the portfolio for labeling. Default is "Portfolio".
    """
    if isinstance(port_value, list) or isinstance(port_value, np.ndarray):
        port_value = pd.Series(port_value, name=name)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(port_value, color=colors[1], lw=2,
            label=f'{port_value.name}\nFinal: ${final_cash:,.2f}')
    ax.set_title(f'{port_value.name}', fontsize=14)
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Value ($)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_test_validation(port_value_test, port_value_val):
    """
    Plots the portfolio values for test and validation datasets.
    Parameters:
        port_value_test (list, np.ndarray, or pd.Series): Portfolio values for the test dataset.
        port_value_val (list, np.ndarray, or pd.Series): Portfolio values for the validation dataset.
    """
    plt.figure(figsize=(12, 6))

    x_test = list(range(len(port_value_test)))
    x_val = list(range(len(port_value_test), len(
        port_value_test) + len(port_value_val)))

    plt.plot(x_test, port_value_test,
             color=colors[0], linewidth=2, label="Test")
    plt.plot(x_val, port_value_val,
             color=colors[1], linewidth=2, label="Validation")

    plt.title("Evolución del valor del portafolio (Test + Validation)")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_best_hyperparams(params: dict) -> None:
    """
    Prints the best hyperparameters from an Optuna study.

    Parameters
    ----------
    study : optuna.study.Study
        Optuna study object containing optimization results.
    """
    print("-----------------------------------\n")
    print('\n--- Best Hyperparameters ---')
    for param, value in params.items():
        if isinstance(value, float):
            print(f'  {param}: {value:.4f}')
        else:
            print(f'  {param}: {value}')
    print("-----------------------------------\n")


def print_metricas(metrics: dict):
    """
    Prints the trading performance metrics in a formatted manner.

    Parameters
    ----------
    metrics : dict
        Dictionary containing metric names and their corresponding values.
    """
    print("\n--- Trading Performance Metrics ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ["Calmar", "Sharpe", "Sortino"]:
                print(f"{key}: {value:.4f}")
            elif key == "Maximum Drawdown":
                print(f"{key}: {value:.2%}")
            elif key == "Win Rate":
                print(f"{key}: {value:.2%}")
            elif key == "Total Return (%)":
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    print("-----------------------------------\n")
