from libraries import *

def plot_portfolio(port_value, final_cash, name="Portfolio"):
    """
    Plots a portfolio over time with a given name and final cash value.

    Parameters:
        port_value (list, np.ndarray, or pd.Series): Portfolio values over time.
        final_cash (float): Final value of the portfolio.
        name (str, optional): Name of the portfolio for labeling. Default is "Portfolio".
    """

    # Convert to pd.Series if a list or ndarray
    if isinstance(port_value, list) or isinstance(port_value, np.ndarray):
        port_value = pd.Series(port_value, name=name)
    
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(port_value, color=colors[1], lw=2, label=f'{port_value.name}\nFinal: ${final_cash:,.2f}')
    ax.set_title(f'{port_value.name}', fontsize=14)
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Value ($)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_test_validation(test_data, validation_data, backtest_func, best_params):
    """
    Plots TEST and VALIDATION portfolios continuously to visualize portfolio progression.

    Parameters:
        test_data (pd.DataFrame): DataFrame containing the TEST dataset with a 'Date' column.
        validation_data (pd.DataFrame): DataFrame containing the VALIDATION dataset with a 'Date' column.
        backtest_func (function): Backtesting function that returns (portfolio, metrics, final_cash).
        best_params (dict): Dictionary containing the optimal hyperparameters.
    
    Returns:
        tuple: test_port, val_port, final_cash_test, final_cash_val
    """

    # --- Backtest ---
    test_port, _, final_cash_test = backtest_func(test_data, best_params)
    val_port, _, final_cash_val = backtest_func(validation_data, best_params)

    # Full dates
    test_dates = test_data['Date'].iloc[:len(test_port)]
    val_dates = validation_data['Date'].iloc[:len(val_port)]

    # Adjust VALIDATION to continue from end of TEST
    val_adjusted = np.array(val_port) - val_port[0] + test_port[-1]

    # Plot
    plt.figure(figsize=(14,6))
    plt.plot(test_dates, test_port, color=colors[0], lw=2, label='TEST')
    plt.plot(val_dates, val_adjusted, color=colors[1], lw=2, label='VALIDATION')
    plt.title("TEST / VALIDATION Portfolio", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return test_port, val_port, final_cash_test, final_cash_val