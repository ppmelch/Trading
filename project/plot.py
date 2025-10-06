from libraries import *


def plot_portfolio(port_value, final_cash, name="Portfolio")-> None:
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


def plot_test_validation(port_value_test, port_value_val)-> None:
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
    Prints the best hyperparameters from the optimization process.
    Parameters
    ----------
    params : dict
        Dictionary containing hyperparameter names and their best values.
    """
    print('\n--- Best Hyperparameters ---')
    for param, value in params.items():
        if isinstance(value, float):
            print(f'  {param}: {value:.4f}')
        else:
            print(f'  {param}: {value}')
    print("-----------------------------------\n")


def print_metricas(metrics: dict, name: str = "Portfolio")-> None:
    """
    Prints the trading performance metrics in a formatted manner.

    Parameters
    ----------
    metrics : dict
        Dictionary containing metric names and their corresponding values.
    name : str, optional
        Name of the portfolio for labeling. Default is "Portfolio".
    """
    print(f"\n--- Trading Performance {name} Metrics ---")
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

def returns_overtime_test_val(port_value_test, test_dates, port_value_val, val_dates, name="TEST + VALIDATION")-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    """
    Compute and display compounded returns for TEST + VALIDATION sets, aligning dates internally.
    Shows monthly, quarterly, and annual returns as tables and bar plots with positive returns in green
    and negative returns in red.

    Parameters
    ----------
    port_value_test : list or pd.Series
        Portfolio values for TEST set.
    test_dates : pd.Series
        Corresponding dates for TEST.
    port_value_val : list or pd.Series
        Portfolio values for VALIDATION set.
    val_dates : pd.Series
        Corresponding dates for VALIDATION.
    name : str, optional
        Name for display and plot titles. Default is "TEST + VALIDATION".

    Returns
    -------
    tuple of pd.DataFrame
        (monthly_df, quarterly_df, annual_df) containing compounded returns.
    """

    # --- Align dates ---
    test_dates_aligned = test_dates.iloc[-len(port_value_test):].reset_index(drop=True)
    val_dates_aligned = val_dates.iloc[-len(port_value_val):].reset_index(drop=True)

    # --- Concatenate portfolio values and dates ---
    total_portfolio = port_value_test + port_value_val
    dates = pd.concat([test_dates_aligned, val_dates_aligned]).reset_index(drop=True)

    # --- Create DataFrame with datetime index ---
    port_df = pd.DataFrame({"Portfolio_Value": total_portfolio}, index=pd.to_datetime(dates))

    # --- Calculate daily returns ---
    daily_returns = port_df['Portfolio_Value'].pct_change()

    # --- Compute compounded returns ---
    monthly_df = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1).to_frame("Monthly_Returns")
    quarterly_df = daily_returns.resample('QE').apply(lambda x: (1 + x).prod() - 1).to_frame("Quarterly_Returns")
    annual_df = daily_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1).to_frame("Annual_Returns")

    # --- Display tables ---
    print(f"\n--- {name} Monthly Returns ---")
    display(monthly_df)
    print(f"\n--- {name} Quarterly Returns ---")
    display(quarterly_df)
    print(f"\n--- {name} Annual Returns ---")
    display(annual_df)

    # --- Internal function to plot returns ---
    def plot_returns(series: pd.Series, title: str):
        """
        Plots a series of returns as a bar chart with positive returns in green and negative in red,
        annotating each bar with the percentage.

        Parameters
        ----------
        series : pd.Series
            Series of returns (decimal form, e.g., 0.05 = 5%).
        title : str
            Title for the plot.
        """
        plt.figure(figsize=(10,4))
        bars = plt.bar(series.index.strftime('%Y-%m-%d'), series*100,
                       color=['#2E8B57' if v >= 0 else '#A30406' for v in series], alpha=0.7)
        plt.title(f"{title} Returns ({name})")
        plt.ylabel("Return (%)")
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(alpha=0.3)
        for bar, val in zip(bars, series):
            height = val*100
            if not pd.isna(height):
                plt.text(bar.get_x() + bar.get_width()/2, height,
                         f"{height:.2f}%", ha='center',
                         va='bottom' if height >= 0 else 'top',
                         fontsize=6, color='black')
        plt.tight_layout()
        plt.show()

    # --- Plot all periods ---
    plot_returns(monthly_df['Monthly_Returns'], "Monthly")
    plot_returns(quarterly_df['Quarterly_Returns'].dropna(), "Quarterly")
    plot_returns(annual_df['Annual_Returns'].dropna(), "Annual")

    return monthly_df, quarterly_df, annual_df
