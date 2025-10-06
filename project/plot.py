from libraries import *


def plot_portfolio(port_value, final_cash, name="Portfolio") -> None:
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


def plot_test_validation(port_value_test, port_value_val) -> None:
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

    plt.title("Test + Validation")
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


def print_metricas(metrics, name: str = "Portfolio") -> None:
    """
    Prints trading performance metrics in a formatted manner.
    Can handle a single dict or a list of dicts (to sum/average them).

    Parameters
    ----------
    metrics : dict or list of dict
        Dictionary (or list of dictionaries) containing metric names and their values.
    name : str, optional
        Name of the portfolio for labeling. Default is "Portfolio".
    """

    # --- Normalize to list ---
    if isinstance(metrics, dict):
        metrics_list = [metrics]
    elif isinstance(metrics, list):
        metrics_list = metrics
    else:
        raise TypeError("metrics must be a dict or list of dicts")

    # --- Initialize combined metrics ---
    combined = {}
    keys = metrics_list[0].keys()

    for key in keys:
        # Monetary / percentage fields that should be summed
        if key in ["Total Return (%)", "Profit ($)", "Final Capital ($)"]:
            total = 0
            for m in metrics_list:
                val = m[key]
                # Remove $ or convert string to float if necessary
                if isinstance(val, str):
                    val = float(val.replace("$", "").replace(",", ""))
                total += val
            combined[key] = total
        # Numeric ratios that can be averaged
        elif isinstance(metrics_list[0][key], float):
            combined[key] = sum(m[key]
                                for m in metrics_list) / len(metrics_list)
        # Keep other types as is (e.g., strings)
        else:
            combined[key] = metrics_list[-1][key]

    # --- Print metrics nicely ---
    print(f"\n--- Trading Performance {name} Metrics ---")
    for key, value in combined.items():
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
        elif isinstance(value, (int, float)) and key in ["Profit ($)", "Final Capital ($)"]:
            print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value}")
    print("-----------------------------------\n")


def tables(port_value_test, port_value_val, test_val_dates, name="TEST + VALIDATION") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes, displays, and visualizes compounded returns for combined TEST and VALIDATION portfolio values.

    This function concatenates portfolio values from the TEST and VALIDATION sets, aligns them with the 
    provided date series, and calculates compounded returns at monthly, quarterly, and annual frequencies.
    Each return series is displayed as a formatted DataFrame and plotted as a color-coded bar chart
    (green for positive, red for negative returns).

    Parameters
    ----------
    port_value_test : list or pd.Series
        Portfolio values for the TEST set.
    port_value_val : list or pd.Series
        Portfolio values for the VALIDATION set.
    test_val_dates : list or pd.Series
        Combined date range corresponding to both TEST and VALIDATION portfolios.
    name : str, optional
        Custom name for display headers and plot titles. Default is "TEST + VALIDATION".

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing three DataFrames:
            - `monthly_df` : Monthly compounded returns (%)
            - `quarterly_df` : Quarterly compounded returns (%)
            - `annual_df` : Annual compounded returns (%)

    Notes
    -----
    - Returns are computed as compounded (geometric) returns.
    - Displayed tables and plots are meant for visual performance analysis.
    - Missing or mismatched date lengths are automatically adjusted from the end of the date series.
    """

    # --- Concatenate portfolio values ---
    total_portfolio = port_value_test + port_value_val

    # --- Align dates (use last N matching portfolio length) ---
    dates_aligned = pd.Series(
        test_val_dates).iloc[-len(total_portfolio):].reset_index(drop=True)

    # --- Create DataFrame with datetime index ---
    port_df = pd.DataFrame(
        {"Portfolio_Value": total_portfolio}, index=pd.to_datetime(dates_aligned))

    # --- Calculate daily returns ---
    daily_returns = port_df['Portfolio_Value'].pct_change()

    # --- Compute compounded returns ---
    monthly_df = daily_returns.resample('ME').apply(
        lambda x: (1 + x).prod() - 1).to_frame("Monthly_Returns")
    quarterly_df = daily_returns.resample('QE').apply(
        lambda x: (1 + x).prod() - 1).to_frame("Quarterly_Returns")
    annual_df = daily_returns.resample('YE').apply(
        lambda x: (1 + x).prod() - 1).to_frame("Annual_Returns")

    # --- Display tables ---
    print(f"\n--- {name} Monthly Returns ---")
    display(monthly_df.tail(len(monthly_df)))
    print(f"\n--- {name} Quarterly Returns ---")
    display(quarterly_df.tail(len(quarterly_df)))
    print(f"\n--- {name} Annual Returns ---")
    display(annual_df.tail(len(annual_df)))

    # --- Internal plot function ---
    def plot_returns(series: pd.Series, title: str):
        """
        Plots a bar chart of returns with positive values in green and negative values in red.

        Each bar represents a periods compounded return, labeled with its corresponding percentage value.

        Parameters
        ----------
        series : pd.Series
            Time-indexed return series to be plotted (monthly, quarterly, or annual).
        title : str
            Title of the plot (e.g., "Monthly", "Quarterly", "Annual").

        Notes
        -----
        - Bars are color-coded: from libraries.py colors[0] for positive returns and colors[1] for negative returns.
        - The function automatically formats x-axis labels and scales for readability.
        - Designed for visual inspection of performance across time periods.
        """
        plt.figure(figsize=(10, 4))
        bars = plt.bar(series.index.strftime('%Y-%m-%d'), series * 100,
                       color=[colors[0] if v >= 0 else colors[1] for v in series], alpha=1)
        plt.title(f"{title} Returns")
        plt.ylabel("Return (%)")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --- Plot ---
    plot_returns(monthly_df['Monthly_Returns'], "Monthly")
    plot_returns(quarterly_df['Quarterly_Returns'].dropna(), "Quarterly")
    plot_returns(annual_df['Annual_Returns'].dropna(), "Annual")

    return monthly_df, quarterly_df, annual_df
