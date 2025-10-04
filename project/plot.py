from libraries import *
from backtest import backtest

def plot_portfolio(port_value, final_cash):
    plt.figure(figsize=(12, 6))
    plt.plot(port_value, color='blue', label=f'Portfolio value\nFinal: ${final_cash:,.2f}')
    plt.title('Portfolio value over time')
    plt.xlabel('Time step')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
