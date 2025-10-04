# --- Standard library ---
import os
import warnings

# --- Third-party libraries: Data analysis ---
import ta
import numpy as np
import pandas as pd
import scipy as sp

# --- Third-party libraries: Visualization ---
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# --- Third-party libraries: Machine Learning / Optimization ---
import optuna
from sklearn.model_selection import TimeSeriesSplit

# --- Type hints ---
from typing import List

warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(42)

plt.rcParams['figure.facecolor'] = 'lightgrey'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'


colors = ["cornflowerblue", "indianred", "darkseagreen", "plum", "dimgray"]


# Requirements
# transactions_fee: float = 0.0125  # 0.125% per trade
# No leverage
# Long + Short positions
# Signal Confirmation ( 2 out of 3 indicators agree)
# Walk forward analysis to avoid overfitting
# Performance metrics (Sharpe Ratio, Sortino Ratio, Calmar Ratio, Maximum Drawdown, Win Rate)
# Charts & Tables : Portafolio value through time, Monthly + Quarterly + Annually returns table
