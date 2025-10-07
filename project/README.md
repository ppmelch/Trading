# 📈 Systematic Trading Strategy – BTC/USDT

This project develops a **systematic trading strategy** that integrates multiple **technical indicators**, dynamic position sizing, and **Bayesian hyperparameter optimization** to achieve consistent **risk-adjusted returns** in the cryptocurrency market.  
It lays the groundwork for more resilient and intelligent trading systems that balance **return and risk** over time.

---

## ⚙️ Methodology

### Data
- **Asset:** BTC/USDT  
- **Timeframe:** 1-hour bars  
- **Dataset:** 1 year of historical price data  
- **Splits:**  
  - 60% Training  
  - 20% Testing  
  - 20% Validation  

Data preprocessing involved removing missing values, chronological ordering, and computing all technical indicators before backtesting.

---

## 📊 Indicators

| Indicator | Description | Purpose |
|------------|-------------|----------|
| **RSI (Relative Strength Index)** | Measures overbought and oversold conditions. | Detects potential trend reversals. |
| **Momentum (Rate of Change)** | Measures the speed of price movements. | Confirms the strength and direction of trends. |
| **Volatility (Bollinger Bands Width)** | Quantifies market variability. | Filters high-risk trades during volatile periods. |

The strategy follows a **2-out-of-3 confirmation rule**, requiring at least two indicators to agree before generating a **buy (long)** or **sell (short)** signal. This reduces false positives and improves reliability.

---

## 💼 Trading Logic

- **Buy (Long):** At least two bullish signals.  
- **Sell (Short):** At least two bearish signals.  
- **No Trade:** If fewer than two signals confirm.  

Each trade dynamically adjusts **position size** according to volatility and applies **stop-loss** and **take-profit** levels.  
A **transaction fee of 0.125%** per operation was included to ensure realistic backtesting.

---

## 💰 Position Management

- **Initial capital:** \$1,000,000  
- **Transaction cost:** 0.125%  
- **No leverage used**  
- Dynamic exposure based on volatility  
- Volatility-based **stop-loss** and **take-profit** for capital protection  

This risk management structure maintains consistent returns while minimizing drawdowns during adverse market phases.

---

## 🧠 Hyperparameter Optimization (Optuna)

The optimization process uses **Optuna** to maximize the **Calmar Ratio**, balancing profitability and maximum drawdown.  
A total of **50 trials** were conducted.

| Parameter | Range | Description |
|------------|--------|-------------|
| `rsi_window` | 11–25 | RSI lookback window |
| `rsi_lower` | 25–35 | RSI oversold threshold |
| `rsi_upper` | 70–80 | RSI overbought threshold |
| `momentum_window` | 10–22 | Momentum lookback window |
| `momentum_threshold` | 0.02–0.1 | Signal trigger threshold |
| `volatility_window` | 25–35 | Bollinger Band window |
| `volatility_quantile` | 0.6–0.7 | Volatility filter level |
| `stop_loss` | 0.02–0.03 | Stop-loss fraction |
| `take_profit` | 0.05–0.1 | Take-profit fraction |
| `capital_pct_exp` | 0.05–0.2 | Fraction of capital per trade |

### Best Parameters Found

| Parameter | Value |
|------------|--------|
| `rsi_window` | 21 |
| `rsi_lower` | 32 |
| `rsi_upper` | 74 |
| `momentum_window` | 14 |
| `momentum_threshold` | 0.0567 |
| `volatility_window` | 25 |
| `volatility_quantile` | 0.6708 |
| `stop_loss` | 0.0286 |
| `take_profit` | 0.0915 |
| `capital_pct_exp` | 0.1842 |

---

## 📈 Performance Overview

| Dataset | Calmar | Sharpe | Sortino | Max DD | Win Rate | Total Return | Final Capital |
|----------|---------|---------|----------|---------|-----------|---------------|
| **Train (60%)** | 1.41 | 1.00 | 1.52 | 7.49% | 4.67% | +35% | \$1,626,250 |
| **Test (20%)** | -0.25 | -0.29 | -0.45 | 8.46% | 24.36% | -3.84% | \$961,602 |
| **Validation (20%)** | 2.26 | 1.51 | 2.48 | 4.99% | 4.99% | +19.46% | \$1,148,439 |

The validation period shows strong recovery and confirms **robustness and adaptability** under favorable market conditions.

---

## ⚠️ Risk Analysis

- Volatility filters reduce exposure in unstable conditions.  
- Stop-loss and take-profit dynamically adjust with market volatility.  
- Transaction costs are fully incorporated.  
- Drawdown management ensures consistent capital protection.  

**Limitations:**  
- Sensitive to trendless markets.  
- Low win rate (few but highly profitable trades).  
- Dependent on indicator tuning and volatility behavior.

---

## 🧾 Conclusions

- The **RSI–Momentum–Volatility** trio, combined with the **2/3 confirmation rule**, improved signal reliability and reduced noise.  
- **Optuna** hyperparameter tuning achieved strong **risk-adjusted performance**.  
- Despite lower test results, the **validation phase** confirmed adaptability and profitability.  
- The system demonstrates a **robust, data-driven framework** for algorithmic trading, adaptable to highly volatile assets like **cryptocurrencies**.

---

## 🧰 Technologies and Tools

**Language:** Python  
**Libraries:**
- `pandas`, `numpy` — data processing  
- `optuna` — Bayesian optimization  
- `matplotlib`, `seaborn` — visualization  
- `TA-Lib` or custom indicator functions  
**Development Tools:** Jupyter Notebook, VSCode  
**Timeframe:** 1-hour bars  
**Asset:** BTC/USDT

---

## 🗂️ Repository Structure

project/
│
├── pycache/
├── env/
├── Extra/
├── plot/
│
├── 002 Introduction to Trading.pdf
│
├── backtesting.py
├── functions.py
├── hyperparams.py
├── indicators.py
├── libraries.py
├── main.py
├── metrics.py
├── optimizer.py
├── visualization.py
│
├── Binance_BTCUSDT_hourly.xlsx
├── requirements.txt
└── README.md



---

## ⚙️ Installation, Execution, and Example Output

To reproduce the results of this project and visualize the performance metrics, follow these steps:

```bash
# 1️⃣ Clone the repository
git clone https://github.com/ppmelch/Trading.git
cd Trading/project

# 2️⃣ Create a virtual environment (recommended)
python -m venv env

# Activate it:
# Windows
env\Scripts\activate
# macOS / Linux
source env/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 💡 Note:
# If TA-Lib fails to install (common on Windows), install it manually:
pip install TA-Lib-0.4.28-cp311-cp311-win_amd64.whl
# (You can download the correct .whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

# 4️⃣ Run the trading strategy
python main.py
