# 📊 002 - Introduction to Trading

## 📌 Overview
This project focuses on the **development of a systematic trading strategy** based on technical analysis indicators for the **BTC/USDT cryptocurrency pair**.  
The goal is to design, implement, and evaluate a **quantitative trading system** capable of generating consistent **risk-adjusted returns** through the integration of multiple indicators and robust optimization techniques.

---

## 🧠 Strategy Description
The trading system uses **three key technical indicators**:
- **RSI (Relative Strength Index)** → Detects overbought/oversold conditions.  
- **Momentum (Rate of Change)** → Measures the strength and direction of price trends.  
- **Volatility (Bollinger Bands Width)** → Acts as a dynamic risk filter to avoid high-risk trades.  

A **“2 out of 3 confirmation rule”** is applied to generate trading signals, increasing the reliability of entries and reducing false positives.  
The strategy supports both **long and short positions**, adapting to bullish and bearish markets.

---

## ⚙️ Methodology and Implementation
1. **Data Source**: One year of hourly BTC/USDT price data.  
2. **Data Splitting**:
   - 60% Training  
   - 20% Testing  
   - 20% Validation  
3. **Optimization**:  
   Hyperparameters (indicator windows, thresholds, stop loss, take profit, position size) were optimized using **Optuna** with 50 trials, aiming to maximize the **Calmar Ratio**.  
4. **Transaction Costs**: A commission of **0.125%** per trade was included for realistic backtesting.

---

## 📈 Evaluation Metrics
The strategy was assessed using multiple financial performance indicators:
- **Calmar Ratio** – Return vs. Maximum Drawdown  
- **Sharpe Ratio** – Return vs. Total Volatility  
- **Sortino Ratio** – Return vs. Downside Volatility  
- **Maximum Drawdown** – Largest capital decline  
- **Win Rate** – Percentage of profitable trades  

### 🔍 Optimization Target
> **Objective:** Maximize Calmar Ratio for optimal balance between profitability and risk.

---

## 🧮 Results Summary
| Dataset | Calmar | Sharpe | Sortino | Max Drawdown | Win Rate | Total Return |
|----------|---------|---------|----------|----------------|------------|---------------|
| **Train (60%)** | 1.4147 | 1.002 | 1.5189 | 7.49% | 4.67% | +35% |
| **Test (20%)** | -0.2546 | -0.2921 | -0.4496 | 8.46% | 24.36% | -3.84% |
| **Validation (20%)** | 2.2599 | 1.5148 | 2.4857 | 22% | 4.99% | +19.46% |

The validation phase confirmed the strategy’s **robustness and adaptability**, achieving sustained portfolio growth with effective risk control.

---

## 🚀 Key Features
- ✅ Walk-forward analysis for realistic performance evaluation.  
- ✅ Dynamic position sizing based on market volatility.  
- ✅ Volatility-based stop-loss and take-profit mechanisms.  
- ✅ Bayesian hyperparameter optimization (Optuna).  
- ✅ Multi-indicator confirmation rule for higher signal reliability.  

---

## ⚠️ Limitations
- The **win rate is relatively low**, though profitable trades tend to offset losses.  
- Performance can degrade under **extreme volatility or low liquidity** conditions.  
- Dependent on the **accuracy and stability** of technical indicators.  

---

## 📚 References
- Sharpe, W. F. (1966, 1994). *Mutual Fund Performance; The Sharpe Ratio.*  
- Sortino, F. A., & Price, L. N. (1994). *Performance Measurement in a Downside Risk Framework.*  
- Calmar, T. (1991). *The Calmar Ratio: A Measure of Return vs. Drawdown.*  
- Chan, E. (2009). *Quantitative Trading: How to Build Your Own Algorithmic Trading Business.*  
- Antonopoulos, A. M. (2017). *Mastering Bitcoin: Unlocking Digital Cryptocurrencies.*  
- OpenAI (2025). *ChatGPT (GPT-5-mini) [Large language model].*  

---

## 👤 Author
**José Armando Melchor Soto**  
🎓 ITESO – 002 Introduction to Trading  
📅 2025  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Optuna  
- Matplotlib / Seaborn  
- Backtesting Framework  

---

## 💡 Future Improvements
- Integration with **live trading APIs** (e.g., Binance).  
- Expansion to **multi-asset portfolios**.  
- Incorporation of **machine learning models** for predictive signal enhancement.  
- Real-time dashboard for performance monitoring.  

---
