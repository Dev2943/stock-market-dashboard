# 📊 Stock Market Analysis Platform

**Built by Dev Golakiya | UMass Amherst — MS in Business Analytics**

A professional-grade, always-on stock market analysis platform integrating four quantitative finance projects into a single interactive dashboard. Features real-time Yahoo Finance data, institutional-grade risk analytics, options pricing, and factor model analysis — the kind of tooling used by bank market risk, quant research, and portfolio management teams.

## 🚀 Live Demo

**[View Live Dashboard](https://stock-market-dashboard-909492874362.us-central1.run.app)**

> Deployed on **Google Cloud Run** — always on, no cold starts, auto-redeploys on every GitHub push.

---

## 📐 Dashboard Overview — 9 Tabs

| Tab | What it does |
|-----|-------------|
| 📊 Overview | Real-time price cards, sector performance, key metrics |
| 🎯 Risk Analysis | Proprietary multi-factor risk score (volatility, momentum, liquidity, technical, drawdown) |
| 📈 Technical | Candlestick charts with Bollinger Bands, RSI, MACD, volume analysis |
| 🔮 Forecast | 30-day price prediction with confidence intervals and mean-reversion model |
| 📉 Performance | Sharpe/Sortino ratios, cumulative returns, monthly heatmap |
| 💼 Portfolio | Allocation, sector diversification, correlation matrix, investment recommendations |
| ⚠️ VaR & Stress Test | Three VaR/ES methods, Kupiec backtest, 2008 GFC & 2020 COVID stress replays |
| 📐 Options Pricer | Black-Scholes pricer, all 5 Greeks, volatility smile, put-call parity verification |
| 🔬 Factor Model | Fama-French 4-factor regression, 12-1 momentum signal, long/short backtest |

---

## ✨ Key Features

- **Real-time data** from Yahoo Finance with 30-second auto-refresh
- **Three VaR methodologies** — Historical Simulation, Variance-Covariance, Monte Carlo — with full diagnostic comparison
- **Kupiec POF backtest** over rolling windows with exception clustering analysis
- **2008 GFC and 2020 COVID stress replays** applied to the live portfolio
- **Black-Scholes options pricer** with Newton-Raphson IV solver and volatility smile
- **Fama-French 4-factor model** using real ETF proxies (SPY, IWM, IVE/IVW, MTUM)
- **12-1 momentum strategy** with long/short backtest, Sharpe ratio and drawdown analysis
- **Proprietary risk scoring** — multi-factor model: Volatility (30%), Momentum (25%), Liquidity (20%), Technical (15%), Drawdown (10%)
- **Always-on deployment** on Google Cloud Run — no sleeping, no cold starts

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Web framework | Streamlit |
| Data | yfinance (Yahoo Finance) |
| Charts | Plotly |
| Numerics | Pandas, NumPy, SciPy |
| Deployment | Google Cloud Run + Docker |
| CI/CD | GitHub → Cloud Build (auto-deploy on push) |

---

## 📊 Quantitative Finance Projects Integrated

This dashboard integrates four standalone quant projects built during the MS in Business Analytics at UMass Amherst:

### Project 1 — Black-Scholes Options Pricer
Closed-form BSM pricer for European calls and puts, all 5 Greeks (delta, gamma, vega, theta, rho), Newton-Raphson IV solver with bisection fallback, and volatility smile recovery. The skew shown in the dashboard is direct evidence that constant-vol assumptions fail in real markets.

### Project 3 — Multi-Factor Equity Model
Fama-French 4-factor regression (α, β, SMB, HML, MOM) using independent ETF proxies. 12-1 month momentum signal ranked across the selected portfolio. Long/short momentum strategy backtest with Sharpe ratio, max drawdown, and hit rate.

### Project 4 — VaR & Expected Shortfall Calculator
Three VaR/ES methods at 99% confidence on a $100K portfolio. Rolling Kupiec POF backtest measuring exception rates vs the 1% target. Historical stress replays through the 2008 GFC (9.3× VaR peak drawdown) and 2020 COVID crash (7.5×). Three hypothetical scenarios including a correlation-breakdown scenario matching the 2008 peak loss.

---

## 🔧 Local Setup

```bash
# Clone the repository
git clone https://github.com/Dev2943/stock-market-dashboard.git
cd stock-market-dashboard

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run dashboard.py
```

### requirements.txt
```
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.5.3
numpy>=1.24.3
scipy>=1.10.0
yfinance>=0.2.18
```

---

## 🐳 Docker / GCP Deployment

The app is containerised with Docker and deployed on Google Cloud Run.

```bash
# Build the container
docker build -t stock-dashboard .

# Run locally in Docker
docker run -p 8080:8080 stock-dashboard
```

The `Dockerfile` configures Streamlit to run headlessly on port 8080 — the standard for Cloud Run. Every push to the `main` branch triggers an automatic rebuild and redeploy via Cloud Build.

---

## 📁 Project Structure

```
stock-market-dashboard/
├── dashboard.py        # Main application — all 9 tabs, ~1,950 lines
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container config for GCP Cloud Run
└── README.md
```

---

## 👨‍💻 About

**Dev Golakiya**
MS in Business Analytics — UMass Amherst

- 📧 devgolakiya31@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/devgolakiya)
- 🐙 [GitHub](https://github.com/Dev2943)

### Other Projects in the Quant Portfolio
- [Project 1 — Black-Scholes Pricer with Greeks & IV](https://github.com/Dev2943/bsm-pricer)
- [Project 2 — Monte Carlo with Variance Reduction & Exotic Payoffs](https://github.com/Dev2943/mc-pricer)
- [Project 3 — Multi-Factor Equity Model with Fama-MacBeth & Momentum](https://github.com/Dev2943/factor-model)
- [Project 4 — VaR & ES Calculator with Backtesting & Stress Testing](https://github.com/Dev2943/var-calculator)

---

*Deployed on Google Cloud Run | Auto-deploys from GitHub | Last updated June 2026*
