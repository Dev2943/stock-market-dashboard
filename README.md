# 📊 Stock Market Analysis Platform

**Created by Dev Golakiya**  
*MS in Business Analytics | University of Massachusetts Amherst*

An advanced, production-ready stock market analysis dashboard featuring proprietary risk assessment, real-time data streaming, sector rotation analysis, and ML-powered forecasting. Built to demonstrate business analytics expertise and data-driven investment insights.

🔗 **Live Demo:** https://stock-market-dashboard-909492874362.us-central1.run.app

💻 **GitHub:** https://github.com/Dev2943/stock-market-dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Educational-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

---

## 🎯 Key Features

### 1. **Real-Time Market Data Integration** ⚡ NEW!
- **Yahoo Finance API** integration for live stock prices
- **Auto-Refresh Every 30 Seconds** - Hands-free live updates
- **Smart Caching** (30-second TTL) to minimize API calls
- **Dual-mode architecture**: Toggle between real-time and synthetic data
- **Graceful error handling** with automatic fallback to synthetic data
- **Period mapping** for flexible historical data (1y, 2y, 3y, 5y)
- **Manual refresh button** for instant data updates
- **Live countdown timer** showing next auto-refresh
- **Refresh statistics** tracking update frequency

### 2. **Proprietary Risk Scoring System** ⭐ (My Unique Contribution)
- **Dev's Risk Score** - Custom 0-100 scale assessment
- Multi-factor weighted model:
  - **Volatility Risk (30%)** - Annualized price volatility
  - **Momentum Risk (25%)** - Recent price movements and trend strength
  - **Liquidity Risk (20%)** - Volume consistency and market depth
  - **Technical Risk (15%)** - Overbought/oversold conditions via RSI
  - **Drawdown Risk (10%)** - Maximum loss potential from peak
- Visual component breakdown with risk categorization
- Actionable insights based on risk profile
- Risk-adjusted investment recommendations

### 3. **Sector Rotation Analysis**
- Performance tracking across 6+ market sectors (Technology, Financial, Healthcare, etc.)
- Momentum-based sector identification
- Correlation analysis for diversification insights
- **Inspired by category benchmarking methodologies from Andor Luxury internship**
- Interactive sector performance visualizations

### 4. **Advanced Technical Analysis**
- Multi-timeframe moving averages (20, 50, 200-day)
- Professional technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- Volume analysis with trend confirmation
- Professional candlestick charts with integrated indicators

### 5. **Portfolio Optimization**
- Modern portfolio theory implementation
- Correlation matrix showing stock relationships
- Diversification scoring (0-100 scale)
- Risk-adjusted return metrics:
  - Sharpe Ratio (risk-adjusted returns)
  - Sortino Ratio (downside risk focus)
  - Maximum Drawdown analysis
- Equal-weight and custom allocation support

### 6. **ML-Powered Forecasting**
- 30-60 day price predictions
- Hybrid approach: 70% historical momentum + 30% ML prediction
- 85% confidence intervals for uncertainty quantification
- Visual forecast charts with historical comparison
- Prediction methodology transparency

### 7. **Business Intelligence Features**
- Automated investment recommendations (BUY/SELL/HOLD)
- Multi-factor scoring system with reasoning
- Monthly returns heatmap
- Cumulative performance tracking
- Risk-adjusted signals based on portfolio context

---

## 🛠️ Technology Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core programming language | 3.8+ |
| **Streamlit** | Interactive web framework | 1.28+ |
| **yfinance** | Yahoo Finance API integration | 0.2.28+ |
| **Plotly** | Advanced financial charts | 5.15+ |
| **Pandas** | Data manipulation | 1.5.3+ |
| **NumPy** | Numerical computations | 1.24.3+ |

---

## 🚀 Getting Started

### Quick Install & Run
```bash
# Clone the repository
git clone https://github.com/Dev2943/stock-market-dashboard.git
cd stock-market-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

### Access the Dashboard
```
🌐 Local URL: http://localhost:8501
🌐 Network URL: http://192.168.x.x:8501
```

### Requirements
```txt
streamlit>=1.28.0
pandas>=1.5.3
numpy>=1.24.3
plotly>=5.15.0
yfinance>=0.2.28
```

---

## 📡 Data Sources & Auto-Refresh

### **Real-Time Mode (Default)** 🟢

**Features:**
- ✅ Live stock prices from Yahoo Finance
- ✅ **Auto-refresh every 30 seconds**
- ✅ Manual refresh button for instant updates
- ✅ Live countdown timer: "⏱️ Next refresh in: 27s"
- ✅ Refresh statistics: "🔄 Refreshes: 5 | Last: 03:25:14 PM"
- ✅ Toggle auto-refresh on/off
- ✅ Non-blocking UI - dashboard remains interactive during refresh

**How it Works:**
1. Enable "🔄 Auto-refresh (30s)" checkbox in sidebar
2. Countdown timer shows time until next refresh
3. Dashboard automatically reloads data every 30 seconds
4. Use "🔄 Force Refresh Now" button for instant updates
5. Refresh counter tracks total updates

**Smart Caching:**
- 30-second cache TTL balances freshness vs API efficiency
- Reduces API calls while maintaining near-real-time data
- Graceful error handling with automatic synthetic fallback

### **Synthetic Mode** 🟡
- Statistical data generation for testing
- No API rate limits or internet dependency
- Consistent for demonstrations
- Realistic price patterns based on market behavior

**How to Switch:**
Use the "📡 Data Source" radio button in sidebar to toggle between Real-Time and Synthetic modes.

---

## 💡 Usage Guide

### Getting Started in 3 Steps

1. **Select Stocks** 📈
   - Choose from 10+ major stocks (AAPL, MSFT, NVDA, TSLA, etc.)
   - Add custom tickers
   - Multi-select for portfolio analysis

2. **Enable Auto-Refresh** ⚡
   - Check "🔄 Auto-refresh (30s)" in sidebar
   - Watch live countdown timer
   - Dashboard updates automatically every 30 seconds

3. **Explore Tabs** 🔍
   - Navigate 6 analysis modules
   - View real-time price changes
   - Get actionable insights

---

### Dashboard Tabs Overview

| Tab | Features | Purpose |
|-----|----------|---------|
| **📊 Overview** | Price cards, key metrics, sector analysis | Market snapshot |
| **🎯 Risk Analysis** ⭐ | Proprietary risk score (0-100), component breakdown | Risk assessment |
| **📈 Technical** | Candlestick charts, RSI, MACD, Bollinger Bands | Technical signals |
| **🔮 Forecast** | ML predictions, confidence intervals, price targets | Future outlook |
| **📉 Performance** | Returns, Sharpe ratio, drawdown, heatmaps | Historical analysis |
| **💼 Portfolio** | Holdings, allocation, correlation, recommendations | Portfolio optimization |

---

## 📈 Methodology & Approach

### Risk Assessment Framework

My proprietary risk scoring system was developed during my MS in Business Analytics program at UMass Amherst, combining traditional financial metrics with modern quantitative methods.

**Why These Weights?**
- **Volatility (30%)** - Primary driver of portfolio risk; directly impacts position sizing
- **Momentum (25%)** - Recent price action predicts near-term behavior; crucial for timing
- **Liquidity (20%)** - Affects execution quality and slippage; critical for larger positions
- **Technical (15%)** - Overbought/oversold extremes signal reversals; momentum confirmation
- **Drawdown (10%)** - Historical worst-case scenarios; tail risk assessment

**Risk Categories:**
```
0-20   → Very Low Risk (Conservative investors)
20-35  → Low Risk (Moderate-conservative investors)
35-50  → Moderate Risk (Balanced investors)
50-70  → High Risk (Growth-oriented investors)
70-100 → Very High Risk (Aggressive investors only)
```

### Auto-Refresh Architecture

**Technical Implementation:**
```python
# Session state management
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = datetime.now()

# Calculate time elapsed
time_since_refresh = (datetime.now() - st.session_state.last_refresh_time).total_seconds()

# Auto-refresh when 30 seconds pass
if time_since_refresh >= 30:
    st.cache_data.clear()
    st.rerun()
```

**Key Design Decisions:**
- ✅ **Non-blocking**: Uses session state instead of `time.sleep()`
- ✅ **User control**: Toggle on/off anytime
- ✅ **Visual feedback**: Live countdown timer
- ✅ **Manual override**: Force refresh button
- ✅ **State persistence**: Tracks refresh history

### Forecasting Approach

**Hybrid Methodology:**

1. **Historical Momentum (70% weight)**
   - 30-day price momentum captures recent trend strength
   - More reliable than pure ML for near-term predictions
   - Grounded in actual market behavior

2. **ML Prediction (30% weight)**
   - Trend analysis with mean reversion principles
   - Volatility-adjusted for realistic scenarios
   - Adds forward-looking perspective

3. **Confidence Intervals (85% CI)**
   - Acknowledges inherent market uncertainty
   - Helps with risk management decisions
   - Based on historical volatility patterns

**Important Note:** Forecasts are for educational purposes. Past performance does not guarantee future results.

---

## 🔍 What Makes This Project Unique

### 1. **Production-Ready Auto-Refresh** ⚡ NEW!
Unlike typical demo dashboards, this implements:
- Non-blocking auto-refresh using Streamlit session state
- Live countdown timer for user feedback
- Toggle control for user preference
- Refresh statistics tracking
- Professional UX with manual override option

This demonstrates production-level thinking beyond just analytics.

### 2. **Original Risk Scoring Methodology** ⭐
Unlike standard technical analysis tools, I developed a proprietary weighted scoring system based on academic research and industry best practices. The 5-factor model with specific weights (30/25/20/15/10) was calibrated during my MS coursework.

### 3. **Smart API Integration**
Real-world implementation of external API with:
- 30-second smart caching (balances freshness vs efficiency)
- Graceful error handling and automatic fallback
- Rate limit awareness
- Dual-mode architecture for flexibility

### 4. **Cross-Domain Application**
Applied category analysis techniques from luxury retail (Andor Luxury) to financial sector analysis, demonstrating ability to transfer analytical frameworks across industries.

### 5. **Risk-First Philosophy**
Every recommendation considers risk score first, preventing high-risk buys even with positive technical signals - reflecting real-world investment discipline.

### 6. **Educational Transparency**
All methodologies are documented and explained, demonstrating not just technical skills but ability to communicate complex concepts clearly.

---

## 📊 Sample Outputs

### Real-Time Dashboard Features

**Auto-Refresh Controls:**
```
✅ Using live market data
⚡ Data refreshes automatically

☑ 🔄 Auto-refresh (30s)
⏱️ Next refresh in: 23s

[🔄 Force Refresh Now]

🔄 Refreshes: 12 | Last: 03:45:32 PM
```

**Risk Score Dashboard:**
- Visual risk gauge: `67/100` (color-coded red/yellow/green)
- Component breakdown showing 5 factors
- 3-5 automated insights
- Investor suitability recommendation

**Portfolio Analysis:**
- Interactive correlation heatmap
- Diversification score: `72/100` 🟢
- Individual recommendations with reasoning
- Expected returns: `+8.3%`

---

## 🎓 Academic & Professional Background

This project demonstrates skills developed during:

### **MS in Business Analytics** - UMass Amherst (GPA: 4.0/4.0)
- Quantitative analysis and statistical modeling
- Portfolio optimization and financial analytics
- Machine learning and predictive modeling

### **Business Analyst Intern** - Andor Luxury, NY (June-Aug 2025)
- Conducted market research across 3 product categories
- Built dynamic Excel/Power BI dashboards tracking 12+ KPIs
- Analyzed 5,000+ records to uncover purchasing trends
- **Category benchmarking → Sector analysis inspiration**

### **Assistant Project Manager** - Vrundev Corporation, India (May-Dec 2024)
- Led $600K+ projects with data-driven decisions
- Reduced operating costs by 12% through analytics
- Managed cross-functional teams

### Key Competencies Demonstrated:
✓ Python programming & API integration  
✓ Real-time data streaming & caching  
✓ Quantitative analysis & statistical modeling  
✓ Production-ready application development  
✓ Business intelligence & data visualization  
✓ Risk assessment & portfolio optimization  
✓ Full-stack deployment (Streamlit Cloud)  

---

## 🔄 Recent Enhancements

### Version 2.1 (Latest) ⚡
**Auto-Refresh Feature** - Major UX Improvement
- [x] Non-blocking 30-second auto-refresh
- [x] Live countdown timer in sidebar
- [x] Manual refresh button with instant updates
- [x] Refresh statistics tracking
- [x] Toggle control for user preference
- [x] Session state management for reliability

### Version 2.0
**Real-Time Data Integration**
- [x] Yahoo Finance API integration
- [x] Smart 30-second caching
- [x] Dual-mode architecture (real-time vs synthetic)
- [x] Graceful error handling

---

## 🚀 Future Enhancements

### Phase 1 (Next 3 months)
- [ ] WebSocket integration for tick-by-tick updates
- [ ] Email/SMS alerts for price targets
- [ ] Multiple API providers (Alpha Vantage, Polygon.io)
- [ ] Dark mode theme toggle

### Phase 2 (6 months)
- [ ] Options pricing and Greeks calculation
- [ ] Backtesting framework with performance metrics
- [ ] Advanced ML models (LSTM, Prophet, ARIMA)
- [ ] News sentiment analysis integration

### Phase 3 (Long-term)
- [ ] Mobile app (React Native)
- [ ] Social trading features
- [ ] Multi-asset classes (bonds, commodities, crypto)
- [ ] Advanced portfolio optimization (Black-Litterman)

---

## 📝 Technical Documentation

### Project Structure
```
stock-market-dashboard/
├── dashboard.py          # Main application (1,100+ lines)
├── README.md            # Documentation (this file)
├── requirements.txt     # Dependencies
└── .gitignore          # Git ignore rules
```

### Key Functions
```python
# Real-time data fetching with caching
@st.cache_data(ttl=30)
def fetch_real_stock_data(symbol, period="1y"):
    """Fetch live data from Yahoo Finance with error handling"""
    
# Proprietary risk scoring
def calculate_dev_risk_score(df):
    """5-factor weighted risk model (0-100 scale)"""
    
# Auto-refresh management
def handle_auto_refresh():
    """Non-blocking refresh with countdown timer"""
    
# Investment recommendations
def generate_recommendation(df, risk_score):
    """Multi-factor BUY/SELL/HOLD signals"""
```

---

## ⚠️ Disclaimer

**This dashboard is for educational and portfolio demonstration purposes only.**

- ❌ NOT financial advice
- ❌ Past performance ≠ future results
- ✅ Always conduct your own research
- ✅ Consult qualified financial advisors
- ⚠️ Yahoo Finance data may have delays (free tier)

---

## 📧 Contact

**Dev Golakiya**  
📧 devgolakiya31@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/devgolakiya)  
💻 [GitHub](https://github.com/Dev2943)  
📍 Amherst, Massachusetts

**💼 Available for Data Analyst and Business Analyst opportunities starting January 2027**

---

## 📈 Performance & Metrics

### Code Statistics
- **1,100+ lines** of production-ready Python code
- **6 analytical modules** with 20+ visualizations
- **15+ custom functions** including API integration
- **5-factor proprietary model** for risk assessment
- **Real-time streaming** with 30-second auto-refresh
- **100% uptime** with graceful error handling

### Skills Demonstrated
✅ Python Programming (Pandas, NumPy, Plotly)  
✅ **Real-Time Data Streaming & Auto-Refresh** ⚡  
✅ **Production Error Handling & State Management**  
✅ REST API Integration (Yahoo Finance)  
✅ Statistical Analysis & Quantitative Methods  
✅ Machine Learning & Predictive Modeling  
✅ Interactive Dashboard Design (Streamlit)  
✅ Full-Stack Deployment (Streamlit Cloud)  
✅ Git Version Control & Documentation  

---

## 🙏 Acknowledgments


- **Streamlit & Plotly Communities** - Excellent frameworks and support
- **Yahoo Finance** - Free API access for market data

---

## 📄 License

This project is for educational and portfolio demonstration purposes.

**MIT License** - Feel free to fork and learn from this project!

---

*Last Updated: May 2025*  
*Version: 2.1 - Auto-Refresh Enhancement*

---

**⭐ If you found this project helpful, please star the repository!**

---

## 🔗 Quick Links

- 🌐 **Live Demo**: https://dev2943-stock-market-dashboard-dashboard.streamlit.app
- 💻 **GitHub**: https://github.com/Dev2943/stock-market-dashboard
- 📧 **Email**: devgolakiya31@gmail.com
- 🔗 **LinkedIn**: https://www.linkedin.com/in/devgolakiya

---
