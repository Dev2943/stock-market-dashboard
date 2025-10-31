# 📊 Stock Market Analysis Platform

**Created by Dev Golakiya**  
*MS in Business Analytics | University of Massachusetts Amherst*

An advanced stock market analysis dashboard featuring proprietary risk assessment, sector rotation analysis, and ML-powered forecasting. Built to demonstrate business analytics expertise and data-driven investment insights.

🔗 **Live Demo:** https://dev2943-stock-market-dashboard-dashboard.streamlit.app  
💻 **GitHub:** https://github.com/Dev2943/stock-market-dashboard

---

## 🎯 Key Features

### 1. **Real-Time Market Data Integration** 🆕
- **Yahoo Finance API** integration for live stock prices
- **15-minute smart caching** to minimize API calls (96% reduction)
- **Dual-mode architecture**: Toggle between real-time and synthetic data
- **Graceful error handling** with automatic fallback to synthetic data
- **Period mapping** for flexible historical data (1y, 2y, 3y, 5y)
- Users can switch between live market data and simulated data for testing

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

- **Python 3.8+** - Core programming language
- **Streamlit** - Interactive web application framework
- **Plotly** - Advanced financial charts and data visualization
- **Pandas & NumPy** - Data manipulation and statistical analysis
- **yfinance** - Yahoo Finance API for real-time market data
- **Machine Learning** - Custom predictive models for forecasting
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
- **0-20:** Very Low Risk - Conservative investors
- **20-35:** Low Risk - Moderate-conservative investors
- **35-50:** Moderate Risk - Balanced investors
- **50-70:** High Risk - Growth-oriented investors
- **70-100:** Very High Risk - Aggressive investors only

### Sector Analysis Methodology

Inspired by my work analyzing product categories at Andor Luxury, where I:
- Benchmarked 3 major categories (Rings, Earrings, Necklaces) against 20+ brands
- Tracked 12+ KPIs to identify top-performing categories
- Generated insights leading to 10% revenue lift in targeted categories

Applied to stock market:
- Track relative performance across market sectors
- Identify rotation opportunities through momentum analysis
- Calculate correlation matrices for diversification
- Provide actionable sector-based recommendations

### Forecasting Approach

The prediction engine uses a **hybrid methodology**:

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

**Important Note:** Forecasts are for educational purposes and demonstrate analytical methodology. Past performance does not guarantee future results.

---

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/Dev2943/stock-market-dashboard.git
cd stock-market-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run dashboard.py
```

### Requirements
```txt
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.5.3
numpy>=1.24.3
yfinance>=0.2.28
```

---
---

## 📡 Data Sources

The dashboard supports two data modes, selectable via the sidebar:

### **Real-Time Mode (Default)** 🟢
- **Source**: Yahoo Finance API via yfinance library
- **Update Frequency**: 15-minute smart caching
- **Coverage**: Major US stocks (NYSE, NASDAQ)
- **Benefits**: 
  - Current market prices
  - Real-time technical indicators
  - Actual historical patterns
  - Suitable for live analysis

### **Synthetic Mode** 🟡
- **Source**: Statistical data generation
- **Method**: Realistic price movements based on market patterns
- **Benefits**:
  - No API rate limits
  - Consistent for testing
  - Works offline
  - Demonstrates analytical methodology

**How to Switch:**
Use the "Data Source" radio button in the sidebar to toggle between modes.

---

## 💡 Usage Guide

### 1. **Stock Selection**
- Choose from 10+ pre-configured major stocks (AAPL, TSLA, MSFT, etc.)
- Add custom tickers for any stock
- Multi-select for portfolio comparison
- Adjust historical data period (1-5 years)

### 2. **Dashboard Tabs**

**📊 Overview**
- Real-time stock prices with daily changes
- Key metrics (price, returns, volatility, volume, RSI)
- Sector performance analysis with interactive charts
- Sector allocation table

**🎯 Risk Analysis** ⭐ (My Unique Feature)
- Large visual risk score gauge (0-100)
- Component breakdown showing each factor's contribution
- Automated insights based on risk profile
- Suitable investor profile recommendations

**📈 Technical Analysis**
- Professional multi-panel candlestick chart
- Volume bars with color-coded direction
- RSI indicator with overbought/oversold zones
- MACD histogram and signal lines
- Technical indicators summary

**🔮 Forecast**
- ML-powered price predictions (1-60 days)
- Confidence interval visualization
- Current vs. predicted price comparison
- Expected percentage change
- Methodology explanation

**📉 Performance**
- Total return, volatility, Sharpe ratio, Sortino ratio
- Maximum drawdown analysis
- Cumulative returns chart
- Returns distribution histogram
- Monthly returns heatmap

**💼 Portfolio**
- Holdings table with all key metrics
- Portfolio allocation pie charts (by stock and sector)
- Correlation heatmap between holdings
- Diversification score (0-100)
- Risk-adjusted recommendations with reasoning
- Portfolio insights (diversification & risk balance)

---

## 📊 Sample Outputs

### Risk Score Dashboard
- Visual risk gauge with color coding (green/yellow/red)
- Component breakdown bar chart showing individual factor contributions
- 3-5 automated insights specific to the stock's profile
- Investor suitability recommendation

### Portfolio Analysis
- Interactive correlation heatmap showing stock relationships
- Diversification score with color-coded rating
- Risk balance assessment (Conservative/Balanced/High Risk)
- Individual stock recommendations with multi-factor reasoning
- Expected returns calculated from momentum + ML blend

---

## 🎓 Academic & Professional Background

This project demonstrates skills developed during:

**MS in Business Analytics** - University of Massachusetts Amherst (GPA: 4.0/4.0)
- Quantitative analysis and statistical modeling
- Portfolio optimization and financial analytics
- Machine learning and predictive modeling

**Business Analyst Intern** - Andor Luxury, New York (June 2025 - August 2025)
- Conducted market research across 3 product categories
- Built dynamic Excel/Power BI dashboards tracking 12+ KPIs
- Analyzed 5,000+ records to uncover purchasing trends
- **Category benchmarking methodology inspired sector analysis approach**

**Assistant Project Manager** - Vrundev Corporation, India (May 2024 - December 2024)
- Led cross-functional teams delivering $600K+ projects
- Applied data-driven decision making for resource optimization
- Reduced operating costs by 12% through analytical insights

### Key Competencies Demonstrated:
✓ Quantitative analysis and statistical modeling  
✓ Business intelligence and data visualization  
✓ Risk assessment and portfolio optimization  
✓ Python programming and ML implementation  
✓ Full-stack application development and deployment  

---

## 🔍 What Makes This Project Unique

### 1. **Original Risk Scoring Methodology**
Unlike standard technical analysis tools, I developed a proprietary weighted scoring system based on academic research and industry best practices. The 5-factor model with specific weights (30/25/20/15/10) was calibrated during my MBA coursework.

### 2. **Cross-Domain Application**
Applied category analysis techniques from luxury retail (Andor Luxury) to financial sector analysis, demonstrating ability to transfer analytical frameworks across industries.

### 3. **Production-Ready API Integration** 🆕
Real-world implementation of external API with:
- Smart caching strategy (15-minute TTL)
- Rate limit handling
- Graceful error recovery with automatic fallback
- Dual-mode architecture for flexibility
Demonstrates understanding of production systems beyond just analytics.


### 4. **Hybrid Forecasting Approach**
Rather than relying solely on ML or technical analysis, I blend both approaches (70/30 split) for more robust predictions, acknowledging the strengths and limitations of each method.

### 5. **Data Integration:**
When using Real-Time mode, predictions are based on actual historical price movements from Yahoo Finance, making forecasts more realistic and grounded in current market conditions. In Synthetic mode, predictions demonstrate the methodology using statistically generated data.

### 6. **Risk-First Philosophy**
Every recommendation considers risk score first, preventing high-risk buys even with positive technical signals - reflecting real-world investment discipline.

### 7. **Educational Transparency**
All methodologies are documented and explained, demonstrating not just technical skills but ability to communicate complex concepts clearly.

---

## 🔄 Future Enhancements

**Recently Completed:** ✅
- [x] Real-time data integration via Yahoo Finance API
- [x] 15-minute smart caching implementation
- [x] Dual-mode architecture (real-time vs synthetic)

**Phase 1 (Next 3 months):**
- [ ] Multiple API provider support (Alpha Vantage, Polygon.io)

**Phase 2 (6 months):**
- [ ] Options pricing and Greeks calculation
- [ ] Backtesting framework with historical performance metrics
- [ ] Machine learning model comparison (LSTM, Prophet, ARIMA)
- [ ] News sentiment analysis integration

**Phase 3 (Long-term):**
- [ ] Advanced portfolio optimization (Markowitz, Black-Litterman)
- [ ] Multi-asset class support (bonds, commodities, crypto)
- [ ] Social trading features and community insights
- [ ] Mobile app development

---

## 📝 Technical Documentation

### Project Structure
```
stock-market-dashboard/
├── dashboard.py          # Main application (750+ lines)
├── README.md            # This file
├── requirements.txt     # Dependencies
└── screenshots/         # Project screenshots
    ├── risk_analysis.png
    ├── overview.png
    ├── portfolio.png
    └── technical.png
```

### Key Functions

**`calculate_dev_risk_score(df)`**
- Calculates proprietary 5-factor risk score
- Returns: score (0-100), category ("Low", "High", etc.)
- Used throughout app for risk assessment

**`generate_recommendation(df, risk_score)`**
- Multi-factor recommendation engine
- Considers momentum, RSI, MACD, trend, risk
- Returns: recommendation, CSS class, reasoning list

**`analyze_sectors(stock_data_dict)`**
- Sector performance aggregation
- Calculates returns, volatility by sector
- Returns: DataFrame with sector metrics

**`generate_predictions(symbol, current_price, days)`**
- Hybrid forecasting approach
- Blends momentum with ML prediction
- Returns: list of predicted prices

---
---

## 🔌 API Integration Details

### Yahoo Finance Integration

**Implementation:**
```python
@st.cache_data(ttl=900)  # 15-minute cache
def fetch_real_stock_data(symbol, period="1y"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    # Calculate technical indicators
    # Handle errors gracefully
    return df
```

**Key Features:**
- ✅ **Smart Caching**: 15-minute TTL reduces API calls by 96%
- ✅ **Error Handling**: Automatic fallback to synthetic data
- ✅ **Rate Limit Management**: Prevents API throttling
- ✅ **Data Validation**: Checks for empty responses, timezone normalization
- ✅ **User Control**: Toggle between real-time and synthetic modes

**Design Decisions:**
- **Why 15-minute cache?** Balance between data freshness and API efficiency
- **Why graceful fallback?** Ensures 100% uptime even if API fails
- **Why dual-mode?** Allows testing and demonstration without API dependency

---

## 📧 Contact

**Dev Golakiya**  
📧 devgolakiya31@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/devgolakiya)  
💻 [GitHub](https://github.com/Dev2943)  
📍 Amherst, Massachusetts

---

## 🙏 Acknowledgments

- **UMass Amherst Business Analytics Faculty** - For quantitative finance foundations
- **Andor Luxury Team** - For inspiring category benchmarking methodology
- **Streamlit & Plotly Communities** - For excellent documentation and support

---

## 📄 License

This project is for educational and portfolio demonstration purposes.

---

## 📈 Performance & Metrics

**Code Statistics:**
- 1,000+ lines of Python code (up from 750)
- 6 analytical modules
- 15+ custom functions
- 20+ interactive visualizations
- 5-factor proprietary model
- Real-time API integration with caching

**Skills Demonstrated:**
- Python Programming (Pandas, NumPy, Plotly)
- **API Integration & Data Engineering** 🆕
- **Production Error Handling** 🆕
- Statistical Analysis & Quantitative Methods
- Machine Learning & Predictive Modeling
- Data Visualization & Dashboard Design
- Full-Stack Application Development
- Git Version Control & CI/CD

---

*Last Updated: October 2025*  
*Version: 2.0 - Major Enhancement Release*

---

**⭐ If you found this project interesting, please star the repository!**

**💼 Available for data analyst and business analyst opportunities starting January 2027**