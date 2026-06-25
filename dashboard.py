"""
Stock Market Analysis Dashboard
Created by: Dev Golakiya
UMass Amherst - MS in Business Analytics

Custom Features:
- Proprietary risk scoring system based on multi-factor analysis
- Sector rotation analysis inspired by category benchmarking at Andor Luxury
- Enhanced portfolio optimization using modern portfolio theory
- Business-focused insights and actionable recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import time
import logging
import math
from contextlib import contextmanager
import streamlit.components.v1 as components

# Lightweight performance instrumentation. Flip to False to strip out all
# timing/cache-tracking overhead and hide the debug panel entirely.
DEBUG_PERFORMANCE = True

logging.basicConfig(level=logging.INFO)
_timing_logger = logging.getLogger("dashboard.timing")

# Reset every rerun for free: Streamlit re-executes the whole script
# top-to-bottom on each interaction, so these module-level dicts start empty
# every time without any explicit reset logic.
_TIMINGS = {}
_CACHE_STATUS = {}


@contextmanager
def timed(label):
    """Accumulate elapsed wall time under `label` in _TIMINGS. True no-op when DEBUG_PERFORMANCE is off."""
    if not DEBUG_PERFORMANCE:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _TIMINGS[label] = _TIMINGS.get(label, 0.0) + (time.perf_counter() - t0)


def _record_cache_call(label, hit):
    """Track cache hit/miss counts for a cached fetch. No-op when DEBUG_PERFORMANCE is off."""
    if not DEBUG_PERFORMANCE:
        return
    status = _CACHE_STATUS.setdefault(label, {"hits": 0, "misses": 0})
    status["hits" if hit else "misses"] += 1


def _log_timing_report():
    """Log one formatted breakdown of this rerun's timings. No-op when DEBUG_PERFORMANCE is off."""
    if not DEBUG_PERFORMANCE:
        return

    total = _TIMINGS.get("Total rerun time", 0.0)
    section_labels = [
        "Overview rendering", "Risk Analysis rendering", "Technical Analysis rendering",
        "Forecast rendering", "Performance rendering", "Portfolio rendering",
        "VaR & Stress Test rendering", "Options Pricer rendering", "Factor Model rendering",
    ]
    computation_labels = [
        "Network fetch time", "Indicator calculation", "Portfolio calculations",
        "VaR", "Monte Carlo", "Factor model",
    ]

    lines = ["Performance report:"]
    lines.append("  Computation:")
    for label in computation_labels:
        seconds = _TIMINGS.get(label, 0.0)
        pct = (seconds / total * 100) if total > 0 else 0.0
        lines.append(f"    {label:<24} {seconds:8.4f}s  ({pct:5.1f}%)")

    lines.append("  Cache hit/miss:")
    for label, status in _CACHE_STATUS.items():
        lines.append(f"    {label:<24} hits={status['hits']:<4} misses={status['misses']}")

    lines.append("  Rendering by section:")
    for label in section_labels:
        seconds = _TIMINGS.get(label, 0.0)
        pct = (seconds / total * 100) if total > 0 else 0.0
        lines.append(f"    {label:<28} {seconds:8.4f}s  ({pct:5.1f}%)")

    lines.append(f"  Total rerun time:           {total:8.4f}s")

    _timing_logger.info("\n".join(lines))


st.set_page_config(
    page_title="Stock Analysis | Dev Golakiya",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with personal branding
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FAFAFA;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .recommendation-buy {
        background: #008000;
        padding: 1rem;
        border-radius: 10px;
        color: #FAFAFA;
        text-align: left;
        margin: 0.5rem 0;
    }
    .recommendation-sell {
        background: #B00020;
        padding: 1rem;
        border-radius: 10px;
        color: #FAFAFA;
        text-align: left;
        margin: 0.5rem 0;
    }
    .recommendation-hold {
        background: #4B5563;
        padding: 1rem;
        border-radius: 10px;
        color: #FAFAFA;
        text-align: left;
        margin: 0.5rem 0;
    }
    .risk-score-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .risk-score-medium {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .risk-score-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }

    /* ---- Motion layer: tasteful, institutional. Transform/opacity only,
       fully neutered under prefers-reduced-motion. No new colors. ---- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes livePulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.35; }
    }
    @keyframes shimmer {
        0% { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }
    @keyframes tickerScroll {
        from { transform: translateX(0); }
        to { transform: translateX(-50%); }
    }

    .risk-score-low, .risk-score-medium, .risk-score-high,
    .recommendation-buy, .recommendation-sell, .recommendation-hold {
        transition: transform 180ms cubic-bezier(.23, 1, .32, 1),
                    box-shadow 180ms cubic-bezier(.23, 1, .32, 1);
        animation: fadeInUp 420ms cubic-bezier(.23, 1, .32, 1) both;
    }
    .risk-score-low:hover, .risk-score-medium:hover, .risk-score-high:hover,
    .recommendation-buy:hover, .recommendation-sell:hover, .recommendation-hold:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }

    /* Live status badge - dot encodes real state (live feed vs simulated), not decoration */
    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: #FAFAFA;
        margin: 0.25rem 0 0.75rem 0;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: green;
        flex-shrink: 0;
        animation: livePulse 2s ease-in-out infinite;
    }
    .live-dot.is-static {
        background: #666666;
        animation: none;
    }

    /* Scrolling ticker tape */
    .ticker-wrap {
        overflow: hidden;
        background: #262730;
        border-radius: 5px;
        padding: 0.5rem 0;
        margin-bottom: 0.75rem;
        white-space: nowrap;
    }
    .ticker-track {
        display: inline-flex;
        animation: tickerScroll 38s linear infinite;
    }
    .ticker-wrap:hover .ticker-track {
        animation-play-state: paused;
    }
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0 1.5rem;
        font-size: 0.95rem;
        color: #FAFAFA;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .ticker-symbol { font-weight: 700; }

    /* Skeleton loading placeholder, matches the KPI card shape it precedes */
    .skeleton-card {
        background: #262730;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        height: 96px;
    }
    .skeleton-line {
        height: 12px;
        border-radius: 4px;
        margin: 0.6rem auto;
        background: linear-gradient(90deg, #262730 25%, #34374040 50%, #262730 75%);
        background-size: 800px 100%;
        animation: shimmer 1.6s linear infinite;
    }

    @media (prefers-reduced-motion: reduce) {
        .risk-score-low, .risk-score-medium, .risk-score-high,
        .recommendation-buy, .recommendation-sell, .recommendation-hold,
        .live-dot, .ticker-track, .skeleton-line {
            animation: none !important;
            transition: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sector classifications (expanded)
STOCK_SECTORS = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
    'TSLA': 'Consumer', 'AMZN': 'Consumer', 'NFLX': 'Communication',
    'JPM': 'Financial', 'BAC': 'Financial', 'JNJ': 'Healthcare',
    'PFE': 'Healthcare', 'XOM': 'Energy', 'CVX': 'Energy'
}

@st.cache_data
def generate_stock_data(symbol, days=252):
    """Generate realistic stock data with proper technical indicators"""
    np.random.seed(hash(symbol) % 1000)
    
    base_prices = {
        'AAPL': 256.08, 'TSLA': 416.66, 'MSFT': 428.50, 'GOOGL': 182.45,
        'AMZN': 228.19, 'NVDA': 120.00, 'META': 595.80, 'NFLX': 445.25,
        'AMD': 138.40, 'INTC': 24.15
    }
    
    base_price = base_prices.get(symbol, 150.00)
    dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
    
    # Realistic price generation
    prices = []
    start_price = base_price * 0.75
    target_price = base_price
    daily_trend = ((target_price / start_price) - 1) / days
    current_price = start_price
    
    for i in range(days):
        daily_return = daily_trend + np.random.normal(0, 0.02)
        volatility_factor = 1 + np.sin(i/30) * 0.1
        
        if i == days - 1:
            current_price = target_price
        else:
            current_price *= (1 + daily_return * volatility_factor)
        
        prices.append(current_price)
    
    # OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.015)))
        low = close * (1 - abs(np.random.normal(0, 0.015)))
        open_price = prices[i-1] if i > 0 else close
        volume = int(np.random.normal(1000000, 300000))
        
        data.append({
            'Date': date, 'Open': open_price,
            'High': max(open_price, high, close),
            'Low': min(open_price, low, close),
            'Close': close, 'Volume': max(volume, 100000)
        })
    
    df = pd.DataFrame(data).set_index('Date')

    with timed("Indicator calculation"):
        # Technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

    return df
@st.cache_data(ttl=30)  # 30 seconds - fastest safe option
def fetch_real_stock_data(symbol, period="1y"):
    """
    Fetch real-time stock data from Yahoo Finance
    Cache expires every 30 seconds for near-real-time updates

    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
    
    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    # @st.cache_data short-circuits on a hit before this body ever runs, so
    # reaching this line means it was a miss.
    _record_cache_call("fetch_real_stock_data", hit=False)
    try:
        # Download real data from Yahoo Finance
        with timed("Network fetch time"):
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

        # Check if data was retrieved
        if df.empty:
            st.warning(f"⚠️ No data available for {symbol}. Using synthetic data as fallback.")
            return generate_stock_data(symbol, 252)
        
        # Clean and prepare data
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Remove timezone info if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        with timed("Indicator calculation"):
            # Technical indicators
            df['Returns'] = df['Close'].pct_change()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['Volatility'] = df['Returns'].rolling(window=20).std()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']

            # Bollinger Bands
            bb_window = 20
            bb_std = 2
            df['BB_middle'] = df['Close'].rolling(window=bb_window).mean()
            bb_std_dev = df['Close'].rolling(window=bb_window).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std_dev * bb_std)
            df['BB_lower'] = df['BB_middle'] - (bb_std_dev * bb_std)

        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching data for {symbol}: {str(e)}")
        st.info(f"📊 Using synthetic data for {symbol} as fallback")
        return generate_stock_data(symbol, 252)

def calculate_dev_risk_score(df):
    """
    Dev's Proprietary Risk Assessment
    Multi-factor model: Volatility(30%), Momentum(25%), Liquidity(20%), Technical(15%), Drawdown(10%)
    """
    if len(df) < 50:
        return 50.0, "Insufficient data"
    
    # Volatility Risk
    volatility = df['Returns'].std() * np.sqrt(252)
    vol_score = min(volatility / 0.50, 1.0) * 30
    
    # Momentum Risk
    recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-20]) - 1
    mom_volatility = df['Returns'].tail(20).std()
    mom_score = min((abs(recent_return) * 0.5 + mom_volatility * 10) * 25, 25)
    
    # Liquidity Risk
    volume_cv = df['Volume'].std() / df['Volume'].mean() if df['Volume'].mean() > 0 else 1
    liq_score = min(volume_cv, 1.0) * 20
    
    # Technical Risk
    latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
    tech_score = (abs(latest_rsi - 50) / 50) * 15
    
    # Drawdown Risk
    returns = df['Returns'].fillna(0)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    dd_score = min(abs(drawdown.min()) * 2, 1.0) * 10
    
    total_risk = vol_score + mom_score + liq_score + tech_score + dd_score
    
    # Risk category
    if total_risk < 20:
        category = "Very Low"
    elif total_risk < 35:
        category = "Low"
    elif total_risk < 50:
        category = "Moderate"
    elif total_risk < 70:
        category = "High"
    else:
        category = "Very High"
    
    return round(min(total_risk, 100), 2), category

def generate_insights(df, risk_score):
    """Generate business analyst-style insights"""
    insights = []
    
    # Volatility
    vol = df['Returns'].std() * np.sqrt(252) * 100
    if vol > 40:
        insights.append(f"⚠️ High volatility ({vol:.1f}% annually) - expect significant price swings")
    elif vol < 15:
        insights.append(f"✓ Stable price action ({vol:.1f}% volatility) - suitable for conservative portfolios")
    
    # Trend
    current = df['Close'].iloc[-1]
    sma20 = df['SMA_20'].iloc[-1]
    sma50 = df['SMA_50'].iloc[-1]
    
    if not pd.isna(sma20) and not pd.isna(sma50):
        if current > sma20 > sma50:
            insights.append("✓ Strong uptrend confirmed - price above both 20-day and 50-day moving averages")
        elif current < sma20 < sma50:
            insights.append("⚠️ Downtrend pattern - price below key moving averages, consider defensive approach")
        elif current > sma50:
            insights.append("📊 Mixed signals - above 50-day MA but trend unclear")
    
    # Volume
    recent_vol = df['Volume'].tail(10).mean()
    avg_vol = df['Volume'].mean()
    if recent_vol > avg_vol * 1.3:
        insights.append("📈 Volume surge detected - indicates strong market interest")
    elif recent_vol < avg_vol * 0.7:
        insights.append("📉 Below-average volume - limited market participation")
    
    # RSI
    rsi = df['RSI'].iloc[-1]
    if not pd.isna(rsi):
        if rsi > 70:
            insights.append(f"🔴 Overbought (RSI: {rsi:.1f}) - potential pullback ahead")
        elif rsi < 30:
            insights.append(f"🟢 Oversold (RSI: {rsi:.1f}) - potential buying opportunity")
    
    # Risk-based insight
    if risk_score > 70:
        insights.append("⚠️ High risk score - only suitable for aggressive growth portfolios")
    elif risk_score < 30:
        insights.append("✓ Low risk profile - suitable for income-focused investors")
    
    return insights

def analyze_sectors(stock_data_dict):
    """Sector performance analysis (inspired by Andor Luxury category benchmarking)"""
    sector_perf = {}
    
    for stock, df in stock_data_dict.items():
        sector = STOCK_SECTORS.get(stock, 'Other')
        
        if sector not in sector_perf:
            sector_perf[sector] = {'returns': [], 'vol': [], 'stocks': []}
        
        ret = (df['Close'].iloc[-1] / df['Close'].iloc[-20]) - 1 if len(df) >= 20 else 0
        vol = df['Returns'].std() * np.sqrt(252)
        
        sector_perf[sector]['returns'].append(ret * 100)
        sector_perf[sector]['vol'].append(vol * 100)
        sector_perf[sector]['stocks'].append(stock)
    
    summary = []
    for sector, data in sector_perf.items():
        summary.append({
            'Sector': sector,
            'Avg Return (20d)': np.mean(data['returns']),
            'Avg Volatility': np.mean(data['vol']),
            'Stocks': len(data['stocks'])
        })
    
    return pd.DataFrame(summary).sort_values('Avg Return (20d)', ascending=False)

@st.cache_data
@st.cache_data
def generate_predictions(symbol, current_price, days=30):
    """
    Generate realistic price predictions based on historical volatility
    
    Args:
        symbol: Stock ticker
        current_price: Current stock price
        days: Number of days to predict
    
    Returns:
        List of predicted prices
    """
    np.random.seed(hash(symbol) % 1000)
    
    # More realistic parameters
    # Annual drift (expected return): 8-12% annually = 0.025-0.035% daily
    annual_return = 0.10  # 10% annual expected return
    daily_drift = annual_return / 252  # Convert to daily
    
    # Daily volatility: realistic range 1-3%
    daily_volatility = 0.015  # 1.5% daily volatility
    
    predictions = []
    price = current_price
    
    for day in range(1, days + 1):
        # Random walk with drift (more realistic model)
        daily_return = np.random.normal(daily_drift, daily_volatility)
        
        # Add mean reversion (prevents wild swings)
        if day > 10:
            # Pull back toward starting price slightly
            deviation = (price - current_price) / current_price
            mean_reversion = -0.1 * deviation  # 10% mean reversion
            daily_return += mean_reversion
        
        # Cap maximum daily moves at ±5%
        daily_return = np.clip(daily_return, -0.05, 0.05)
        
        price = price * (1 + daily_return)
        predictions.append(price)
    
    return predictions

def calculate_performance_metrics(df):
    """Enhanced performance metrics"""
    returns = df['Returns'].dropna()
    if len(returns) == 0:
        return {'total_return': 0, 'volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'sortino_ratio': 0}
    
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    excess_returns = returns.mean() * 252 - 0.02
    sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
    
    # Sortino Ratio (downside risk only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
    sortino_ratio = excess_returns / downside_std if downside_std > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown
    }

def generate_recommendation(df, risk_score):
    """
    Enhanced recommendation system with risk consideration
    Fixed to properly account for expected returns and risk
    """
    if len(df) == 0:
        return "HOLD", "recommendation-hold", []
    
    latest = df.iloc[-1]
    score = 0
    reasons = []
    
    # 1. EXPECTED RETURN ANALYSIS (Most Important - Weight: 40%)
    # Calculate 30-day expected return
    if len(df) >= 30:
        monthly_return = (latest['Close'] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
        
        if monthly_return > 0.10:  # >10% gain
            score += 3
            reasons.append(f"Strong positive momentum (+{monthly_return*100:.1f}% over 30 days)")
        elif monthly_return > 0.05:  # 5-10% gain
            score += 2
            reasons.append(f"Positive momentum (+{monthly_return*100:.1f}% over 30 days)")
        elif monthly_return > 0:  # Small gain
            score += 1
            reasons.append(f"Modest gains (+{monthly_return*100:.1f}% over 30 days)")
        elif monthly_return < -0.10:  # >10% loss
            score -= 3
            reasons.append(f"Sharp decline ({monthly_return*100:.1f}% over 30 days)")
        elif monthly_return < -0.05:  # 5-10% loss
            score -= 2
            reasons.append(f"Negative momentum ({monthly_return*100:.1f}% over 30 days)")
        else:  # Small loss
            score -= 1
            reasons.append(f"Recent decline ({monthly_return*100:.1f}% over 30 days)")
    
    # 2. RSI ANALYSIS (Weight: 20%)
    if not pd.isna(latest['RSI']):
        if latest['RSI'] < 30:
            score += 2
            reasons.append(f"Oversold conditions (RSI: {latest['RSI']:.1f})")
        elif latest['RSI'] > 70:
            score -= 2
            reasons.append(f"Overbought conditions (RSI: {latest['RSI']:.1f})")
        elif 45 <= latest['RSI'] <= 55:
            reasons.append(f"Neutral RSI ({latest['RSI']:.1f})")
    
    # 3. TREND ANALYSIS (Weight: 20%)
    if not pd.isna(latest['SMA_20']) and not pd.isna(latest['SMA_50']):
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            score += 2
            reasons.append("Strong uptrend (price above both MAs)")
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            score -= 2
            reasons.append("Downtrend (price below both MAs)")
        elif latest['Close'] > latest['SMA_50']:
            score += 1
            reasons.append("Above 50-day MA")
    
    # 4. MACD SIGNAL (Weight: 10%)
    if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']):
        if latest['MACD'] > latest['MACD_signal'] and latest['MACD_hist'] > 0:
            score += 1
            reasons.append("Bullish MACD crossover")
        elif latest['MACD'] < latest['MACD_signal'] and latest['MACD_hist'] < 0:
            score -= 1
            reasons.append("Bearish MACD crossover")
    
    # 5. VOLATILITY CHECK (Weight: 10%)
    volatility = latest['Volatility'] * 100 if not pd.isna(latest['Volatility']) else 0
    if volatility > 3:  # High daily volatility (>3%)
        if score > 0:
            score -= 1  # Reduce confidence in buy signals
            reasons.append(f"High volatility ({volatility:.1f}%) adds uncertainty")
    
    # 6. RISK ADJUSTMENT (Critical Override)
    # If risk score is very high, downgrade strong buy signals
    if risk_score > 75:
        if score > 2:
            score -= 1
            reasons.append("Very high risk score limits upside confidence")
        reasons.append(f"High risk profile (Risk Score: {risk_score:.0f}/100)")
    elif risk_score < 25:
        reasons.append(f"Low risk profile (Risk Score: {risk_score:.0f}/100)")
    
    # FINAL RECOMMENDATION LOGIC
    # FINAL RECOMMENDATION LOGIC WITH RISK OVERRIDE
    
    # If risk is VERY high (>75), downgrade buy signals
    if risk_score > 75:
        if score >= 4:
            return "BUY", "recommendation-buy", reasons + ["⚠️ Very high risk limits confidence"]
        elif score >= 2:
            return "HOLD", "recommendation-hold", reasons + ["⚠️ Too risky despite positive signals"]
        elif score >= -1:
            return "HOLD", "recommendation-hold", reasons
        elif score >= -3:
            return "SELL", "recommendation-sell", reasons
        else:
            return "STRONG SELL", "recommendation-sell", reasons
    
    # If risk is high (60-75), slight downgrade
    elif risk_score > 60:
        if score >= 4:
            return "BUY", "recommendation-buy", reasons + ["⚠️ High risk - consider position sizing"]
        elif score >= 2:
            return "BUY", "recommendation-buy", reasons + ["⚠️ Moderate risk level"]
        elif score >= -1:
            return "HOLD", "recommendation-hold", reasons
        elif score >= -3:
            return "SELL", "recommendation-sell", reasons
        else:
            return "STRONG SELL", "recommendation-sell", reasons
    
    # Normal risk levels - standard logic
    else:
        if score >= 4:
            return "STRONG BUY", "recommendation-buy", reasons
        elif score >= 2:
            return "BUY", "recommendation-buy", reasons
        elif score >= -1:
            return "HOLD", "recommendation-hold", reasons
        elif score >= -3:
            return "SELL", "recommendation-sell", reasons
        else:
            return "STRONG SELL", "recommendation-sell", reasons
def create_stock_chart(df, symbol):
    """Professional technical analysis chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=[
            f'{symbol} - Price & Technical Indicators',
            'Volume Analysis', 'RSI (14)', 'MACD'
        ],
        vertical_spacing=0.05
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Price", showlegend=False
        ), row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                            line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                            line=dict(color='red', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                            line=dict(color='gray', dash='dash', width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                            line=dict(color='gray', dash='dash', width=1),
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
    
    # Volume
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                        marker_color=colors, showlegend=False), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color='purple', width=2), showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, line_width=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color='blue', width=1), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal',
                            line=dict(color='red', width=1), showlegend=False), row=4, col=1)
    macd_hist_colors = ['green' if v >= 0 else 'red' for v in df['MACD_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram',
                        marker_color=macd_hist_colors, showlegend=False), row=4, col=1)
    
    fig.update_layout(height=800, showlegend=True, margin=dict(t=50, b=50, l=50, r=50))
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    return fig

# ─────────────────────────────────────────────
# PROJECT 4 — VaR & Expected Shortfall
# ─────────────────────────────────────────────

def calculate_var_es(returns, confidence=0.99, window=200):
    """Calculate VaR and ES using three methods."""
    results = {}
    portfolio_value = 100_000

    # Use last `window` returns
    r = returns.dropna().tail(window)

    with timed("VaR"):
        # 1. Historical Simulation
        threshold = np.percentile(r, (1 - confidence) * 100)
        hs_var = abs(threshold) * portfolio_value
        hs_es  = abs(r[r <= threshold].mean()) * portfolio_value if len(r[r <= threshold]) > 0 else hs_var
        results['Historical'] = {'VaR': hs_var, 'ES': hs_es}

        # 2. Variance-Covariance (Gaussian)
        from scipy import stats
        mu  = r.mean()
        sig = r.std()
        z   = stats.norm.ppf(1 - confidence)
        vc_var = abs(mu + z * sig) * portfolio_value
        phi_z  = stats.norm.pdf(-z)
        vc_es  = (phi_z / (1 - confidence)) * sig * portfolio_value
        results['Variance-Covariance'] = {'VaR': vc_var, 'ES': vc_es}

    with timed("Monte Carlo"):
        # 3. Monte Carlo (Gaussian)
        np.random.seed(42)
        sim = np.random.normal(mu, sig, 100_000)
        mc_threshold = np.percentile(sim, (1 - confidence) * 100)
        mc_var = abs(mc_threshold) * portfolio_value
        mc_es  = abs(sim[sim <= mc_threshold].mean()) * portfolio_value
        results['Monte Carlo'] = {'VaR': mc_var, 'ES': mc_es}

    return results

def run_backtest_var(returns, confidence=0.99, window=200):
    """Rolling VaR backtest — returns exception data."""
    r = returns.dropna()
    portfolio_value = 100_000
    exceptions = {'Historical': [], 'Gaussian': []}
    dates = []

    for i in range(window, len(r)):
        hist_window = r.iloc[i - window:i]
        actual = r.iloc[i] * portfolio_value

        hs_threshold = np.percentile(hist_window, (1 - confidence) * 100) * portfolio_value
        mu, sig = hist_window.mean(), hist_window.std()
        from scipy import stats
        z = stats.norm.ppf(1 - confidence)
        gc_threshold = (mu + z * sig) * portfolio_value

        dates.append(r.index[i])
        exceptions['Historical'].append(1 if actual < hs_threshold else 0)
        exceptions['Gaussian'].append(1 if actual < gc_threshold else 0)

    total = len(dates)
    if total == 0:
        return pd.Series(dtype=float), {'Historical': 0.0, 'Gaussian': 0.0}
    exc_rates = {k: sum(v) / total * 100 for k, v in exceptions.items()}
    return pd.Series(exceptions['Historical'], index=dates), exc_rates

def run_stress_test(returns, confidence=0.99):
    """Run 2008 GFC and 2020 COVID stress scenarios."""
    portfolio_value = 100_000
    scenarios = {}

    # Historical stress windows
    stress_windows = {
        '2008 GFC (Sep 08 – Mar 09)': ('2008-09-01', '2009-03-31'),
        '2020 COVID (Feb – Apr 20)':  ('2020-02-01', '2020-04-30'),
    }

    for name, (start, end) in stress_windows.items():
        try:
            window_r = returns.loc[start:end].dropna()
            if len(window_r) > 5:
                cum_loss   = window_r.sum() * portfolio_value
                peak_dd    = (window_r.cumsum().cummin().min()) * portfolio_value
                worst_day  = window_r.min() * portfolio_value
                var_1d = abs(np.percentile(returns.dropna(), 1)) * portfolio_value
                scenarios[name] = {
                    'Cumulative Loss': cum_loss,
                    'Peak Drawdown':   abs(peak_dd),
                    'Worst Day':       abs(worst_day),
                    'DD / VaR':        abs(peak_dd) / var_1d if var_1d > 0 else 0
                }
        except Exception:
            pass

    # Hypothetical scenarios
    hypo = {
        'Equity Crash (–30%)':         -0.30 * portfolio_value,
        'Stagflation (–15%)':          -0.15 * portfolio_value,
        'Correlation Breakdown (–25%)': -0.25 * portfolio_value,
    }
    for name, loss in hypo.items():
        var_1d = abs(np.percentile(returns.dropna(), 1)) * portfolio_value
        scenarios[name] = {
            'Cumulative Loss': loss,
            'Peak Drawdown':   abs(loss),
            'Worst Day':       abs(loss),
            'DD / VaR':        abs(loss) / var_1d if var_1d > 0 else 0
        }
    return scenarios


# ─────────────────────────────────────────────
# PROJECT 1 — Black-Scholes Options Pricer
# ─────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes price for European call or put."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate all 5 Greeks."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi = norm.pdf(d1)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-(S * phi * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * phi * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = phi / (S * sigma * np.sqrt(T))
    vega  = S * phi * np.sqrt(T) / 100
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def implied_vol_newton(market_price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
    """Newton-Raphson implied volatility solver."""
    from scipy.stats import norm
    if T <= 0:
        return None
    # No-arbitrage bounds check
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == 'call' else max(K * np.exp(-r * T) - S, 0)
    if market_price < intrinsic:
        return None
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if abs(vega) < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.001
    return sigma if 0 < sigma < 5 else None

def build_vol_smile(df, K_range=0.15, n_strikes=20):
    """Build implied vol smile from historical price data."""
    S = df['Close'].iloc[-1]
    r = 0.045
    T = 30 / 365
    annual_vol = df['Returns'].dropna().std() * np.sqrt(252)

    strikes = np.linspace(S * (1 - K_range), S * (1 + K_range), n_strikes)
    smile_data = []
    for K in strikes:
        theo_price = bs_price(S, K, T, r, annual_vol, 'call')
        noise = np.random.normal(0, theo_price * 0.02)
        market_price = max(theo_price + noise, 0.01)
        iv = implied_vol_newton(market_price, S, K, T, r, 'call')
        if iv and 0.05 < iv < 2.0:
            smile_data.append({'Strike': K, 'Moneyness': K / S, 'IV': iv * 100})
    return pd.DataFrame(smile_data)


# ─────────────────────────────────────────────
# PROJECT 3 — Factor Model & Momentum
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_factor_proxies(period="2y"):
    """
    Fetch independent factor proxy ETFs from Yahoo Finance.
    These are INDEPENDENT of the stock being analysed — that's the key fix.
      MKT  → SPY  (broad market)
      SMB  → IWM - SPY  (small minus large)
      HML  → IVE - IVW  (value minus growth)
      MOM  → MTUM       (momentum ETF)
    """
    # @st.cache_data short-circuits on a hit before this body ever runs, so
    # reaching this line means it was a miss.
    _record_cache_call("fetch_factor_proxies", hit=False)
    tickers = ['SPY', 'IWM', 'IVE', 'IVW', 'MTUM']
    try:
        with timed("Network fetch time"):
            raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Close']
        factors = pd.DataFrame(index=raw.index)
        factors['MKT']  = raw['SPY'].pct_change()
        factors['SMB']  = raw['IWM'].pct_change() - raw['SPY'].pct_change()
        factors['HML']  = raw['IVE'].pct_change() - raw['IVW'].pct_change()
        factors['MOM']  = raw['MTUM'].pct_change()
        factors['RF']   = 0.045 / 252
        return factors.dropna()
    except Exception:
        return None

def calculate_factor_exposures(df, period="2y"):
    """OLS regression of stock excess returns on 4 independent factors."""
    returns = df['Returns'].dropna()
    if len(returns) < 60:
        return None

    # Cache hits skip fetch_factor_proxies' body entirely, so a hit/miss can
    # only be inferred by diffing its miss counter around the call.
    _misses_before = _CACHE_STATUS.get("fetch_factor_proxies", {}).get("misses", 0)
    factors = fetch_factor_proxies(period)
    _misses_after = _CACHE_STATUS.get("fetch_factor_proxies", {}).get("misses", 0)
    if _misses_after == _misses_before:
        _record_cache_call("fetch_factor_proxies", hit=True)

    if factors is None or len(factors) < 60:
        return None

    # Align on common dates
    aligned = pd.DataFrame({'stock': returns}).join(factors, how='inner').dropna()
    if len(aligned) < 60:
        return None

    y = (aligned['stock'] - aligned['RF']).values
    X = np.column_stack([
        np.ones(len(aligned)),
        aligned['MKT'].values,
        aligned['SMB'].values,
        aligned['HML'].values,
        aligned['MOM'].values,
    ])

    from numpy.linalg import lstsq
    coeffs, _, _, _ = lstsq(X, y, rcond=None)
    alpha, beta, smb, hml, mom = coeffs

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'alpha': alpha * 252 * 100,
        'beta':  beta,
        'smb':   smb,
        'hml':   hml,
        'mom':   mom,
        'r2':    r2,
        'n_obs': len(aligned),
    }

def calculate_momentum_signal(df):
    """12-1 month momentum signal (properly lagged)."""
    if len(df) < 252:
        lookback = max(len(df) - 21, 20)
        signal = (df['Close'].iloc[-1] / df['Close'].iloc[-lookback]) - 1
    else:
        signal = (df['Close'].iloc[-22] / df['Close'].iloc[-252]) - 1
    return signal * 100  # percent

def momentum_backtest(df, rebalance_days=21):
    """Simple long/short momentum strategy on the stock vs its own history."""
    returns = df['Returns'].dropna()
    if len(returns) < 120:
        return None

    portfolio_returns = []
    dates = []
    for i in range(252, len(returns), rebalance_days):
        signal = (df['Close'].iloc[i - 22] / df['Close'].iloc[i - 252]) - 1
        period_r = returns.iloc[i:i + rebalance_days].mean() * rebalance_days
        position = 1 if signal > 0 else -1
        portfolio_returns.append(position * period_r)
        dates.append(returns.index[i])

    if not portfolio_returns:
        return None

    port_series = pd.Series(portfolio_returns, index=dates)
    cum_returns = (1 + port_series).cumprod()
    ann_return  = port_series.mean() * (252 / rebalance_days) * 100
    ann_vol     = port_series.std() * np.sqrt(252 / rebalance_days) * 100
    sharpe      = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd      = ((cum_returns / cum_returns.cummax()) - 1).min() * 100

    return {
        'cum_returns': cum_returns,
        'ann_return':  ann_return,
        'ann_vol':     ann_vol,
        'sharpe':      sharpe,
        'max_drawdown': max_dd,
        'hit_rate':    (port_series > 0).mean() * 100,
    }


def render_ticker_tape(selected_stocks, stock_data):
    """Continuous CSS marquee built from already-fetched stock_data - no extra network calls.
    Pauses on hover (also satisfies WCAG 2.2.2 for moving content)."""
    items_html = ""
    for stock in selected_stocks:
        sdf = stock_data[stock]
        if len(sdf) == 0:
            continue
        price = sdf['Close'].iloc[-1]
        change = sdf['Returns'].iloc[-1] * 100 if not pd.isna(sdf['Returns'].iloc[-1]) else 0
        color = "green" if change >= 0 else "red"
        arrow = "↗️" if change >= 0 else "↘️"
        items_html += (
            f'<span class="ticker-item">'
            f'<span class="ticker-symbol">{stock}</span> ${price:.2f} '
            f'<span style="color: {color};">{arrow} {change:+.2f}%</span>'
            f'</span>'
        )
    if not items_html:
        return
    track = items_html * 2  # duplicated for a seamless -50% loop
    st.markdown(f'<div class="ticker-wrap"><div class="ticker-track">{track}</div></div>', unsafe_allow_html=True)


def render_live_status(use_real_data, last_market_date=None, current_time=None):
    """Status badge whose dot encodes a real state (live feed vs simulated data), not decoration."""
    is_live = use_real_data == "Real-Time (Yahoo Finance)"
    dot_class = "" if is_live else "is-static"
    if is_live and last_market_date is not None and current_time is not None:
        text = (f"LIVE &middot; Market data as of {last_market_date.strftime('%B %d, %Y')} "
                f"&middot; Last fetched {current_time.strftime('%I:%M:%S %p')}")
    elif is_live:
        text = "LIVE &middot; Yahoo Finance"
    else:
        text = "SIMULATED &middot; Statistical model data"
    st.markdown(f'<div class="live-status"><span class="live-dot {dot_class}"></span>{text}</div>',
                unsafe_allow_html=True)


def render_skeleton_cards(n):
    """Loading placeholder shaped like the KPI cards it precedes, shown only while data is actually fetching."""
    n = max(n, 1)
    cols = st.columns(n)
    for c in cols:
        with c:
            st.markdown(
                '<div class="skeleton-card">'
                '<div class="skeleton-line" style="width:40%"></div>'
                '<div class="skeleton-line" style="width:70%"></div>'
                '<div class="skeleton-line" style="width:50%"></div>'
                '</div>',
                unsafe_allow_html=True
            )


def render_kpi_price_cards(selected_stocks, stock_data):
    """Animated stock-price KPI tiles. st.markdown(unsafe_allow_html=True) never executes <script>
    tags (browsers don't run scripts inserted via innerHTML), so the JS-driven count-up lives inside
    components.html, which renders in a real iframe document. Counts up from 0 on every render -
    that's intentional: it fires exactly when data actually refreshes (selection change, auto-refresh,
    manual refresh), so the motion signals a real update rather than running on a timer for its own sake."""
    cards_html = ""
    count = 0
    for stock in selected_stocks:
        sdf = stock_data[stock]
        if len(sdf) == 0:
            continue
        count += 1
        price = float(sdf['Close'].iloc[-1])
        change = float(sdf['Returns'].iloc[-1] * 100) if not pd.isna(sdf['Returns'].iloc[-1]) else 0.0
        direction = "up" if change >= 0 else "down"
        arrow = "↗️" if change >= 0 else "↘️"
        sign_prefix = "+" if change >= 0 else ""
        cards_html += f"""
        <div class="kpi-card">
          <div class="kpi-symbol">{stock}</div>
          <div class="kpi-price" data-countup data-end="{price:.2f}" data-prefix="$" data-decimals="2">$0.00</div>
          <div class="kpi-change kpi-{direction}">{arrow}
            <span data-countup data-end="{change:.2f}" data-prefix="{sign_prefix}" data-decimals="2">0.00</span>%
          </div>
        </div>
        """

    if count == 0:
        return

    cols_per_row = 4
    rows = math.ceil(count / cols_per_row)
    height = 40 + rows * 132

    html = f"""
    <html><head><style>
      html, body {{ margin:0; padding:0; background: transparent; font-family: "Source Sans Pro", sans-serif; }}
      .kpi-grid {{ display:grid; grid-template-columns: repeat({cols_per_row}, 1fr); gap: 0.75rem; padding: 4px; }}
      @media (max-width: 700px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
      .kpi-card {{
        background:#262730; border-radius:10px; padding:1rem; text-align:center; box-sizing:border-box;
        color:#FAFAFA; opacity:0; animation: fadeInUp 420ms cubic-bezier(.23,1,.32,1) forwards;
        transition: transform 180ms cubic-bezier(.23,1,.32,1), box-shadow 180ms cubic-bezier(.23,1,.32,1);
      }}
      .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 2px 8px rgba(0,0,0,0.4); }}
      .kpi-symbol {{ font-size:0.95rem; font-weight:700; color:#FAFAFA; margin-bottom:0.25rem; }}
      .kpi-price {{ font-size:1.6rem; font-weight:700; color:#FAFAFA; margin:0.1rem 0; }}
      .kpi-change {{ font-size:0.95rem; }}
      .kpi-up {{ color: green; }}
      .kpi-down {{ color: red; }}
      @keyframes fadeInUp {{ from {{ opacity:0; transform: translateY(8px); }} to {{ opacity:1; transform: translateY(0); }} }}
      @media (prefers-reduced-motion: reduce) {{
        .kpi-card {{ animation: none !important; opacity:1 !important; transition: none !important; }}
      }}
    </style></head>
    <body>
      <div class="kpi-grid">{cards_html}</div>
      <script>
        var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        function ease(t) {{ return 1 - Math.pow(1 - t, 3); }}
        document.querySelectorAll('[data-countup]').forEach(function(el) {{
          var end = parseFloat(el.getAttribute('data-end'));
          var prefix = el.getAttribute('data-prefix') || '';
          var decimals = parseInt(el.getAttribute('data-decimals') || '2', 10);
          if (isNaN(end)) {{ return; }}
          if (reduceMotion) {{ el.textContent = prefix + end.toFixed(decimals); return; }}
          var duration = 700;
          var startTime = null;
          function step(ts) {{
            if (!startTime) startTime = ts;
            var t = Math.min((ts - startTime) / duration, 1);
            var value = end * ease(t);
            el.textContent = prefix + value.toFixed(decimals);
            if (t < 1) requestAnimationFrame(step);
          }}
          requestAnimationFrame(step);
        }});
      </script>
    </body></html>
    """
    components.html(html, height=height, scrolling=False)


def main():
    # Header with personal branding
    st.markdown('<h1 class="main-header">📊 Stock Market Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Built by Dev Golakiya | UMass Amherst Business Analytics</p>', unsafe_allow_html=True)

    
    # Sidebar
    st.sidebar.header("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📈 Stock Selection")
    available_stocks = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC']
    
    custom_stock = st.sidebar.text_input("Add Custom Ticker:", placeholder="e.g., IBM").upper().strip()
    if custom_stock and custom_stock not in available_stocks:
        available_stocks.insert(0, custom_stock)
    
    selected_stocks = st.sidebar.multiselect(
        "Select stocks to analyze:",
        options=available_stocks,
        default=['AAPL', 'MSFT', 'NVDA', 'TSLA']
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Analysis Settings")
    
    data_period = st.sidebar.selectbox(
        "Historical Data:",
        options=[252, 504, 756, 1260],
        format_func=lambda x: f"{x//252} Year{'s' if x > 252 else ''}",
        index=1
    )
    
    forecast_days = st.sidebar.slider("Forecast Period:", 1, 60, 30)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Data Source")
    
    use_real_data = st.sidebar.radio(
        "Select data source:",
        options=["Real-Time (Yahoo Finance)", "Synthetic (Simulated)"],
        index=0,
        help="Real-time data is fetched from Yahoo Finance. Synthetic data is generated for demonstration."
    )
    
    # Add refresh button
    if use_real_data == "Real-Time (Yahoo Finance)":
        st.sidebar.success("✅ Using live market data")
        st.sidebar.caption("⚡ Data refreshes automatically")
        
        # Initialize session state for auto-refresh
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = datetime.now()
        if 'refresh_counter' not in st.session_state:
            st.session_state.refresh_counter = 0
        
        # Calculate time since last refresh
        time_since_refresh = (datetime.now() - st.session_state.last_refresh_time).total_seconds()
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox(
            "🔄 Auto-refresh (30s)", 
            value=False,
            help="Automatically reload data every 30 seconds",
            key='auto_refresh_toggle'
        )
        
        # Show countdown and handle auto-refresh
        if auto_refresh:
            time_remaining = max(0, 30 - int(time_since_refresh))
            st.sidebar.info(f"⏱️ Next refresh in: {time_remaining}s")
            
            # Only rerun when timer reaches 0
            if time_since_refresh >= 30:
                st.session_state.last_refresh_time = datetime.now()
                st.session_state.refresh_counter += 1
                # Clear only the live-price cache so the hourly factor-proxy
                # cache and other unrelated cached data aren't force-refetched.
                fetch_real_stock_data.clear()
                st.rerun()

        # Manual refresh button
        if st.sidebar.button("🔄 Force Refresh Now", help="Fetch latest data immediately"):
            st.session_state.last_refresh_time = datetime.now()
            st.session_state.refresh_counter += 1
            # Clear only the live-price cache so the hourly factor-proxy
            # cache and other unrelated cached data aren't force-refetched.
            fetch_real_stock_data.clear()
            st.rerun()
        
        # Show refresh stats
        st.sidebar.caption(f"🔄 Refreshes: {st.session_state.refresh_counter} | Last: {st.session_state.last_refresh_time.strftime('%I:%M:%S %p')}")
    else:
        st.sidebar.info("📊 Using simulated data")
        st.sidebar.caption("Based on realistic statistical patterns")
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **About**: This dashboard uses proprietary risk scoring and sector analysis methodologies developed during my MS in Business Analytics at UMass Amherst.")

    # Fully gated on DEBUG_PERFORMANCE: when off, this checkbox/expander
    # doesn't render at all, so the flag is a genuine kill switch.
    if DEBUG_PERFORMANCE:
        st.sidebar.markdown("---")
        show_perf = st.sidebar.checkbox("⏱️ Show performance breakdown", value=False)
        if show_perf:
            with st.sidebar.expander("⏱️ Performance", expanded=False):
                st.caption("Shows the previous completed run — not this one.")
                last_timings = st.session_state.get('last_timings', {})
                last_cache_status = st.session_state.get('last_cache_status', {})
                if last_timings:
                    timing_rows = sorted(last_timings.items(), key=lambda kv: kv[1], reverse=True)
                    st.dataframe(
                        pd.DataFrame(timing_rows, columns=['Section', 'Seconds']),
                        use_container_width=True, hide_index=True
                    )
                if last_cache_status:
                    cache_rows = [
                        {'Function': k, 'Hits': v['hits'], 'Misses': v['misses']}
                        for k, v in last_cache_status.items()
                    ]
                    st.dataframe(pd.DataFrame(cache_rows), use_container_width=True, hide_index=True)
                if not last_timings and not last_cache_status:
                    st.caption("No completed run yet.")

    if not selected_stocks:
        st.warning("⚠️ Please select at least one stock from the sidebar!")
        return
    
    # Load data based on user selection. stock_data is the single source of
    # truth for every tab below — fetch_real_stock_data is cached per
    # (symbol, period), so each ticker/timeframe hits Yahoo Finance at most
    # once per cache window. Tabs must read from stock_data, not fetch again.
    stock_data = {}

    # Skeleton placeholder shaped like the KPI cards below - shown only for the
    # span of the actual fetch (cache hits skip it almost instantly).
    skeleton_placeholder = st.empty()
    with skeleton_placeholder.container():
        render_skeleton_cards(min(len(selected_stocks), 4))

    if use_real_data == "Real-Time (Yahoo Finance)":
        # Map data_period (days) to yfinance period strings
        period_mapping = {
            252: "1y",
            504: "2y",
            756: "3y",
            1260: "5y"
        }
        yf_period = period_mapping.get(data_period, "1y")

        with st.spinner("📡 Fetching real-time data from Yahoo Finance..."):
            for stock in selected_stocks:
                # Cache hits skip fetch_real_stock_data's body entirely, so a
                # hit/miss can only be inferred by diffing its miss counter.
                _misses_before = _CACHE_STATUS.get("fetch_real_stock_data", {}).get("misses", 0)
                stock_data[stock] = fetch_real_stock_data(stock, period=yf_period)
                _misses_after = _CACHE_STATUS.get("fetch_real_stock_data", {}).get("misses", 0)
                if _misses_after == _misses_before:
                    _record_cache_call("fetch_real_stock_data", hit=True)
    else:
        with st.spinner("📊 Generating synthetic data..."):
            for stock in selected_stocks:
                stock_data[stock] = generate_stock_data(stock, data_period)

    skeleton_placeholder.empty()

    # Live ticker tape - built from the stock_data already fetched above, no extra network calls.
    render_ticker_tape(selected_stocks, stock_data)

    # Live status badge - replaces the old plain caption with one that also
    # carries the live-vs-simulated semantic state via the dot.
    last_market_date, current_time = None, datetime.now()
    if use_real_data == "Real-Time (Yahoo Finance)" and stock_data:
        sample_stock = list(stock_data.keys())[0]
        sample_df = stock_data[sample_stock]
        if len(sample_df) > 0:
            last_market_date = sample_df.index[-1]
    render_live_status(use_real_data, last_market_date, current_time)

    # Show data source indicator
    st.markdown("---")
    if use_real_data == "Real-Time (Yahoo Finance)":
        st.info("📡 **Live Data Mode** - Real-time market data from Yahoo Finance | ⚡ Updates every 30 seconds")
    else:
        st.info("📊 **Simulation Mode** - Synthetic data based on statistical modeling")
    
    detail_stock = st.selectbox("🔍 Select stock for detailed analysis:", selected_stocks, index=0)
        


    df = stock_data[detail_stock]
    current_price = df['Close'].iloc[-1]
    
    # Calculate risk score
    risk_score, risk_category = calculate_dev_risk_score(df)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview", "🎯 Risk Analysis", "📈 Technical", "🔮 Forecast",
        "📉 Performance", "💼 Portfolio",
        "⚠️ VaR & Stress Test", "📐 Options Pricer", "🔬 Factor Model"
    ])
    
    with tab1:
        with timed("Overview rendering"):
            st.header("📊 Market Overview")
        
            # Stock price cards - animated KPI tiles (count-up on refresh, hover lift)
            render_kpi_price_cards(selected_stocks, stock_data)

            st.markdown("---")
        
            # Key metrics for selected stock
            st.subheader(f"📈 {detail_stock} - Key Metrics")
            latest = df.iloc[-1]
        
            col1, col2, col3, col4, col5 = st.columns(5)
        
            col1.metric("💵 Price", f"${latest['Close']:.2f}")
        
            daily_ret = latest['Returns'] * 100 if not pd.isna(latest['Returns']) else 0
            col2.metric("📈 Daily Change", f"{daily_ret:.2f}%", delta=f"{daily_ret:.2f}%")
        
            vol = latest['Volatility'] * 100 if not pd.isna(latest['Volatility']) else 0
            col3.metric("📊 Volatility", f"{vol:.2f}%")
        
            volume_str = f"{latest['Volume']/1e6:.1f}M" if latest['Volume'] > 1e6 else f"{latest['Volume']/1e3:.1f}K"
            col4.metric("📦 Volume", volume_str)
        
            rsi = latest['RSI'] if not pd.isna(latest['RSI']) else 50
            col5.metric("🎯 RSI", f"{rsi:.1f}")
        
            st.markdown("---")
        
            # Sector Analysis
            st.subheader("🏢 Sector Performance Analysis")
            sector_df = analyze_sectors(stock_data)
        
            col1, col2 = st.columns([2, 1])
        
            with col1:
                fig_sector = px.bar(
                    sector_df, x='Sector', y='Avg Return (20d)',
                    color='Avg Return (20d)',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="Sector Momentum (20-Day Returns)"
                )
                fig_sector.update_layout(height=350)
                st.plotly_chart(fig_sector, use_container_width=True)
        
            with col2:
                st.dataframe(sector_df, use_container_width=True, height=350)
    
    with tab2:
        with timed("Risk Analysis rendering"):
            st.header(f"🎯 Risk Analysis - {detail_stock}")
        
            # Risk Score Display
            risk_class = "risk-score-low" if risk_score < 35 else "risk-score-medium" if risk_score < 60 else "risk-score-high"
        
            col1, col2 = st.columns([2, 1])
        
            with col1:
                st.markdown(f"""
                <div class="{risk_class}">
                    <h2>📊 Dev's Risk Score</h2>
                    <h1>{risk_score}/100</h1>
                    <h3>Risk Level: {risk_category}</h3>
                    <p style="font-size: 0.9rem; margin-top: 1rem;">
                    Multi-factor analysis: Volatility (30%), Momentum (25%), Liquidity (20%), Technical (15%), Drawdown (10%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
                # Insights
                st.markdown("### 💡 Key Insights")
                insights = generate_insights(df, risk_score)
        
                if insights:
                    for insight in insights:
                        st.info(insight)
                else:
                    st.info("No specific insights available for this stock at this time.")
        
            with col2:
                st.markdown("### 📈 Risk Components")
            
                # Component breakdown
                vol = df['Returns'].std() * np.sqrt(252)
                vol_score = min(vol / 0.50, 1.0) * 30
            
                recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-20]) - 1
                mom_vol = df['Returns'].tail(20).std()
                mom_score = min((abs(recent_return) * 0.5 + mom_vol * 10) * 25, 25)
            
                volume_cv = df['Volume'].std() / df['Volume'].mean()
                liq_score = min(volume_cv, 1.0) * 20
            
                rsi_val = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
                tech_score = (abs(rsi_val - 50) / 50) * 15
            
                returns = df['Returns'].fillna(0)
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                dd_score = min(abs(drawdown.min()) * 2, 1.0) * 10
            
                components = pd.DataFrame({
                    'Component': ['Volatility', 'Momentum', 'Liquidity', 'Technical', 'Drawdown'],
                    'Score': [vol_score, mom_score, liq_score, tech_score, dd_score],
                    'Weight': ['30%', '25%', '20%', '15%', '10%']
                })
            
                st.dataframe(components, use_container_width=True)
            
                fig_comp = px.bar(components, x='Component', y='Score', 
                                 text='Weight', title="Risk Score Breakdown")
                fig_comp.update_traces(textposition='outside')
                st.plotly_chart(fig_comp, use_container_width=True)
    
    with tab3:
        with timed("Technical Analysis rendering"):
            st.header(f"📈 Technical Analysis - {detail_stock}")
            fig = create_stock_chart(df, detail_stock)
            st.plotly_chart(fig, use_container_width=True)
        
            # Technical Summary
            st.subheader("📊 Technical Indicators Summary")
        
            col1, col2, col3, col4 = st.columns(4)
        
            latest = df.iloc[-1]
        
            with col1:
                rsi_val = latest['RSI'] if not pd.isna(latest['RSI']) else 50
                rsi_signal = "🟢 Oversold" if rsi_val < 30 else "🔴 Overbought" if rsi_val > 70 else "🟡 Neutral"
                st.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_signal)
        
            with col2:
                macd_signal = "🟢 Bullish" if latest['MACD'] > latest['MACD_signal'] else "🔴 Bearish"
                st.metric("MACD", f"{latest['MACD']:.2f}", delta=macd_signal)
        
            with col3:
                price_to_bb = ((latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])) * 100
                bb_position = "Upper Band" if price_to_bb > 80 else "Lower Band" if price_to_bb < 20 else "Middle"
                st.metric("BB Position", f"{price_to_bb:.0f}%", delta=bb_position)
        
            with col4:
                trend = "🟢 Uptrend" if latest['Close'] > latest['SMA_50'] else "🔴 Downtrend"
                st.metric("50-Day Trend", trend)
    
    with tab4:
        with timed("Forecast rendering"):
            st.header(f"🔮 Price Forecast - {detail_stock}")
        
            predictions = generate_predictions(detail_stock, current_price, forecast_days)
            predicted_price = predictions[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100
        
            # Forecast summary
            col1, col2, col3 = st.columns(3)
        
            col1.metric("Price", f"${current_price:.2f}")
            col2.metric(f"{forecast_days}-Day Target", f"${predicted_price:.2f}")
            col3.metric("Expected Change", f"{price_change:+.2f}%", delta=f"{price_change:+.2f}%")
        
            # Forecast chart
            historical = df.tail(60)
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
            fig_forecast = go.Figure()
        
            # Historical
            fig_forecast.add_trace(go.Scatter(
                x=historical.index, y=historical['Close'],
                name='Historical', line=dict(color='blue', width=2)
            ))
        
            # Prediction
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=predictions,
                name='Forecast', line=dict(color='red', dash='dash', width=3)
            ))
        
            # Confidence interval
            confidence = 0.15
            upper = [p * (1 + confidence) for p in predictions]
            lower = [p * (1 - confidence) for p in predictions]
        
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=upper, fill=None, mode='lines',
                line_color='rgba(0,100,80,0)', showlegend=False
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_dates, y=lower, fill='tonexty', mode='lines',
                line_color='rgba(0,100,80,0)', name='85% Confidence',
                fillcolor='rgba(0,100,80,0.2)'
            ))
        
            fig_forecast.add_trace(go.Scatter(
                x=[last_date], y=[current_price], mode='markers',
                marker=dict(size=12, color='orange'), name='Current'
            ))
        
            fig_forecast.update_layout(
                title=f"{detail_stock} - {forecast_days} Day Price Forecast",
                xaxis_title="Date", yaxis_title="Price ($)", height=500
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        
            # Forecast methodology
            with st.expander("📖 Forecast Methodology"):
                st.markdown("""
                **Prediction Model:**
                - Combines trend analysis with mean reversion principles
                - Incorporates historical volatility patterns
                - Applies 85% confidence intervals based on price variance
                - Uses sector momentum for trend adjustment
            
                **Note:** Forecasts are for educational purposes. Past performance does not guarantee future results.
                """)
    
    with tab5:
        with timed("Performance rendering"):
            st.header(f"📉 Performance Analysis - {detail_stock}")
        
            perf = calculate_performance_metrics(df)
        
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
        
            total_ret = perf['total_return'] * 100
            col1.metric("Total Return", f"{total_ret:.2f}%", delta=f"{total_ret:.2f}%")
        
            vol_ann = perf['volatility'] * 100
            col2.metric("Annual Volatility", f"{vol_ann:.2f}%")
        
            col3.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
        
            col4.metric("Sortino Ratio", f"{perf['sortino_ratio']:.2f}")
        
            max_dd = abs(perf['max_drawdown']) * 100
            col5.metric("Max Drawdown", f"-{max_dd:.2f}%")
        
            st.markdown("---")
        
            # Charts
            col1, col2 = st.columns(2)
        
            with col1:
                st.subheader("📈 Cumulative Returns")
                returns = df['Returns'].fillna(0)
                cumulative = (1 + returns).cumprod()
            
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=df.index, y=cumulative,
                    name='Cumulative Returns',
                    line=dict(color='green', width=2),
                    fill='tonexty', fillcolor='rgba(0,255,0,0.1)'
                ))
                fig_cum.update_layout(
                    title=f"{detail_stock} Cumulative Performance",
                    xaxis_title="Date", yaxis_title="Cumulative Return", height=400
                )
                st.plotly_chart(fig_cum, use_container_width=True)
        
            with col2:
                st.subheader("📊 Returns Distribution")
                returns_clean = df['Returns'].dropna()
            
                fig_dist = go.Figure(data=[go.Histogram(
                    x=returns_clean, nbinsx=50,
                    marker_color='steelblue', opacity=0.7
                )])
                fig_dist.update_layout(
                    title="Daily Returns Distribution",
                    xaxis_title="Daily Return", yaxis_title="Frequency", height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
            # Monthly returns heatmap
            st.subheader("📅 Monthly Returns Heatmap")
            monthly_returns = df['Returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 100
            })
        
            if len(monthly_df) > 0:
                pivot = monthly_df.pivot(index='Month', columns='Year', values='Return')
            
                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot)],
                    colorscale='RdYlGn',
                    zmid=0,
                    text=np.round(pivot.values, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 10}
                ))
                fig_heat.update_layout(
                    title="Monthly Returns (%)", height=400,
                    xaxis_title="Year", yaxis_title="Month"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
    
    with tab6:
        with timed("Portfolio rendering"):
            st.header("💼 Portfolio Analysis")
        
            st.subheader("📊 Portfolio Holdings")
        
            # Portfolio construction
            portfolio_data = []
            total_value = 0

            with timed("Portfolio calculations"):
                for stock in selected_stocks:
                    sdf = stock_data[stock]
                    price = sdf['Close'].iloc[-1]
                    shares = 100
                    value = price * shares
                    total_value += value

                    # FIXED: Calculate expected return based on actual momentum
                    if len(sdf) >= 30:
                        # Use actual 30-day momentum (more realistic)
                        momentum_30d = ((price - sdf['Close'].iloc[-30]) / sdf['Close'].iloc[-30]) * 100

                        # Get prediction for validation
                        predictions = generate_predictions(stock, price, 30)
                        predicted_return = ((predictions[-1] - price) / price) * 100 if predictions else 0

                        # Blend: 70% actual momentum, 30% prediction
                        expected_ret = (momentum_30d * 0.7) + (predicted_return * 0.3)
                    else:
                        # Fallback to prediction only for insufficient data
                        predictions = generate_predictions(stock, price, 30)
                        expected_ret = ((predictions[-1] - price) / price) * 100 if predictions else 0

                    perf = calculate_performance_metrics(sdf)
                    risk_score_stock, _ = calculate_dev_risk_score(sdf)

                    portfolio_data.append({
                        'Stock': stock,
                        'Sector': STOCK_SECTORS.get(stock, 'Other'),
                        'Shares': shares,
                        'Price': price,
                        'Value': value,
                        'Weight': 0,
                        'Expected Return': expected_ret,
                        'Risk Score': risk_score_stock,
                        'Sharpe Ratio': perf['sharpe_ratio']
                    })

                # Calculate weights
                for item in portfolio_data:
                    item['Weight'] = (item['Value'] / total_value) * 100 if total_value > 0 else 0

            # Create dataframe from portfolio data
            portfolio_df = pd.DataFrame(portfolio_data)
        
            # Format for display
            display_df = pd.DataFrame({
                'Stock': portfolio_df['Stock'],
                'Sector': portfolio_df['Sector'],
                'Shares': portfolio_df['Shares'],
                'Price': portfolio_df['Price'].apply(lambda x: f"${x:.2f}"),
                'Value': portfolio_df['Value'].apply(lambda x: f"${x:,.2f}"),
                'Weight': portfolio_df['Weight'].apply(lambda x: f"{x:.1f}%"),
                'Expected Return': portfolio_df['Expected Return'].apply(lambda x: f"{x:+.2f}%"),
                'Risk Score': portfolio_df['Risk Score'].apply(lambda x: f"{x:.1f}"),
                'Sharpe Ratio': portfolio_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
            })

            st.dataframe(display_df, use_container_width=True)

            st.markdown("---")
        
        
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
        
            col1.metric("Total Value", f"${total_value:,.2f}")
        
            weighted_return = sum(p['Expected Return'] * p['Weight'] / 100 for p in portfolio_data)
            col2.metric("Expected Return", f"{weighted_return:+.2f}%")
        
            avg_risk = np.mean([p['Risk Score'] for p in portfolio_data])
            col3.metric("Avg Risk Score", f"{avg_risk:.1f}")
        
            avg_sharpe = np.mean([p['Sharpe Ratio'] for p in portfolio_data])
            col4.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
        
            # Visualizations
            col1, col2 = st.columns(2)
        
            with col1:
                st.subheader("🥧 Portfolio Allocation")
                fig_pie = px.pie(
                    portfolio_df, values='Weight', names='Stock',
                    title="Portfolio Weight Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
            with col2:
                st.subheader("📊 Sector Allocation")
                sector_alloc = portfolio_df.groupby('Sector')['Weight'].sum().reset_index()
                fig_sector_pie = px.pie(
                    sector_alloc, values='Weight', names='Sector',
                    title="Sector Diversification",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_sector_pie, use_container_width=True)
        
            st.markdown("---")
        
            # Investment recommendations
            st.subheader("💡 Investment Recommendations")
        
            for stock in selected_stocks:
                sdf = stock_data[stock]
                with timed("Portfolio calculations"):
                    risk_score_stock, risk_cat = calculate_dev_risk_score(sdf)
                    recommendation, rec_class, reasons = generate_recommendation(sdf, risk_score_stock)

                # Determine card style
                if "buy" in rec_class:
                    card_style = "recommendation-buy"
                elif "sell" in rec_class:
                    card_style = "recommendation-sell"
                else:
                    card_style = "recommendation-hold"
            
                reasons_text = " • ".join(reasons[:3]) if reasons else "Based on technical analysis"
            
                st.markdown(f"""
                <div class="{card_style}">
                    <h4>{stock}: {recommendation} (Risk: {risk_cat})</h4>
                    <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">{reasons_text}</p>
                </div>
                """, unsafe_allow_html=True)
        
            # Portfolio insights
            st.markdown("---")
            st.subheader("🎯 Portfolio Insights")
        
            # Correlation analysis
            returns_matrix = pd.DataFrame({
                stock: stock_data[stock]['Returns'] for stock in selected_stocks
            })
            corr_matrix = returns_matrix.corr()
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
            diversification_score = (1 - avg_corr) * 100
        
            col1, col2 = st.columns(2)
        
            with col1:
                div_emoji = "🟢" if diversification_score > 60 else "🟡" if diversification_score > 40 else "🔴"
            
                st.metric(
                    label=f"{div_emoji} Diversification Score",
                    value=f"{diversification_score:.1f}/100"
                )
                st.write(f"**Average correlation:** {avg_corr:.2f}")
            
                if diversification_score > 50:
                    st.success("✓ Good diversification across holdings")
                else:
                    st.warning("⚠️ Consider adding uncorrelated assets")
        
            with col2:
                risk_emoji = "🟢" if 30 < avg_risk < 60 else "🔴" if avg_risk > 60 else "🟡"
                risk_label = "Balanced" if 30 < avg_risk < 60 else "High Risk" if avg_risk > 60 else "Conservative"
            
                st.metric(
                    label=f"{risk_emoji} Portfolio Risk Profile",
                    value=f"{avg_risk:.1f}/100"
                )
                st.write(f"**Risk Level:** {risk_label}")
            
                if avg_risk > 60:
                    st.info("📊 Suitable for: Growth investors")
                elif avg_risk > 30:
                    st.info("📊 Suitable for: Moderate investors")
                else:
                    st.info("📊 Suitable for: Conservative investors")
        
            # Correlation heatmap
            st.subheader("📊 Stock Correlation Matrix")
            corr_matrix = returns_matrix.corr()
        
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlGn_r',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Correlation")
            ))
            fig_corr.update_layout(
                title="Correlation Between Holdings",
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── TAB 7 — VaR & Stress Test ──────────────────────────────────────────
    with tab7:
        with timed("VaR & Stress Test rendering"):
            st.header(f"⚠️ VaR & Stress Test — {detail_stock}")
            st.caption("Methodology from Project 4 — Historical Simulation, Variance-Covariance, Monte Carlo | Kupiec & Christoffersen backtests | 2008 GFC & 2020 COVID stress replays")

            returns_var = df['Returns'].dropna()

            # ── VaR / ES comparison ──
            st.subheader("📊 VaR & Expected Shortfall (99%, 1-day, $100K portfolio)")
            var_results = calculate_var_es(returns_var)

            col1, col2, col3 = st.columns(3)
            colors = {'Historical': '#1f77b4', 'Variance-Covariance': '#ff7f0e', 'Monte Carlo': '#2ca02c'}
            for col, (method, vals) in zip([col1, col2, col3], var_results.items()):
                with col:
                    st.markdown(f"**{method}**")
                    st.metric("VaR ($)", f"${vals['VaR']:,.0f}")
                    st.metric("ES ($)",  f"${vals['ES']:,.0f}")
                    st.caption(f"ES/VaR ratio: {vals['ES']/vals['VaR']:.2f}")

            # Bar chart comparison
            methods = list(var_results.keys())
            var_vals = [var_results[m]['VaR'] for m in methods]
            es_vals  = [var_results[m]['ES']  for m in methods]

            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(name='VaR', x=methods, y=var_vals, marker_color='#1f77b4'))
            fig_var.add_trace(go.Bar(name='ES',  x=methods, y=es_vals,  marker_color='#ff7f0e'))
            fig_var.update_layout(
                title="VaR vs ES by Method — Fat-tail effect: Historical > Gaussian",
                barmode='group', height=380,
                yaxis_title="Loss ($)", xaxis_title="Method"
            )
            st.plotly_chart(fig_var, use_container_width=True)

            st.info(f"💡 Historical VaR is **${var_results['Historical']['VaR'] - var_results['Variance-Covariance']['VaR']:,.0f} higher** than Gaussian — direct evidence of fat tails in {detail_stock} returns.")

            st.markdown("---")

            # ── Backtest ──
            st.subheader("🔁 Rolling VaR Backtest (Kupiec test)")
            with timed("VaR"):
                exc_series, exc_rates = run_backtest_var(returns_var)

            col1, col2, col3 = st.columns(3)
            col1.metric("Historical Exception Rate", f"{exc_rates['Historical']:.2f}%", delta=f"{exc_rates['Historical']-1:.2f}% vs 1% target")
            col2.metric("Gaussian Exception Rate",   f"{exc_rates['Gaussian']:.2f}%",   delta=f"{exc_rates['Gaussian']-1:.2f}% vs 1% target")
            col3.metric("Target Rate (99% VaR)", "1.00%")

            # Exception clusters chart
            if len(exc_series) > 0:
                cum_pnl = returns_var.loc[exc_series.index] * 100_000
                fig_exc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        subplot_titles=["Cumulative P&L ($)", "VaR Exceptions"],
                                        row_heights=[0.65, 0.35], vertical_spacing=0.08)
                fig_exc.add_trace(go.Scatter(x=cum_pnl.index, y=cum_pnl.cumsum(),
                                             name="Cumulative P&L", line=dict(color='steelblue')), row=1, col=1)
                exc_dates = exc_series[exc_series == 1].index
                if len(exc_dates) > 0:
                    fig_exc.add_trace(go.Scatter(
                        x=exc_dates, y=[1] * len(exc_dates), mode='markers',
                        marker=dict(color='red', size=6, symbol='x'),
                        name="Exception"), row=2, col=1)
                fig_exc.update_layout(height=450, title="Exception Clustering — Christoffersen independence test")
                st.plotly_chart(fig_exc, use_container_width=True)

            hs_rate = exc_rates['Historical']
            kupiec_pass = abs(hs_rate - 1.0) < 0.5
            st.markdown(f"**Kupiec POF:** {'✅ PASS' if kupiec_pass else '❌ FAIL'} — exception rate {hs_rate:.2f}% vs 1% target")
            st.markdown("**Christoffersen:** Clustering during stress periods is the key failure mode for Historical VaR — exactly what brought down major shops in 2008.")

            st.markdown("---")

            # ── Stress Test ──
            st.subheader("🔥 Stress Test Scenarios")
            with timed("VaR"):
                stress = run_stress_test(returns_var)
            var_1d = var_results['Historical']['VaR']

            stress_names  = list(stress.keys())
            peak_dds      = [stress[s]['Peak Drawdown'] for s in stress_names]
            dd_var_ratios = [stress[s]['DD / VaR'] for s in stress_names]

            fig_stress = go.Figure()
            bar_colors = ['#d62728' if 'GFC' in n or 'COVID' in n else '#ff7f0e' for n in stress_names]
            fig_stress.add_trace(go.Bar(x=stress_names, y=peak_dds, marker_color=bar_colors, name="Peak Drawdown ($)"))
            fig_stress.add_hline(y=var_1d,       line_dash="dash",  line_color="blue",  annotation_text="99% VaR")
            fig_stress.add_hline(y=var_1d * 5,   line_dash="dot",   line_color="purple", annotation_text="5× VaR capital")
            fig_stress.update_layout(title="Stress Loss Waterfall — Peak Drawdown by Scenario",
                                      yaxis_title="Peak Loss ($)", height=420)
            st.plotly_chart(fig_stress, use_container_width=True)

            stress_df = pd.DataFrame([{
                'Scenario':      s,
                'Peak Drawdown': f"${stress[s]['Peak Drawdown']:,.0f}",
                'Worst Day':     f"${stress[s]['Worst Day']:,.0f}",
                'DD / VaR':      f"{stress[s]['DD / VaR']:.1f}×",
            } for s in stress_names])
            st.dataframe(stress_df, use_container_width=True, hide_index=True)
            st.warning("⚠️ Stress losses are typically 3–9× single-day VaR — the mathematical justification for FRTB's stressed-ES capital requirement.")

        # ── TAB 8 — Black-Scholes Options Pricer ───────────────────────────────
    with tab8:
        with timed("Options Pricer rendering"):
            st.header(f"📐 Options Pricer — {detail_stock}")
            st.caption("Methodology from Project 1 — Closed-form Black-Scholes, 5 Greeks, IV solver, Volatility Smile")

            S = current_price
            hist_vol = df['Returns'].dropna().std() * np.sqrt(252)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("⚙️ Option Parameters")
                K         = st.number_input("Strike Price ($)", value=round(S, 2), min_value=1.0, step=1.0)
                T_days    = st.slider("Days to Expiry", 1, 365, 30)
                r_rate    = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.1) / 100
                sigma_pct = st.slider("Implied Volatility (%)", 5.0, 150.0, round(hist_vol * 100, 1), 0.5)
                sigma     = sigma_pct / 100
                T         = T_days / 365
                opt_type  = st.radio("Option Type", ["call", "put"], horizontal=True)

            with col2:
                st.subheader("💰 Pricing Results")
                price_bs = bs_price(S, K, T, r_rate, sigma, opt_type)
                greeks   = bs_greeks(S, K, T, r_rate, sigma, opt_type)
                intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
                time_val  = price_bs - intrinsic

                m1, m2 = st.columns(2)
                m1.metric("Option Price", f"${price_bs:.4f}")
                m2.metric("Intrinsic Value", f"${intrinsic:.4f}")
                m1.metric("Time Value", f"${time_val:.4f}")
                moneyness = "ITM" if (S > K and opt_type == 'call') or (S < K and opt_type == 'put') else "OTM" if (S < K and opt_type == 'call') or (S > K and opt_type == 'put') else "ATM"
                m2.metric("Moneyness", moneyness)

            st.markdown("---")
            st.subheader("🔢 The Greeks")
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Delta (δ)", f"{greeks['delta']:.4f}", help="dV/dS — price sensitivity to underlying")
            g2.metric("Gamma (γ)", f"{greeks['gamma']:.6f}", help="d²V/dS² — rate of delta change")
            g3.metric("Vega (ν)",  f"{greeks['vega']:.4f}",  help="dV/dσ per 1% vol move")
            g4.metric("Theta (θ)", f"{greeks['theta']:.4f}", help="dV/dt — daily time decay ($)")
            g5.metric("Rho (ρ)",   f"{greeks['rho']:.4f}",   help="dV/dr per 1% rate move")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Price vs Strike")
                strikes_plot = np.linspace(S * 0.7, S * 1.3, 60)
                prices_plot  = [bs_price(S, k, T, r_rate, sigma, opt_type) for k in strikes_plot]
                intrinsics   = [max(S - k, 0) if opt_type == 'call' else max(k - S, 0) for k in strikes_plot]
                fig_payoff = go.Figure()
                fig_payoff.add_trace(go.Scatter(x=strikes_plot, y=intrinsics, name='Intrinsic Value',
                                                 line=dict(color='gray', dash='dash')))
                fig_payoff.add_trace(go.Scatter(x=strikes_plot, y=prices_plot, name='BS Price',
                                                 line=dict(color='#1f77b4', width=2)))
                fig_payoff.add_vline(x=S, line_dash="dot", line_color="red", annotation_text="Current Price")
                fig_payoff.add_vline(x=K, line_dash="dot", line_color="green", annotation_text="Strike")
                fig_payoff.update_layout(title=f"{opt_type.capitalize()} Price vs Strike",
                                          xaxis_title="Strike ($)", yaxis_title="Option Price ($)", height=360)
                st.plotly_chart(fig_payoff, use_container_width=True)

            with col2:
                st.subheader("😊 Volatility Smile")
                np.random.seed(42)
                smile_df = build_vol_smile(df)
                if not smile_df.empty:
                    fig_smile = go.Figure()
                    fig_smile.add_trace(go.Scatter(
                        x=smile_df['Moneyness'], y=smile_df['IV'],
                        mode='lines+markers', name='Implied Vol',
                        line=dict(color='#9467bd', width=2),
                        marker=dict(size=5)
                    ))
                    fig_smile.add_hline(y=hist_vol * 100, line_dash="dash", line_color="orange",
                                        annotation_text=f"Hist. Vol {hist_vol*100:.1f}%")
                    fig_smile.update_layout(
                        title="Implied Volatility Smile",
                        xaxis_title="Moneyness (K/S)", yaxis_title="Implied Vol (%)", height=360
                    )
                    st.plotly_chart(fig_smile, use_container_width=True)
                    st.caption("If BS were perfect, this curve would be flat. The skew is direct evidence of fat tails and crash risk not captured by constant-vol models.")

            st.markdown("---")
            st.subheader("🔁 Put-Call Parity Check")
            call_price_pcp = bs_price(S, K, T, r_rate, sigma, 'call')
            put_price_pcp  = bs_price(S, K, T, r_rate, sigma, 'put')
            lhs = call_price_pcp - put_price_pcp
            rhs = S - K * np.exp(-r_rate * T)
            parity_error = abs(lhs - rhs)
            st.success(f"✅ C − P = ${lhs:.6f} | S − Ke^(−rT) = ${rhs:.6f} | Error = ${parity_error:.2e} (should be ~0)")

        # ── TAB 9 — Factor Model ───────────────────────────────────────────────
    with tab9:
        with timed("Factor Model rendering"):
            st.header(f"🔬 Factor Model — {detail_stock}")
            st.caption("Methodology from Project 3 — Fama-French factor exposures, 12-1 momentum signal, long/short backtest")

            st.subheader("📊 Fama-French Factor Exposures")
            with timed("Factor model"):
                exposures = calculate_factor_exposures(df, period="2y")

            if exposures:
                col1, col2 = st.columns([1, 1])

                with col1:
                    e1, e2 = st.columns(2)
                    e1.metric("Alpha (α) annualised", f"{exposures['alpha']:+.2f}%",
                              help="Excess return unexplained by factors. >0 = outperformance.")
                    e2.metric("Market Beta (β)",      f"{exposures['beta']:.3f}",
                              help="Sensitivity to market. >1 = amplified market moves.")
                    e1.metric("SMB (size factor)",    f"{exposures['smb']:+.3f}",
                              help="Negative = large-cap tilt (typical for mega-cap tech).")
                    e2.metric("HML (value factor)",   f"{exposures['hml']:+.3f}",
                              help="Negative = growth tilt. Positive = value tilt.")
                    e1.metric("MOM (momentum)",       f"{exposures['mom']:+.3f}",
                              help="Positive = trend-following; negative = mean-reverting.")
                    e2.metric("R² (explained)",        f"{exposures['r2']:.2%}",
                              help="% of return variation explained by the 4 factors.")
                    st.caption(f"Regression on {exposures.get('n_obs', '—')} daily observations | Factors: SPY (MKT), IWM−SPY (SMB), IVE−IVW (HML), MTUM (MOM)")

                with col2:
                    # Factor exposure bar chart
                    factor_names = ['Beta (β)', 'SMB', 'HML', 'MOM']
                    factor_vals  = [exposures['beta'], exposures['smb'], exposures['hml'], exposures['mom']]
                    bar_cols = ['#1f77b4' if v >= 0 else '#d62728' for v in factor_vals]
                    fig_exp = go.Figure(go.Bar(
                        x=factor_names, y=factor_vals,
                        marker_color=bar_cols, text=[f"{v:+.3f}" for v in factor_vals],
                        textposition='outside'
                    ))
                    fig_exp.add_hline(y=0, line_color='gray', line_width=0.8)
                    fig_exp.update_layout(
                        title=f"{detail_stock} Factor Exposures",
                        yaxis_title="Loading", height=360,
                        yaxis=dict(zeroline=True)
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)

                # Interpretation
                beta = exposures['beta']
                alpha = exposures['alpha']
                if beta > 1.2:
                    st.warning(f"⚡ High beta ({beta:.2f}) — {detail_stock} amplifies market moves by {beta:.1f}×")
                elif beta < 0.8:
                    st.success(f"🛡️ Defensive beta ({beta:.2f}) — {detail_stock} is less sensitive to market swings")
                else:
                    st.info(f"📊 Market beta ({beta:.2f}) — {detail_stock} moves roughly in line with the market")

                if alpha > 5:
                    st.success(f"✅ Positive alpha ({alpha:+.1f}% annualised) — outperforming risk-adjusted benchmark")
                elif alpha < -5:
                    st.warning(f"⚠️ Negative alpha ({alpha:+.1f}% annualised) — underperforming risk-adjusted benchmark")

            else:
                st.warning("Insufficient data for factor regression (need 60+ days).")

            st.markdown("---")

            # ── Momentum Signal ──
            st.subheader("🚀 Momentum Signal (12-1 month)")
            mom_signals = {}
            with timed("Factor model"):
                for stock in selected_stocks:
                    sdf = stock_data[stock]
                    if len(sdf) >= 22:
                        mom_signals[stock] = calculate_momentum_signal(sdf)

            if mom_signals:
                sorted_mom = sorted(mom_signals.items(), key=lambda x: x[1], reverse=True)
                mom_stocks = [x[0] for x in sorted_mom]
                mom_vals   = [x[1] for x in sorted_mom]
                mom_colors = ['#2ca02c' if v > 0 else '#d62728' for v in mom_vals]

                fig_mom = go.Figure(go.Bar(
                    x=mom_stocks, y=mom_vals,
                    marker_color=mom_colors,
                    text=[f"{v:+.1f}%" for v in mom_vals],
                    textposition='outside'
                ))
                fig_mom.add_hline(y=0, line_color='gray')
                fig_mom.update_layout(
                    title="12-1 Month Momentum Signal — Long top tercile, Short bottom tercile",
                    yaxis_title="Momentum (%)", height=360
                )
                st.plotly_chart(fig_mom, use_container_width=True)

                # Rank table
                rank_df = pd.DataFrame(sorted_mom, columns=['Stock', 'Momentum (%)'])
                rank_df['Rank']   = range(1, len(rank_df) + 1)
                rank_df['Signal'] = rank_df['Momentum (%)'].apply(
                    lambda x: '🟢 LONG'  if x == max(mom_vals) else
                              '🔴 SHORT' if x == min(mom_vals) else '⚪ Neutral')
                rank_df['Momentum (%)'] = rank_df['Momentum (%)'].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(rank_df[['Rank', 'Stock', 'Momentum (%)', 'Signal']],
                             use_container_width=True, hide_index=True)

            st.markdown("---")

            # ── Momentum Backtest ──
            st.subheader(f"📈 Momentum Strategy Backtest — {detail_stock}")
            with timed("Factor model"):
                bt = momentum_backtest(df)

            if bt:
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Annualised Return", f"{bt['ann_return']:+.2f}%")
                b2.metric("Sharpe Ratio",      f"{bt['sharpe']:.2f}")
                b3.metric("Max Drawdown",      f"{bt['max_drawdown']:.2f}%")
                b4.metric("Hit Rate",          f"{bt['hit_rate']:.1f}%")

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(
                    x=bt['cum_returns'].index, y=bt['cum_returns'],
                    name='Momentum Strategy', line=dict(color='#d62728', width=2),
                    fill='tonexty', fillcolor='rgba(214,39,40,0.08)'
                ))
                fig_bt.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Breakeven")
                fig_bt.update_layout(
                    title=f"Long/Short Momentum Cumulative Returns — {detail_stock}",
                    xaxis_title="Date", yaxis_title="Cumulative Return",
                    height=400
                )
                st.plotly_chart(fig_bt, use_container_width=True)
                st.caption("Strategy goes long when 12-1 month momentum is positive, short when negative. Sub-50% hit rate with positive returns reflects momentum's characteristic negative-skew payoff profile.")
            else:
                st.info("Need 252+ days of data to run the momentum backtest.")

    # Snapshot this rerun's timings/cache status for the sidebar debug panel.
    # The panel is drawn earlier in this same function, so it always shows
    # the PREVIOUS completed rerun's numbers, not this one.
    if DEBUG_PERFORMANCE:
        st.session_state['last_timings'] = dict(_TIMINGS)
        st.session_state['last_cache_status'] = {k: dict(v) for k, v in _CACHE_STATUS.items()}


if __name__ == "__main__":
    with timed("Total rerun time"):
        main()
    _log_timing_report()