# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A single-file Streamlit dashboard (`dashboard.py`) for quantitative stock market analysis. It integrates four standalone quant projects (Black-Scholes options pricer, Monte Carlo, Fama-French factor model, VaR/ES calculator) into one app with 9 tabs: Overview, Risk Analysis, Technical, Forecast, Performance, Portfolio, VaR & Stress Test, Options Pricer, Factor Model.

Live deployment: Google Cloud Run, auto-deployed via Cloud Build on every push to `main` (no Dockerfile/cloudbuild config is currently checked into this repo, despite the README describing that pipeline — treat deployment config as external/managed elsewhere).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run dashboard.py
```

There is no test suite, linter, or CI config in this repo — don't invent commands for them.

## Repository structure

- `dashboard.py` — the real app (~1950 lines). This is the only file that matters for feature work.
- `dashboard_broken.py`, `dashboard_simple.py` — untracked, older/scratch variants of the dashboard, not wired into deployment. Don't edit these for feature work unless explicitly asked; if asked to "fix the dashboard," confirm which file is meant.

## Architecture of `dashboard.py`

The file is one long script: helper/calculation functions defined top-to-bottom, then a `main()` that builds the Streamlit UI and is invoked at the bottom of the file. There's no separate module structure — when changing behavior, find the relevant function by name (most are unique and descriptive) rather than expecting package boundaries.

**Data layer**
- `fetch_real_stock_data(symbol, period)` — live OHLCV from `yfinance`, `@st.cache_data(ttl=30)` so it refetches at most every 30s. Falls back to `generate_stock_data` (synthetic) on empty result or exception.
- `generate_stock_data(symbol, days)` — synthetic OHLCV generator (seeded by `hash(symbol)`) used as a fallback and as the "Synthetic (Simulated)" data source toggle in the sidebar.
- Both functions independently compute the same technical indicator set (SMA 20/50/200, rolling volatility, RSI, MACD, Bollinger Bands) directly on the DataFrame — if you change an indicator's formula, update it in *both* places.
- `fetch_factor_proxies(period)` — cached (`ttl=3600`) ETF proxy data (SPY, IWM, IVE/IVW, MTUM) used for the Fama-French factor model tab.

**Analytics layer** (each tab's math lives in its own function(s), called from `main()`):
- `calculate_dev_risk_score(df)` — proprietary weighted risk score: Volatility 30%, Momentum 25%, Liquidity 20%, Technical 15%, Drawdown 10%.
- `generate_predictions`, `calculate_performance_metrics`, `generate_recommendation` — forecast/performance/portfolio tabs.
- `calculate_var_es`, `run_backtest_var`, `run_stress_test` — VaR & Stress Test tab (historical sim, variance-covariance, Monte Carlo; Kupiec POF backtest; 2008 GFC / 2020 COVID replays).
- `bs_price`, `bs_greeks`, `implied_vol_newton`, `build_vol_smile` — Black-Scholes options pricer tab (Newton-Raphson IV with bisection fallback).
- `calculate_factor_exposures`, `calculate_momentum_signal`, `momentum_backtest` — Factor Model tab (Fama-French 4-factor regression, 12-1 momentum long/short backtest).

**UI layer**
- `main()` builds the sidebar (stock multiselect + custom ticker entry, historical period, forecast horizon, real-vs-synthetic data toggle, auto-refresh controls using `st.session_state`) and then the 9 `st.tabs(...)` sections in a fixed order (`tab1`..`tab9` matching the list above). Each `with tabN:` block is self-contained — when adding a tab, follow this pattern rather than introducing a different structure.
- Auto-refresh is implemented manually via `st.session_state.last_refresh_time` / `refresh_counter` plus `st.cache_data.clear()` + `st.rerun()`, not a Streamlit built-in — be careful not to clear caches more aggressively than intended if touching this logic.

## Conventions to follow

- Stock universe defaults and per-symbol base prices/sectors are hardcoded dicts near the top of the file (e.g. `STOCK_SECTORS`, `base_prices` in `generate_stock_data`) — extend these dicts rather than special-casing symbols elsewhere.
- Custom tickers typed into the sidebar are accepted ad hoc (not validated against the hardcoded sector/base-price maps), so code reading those maps must handle missing keys gracefully (`.get(symbol, default)`), matching the existing pattern.
- UI strings are emoji-prefixed (`📊`, `⚠️`, etc.) consistently per tab/section — match this style for new UI text.

## Design context

`PRODUCT.md` and `DESIGN.md` (plus the `.impeccable/design.json` sidecar) capture the strategic and visual design system — read them before any UI/styling work. Headline takeaways:
- Register is **product**: this is a tool serving recruiters evaluating quant skill, not a marketing page — design should serve the analysis workflow, not decorate it.
- The current CSS has known anti-patterns flagged for removal: the `.main-header` gradient-clip-text masthead, the `.metric-card` purple gradient (`#667EEA → #764BA2`), and the dead unused `.insight-box` side-stripe class. Don't extend or copy these patterns into new code.
- Color is reserved for five meaningful signals only (gain/loss, risk tier, VaR breach, factor sign, buy/hold/sell) — see DESIGN.md's "Risk Tier Rule".
