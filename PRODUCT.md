# Product

## Register

product

## Users

Primary audience is recruiters and hiring managers evaluating Dev Golakiya's quantitative finance skills (UMass Amherst MS in Business Analytics portfolio piece) — they click through tabs for a few minutes, scanning for technical depth and engineering polish. But the dashboard only earns credibility if it also works as a genuinely usable analysis tool: someone could use it for real to read price action, risk, options pricing, and factor exposures. Design decisions should satisfy both — a tool a quant team would actually use, which is exactly what makes it convincing to a recruiter.

## Product Purpose

A single always-on Streamlit dashboard (deployed on Google Cloud Run) that integrates four standalone quant finance projects — Black-Scholes options pricer, VaR/Expected Shortfall with stress testing, Fama-French factor model, and a proprietary multi-factor risk score — into one coherent 9-tab analysis platform with live Yahoo Finance data. It exists as living proof of quantitative finance ability. Success looks like: a viewer leaves convinced this person can build real institutional-grade analytics, not just wire up `yfinance` and `Plotly` into generic charts.

## Brand Personality

Institutional / terminal-grade. Reference: TradingView — dark, modern, chart-first, professional, but contemporary rather than retro-clunky like classic Bloomberg terminals. Three words: **precise, dense, credible**. Color and layout should read as functional instrumentation for someone who lives in numbers all day, not as a marketing surface.

## Anti-references

- **Generic AI SaaS dashboard look** — explicitly what the current implementation has: purple/blue gradient metric cards (`#667eea → #764ba2`), gradient-clip-text headers, side-stripe-bordered "insight boxes." These read as templated AI output, not institutional tooling, and undercut the credibility this dashboard is supposed to build.
- **Consumer trading app playfulness** — no Robinhood-style rounded cute UI, no confetti, no casual tone. This is not a retail trading app.

## Design Principles

1. **Function before flourish** — every visual choice should make the underlying quant model (VaR method, factor exposure, Greek, risk tier) more legible, not decorate it.
2. **Color carries meaning, not decoration** — red/green and risk-tier colors are load-bearing data signals; gradients for their own sake are not allowed. Pair color with icons/labels so meaning survives without color vision.
3. **Density with hierarchy** — institutional tools pack many numbers per screen; the goal is to do that while staying scannable in the few minutes a recruiter will actually spend, not to minimize information.
4. **Show the rigor, don't caption it** — prove competence through how methodology, confidence intervals, and assumptions are surfaced in the UI, not through copy that asserts "proprietary" or "institutional-grade."
5. **One consistent system, not one-off styling** — extend a small shared set of tokens/classes across all 9 tabs rather than inventing new gradients, card styles, or colors per section.

## Accessibility & Inclusion

- WCAG AA contrast minimums for all text, including against the dark background already in use.
- Never rely on red/green alone for gains/losses or risk tiers — pair with icons, symbols, or text labels (colorblind-safe).
- Respect `prefers-reduced-motion` for any animation or transition work.
