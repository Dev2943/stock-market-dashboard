---
name: Stock Market Analysis Platform
description: A 9-tab institutional risk/analytics dashboard built on Streamlit's dark shell, currently undermined by ad hoc SaaS-gradient cards layered on top of it.
colors:
  canvas-black: "#0E1117"
  panel-charcoal: "#262730"
  ink-white: "#FAFAFA"
  muted-gray: "#666666"
  signal-red: "#FF4B4B"
  gain-green: "#008000"
  loss-red: "#FF0000"
  risk-low-a: "#11998E"
  risk-low-b: "#38EF7D"
  risk-medium-a: "#F2994A"
  risk-medium-b: "#F2C94C"
  risk-high-a: "#EB3349"
  risk-high-b: "#F45C43"
  deprecated-header-a: "#2E3192"
  deprecated-header-b: "#1BFFFF"
  deprecated-metric-a: "#667EEA"
  deprecated-metric-b: "#764BA2"
  dead-insight-bg: "#F8F9FA"
  dead-insight-border: "#667EEA"
typography:
  display:
    fontFamily: "Source Sans Pro, sans-serif"
    fontSize: "2.5rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "normal"
  caption:
    fontFamily: "Source Sans Pro, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: "normal"
  body:
    fontFamily: "Source Sans Pro, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
rounded:
  sm: "5px"
  md: "10px"
  lg: "15px"
spacing:
  sm: "0.5rem"
  md: "1rem"
  lg: "1.5rem"
components:
  risk-score-low:
    backgroundColor: "{colors.risk-low-a}"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.lg}"
    padding: "1.5rem"
  risk-score-medium:
    backgroundColor: "{colors.risk-medium-a}"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.lg}"
    padding: "1.5rem"
  risk-score-high:
    backgroundColor: "{colors.risk-high-a}"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.lg}"
    padding: "1.5rem"
---

# Design System: Stock Market Analysis Platform

## 1. Overview

**Creative North Star: "The Quant Terminal"**

The dashboard sits on a genuinely good foundation it never earned credit for: Streamlit's dark shell (`#0E1117` canvas, `#262730` panels, `#FAFAFA` text) already reads as a TradingView-style instrument panel — dense, dark, low-glare, built for someone who watches numbers all day. That foundation is undermined by a second layer of decoration bolted on top: a gradient-clip-text masthead, a purple-to-violet `metric-card` gradient lifted straight from 2021 SaaS landing pages, and one dead CSS class (`.insight-box`, a side-stripe-bordered callout) that isn't even wired into any markup. The Quant Terminal direction means stripping that decorative layer back to the shell's own dark, flat, color-as-signal language — the same restraint a real risk desk or trading terminal enforces, because every accent color there means something specific (buy, sell, breach, warning), not "this card is important."

This system explicitly rejects the generic-AI-SaaS-dashboard look this codebase currently has (gradient cards, gradient text headers) and consumer-trading-app playfulness (Robinhood-style rounded cuteness). Depth comes from contrast and density, not gradients; color is rationed to data that actually carries meaning — gains/losses, risk tiers, breach flags — never spent on decoration.

**Key Characteristics:**
- Near-black canvas, charcoal panels, off-white text — inherited from Streamlit's dark theme, currently uncommitted to the repo
- Color is semantic only: red/green for direction, a three-stop ramp for risk tiers, nothing else
- Flat by default; elevation reserved for transient surfaces (hover, modal, popover), never base layout
- Dense, multi-metric layouts (4-5 `st.metric` columns, 9 tabs) read at terminal speed, not landing-page speed

## 2. Colors

The palette as it exists today is split between a coherent dark instrument-panel base and a disconnected, decorative gradient layer — the Don'ts in Section 6 exist to close that gap.

### Primary
- **Signal Red** (`#FF4B4B`): Streamlit's unmodified default `primaryColor`. Currently does double duty as the active-tab underline, slider track, and focus ring — the only accent color in the entire UI that wasn't hand-picked for this project.

### Neutral
- **Canvas Black** (`#0E1117`): page background. Comes from Streamlit's default dark theme, not from any setting committed in this repo.
- **Panel Charcoal** (`#262730`): sidebar, `st.dataframe` chrome, and the fill behind every native `st.info` / `st.success` / `st.warning` / `st.error` banner.
- **Ink White** (`#FAFAFA`): primary text color throughout, inherited from the same uncommitted theme.
- **Muted Gray** (`#666666`): the `.sub-header` byline color. Hardcoded assuming a light page background; against `#0E1117` it under-shoots WCAG AA for body-sized text. Flagged in Do's and Don'ts.

### Functional (data-meaning colors)
- **Gain Green** (`#008000`, CSS `green`): positive return text and the ↗️ arrow in the Overview metric cards.
- **Loss Red** (`#FF0000`, CSS `red`): negative return text and the ↘️ arrow. Note this is a *different* red from Signal Red (`#FF4B4B`) — two unrelated reds doing two unrelated jobs is exactly the kind of one-off-per-section drift Design Principle 5 in PRODUCT.md calls out.
- **Risk Tier Ramp** — three two-stop gradients keyed to `calculate_dev_risk_score`'s low/medium/high bands:
  - Low risk: `#11998E → #38EF7D` (teal to green)
  - Medium risk: `#F2994A → #F2C94C` (orange to gold)
  - High risk: `#EB3349 → #F45C43` (crimson to coral)
  
  This is the one gradient usage in the codebase that's actually load-bearing (it encodes the risk tier, not just decoration) — see the Risk Tier Rule below.

### Deprecated (documented so they can be removed, not reused)
- **Deprecated Header Gradient** (`#2E3192 → #1BFFFF`): the `.main-header` gradient-clip-text masthead. Pure decoration, the textbook AI-SaaS tell. Remove; replace with solid Ink White.
- **Deprecated Metric Gradient** (`#667EEA → #764BA2`): the `.metric-card` background behind every Overview price tile. Carries no meaning — it's the same purple on every stock regardless of performance. Remove.
- **Dead Insight Colors** (`#F8F9FA` bg / `#667EEA` border): `.insight-box` is defined in the stylesheet but never referenced by any `class="insight-box"` in the markup — 100% dead CSS, and it's also a side-stripe-border anti-pattern. Delete the rule rather than ever wiring it up.

### Named Rules
**The Risk Tier Rule.** Color may only encode the five things that already have meaning in this codebase: gain/loss direction, risk tier (low/medium/high), VaR breach/no-breach, factor sign (positive/negative alpha or beta), and recommendation (buy/hold/sell). If a color isn't standing in for one of those, it's decoration and it's not allowed.

**The One Red Rule.** Signal Red (`#FF4B4B`, Streamlit's UI accent) and Loss Red (`#FF0000`, the gains/losses color) are not allowed to keep diverging by accident. The next time either is touched, converge them to a single deliberate red used for both "UI accent" and "negative number," or pick a different, explicit Signal Red that isn't just whatever Streamlit shipped.

## 3. Typography

**Body/UI Font:** Source Sans Pro (Streamlit's built-in default stack, with system sans-serif fallback)
**Display Font:** none — the only deliberate type override in the codebase is `.main-header`'s `font-size: 2.5rem; font-weight: bold`, applied on top of the same Source Sans Pro stack, not a distinct typeface

**Character:** Entirely inherited. Every heading, label, `st.metric` value, dataframe cell, and tab title uses Streamlit's default UI font at Streamlit's default sizes — the codebase has made exactly one typographic decision (the masthead size/weight) in ~1,950 lines.

### Hierarchy
- **Display** (700, 2.5rem, line-height 1.2): the `.main-header` masthead only — "📊 Stock Market Analysis Platform." Currently rendered as gradient-clip text; should render as solid Ink White (see Don'ts).
- **Title** (Streamlit default `st.header`/`st.subheader` sizing, not overridden): the per-tab section headers ("🎯 Risk Analysis - NVDA", "🏢 Sector Performance Analysis").
- **Body** (1rem, 400): dataframe cells, `st.info`/`st.success`/`st.warning` banner text, insight bullets.
- **Caption** (1rem, 400, Muted Gray): the `.sub-header` byline under the masthead — "Built by Dev Golakiya | UMass Amherst Business Analytics." Needs a contrast fix (see Don'ts), not a size or weight change.

### Named Rules
**The Inherited-Until-Earned Rule.** Don't introduce a second typeface or a new display size to "make a tab feel special." The system has earned exactly one display style (the masthead); any new typographic emphasis should come from Streamlit's existing `st.header`/`st.subheader`/`st.metric` hierarchy, not a new custom class.

## 4. Elevation

Currently zero shadows anywhere in the codebase — depth is conveyed entirely by flat background-color blocks (Canvas Black vs. Panel Charcoal vs. the saturated gradient cards). Going forward, the system stays flat at rest — that's the correct instrument-panel convention — but introduces restrained elevation as a *state* signal: hover, focus, and transient overlays (popovers, the live-mode variant panel, any future modal/dialog) may lift off the canvas. Base layout, tabs, and metric tiles never do.

### Shadow Vocabulary (to be introduced, not yet present in code)
- **hover-lift** (`box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4)`): subtle lift on interactive cards/rows on hover, against the dark canvas.
- **overlay-lift** (`box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5)`): for any future modal, popover, or dropdown that needs to separate from the page behind it.

### Named Rules
**The Flat-at-Rest Rule.** Nothing in the base layout (tab panels, metric tiles, the sidebar) carries a shadow at rest. Shadows only appear in direct response to interaction state (hover, focus) or for content that's genuinely floating above the page (overlays, popovers).

## 5. Components

### Metric Tiles (Overview tab)
- **Current:** `.metric-card` — `border-radius: 10px`, `padding: 1rem`, white text on the Deprecated Metric Gradient (`#667EEA → #764BA2`), one per selected stock in a 4-column row.
- **Character going forward:** instrumented and quiet — flat Panel Charcoal background, Ink White text, no gradient. The only color allowed inside the tile is the existing inline gain/loss span (`color: green` / `color: red`) on the percent-change line.

### Risk Score Card (Risk Analysis tab)
- **Shape:** `border-radius: 15px`, `padding: 1.5rem`.
- **Color assignment:** solid (not gradient) fill from the Risk Tier Ramp's first stop — `risk-low-a` / `risk-medium-a` / `risk-high-a` — chosen by `calculate_dev_risk_score`'s 35/60 thresholds. This is the one card in the system where color is load-bearing, so it's the one card allowed to differ visually from the otherwise-neutral tile language.
- **Internal content:** score out of 100, risk-level label, one-line methodology caption at `0.9rem`.

### Recommendation Card (Portfolio tab)
- **Current:** the code assigns `card_style` to `"recommendation-buy"` / `"recommendation-sell"` / `"recommendation-hold"` and renders a `<div class="{card_style}">`, but none of those three classes exist in the stylesheet — they render as a plain unstyled `<div>`. This is a gap, not a style choice; document the intent (buy/hold/sell needs its own color treatment, presumably drawing from Gain Green / Loss Red / a neutral hold color) as unresolved rather than inventing colors for it here.

### Native Streamlit Components (undocumented by design, inherited as-is)
- **Tabs** (`st.tabs`, 9 panels): Streamlit default styling — Signal Red active-tab underline, Ink White label text.
- **Metrics** (`st.metric`): native delta-colored (green/red) value display, used throughout instead of custom HTML wherever a card doesn't need a gradient/background.
- **Banners** (`st.info` / `st.success` / `st.warning` / `st.error`): native Panel Charcoal-backed callouts with a colored left icon, used for every contextual insight, methodology note, and data-source notice in the app. These are already the "quiet instrumented" callout the system wants — new callout-style content should use these natively rather than reviving `.insight-box`.
- **DataFrames** (`st.dataframe`): native dark-themed tables for sector breakdowns, correlation matrices, stress-test results.
- **Plotly charts**: no shared template — each chart sets its own `colorscale` (`RdYlGn`, `RdYlGn_r`) or `color_discrete_sequence` (`px.colors.qualitative.Set3`, `.Pastel`) independently. Candlesticks use Plotly's default green/red. No project-wide chart theme is set.

## 6. Do's and Don'ts

### Do:
- **Do** keep the dark Canvas Black / Panel Charcoal / Ink White base — commit it explicitly via a `.streamlit/config.toml` `[theme]` block instead of relying on each viewer's inherited default, so the deployed app looks the same for every recruiter who opens it.
- **Do** reserve color strictly for the five meaningful signals: gain/loss direction, risk tier, VaR breach, factor sign, buy/hold/sell — per the Risk Tier Rule.
- **Do** use the native `st.info` / `st.success` / `st.warning` / `st.error` banners for any new contextual callout — they already deliver the "instrumented and quiet" feel the system wants.
- **Do** keep the base layout flat; introduce shadows only for hover/focus states and transient overlays, per the Flat-at-Rest Rule.
- **Do** fix `.sub-header`'s `color: #666` — it was written against an assumed light page and fails contrast against the `#0E1117` dark canvas it actually renders on.

### Don't:
- **Don't** use gradient-clip-text headings (`-webkit-background-clip: text`) anywhere — explicitly the AI-SaaS tell flagged in PRODUCT.md's anti-references. Remove `.main-header`'s `#2E3192 → #1BFFFF` gradient and render it as solid Ink White.
- **Don't** use decorative gradient card backgrounds where the gradient carries no information — explicitly the `.metric-card` (`#667EEA → #764BA2`) pattern. The only gradient/color-coded card allowed to exist is the risk score card, because its color *is* the data.
- **Don't** use `border-left`/`border-right` as a colored accent stripe on any card or callout — `.insight-box`'s 4px `#667EEA` left border is dead code; delete the rule, don't ever wire it up.
- **Don't** introduce a second display typeface or a new heading size "for emphasis." One display style exists (the masthead); everything else borrows Streamlit's native `st.header`/`st.subheader`/`st.metric` hierarchy.
- **Don't** add a new one-off card style (a new radius, a new gradient, a new padding scale) per tab. Extend the three radius tokens (`sm` 5px / `md` 10px / `lg` 15px) and three spacing tokens (`sm` 0.5rem / `md` 1rem / `lg` 1.5rem) already in use rather than inventing a fourth value.
