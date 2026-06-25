---
name: Stock Market Analysis Platform
description: A 9-tab institutional risk/analytics dashboard built on Streamlit's dark shell, now expressed through a restrained glass/gradient surface system ("Glass Terminal") instead of flat panels.
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
  recommendation-buy-a: "#0F8A3C"
  recommendation-buy-b: "#0A6B2E"
  recommendation-sell-a: "#C2273D"
  recommendation-sell-b: "#8E1B2D"
  recommendation-hold-a: "#5B6472"
  recommendation-hold-b: "#3D434D"
  glass-surface: "rgba(38, 39, 48, 0.55)"
  glass-surface-strong: "rgba(38, 39, 48, 0.85)"
  glass-border: "rgba(255, 255, 255, 0.09)"
  glass-highlight: "rgba(255, 255, 255, 0.07)"
  ambient-mesh-blue: "rgba(46, 84, 138, 0.16)"
  ambient-mesh-teal: "rgba(20, 110, 100, 0.13)"
  ambient-mesh-red: "rgba(255, 75, 75, 0.05)"
  deprecated-header-a: "#2E3192"
  deprecated-header-b: "#1BFFFF"
  deprecated-metric-a: "#667EEA"
  deprecated-metric-b: "#764BA2"
  dead-insight-bg: "#F8F9FA"
  dead-insight-border: "#667EEA"
typography:
  display:
    fontFamily: "Source Sans Pro, sans-serif"
    fontSize: "2.6rem"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "-0.01em"
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
  xl: "20px"
spacing:
  sm: "0.5rem"
  md: "1rem"
  lg: "1.5rem"
blur:
  sm: "10px"
  lg: "22px"
shadow:
  hover: "0 2px 8px rgba(0, 0, 0, 0.4)"
  overlay: "0 8px 32px rgba(0, 0, 0, 0.5)"
  glass: "0 10px 30px rgba(0, 0, 0, 0.38)"
motion:
  ease: "cubic-bezier(.23, 1, .32, 1)"
  hover: "160-200ms"
  entrance: "420-480ms"
components:
  hero-panel:
    backgroundColor: "{colors.glass-surface}"
    blur: "{blur.lg}"
    rounded: "{rounded.xl}"
    border: "{colors.glass-border}"
    shadow: "{shadow.glass}"
  kpi-card:
    backgroundColor: "{colors.glass-surface}"
    blur: "{blur.sm}"
    rounded: "{rounded.md}"
    border: "{colors.glass-border}"
  risk-score-low:
    backgroundColor: "linear-gradient({colors.risk-low-a}, {colors.risk-low-b})"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.lg}"
    padding: "1.5rem"
  risk-score-medium:
    backgroundColor: "linear-gradient({colors.risk-medium-a}, {colors.risk-medium-b})"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.lg}"
    padding: "1.5rem"
  risk-score-high:
    backgroundColor: "linear-gradient({colors.risk-high-a}, {colors.risk-high-b})"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.lg}"
    padding: "1.5rem"
  recommendation-buy:
    backgroundColor: "linear-gradient({colors.recommendation-buy-a}, {colors.recommendation-buy-b})"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.md}"
    padding: "1.25rem 1.5rem"
  recommendation-sell:
    backgroundColor: "linear-gradient({colors.recommendation-sell-a}, {colors.recommendation-sell-b})"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.md}"
    padding: "1.25rem 1.5rem"
  recommendation-hold:
    backgroundColor: "linear-gradient({colors.recommendation-hold-a}, {colors.recommendation-hold-b})"
    textColor: "{colors.ink-white}"
    rounded: "{rounded.md}"
    padding: "1.25rem 1.5rem"
---

# Design System: Stock Market Analysis Platform

## 1. Overview

**Creative North Star: "The Glass Terminal"**

The dashboard sits on a genuinely good foundation: Streamlit's dark shell (`#0E1117` canvas, `#262730` panels, `#FAFAFA` text) already reads as a TradingView-style instrument panel — dense, dark, low-glare, built for someone who watches numbers all day. The system evolved from an earlier, stricter "Quant Terminal" direction (flat panels, zero decoration) to Glass Terminal because flat-only was reading as under-produced rather than disciplined — a real institutional product (Bloomberg, a modern trading desk, Stripe's dashboard surfaces) earns "premium" through material depth, not just restraint. Glass Terminal keeps every credibility rule the old system had — color still only encodes real signals, no gradient-clip-text headlines, no purple SaaS slop — but adds one deliberate new material: frosted glass surfaces with restrained gradient depth, used consistently everywhere a panel needs to feel "elevated" rather than "flat."

The distinction that keeps this from sliding back into AI-SaaS-dashboard territory: **gradients live in surfaces and backdrops, never in text.** The masthead is solid Ink White on a glass panel with a gradient *behind* it — not gradient-clip-text. The ambient background mesh is tonal blue/teal (a trading floor at night), not saturated purple. Every gradient fill on a card either *is* the data (risk tier, buy/sell/hold) or is a low-opacity accent layered under solid content, never the entire visual weight of a tile.

**Key Characteristics:**
- Near-black canvas, charcoal panels, off-white text — inherited from Streamlit's dark theme, committed via `.streamlit/config.toml`
- Color is still semantic-first: red/green for direction, a three-stop ramp for risk tiers, a three-stop ramp for buy/hold/sell — gradients on these cards encode the signal, they don't decorate it
- Glass is a new, deliberate elevation tier: the hero panel, KPI tiles, the ticker strip, risk/recommendation cards, and skeleton loaders all sit on translucent, blurred, bordered surfaces — distinct from the flat base canvas and flat chart/dataframe frames beneath them
- A faint ambient gradient mesh (tonal blue/teal, low opacity) plus the existing grid/sparkline texture drifts slowly behind everything — pure CSS, never competing with text contrast
- Motion is restrained and motivated: entrance fades, hover lift, button shimmer, tab transitions, a live ticker — all transform/opacity only, all neutered under `prefers-reduced-motion`
- Dense, multi-metric layouts (4-5 `st.metric` columns, 9 tabs) still read at terminal speed, not landing-page speed — glass adds depth, it doesn't add scroll

## 2. Colors

### Primary
- **Signal Red** (`#FF4B4B`): Streamlit's unmodified default `primaryColor`. Active-tab underline/glow, button hover glow and shimmer, focus ring, and one of the two hues in the ambient gradient mesh — the only accent hue in the system, used consistently rather than introducing a second uncoordinated color.

### Neutral
- **Canvas Black** (`#0E1117`): page background, committed via `.streamlit/config.toml`.
- **Panel Charcoal** (`#262730`): sidebar, `st.dataframe` chrome, native `st.info`/`st.success`/`st.warning`/`st.error` banners, and the base tone that `--glass-surface` is a translucent variant of.
- **Ink White** (`#FAFAFA`): primary text color throughout. Headlines stay solid Ink White even on glass/gradient surfaces — text is never the gradient.
- **Muted Gray** (`#666666`): legacy `.sub-header` color, deprecated for contrast (see Do's). Current `.sub-header` uses `rgba(250, 250, 250, 0.6)` instead.

### Glass surface tokens
- **Glass Surface** (`rgba(38, 39, 48, 0.55)`) / **Glass Surface Strong** (`rgba(38, 39, 48, 0.85)`, the `prefers-reduced-transparency` fallback): translucent variants of Panel Charcoal, paired with `backdrop-filter: blur()`.
- **Glass Border** (`rgba(255, 255, 255, 0.09)`) / **Glass Highlight** (`rgba(255, 255, 255, 0.07)`, used as an `inset` top highlight): the frosted-edge treatment that reads as "physical glass" rather than "transparent PNG."
- **Blur tiers** — exactly two, no more: `--blur-sm` (10px, for thin/small surfaces: ticker strip, skeleton cards, KPI tiles) and `--blur-lg` (22px, for the hero panel only). Chart and dataframe frames intentionally use a flat rgba tint with **no** blur — dozens of them exist per tab, and blur at that count is a real rendering cost for no added benefit at that size.

### Functional (data-meaning colors)
- **Gain Green** (`#008000`) / **Loss Red** (`#FF0000`): positive/negative return text and arrows. Still two different reds doing two different jobs (Signal Red vs. Loss Red) — the One Red Rule below still applies and is still unresolved; Glass Terminal doesn't fix it, just doesn't make it worse.
- **Risk Tier Ramp** — three two-stop gradients keyed to `calculate_dev_risk_score`'s low/medium/high bands (unchanged, still the one pre-existing load-bearing gradient): Low `#11998E → #38EF7D`, Medium `#F2994A → #F2C94C`, High `#EB3349 → #F45C43`.
- **Recommendation Ramp** — three two-stop gradients keyed to the Portfolio tab's buy/hold/sell logic (newly defined; previously referenced by class name but undefined in CSS): Buy `#0F8A3C → #0A6B2E`, Sell `#C2273D → #8E1B2D`, Hold `#5B6472 → #3D434D`. Same rule as the risk ramp — the gradient *is* the signal (direction of the call), not decoration.

### Ambient mesh (new, decorative by design — see Named Rules)
- **Ambient Mesh Blue** (`rgba(46, 84, 138, 0.16)`), **Ambient Mesh Teal** (`rgba(20, 110, 100, 0.13)`), **Ambient Mesh Red** (`rgba(255, 75, 75, 0.05)`): three very-low-opacity radial gradients layered behind the existing grid/sparkline SVG texture on `.stApp`. This is the one place color is allowed to be purely atmospheric rather than load-bearing — see the Ambient Exception below.

### Deprecated (documented so they can be removed, not reused)
- **Deprecated Header Gradient** (`#2E3192 → #1BFFFF`) and **Deprecated Metric Gradient** (`#667EEA → #764BA2`): the original gradient-clip-text masthead and purple `.metric-card` fill. Both already removed from the codebase. Never reintroduce gradient-clip text, and never reintroduce a card gradient that doesn't encode a real signal.
- **Dead Insight Colors** (`#F8F9FA` bg / `#667EEA` border): `.insight-box` was dead CSS (side-stripe-border anti-pattern), already removed.

### Named Rules
**The Risk Tier Rule.** Saturated, full-strength gradient *fills* (the kind that carry the entire visual weight of a card) may only encode the things that already have meaning in this codebase: gain/loss direction, risk tier, VaR breach/no-breach, factor sign, recommendation (buy/hold/sell). If a fill gradient isn't standing in for one of those, it's decoration and it's not allowed.

**The Ambient Exception.** The one deliberate exception to the Risk Tier Rule: the page-level ambient mesh and the low-opacity accent layers inside glass panels (hero panel backdrop, KPI card corner sheen, chart-frame top hairline) are allowed to be purely atmospheric, because they carry no specific data value and never claim to — they're materially honest "this is glass/depth," not "this number is good." The tell that separates this from AI-slop: these accents stay under ~15% opacity, use only the two established hues (Signal Red, tonal blue/teal), and never appear as a card's primary fill.

**The One Red Rule.** Signal Red (`#FF4B4B`) and Loss Red (`#FF0000`) are still two unrelated reds doing two unrelated jobs. Unresolved by this redesign — flagged for whoever next touches either one to converge deliberately rather than by accident.

## 3. Typography

**Body/UI Font:** Source Sans Pro (Streamlit's built-in default stack, with system sans-serif fallback)
**Display Font:** none — `.main-header` is a heavier weight/size override (`2.6rem`, `800`, `-0.01em` tracking) of the same Source Sans Pro stack, not a distinct typeface.

### Hierarchy
- **Display** (800, 2.6rem, `-0.01em` tracking): the `.main-header` masthead only — "📊 Stock Market Analysis Platform," rendered as solid Ink White inside the hero glass panel. Never gradient-clip text, even though the panel behind it is glass/gradient.
- **Title** (Streamlit default `st.header`/`st.subheader` sizing, not overridden): the per-tab section headers.
- **Body** (1rem, 400): dataframe cells, banner text, insight bullets.
- **Caption** (1rem, 400, `rgba(250,250,250,0.6)`): the `.sub-header` byline — contrast-fixed from the original `#666` (see Do's).

### Named Rules
**The Inherited-Until-Earned Rule.** Don't introduce a second typeface or a new display size to "make a tab feel special." One display style exists (the masthead); everything else borrows Streamlit's native `st.header`/`st.subheader`/`st.metric` hierarchy.

## 4. Elevation

Glass introduces a second elevation tier alongside the original flat one — the system now has two deliberate, named layers instead of one:

1. **Flat base** (unchanged from Quant Terminal): tab panels, `st.metric` tiles, the sidebar body. No shadow at rest. This is still the *majority* of the UI by area.
2. **Glass elevated** (new): the hero panel, KPI tiles, ticker strip, skeleton loaders, risk score card, recommendation card. These are permanently, intentionally elevated — translucent surface, blur, border, and a resting shadow (`--shadow-glass`) — because they're framed as floating instrument readouts, not page furniture. They still respond to hover/focus with a stronger lift, same as before.

Chart and dataframe frames (`[data-testid="stPlotlyChart"]`, `[data-testid="stDataFrame"]`) sit in between: a flat rgba tint + border at rest (cheap, no blur), with the same hover-lift behavior as Flat base elements.

### Shadow Vocabulary
- **hover-lift** (`0 2px 8px rgba(0,0,0,0.4)`): chart/dataframe/metric hover state.
- **overlay-lift** (`0 8px 32px rgba(0,0,0,0.5)`): any future modal, popover, or dropdown.
- **glass** (`0 10px 30px rgba(0,0,0,0.38)`, paired with `inset 0 1px 0 rgba(255,255,255,0.07)`): resting shadow + inner highlight for every Glass-elevated surface. This is the one shadow token allowed to appear at rest rather than only on interaction — Glass-elevated surfaces are the named exception to the old Flat-at-Rest Rule, not a violation of it.

### Named Rules
**The Flat-at-Rest Rule (narrowed).** Flat-base and chart/dataframe-frame elements still carry no shadow at rest; shadows there only appear on hover/focus. Glass-elevated surfaces are exempt by design — they're a different material, not an exception being snuck in per-component.

## 5. Components

### Hero Panel (new)
- **Shape:** `border-radius: 20px` (`--radius-xl`, the one new radius token — used only here, not invented per-tab), `padding: 2rem 2rem 1.5rem`.
- **Surface:** `--glass-surface` + `blur(--blur-lg)`, bordered, with a low-opacity two-tone radial accent (Signal Red top-left, teal bottom-right) behind the text and a 1px gradient hairline along the bottom edge.
- **Content:** masthead (solid Ink White) + byline. No gradient text, ever.

### KPI Price Tiles (Overview tab)
- **Shape:** `border-radius: 10px`, glass surface (`--blur-sm`), corner sheen accent (a 140×80px low-opacity radial highlight, Signal Red, top-left corner only).
- **Content:** symbol, count-up price, count-up percent change colored by direction (Gain Green / Loss Red). Rendered inside `components.html` (a real iframe), so the count-up JS and the `prefers-reduced-motion` gate both run client-side independent of Streamlit's own rerun cycle.

### Risk Score Card (Risk Analysis tab)
- **Shape:** `border-radius: 15px`, `padding: 1.5rem`.
- **Color assignment:** full-strength gradient fill from the Risk Tier Ramp, chosen by `calculate_dev_risk_score`'s 35/60 thresholds — unchanged, the gradient *is* the data. Glass Terminal adds a frosted edge (border + inset highlight + `--shadow-glass`) on top of the existing fill, not a blur (the fill is opaque, so blur would do nothing).
- **Internal content:** score out of 100, risk-level label, one-line methodology caption at `0.9rem`.

### Recommendation Card (Portfolio tab)
- **Shape:** `border-radius: 10px`, `padding: 1.25rem 1.5rem`.
- **Color assignment:** full-strength gradient fill from the new Recommendation Ramp (buy/sell/hold), same frosted-edge treatment as the risk card. Previously these three classes were referenced in markup but undefined in CSS (rendered as plain unstyled divs) — now defined.

### Ticker Strip & Live Status Badge
- **Ticker:** thin glass strip (`--blur-sm`), pauses on hover (also satisfies WCAG 2.2.2 for moving content). Built from already-fetched data, no extra network calls.
- **Live status dot:** encodes a real state (live feed vs. simulated), not decoration — pulses only when genuinely live.

### Native Streamlit Components (inherited, lightly styled)
- **Tabs** (`st.tabs`, 9 panels): active tab now gets a faint glass-pill background + Signal Red underline glow instead of an instant color snap.
- **Metrics** (`st.metric`): native delta-colored value display; hover gets a faint background tint + lift.
- **Banners** (`st.info`/`st.success`/`st.warning`/`st.error`): unchanged, native Panel Charcoal callouts — still the right component for new contextual content.
- **DataFrames / Plotly charts**: flat tinted glass *frame* (border + rgba tint + hover lift + a 2px gradient top hairline), no blur. No shared Plotly color template is set — each chart still sets its own `colorscale`/`color_discrete_sequence` independently; out of scope for this pass.

## 6. Do's and Don'ts

### Do:
- **Do** keep the dark Canvas Black / Panel Charcoal / Ink White base, committed via `.streamlit/config.toml`.
- **Do** keep full-strength gradient fills reserved for cards where the gradient *is* the signal (risk tier, recommendation) — per the Risk Tier Rule.
- **Do** keep the ambient mesh and corner-sheen accents under ~15% opacity, restricted to the two established hues — per the Ambient Exception.
- **Do** use `backdrop-filter` blur only on the handful of Glass-elevated surfaces (hero, KPI tiles, ticker, skeletons, risk/recommendation card edges) — never on the dozens of chart/dataframe frames.
- **Do** keep headline text solid Ink White, even on gradient/glass backgrounds — the gradient is the surface, never the text.
- **Do** provide a `prefers-reduced-transparency` fallback (solid `--glass-surface-strong`, no blur) and a `prefers-reduced-motion` fallback (no animation/transition) for every new glass/motion rule.
- **Do** use the native `st.info`/`st.success`/`st.warning`/`st.error` banners for any new contextual callout.
- **Do** extend the existing token set (`--radius-*`, `--blur-*`, `--shadow-*`, `--ease-out-quint`) rather than inventing new ad hoc values.

### Don't:
- **Don't** use gradient-clip-text headings (`-webkit-background-clip: text`) anywhere — this is the one rule that survived unchanged from Quant Terminal and the line that separates Glass Terminal from AI-SaaS slop.
- **Don't** use saturated purple/violet as an accent or ambient hue — the ambient mesh is tonal blue/teal/Signal-Red specifically to avoid the "2021 SaaS gradient" tell.
- **Don't** apply `backdrop-filter` blur to high-count elements (per-chart frames, per-row dataframe cells, per-metric tiles in dense grids) — flat rgba tint only, for performance.
- **Don't** let a card's gradient fill exceed the Risk Tier Rule's scope (gain/loss, risk tier, VaR breach, factor sign, recommendation). New decorative-only card gradients are not allowed.
- **Don't** introduce a second display typeface or a new heading size "for emphasis" — see the Inherited-Until-Earned Rule.
- **Don't** add a new one-off card style (a new radius, a new blur value, a new padding scale) per tab. The token set is now four radii (`sm` 5px / `md` 10px / `lg` 15px / `xl` 20px), two blur tiers (`sm` 10px / `lg` 22px), three spacing values (`sm`/`md`/`lg`), and three shadow tokens (`hover`/`overlay`/`glass`) — extend these, don't add a fifth/third/fourth.
