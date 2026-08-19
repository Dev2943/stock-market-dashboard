"""
latency_engine.py — live telemetry from the C++ order book engine.

Two modes, chosen automatically:

  LIVE      A bridge is reachable on localhost. Used when you run the dashboard
            on the same machine as the engine.

  RECORDED  No bridge. Replays engine_session.json, captured from real hardware
            by capture_session.py. This is what the Cloud Run deployment shows.

The engine is deliberately *not* deployed alongside the dashboard. Its numbers
only mean something on dedicated cores; the same binary on a shared, throttled
cloud vCPU reports tens of microseconds, and a viewer would reasonably conclude
the engine is slow. A recording from real hardware, labelled as a recording, is
more truthful than a live feed from hardware that cannot do the thing being
measured.

Streamlit also reruns the whole script on every interaction, so it can never be
driven by a 30 Hz feed directly. Metrics come from one frame per rerun; the
continuously-updating panel is an iframe that owns its own connection.
"""

import json
import os
import time
import urllib.error
import urllib.request

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ENGINE_URL = os.environ.get("ENGINE_URL", "http://localhost:8080")
SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "engine_session.json")
HIST_BUCKETS = 80

# use_container_width is deprecated after 2025-12-31 in favour of width, but
# older Streamlit builds don't accept width. Pick whichever this install
# supports so the tab keeps working across versions rather than warning on new
# ones and breaking on old ones.
try:
    from streamlit import __version__ as _ST_VER
    _WIDTH = ({"width": "stretch"}
              if tuple(int(x) for x in _ST_VER.split(".")[:2]) >= (1, 49)
              else {"use_container_width": True})
except Exception:
    _WIDTH = {"use_container_width": True}


# ---------------------------------------------------------------- transport

def _bucket_ns(i: int) -> int:
    """Lower edge of histogram bucket i, mirroring LatencyHistogram in C++.

    20 octaves x 4 linear sub-buckets: resolution stays within 25% of the value
    from nanoseconds to milliseconds.
    """
    if i < 4:
        return i
    octave, sub = i // 4 + 1, i % 4
    return (4 + sub) * (2 ** (octave - 2))


def _fetch_live(timeout: float = 1.5):
    """Read one frame off the SSE stream, then disconnect.

    Not cached: a stale latency figure is worse than none. One short round trip
    per rerun is negligible beside the yfinance calls this dashboard makes.
    """
    try:
        req = urllib.request.Request(
            f"{ENGINE_URL}/stream", headers={"Accept": "text/event-stream"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for _ in range(40):
                line = resp.readline()
                if not line:
                    break
                if line.startswith(b"data: "):
                    return json.loads(line[6:].decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    return None


@st.cache_data(show_spinner=False)
def _load_session():
    """The recording is immutable, so caching it is safe and worth it."""
    try:
        with open(SESSION_FILE) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def get_frame():
    """Return (frame, mode, session). mode is 'live', 'recorded' or 'none'."""
    live = _fetch_live()
    if live is not None:
        return live, "live", None

    session = _load_session()
    if session and session.get("frames"):
        frames = session["frames"]
        # Walk the recording on wall-clock time so successive reruns advance
        # rather than always showing frame zero.
        idx = int(time.time() * 2) % len(frames)
        return frames[idx], "recorded", session
    return None, "none", None


# ------------------------------------------------------------------ display

def _fmt_ns(ns) -> str:
    if ns is None:
        return "—"
    ns = float(ns)
    if ns < 1000:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns/1000:.1f} µs"
    return f"{ns/1e6:.2f} ms"


def _latency_figure(hist: dict, title: str, accent: str = "#38ef7d"):
    """Log-log histogram with percentile markers.

    Counts span several orders of magnitude, so the y-axis is log as well. A
    linear y would flatten the entire tail onto zero, and the tail is the part
    worth looking at.
    """
    buckets = hist.get("buckets") or []
    xs, ys = [], []
    for i, count in enumerate(buckets):
        if count:
            xs.append(max(_bucket_ns(i), 1))
            ys.append(count)

    fig = go.Figure()
    if xs:
        fig.add_trace(go.Bar(
            x=xs, y=ys, marker_color=accent, marker_line_width=0,
            hovertemplate="%{x:,.0f} ns<br>%{y:,} samples<extra></extra>",
        ))
    for label, key, colour in (("p50", "p50", "#FAFAFA"),
                               ("p99", "p99", "#f2c94c"),
                               ("p99.9", "p999", "#FF4B4B")):
        v = hist.get(key)
        if v:
            fig.add_vline(x=v, line_dash="dash", line_color=colour, line_width=1,
                          annotation_text=f"{label} {_fmt_ns(v)}",
                          annotation_position="top",
                          annotation_font_color=colour, annotation_font_size=11)
    fig.update_layout(
        title=title, template="plotly_dark", height=340,
        margin=dict(l=10, r=10, t=56, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(type="log", title="latency (ns, log scale)", gridcolor="#2A2E39"),
        yaxis=dict(type="log", title="samples", gridcolor="#2A2E39"),
        bargap=0.15,
    )
    return fig


def _provenance(mode: str, session: dict):
    if mode == "live":
        st.success("**Live** — reading the engine running on this machine.")
        return
    hw = session.get("hardware", "unknown hardware")
    p, e = session.get("performance_cores"), session.get("efficiency_cores")
    cores = f", {p}P/{e}E cores" if p else ""
    when = (session.get("captured_at") or "")[:10]
    st.info(
        f"**Recorded session** — {session.get('frame_count', 0)} frames captured "
        f"on {hw}{cores}"
        + (f" on {when}" if when else "")
        + ". These are real measurements replayed, not simulated numbers.\n\n"
        "The engine isn't deployed alongside this dashboard on purpose. Its "
        "figures only mean something on dedicated cores — the same binary on a "
        "shared cloud vCPU reports tens of microseconds, which would say more "
        "about the hypervisor than the engine. Run it locally to see it live."
    )


def render_latency_engine_tab():
    """Body of the Latency Engine tab. Safe when no engine and no recording."""
    st.header("⚡ Low-Latency Order Book Engine")
    st.caption(
        "A C++20 market data feed handler and price-time-priority order book. "
        "Every figure below is measured by the engine itself."
    )

    frame, mode, session = get_frame()

    if frame is None:
        st.warning("No engine running and no recorded session found.")
        st.markdown(
            """
**To run it live**, from the order-book project directory:

```bash
make
node web/bridge.js
```

**To ship a recording** with the deployed dashboard, capture one on the machine
the engine performs on and commit it next to `dashboard.py`:

```bash
python3 capture_session.py --seconds 30
```
            """
        )
        return

    _provenance(mode, session)

    t2b = frame.get("tick_to_book", {}) or {}
    svc = frame.get("service", {}) or {}

    if frame.get("saturated"):
        st.warning(
            "**Queue saturated.** The feed is arriving faster than the book can "
            "drain it, so tick-to-book reflects queue depth rather than engine "
            "speed. Restart at a lower rate: `RATE=4 node web/bridge.js`"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Throughput", f"{frame.get('mps', 0)/1e6:.2f} M msg/s")
    c2.metric("Tick to book (p50)", _fmt_ns(t2b.get("p50")),
              help="Bytes arriving to order book updated, end to end.")
    c3.metric("Book apply (p50)", _fmt_ns(svc.get("p50")),
              help="The engine's own cost, with queueing excluded.")
    c4.metric("Resting orders", f"{frame.get('live_orders', 0):,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Messages processed", f"{frame.get('applied', 0)/1e6:.1f} M")
    c6.metric("Updates published", f"{frame.get('publishes', 0):,}",
              help="Only messages that actually moved the best bid or ask.")
    c7.metric("Publish suppression", f"{frame.get('suppress_pct', 0):.2f}%",
              help="Messages that left the touch price and size unchanged.")
    stale = frame.get("stale_refs", 0)
    c8.metric("Stale references", f"{stale:,}",
              help="References to orders the book no longer holds. Should be zero.")

    st.markdown("---")
    left, right = st.columns(2)
    # Explicit keys: Streamlit derives an element's internal ID from its type
    # and parameters, and these two charts are structurally similar enough to
    # collide. Without keys the second one raises a duplicate-ID error.
    with left:
        st.plotly_chart(_latency_figure(t2b, "Tick to book", "#38ef7d"),
                        key="lat_engine_tick_to_book", **_WIDTH)
    with right:
        st.plotly_chart(_latency_figure(svc, "Book apply time", "#11998e"),
                        key="lat_engine_service_time", **_WIDTH)
    st.caption(
        f"Distributions cover the last {frame.get('window_s', 0):.2f} s "
        f"({t2b.get('n', 0):,} samples) rather than the whole session, so they "
        "track what the engine is doing now."
    )

    books = frame.get("books") or []
    if books:
        st.subheader("Top of book")
        for col, b in zip(st.columns(len(books)), books):
            with col:
                st.markdown(f"**SYM {b.get('sym')}**")
                st.markdown(
                    "<div style='font-family:monospace;font-size:1.25rem;line-height:1.5'>"
                    f"<span style='color:#38ef7d'>{b.get('bid', -1):,}</span>"
                    f"<span style='color:#8B92A0;font-size:0.8rem'> × {b.get('bid_qty',0):,}</span><br>"
                    f"<span style='color:#f2994a'>{b.get('ask', -1):,}</span>"
                    f"<span style='color:#8B92A0;font-size:0.8rem'> × {b.get('ask_qty',0):,}</span>"
                    "</div>"
                    f"<div style='color:#8B92A0;font-size:0.75rem;margin-top:0.25rem'>"
                    f"spread {b.get('spread', -1)}</div>",
                    unsafe_allow_html=True,
                )

    if mode == "live":
        st.markdown("---")
        # The iframe holds its own EventSource, so it updates ~30x/sec without
        # triggering a Streamlit rerun. Off by default: it keeps a connection
        # open for as long as it's rendered.
        if st.checkbox("Show live view (updates continuously)", value=False):
            components.iframe(ENGINE_URL, height=1150, scrolling=True)

    with st.expander("What this is measuring"):
        st.markdown(
            """
**Tick to book** is the whole path: bytes arrive, get decoded from a binary
ITCH-style wire format, cross a lock-free queue to the book thread, and the book
is updated. **Book apply** is only the last step. Keeping them apart matters —
a single blended number hides whether a slow result came from the book or from
the queue in front of it.

**Publish suppression** is the design decision that makes the feed usable. Most
messages are adds and cancels deep in the book that leave the best bid and ask
untouched, so publishing on each one would re-broadcast identical state. The
engine compares first and publishes only real changes, typically well under
0.1% of messages.

**A reader cannot slow the engine down.** Top of book is published through a
seqlock, which conflates: a consumer that falls behind sees the current book
rather than a backlog. If this page stalls or you close the tab, the engine
keeps running at full rate and simply drops frames — which is why a browser can
safely read the same publisher a trading strategy would.
            """
        )
