import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ORB Scanner — Nifty 50",
    page_icon="📈",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e1a;
    color: #e0e6f0;
}

.stApp { background-color: #0a0e1a; }

/* Header */
.orb-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0a192f 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.orb-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,200,150,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.orb-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #00e5a0;
    margin: 0;
    letter-spacing: -0.5px;
}
.orb-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #4a7fa5;
    margin-top: 6px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric Cards */
.metric-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 18px 24px;
    flex: 1;
    min-width: 140px;
}
.metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #4a7fa5;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #00e5a0;
}
.metric-card .value.warn { color: #f59e0b; }
.metric-card .value.danger { color: #f87171; }

/* Signal table */
.signal-table-wrap {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 0;
    overflow: hidden;
}
.signal-table-wrap table { border-collapse: collapse; width: 100%; }
.signal-table-wrap th {
    background: #112240;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4a7fa5;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 14px 16px;
    text-align: left;
    border-bottom: 1px solid #1e3a5f;
}
.signal-table-wrap td {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    padding: 12px 16px;
    border-bottom: 1px solid #0f2035;
    vertical-align: middle;
}
.signal-table-wrap tr:last-child td { border-bottom: none; }
.signal-table-wrap tr:hover td { background: #0f2035; }

/* Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    font-family: 'Space Mono', monospace;
}
.badge-green  { background: rgba(0,229,160,0.15); color: #00e5a0; border: 1px solid rgba(0,229,160,0.3); }
.badge-yellow { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-red    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

/* Filter sidebar */
section[data-testid="stSidebar"] {
    background: #0d1b2a;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] .css-1d391kg { padding-top: 2rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00e5a0, #00b07a);
    color: #0a0e1a;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    cursor: pointer;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #00e5a0, #00b07a); }

/* Divider */
hr { border-color: #1e3a5f; }

/* Info box */
.info-box {
    background: rgba(0,229,160,0.06);
    border: 1px solid rgba(0,229,160,0.2);
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #7ec8a4;
    line-height: 1.7;
    margin-bottom: 20px;
}
.warn-box {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px;
    padding: 12px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #f59e0b;
    margin-bottom: 14px;
}

/* Stock detail expander */
.detail-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-top: 10px; }
.detail-cell {
    background: #112240;
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #1e3a5f;
}
.detail-cell .d-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #4a7fa5; text-transform: uppercase; letter-spacing: 1px; }
.detail-cell .d-val   { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #e0e6f0; margin-top: 4px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Nifty 50 Symbols ────────────────────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN",
    "HINDUNILVR","ITC","BAJFINANCE","LT","KOTAKBANK","HCLTECH","MARUTI",
    "AXISBANK","ASIANPAINT","SUNPHARMA","TITAN","WIPRO","ONGC","NTPC","POWERGRID",
    "ULTRACEMCO","BAJAJFINSV","TECHM","ADANIPORTS","NESTLE","TATASTEEL","JSWSTEEL",
    "TATAMOTORS","INDUSINDBK","CIPLA","DRREDDY","HINDALCO","M&M",
    "BPCL","GRASIM","COALINDIA","EICHERMOT","SBILIFE","HDFCLIFE","DIVISLAB",
    "APOLLOHOSP","BAJAJ-AUTO","HEROMOTOCO","BRITANNIA","TATACONSUM",
    "ADANIENT","BEL","SHRIRAMFIN"
]

# ─── Helper Functions ────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute cumulative VWAP from intraday data."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tpv = (tp * df['Volume']).cumsum()
    cum_vol  = df['Volume'].cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)

def compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Relative Volume vs rolling mean."""
    rolling_avg = df['Volume'].rolling(window=window, min_periods=1).mean().shift(1)
    return df['Volume'] / rolling_avg.replace(0, np.nan)

def get_opening_range(df: pd.DataFrame, candles: int = 3) -> tuple:
    """Return (OR_High, OR_Low) based on first `candles` bars of the latest day."""
    if df.empty:
        return None, None
    latest_day = df.index.normalize().max()
    day_df = df[df.index.normalize() == latest_day].head(candles)
    if day_df.empty:
        return None, None
    return day_df['High'].max(), day_df['Low'].min()

def analyze_stock(ticker_ns: str, params: dict) -> dict | None:
    """
    Download 1-month 15m data, compute indicators, check ORB signal.
    Returns a result dict or None if data unavailable / no signal.
    """
    try:
        df = yf.download(
            ticker_ns, period="1mo", interval="15m",
            progress=False, auto_adjust=True
        )
        if df is None or df.empty or len(df) < 10:
            return None

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open','High','Low','Close','Volume']].copy()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)

        # ── Indicators ──────────────────────────────────────────────────────
        df['VWAP']      = compute_vwap(df)
        df['RVOL']      = compute_rvol(df, window=params['rvol_window'])
        df['VWAP_Dist'] = ((df['Close'] - df['VWAP']) / df['VWAP']) * 100

        # ── Opening Range (latest trading day) ──────────────────────────────
        or_high, or_low = get_opening_range(df, candles=params['or_candles'])
        if or_high is None:
            return None

        # Latest day candles
        latest_day = df.index.normalize().max()
        day_df = df[df.index.normalize() == latest_day].copy()
        if len(day_df) < params['or_candles'] + 2:
            return None

        # Post-OR candles
        post_or = day_df.iloc[params['or_candles']:]
        if post_or.empty:
            return None

        # ── Breakout Detection ───────────────────────────────────────────────
        breakout_idx = None
        for i in range(len(post_or)):
            if post_or['Close'].iloc[i] > or_high:
                breakout_idx = i
                break

        if breakout_idx is None:
            return None

        bo_row    = post_or.iloc[breakout_idx]
        bo_rvol   = bo_row['RVOL']
        bo_vwap_d = bo_row['VWAP_Dist']

        # ── Filter I: RVOL ───────────────────────────────────────────────────
        rvol_pass = (not np.isnan(bo_rvol)) and (bo_rvol >= params['min_rvol'])

        # ── Filter II: VWAP Distance ─────────────────────────────────────────
        vwap_pass = (not np.isnan(bo_vwap_d)) and (bo_vwap_d <= params['max_vwap_dist'])

        # ── Filter III: Momentum (volume-price divergence check) ─────────────
        if breakout_idx >= 1:
            prev_vol   = post_or['Volume'].iloc[breakout_idx - 1]
            curr_vol   = post_or['Volume'].iloc[breakout_idx]
            prev_close = post_or['Close'].iloc[breakout_idx - 1]
            curr_close = post_or['Close'].iloc[breakout_idx]
            momentum_pass = (curr_close > prev_close) and \
                            (curr_vol >= prev_vol * params['momentum_ratio'])
        else:
            momentum_pass = True  # first post-OR candle — no prior to compare

        # ── Pullback / Retest Check ──────────────────────────────────────────
        retest_pass = False
        retest_note = "No retest yet"
        if breakout_idx + 1 < len(post_or):
            retest_row = post_or.iloc[breakout_idx + 1]
            body_size  = abs(retest_row['Close'] - retest_row['Open'])
            atr_proxy  = (bo_row['High'] - bo_row['Low'])
            low_vol    = retest_row['Volume'] < bo_row['Volume'] * 0.6
            holds_orh  = retest_row['Close'] >= or_high * 0.995   # within 0.5%
            small_body = body_size <= atr_proxy * 0.5
            retest_pass = low_vol and holds_orh and small_body
            retest_note = "✓ Clean retest" if retest_pass else "✗ Weak / no retest"

        # ── Signal Score (0-4) ───────────────────────────────────────────────
        score = sum([rvol_pass, vwap_pass, momentum_pass, retest_pass])

        # ── Only return if at least 2 filters pass ───────────────────────────
        if score < 2:
            return None

        current_price = day_df['Close'].iloc[-1]
        or_pct_above  = ((current_price - or_high) / or_high) * 100

        return {
            "ticker":        ticker_ns.replace(".NS", ""),
            "score":         score,
            "current_price": round(float(current_price), 2),
            "or_high":       round(float(or_high), 2),
            "or_low":        round(float(or_low), 2),
            "or_pct_above":  round(float(or_pct_above), 2),
            "rvol":          round(float(bo_rvol), 2) if not np.isnan(bo_rvol) else 0,
            "rvol_pass":     rvol_pass,
            "vwap_dist":     round(float(bo_vwap_d), 2) if not np.isnan(bo_vwap_d) else 0,
            "vwap_pass":     vwap_pass,
            "momentum_pass": momentum_pass,
            "retest_pass":   retest_pass,
            "retest_note":   retest_note,
            "vwap_price":    round(float(bo_row['VWAP']), 2),
            "breakout_vol":  int(bo_row['Volume']),
        }

    except Exception:
        return None


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Scanner Settings")
    st.markdown("---")

    st.markdown("**Opening Range**")
    or_candles = st.slider("OR Candles (15m bars)", 1, 6, 3,
                           help="How many 15-min candles form the Opening Range")

    st.markdown("**Filter Thresholds**")
    min_rvol = st.slider("Min RVOL (Relative Volume)", 1.0, 5.0, 2.0, 0.1,
                         help="Breakout candle must show this multiple of avg volume")
    max_vwap_dist = st.slider("Max VWAP Distance (%)", 0.5, 4.0, 1.5, 0.1,
                               help="Reject if price is overextended beyond VWAP")
    momentum_ratio = st.slider("Momentum Vol Ratio", 0.7, 1.0, 0.9, 0.05,
                                help="2nd breakout candle vol ≥ this × 1st candle vol")

    st.markdown("**Stock Universe**")
    selected = st.multiselect("Select stocks (default = all Nifty 50)",
                               NIFTY50, default=[], placeholder="All 50 if empty")
    scan_list = selected if selected else NIFTY50

    st.markdown("---")
    st.markdown('<div class="info-box">📌 Data: yfinance 15m / 1 month<br>⏱ Scan may take 1–2 min for all 50<br>🔄 No TA-Lib / pandas-ta dependency</div>', unsafe_allow_html=True)
    run_scan = st.button("🚀 Run ORB Scan")

# ─── Main Panel ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="orb-header">
    <div class="orb-title">⚡ ORB Scanner — Nifty 50</div>
    <div class="orb-subtitle">Opening Range Breakout · VWAP · RVOL · Momentum · Retest Filter</div>
</div>
""", unsafe_allow_html=True)

# Legend
st.markdown("""
<div class="info-box">
<b>Signal Score:</b> Each stock is scored 0–4 across four independent filters.
&nbsp;🟢 Score 4 = All filters pass (Highest Confidence) &nbsp;|&nbsp;
🟡 Score 3 = Strong Setup &nbsp;|&nbsp; 🟠 Score 2 = Marginal — review manually
</div>
""", unsafe_allow_html=True)

if not run_scan:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #4a7fa5;">
        <div style="font-size:3rem;">📊</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.9rem; margin-top:12px;">
            Configure filters in the sidebar, then click <b style="color:#00e5a0;">Run ORB Scan</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Run Scan ─────────────────────────────────────────────────────────────────────
params = {
    "or_candles":     or_candles,
    "min_rvol":       min_rvol,
    "max_vwap_dist":  max_vwap_dist,
    "momentum_ratio": momentum_ratio,
    "rvol_window":    20,
}

results      = []
errors       = []
progress_bar = st.progress(0, text="Initializing scan…")
status_text  = st.empty()

for idx, sym in enumerate(scan_list):
    ticker_ns = f"{sym}.NS"
    status_text.markdown(
        f'<span style="font-family:Space Mono,monospace;font-size:0.78rem;color:#4a7fa5;">'
        f'Scanning {ticker_ns}… ({idx+1}/{len(scan_list)})</span>',
        unsafe_allow_html=True
    )
    res = analyze_stock(ticker_ns, params)
    if res:
        results.append(res)
    else:
        errors.append(sym)
    progress_bar.progress((idx + 1) / len(scan_list),
                           text=f"Scanned {idx+1}/{len(scan_list)}")

progress_bar.empty()
status_text.empty()

# ─── Summary Metrics ──────────────────────────────────────────────────────────────
total        = len(scan_list)
signals      = len(results)
high_conf    = sum(1 for r in results if r['score'] == 4)
strong       = sum(1 for r in results if r['score'] == 3)
marginal     = sum(1 for r in results if r['score'] == 2)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Stocks Scanned</div>
        <div class="value">{total}</div>
    </div>
    <div class="metric-card">
        <div class="label">ORB Signals</div>
        <div class="value {'warn' if signals==0 else ''}">{signals}</div>
    </div>
    <div class="metric-card">
        <div class="label">🟢 Score 4</div>
        <div class="value">{high_conf}</div>
    </div>
    <div class="metric-card">
        <div class="label">🟡 Score 3</div>
        <div class="value warn">{strong}</div>
    </div>
    <div class="metric-card">
        <div class="label">🟠 Score 2</div>
        <div class="value danger">{marginal}</div>
    </div>
    <div class="metric-card">
        <div class="label">No Signal</div>
        <div class="value danger">{total - signals}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not results:
    st.markdown("""
    <div class="warn-box">
    ⚠️ No ORB signals found with current filter settings.
    Try lowering the Min RVOL threshold or relaxing the VWAP Distance limit.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Sort Results ─────────────────────────────────────────────────────────────────
results.sort(key=lambda x: (-x['score'], -x['rvol']))

# ─── Results Table ────────────────────────────────────────────────────────────────
def score_badge(score):
    if score == 4:
        return f'<span class="badge badge-green">★★★★ {score}/4</span>'
    elif score == 3:
        return f'<span class="badge badge-yellow">★★★☆ {score}/4</span>'
    else:
        return f'<span class="badge badge-red">★★☆☆ {score}/4</span>'

def check(val):
    return "✅" if val else "❌"

rows_html = ""
for r in results:
    rows_html += f"""
    <tr>
        <td><b style="color:#e0e6f0;font-family:'Syne',sans-serif;">{r['ticker']}</b></td>
        <td>{score_badge(r['score'])}</td>
        <td style="color:#e0e6f0;">₹{r['current_price']}</td>
        <td style="color:#4a7fa5;">₹{r['or_high']}</td>
        <td style="color:{'#00e5a0' if r['or_pct_above']>=0 else '#f87171'};">
            {'+' if r['or_pct_above']>=0 else ''}{r['or_pct_above']}%
        </td>
        <td style="color:{'#00e5a0' if r['rvol_pass'] else '#f87171'};">{r['rvol']}x {check(r['rvol_pass'])}</td>
        <td style="color:{'#00e5a0' if r['vwap_pass'] else '#f87171'};">{r['vwap_dist']}% {check(r['vwap_pass'])}</td>
        <td>{check(r['momentum_pass'])}</td>
        <td style="font-size:0.72rem;color:#7ec8a4;">{r['retest_note']}</td>
    </tr>
    """

st.markdown(f"""
<div class="signal-table-wrap">
<table>
<thead>
<tr>
    <th>Ticker</th>
    <th>Score</th>
    <th>Price (₹)</th>
    <th>OR High</th>
    <th>% Above OR</th>
    <th>RVOL</th>
    <th>VWAP Dist</th>
    <th>Momentum</th>
    <th>Retest</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ─── Detailed Drill-Down ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔍 Stock Detail Drill-Down")

ticker_choices = [r['ticker'] for r in results]
selected_ticker = st.selectbox("Select a stock for details", ticker_choices)

if selected_ticker:
    det = next(r for r in results if r['ticker'] == selected_ticker)
    st.markdown(f"""
    <div class="detail-grid">
        <div class="detail-cell">
            <div class="d-label">Current Price</div>
            <div class="d-val">₹{det['current_price']}</div>
        </div>
        <div class="detail-cell">
            <div class="d-label">OR High</div>
            <div class="d-val">₹{det['or_high']}</div>
        </div>
        <div class="detail-cell">
            <div class="d-label">OR Low</div>
            <div class="d-val">₹{det['or_low']}</div>
        </div>
        <div class="detail-cell">
            <div class="d-label">VWAP at Breakout</div>
            <div class="d-val">₹{det['vwap_price']}</div>
        </div>
        <div class="detail-cell">
            <div class="d-label">Relative Volume</div>
            <div class="d-val" style="color:{'#00e5a0' if det['rvol_pass'] else '#f87171'};">{det['rvol']}x</div>
        </div>
        <div class="detail-cell">
            <div class="d-label">VWAP Distance</div>
            <div class="d-val" style="color:{'#00e5a0' if det['vwap_pass'] else '#f87171'};">{det['vwap_dist']}%</div>
        </div>
        <div class="detail-cell">
            <div class="d-label">Breakout Volume</div>
            <div class="d-val">{det['breakout_vol']:,}</div>
        </div>
        <div class="detail-cell">
            <div class="d-label">% Above OR High</div>
            <div class="d-val" style="color:#00e5a0;">+{det['or_pct_above']}%</div>
        </div>
        <div class="detail-cell">
            <div class="d-label">Signal Score</div>
            <div class="d-val" style="color:#00e5a0;">{det['score']} / 4</div>
        </div>
    </div>
    <br>
    <div class="info-box">
    <b>Filter Summary for {det['ticker']}</b><br>
    {'✅' if det['rvol_pass'] else '❌'} RVOL ≥ {min_rvol}x &nbsp;|&nbsp;
    {'✅' if det['vwap_pass'] else '❌'} VWAP Dist ≤ {max_vwap_dist}% &nbsp;|&nbsp;
    {'✅' if det['momentum_pass'] else '❌'} Momentum Confirmed &nbsp;|&nbsp;
    {'✅' if det['retest_pass'] else '❌'} {det['retest_note']}
    </div>
    """, unsafe_allow_html=True)

# ─── Failed stocks note ───────────────────────────────────────────────────────────
if errors:
    with st.expander(f"⚠️ {len(errors)} stocks with no data / no signal"):
        st.markdown(
            f'<span style="font-family:Space Mono,monospace;font-size:0.78rem;color:#4a7fa5;">'
            f'{", ".join(errors)}</span>',
            unsafe_allow_html=True
        )

# ─── Disclaimer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:#2d5070;text-align:center;padding:10px;">
⚠️ FOR EDUCATIONAL & RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE. ALWAYS DO YOUR OWN DUE DILIGENCE.
</div>
""", unsafe_allow_html=True)
