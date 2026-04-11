"""
Elliott Wave Trader — Streamlit App  v3
========================================
Sidebar:   Global config (EW params, SL/TP mode, interval)
Tab 1:     📡 Scanner  — live multi-ticker signal scan + one-click Apply & Trade
Tab 2:     📊 Backtest
Tab 3:     🔴 Live Trading — continuous loop with 1.5 s throttle + st.rerun()
Tab 4:     ⚙️  Optimization — grid search with min-accuracy filter

Fixes over v2:
  • No more AttributeError — _fetch_live() calls yf.download() directly
  • confirm_idx placement (no lookahead / repainting)
  • live_mode drops incomplete last candle
  • Continuous live trading via st.rerun() when lv_running=True
  • SL/TP in % OR absolute (sidebar toggle)
  • All times displayed in IST
  • Shared config from sidebar flows into all tabs + optimization
  • Scanner: 12 tickers, IST signal time, time-since-signal,
    remaining-to-target, remaining-to-SL, "Apply & Trade" button
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import itertools
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# TIMEZONE
# ─────────────────────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> datetime:
    return datetime.now(IST)

def fmt_ist(dt) -> str:
    """Convert any datetime-like to IST string HH:MM:SS DD-Mon-YY."""
    if dt is None:
        return "—"
    if isinstance(dt, str):
        try:
            dt = pd.Timestamp(dt)
        except Exception:
            return str(dt)
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(IST)
        return dt.strftime("%H:%M:%S  %d-%b-%y")
    return str(dt)

def df_index_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with DatetimeIndex converted to IST-aware."""
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df = df.copy()
            df.index = df.index.tz_convert(IST)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TICKER UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
SCANNER_TICKERS = [
    ("^NSEI",       "Nifty 50"),
    ("^NSEBANK",    "Bank Nifty"),
    ("^BSESN",      "Sensex"),
    ("BTC-USD",     "BTC/USD"),
    ("ETH-USD",     "ETH/USD"),
    ("USDINR=X",    "USD/INR"),
    ("GC=F",        "Gold"),
    ("SI=F",        "Silver"),
    ("INFY.NS",     "Infosys"),
    ("RELIANCE.NS", "Reliance"),
    ("KAYNES.NS",   "Kaynes Tech"),
    ("HDFCBANK.NS", "HDFC Bank"),
]

TICKER_GROUPS = {
    "── Indices ──": [("^NSEI","Nifty 50"),("^NSEBANK","Bank Nifty"),("^BSESN","Sensex")],
    "── Nifty 50 Stocks ──": [
        ("RELIANCE.NS","Reliance"),("TCS.NS","TCS"),("HDFCBANK.NS","HDFC Bank"),
        ("INFY.NS","Infosys"),("ICICIBANK.NS","ICICI Bank"),
        ("HINDUNILVR.NS","HUL"),("BAJFINANCE.NS","Bajaj Finance"),
        ("BHARTIARTL.NS","Airtel"),("KOTAKBANK.NS","Kotak Bank"),("ITC.NS","ITC"),
        ("LT.NS","L&T"),("HCLTECH.NS","HCL Tech"),("ASIANPAINT.NS","Asian Paints"),
        ("AXISBANK.NS","Axis Bank"),("MARUTI.NS","Maruti"),("TITAN.NS","Titan"),
        ("SUNPHARMA.NS","Sun Pharma"),("ULTRACEMCO.NS","UltraCemco"),
        ("WIPRO.NS","Wipro"),("NESTLEIND.NS","Nestle"),("KAYNES.NS","Kaynes Tech"),
    ],
    "── Crypto ──": [
        ("BTC-USD","BTC/USD"),("ETH-USD","ETH/USD"),("BNB-USD","BNB/USD"),
        ("SOL-USD","SOL/USD"),("XRP-USD","XRP/USD"),
    ],
    "── Commodities ──": [
        ("GC=F","Gold"),("SI=F","Silver"),("CL=F","Crude Oil"),("NG=F","Nat Gas"),
    ],
    "── Forex ──": [
        ("USDINR=X","USD/INR"),("EURUSD=X","EUR/USD"),
        ("GBPUSD=X","GBP/USD"),("USDJPY=X","USD/JPY"),
    ],
    "── Custom ──": [("__CUSTOM__","Custom Ticker")],
}

INTERVALS   = ["1m","2m","5m","15m","30m","1h","1d"]
PERIODS_MAP = {
    "1m":"7d","2m":"60d","5m":"60d","15m":"60d",
    "30m":"60d","1h":"730d","1d":"max",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=90, show_spinner=False)
def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def _fetch_live(ticker: str, interval: str, period: str,
                throttle: float = 1.5) -> pd.DataFrame:
    """Uncached, always-fresh fetch. Sleeps throttle seconds BEFORE request."""
    time.sleep(throttle)
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_sl_tp(entry: float, side: int,
                  sl_val: float, tp_val: float, mode: str):
    """
    mode = "Percentage"  → sl_val / tp_val are % values
    mode = "Absolute"    → sl_val / tp_val are absolute price distances
    """
    if mode == "Percentage":
        sl = entry*(1-sl_val/100) if side==1 else entry*(1+sl_val/100)
        tp = entry*(1+tp_val/100) if side==1 else entry*(1-tp_val/100)
    else:
        sl = entry - sl_val if side==1 else entry + sl_val
        tp = entry + tp_val if side==1 else entry - tp_val
    return sl, tp


# ─────────────────────────────────────────────────────────────────────────────
# FIXED PIVOT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _ew_build_pivots(df: pd.DataFrame, min_wave_pct: float = 1.0,
                     live_mode: bool = False):
    cl = df["Close"]
    n  = len(cl)
    if live_mode and n > 1:
        n -= 1

    min_mv  = min_wave_pct / 100.0
    pivots  = []
    last_dir = 0
    last_px  = float(cl.iloc[0])
    last_idx = 0

    for i in range(1, n):
        p = float(cl.iloc[i])
        if last_dir == 0:
            if   p > last_px * (1 + min_mv): last_dir =  1
            elif p < last_px * (1 - min_mv): last_dir = -1
        elif last_dir == 1:
            if p < float(cl.iloc[last_idx]) * (1 - min_mv):
                pivots.append((last_idx, float(cl.iloc[last_idx]), 1, i))
                last_dir, last_px, last_idx = -1, p, i
            elif p > last_px:
                last_px, last_idx = p, i
        else:
            if p > float(cl.iloc[last_idx]) * (1 + min_mv):
                pivots.append((last_idx, float(cl.iloc[last_idx]), -1, i))
                last_dir, last_px, last_idx = 1, p, i
            elif p < last_px:
                last_px, last_idx = p, i

    return pivots, (last_dir, last_px, last_idx)


# ─────────────────────────────────────────────────────────────────────────────
# FIXED SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def sig_elliott_wave(df: pd.DataFrame, swing_lookback: int = 10,
                     min_wave_pct: float = 1.0,
                     live_mode: bool = False):
    n = len(df)
    s = pd.Series(0, index=df.index)
    pivots, _ = _ew_build_pivots(df, min_wave_pct, live_mode=live_mode)
    if swing_lookback and len(pivots) > swing_lookback:
        pivots = pivots[-swing_lookback:]
    for k in range(2, len(pivots)):
        p0, p1, p2 = pivots[k-2], pivots[k-1], pivots[k]
        c = p2[3]
        if c >= n: continue
        if p0[2]==-1 and p1[2]==1 and p2[2]==-1 and p2[1]>p0[1]:
            s.iloc[c] = 1
        elif p0[2]==1 and p1[2]==-1 and p2[2]==1 and p2[1]<p0[1]:
            s.iloc[c] = -1
    return s, {"EMA_20": ema(df["Close"], 20)}


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signals, sl_val, tp_val, sl_tp_mode, initial_capital=100_000.0):
    trades, capital, equity, position = [], initial_capital, [], None
    prices  = df[["Open","High","Low","Close"]].reset_index()
    sig_arr = signals.values

    for i in range(len(prices)):
        row = prices.iloc[i]
        hi, lo, op = float(row["High"]), float(row["Low"]), float(row["Open"])

        if position:
            pnl, closed, exit_px = 0.0, False, 0.0
            if position["side"] == 1:
                if lo <= position["sl"]:
                    pnl, exit_px, closed = position["sl"]-position["entry"], position["sl"], True
                elif hi >= position["tp"]:
                    pnl, exit_px, closed = position["tp"]-position["entry"], position["tp"], True
            else:
                if hi >= position["sl"]:
                    pnl, exit_px, closed = position["entry"]-position["sl"], position["sl"], True
                elif lo <= position["tp"]:
                    pnl, exit_px, closed = position["entry"]-position["tp"], position["tp"], True
            if closed:
                pct = pnl / position["entry"] * 100
                capital += pnl * (capital / position["entry"])
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time" : str(row.iloc[0]),
                    "side"      : "BUY" if position["side"]==1 else "SELL",
                    "entry"     : round(position["entry"],4),
                    "exit"      : round(exit_px,4),
                    "sl"        : round(position["sl"],4),
                    "tp"        : round(position["tp"],4),
                    "pnl_pct"   : round(pct,2),
                    "result"    : "WIN" if pnl>0 else "LOSS",
                })
                position = None

        if position is None and i > 0 and sig_arr[i-1] != 0:
            side  = int(sig_arr[i-1])
            entry = op
            sl, tp = compute_sl_tp(entry, side, sl_val, tp_val, sl_tp_mode)
            position = {"side":side,"entry":entry,"sl":sl,"tp":tp,
                        "entry_time":str(row.iloc[0])}
        equity.append(capital)

    return trades, pd.Series(equity, index=df.index), capital


def trade_stats(trades, initial_capital, final_capital):
    if not trades: return {}
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["result"]=="WIN"]
    loss = df_t[df_t["result"]=="LOSS"]
    wr   = len(wins)/len(df_t)*100
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    rr   = abs(aw/al) if al else float("inf")
    ret  = (final_capital-initial_capital)/initial_capital*100
    return {
        "Total Trades"   : len(df_t),
        "Win Rate %"     : round(wr,1),
        "Avg Win %"      : round(aw,2),
        "Avg Loss %"     : round(al,2),
        "Risk/Reward"    : round(rr,2),
        "Total Return %" : round(ret,2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TICKER SELECTOR WIDGET (for backtest / live / opt)
# ─────────────────────────────────────────────────────────────────────────────
def ticker_selector(key_prefix: str):
    display, values = [], []
    for grp, pairs in TICKER_GROUPS.items():
        display.append(grp); values.append(None)
        for sym, lbl in pairs:
            display.append(f"  {sym}  ({lbl})" if sym != "__CUSTOM__" else "  Custom Ticker")
            values.append(sym)
    default_idx = next((i for i,v in enumerate(values) if v=="BTC-USD"), 1)
    sel = st.selectbox("Instrument", range(len(display)),
                       format_func=lambda i: display[i],
                       index=default_idx, key=f"{key_prefix}_sel")
    chosen = values[sel]
    if chosen is None:
        st.warning("Select an instrument, not a group header.")
        return None, None
    if chosen == "__CUSTOM__":
        val = st.text_input("Ticker (yfinance format)", "HDFCBANK.NS",
                            key=f"{key_prefix}_cust")
        t = val.strip().upper()
        return t, t
    lbl_map = {sym:lbl for g,pairs in TICKER_GROUPS.items() for sym,lbl in pairs}
    return chosen, lbl_map.get(chosen, chosen)


# ═══════════════════════════════════════════════════════════════════════════
# ░░  STREAMLIT APP  ░░
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="EW Trader", layout="wide", page_icon="〰️",
                   initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Global Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Global Configuration")
    st.markdown("*These settings apply to all tabs.*")
    st.divider()

    st.markdown("### 📐 Elliott Wave")
    cfg_interval  = st.selectbox("Interval",        INTERVALS, index=3, key="cfg_iv")
    cfg_min_wave  = st.slider("Min Wave %",          0.3, 5.0, 1.0, 0.1, key="cfg_mw")
    cfg_lookback  = st.slider("Swing Lookback",      4,   30,  10,  1,   key="cfg_lb")

    st.divider()
    st.markdown("### 🛡️ Stop Loss / Target")
    cfg_sltp_mode = st.radio("SL / TP Mode",
                             ["Percentage", "Absolute"],
                             horizontal=True, key="cfg_mode")
    if cfg_sltp_mode == "Percentage":
        cfg_sl_val = st.slider("Stop Loss %",   0.1, 10.0, 1.5, 0.1, key="cfg_sl")
        cfg_tp_val = st.slider("Take Profit %", 0.1, 20.0, 3.0, 0.1, key="cfg_tp")
        cfg_sl_label = f"{cfg_sl_val}%"
        cfg_tp_label = f"{cfg_tp_val}%"
    else:
        cfg_sl_val = st.number_input("Stop Loss (abs points)", 0.01, 10000.0,
                                     50.0, step=0.5, key="cfg_sl_abs")
        cfg_tp_val = st.number_input("Take Profit (abs points)", 0.01, 10000.0,
                                     100.0, step=0.5, key="cfg_tp_abs")
        cfg_sl_label = f"{cfg_sl_val} pts"
        cfg_tp_label = f"{cfg_tp_val} pts"

    st.divider()
    st.markdown("### 💰 Capital")
    cfg_capital   = st.number_input("Capital (₹)", 1000, 10_000_000,
                                    100_000, step=10_000, key="cfg_cap")

    st.divider()
    st.markdown("### 📡 Scanner")
    cfg_scan_iv   = st.selectbox("Scanner Interval", ["1m","2m","5m","15m"],
                                 index=2, key="cfg_scan_iv")
    cfg_throttle  = st.slider("API Throttle (s)", 1.0, 5.0, 1.5, 0.5, key="cfg_thr")

    st.divider()
    st.markdown("### 🔴 Live Trading")
    cfg_lv_polls  = st.slider("Polls per Cycle", 1, 5, 2, key="cfg_polls")

    st.divider()
    st.caption(f"SL: **{cfg_sl_label}**  |  TP: **{cfg_tp_label}**  "
               f"|  {cfg_interval}  |  EW min {cfg_min_wave}%")

# Initialise session state
_live_defaults = {
    "lv_running"  : False,
    "lv_position" : None,
    "lv_capital"  : float(cfg_capital),
    "lv_init_cap" : float(cfg_capital),
    "lv_trades"   : [],
    "lv_log"      : [],
    "lv_poll_cnt" : 0,
    "lv_last_sig" : 0,
    "lv_last_px"  : None,
    "lv_ticker"   : "BTC-USD",
    "lv_label"    : "BTC/USD",
    "lv_rerun_after": False,
}
for k, v in _live_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
st.title("〰️  Elliott Wave Trader")
st.caption(
    f"Interval: **{cfg_interval}**  ·  "
    f"EW: min {cfg_min_wave}% / lookback {cfg_lookback}  ·  "
    f"SL {cfg_sl_label}  ·  TP {cfg_tp_label}  ·  "
    f"Capital ₹{cfg_capital:,.0f}  ·  "
    f"All times in **IST**"
)

tab_scan, tab_bt, tab_live, tab_opt = st.tabs(
    ["📡 Scanner", "📊 Backtest", "🔴 Live Trading", "⚙️ Optimization"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SCANNER
# ═══════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.subheader("📡 Live Signal Scanner")
    st.markdown(
        f"Scans **{len(SCANNER_TICKERS)}** instruments using sidebar config.  "
        f"Each request throttled **{cfg_throttle} s**.  "
        f"Est. scan time: **~{len(SCANNER_TICKERS)*cfg_throttle:.0f} s**."
    )

    # Custom ticker toggle
    with st.expander("➕ Add custom ticker to scan"):
        cust_sym  = st.text_input("Ticker symbol", "SBIN.NS", key="scan_cust_sym")
        cust_lbl  = st.text_input("Display label",  "SBI",    key="scan_cust_lbl")
        add_cust  = st.button("Add", key="scan_add_cust")

    if "scan_extra" not in st.session_state:
        st.session_state.scan_extra = []
    if add_cust and cust_sym.strip():
        entry_c = (cust_sym.strip().upper(), cust_lbl.strip() or cust_sym.strip().upper())
        if entry_c not in st.session_state.scan_extra:
            st.session_state.scan_extra.append(entry_c)

    all_scan_tickers = SCANNER_TICKERS + st.session_state.scan_extra

    col_run, col_clr = st.columns(2)
    run_scan  = col_run.button("🔍 Scan Now", type="primary", key="btn_scan")
    clr_scan  = col_clr.button("🗑️ Clear Results", key="btn_scan_clr")

    if clr_scan:
        st.session_state.pop("scan_results", None)

    if run_scan:
        period = PERIODS_MAP.get(cfg_scan_iv, "60d")
        prog   = st.progress(0, text="Starting scan…")
        scan_results = []

        for idx, (sym, lbl) in enumerate(all_scan_tickers):
            pct = int((idx / len(all_scan_tickers)) * 100)
            prog.progress(pct, text=f"[{idx+1}/{len(all_scan_tickers)}] "
                          f"Fetching {lbl} ({cfg_throttle} s throttle)…")
            try:
                df_s = _fetch_live(sym, cfg_scan_iv, period, throttle=cfg_throttle)
            except Exception as exc:
                scan_results.append({
                    "sym":sym, "label":lbl, "error":str(exc),
                    "sig":0, "sig_time":None, "price":None
                })
                continue

            if df_s.empty:
                scan_results.append({"sym":sym,"label":lbl,"error":"No data","sig":0,
                                      "sig_time":None,"price":None})
                continue

            df_s = df_index_to_ist(df_s)
            sigs, _ = sig_elliott_wave(
                df_s, cfg_lookback, cfg_min_wave, live_mode=True
            )

            # Find latest non-zero signal in last 5 bars
            recent_sigs = sigs.iloc[-6:-1]
            sig_val, sig_time = 0, None
            for t, v in reversed(list(recent_sigs.items())):
                if v != 0:
                    sig_val, sig_time = int(v), t
                    break

            cur_px = float(df_s["Close"].iloc[-2]) if len(df_s) > 1 else float(df_s["Close"].iloc[-1])

            sl, tp = compute_sl_tp(cur_px, sig_val if sig_val != 0 else 1,
                                   cfg_sl_val, cfg_tp_val, cfg_sltp_mode)
            scan_results.append({
                "sym"     : sym,
                "label"   : lbl,
                "error"   : None,
                "sig"     : sig_val,
                "sig_time": sig_time,
                "price"   : cur_px,
                "sl"      : sl,
                "tp"      : tp,
            })

        prog.progress(100, text="Scan complete!")
        time.sleep(0.3)
        prog.empty()
        st.session_state.scan_results = scan_results

    # ── Display scan results ──────────────────────────────────────────────
    if "scan_results" in st.session_state and st.session_state.scan_results:
        results = st.session_state.scan_results
        now     = now_ist()

        active_signals = [r for r in results if r["sig"] != 0 and not r.get("error")]
        no_signal      = [r for r in results if r["sig"] == 0 and not r.get("error")]
        errors         = [r for r in results if r.get("error")]

        st.markdown(f"### 🎯 Active Signals  ({len(active_signals)} found)")

        if not active_signals:
            st.info("No EW signals detected on current bars across scanned instruments.")
        else:
            for r in active_signals:
                sig_icon  = "🟢 BUY" if r["sig"]==1 else "🔴 SELL"
                sig_ts    = fmt_ist(r["sig_time"])
                cur_px    = r["price"]

                # Time elapsed
                if r["sig_time"] is not None:
                    try:
                        st_dt = r["sig_time"]
                        if hasattr(st_dt, "tzinfo") and st_dt.tzinfo:
                            elapsed = now - st_dt.astimezone(IST)
                        else:
                            elapsed = timedelta(0)
                        mins = int(elapsed.total_seconds() // 60)
                        secs = int(elapsed.total_seconds() % 60)
                        elapsed_str = f"{mins}m {secs}s ago"
                    except Exception:
                        elapsed_str = "—"
                else:
                    elapsed_str = "—"

                # Remaining distance
                if r["sig"] == 1:
                    rem_tp_abs = r["tp"] - cur_px
                    rem_sl_abs = cur_px - r["sl"]
                else:
                    rem_tp_abs = cur_px - r["tp"]
                    rem_sl_abs = r["sl"] - cur_px
                rem_tp_pct = rem_tp_abs / cur_px * 100 if cur_px else 0
                rem_sl_pct = rem_sl_abs / cur_px * 100 if cur_px else 0

                with st.container(border=True):
                    hc1, hc2 = st.columns([3, 1])
                    hc1.markdown(
                        f"#### {r['label']}  `{r['sym']}`  —  {sig_icon}"
                    )

                    # Apply & Trade button
                    apply_key = f"apply_{r['sym'].replace('.','_').replace('^','').replace('=','')}"
                    if hc2.button(f"▶ Apply & Trade", key=apply_key, type="primary"):
                        st.session_state.lv_ticker   = r["sym"]
                        st.session_state.lv_label    = r["label"]
                        st.session_state.lv_running  = True
                        st.session_state.lv_capital  = float(cfg_capital)
                        st.session_state.lv_init_cap = float(cfg_capital)
                        st.session_state.lv_trades   = []
                        st.session_state.lv_log      = []
                        st.session_state.lv_poll_cnt = 0
                        st.session_state.lv_position = None
                        st.success(
                            f"✅ Live trading started for **{r['label']}**. "
                            f"Switch to the **🔴 Live Trading** tab."
                        )

                    dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
                    dc1.metric("Current Price",  f"{cur_px:.4f}")
                    dc2.metric("Signal Time (IST)", sig_ts)
                    dc3.metric("Current Time (IST)", fmt_ist(now))
                    dc4.metric("Signal Age",     elapsed_str)
                    dc5.metric("🎯 Rem. to TP",
                               f"{rem_tp_abs:+.4f}",
                               delta=f"{rem_tp_pct:+.2f}%",
                               delta_color="normal")
                    dc6.metric("🛑 Rem. to SL",
                               f"{rem_sl_abs:.4f}",
                               delta=f"{rem_sl_pct:.2f}%",
                               delta_color="inverse")

                    st.caption(
                        f"SL: **{r['sl']:.4f}**  ·  TP: **{r['tp']:.4f}**  ·  "
                        f"Mode: {cfg_sltp_mode}"
                    )

        # Collapsible: no-signal list
        with st.expander(f"📭 No Signal  ({len(no_signal)} instruments)"):
            for r in no_signal:
                px_str = f"{r['price']:.4f}" if r["price"] else "—"
                st.caption(f"• **{r['label']}** `{r['sym']}`  —  price: {px_str}")

        if errors:
            with st.expander(f"⚠️ Fetch Errors ({len(errors)})"):
                for r in errors:
                    st.error(f"{r['label']} `{r['sym']}`: {r['error']}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("📊 Backtest — Elliott Wave (no lookahead)")
    st.caption("Uses sidebar config for EW params, SL/TP mode, and capital.")

    col1, col2 = st.columns(2)
    with col1:
        bt_ticker, bt_label = ticker_selector("bt")
    with col2:
        bt_period_override = st.selectbox(
            "Override period (optional)",
            ["(use default)", "7d","30d","60d","180d","1y","2y","max"],
            key="bt_period"
        )

    if st.button("▶ Run Backtest", key="run_bt") and bt_ticker:
        period = (PERIODS_MAP.get(cfg_interval, "60d")
                  if bt_period_override == "(use default)"
                  else bt_period_override)
        with st.spinner(f"Fetching {bt_label}…"):
            df = fetch_data(bt_ticker, cfg_interval, period)
        if df.empty:
            st.error("No data returned.")
        else:
            df = df_index_to_ist(df)
            sigs, indics = sig_elliott_wave(df, cfg_lookback, cfg_min_wave, live_mode=False)
            trades, equity, final = run_backtest(
                df, sigs, cfg_sl_val, cfg_tp_val, cfg_sltp_mode, float(cfg_capital)
            )
            stats = trade_stats(trades, float(cfg_capital), final)

            st.markdown("#### Performance")
            if stats:
                mc = st.columns(len(stats))
                for c, (k, v) in zip(mc, stats.items()):
                    c.metric(k, v)

            st.markdown("#### Equity Curve")
            st.line_chart(equity.rename("Equity"))

            if trades:
                st.markdown("#### Trade Log")
                df_t = pd.DataFrame(trades)
                st.dataframe(df_t, use_container_width=True)

            st.markdown("#### Price + EMA 20 (last 300 bars)")
            df_p = df.tail(300).copy()
            df_p["EMA_20"] = indics["EMA_20"].reindex(df_p.index)
            st.line_chart(df_p[["Close","EMA_20"]])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE TRADING  (continuous via st.rerun)
# ═══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("🔴 Live Trading")
    st.caption(
        "Runs **continuously** — each cycle fetches fresh data with "
        f"**{cfg_throttle} s** throttle per request, then calls st.rerun() "
        "automatically when running."
    )

    # ── Ticker override (or use whatever scanner applied) ─────────────────
    col_ti, col_iv = st.columns(2)
    with col_ti:
        lv_ticker_sel, lv_label_sel = ticker_selector("lv")
    with col_iv:
        lv_iv_override = st.selectbox(
            "Interval override",
            ["(use sidebar)", "1m","2m","5m","15m","30m","1h"],
            key="lv_iv_ovr"
        )

    if lv_ticker_sel:
        st.session_state.lv_ticker = lv_ticker_sel
        st.session_state.lv_label  = lv_label_sel

    lv_iv = (cfg_interval if lv_iv_override == "(use sidebar)"
             else lv_iv_override)

    # ── Start / Stop / Reset ─────────────────────────────────────────────
    bc1, bc2, bc3, bc4 = st.columns(4)
    start_btn = bc1.button("▶ Start", type="primary",
                           disabled=st.session_state.lv_running, key="lv_start")
    stop_btn  = bc2.button("⏹ Stop",
                           disabled=not st.session_state.lv_running, key="lv_stop")
    reset_btn = bc3.button("🔄 Reset", key="lv_reset")
    step_btn  = bc4.button("⏭ Single Poll", key="lv_step")   # manual one-shot

    if start_btn:
        st.session_state.lv_running  = True
        st.session_state.lv_capital  = float(cfg_capital)
        st.session_state.lv_init_cap = float(cfg_capital)
    if stop_btn:
        st.session_state.lv_running  = False
    if reset_btn:
        for k, v in _live_defaults.items():
            st.session_state[k] = v
        st.session_state.lv_capital  = float(cfg_capital)
        st.session_state.lv_init_cap = float(cfg_capital)
        st.success("Session reset.")

    st.divider()

    # ── Status bar ───────────────────────────────────────────────────────
    icon   = "🟢" if st.session_state.lv_running else "⚫"
    active_ticker = st.session_state.lv_ticker
    active_label  = st.session_state.lv_label
    st.markdown(
        f"{icon} **{'RUNNING' if st.session_state.lv_running else 'STOPPED'}**  "
        f"|  Instrument: **{active_label}** `{active_ticker}`  "
        f"|  Interval: **{lv_iv}**  "
        f"|  Cycles: **{st.session_state.lv_poll_cnt}**  "
        f"|  Now: **{fmt_ist(now_ist())}**"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # POLL EXECUTION  (runs when lv_running=True OR step_btn clicked)
    # ═══════════════════════════════════════════════════════════════════════
    def _do_poll(ticker, lv_iv, throttle, polls):
        """Execute one polling cycle. Updates session_state in-place."""
        period = PERIODS_MAP.get(lv_iv, "60d")
        prog   = st.progress(0, text=f"Polling {active_label}…")
        err_ph = st.empty()

        for rnd in range(polls):
            prog.progress(
                int((rnd+0.2)/polls*100),
                text=f"Cycle {st.session_state.lv_poll_cnt+1}  "
                     f"Round {rnd+1}/{polls} — fetching {active_label}…"
            )
            try:
                df_live = _fetch_live(ticker, lv_iv, period, throttle=throttle)
            except Exception as exc:
                err_ph.error(f"Fetch error: {exc}")
                break

            if df_live.empty:
                err_ph.warning("Empty data — skipping.")
                continue

            df_live = df_index_to_ist(df_live)
            st.session_state.lv_poll_cnt += 1

            sigs, indics = sig_elliott_wave(
                df_live, cfg_lookback, cfg_min_wave, live_mode=True
            )

            bar_idx    = -2 if len(df_live) > 1 else -1
            last_close = float(df_live["Close"].iloc[bar_idx])
            last_hi    = float(df_live["High"].iloc[bar_idx])
            last_lo    = float(df_live["Low"].iloc[bar_idx])
            last_time  = df_live.index[bar_idx]
            ema20      = float(indics["EMA_20"].iloc[bar_idx])
            last_sig   = int(sigs.iloc[bar_idx])

            st.session_state.lv_last_px  = last_close
            st.session_state.lv_last_sig = last_sig

            # ── Manage open position ──────────────────────────────────
            pos = st.session_state.lv_position
            if pos:
                pnl, closed, exit_px = 0.0, False, 0.0
                if pos["side"] == 1:
                    if last_lo <= pos["sl"]:
                        pnl, exit_px, closed = pos["sl"]-pos["entry"], pos["sl"], True
                    elif last_hi >= pos["tp"]:
                        pnl, exit_px, closed = pos["tp"]-pos["entry"], pos["tp"], True
                else:
                    if last_hi >= pos["sl"]:
                        pnl, exit_px, closed = pos["entry"]-pos["sl"], pos["sl"], True
                    elif last_lo <= pos["tp"]:
                        pnl, exit_px, closed = pos["entry"]-pos["tp"], pos["tp"], True
                if closed:
                    pct = pnl / pos["entry"] * 100
                    st.session_state.lv_capital += pnl * (
                        st.session_state.lv_capital / pos["entry"]
                    )
                    st.session_state.lv_trades.append({
                        "entry_time": fmt_ist(pos["entry_time"]),
                        "exit_time" : fmt_ist(last_time),
                        "side"      : "BUY" if pos["side"]==1 else "SELL",
                        "entry"     : round(pos["entry"],4),
                        "exit"      : round(exit_px,4),
                        "sl"        : round(pos["sl"],4),
                        "tp"        : round(pos["tp"],4),
                        "pnl_pct"   : round(pct,2),
                        "result"    : "WIN" if pnl>0 else "LOSS",
                    })
                    st.session_state.lv_position = None

            # ── Enter new position ────────────────────────────────────
            if st.session_state.lv_position is None and last_sig != 0:
                entry = last_close
                sl, tp = compute_sl_tp(entry, last_sig, cfg_sl_val, cfg_tp_val, cfg_sltp_mode)
                st.session_state.lv_position = {
                    "side"      : last_sig,
                    "entry"     : entry,
                    "sl"        : sl,
                    "tp"        : tp,
                    "entry_time": last_time,
                }

            # ── Log ───────────────────────────────────────────────────
            pos_now = st.session_state.lv_position
            st.session_state.lv_log.insert(0, {
                "time (IST)" : fmt_ist(last_time),
                "close"      : round(last_close, 4),
                "ema_20"     : round(ema20, 4),
                "signal"     : {1:"🟢 BUY",-1:"🔴 SELL",0:"—"}.get(last_sig,"—"),
                "position"   : ("LONG"  if pos_now and pos_now["side"]==1 else
                                 "SHORT" if pos_now else "FLAT"),
                "capital"    : round(st.session_state.lv_capital, 2),
            })

            prog.progress(int((rnd+1)/polls*100),
                          text=f"Round {rnd+1}/{polls} done.")

        prog.empty()

    if st.session_state.lv_running or step_btn:
        _do_poll(active_ticker, lv_iv, cfg_throttle, cfg_lv_polls)

    # ═══════════════════════════════════════════════════════════════════════
    # LIVE PnL DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📊 Live Dashboard")

    cap      = st.session_state.lv_capital
    init_cap = st.session_state.lv_init_cap
    total_ret = (cap-init_cap)/init_cap*100 if init_cap else 0
    n_trades  = len(st.session_state.lv_trades)
    wins_l    = [t for t in st.session_state.lv_trades if t["result"]=="WIN"]
    wr        = len(wins_l)/n_trades*100 if n_trades else 0

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("💰 Capital", f"₹{cap:,.2f}",
              delta=f"{total_ret:+.2f}%",
              delta_color="normal" if total_ret>=0 else "inverse")
    m2.metric("📈 Trades",    n_trades)
    m3.metric("🎯 Win Rate",  f"{wr:.1f}%")
    m4.metric("🔄 Cycles",   st.session_state.lv_poll_cnt)
    m5.metric("📍 Last Sig",
              {1:"🟢 BUY",-1:"🔴 SELL",0:"—"}.get(st.session_state.lv_last_sig,"—"))
    m6.metric("🕐 Now (IST)", fmt_ist(now_ist()).split("  ")[0])  # just time part

    st.divider()

    # ── Open Position Card ────────────────────────────────────────────────
    st.markdown("#### 🏦 Open Position")
    pos = st.session_state.lv_position
    if pos:
        cur_px = st.session_state.lv_last_px or pos["entry"]
        if pos["side"] == 1:
            live_pnl   = cur_px - pos["entry"]
            prog_val   = max(0.0, min(1.0,
                (cur_px-pos["entry"])/(pos["tp"]-pos["entry"])
                if pos["tp"] != pos["entry"] else 0))
            rem_tp_abs = pos["tp"] - cur_px
            rem_sl_abs = cur_px  - pos["sl"]
        else:
            live_pnl   = pos["entry"] - cur_px
            prog_val   = max(0.0, min(1.0,
                (pos["entry"]-cur_px)/(pos["entry"]-pos["tp"])
                if pos["entry"] != pos["tp"] else 0))
            rem_tp_abs = cur_px  - pos["tp"]
            rem_sl_abs = pos["sl"] - cur_px

        live_pnl_pct = live_pnl / pos["entry"] * 100
        rem_tp_pct   = rem_tp_abs / pos["tp"]   * 100 if pos["tp"]  else 0
        rem_sl_pct   = rem_sl_abs / pos["sl"]   * 100 if pos["sl"]  else 0
        sl_dist      = abs(pos["entry"] - pos["sl"])
        tp_dist      = abs(pos["tp"]    - pos["entry"])
        rr_trade     = tp_dist/sl_dist if sl_dist > 0 else 0
        side_lbl     = "🟢 LONG" if pos["side"]==1 else "🔴 SHORT"

        pc1,pc2,pc3,pc4,pc5,pc6 = st.columns(6)
        pc1.metric("Direction",    side_lbl)
        pc2.metric("Entry",        f"{pos['entry']:.4f}")
        pc3.metric("Current",      f"{cur_px:.4f}",
                   delta=f"{live_pnl_pct:+.2f}%",
                   delta_color="normal" if live_pnl_pct>=0 else "inverse")
        pc4.metric("🛑 Stop Loss", f"{pos['sl']:.4f}",
                   delta=f"−{rem_sl_abs:.4f} ({rem_sl_pct:.2f}%)",
                   delta_color="inverse")
        pc5.metric("🎯 Target",    f"{pos['tp']:.4f}",
                   delta=f"+{rem_tp_abs:.4f} ({rem_tp_pct:.2f}%)",
                   delta_color="normal")
        pc6.metric("R/R",          f"{rr_trade:.2f}")

        st.markdown("**Progress toward target**")
        st.progress(min(prog_val, 1.0))

        st.caption(
            f"Entry time (IST): **{fmt_ist(pos['entry_time'])}**  ·  "
            f"SL distance: **{sl_dist:.4f}**  ·  TP distance: **{tp_dist:.4f}**"
        )
    else:
        sig = st.session_state.lv_last_sig
        msgs = {1:"🟢 BUY signal last seen — will enter on next cycle.",
               -1:"🔴 SELL signal last seen — will enter on next cycle.",
                0:"⚫ Flat — no open position. Watching for EW signal…"}
        {"normal":st.success, "warning":st.warning, "info":st.info}.get(
            {1:"normal",-1:"warning",0:"info"}.get(sig,"info"), st.info
        )(msgs.get(sig, "—"))

    # ── Closed Trades ─────────────────────────────────────────────────────
    if st.session_state.lv_trades:
        st.divider()
        st.markdown("#### 📋 Closed Trades")
        df_ct = pd.DataFrame(st.session_state.lv_trades)
        st.dataframe(df_ct, use_container_width=True)
        s1,s2,s3 = st.columns(3)
        s1.metric("Avg PnL %", f"{df_ct['pnl_pct'].mean():+.2f}%")
        s2.metric("Best %",    f"{df_ct['pnl_pct'].max():+.2f}%")
        s3.metric("Worst %",   f"{df_ct['pnl_pct'].min():+.2f}%")

    # ── Poll log ──────────────────────────────────────────────────────────
    if st.session_state.lv_log:
        with st.expander("🗒️ Poll Log (latest 20)", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.lv_log).head(20),
                         use_container_width=True)

    # ── CONTINUOUS RERUN ─────────────────────────────────────────────────
    # After rendering the full dashboard, trigger st.rerun() so the loop
    # continues automatically.  The 1.5 s sleep is inside _fetch_live().
    if st.session_state.lv_running:
        st.caption("⏳ Auto-refreshing… click **⏹ Stop** to halt.")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("⚙️ Grid Search Optimization")
    st.markdown(
        "Uses sidebar **SL/TP mode**, **capital**, **interval**, and **EW lookback** as "
        "the fixed context. Grid searches over `min_wave_pct`, `swing_lookback`, "
        "`sl_val`, `tp_val`.  Filtered by **min win rate** and **min trades**."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        opt_ticker, opt_label = ticker_selector("opt")
        opt_period_ovr = st.selectbox(
            "Override period",
            ["(use default)", "30d","60d","180d","1y","2y","max"],
            key="opt_period"
        )
    with col2:
        opt_mr  = st.slider("Min Wave % range",  0.3, 5.0, (0.5, 2.0), 0.1, key="opt_mr")
        opt_ms  = st.selectbox("Min Wave % step", [0.1, 0.25, 0.5], index=1, key="opt_ms")
        opt_lbr = st.slider("Lookback range",     4,  30,  (6, 16), 1,  key="opt_lbr")
        opt_lbs = st.selectbox("Lookback step",   [1, 2, 4], index=1,    key="opt_lbs")
    with col3:
        if cfg_sltp_mode == "Percentage":
            opt_sl_r = st.slider("SL % range",  0.1, 10.0, (0.5, 3.0), 0.5, key="opt_sl")
            opt_sl_s = st.selectbox("SL step",  [0.25, 0.5, 1.0], index=1,   key="opt_sls")
            opt_tp_r = st.slider("TP % range",  0.5, 20.0, (1.0, 6.0), 0.5,  key="opt_tp")
            opt_tp_s = st.selectbox("TP step",  [0.25, 0.5, 1.0], index=1,   key="opt_tps")
        else:
            opt_sl_r = st.slider("SL abs range", 1.0, 500.0, (10.0, 100.0), 5.0, key="opt_sl_a")
            opt_sl_s = st.selectbox("SL step",   [5.0, 10.0, 25.0], index=1,      key="opt_sls_a")
            opt_tp_r = st.slider("TP abs range", 1.0, 500.0, (20.0, 200.0), 5.0,  key="opt_tp_a")
            opt_tp_s = st.selectbox("TP step",   [5.0, 10.0, 25.0], index=1,      key="opt_tps_a")

        opt_min_acc    = st.slider("Min Accuracy (Win Rate) %", 0, 100, 60, 5,  key="opt_acc")
        opt_min_trades = st.number_input("Min Trades",          1,  200, 5,      key="opt_mt")

    if st.button("⚙️ Run Optimization", key="run_opt") and opt_ticker:
        period = (PERIODS_MAP.get(cfg_interval, "60d")
                  if opt_period_ovr == "(use default)"
                  else opt_period_ovr)

        with st.spinner(f"Fetching {opt_label}… (1.5 s throttle)"):
            time.sleep(1.5)
            df_opt = fetch_data(opt_ticker, cfg_interval, period)

        if df_opt.empty:
            st.error("No data returned.")
        else:
            df_opt = df_index_to_ist(df_opt)

            def _arange(lo, hi, step):
                vals, v = [], float(lo)
                while round(v,6) <= round(float(hi),6):
                    vals.append(round(v,6)); v += float(step)
                return vals

            g_mw = _arange(opt_mr[0],  opt_mr[1],  opt_ms)
            g_lb = list(range(opt_lbr[0], opt_lbr[1]+1, int(opt_lbs)))
            g_sl = _arange(opt_sl_r[0], opt_sl_r[1], opt_sl_s)
            g_tp = _arange(opt_tp_r[0], opt_tp_r[1], opt_tp_s)

            combos   = list(itertools.product(g_mw, g_lb, g_sl, g_tp))
            n_combos = len(combos)
            st.info(f"Evaluating **{n_combos}** combinations  "
                    f"(SL/TP mode: **{cfg_sltp_mode}**)…")

            bar     = st.progress(0, text="Running…")
            results = []

            for idx, (mw, lb, sl, tp) in enumerate(combos):
                if idx % max(1, n_combos//80) == 0:
                    bar.progress(int(idx/n_combos*100), text=f"{idx}/{n_combos}…")

                sigs, _ = sig_elliott_wave(df_opt, lb, mw, live_mode=False)
                trades, _, final = run_backtest(
                    df_opt, sigs, sl, tp, cfg_sltp_mode, float(cfg_capital)
                )
                if len(trades) < int(opt_min_trades):
                    continue

                s = trade_stats(trades, float(cfg_capital), final)
                if s.get("Win Rate %", 0) < opt_min_acc:
                    continue

                results.append({
                    "min_wave_pct"  : mw,
                    "swing_lookback": lb,
                    "sl_val"        : sl,
                    "tp_val"        : tp,
                    "sl_tp_mode"    : cfg_sltp_mode,
                    **s
                })

            bar.progress(100, text="Done!")

            if not results:
                st.warning(
                    f"No combinations met the filters: "
                    f"min {opt_min_trades} trades AND win rate ≥ {opt_min_acc}%."
                )
            else:
                df_res = (pd.DataFrame(results)
                          .sort_values("Total Return %", ascending=False)
                          .reset_index(drop=True))

                best = df_res.iloc[0]
                st.markdown("#### 🏆 Best Parameters")
                b1,b2,b3,b4,b5 = st.columns(5)
                b1.metric("Min Wave %",     best["min_wave_pct"])
                b2.metric("Swing Lookback", int(best["swing_lookback"]))
                b3.metric(f"SL ({cfg_sltp_mode})",  best["sl_val"])
                b4.metric(f"TP ({cfg_sltp_mode})",  best["tp_val"])
                b5.metric("SL/TP Mode",     cfg_sltp_mode)

                r1,r2,r3,r4 = st.columns(4)
                r1.metric("Total Return %", f"{best['Total Return %']}%")
                r2.metric("Win Rate %",     f"{best['Win Rate %']}%")
                r3.metric("Risk/Reward",    best["Risk/Reward"])
                r4.metric("Total Trades",   int(best["Total Trades"]))

                # Button to apply best params to sidebar hint
                st.info(
                    f"💡 Best params: Min Wave **{best['min_wave_pct']}%** · "
                    f"Lookback **{int(best['swing_lookback'])}** · "
                    f"SL **{best['sl_val']}** · TP **{best['tp_val']}** "
                    f"({cfg_sltp_mode}). Update sidebar sliders to apply."
                )

                st.markdown("#### Full Results")
                st.dataframe(
                    df_res.style.background_gradient(
                        subset=["Total Return %","Win Rate %"], cmap="RdYlGn"
                    ),
                    use_container_width=True
                )

                st.markdown("#### Parameter Correlation")
                num_cols = ["min_wave_pct","swing_lookback","sl_val","tp_val",
                            "Total Return %","Win Rate %","Risk/Reward"]
                corr = df_res[num_cols].corr()[["Total Return %","Win Rate %"]]
                st.dataframe(
                    corr.style.background_gradient(cmap="RdYlGn"),
                    use_container_width=True
                )
