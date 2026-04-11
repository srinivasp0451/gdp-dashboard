"""
Elliott Wave Trader — Streamlit App
====================================
Tabs: Backtest | Live Trading | Optimization

FIXES vs original:
  1. confirm_idx placement — no lookahead / repainting
  2. live_mode strips incomplete last candle
  3. _fetch_live() calls yf.download directly (no .func on CachedFunc)
  4. Expanded ticker universe: Nifty50 / BankNifty / Sensex /
     Crypto / Commodities / Forex + custom input
  5. Live tab: start/stop loop, live PnL card, entry/SL/TP display
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import itertools

# ─────────────────────────────────────────────────────────────────────────────
# TICKER UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
TICKER_GROUPS = {
    "── Indices ──": [
        "^NSEI",       # Nifty 50
        "^NSEBANK",    # Bank Nifty
        "^BSESN",      # Sensex
    ],
    "── Nifty 50 Stocks ──": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","BAJFINANCE.NS","BHARTIARTL.NS","KOTAKBANK.NS","ITC.NS",
        "LT.NS","HCLTECH.NS","ASIANPAINT.NS","AXISBANK.NS","MARUTI.NS",
        "TITAN.NS","SUNPHARMA.NS","ULTRACEMCO.NS","WIPRO.NS","NESTLEIND.NS",
    ],
    "── Crypto ──": [
        "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
    ],
    "── Commodities ──": [
        "GC=F",   # Gold futures
        "SI=F",   # Silver futures
        "CL=F",   # Crude Oil
        "NG=F",   # Natural Gas
    ],
    "── Forex ──": [
        "USDINR=X",
        "EURUSD=X",
        "GBPUSD=X",
        "USDJPY=X",
    ],
    "── Custom ──": ["__CUSTOM__"],
}

TICKER_LABELS = {
    "^NSEI":"Nifty 50","^NSEBANK":"Bank Nifty","^BSESN":"Sensex",
    "BTC-USD":"BTC/USD","ETH-USD":"ETH/USD","BNB-USD":"BNB/USD",
    "SOL-USD":"SOL/USD","XRP-USD":"XRP/USD",
    "GC=F":"Gold","SI=F":"Silver","CL=F":"Crude Oil","NG=F":"Nat Gas",
    "USDINR=X":"USD/INR","EURUSD=X":"EUR/USD",
    "GBPUSD=X":"GBP/USD","USDJPY=X":"USD/JPY",
    "__CUSTOM__":"Custom",
}

# Build flat display list + values list for selectbox
TICKER_DISPLAY: list[str] = []
TICKER_VALUES:  list      = []
for grp, tickers in TICKER_GROUPS.items():
    TICKER_DISPLAY.append(grp)
    TICKER_VALUES.append(None)
    for t in tickers:
        lbl = TICKER_LABELS.get(t, t)
        TICKER_DISPLAY.append(f"  {t}" if lbl == t else f"  {t}  ({lbl})")
        TICKER_VALUES.append(t)

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
    """Cached fetch — used for backtest and optimization."""
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def _fetch_live(ticker: str, interval: str, period: str,
                throttle: float = 1.5) -> pd.DataFrame:
    """
    Always fetches fresh data — NO cache, NO .func attribute access.
    Sleeps `throttle` seconds BEFORE the request to prevent yfinance 429s.
    """
    time.sleep(throttle)
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# FIXED PIVOT BUILDER  — stores confirm_idx
# ─────────────────────────────────────────────────────────────────────────────
def _ew_build_pivots(df: pd.DataFrame, min_wave_pct: float = 1.0,
                     live_mode: bool = False):
    """
    Returns: pivots list of (pivot_idx, price, direction, confirm_idx)
      confirm_idx = bar where price moved min_wave_pct past the pivot.
    In live_mode the last (incomplete) candle is excluded.
    """
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
                     live_mode: bool = False, **_):
    n = len(df)
    s = pd.Series(0, index=df.index)
    pivots, _ = _ew_build_pivots(df, min_wave_pct, live_mode=live_mode)
    if swing_lookback and len(pivots) > swing_lookback:
        pivots = pivots[-swing_lookback:]
    for k in range(2, len(pivots)):
        p0, p1, p2 = pivots[k-2], pivots[k-1], pivots[k]
        confirm_at = p2[3]
        if confirm_at >= n:
            continue
        if p0[2]==-1 and p1[2]==1 and p2[2]==-1 and p2[1] > p0[1]:
            s.iloc[confirm_at] = 1
        elif p0[2]==1 and p1[2]==-1 and p2[2]==1 and p2[1] < p0[1]:
            s.iloc[confirm_at] = -1
    return s, {"EMA_20": ema(df["Close"], 20)}


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, signals: pd.Series,
                 sl_pct: float = 1.5, tp_pct: float = 3.0,
                 initial_capital: float = 100_000.0):
    trades   = []
    capital  = initial_capital
    equity   = []
    position = None
    prices   = df[["Open","High","Low","Close"]].reset_index()
    sig_arr  = signals.values

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
            sl    = entry*(1-sl_pct/100) if side==1 else entry*(1+sl_pct/100)
            tp    = entry*(1+tp_pct/100) if side==1 else entry*(1-tp_pct/100)
            position = {"side":side,"entry":entry,"sl":sl,"tp":tp,
                        "entry_time":str(row.iloc[0])}

        equity.append(capital)

    return trades, pd.Series(equity, index=df.index), capital


def trade_stats(trades, initial_capital, final_capital):
    if not trades:
        return {}
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["result"]=="WIN"]
    loss = df_t[df_t["result"]=="LOSS"]
    wr   = len(wins)/len(df_t)*100
    aw   = wins["pnl_pct"].mean() if len(wins) else 0
    al   = loss["pnl_pct"].mean() if len(loss) else 0
    rr   = abs(aw/al)             if al         else float("inf")
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
# TICKER SELECTOR WIDGET
# ─────────────────────────────────────────────────────────────────────────────
def ticker_selector(key_prefix: str):
    """Returns (ticker_symbol, display_label). Handles headers + custom."""
    # find default index = BTC-USD
    default_idx = next(
        (i for i, v in enumerate(TICKER_VALUES) if v == "BTC-USD"), 1
    )
    sel = st.selectbox(
        "Instrument",
        options=range(len(TICKER_DISPLAY)),
        format_func=lambda i: TICKER_DISPLAY[i],
        index=default_idx,
        key=f"{key_prefix}_sel",
    )
    chosen = TICKER_VALUES[sel]
    if chosen is None:
        st.warning("Select an instrument, not a group header.")
        return None, None
    if chosen == "__CUSTOM__":
        val = st.text_input("Ticker (yfinance format)", value="AAPL",
                            key=f"{key_prefix}_custom")
        t = val.strip().upper()
        return t, t
    return chosen, TICKER_LABELS.get(chosen, chosen)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EW Trader", layout="wide", page_icon="〰️")
st.title("〰️  Elliott Wave Trader")
st.caption("No-lookahead zigzag  ·  Backtest  ·  Live  ·  Optimize")

tab_bt, tab_live, tab_opt = st.tabs(
    ["📊 Backtest", "🔴 Live Trading", "⚙️ Optimization"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("Backtest — Elliott Wave (fixed, no lookahead)")
    col1, col2, col3 = st.columns(3)
    with col1:
        bt_ticker, bt_label = ticker_selector("bt")
        bt_iv = st.selectbox("Interval", INTERVALS, index=4, key="bt_iv")
    with col2:
        bt_mw = st.slider("Min Wave %",     0.3, 5.0, 1.0, 0.1, key="bt_mw")
        bt_lb = st.slider("Swing Lookback", 4,   30,  10,  1,   key="bt_lb")
    with col3:
        bt_sl  = st.slider("Stop Loss %",   0.5, 5.0,  1.5, 0.1, key="bt_sl")
        bt_tp  = st.slider("Take Profit %", 0.5, 10.0, 3.0, 0.1, key="bt_tp")
        bt_cap = st.number_input("Capital",  value=100000, step=10000, key="bt_cap")

    if st.button("▶ Run Backtest", key="run_bt") and bt_ticker:
        period = PERIODS_MAP.get(bt_iv, "60d")
        with st.spinner(f"Fetching {bt_label}…"):
            df = fetch_data(bt_ticker, bt_iv, period)
        if df.empty:
            st.error("No data returned.")
        else:
            sigs, indics = sig_elliott_wave(df, bt_lb, bt_mw, live_mode=False)
            trades, equity, final = run_backtest(df, sigs, bt_sl, bt_tp, float(bt_cap))
            stats = trade_stats(trades, float(bt_cap), final)

            st.markdown("#### Performance")
            if stats:
                mc = st.columns(len(stats))
                for c, (k, v) in zip(mc, stats.items()):
                    c.metric(k, v)

            st.markdown("#### Equity Curve")
            st.line_chart(equity.rename("Equity"))

            if trades:
                st.markdown("#### Trade Log")
                st.dataframe(pd.DataFrame(trades), use_container_width=True)

            st.markdown("#### Price + EMA 20 (last 200 bars)")
            df_p = df.tail(200).copy()
            df_p["EMA_20"] = indics["EMA_20"].reindex(df_p.index)
            st.line_chart(df_p[["Close","EMA_20"]])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ═══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("🔴 Live Trading  (yfinance · 1.5 s throttle per request)")

    # ── Session state ────────────────────────────────────────────────────
    _lv_defaults = {
        "lv_running"  : False,
        "lv_position" : None,
        "lv_capital"  : 100_000.0,
        "lv_init_cap" : 100_000.0,
        "lv_trades"   : [],
        "lv_log"      : [],
        "lv_poll_cnt" : 0,
        "lv_last_sig" : 0,
        "lv_last_px"  : None,
    }
    for k, v in _lv_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Controls ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        lv_ticker, lv_label = ticker_selector("lv")
        lv_iv    = st.selectbox("Interval", ["1m","2m","5m","15m"], key="lv_iv")
        lv_cap   = st.number_input("Capital", value=100000, step=10000, key="lv_cap")
    with c2:
        lv_mw    = st.slider("Min Wave %",     0.3, 5.0,  1.0, 0.1, key="lv_mw")
        lv_lb    = st.slider("Swing Lookback", 4,   30,   10,  1,   key="lv_lb")
        lv_polls = st.slider("Poll rounds",    1,   10,   3,   1,   key="lv_polls")
    with c3:
        lv_sl    = st.slider("Stop Loss %",   0.5, 5.0,  1.5, 0.1, key="lv_sl")
        lv_tp    = st.slider("Take Profit %", 0.5, 10.0, 3.0, 0.1, key="lv_tp")

    # ── Start / Stop / Reset ─────────────────────────────────────────────
    bc1, bc2, bc3 = st.columns(3)
    start_btn = bc1.button("▶ Start Trading",  type="primary",
                           disabled=st.session_state.lv_running,
                           key="lv_start")
    stop_btn  = bc2.button("⏹ Stop Trading",
                           disabled=not st.session_state.lv_running,
                           key="lv_stop")
    reset_btn = bc3.button("🔄 Reset Session", key="lv_reset")

    if start_btn:
        st.session_state.lv_running  = True
        st.session_state.lv_capital  = float(lv_cap)
        st.session_state.lv_init_cap = float(lv_cap)
    if stop_btn:
        st.session_state.lv_running = False
    if reset_btn:
        for k, v in _lv_defaults.items():
            st.session_state[k] = v
        st.session_state.lv_capital  = float(lv_cap)
        st.session_state.lv_init_cap = float(lv_cap)
        st.success("Session reset.")

    st.divider()

    # ── Status badge ──────────────────────────────────────────────────────
    icon = "🟢" if st.session_state.lv_running else "🔴"
    st.markdown(
        f"**Status:** {icon} {'RUNNING' if st.session_state.lv_running else 'STOPPED'}  "
        f"| Polls completed: **{st.session_state.lv_poll_cnt}**"
    )

    # ═════════════════════════════════════════════════════════════════════
    # POLL LOOP
    # ═════════════════════════════════════════════════════════════════════
    if st.session_state.lv_running and lv_ticker:
        period = PERIODS_MAP.get(lv_iv, "60d")
        prog   = st.progress(0, text="Starting…")
        err_ph = st.empty()

        for rnd in range(int(lv_polls)):
            prog.progress(
                int((rnd + 0.2) / lv_polls * 100),
                text=f"Round {rnd+1}/{lv_polls} — fetching {lv_label} "
                     f"(1.5 s throttle enforced)…"
            )
            try:
                df_live = _fetch_live(lv_ticker, lv_iv, period, throttle=1.5)
            except Exception as exc:
                err_ph.error(f"Fetch error: {exc}")
                break

            if df_live.empty:
                err_ph.warning("Empty data — skipping round.")
                continue

            st.session_state.lv_poll_cnt += 1

            # Use second-to-last bar (last confirmed complete candle)
            sigs, indics = sig_elliott_wave(
                df_live, lv_lb, lv_mw, live_mode=True
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
                        "entry_time": pos["entry_time"],
                        "exit_time" : str(last_time),
                        "side"      : "BUY" if pos["side"]==1 else "SELL",
                        "entry"     : round(pos["entry"],4),
                        "exit"      : round(exit_px,4),
                        "sl"        : round(pos["sl"],4),
                        "tp"        : round(pos["tp"],4),
                        "pnl_pct"   : round(pct,2),
                        "result"    : "WIN" if pnl>0 else "LOSS",
                    })
                    st.session_state.lv_position = None

            # ── Enter new position on signal ──────────────────────────
            if st.session_state.lv_position is None and last_sig != 0:
                entry = last_close
                sl    = entry*(1-lv_sl/100) if last_sig==1 else entry*(1+lv_sl/100)
                tp    = entry*(1+lv_tp/100) if last_sig==1 else entry*(1-lv_tp/100)
                st.session_state.lv_position = {
                    "side"      : last_sig,
                    "entry"     : entry,
                    "sl"        : sl,
                    "tp"        : tp,
                    "entry_time": str(last_time),
                }

            # ── Log ───────────────────────────────────────────────────
            pos_now = st.session_state.lv_position
            st.session_state.lv_log.insert(0, {
                "time"    : str(last_time),
                "close"   : round(last_close, 4),
                "ema_20"  : round(ema20, 4),
                "signal"  : {1:"🟢 BUY",-1:"🔴 SELL",0:"—"}.get(last_sig, "—"),
                "position": ("LONG"  if pos_now and pos_now["side"]==1 else
                              "SHORT" if pos_now else "FLAT"),
                "capital" : round(st.session_state.lv_capital, 2),
            })

            prog.progress(
                int((rnd+1)/lv_polls*100),
                text=f"Round {rnd+1}/{lv_polls} complete."
            )

        prog.empty()

    # ═════════════════════════════════════════════════════════════════════
    # LIVE PnL DASHBOARD
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📊 Live Dashboard")

    cap      = st.session_state.lv_capital
    init_cap = st.session_state.lv_init_cap
    total_ret = (cap - init_cap) / init_cap * 100 if init_cap else 0
    n_trades  = len(st.session_state.lv_trades)
    wins_list = [t for t in st.session_state.lv_trades if t["result"]=="WIN"]
    wr        = len(wins_list)/n_trades*100 if n_trades else 0

    # ── Top metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💰 Capital",       f"{cap:,.2f}",
              delta=f"{total_ret:+.2f}%",
              delta_color="normal" if total_ret >= 0 else "inverse")
    m2.metric("📈 Total Trades",  n_trades)
    m3.metric("🎯 Win Rate",      f"{wr:.1f}%")
    m4.metric("🔄 Poll Count",    st.session_state.lv_poll_cnt)
    m5.metric("📍 Last Signal",
              {1:"🟢 BUY",-1:"🔴 SELL",0:"—"}.get(
                  st.session_state.lv_last_sig,"—"))

    st.divider()

    # ── Open Position Card ────────────────────────────────────────────────
    st.markdown("#### 🏦 Open Position")
    pos = st.session_state.lv_position

    if pos:
        cur_px = st.session_state.lv_last_px or pos["entry"]
        if pos["side"] == 1:
            live_pnl = cur_px - pos["entry"]
            progress_val = max(0.0, min(1.0,
                (cur_px - pos["entry"]) / (pos["tp"] - pos["entry"])
                if pos["tp"] != pos["entry"] else 0))
        else:
            live_pnl = pos["entry"] - cur_px
            progress_val = max(0.0, min(1.0,
                (pos["entry"] - cur_px) / (pos["entry"] - pos["tp"])
                if pos["entry"] != pos["tp"] else 0))

        live_pnl_pct  = live_pnl / pos["entry"] * 100
        dist_sl_pct   = abs(cur_px - pos["sl"])  / pos["entry"] * 100
        dist_tp_pct   = abs(pos["tp"] - cur_px)  / pos["entry"] * 100
        sl_dist_abs   = abs(pos["entry"] - pos["sl"])
        tp_dist_abs   = abs(pos["tp"]   - pos["entry"])
        rr_trade      = tp_dist_abs / sl_dist_abs if sl_dist_abs > 0 else 0
        side_lbl      = "🟢 LONG" if pos["side"]==1 else "🔴 SHORT"
        d_color       = "normal" if live_pnl_pct >= 0 else "inverse"

        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("Direction",  side_lbl)
        pc2.metric("Entry",      f"{pos['entry']:.4f}")
        pc3.metric("Current LTP",f"{cur_px:.4f}",
                   delta=f"{live_pnl_pct:+.2f}%",
                   delta_color=d_color)
        pc4.metric("🛑 Stop Loss",  f"{pos['sl']:.4f}",
                   delta=f"-{dist_sl_pct:.2f}% away",
                   delta_color="inverse")
        pc5.metric("🎯 Target",     f"{pos['tp']:.4f}",
                   delta=f"+{dist_tp_pct:.2f}% away",
                   delta_color="normal")

        st.markdown(f"**Progress toward target:**")
        st.progress(min(progress_val, 1.0))

        info_cols = st.columns(4)
        info_cols[0].caption(f"Entered: {pos['entry_time']}")
        info_cols[1].caption(f"R/R: **{rr_trade:.2f}**")
        info_cols[2].caption(f"SL risk: **{sl_dist_abs:.4f}**")
        info_cols[3].caption(f"TP reward: **{tp_dist_abs:.4f}**")

    else:
        sig = st.session_state.lv_last_sig
        if sig == 0:
            st.info("Flat — no open position. Watching for EW signal…")
        elif sig == 1:
            st.success("🟢 BUY signal on last bar — will enter on next poll.")
        else:
            st.warning("🔴 SELL signal on last bar — will enter on next poll.")

    st.divider()

    # ── Closed Trade Log ──────────────────────────────────────────────────
    if st.session_state.lv_trades:
        st.markdown("#### 📋 Closed Trades")
        df_ct = pd.DataFrame(st.session_state.lv_trades)
        st.dataframe(df_ct, use_container_width=True)

        # Summary row
        avg_pnl = df_ct["pnl_pct"].mean()
        best    = df_ct["pnl_pct"].max()
        worst   = df_ct["pnl_pct"].min()
        s1,s2,s3 = st.columns(3)
        s1.metric("Avg PnL %", f"{avg_pnl:+.2f}%")
        s2.metric("Best %",    f"{best:+.2f}%")
        s3.metric("Worst %",   f"{worst:+.2f}%")

    # ── Poll Log ──────────────────────────────────────────────────────────
    if st.session_state.lv_log:
        with st.expander("🗒️ Poll Log (latest 20)", expanded=False):
            st.dataframe(
                pd.DataFrame(st.session_state.lv_log).head(20),
                use_container_width=True
            )

    st.caption(
        "ℹ️ Each click of '▶ Start Trading' runs one batch of poll rounds. "
        "1.5 s throttle is enforced between every yfinance request. "
        "Stop Trading pauses the loop; Reset clears all state."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("⚙️ Grid Search Optimization")
    st.markdown("Grid over `min_wave_pct × swing_lookback × sl_pct × tp_pct`. "
                "One data fetch, then pure CPU. Ranked by Total Return %.")

    c1, c2, c3 = st.columns(3)
    with c1:
        opt_ticker, opt_label = ticker_selector("opt")
        opt_iv  = st.selectbox("Interval", INTERVALS, index=4, key="opt_iv")
        opt_cap = st.number_input("Capital", value=100000, step=10000, key="opt_cap")
    with c2:
        opt_mr  = st.slider("Min Wave % range",  0.3, 5.0, (0.5, 2.0), 0.1, key="opt_mr")
        opt_ms  = st.selectbox("Min Wave % step", [0.1, 0.25, 0.5], index=1, key="opt_ms")
        opt_lbr = st.slider("Lookback range",     4,  30,  (6, 16), 1,  key="opt_lbr")
        opt_lbs = st.selectbox("Lookback step",   [1, 2, 4], index=1,    key="opt_lbs")
    with c3:
        opt_sl  = st.slider("SL % range",  0.5, 5.0,  (1.0, 2.5), 0.5, key="opt_sl")
        opt_sls = st.selectbox("SL step",  [0.5, 1.0],              key="opt_sls")
        opt_tp  = st.slider("TP % range",  1.0, 10.0, (2.0, 5.0),  0.5, key="opt_tp")
        opt_tps = st.selectbox("TP step",  [0.5, 1.0],              key="opt_tps")
        opt_mt  = st.number_input("Min trades filter", 1, 50, 5,    key="opt_mt")

    if st.button("⚙️ Run Optimization", key="run_opt") and opt_ticker:
        period = PERIODS_MAP.get(opt_iv, "60d")
        with st.spinner(f"Fetching {opt_label}… (1.5 s throttle)"):
            time.sleep(1.5)
            df_opt = fetch_data(opt_ticker, opt_iv, period)

        if df_opt.empty:
            st.error("No data returned.")
        else:
            def _arange(lo, hi, step):
                vals, v = [], lo
                while round(v, 5) <= round(hi, 5):
                    vals.append(round(v, 5)); v += step
                return vals

            g_mw = _arange(opt_mr[0],  opt_mr[1],  float(opt_ms))
            g_lb = list(range(opt_lbr[0], opt_lbr[1]+1, int(opt_lbs)))
            g_sl = _arange(opt_sl[0],  opt_sl[1],  float(opt_sls))
            g_tp = _arange(opt_tp[0],  opt_tp[1],  float(opt_tps))

            combos   = list(itertools.product(g_mw, g_lb, g_sl, g_tp))
            n_combos = len(combos)
            st.info(f"Evaluating **{n_combos}** combinations…")

            bar     = st.progress(0, text="Running…")
            results = []

            for idx, (mw, lb, sl, tp) in enumerate(combos):
                if idx % max(1, n_combos // 60) == 0:
                    bar.progress(int(idx/n_combos*100), text=f"{idx}/{n_combos}…")
                sigs, _ = sig_elliott_wave(df_opt, lb, mw, live_mode=False)
                trades, _, final = run_backtest(
                    df_opt, sigs, sl, tp, float(opt_cap)
                )
                if len(trades) < int(opt_mt):
                    continue
                s = trade_stats(trades, float(opt_cap), final)
                results.append({"min_wave_pct":mw,"swing_lookback":lb,
                                 "sl_pct":sl,"tp_pct":tp, **s})

            bar.progress(100, text="Done!")

            if not results:
                st.warning("No combinations met the min-trade filter.")
            else:
                df_res = (pd.DataFrame(results)
                          .sort_values("Total Return %", ascending=False)
                          .reset_index(drop=True))

                best = df_res.iloc[0]
                st.markdown("#### 🏆 Best Parameters")
                b1,b2,b3,b4 = st.columns(4)
                b1.metric("Min Wave %",      best["min_wave_pct"])
                b2.metric("Swing Lookback",  int(best["swing_lookback"]))
                b3.metric("SL %",            best["sl_pct"])
                b4.metric("TP %",            best["tp_pct"])
                r1,r2,r3 = st.columns(3)
                r1.metric("Total Return %",  f"{best['Total Return %']}%")
                r2.metric("Win Rate %",      f"{best['Win Rate %']}%")
                r3.metric("Total Trades",    int(best["Total Trades"]))

                st.markdown("#### Full Results (sorted by Return %)")
                st.dataframe(
                    df_res.style.background_gradient(
                        subset=["Total Return %","Win Rate %"], cmap="RdYlGn"
                    ),
                    use_container_width=True
                )

                st.markdown("#### Parameter Correlation with Return / Win Rate")
                num_cols = ["min_wave_pct","swing_lookback","sl_pct","tp_pct",
                            "Total Return %","Win Rate %","Risk/Reward"]
                corr = df_res[num_cols].corr()[["Total Return %","Win Rate %"]]
                st.dataframe(
                    corr.style.background_gradient(cmap="RdYlGn"),
                    use_container_width=True
                )
