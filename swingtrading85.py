"""
Elliott Wave Trader — Streamlit App
====================================
Tabs: Backtest | Live Trading | Optimization

KEY FIX over original code
──────────────────────────
Original places the signal at the *pivot bar* (idx2).
In backtest this looks correct because the full history confirms the pivot.
In live trading the pivot isn't yet confirmed at that bar — only later,
when price moves min_wave_pct away, do we KNOW it was a pivot.
This creates lookahead bias / repainting that blows up live performance.

Fix: every pivot record now carries a confirm_idx = the bar at which the
pivot was confirmed.  Signals are placed at confirm_idx, not pivot_idx.
Additionally the last (incomplete) candle is always stripped in live mode.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import itertools
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
NIFTY50_TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","BAJFINANCE.NS","BHARTIARTL.NS","KOTAKBANK.NS","ITC.NS",
    "LT.NS","HCLTECH.NS","ASIANPAINT.NS","AXISBANK.NS","MARUTI.NS",
    "TITAN.NS","SUNPHARMA.NS","ULTRACEMCO.NS","WIPRO.NS","NESTLEIND.NS",
]

INTERVALS   = ["1m","2m","5m","15m","30m","1h","1d"]
PERIODS_MAP = {
    "1m":"7d","2m":"60d","5m":"60d","15m":"60d",
    "30m":"60d","1h":"730d","1d":"max"
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

@st.cache_data(ttl=90, show_spinner=False)
def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Cache yfinance data with 90-second TTL."""
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CORE ELLIOTT WAVE  (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
def _ew_build_pivots(df: pd.DataFrame, min_wave_pct: float = 1.0,
                     live_mode: bool = False):
    """
    Build confirmed zigzag pivots.

    Returns
    -------
    pivots : list of (pivot_idx, price, direction, confirm_idx)
        pivot_idx   – bar index of the extreme price
        price       – price at the pivot
        direction   – +1 swing-high, -1 swing-low
        confirm_idx – bar index at which the pivot was CONFIRMED
                      (first bar where price moved min_wave_pct away)
    in_progress : (last_dir, last_px, last_idx)
        The unconfirmed swing still building.

    LIVE FIX: when live_mode=True the last (possibly incomplete) candle
    is excluded so a half-built candle cannot contaminate the pivot logic.
    """
    cl = df["Close"]
    n  = len(cl)
    if live_mode and n > 1:
        n -= 1                      # ← drop incomplete last candle

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

        elif last_dir == 1:                         # tracking a high
            if p < float(cl.iloc[last_idx]) * (1 - min_mv):
                # HIGH confirmed at bar i
                pivots.append((last_idx, float(cl.iloc[last_idx]), 1, i))
                last_dir, last_px, last_idx = -1, p, i
            elif p > last_px:
                last_px, last_idx = p, i            # extend the high

        else:                                       # tracking a low
            if p > float(cl.iloc[last_idx]) * (1 + min_mv):
                # LOW confirmed at bar i
                pivots.append((last_idx, float(cl.iloc[last_idx]), -1, i))
                last_dir, last_px, last_idx = 1, p, i
            elif p < last_px:
                last_px, last_idx = p, i            # extend the low

    return pivots, (last_dir, last_px, last_idx)


def sig_elliott_wave(df: pd.DataFrame, swing_lookback: int = 10,
                     min_wave_pct: float = 1.0,
                     live_mode: bool = False, **_):
    """
    FIXED Elliott Wave signal generator.

    Signal is placed at confirm_idx (bar where the pivot was confirmed),
    NOT at the pivot bar — eliminates lookahead bias / repainting.

    Pattern logic (unchanged from original):
      BUY  : …low → high → lower-low  (corrective wave 2 / 4 in uptrend)
      SELL : …high → low → higher-high (corrective wave 2 / 4 in downtrend)
    """
    n = len(df)
    s = pd.Series(0, index=df.index)

    pivots, _ = _ew_build_pivots(df, min_wave_pct, live_mode=live_mode)

    # Optionally restrict to last swing_lookback pivots
    if swing_lookback and len(pivots) > swing_lookback:
        pivots = pivots[-swing_lookback:]

    for k in range(2, len(pivots)):
        p0, p1, p2 = pivots[k-2], pivots[k-1], pivots[k]
        # (pivot_idx, price, direction, confirm_idx)
        confirm_at = p2[3]          # ← KEY FIX: use confirm bar
        if confirm_at >= n:
            continue

        if (p0[2] == -1 and p1[2] == 1 and p2[2] == -1 and p2[1] > p0[1]):
            s.iloc[confirm_at] = 1   # BUY signal

        elif (p0[2] == 1 and p1[2] == -1 and p2[2] == 1 and p2[1] < p0[1]):
            s.iloc[confirm_at] = -1  # SELL signal

    e20 = ema(df["Close"], 20)
    return s, {"EMA_20": e20}


# ─────────────────────────────────────────────────────────────────────────────
# BACK-TEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, signals: pd.Series,
                 sl_pct: float = 1.5, tp_pct: float = 3.0,
                 initial_capital: float = 100_000.0):
    """
    Simple bar-by-bar backtest.
    Entry at next-bar open after signal.
    SL / TP checked against High/Low of each bar.
    Returns trades list + equity curve.
    """
    trades   = []
    equity   = [initial_capital]
    capital  = initial_capital
    position = None             # {"side","entry","sl","tp","entry_bar","entry_time"}

    prices = df[["Open","High","Low","Close"]].reset_index()
    sig_arr = signals.values

    for i in range(len(prices)):
        row = prices.iloc[i]

        # ── manage open position ──────────────────────────────────────────
        if position:
            hi, lo = float(row["High"]), float(row["Low"])
            pnl = 0
            closed = False

            if position["side"] == 1:           # LONG
                if lo <= position["sl"]:
                    pnl    = position["sl"] - position["entry"]
                    closed = True
                elif hi >= position["tp"]:
                    pnl    = position["tp"] - position["entry"]
                    closed = True
            else:                               # SHORT
                if hi >= position["sl"]:
                    pnl    = position["entry"] - position["sl"]
                    closed = True
                elif lo <= position["tp"]:
                    pnl    = position["entry"] - position["tp"]
                    closed = True

            if closed:
                pct_pnl = pnl / position["entry"] * 100
                capital += pnl * (capital / position["entry"])
                trades.append({
                    "entry_time" : position["entry_time"],
                    "exit_time"  : row.iloc[0] if hasattr(row.iloc[0], "strftime") else str(row.iloc[0]),
                    "side"       : "BUY" if position["side"] == 1 else "SELL",
                    "entry"      : round(position["entry"], 2),
                    "exit"       : round(position["sl"] if (
                                       (position["side"]==1 and lo<=position["sl"]) or
                                       (position["side"]==-1 and hi>=position["sl"])) else position["tp"], 2),
                    "pnl_pct"    : round(pct_pnl, 2),
                    "result"     : "WIN" if pnl > 0 else "LOSS",
                })
                position = None

        # ── new signal (enter next bar → use current Open as proxy) ──────
        if position is None and i > 0 and sig_arr[i-1] != 0:
            side   = int(sig_arr[i-1])
            entry  = float(row["Open"])
            sl     = entry * (1 - sl_pct/100) if side == 1 else entry * (1 + sl_pct/100)
            tp     = entry * (1 + tp_pct/100) if side == 1 else entry * (1 - tp_pct/100)
            position = {
                "side"      : side,
                "entry"     : entry,
                "sl"        : sl,
                "tp"        : tp,
                "entry_time": row.iloc[0],
            }

        equity.append(capital)

    equity_s = pd.Series(equity[1:], index=df.index)
    return trades, equity_s, capital


def trade_stats(trades: list, initial_capital: float, final_capital: float):
    if not trades:
        return {}
    df_t    = pd.DataFrame(trades)
    wins    = df_t[df_t["result"] == "WIN"]
    losses  = df_t[df_t["result"] == "LOSS"]
    wr      = len(wins) / len(df_t) * 100
    avg_win = wins["pnl_pct"].mean()   if len(wins)   else 0
    avg_los = losses["pnl_pct"].mean() if len(losses) else 0
    rr      = abs(avg_win / avg_los)   if avg_los     else float("inf")
    total_r = (final_capital - initial_capital) / initial_capital * 100
    return {
        "Total Trades"   : len(df_t),
        "Win Rate %"     : round(wr, 1),
        "Avg Win %"      : round(avg_win, 2),
        "Avg Loss %"     : round(avg_los, 2),
        "Risk/Reward"    : round(rr, 2),
        "Total Return %": round(total_r, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_metric_row(stats: dict):
    cols = st.columns(len(stats))
    for col, (k, v) in zip(cols, stats.items()):
        col.metric(k, v)


def render_equity_chart(equity: pd.Series):
    df_eq = equity.reset_index()
    df_eq.columns = ["Time", "Equity"]
    st.line_chart(df_eq.set_index("Time")["Equity"])


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EW Trader", layout="wide", page_icon="〰️")
st.title("〰️ Elliott Wave Trader")
st.caption("Backtest · Live · Optimize  |  Lookahead-free zigzag signals")

tab_bt, tab_live, tab_opt = st.tabs(["📊 Backtest", "🔴 Live Trading", "⚙️ Optimization"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("Backtest — Elliott Wave (fixed, no lookahead)")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_ticker   = st.selectbox("Ticker", NIFTY50_TICKERS, key="bt_ticker")
        bt_interval = st.selectbox("Interval", INTERVALS, index=4, key="bt_interval")
    with col2:
        bt_min_pct  = st.slider("Min Wave %", 0.3, 5.0, 1.0, 0.1, key="bt_min_pct")
        bt_lookback = st.slider("Swing Lookback (pivots)", 4, 30, 10, 1, key="bt_lb")
    with col3:
        bt_sl   = st.slider("Stop Loss %",    0.5, 5.0, 1.5, 0.1, key="bt_sl")
        bt_tp   = st.slider("Take Profit %",  0.5, 10.0, 3.0, 0.1, key="bt_tp")
        bt_cap  = st.number_input("Capital ₹", value=100000, step=10000, key="bt_cap")

    if st.button("▶ Run Backtest", key="run_bt"):
        period = PERIODS_MAP.get(bt_interval, "60d")
        with st.spinner("Fetching data…"):
            df = fetch_data(bt_ticker, bt_interval, period)

        if df.empty:
            st.error("No data returned. Check ticker / interval.")
        else:
            sigs, indics = sig_elliott_wave(
                df, swing_lookback=bt_lookback,
                min_wave_pct=bt_min_pct, live_mode=False
            )

            trades, equity, final_cap = run_backtest(
                df, sigs, sl_pct=bt_sl, tp_pct=bt_tp, initial_capital=bt_cap
            )
            stats = trade_stats(trades, bt_cap, final_cap)

            st.markdown("#### Performance")
            render_metric_row(stats)

            st.markdown("#### Equity Curve")
            render_equity_chart(equity)

            if trades:
                st.markdown("#### Trade Log")
                st.dataframe(pd.DataFrame(trades), use_container_width=True)

            # ── signal overlay ─────────────────────────────────────────────
            st.markdown("#### Signal Overview (last 200 bars)")
            df_plot = df.tail(200).copy()
            df_plot["EMA_20"] = indics["EMA_20"].reindex(df_plot.index)
            df_plot["Signal"] = sigs.reindex(df_plot.index)
            st.line_chart(df_plot[["Close","EMA_20"]])

            buys  = df_plot[df_plot["Signal"] ==  1]["Close"]
            sells = df_plot[df_plot["Signal"] == -1]["Close"]
            st.markdown(f"**Signals in last 200 bars →** "
                        f"🟢 BUY: {len(buys)}  🔴 SELL: {len(sells)}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRADING
# ═════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("Live Trading Simulation (yfinance, 1.5 s throttle)")
    st.info("Runs a polling loop — each refresh fetches fresh data with a 1.5 s "
            "delay between requests to avoid yfinance rate-limits.  "
            "Signals use **live_mode=True** (drops last incomplete candle + "
            "confirm_idx placement) so they match what the fixed backtest sees.")

    col1, col2 = st.columns(2)
    with col1:
        lv_ticker   = st.selectbox("Ticker",   NIFTY50_TICKERS, key="lv_ticker")
        lv_interval = st.selectbox("Interval", ["1m","2m","5m","15m"], key="lv_interval")
        lv_min_pct  = st.slider("Min Wave %", 0.3, 5.0, 1.0, 0.1, key="lv_min")
        lv_lookback = st.slider("Swing Lookback", 4, 30, 10, key="lv_lb")
    with col2:
        lv_sl       = st.slider("Stop Loss %",   0.5, 5.0, 1.5, 0.1, key="lv_sl")
        lv_tp       = st.slider("Take Profit %", 0.5,10.0, 3.0, 0.1, key="lv_tp")
        lv_refresh  = st.slider("Auto-refresh (s)", 10, 120, 30, key="lv_refresh")
        lv_loops    = st.slider("Poll rounds per refresh", 1, 5, 2, key="lv_loops")

    # ── session state ──────────────────────────────────────────────────────
    if "lv_trades"   not in st.session_state: st.session_state.lv_trades   = []
    if "lv_position" not in st.session_state: st.session_state.lv_position = None
    if "lv_capital"  not in st.session_state: st.session_state.lv_capital  = 100_000.0
    if "lv_log"      not in st.session_state: st.session_state.lv_log      = []

    def _throttled_fetch(ticker, interval, period, delay=1.5):
        """Fetch with mandatory delay to respect yfinance rate limits."""
        time.sleep(delay)
        return fetch_data.func(ticker, interval, period)   # bypass cache for live

    col_run, col_rst = st.columns(2)
    run_clicked   = col_run.button("▶ Fetch & Evaluate Signal", key="lv_run")
    reset_clicked = col_rst.button("🔄 Reset Session", key="lv_reset")

    if reset_clicked:
        st.session_state.lv_trades   = []
        st.session_state.lv_position = None
        st.session_state.lv_capital  = 100_000.0
        st.session_state.lv_log      = []
        st.success("Session reset.")

    if run_clicked:
        period = PERIODS_MAP.get(lv_interval, "60d")
        progress_bar = st.progress(0, text="Polling yfinance…")

        for loop in range(lv_loops):
            progress_bar.progress(
                int((loop + 0.5) / lv_loops * 100),
                text=f"Round {loop+1}/{lv_loops} — fetching (1.5 s throttle)…"
            )
            # ── 1.5 s enforced delay ──────────────────────────────────────
            df_live = _throttled_fetch(lv_ticker, lv_interval, period, delay=1.5)

            if df_live.empty:
                st.warning("Empty data returned.")
                break

            # ── generate signal in LIVE mode ─────────────────────────────
            sigs, indics = sig_elliott_wave(
                df_live,
                swing_lookback=lv_lookback,
                min_wave_pct=lv_min_pct,
                live_mode=True,             # ← drops incomplete last candle
            )

            last_sig   = int(sigs.iloc[-1])
            last_close = float(df_live["Close"].iloc[-1])
            last_time  = df_live.index[-1]
            ema20_now  = float(indics["EMA_20"].iloc[-1])

            log_entry = {
                "time"    : str(last_time),
                "close"   : round(last_close, 2),
                "ema20"   : round(ema20_now, 2),
                "signal"  : {1:"BUY", -1:"SELL", 0:"—"}.get(last_sig, "—"),
                "position": "LONG" if st.session_state.lv_position and
                             st.session_state.lv_position["side"]==1
                             else ("SHORT" if st.session_state.lv_position else "FLAT"),
            }
            st.session_state.lv_log.insert(0, log_entry)

            pos = st.session_state.lv_position

            # ── check SL / TP on open position ───────────────────────────
            if pos:
                hi = float(df_live["High"].iloc[-1])
                lo = float(df_live["Low"].iloc[-1])
                pnl, closed, exit_px = 0, False, 0
                if pos["side"] == 1:
                    if lo <= pos["sl"]:   pnl = pos["sl"] - pos["entry"]; exit_px=pos["sl"]; closed=True
                    elif hi >= pos["tp"]: pnl = pos["tp"] - pos["entry"]; exit_px=pos["tp"]; closed=True
                else:
                    if hi >= pos["sl"]:   pnl = pos["entry"] - pos["sl"]; exit_px=pos["sl"]; closed=True
                    elif lo <= pos["tp"]: pnl = pos["entry"] - pos["tp"]; exit_px=pos["tp"]; closed=True
                if closed:
                    pct = pnl / pos["entry"] * 100
                    st.session_state.lv_capital += pnl * (st.session_state.lv_capital / pos["entry"])
                    st.session_state.lv_trades.append({
                        "entry_time": pos["entry_time"],
                        "exit_time" : str(last_time),
                        "side"      : "BUY" if pos["side"]==1 else "SELL",
                        "entry"     : round(pos["entry"],2),
                        "exit"      : round(exit_px,2),
                        "pnl_pct"   : round(pct,2),
                        "result"    : "WIN" if pnl>0 else "LOSS",
                    })
                    st.session_state.lv_position = None
                    pos = None

            # ── enter new position on signal ──────────────────────────────
            if pos is None and last_sig != 0:
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

            progress_bar.progress(
                int((loop + 1) / lv_loops * 100),
                text=f"Round {loop+1}/{lv_loops} complete."
            )

        progress_bar.empty()

        # ── display ───────────────────────────────────────────────────────
        st.markdown("#### Current State")
        c1, c2, c3 = st.columns(3)
        c1.metric("Capital ₹", f"₹{st.session_state.lv_capital:,.0f}")
        c2.metric("Open Position",
                  "LONG"  if st.session_state.lv_position and st.session_state.lv_position["side"]==1 else
                  "SHORT" if st.session_state.lv_position else "FLAT")
        c3.metric("Trades", len(st.session_state.lv_trades))

        if st.session_state.lv_position:
            pos = st.session_state.lv_position
            st.info(f"Open {'LONG' if pos['side']==1 else 'SHORT'}  |  "
                    f"Entry: {pos['entry']:.2f}  |  SL: {pos['sl']:.2f}  |  TP: {pos['tp']:.2f}")

        if st.session_state.lv_log:
            st.markdown("#### Poll Log (latest first)")
            st.dataframe(pd.DataFrame(st.session_state.lv_log).head(20),
                         use_container_width=True)

        if st.session_state.lv_trades:
            st.markdown("#### Closed Trades")
            df_t = pd.DataFrame(st.session_state.lv_trades)
            st.dataframe(df_t, use_container_width=True)
            wins = (df_t["result"]=="WIN").sum()
            st.markdown(f"Win Rate: **{wins/len(df_t)*100:.1f}%**  "
                        f"| Total: **{len(df_t)}** trades")

    # ── auto-refresh hint ──────────────────────────────────────────────────
    st.markdown(f"---\n💡 Tip: use `st-autorefresh` or browser-side polling every "
                f"**{lv_refresh}s** to simulate continuous live trading.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("Grid Search Optimization")
    st.markdown("Exhaustive grid search over `min_wave_pct` × `swing_lookback` "
                "× `sl_pct` × `tp_pct`.  Each combination runs a full backtest "
                "(fixed, no-lookahead).  Ranked by Total Return %.")

    col1, col2, col3 = st.columns(3)
    with col1:
        opt_ticker   = st.selectbox("Ticker",   NIFTY50_TICKERS, key="opt_tk")
        opt_interval = st.selectbox("Interval", INTERVALS, index=4, key="opt_iv")
        opt_capital  = st.number_input("Capital ₹", value=100000, step=10000, key="opt_cap")
    with col2:
        opt_min_range  = st.slider("Min Wave % range",  0.3, 5.0, (0.5, 2.0), 0.1, key="opt_mr")
        opt_min_step   = st.selectbox("Min Wave % step", [0.1, 0.25, 0.5], index=1, key="opt_ms")
        opt_lb_range   = st.slider("Lookback range",     4,  30, (6, 16), 1, key="opt_lbr")
        opt_lb_step    = st.selectbox("Lookback step",   [1, 2, 4], index=1, key="opt_lbs")
    with col3:
        opt_sl_range   = st.slider("SL % range", 0.5, 5.0, (1.0, 2.5), 0.5, key="opt_sl")
        opt_sl_step    = st.selectbox("SL step",  [0.5, 1.0], key="opt_sls")
        opt_tp_range   = st.slider("TP % range", 1.0, 10.0, (2.0, 5.0), 0.5, key="opt_tp")
        opt_tp_step    = st.selectbox("TP step",  [0.5, 1.0], key="opt_tps")
        opt_min_trades = st.number_input("Min trades filter", 1, 50, 5, key="opt_mt")

    if st.button("⚙️ Run Optimization", key="run_opt"):
        period = PERIODS_MAP.get(opt_interval, "60d")

        with st.spinner("Fetching data…"):
            time.sleep(1.5)                 # throttle before fetch
            df_opt = fetch_data(opt_ticker, opt_interval, period)

        if df_opt.empty:
            st.error("No data. Check ticker / interval.")
        else:
            # ── build grid ────────────────────────────────────────────────
            def _arange(lo, hi, step):
                vals = []
                v = lo
                while round(v, 4) <= round(hi, 4):
                    vals.append(round(v, 4))
                    v += step
                return vals

            grid_min  = _arange(opt_min_range[0],  opt_min_range[1],  float(opt_min_step))
            grid_lb   = list(range(opt_lb_range[0], opt_lb_range[1]+1, int(opt_lb_step)))
            grid_sl   = _arange(opt_sl_range[0],   opt_sl_range[1],   float(opt_sl_step))
            grid_tp   = _arange(opt_tp_range[0],   opt_tp_range[1],   float(opt_tp_step))

            combos   = list(itertools.product(grid_min, grid_lb, grid_sl, grid_tp))
            n_combos = len(combos)
            st.info(f"Running **{n_combos}** combinations…")

            bar = st.progress(0, text="Optimizing…")
            results = []

            for idx, (mw, lb, sl, tp) in enumerate(combos):
                if idx % 20 == 0:
                    bar.progress(int(idx / n_combos * 100),
                                 text=f"{idx}/{n_combos} evaluated…")

                sigs, _ = sig_elliott_wave(
                    df_opt, swing_lookback=lb,
                    min_wave_pct=mw, live_mode=False
                )
                trades, _, final_cap = run_backtest(
                    df_opt, sigs, sl_pct=sl, tp_pct=tp,
                    initial_capital=float(opt_capital)
                )
                if len(trades) < opt_min_trades:
                    continue

                stats = trade_stats(trades, float(opt_capital), final_cap)
                results.append({
                    "min_wave_pct" : mw,
                    "swing_lookback": lb,
                    "sl_pct"       : sl,
                    "tp_pct"       : tp,
                    **stats,
                })

            bar.progress(100, text="Done!")

            if not results:
                st.warning("No combinations met the minimum trade filter.")
            else:
                df_res = pd.DataFrame(results).sort_values(
                    "Total Return %", ascending=False
                ).reset_index(drop=True)

                st.success(f"✅ {len(df_res)} valid combinations found.")

                # ── top result highlight ───────────────────────────────────
                best = df_res.iloc[0]
                st.markdown("#### 🏆 Best Parameters")
                bc = st.columns(4)
                bc[0].metric("Min Wave %",     best["min_wave_pct"])
                bc[1].metric("Swing Lookback", int(best["swing_lookback"]))
                bc[2].metric("SL %",           best["sl_pct"])
                bc[3].metric("TP %",           best["tp_pct"])

                c2 = st.columns(3)
                c2[0].metric("Total Return %", f"{best['Total Return %']}%")
                c2[1].metric("Win Rate %",     f"{best['Win Rate %']}%")
                c2[2].metric("Total Trades",   int(best["Total Trades"]))

                st.markdown("#### Full Results (sorted by Return)")
                st.dataframe(
                    df_res.style.background_gradient(
                        subset=["Total Return %","Win Rate %"], cmap="RdYlGn"
                    ),
                    use_container_width=True
                )

                # ── correlation heatmap (numeric columns) ─────────────────
                st.markdown("#### Parameter Correlation with Return")
                numeric_cols = ["min_wave_pct","swing_lookback","sl_pct",
                                "tp_pct","Total Return %","Win Rate %","Risk/Reward"]
                corr = df_res[numeric_cols].corr()[["Total Return %","Win Rate %"]]
                st.dataframe(corr.style.background_gradient(cmap="RdYlGn"),
                             use_container_width=True)
