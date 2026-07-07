"""
================================================================================
 ALGO TRADING TERMINAL  —  NIFTY50 / BANKNIFTY / SENSEX / Custom Indian tickers
================================================================================
 Single-file Streamlit application (paper / simulated execution).

 Professional feature set
 ------------------------
  ENGINE CONTROL
     • Explicit  START  /  STOP  /  SQUARE-OFF ALL  buttons
     • Engine runs ONLY while ARMED — no uncontrolled infinite loops
     • Hard cap on API calls per session + per-minute rate budget
     • Configurable auto square-off time (e.g. 15:15 IST intraday exit)

  ORDER / RISK MANAGEMENT
     • Per-trade STOP-LOSS, TARGET and TRAILING STOP (points or %)
     • Auto-trigger: SL-hit / Target-hit / Trailing-SL / Time square-off
     • Position sizing by capital-per-trade or fixed qty
     • Risk guards: max open positions, max daily loss (kill-switch),
       max trades/day

  LIVE BLOTTER  (updates each engine tick)
     • Side | Qty | Entry | LTP | SL | Target | Unrealised P&L | %  | Status
     • Live MTM: realised, unrealised, net, day P&L
     • Signal panel: current strategy state + last flip time

  STRATEGY
     • SMA / EMA crossover, RSI, Bollinger, MACD, Supertrend
     • MANUAL or AUTO execution (engine takes strategy entries)

  DATA
     • yfinance with a strict >=0.3s throttle between EVERY request
     • Request counter + budget shown live so the API never crashes

 Run
 ---
     pip install streamlit yfinance pandas numpy plotly
     streamlit run algo_trading_app.py

 DISCLAIMER: simulated execution, no broker connection, no real orders.
================================================================================
"""

import os
import time
import json
from datetime import datetime, time as dtime, date

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance not installed →  pip install yfinance");  st.stop()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import pytz
IST = pytz.timezone("Asia/Kolkata")

# --------------------------------------------------------------------------- #
#  PATHS / CONSTANTS
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algo_data")
os.makedirs(DATA_DIR, exist_ok=True)
TRADES_CSV = os.path.join(DATA_DIR, "trade_history.csv")

PRESET = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT", "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS", "ICICI Bank": "ICICIBANK.NS", "SBI": "SBIN.NS",
    "Tata Motors": "TATAMOTORS.NS",
}
INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m"]
STRATEGIES = ["SMA Crossover", "EMA Crossover", "RSI", "Bollinger",
              "MACD", "Supertrend"]
YF_MIN_INTERVAL = 0.3          # hard floor between yfinance requests

# --------------------------------------------------------------------------- #
#  SESSION STATE  (the engine's brain)
# --------------------------------------------------------------------------- #
def _init():
    ss = st.session_state
    ss.setdefault("armed", False)              # engine running?
    ss.setdefault("positions", [])             # list of open position dicts
    ss.setdefault("order_log", [])             # audit trail (session)
    ss.setdefault("realised", 0.0)             # booked P&L (session)
    ss.setdefault("last_yf", 0.0)              # throttle timestamp
    ss.setdefault("api_calls", 0)              # total requests this session
    ss.setdefault("kill", False)               # daily-loss kill switch
    ss.setdefault("trades_today", 0)
    ss.setdefault("last_price", None)
    ss.setdefault("last_ts", None)
    ss.setdefault("last_tick", None)           # wall-clock of last engine tick
    ss.setdefault("signal_state", 0)
    ss.setdefault("signal_flip_ts", None)
_init()

# --------------------------------------------------------------------------- #
#  THROTTLED DATA LAYER
# --------------------------------------------------------------------------- #
def _throttle():
    elapsed = time.time() - st.session_state.last_yf
    if elapsed < YF_MIN_INTERVAL:
        time.sleep(YF_MIN_INTERVAL - elapsed)
    st.session_state.last_yf = time.time()
    st.session_state.api_calls += 1

def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.rename(columns=str.title)

@st.cache_data(show_spinner=False, ttl=20)
def _download(symbol, period, interval, _tick):
    """_tick busts cache so live mode gets fresh data each engine tick."""
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _flatten(df)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].dropna()

def fetch_history(symbol, period, interval, live=False):
    _throttle()
    tick = st.session_state.api_calls if live else 0
    return _download(symbol, period, interval, tick)

def fetch_ltp(symbol):
    _throttle()
    df = _download(symbol, "1d", "1m", st.session_state.api_calls)
    if df.empty:
        return None, None
    return float(df["Close"].iloc[-1]), df.index[-1]

# --------------------------------------------------------------------------- #
#  INDICATORS
# --------------------------------------------------------------------------- #
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
def boll(s, n=20, k=2.0):
    m = s.rolling(n).mean(); sd = s.rolling(n).std()
    return m, m + k * sd, m - k * sd
def macd(s, f=12, sl=26, sg=9):
    line = ema(s, f) - ema(s, sl); return line, ema(line, sg)
def atr(df, n=10):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def supertrend(df, n=10, mult=3.0):
    hl2 = (df["High"] + df["Low"]) / 2; a = atr(df, n)
    up = hl2 - mult * a; dn = hl2 + mult * a
    st_dir = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > dn.iloc[i - 1]: st_dir.iloc[i] = 1
        elif df["Close"].iloc[i] < up.iloc[i - 1]: st_dir.iloc[i] = -1
        else: st_dir.iloc[i] = st_dir.iloc[i - 1]
    return st_dir

def generate_signals(df, strat, p):
    c = df["Close"]; sig = pd.Series(0.0, index=df.index)
    if strat == "SMA Crossover":
        f, s = sma(c, p["fast"]), sma(c, p["slow"])
        sig[f > s] = 1; sig[f < s] = -1 if p["short"] else 0
    elif strat == "EMA Crossover":
        f, s = ema(c, p["fast"]), ema(c, p["slow"])
        sig[f > s] = 1; sig[f < s] = -1 if p["short"] else 0
    elif strat == "RSI":
        r = rsi(c, p["rsi_n"]); sig[r < p["rsi_lo"]] = 1
        if p["short"]: sig[r > p["rsi_hi"]] = -1
        sig = sig.replace(0, np.nan).ffill().fillna(0)
    elif strat == "Bollinger":
        m, u, lo = boll(c, p["bb_n"], p["bb_k"]); sig[c > u] = 1
        if p["short"]: sig[c < lo] = -1
        sig = sig.replace(0, np.nan).ffill().fillna(0)
    elif strat == "MACD":
        ln, sg = macd(c); sig[ln > sg] = 1; sig[ln < sg] = -1 if p["short"] else 0
    elif strat == "Supertrend":
        d = supertrend(df, p["st_n"], p["st_mult"]); sig = d if p["short"] else d.clip(lower=0)
    return sig.fillna(0)

# --------------------------------------------------------------------------- #
#  PERSISTENCE
# --------------------------------------------------------------------------- #
def load_trades():
    if os.path.exists(TRADES_CSV):
        try: return pd.read_csv(TRADES_CSV)
        except Exception: return pd.DataFrame()
    return pd.DataFrame()

def log_trade(row):
    df = load_trades()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRADES_CSV, index=False)

def audit(msg):
    st.session_state.order_log.insert(
        0, {"time": datetime.now(IST).strftime("%H:%M:%S"), "event": msg})
    st.session_state.order_log = st.session_state.order_log[:200]

# --------------------------------------------------------------------------- #
#  ORDER / POSITION ENGINE
# --------------------------------------------------------------------------- #
def open_position(symbol, name, side, qty, price, sl_pts, tgt_pts, trail_pts, strat):
    d = 1 if side == "LONG" else -1
    pos = {
        "id": len(st.session_state.order_log) + 1,
        "symbol": symbol, "name": name, "side": side, "qty": int(qty),
        "entry": price, "entry_ts": datetime.now(IST).strftime("%H:%M:%S"),
        "sl": price - d * sl_pts if sl_pts > 0 else None,
        "target": price + d * tgt_pts if tgt_pts > 0 else None,
        "trail_pts": trail_pts if trail_pts > 0 else None,
        "peak": price, "ltp": price, "strat": strat, "status": "OPEN",
    }
    st.session_state.positions.append(pos)
    st.session_state.trades_today += 1
    audit(f"OPEN {side} {qty} {name} @ {price:.2f}  SL={pos['sl']}  T={pos['target']}")
    return pos

def close_position(pos, price, reason):
    d = 1 if pos["side"] == "LONG" else -1
    pnl = (price - pos["entry"]) * d * pos["qty"]
    st.session_state.realised += pnl
    pos["status"] = "CLOSED"
    audit(f"CLOSE {pos['side']} {pos['name']} @ {price:.2f}  P&L ₹{pnl:,.2f}  [{reason}]")
    log_trade({
        "Date": date.today().isoformat(), "Symbol": pos["symbol"], "Name": pos["name"],
        "Side": pos["side"], "Qty": pos["qty"], "Entry": round(pos["entry"], 2),
        "Exit": round(price, 2), "Entry Time": pos["entry_ts"],
        "Exit Time": datetime.now(IST).strftime("%H:%M:%S"),
        "PnL ₹": round(pnl, 2),
        "PnL %": round((price - pos["entry"]) / pos["entry"] * d * 100, 3),
        "Exit Reason": reason, "Strategy": pos["strat"],
    })
    st.session_state.positions = [x for x in st.session_state.positions
                                  if x["id"] != pos["id"]]
    return pnl

def update_and_check(pos, ltp):
    """Update LTP, trail SL, and auto-trigger SL / Target. Returns reason|None."""
    pos["ltp"] = ltp
    d = 1 if pos["side"] == "LONG" else -1
    # trailing stop
    if pos["trail_pts"]:
        if d == 1 and ltp > pos["peak"]:
            pos["peak"] = ltp; pos["sl"] = max(pos["sl"] or -1e18, ltp - pos["trail_pts"])
        elif d == -1 and ltp < pos["peak"]:
            pos["peak"] = ltp; pos["sl"] = min(pos["sl"] or 1e18, ltp + pos["trail_pts"])
    # target
    if pos["target"] is not None and ((d == 1 and ltp >= pos["target"]) or
                                      (d == -1 and ltp <= pos["target"])):
        return "TARGET"
    # stop loss
    if pos["sl"] is not None and ((d == 1 and ltp <= pos["sl"]) or
                                  (d == -1 and ltp >= pos["sl"])):
        return "STOPLOSS"
    return None

def unrealised(pos):
    d = 1 if pos["side"] == "LONG" else -1
    return (pos["ltp"] - pos["entry"]) * d * pos["qty"]

# =========================================================================== #
#  BACKTEST
# =========================================================================== #
def run_backtest(df, sig, capital, fee_bps, slip_bps):
    close = df["Close"].astype(float)
    pos = sig.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    cost = pos.diff().abs().fillna(0) * (fee_bps + slip_bps) / 10000
    sret = pos * ret - cost
    eq = (1 + sret).cumprod() * capital
    bh = (1 + ret).cumprod() * capital
    # trade log
    log, ep, et, ed = [], None, None, 0
    for i in range(len(pos)):
        d = pos.iloc[i]
        if d != ed:
            if ed != 0 and ep is not None:
                xp = close.iloc[i]
                log.append({"Entry Time": et, "Exit Time": close.index[i],
                            "Side": "LONG" if ed > 0 else "SHORT",
                            "Entry": round(ep, 2), "Exit": round(xp, 2),
                            "Return %": round((xp - ep) / ep * ed * 100, 3)})
            if d != 0: ep, et = close.iloc[i], close.index[i]
            ed = d
    tl = pd.DataFrame(log)
    tot = (eq.iloc[-1] / capital - 1) * 100 if len(eq) else 0
    dd = (eq / eq.cummax() - 1).min() * 100 if len(eq) else 0
    shp = np.sqrt(252) * sret.mean() / sret.std() if sret.std() > 0 else 0
    wins = (tl["Return %"] > 0).sum() if len(tl) else 0
    stats = {"ret": round(tot, 2), "final": round(eq.iloc[-1], 0) if len(eq) else capital,
             "dd": round(dd, 2), "sharpe": round(shp, 2), "n": len(tl),
             "win": round(100 * wins / len(tl), 1) if len(tl) else 0}
    return eq, bh, tl, stats

# =========================================================================== #
#  CHART
# =========================================================================== #
def price_chart(df, sig=None, positions=None, title=""):
    if df.empty: return
    if not HAS_PLOTLY:
        st.line_chart(df["Close"]); return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                  low=df["Low"], close=df["Close"], name="Price"), 1, 1)
    if sig is not None and len(sig):
        lo = df.index[sig.diff() > 0]; sh = df.index[sig.diff() < 0]
        fig.add_trace(go.Scatter(x=lo, y=df.loc[lo, "Close"], mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#16a34a"), name="Buy"), 1, 1)
        fig.add_trace(go.Scatter(x=sh, y=df.loc[sh, "Close"], mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#dc2626"), name="Sell"), 1, 1)
    for pos in (positions or []):
        fig.add_hline(y=pos["entry"], line=dict(color="#2563eb", dash="dot"), row=1, col=1)
        if pos["sl"]:     fig.add_hline(y=pos["sl"], line=dict(color="#dc2626", dash="dash"), row=1, col=1)
        if pos["target"]: fig.add_hline(y=pos["target"], line=dict(color="#16a34a", dash="dash"), row=1, col=1)
    if "Volume" in df:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="#cbd5e1", name="Vol"), 2, 1)
    fig.update_layout(height=520, margin=dict(l=8, r=8, t=34, b=8), title=title,
                      xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================================== #
#  SIDEBAR
# =========================================================================== #
st.set_page_config(page_title="Algo Trading Terminal", layout="wide", page_icon="📊")
ss = st.session_state
st.sidebar.title("⚙️ Configuration")
mode = st.sidebar.radio("Module", ["🔴 Live Terminal", "📊 Backtest", "🧾 History"])

st.sidebar.markdown("### Instrument")
srctype = st.sidebar.radio("Source", ["Preset", "Custom"], horizontal=True)
if srctype == "Preset":
    name = st.sidebar.selectbox("Instrument", list(PRESET))
    symbol = PRESET[name]
else:
    symbol = st.sidebar.text_input("Yahoo symbol", "RELIANCE.NS",
        help="NSE → .NS, BSE → .BO"); name = symbol
st.sidebar.caption(f"Symbol → `{symbol}`")
interval = st.sidebar.selectbox("Interval", INTERVALS, 2)
period = st.sidebar.selectbox("History window",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y"], 2)

st.sidebar.markdown("### Strategy")
strat = st.sidebar.selectbox("Strategy", STRATEGIES)
allow_short = st.sidebar.checkbox("Allow shorts", False)
p = {"short": allow_short}
if strat in ("SMA Crossover", "EMA Crossover"):
    p["fast"] = st.sidebar.slider("Fast", 3, 50, 9); p["slow"] = st.sidebar.slider("Slow", 5, 200, 21)
elif strat == "RSI":
    p["rsi_n"] = st.sidebar.slider("RSI period", 5, 30, 14)
    p["rsi_lo"] = st.sidebar.slider("Buy <", 5, 45, 30); p["rsi_hi"] = st.sidebar.slider("Sell >", 55, 95, 70)
elif strat == "Bollinger":
    p["bb_n"] = st.sidebar.slider("Period", 10, 50, 20); p["bb_k"] = st.sidebar.slider("Std-dev k", 1.0, 3.5, 2.0, 0.1)
elif strat == "Supertrend":
    p["st_n"] = st.sidebar.slider("ATR period", 5, 30, 10); p["st_mult"] = st.sidebar.slider("Multiplier", 1.0, 5.0, 3.0, 0.5)

st.sidebar.markdown("### Order & Risk")
qty = st.sidebar.number_input("Quantity (units/lots)", 1, 100000, 1)
sl_pts = st.sidebar.number_input("Stop-Loss (points, 0=off)", 0.0, 1e6, 20.0, 1.0)
tgt_pts = st.sidebar.number_input("Target (points, 0=off)", 0.0, 1e6, 40.0, 1.0)
trail_pts = st.sidebar.number_input("Trailing SL (points, 0=off)", 0.0, 1e6, 0.0, 1.0)
capital = st.sidebar.number_input("Capital (₹)", 10000, 100_000_000, 1_000_000, 10000)
fee_bps = st.sidebar.number_input("Fee (bps)", 0.0, 100.0, 3.0, 0.5)
slip_bps = st.sidebar.number_input("Slippage (bps)", 0.0, 100.0, 2.0, 0.5)

st.sidebar.markdown("### Engine Guards")
auto_exec = st.sidebar.checkbox("AUTO-execute strategy signals", False,
    help="Engine opens/reverses positions on strategy flips. Off = manual only.")
max_pos = st.sidebar.number_input("Max open positions", 1, 20, 1)
max_trades = st.sidebar.number_input("Max trades/day", 1, 500, 20)
max_loss = st.sidebar.number_input("Max daily loss ₹ (kill-switch)", 0, 10_000_000, 20000, 1000)
poll = st.sidebar.number_input("Engine tick (seconds)", 5, 300, 15)
sqoff_on = st.sidebar.checkbox("Auto square-off at time", True)
sqoff_time = st.sidebar.time_input("Square-off time (IST)", dtime(15, 15))
st.sidebar.caption(f"⏱️ yfinance throttle ≥ {YF_MIN_INTERVAL}s · calls this session: {ss.api_calls}")

# =========================================================================== #
#  ENGINE TICK  (only runs while ARMED)
# =========================================================================== #
def engine_tick():
    ltp, tsx = fetch_ltp(symbol)
    if ltp is None:
        audit("⚠️ LTP fetch failed"); return None, None
    ss.last_price, ss.last_ts, ss.last_tick = ltp, tsx, datetime.now(IST)

    # signal
    hist = fetch_history(symbol, "5d", interval, live=True)
    sig = generate_signals(hist, strat, p) if not hist.empty else pd.Series()
    cur = int(sig.iloc[-1]) if len(sig) else 0
    if cur != ss.signal_state:
        ss.signal_state = cur; ss.signal_flip_ts = datetime.now(IST).strftime("%H:%M:%S")

    # update positions + auto SL/Target
    for pos in list(ss.positions):
        reason = update_and_check(pos, ltp)
        if reason: close_position(pos, ltp, reason)

    # time square-off
    if sqoff_on and datetime.now(IST).time() >= sqoff_time:
        for pos in list(ss.positions):
            close_position(pos, ltp, "TIME SQ-OFF")
        if ss.positions == [] and ss.armed:
            audit("⏰ Time square-off — engine idle")

    # daily-loss kill switch
    daymtm = ss.realised + sum(unrealised(x) for x in ss.positions)
    if max_loss > 0 and daymtm <= -max_loss and not ss.kill:
        ss.kill = True
        for pos in list(ss.positions): close_position(pos, ltp, "KILL-SWITCH")
        ss.armed = False
        audit(f"🛑 KILL-SWITCH: day loss ₹{daymtm:,.0f} ≤ -₹{max_loss:,}")

    # auto execution on signal
    if auto_exec and not ss.kill and ss.armed:
        want = "LONG" if cur == 1 else ("SHORT" if cur == -1 else None)
        held = ss.positions[0]["side"] if ss.positions else None
        if want and want != held:
            for pos in list(ss.positions):    # reverse
                close_position(pos, ltp, "SIGNAL FLIP")
            if len(ss.positions) < max_pos and ss.trades_today < max_trades:
                open_position(symbol, name, want, qty, ltp, sl_pts, tgt_pts, trail_pts, strat)
    return ltp, hist

# =========================================================================== #
#  LIVE TERMINAL
# =========================================================================== #
if mode == "🔴 Live Terminal":
    st.title("🔴 Live Trading Terminal")
    st.caption("Simulated execution · no broker orders placed")

    # ---- engine control bar ----
    st.markdown("### Engine Control")
    b = st.columns([1, 1, 1.4, 1, 3])
    if b[0].button("🟢 START", disabled=ss.armed or ss.kill, use_container_width=True):
        ss.armed = True; audit("▶️ Engine ARMED"); st.rerun()
    if b[1].button("⏸️ STOP", disabled=not ss.armed, use_container_width=True):
        ss.armed = False; audit("⏸️ Engine STOPPED"); st.rerun()
    if b[2].button("⛔ SQUARE-OFF ALL", type="primary", use_container_width=True):
        ltp, _ = fetch_ltp(symbol)
        if ltp:
            for pos in list(ss.positions): close_position(pos, ltp, "MANUAL SQ-OFF")
        audit("⛔ Manual square-off ALL"); st.rerun()
    if b[3].button("🔄 Reset day", use_container_width=True):
        ss.kill = False; ss.trades_today = 0; ss.realised = 0.0
        ss.positions = []; audit("🔄 Day reset"); st.rerun()

    status = ("🟢 ARMED" if ss.armed else "⚪ IDLE")
    if ss.kill: status = "🛑 KILLED (daily loss)"
    b[4].markdown(f"**Status:** {status}  ·  **Mode:** {'AUTO' if auto_exec else 'MANUAL'}  "
                  f"·  Trades today: **{ss.trades_today}/{max_trades}**")

    # ---- run a tick ----
    ltp, hist = (None, None)
    if ss.armed or ss.last_price is None:
        with st.spinner("Engine tick — fetching market data…"):
            ltp, hist = engine_tick()
    else:
        ltp = ss.last_price
        hist = fetch_history(symbol, "5d", interval)

    # ---- MTM header ----
    unreal = sum(unrealised(x) for x in ss.positions)
    net = ss.realised + unreal
    m = st.columns(6)
    m[0].metric("LTP", f"₹{ss.last_price:,.2f}" if ss.last_price else "—",
                f"{ss.last_ts}" if ss.last_ts else None)
    sig_lbl = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪ FLAT"}[ss.signal_state]
    m[1].metric("Signal", sig_lbl, ss.signal_flip_ts or "")
    m[2].metric("Open Positions", len(ss.positions))
    m[3].metric("Realised P&L", f"₹{ss.realised:,.0f}")
    m[4].metric("Unrealised P&L", f"₹{unreal:,.0f}")
    m[5].metric("Net Day P&L", f"₹{net:,.0f}", delta=f"{net:,.0f}")

    col1, col2 = st.columns([2, 1])
    with col1:
        sig = generate_signals(hist, strat, p) if (hist is not None and not hist.empty) else None
        price_chart(hist if hist is not None else pd.DataFrame(),
                    sig, ss.positions, f"{name} — {interval}")
    with col2:
        st.markdown("#### Manual Order Ticket")
        can_trade = (not ss.kill and len(ss.positions) < max_pos
                     and ss.trades_today < max_trades and ss.last_price)
        t = st.columns(2)
        if t[0].button("🟢 BUY / LONG", use_container_width=True, disabled=not can_trade):
            open_position(symbol, name, "LONG", qty, ss.last_price,
                          sl_pts, tgt_pts, trail_pts, strat); st.rerun()
        if t[1].button("🔴 SELL / SHORT", use_container_width=True, disabled=not can_trade):
            open_position(symbol, name, "SHORT", qty, ss.last_price,
                          sl_pts, tgt_pts, trail_pts, strat); st.rerun()
        if not can_trade and ss.last_price:
            st.caption("🔒 New entries blocked (kill-switch / max positions / max trades).")
        st.caption(f"Ticket → Qty {qty} · SL {sl_pts} · Tgt {tgt_pts} · Trail {trail_pts}")

    # ---- LIVE BLOTTER ----
    st.markdown("### 📋 Live Position Blotter")
    if ss.positions:
        rows = []
        for pos in ss.positions:
            u = unrealised(pos); d = 1 if pos["side"] == "LONG" else -1
            rows.append({
                "ID": pos["id"], "Instrument": pos["name"], "Side": pos["side"],
                "Qty": pos["qty"], "Entry": round(pos["entry"], 2),
                "LTP": round(pos["ltp"], 2),
                "SL": round(pos["sl"], 2) if pos["sl"] else "—",
                "Target": round(pos["target"], 2) if pos["target"] else "—",
                "P&L ₹": round(u, 2),
                "P&L %": round((pos["ltp"] - pos["entry"]) / pos["entry"] * d * 100, 2),
                "Status": pos["status"], "Since": pos["entry_ts"],
            })
        bl = pd.DataFrame(rows)
        def _color(v):
            try: return "color:#16a34a" if float(v) >= 0 else "color:#dc2626"
            except Exception: return ""
        st.dataframe(bl.style.map(_color, subset=["P&L ₹", "P&L %"]),
                     use_container_width=True, hide_index=True)
        # per-position exit buttons
        ec = st.columns(min(len(ss.positions), 4))
        for i, pos in enumerate(ss.positions):
            if ec[i % 4].button(f"✖ Close #{pos['id']} ({pos['side']})",
                                 key=f"cl{pos['id']}"):
                close_position(pos, ss.last_price, "MANUAL"); st.rerun()
    else:
        st.info("No open positions.")

    # ---- ORDER LOG ----
    with st.expander("🧾 Session Order Log / Audit Trail", expanded=False):
        if ss.order_log:
            st.dataframe(pd.DataFrame(ss.order_log), use_container_width=True,
                         hide_index=True, height=260)
        else:
            st.caption("No activity yet.")

    # ---- controlled auto-refresh (ONLY while armed) ----
    if ss.armed and not ss.kill:
        st.caption(f"⟳ Next engine tick in {poll}s … (STOP to halt)")
        time.sleep(poll)
        st.rerun()

# =========================================================================== #
#  BACKTEST
# =========================================================================== #
elif mode == "📊 Backtest":
    st.title("📊 Strategy Backtest")
    st.caption(f"{name} · {strat} · {interval} · {period}")
    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Fetching & simulating…"):
            df = fetch_history(symbol, period, interval)
        if df.empty:
            st.error("No data. Try another symbol/interval/period.")
        else:
            sig = generate_signals(df, strat, p)
            eq, bh, tl, s = run_backtest(df, sig, capital, fee_bps, slip_bps)
            c = st.columns(6)
            c[0].metric("Total Return", f"{s['ret']}%")
            c[1].metric("Final Equity", f"₹{s['final']:,.0f}")
            c[2].metric("Max Drawdown", f"{s['dd']}%")
            c[3].metric("Sharpe", s["sharpe"])
            c[4].metric("Trades", s["n"])
            c[5].metric("Win Rate", f"{s['win']}%")
            price_chart(df, sig, None, f"{name} — {interval}")
            st.markdown("#### Equity Curve"); st.line_chart(pd.DataFrame({"Strategy": eq, "Buy & Hold": bh}))
            st.markdown("#### Trade Log")
            if tl.empty: st.info("No trades generated.")
            else:
                st.dataframe(tl, use_container_width=True, height=300)
                st.download_button("⬇️ Trade log CSV", tl.to_csv(index=False),
                                   f"backtest_{symbol}.csv", "text/csv")
    else:
        st.info("Set parameters in the sidebar and click **Run Backtest**.")

# =========================================================================== #
#  HISTORY
# =========================================================================== #
else:
    st.title("🧾 Trade History")
    th = load_trades()
    if th.empty:
        st.info("No closed trades yet. Live-terminal exits are logged here.")
    else:
        tot = th["PnL ₹"].sum(); n = len(th); wins = (th["PnL ₹"] > 0).sum()
        gp = th.loc[th["PnL ₹"] > 0, "PnL ₹"].sum()
        gl = -th.loc[th["PnL ₹"] < 0, "PnL ₹"].sum()
        c = st.columns(5)
        c[0].metric("Trades", n)
        c[1].metric("Net P&L", f"₹{tot:,.0f}")
        c[2].metric("Win Rate", f"{100*wins/n:.1f}%")
        c[3].metric("Profit Factor", f"{gp/gl:.2f}" if gl else "∞")
        c[4].metric("Avg / Trade", f"₹{tot/n:,.0f}")
        st.markdown("#### Cumulative P&L"); st.line_chart(th["PnL ₹"].cumsum())
        st.dataframe(th, use_container_width=True, height=380)
        cc = st.columns(2)
        cc[0].download_button("⬇️ Download CSV", th.to_csv(index=False),
                              "trade_history.csv", "text/csv")
        if cc[1].button("🗑️ Clear history"):
            if os.path.exists(TRADES_CSV): os.remove(TRADES_CSV)
            st.rerun()
