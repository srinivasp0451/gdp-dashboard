"""
Algo Trading Platform — NIFTY50 / BANKNIFTY / SENSEX / Custom Indian tickers
=============================================================================
Single-file Streamlit app.

Features
--------
- Sidebar-driven configuration for everything (instrument, strategy, params, capital, etc.)
- Backtesting engine with multiple built-in strategies
- Live (paper) trading loop that polls yfinance
- Persistent trade history (CSV)
- yfinance data layer with a 0.3s throttle between every request to avoid rate-limit crashes

Run
---
    pip install streamlit yfinance pandas numpy plotly
    streamlit run algo_trading_app.py

NOTE: This is a paper-trading / educational tool. It places NO real orders.
"""

import os
import time
import json
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance is not installed. Run:  pip install yfinance")
    st.stop()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# --------------------------------------------------------------------------- #
#  CONSTANTS & PATHS
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algo_data")
os.makedirs(DATA_DIR, exist_ok=True)
TRADES_CSV = os.path.join(DATA_DIR, "trade_history.csv")
POSITIONS_JSON = os.path.join(DATA_DIR, "open_positions.json")

# Global throttle: enforce >= 0.3s between ANY two yfinance network calls.
_YF_LOCK = threading.Lock()
_LAST_YF_CALL = [0.0]
YF_MIN_INTERVAL = 0.3  # seconds

# Predefined Indian instruments (Yahoo Finance symbols)
PRESET_TICKERS = {
    "NIFTY 50":       "^NSEI",
    "BANK NIFTY":     "^NSEBANK",
    "SENSEX":         "^BSESN",
    "NIFTY IT":       "^CNXIT",
    "NIFTY FIN":      "NIFTY_FIN_SERVICE.NS",
    "Reliance":       "RELIANCE.NS",
    "TCS":            "TCS.NS",
    "HDFC Bank":      "HDFCBANK.NS",
    "Infosys":        "INFY.NS",
    "ICICI Bank":     "ICICIBANK.NS",
    "SBI":            "SBIN.NS",
}

INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "1d", "1wk"]
STRATEGIES = ["SMA Crossover", "EMA Crossover", "RSI Mean Reversion",
              "Bollinger Band Breakout", "MACD"]


# --------------------------------------------------------------------------- #
#  DATA LAYER  (throttled yfinance)
# --------------------------------------------------------------------------- #
def _throttle():
    """Block until at least YF_MIN_INTERVAL has passed since the last yf call."""
    with _YF_LOCK:
        elapsed = time.time() - _LAST_YF_CALL[0]
        wait = YF_MIN_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)
        _LAST_YF_CALL[0] = time.time()


@st.cache_data(show_spinner=False, ttl=30)
def fetch_history(symbol, period, interval):
    """Download historical OHLCV with a mandatory 0.3s throttle."""
    _throttle()
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Flatten possible multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def fetch_last_price(symbol):
    """Fetch the most recent price (throttled). Returns (price, timestamp)."""
    _throttle()
    df = yf.download(symbol, period="1d", interval="1m",
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    price = float(df["Close"].dropna().iloc[-1])
    ts = df.index[-1]
    return price, ts


# --------------------------------------------------------------------------- #
#  INDICATORS
# --------------------------------------------------------------------------- #
def sma(s, n):  return s.rolling(n).mean()
def ema(s, n):  return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger(s, n=20, k=2.0):
    mid = s.rolling(n).mean()
    std = s.rolling(n).std()
    return mid, mid + k * std, mid - k * std

def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig


# --------------------------------------------------------------------------- #
#  SIGNAL GENERATION
#  Returns a Series of position signals: 1 = long, -1 = short, 0 = flat
# --------------------------------------------------------------------------- #
def generate_signals(df, strategy, p):
    close = df["Close"]
    sig = pd.Series(0, index=df.index, dtype=float)

    if strategy == "SMA Crossover":
        fast, slow = sma(close, p["fast"]), sma(close, p["slow"])
        sig[fast > slow] = 1
        if p["allow_short"]:
            sig[fast < slow] = -1

    elif strategy == "EMA Crossover":
        fast, slow = ema(close, p["fast"]), ema(close, p["slow"])
        sig[fast > slow] = 1
        if p["allow_short"]:
            sig[fast < slow] = -1

    elif strategy == "RSI Mean Reversion":
        r = rsi(close, p["rsi_period"])
        sig[r < p["rsi_low"]] = 1
        if p["allow_short"]:
            sig[r > p["rsi_high"]] = -1
        # hold until neutral
        sig = sig.replace(0, np.nan).ffill().fillna(0)

    elif strategy == "Bollinger Band Breakout":
        mid, up, lo = bollinger(close, p["bb_period"], p["bb_k"])
        sig[close > up] = 1
        if p["allow_short"]:
            sig[close < lo] = -1
        sig = sig.replace(0, np.nan).ffill().fillna(0)

    elif strategy == "MACD":
        line, sg = macd(close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
        sig[line > sg] = 1
        if p["allow_short"]:
            sig[line < sg] = -1

    return sig.fillna(0)


# --------------------------------------------------------------------------- #
#  BACKTEST ENGINE
# --------------------------------------------------------------------------- #
def run_backtest(df, signals, capital, fee_bps, slippage_bps):
    """Vectorized long/short backtest on close prices."""
    close = df["Close"].astype(float)
    pos = signals.shift(1).fillna(0)          # act on next bar
    ret = close.pct_change().fillna(0)

    cost_rate = (fee_bps + slippage_bps) / 10000.0
    trades = pos.diff().abs().fillna(0)        # 0->1, 1->-1 etc.
    costs = trades * cost_rate

    strat_ret = pos * ret - costs
    equity = (1 + strat_ret).cumprod() * capital
    bh_equity = (1 + ret).cumprod() * capital

    # Trade log
    log = []
    entry_price = None
    entry_time = None
    entry_dir = 0
    for i in range(len(pos)):
        d = pos.iloc[i]
        if d != entry_dir:
            # close previous
            if entry_dir != 0 and entry_price is not None:
                exit_price = close.iloc[i]
                pnl = (exit_price - entry_price) / entry_price * entry_dir * 100
                log.append({
                    "Entry Time": entry_time, "Exit Time": close.index[i],
                    "Direction": "LONG" if entry_dir > 0 else "SHORT",
                    "Entry": round(entry_price, 2), "Exit": round(exit_price, 2),
                    "Return %": round(pnl, 3),
                })
            # open new
            if d != 0:
                entry_price = close.iloc[i]
                entry_time = close.index[i]
            entry_dir = d

    trade_df = pd.DataFrame(log)
    stats = compute_stats(equity, strat_ret, trade_df, capital)
    return equity, bh_equity, trade_df, stats


def compute_stats(equity, strat_ret, trade_df, capital):
    total_ret = (equity.iloc[-1] / capital - 1) * 100 if len(equity) else 0
    dd = (equity / equity.cummax() - 1).min() * 100 if len(equity) else 0
    ann = np.sqrt(252) * (strat_ret.mean() / strat_ret.std()) if strat_ret.std() > 0 else 0
    wins = (trade_df["Return %"] > 0).sum() if len(trade_df) else 0
    n = len(trade_df)
    return {
        "Total Return %": round(total_ret, 2),
        "Final Equity": round(equity.iloc[-1], 2) if len(equity) else capital,
        "Max Drawdown %": round(dd, 2),
        "Sharpe (ann.)": round(ann, 2),
        "Trades": n,
        "Win Rate %": round(100 * wins / n, 1) if n else 0,
    }


# --------------------------------------------------------------------------- #
#  TRADE HISTORY PERSISTENCE
# --------------------------------------------------------------------------- #
def load_trades():
    if os.path.exists(TRADES_CSV):
        try:
            return pd.read_csv(TRADES_CSV)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def append_trade(row):
    df = load_trades()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRADES_CSV, index=False)


def load_positions():
    if os.path.exists(POSITIONS_JSON):
        try:
            with open(POSITIONS_JSON) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_positions(pos):
    with open(POSITIONS_JSON, "w") as f:
        json.dump(pos, f, indent=2)


# --------------------------------------------------------------------------- #
#  UI  ─  SIDEBAR CONFIG
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Algo Trading Platform", layout="wide",
                   page_icon="📈")

st.sidebar.title("⚙️ Configuration")

mode = st.sidebar.radio("Mode", ["📊 Backtest", "🔴 Live (Paper)", "🧾 Trade History"])

st.sidebar.markdown("### Instrument")
src = st.sidebar.radio("Ticker source", ["Preset", "Custom"], horizontal=True)
if src == "Preset":
    name = st.sidebar.selectbox("Instrument", list(PRESET_TICKERS.keys()))
    symbol = PRESET_TICKERS[name]
else:
    symbol = st.sidebar.text_input(
        "Yahoo symbol", value="RELIANCE.NS",
        help="Use the Yahoo Finance symbol. NSE stocks end in .NS, BSE in .BO")
    name = symbol
st.sidebar.caption(f"Resolved symbol: `{symbol}`")

st.sidebar.markdown("### Data")
interval = st.sidebar.selectbox("Interval", INTERVALS, index=2)
period = st.sidebar.selectbox(
    "Look-back period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2)

st.sidebar.markdown("### Strategy")
strategy = st.sidebar.selectbox("Strategy", STRATEGIES)
allow_short = st.sidebar.checkbox("Allow short positions", value=False)

p = {"allow_short": allow_short}
if strategy in ("SMA Crossover", "EMA Crossover"):
    p["fast"] = st.sidebar.slider("Fast window", 3, 50, 9)
    p["slow"] = st.sidebar.slider("Slow window", 5, 200, 21)
elif strategy == "RSI Mean Reversion":
    p["rsi_period"] = st.sidebar.slider("RSI period", 5, 30, 14)
    p["rsi_low"] = st.sidebar.slider("Oversold (buy) <", 5, 45, 30)
    p["rsi_high"] = st.sidebar.slider("Overbought (sell) >", 55, 95, 70)
elif strategy == "Bollinger Band Breakout":
    p["bb_period"] = st.sidebar.slider("BB period", 10, 50, 20)
    p["bb_k"] = st.sidebar.slider("BB std-dev (k)", 1.0, 3.5, 2.0, 0.1)
elif strategy == "MACD":
    p["macd_fast"] = st.sidebar.slider("MACD fast", 5, 20, 12)
    p["macd_slow"] = st.sidebar.slider("MACD slow", 20, 50, 26)
    p["macd_signal"] = st.sidebar.slider("MACD signal", 5, 20, 9)

st.sidebar.markdown("### Risk & Costs")
capital = st.sidebar.number_input("Starting capital (₹)", 10000, 100_000_000,
                                  1_000_000, step=10000)
qty = st.sidebar.number_input("Order quantity (units)", 1, 100000, 1)
fee_bps = st.sidebar.number_input("Fee (bps per trade)", 0.0, 100.0, 3.0, 0.5)
slippage_bps = st.sidebar.number_input("Slippage (bps)", 0.0, 100.0, 2.0, 0.5)

st.sidebar.caption(f"⏱️ yfinance throttle: {YF_MIN_INTERVAL}s between requests")


# --------------------------------------------------------------------------- #
#  MAIN PANEL
# --------------------------------------------------------------------------- #
st.title("📈 Algo Trading Platform")
st.caption("Paper-trading & backtesting for Indian indices and equities. "
           "No real orders are placed.")


def price_chart(df, signals=None, title=""):
    if not HAS_PLOTLY:
        st.line_chart(df["Close"])
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    if signals is not None:
        longs = df.index[signals.diff() > 0]
        shorts = df.index[signals.diff() < 0]
        fig.add_trace(go.Scatter(
            x=longs, y=df.loc[longs, "Close"], mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#16a34a"),
            name="Buy"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=shorts, y=df.loc[shorts, "Close"], mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#dc2626"),
            name="Sell"), row=1, col=1)
    if "Volume" in df:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol",
                             marker_color="#94a3b8"), row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, title=title,
                      showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# ============================  BACKTEST  =================================== #
if mode == "📊 Backtest":
    st.subheader(f"Backtest — {name}  ({strategy})")
    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Fetching data & running backtest…"):
            df = fetch_history(symbol, period, interval)
        if df.empty:
            st.error("No data returned. Try a different symbol / interval / period.")
        else:
            signals = generate_signals(df, strategy, p)
            equity, bh, trades, stats = run_backtest(
                df, signals, capital, fee_bps, slippage_bps)

            c = st.columns(6)
            c[0].metric("Total Return", f"{stats['Total Return %']}%")
            c[1].metric("Final Equity", f"₹{stats['Final Equity']:,.0f}")
            c[2].metric("Max Drawdown", f"{stats['Max Drawdown %']}%")
            c[3].metric("Sharpe", stats["Sharpe (ann.)"])
            c[4].metric("Trades", stats["Trades"])
            c[5].metric("Win Rate", f"{stats['Win Rate %']}%")

            price_chart(df, signals, f"{name} — {interval}")

            st.markdown("#### Equity Curve")
            eq_df = pd.DataFrame({"Strategy": equity, "Buy & Hold": bh})
            st.line_chart(eq_df)

            st.markdown("#### Trade Log")
            if trades.empty:
                st.info("No trades were generated for these parameters.")
            else:
                st.dataframe(trades, use_container_width=True, height=300)
                st.download_button("⬇️ Download trade log (CSV)",
                                   trades.to_csv(index=False),
                                   f"backtest_{symbol}.csv", "text/csv")
    else:
        st.info("Configure the strategy in the sidebar, then click **Run Backtest**.")


# ============================  LIVE (PAPER)  =============================== #
elif mode == "🔴 Live (Paper)":
    st.subheader(f"Live Paper Trading — {name}")
    st.warning("Paper trading only — simulated fills, no broker connection.")

    poll = st.number_input("Refresh every (seconds)", 5, 300, 15)
    auto = st.checkbox("🔄 Auto-refresh loop")

    positions = load_positions()
    cur = positions.get(symbol)

    col1, col2 = st.columns([2, 1])

    with st.spinner("Fetching latest price…"):
        price, ts = fetch_last_price(symbol)
        hist = fetch_history(symbol, "5d", interval)

    if price is None:
        st.error("Could not fetch a live price for this symbol.")
    else:
        signals = generate_signals(hist, strategy, p) if not hist.empty else pd.Series()
        latest_sig = int(signals.iloc[-1]) if len(signals) else 0
        sig_label = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪ FLAT"}[latest_sig]

        with col1:
            price_chart(hist, signals, f"{name} — live {interval}")
        with col2:
            st.metric("Last Price", f"₹{price:,.2f}")
            st.caption(f"As of {ts}")
            st.metric("Strategy Signal", sig_label)
            if cur:
                entry = cur["entry_price"]
                d = 1 if cur["direction"] == "LONG" else -1
                upnl = (price - entry) / entry * d * 100
                st.metric("Open Position",
                          f"{cur['direction']} {cur['qty']} @ ₹{entry:,.2f}",
                          f"{upnl:+.2f}%")
            else:
                st.info("No open position.")

        st.markdown("#### Manual / Signal Order")
        b1, b2, b3 = st.columns(3)

        def do_open(direction):
            positions[symbol] = {
                "direction": direction, "qty": int(qty),
                "entry_price": price, "entry_time": str(ts), "name": name}
            save_positions(positions)
            st.success(f"Opened {direction} {qty} {name} @ ₹{price:,.2f}")

        def do_close():
            if not cur:
                st.warning("No open position to close.")
                return
            entry = cur["entry_price"]
            d = 1 if cur["direction"] == "LONG" else -1
            pnl_pct = (price - entry) / entry * d * 100
            pnl_abs = (price - entry) * d * cur["qty"]
            append_trade({
                "Symbol": symbol, "Name": name,
                "Direction": cur["direction"], "Qty": cur["qty"],
                "Entry Time": cur["entry_time"], "Entry Price": entry,
                "Exit Time": str(ts), "Exit Price": price,
                "PnL %": round(pnl_pct, 3), "PnL ₹": round(pnl_abs, 2),
                "Strategy": strategy, "Logged": str(datetime.now()),
            })
            del positions[symbol]
            save_positions(positions)
            st.success(f"Closed {cur['direction']} — PnL ₹{pnl_abs:,.2f} ({pnl_pct:+.2f}%)")

        if b1.button("🟢 Buy / Long"):
            do_open("LONG")
            st.rerun()
        if b2.button("🔴 Sell / Short"):
            do_open("SHORT")
            st.rerun()
        if b3.button("✖️ Close Position"):
            do_close()
            st.rerun()

        st.markdown("#### Recent Trades")
        th = load_trades()
        if not th.empty:
            st.dataframe(th.tail(10), use_container_width=True)
        else:
            st.caption("No trades logged yet.")

    if auto:
        time.sleep(poll)
        st.rerun()


# ============================  TRADE HISTORY  ============================= #
elif mode == "🧾 Trade History":
    st.subheader("Trade History")
    th = load_trades()
    if th.empty:
        st.info("No trades logged yet. Trades from Live (Paper) mode appear here.")
    else:
        # Summary
        total_pnl = th["PnL ₹"].sum() if "PnL ₹" in th else 0
        wins = (th["PnL ₹"] > 0).sum() if "PnL ₹" in th else 0
        n = len(th)
        c = st.columns(4)
        c[0].metric("Total Trades", n)
        c[1].metric("Net PnL", f"₹{total_pnl:,.2f}")
        c[2].metric("Win Rate", f"{100*wins/n:.1f}%" if n else "0%")
        c[3].metric("Avg PnL/Trade", f"₹{total_pnl/n:,.2f}" if n else "₹0")

        if "PnL ₹" in th:
            st.markdown("#### Cumulative PnL")
            st.line_chart(th["PnL ₹"].cumsum())

        st.dataframe(th, use_container_width=True, height=400)
        st.download_button("⬇️ Download full history (CSV)",
                           th.to_csv(index=False), "trade_history.csv", "text/csv")
        if st.button("🗑️ Clear history"):
            if os.path.exists(TRADES_CSV):
                os.remove(TRADES_CSV)
            st.rerun()
