# streamlit_swing_dashboard.py
"""
Auto-Optimized Swing Trading Dashboard (long-only)

Features:
- Robust CSV loader: case-insensitive column handling + numeric coercion + date parsing.
- Indicators: MA, RSI, MACD, ATR, ADX.
- Auto-optimization (random search) to pick best params (net PnL + reasonable win-rate).
- Backtest engine (entry: MA crossover + RSI + MACD + ADX; exits: target/SL/hold limit).
- Live recommendation using optimized params.
- Modern UI: top metric cards, tabs, Plotly interactive chart with buy/sell markers.
- No persistence — idempotent behavior on same data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import random
from math import isnan

# ---------------------------
# Utilities / Indicator code
# ---------------------------
st.set_page_config(layout="wide", page_title="Auto-Optimized Swing Dashboard", initial_sidebar_state="collapsed")


@st.cache_data
def normalize_and_prepare(df_raw):
    """Normalize columns, coerce numeric, set datetime index if possible, drop bad rows."""
    df = df_raw.copy()
    # Normalize cols
    df.columns = [str(c).strip().lower() for c in df.columns]

    # If there's a 'date' column try to parse and set index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")

    # If index is not datetime but first column looks like date, try it
    if not isinstance(df.index, pd.DatetimeIndex):
        # try detect first column as date
        first_col = df.columns[0]
        try:
            maybe_date = pd.to_datetime(df[first_col], errors="coerce")
            if maybe_date.notna().sum() > 0:
                df.index = maybe_date
                df.drop(columns=[first_col], inplace=True, errors="ignore")
        except Exception:
            pass

    # Force numeric on common OHLCV columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essential OHLC data
    required = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if required:
        df.dropna(subset=required, inplace=True)

    # Sort by index if datetime
    if isinstance(df.index, pd.DatetimeIndex):
        df.sort_index(inplace=True)

    return df


@st.cache_data
def compute_indicators(df, short=20, long=50, rsi_period=14):
    """Return DataFrame with added indicators. Assumes normalized lowercase columns."""
    df = df.copy()
    if not {"open", "high", "low", "close"}.issubset(set(df.columns)):
        raise ValueError("Input DataFrame must contain open, high, low, close columns (case-insensitive).")

    # MAs
    df["ma_short"] = df["close"].rolling(short, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(long, min_periods=1).mean()

    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / rsi_period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(50, inplace=True)

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()

    # ADX (simplified)
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr14 = tr.rolling(14, min_periods=1).sum()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / tr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / tr14.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.rolling(14, min_periods=1).mean()

    return df


# ---------------------------
# Backtester & optimizer
# ---------------------------
def backtest(df_raw, params):
    """Run backtest on df_raw (raw) with given params. Returns trades_df, metrics."""
    df = normalize_and_prepare(df_raw)
    df = compute_indicators(df, short=params["short_ma"], long=params["long_ma"], rsi_period=params["rsi_period"])

    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is None:
            # entry conditions
            ma_cross = prev["ma_short"] <= prev["ma_long"] and row["ma_short"] > row["ma_long"]
            rsi_ok = row["rsi"] <= params["rsi_entry"]
            macd_ok = row["macd"] > row["macd_signal"]
            adx_ok = row["adx"] >= params["adx_min"]

            if ma_cross and rsi_ok and macd_ok and adx_ok:
                entry_price = row["open"]
                sl_price = (row["close"] - row["atr"] * params["atr_mult"]) if params["use_atr_sl"] else entry_price * (1 - params["sl_pct"] / 100)
                target_price = entry_price * (1 + params["target_pct"] / 100)
                position = {
                    "entry_index": row.name,
                    "entry_price": entry_price,
                    "sl": sl_price,
                    "target": target_price,
                    "hold_days": 0
                }
        else:
            position["hold_days"] += 1
            # check exits using bar extremes
            if row["high"] >= position["target"]:
                exit_price = position["target"]
                reason = "Target"
            elif row["low"] <= position["sl"]:
                exit_price = position["sl"]
                reason = "StopLoss"
            elif position["hold_days"] >= params["max_hold"]:
                exit_price = row["close"]
                reason = "MaxHold"
            else:
                exit_price = None
                reason = None

            if exit_price is not None:
                pnl = exit_price - position["entry_price"]
                trades.append({
                    "entry_date": position["entry_index"],
                    "exit_date": row.name,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "hold_days": position["hold_days"],
                    "reason": reason
                })
                position = None

    # if still open at end, close at last close
    if position is not None:
        last = df.iloc[-1]
        pnl = last["close"] - position["entry_price"]
        trades.append({
            "entry_date": position["entry_index"],
            "exit_date": last.name,
            "entry_price": position["entry_price"],
            "exit_price": last["close"],
            "pnl": pnl,
            "hold_days": position["hold_days"],
            "reason": "EOD"
        })

    trades_df = pd.DataFrame(trades)
    metrics = {"net_pnl": 0.0, "n_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    if not trades_df.empty:
        metrics["net_pnl"] = trades_df["pnl"].sum()
        metrics["n_trades"] = len(trades_df)
        metrics["win_rate"] = (trades_df["pnl"] > 0).mean()
        metrics["avg_win"] = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if (trades_df["pnl"] > 0).any() else 0.0
        metrics["avg_loss"] = trades_df[trades_df["pnl"] <= 0]["pnl"].mean() if (trades_df["pnl"] <= 0).any() else 0.0

    return trades_df, metrics


def score_metrics(metrics, min_trades=3):
    """
    Score function for optimizer:
    - prefer higher net_pnl
    - penalize if trades < min_trades (to avoid one-off lucky trades)
    - combine with win_rate to prefer stable strategies
    """
    net = metrics["net_pnl"]
    trades = metrics["n_trades"]
    win = metrics["win_rate"]
    if trades < min_trades:
        net *= 0.5  # penalize low trade count
    # final score
    score = net * (1 + win)
    return score


def auto_optimize(df_raw, search_space, max_evals=200, random_seed=42):
    """Random search optimizer returning best_params, best_metrics, summary_list"""
    random.seed(random_seed)
    best = None
    summary = []

    for i in range(max_evals):
        params = {k: random.choice(v) for k, v in search_space.items()}
        trades, metrics = backtest(df_raw, params)
        sc = score_metrics(metrics, min_trades=search_space.get("min_trades", 3))
        summary.append({"params": params, "metrics": metrics, "score": sc})
        if best is None or sc > best["score"]:
            best = {"params": params, "metrics": metrics, "score": sc}
    return best, summary


# ---------------------------
# Plot helpers
# ---------------------------
def plot_with_trades(df_raw, trades_df, title="Price chart with trades"):
    df = normalize_and_prepare(df_raw)
    df = compute_indicators(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close", line=dict(width=1.5)))
    # add MAs
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_short"], mode="lines", name="MA short", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_long"], mode="lines", name="MA long", line=dict(width=1)))

    # plot entries/exits
    if trades_df is not None and not trades_df.empty:
        buys = trades_df.copy()
        buys["entry_date"] = pd.to_datetime(buys["entry_date"])
        buys["exit_date"] = pd.to_datetime(buys["exit_date"])

        # entry markers
        fig.add_trace(go.Scatter(
            x=buys["entry_date"], y=buys["entry_price"],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"), name="Entry"
        ))
        # exit markers colored by pnl
        colors = ["green" if p > 0 else "red" for p in buys["pnl"]]
        fig.add_trace(go.Scatter(
            x=buys["exit_date"], y=buys["exit_price"],
            mode="markers", marker=dict(symbol="x", size=10, color=colors), name="Exit"
        ))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template="plotly_dark", height=500)
    return fig


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 Auto-Optimized Swing Trading Dashboard (Long Only)")
st.markdown(
    "Upload a CSV with OHLC data (columns: Date/DateIndex, Open, High, Low, Close, Volume). "
    "The app will auto-normalize columns and perform an automatic parameter search."
)

uploaded = st.file_uploader("Upload OHLC CSV (CSV or Excel). No persistence — uploads only.", type=["csv", "xlsx"])

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### Upload & Settings")
    st.caption("If your CSV has a 'Date' column it will be used as index. Column names are case-insensitive.")
    st.caption("Optimization may take time depending on `Evaluations`. Start with 50 for quick results.")

with col2:
    st.markdown("### Auto-optimize controls")
    max_evals = st.number_input("Evaluations (random search)", min_value=20, max_value=800, value=120, step=20)
    run_opt = st.button("Run Auto Optimize & Backtest")

# default search space (can be changed here)
default_search_space = {
    "short_ma": list(range(5, 31, 1)),
    "long_ma": list(range(20, 101, 5)),
    "rsi_entry": [20, 25, 30, 35, 40],
    "target_pct": [0.8, 1.0, 1.5, 2.0, 3.0],
    "sl_pct": [0.5, 1.0, 1.5, 2.0, 3.0],
    "atr_mult": [0.8, 1.0, 1.5, 2.0, 2.5],
    "use_atr_sl": [True, False],
    "max_hold": [3, 5, 7, 10],
    "rsi_period": [14],
    "adx_min": [12, 15, 20],
    "min_trades": 3
}

if uploaded is None:
    st.info("Upload an OHLC CSV to proceed. Use sample data if you want to test quickly.")
    st.stop()

# read file robustly
try:
    if uploaded.name.lower().endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# prepare data
df = normalize_and_prepare(df_raw)
if df.empty or "close" not in df.columns:
    st.error("Uploaded file doesn't contain usable OHLC data after normalization. Ensure columns include Open/High/Low/Close.")
    st.stop()

st.success(f"Data loaded: {len(df)} rows — date range {df.index.min().date() if isinstance(df.index, pd.DatetimeIndex) else 'N/A'} to {df.index.max().date() if isinstance(df.index, pd.DatetimeIndex) else 'N/A'}")

# Run optimizer when user clicks
if run_opt:
    st.info("Running optimization — this may take some time. Progress updates will be shown below.")
    placeholder = st.empty()
    best, summary = auto_optimize(df, default_search_space, max_evals=int(max_evals), random_seed=42)

    # show best params & metrics
    st.markdown("## ✅ Best Parameters (found automatically)")
    st.json(best["params"])
    st.markdown("### Backtest Metrics (with those parameters)")
    st.json(best["metrics"])

    # produce backtest trades for best params
    trades_df, metrics = backtest(df, best["params"])

    # Top metrics cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net PnL", f"{metrics['net_pnl']:.2f}")
    m2.metric("Trades", f"{metrics['n_trades']}")
    m3.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
    m4.metric("Avg Win", f"{metrics['avg_win']:.2f}")

    # Tabs: chart, trades, optimization details
    tab1, tab2, tab3 = st.tabs(["Chart", "Trades", "Optimizer Details"])

    with tab1:
        st.markdown("### Price chart with entry & exit markers")
        fig = plot_with_trades(df, trades_df, title="Close with MA short/long and trades")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Live Recommendation (based on optimized params)")
        rec = None
        try:
            rec = None
            # live_recommendation: reuse logic from backtest but for last bar only
            df_ind = compute_indicators(df, short=best["params"]["short_ma"], long=best["params"]["long_ma"], rsi_period=best["params"]["rsi_period"])
            latest = df_ind.iloc[-1]
            prev = df_ind.iloc[-2]
            ma_cross = prev["ma_short"] <= prev["ma_long"] and latest["ma_short"] > latest["ma_long"]
            rsi_ok = latest["rsi"] <= best["params"]["rsi_entry"]
            macd_ok = latest["macd"] > latest["macd_signal"]
            adx_ok = latest["adx"] >= best["params"]["adx_min"]
            if ma_cross and rsi_ok and macd_ok and adx_ok:
                entry_price = latest["open"]
                sl_price = (latest["close"] - latest["atr"] * best["params"]["atr_mult"]) if best["params"]["use_atr_sl"] else latest["close"] * (1 - best["params"]["sl_pct"] / 100)
                target_price = latest["close"] * (1 + best["params"]["target_pct"] / 100)
                # confidence based on historical win rate
                confidence_pct = best["metrics"]["win_rate"] * 100 if best["metrics"]["n_trades"] > 0 else 0.0
                rec = {
                    "date": str(latest.name),
                    "entry_price": float(entry_price),
                    "sl": float(sl_price),
                    "target": float(target_price),
                    "confidence_pct": round(confidence_pct, 1),
                    "logic": "MA crossover + RSI + MACD + ADX (optimized params)"
                }
        except Exception as e:
            st.error(f"Failed to generate live recommendation: {e}")

        if rec:
            st.success(f"Buy Signal — Confidence: {rec['confidence_pct']}%")
            st.json(rec)
        else:
            st.info("No buy signal on the last bar with the optimized parameters.")

    with tab2:
        st.markdown("### Trades table (chronological)")
        if trades_df is None or trades_df.empty:
            st.write("No trades generated with optimized parameters.")
        else:
            # sort descending by entry date
            trades_display = trades_df.sort_values("entry_date", ascending=False).reset_index(drop=True)
            # format dates
            trades_display["entry_date"] = pd.to_datetime(trades_display["entry_date"])
            trades_display["exit_date"] = pd.to_datetime(trades_display["exit_date"])
            st.dataframe(trades_display)

    with tab3:
        st.markdown("### Optimizer summary (top 10 by score)")
        summary_sorted = sorted(summary, key=lambda x: x["score"], reverse=True)[:10]
        for i, it in enumerate(summary_sorted, 1):
            st.markdown(f"**#{i} — Score {it['score']:.2f}**")
            st.json({"params": it["params"], "metrics": it["metrics"]})

    st.balloons()

else:
    st.info("Click **Run Auto Optimize & Backtest** to start automatic optimization and backtest. Use fewer evaluations for faster results.")
