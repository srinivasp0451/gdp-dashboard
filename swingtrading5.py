# streamlit_swing_dashboard_fixed.py
"""
Auto-Optimized Swing Trading Dashboard (long-only)
Robust column detection + numeric coercion fix (handles LTP, Last, 'Close Price', 'close.1', etc.)
"""

import re
import random
from math import isnan
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Auto-Optimized Swing Dashboard (Fixed)", initial_sidebar_state="collapsed")


# ---------------------------
# Robust normalization util
# ---------------------------
def _clean_name(s: str) -> str:
    s = str(s).lower().strip()
    # remove non-alphanumeric characters
    return re.sub(r'[^a-z0-9]', '', s)


def normalize_and_prepare(df_raw: pd.DataFrame):
    """
    Returns (df_prepared, col_map).
    df_prepared always has lowercase columns (open, high, low, close, volume) and datetime index if possible.
    col_map maps canonical names -> original column names (or None).
    """
    df = df_raw.copy()
    original_cols = list(df.columns)

    # Build mapping of cleaned_name -> original_name
    cleaned_map = {_clean_name(c): c for c in original_cols}

    # candidate lists for each field (ordered)
    open_cands = ["open", "openprice", "open_price", "o"]
    high_cands = ["high", "highprice", "high_price", "h"]
    low_cands = ["low", "lowprice", "low_price", "l"]
    close_cands = ["close", "closeprice", "close_price", "ltp", "last", "lastprice", "pxlast", "price", "adjclose", "close1"]
    volume_cands = ["volume", "vol", "tradedqty", "qty", "quantity"]
    date_cands = ["date", "datetime", "timestamp", "time", "trade_date", "tradedate"]

    def find_candidate(cands):
        # exact match first
        for cand in cands:
            k = _clean_name(cand)
            if k in cleaned_map:
                return cleaned_map[k]
        # substring match (cleaned)
        for k_clean, orig in cleaned_map.items():
            for cand in cands:
                if _clean_name(cand) in k_clean:
                    return orig
        # last resort: try looking for keywords inside original column (case-insensitive)
        for orig in original_cols:
            for cand in cands:
                if cand in str(orig).lower().replace(' ', ''):
                    return orig
        return None

    col_map = {}
    col_map["open"] = find_candidate(open_cands)
    col_map["high"] = find_candidate(high_cands)
    col_map["low"] = find_candidate(low_cands)
    col_map["close"] = find_candidate(close_cands)
    col_map["volume"] = find_candidate(volume_cands)
    col_map["date"] = find_candidate(date_cands)

    # If date not found, check if index is datetime-like
    if col_map["date"] is None and isinstance(df.index, pd.DatetimeIndex):
        # keep index as date
        pass
    elif col_map["date"] is not None:
        # try to set index
        try:
            df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors="coerce")
            df = df.set_index(col_map["date"])
        except Exception:
            # leave as is; we'll try to detect index below
            pass
    else:
        # attempt to interpret first column as date
        first_col = original_cols[0] if original_cols else None
        if first_col:
            maybe_date = pd.to_datetime(df[first_col], errors="coerce")
            if maybe_date.notna().sum() > 0 and maybe_date.notna().sum() / len(maybe_date) > 0.5:
                df.index = maybe_date
                try:
                    df.drop(columns=[first_col], inplace=True)
                except Exception:
                    pass

    # Now create a new DataFrame with canonical column names if possible
    new_df = pd.DataFrame(index=df.index)

    def to_numeric_series(series):
        # Remove any non-digit except dot and minus then convert
        s = series.astype(str).fillna("")
        s = s.str.replace(r'[^0-9\.\-]', '', regex=True)
        return pd.to_numeric(s, errors="coerce")

    for canon in ["open", "high", "low", "close", "volume"]:
        orig_col = col_map.get(canon)
        if orig_col is not None and orig_col in df.columns:
            new_df[canon] = to_numeric_series(df[orig_col])
        else:
            new_df[canon] = np.nan

    # Drop rows that miss essential OHLC values
    required = ["open", "high", "low", "close"]
    new_df.dropna(subset=required, inplace=True)

    # Ensure index is datetime if possible
    if not isinstance(new_df.index, pd.DatetimeIndex):
        # try to parse the index to datetime
        try:
            new_index = pd.to_datetime(new_df.index, errors="coerce")
            if new_index.notna().sum() > 0:
                new_df.index = new_index
                new_df = new_df[~new_df.index.isna()]
        except Exception:
            pass

    # final sanity: check we have numeric close column
    if "close" not in new_df.columns or new_df["close"].dropna().empty:
        # return mapping and original cols so UI can show helpful debugging
        return None, {"col_map": col_map, "original_columns": original_cols}

    # Sort index if datetime
    if isinstance(new_df.index, pd.DatetimeIndex):
        new_df.sort_index(inplace=True)

    # return prepared df and mapping (mapped canonical -> original name or None)
    return new_df, col_map


# ---------------------------
# Indicators, backtest, optimizer (same as before)
# ---------------------------
@st.cache_data
def compute_indicators(df, short=20, long=50, rsi_period=14):
    df = df.copy()
    # assume df has lowercase 'open','high','low','close'
    df["ma_short"] = df["close"].rolling(short, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(long, min_periods=1).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / rsi_period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(50, inplace=True)

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()

    # ADX simplified
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


def backtest(df_raw, params):
    prepared, meta = normalize_and_prepare(df_raw)
    if prepared is None:
        # propagate helpful error
        raise ValueError(f"Unable to detect OHLC numeric columns. Mapping attempt: {meta['col_map']}. File columns: {meta['original_columns']}")

    df = compute_indicators(prepared, short=params["short_ma"], long=params["long_ma"], rsi_period=params["rsi_period"])

    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is None:
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
            if row["high"] >= position["target"]:
                exit_price = position["target"]; reason = "Target"
            elif row["low"] <= position["sl"]:
                exit_price = position["sl"]; reason = "StopLoss"
            elif position["hold_days"] >= params["max_hold"]:
                exit_price = row["close"]; reason = "MaxHold"
            else:
                exit_price = None; reason = None

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

    # close open
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
    net = metrics["net_pnl"]
    trades = metrics["n_trades"]
    win = metrics["win_rate"]
    if trades < min_trades:
        net *= 0.5
    return net * (1 + win)


def auto_optimize(df_raw, search_space, max_evals=200, random_seed=42):
    random.seed(random_seed)
    best = None
    summary = []
    for _ in range(max_evals):
        params = {k: random.choice(v) for k, v in search_space.items() if k != "min_trades"}
        # attach other fixed params
        params["rsi_period"] = search_space.get("rsi_period", [14])[0]
        params["adx_min"] = random.choice(search_space.get("adx_min", [12]))
        trades, metrics = backtest(df_raw, params)
        sc = score_metrics(metrics, min_trades=search_space.get("min_trades", 3))
        summary.append({"params": params, "metrics": metrics, "score": sc})
        if best is None or sc > best["score"]:
            best = {"params": params, "metrics": metrics, "score": sc}
    return best, summary


# Plot helper
def plot_with_trades(df_raw, trades_df, title="Price chart with trades"):
    prepared, meta = normalize_and_prepare(df_raw)
    if prepared is None:
        raise ValueError(f"Cannot plot: failed normalization. Meta: {meta}")
    df = compute_indicators(prepared)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_short"], mode="lines", name="MA short"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_long"], mode="lines", name="MA long"))

    if trades_df is not None and not trades_df.empty:
        buys = trades_df.copy()
        buys["entry_date"] = pd.to_datetime(buys["entry_date"])
        buys["exit_date"] = pd.to_datetime(buys["exit_date"])
        fig.add_trace(go.Scatter(x=buys["entry_date"], y=buys["entry_price"], mode="markers",
                                 marker=dict(symbol="triangle-up", size=10, color="green"), name="Entry"))
        colors = ["green" if p > 0 else "red" for p in buys["pnl"]]
        fig.add_trace(go.Scatter(x=buys["exit_date"], y=buys["exit_price"], mode="markers",
                                 marker=dict(symbol="x", size=10, color=colors), name="Exit"))

    fig.update_layout(title=title, template="plotly_dark", height=500)
    return fig


# ---------------------------
# Streamlit UI main
# ---------------------------
st.title("📊 Auto-Optimized Swing Dashboard — Robust Loader Fix")

st.markdown("Upload your OHLC CSV (or XLSX). This version will automatically detect columns like `LTP`, `Last`, `Close Price`, `close.1` etc.")

uploaded = st.file_uploader("Upload file", type=["csv", "xls","xlsx"])
if uploaded is None:
    st.info("Upload a file to continue.")
    st.stop()

# load file
try:
    if uploaded.name.lower().endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.markdown("**Detected file columns (raw)**")
st.write(list(df_raw.columns))

prepared, meta = normalize_and_prepare(df_raw)
if prepared is None:
    st.error("Uploaded file doesn't contain usable OHLC data after normalization.")
    st.markdown("Mapping attempt to canonical columns:")
    st.json(meta["col_map"])
    st.markdown("Original columns in your file:")
    st.write(meta["original_columns"])
    st.warning("If none of the original column names look like OHLC, paste your header here and I'll map it.")
    st.stop()

# show mapping to user
st.markdown("### Column mapping (canonical -> original)")
st.json(meta)

st.success(f"Loaded {len(prepared)} rows. Date index: {isinstance(prepared.index, pd.DatetimeIndex)}")

# default search space
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

col1, col2 = st.columns([2, 1])
with col2:
    max_evals = st.number_input("Evaluations (random search)", min_value=20, max_value=800, value=120, step=20)
    run_opt = st.button("Run Auto Optimize & Backtest")

if not run_opt:
    st.info("Press 'Run Auto Optimize & Backtest' to start.")
    st.stop()

st.info("Optimization started — this may take a bit depending on evaluations.")

best, summary = auto_optimize(df_raw, default_search_space, max_evals=int(max_evals), random_seed=42)

st.markdown("## ✅ Best parameters found")
st.json(best["params"])
st.markdown("### Backtest metrics for those params")
st.json(best["metrics"])

trades_df, metrics = backtest(df_raw, best["params"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Net PnL", f"{metrics['net_pnl']:.2f}")
m2.metric("Trades", f"{metrics['n_trades']}")
m3.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
m4.metric("Avg Win", f"{metrics['avg_win']:.2f}")

tabs = st.tabs(["Chart", "Trades", "Optimizer details"])
with tabs[0]:
    fig = plot_with_trades(df_raw, trades_df, title="Price with trades")
    st.plotly_chart(fig, use_container_width=True)

    # live recommendation
    st.markdown("### Live recommendation (on last bar)")
    try:
        df_ind = compute_indicators(prepared, short=best["params"]["short_ma"], long=best["params"]["long_ma"], rsi_period=best["params"]["rsi_period"])
        latest = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        ma_cross = prev["ma_short"] <= prev["ma_long"] and latest["ma_short"] > latest["ma_long"]
        rsi_ok = latest["rsi"] <= best["params"]["rsi_entry"]
        macd_ok = latest["macd"] > latest["macd_signal"]
        adx_ok = latest["adx"] >= best["params"]["adx_min"]
        rec = None
        if ma_cross and rsi_ok and macd_ok and adx_ok:
            entry_price = latest["open"]
            sl_price = (latest["close"] - latest["atr"] * best["params"]["atr_mult"]) if best["params"]["use_atr_sl"] else latest["close"] * (1 - best["params"]["sl_pct"] / 100)
            target_price = latest["close"] * (1 + best["params"]["target_pct"] / 100)
            conf = best["metrics"]["win_rate"] * 100 if best["metrics"]["n_trades"] > 0 else 0.0
            rec = {"date": str(latest.name), "entry": float(entry_price), "sl": float(sl_price), "target": float(target_price), "confidence_pct": round(conf, 1)}
    except Exception as e:
        st.error(f"Live rec generation failed: {e}")
        rec = None

    if rec:
        st.success(f"Buy signal on {rec['date']} — confidence: {rec['confidence_pct']}%")
        st.json(rec)
    else:
        st.info("No buy signal on the latest bar with optimized params.")

with tabs[1]:
    st.markdown("### Trades (descending)")
    if trades_df is None or trades_df.empty:
        st.write("No trades generated.")
    else:
        tdisp = trades_df.sort_values("entry_date", ascending=False).reset_index(drop=True)
        st.dataframe(tdisp)

with tabs[2]:
    st.markdown("### Top optimizer candidates (top 8)")
    summary_sorted = sorted(summary, key=lambda x: x["score"], reverse=True)[:8]
    for i, it in enumerate(summary_sorted, 1):
        st.markdown(f"**#{i} — Score {it['score']:.2f}**")
        st.json({"params": it["params"], "metrics": it["metrics"]})

st.balloons()
