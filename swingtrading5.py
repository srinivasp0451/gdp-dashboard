# streamlit_swing_advanced.py
"""
Auto-Optimized Swing Trading Dashboard — Advanced (long-only)
Features:
- Robust loader (fuzzy mapping for OHLCV/date), numeric coercion.
- Indicators: MA, EMA, RSI, MACD, ATR, ADX, Bollinger Bands, MACD hist.
- Confluences: MA crossover, EMA trend, Bollinger breakout, volume spike, MACD hist rising, ATR threshold.
- Rule modes: strict (all), any_k (any K of N).
- Two-stage optimizer: Stage A random search, Stage B local hill-climb.
- Scoring: net_pnl*(1+win_rate) with penalties for too few trades & tiny avg pnl.
- Trade-level confluence tags and Plotly chart.
- Live recommendation using optimized params with confidence from historical win_rate.
- No persistence; runs per uploaded CSV.
"""

from __future__ import annotations
import re
import random
from typing import Tuple, Dict, Any, List
from math import isnan

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Auto-Optimized Swing — Advanced", initial_sidebar_state="collapsed")


# -----------------------------
# Robust loader / normalization
# -----------------------------
def _clean_name(s: str) -> str:
    s = str(s).lower().strip()
    return re.sub(r'[^a-z0-9]', '', s)


def detect_and_prepare(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Return (prepared_df, col_map).
    prepared_df has lowercase canonical columns: open, high, low, close, volume, index datetime if possible.
    col_map maps canonical -> original
    """
    df = df_raw.copy()
    orig_cols = list(df.columns)

    cleaned_map = {_clean_name(c): c for c in orig_cols}

    # candidate tokens
    open_cands = ["open", "openprice", "o"]
    high_cands = ["high", "highprice", "h"]
    low_cands = ["low", "lowprice", "l"]
    close_cands = ["close", "closeprice", "ltp", "last", "price", "adjclose"]
    vol_cands = ["volume", "vol", "quantity", "tradedqty"]
    date_cands = ["date", "datetime", "timestamp", "time"]

    def find(orig_candidate_list):
        for cand in orig_candidate_list:
            key = _clean_name(cand)
            if key in cleaned_map:
                return cleaned_map[key]
        # substring match in cleaned_map
        for k, v in cleaned_map.items():
            for cand in orig_candidate_list:
                if _clean_name(cand) in k:
                    return v
        # fallback
        for orig in orig_cols:
            low = str(orig).lower().replace(' ', '')
            for cand in orig_candidate_list:
                if cand in low:
                    return orig
        return None

    col_map = {
        "open": find(open_cands),
        "high": find(high_cands),
        "low": find(low_cands),
        "close": find(close_cands),
        "volume": find(vol_cands),
        "date": find(date_cands)
    }

    # try to set date index
    if col_map["date"] and col_map["date"] in df.columns:
        try:
            df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors="coerce")
            if df[col_map["date"]].notna().sum() > 0:
                df = df.set_index(col_map["date"])
        except Exception:
            pass
    else:
        # if index looks date-like keep it, else try to parse first column
        if not isinstance(df.index, pd.DatetimeIndex):
            first = orig_cols[0] if orig_cols else None
            if first:
                maybe = pd.to_datetime(df[first], errors="coerce")
                if maybe.notna().sum() / max(1, len(maybe)) > 0.5:
                    df.index = maybe
                    try:
                        df.drop(columns=[first], inplace=True)
                    except Exception:
                        pass

    # coerce numeric for canonical fields
    def to_numeric_series(s):
        s = s.astype(str).fillna("")
        s = s.str.replace(r'[^0-9\.\-]', '', regex=True)
        return pd.to_numeric(s, errors="coerce")

    prepared = pd.DataFrame(index=df.index)
    for canon in ["open", "high", "low", "close", "volume"]:
        orig = col_map.get(canon)
        if orig is not None and orig in df.columns:
            prepared[canon] = to_numeric_series(df[orig])
        else:
            prepared[canon] = np.nan

    # drop rows missing core OHLC
    prepared.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # ensure datetime index if possible
    if not isinstance(prepared.index, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(prepared.index, errors="coerce")
            if idx.notna().sum() > 0:
                prepared.index = idx
                prepared = prepared[~prepared.index.isna()]
        except Exception:
            pass

    # final check
    if prepared["close"].dropna().empty:
        return None, {"col_map": col_map, "original_columns": orig_cols}

    prepared.sort_index(inplace=True)
    return prepared, col_map


# -----------------------------
# Indicators & Confluences
# -----------------------------
def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add indicators used for confluences. Returns df copy with fields (ema9, ema21, ma_short, ma_long, rsi, macd, macd_signal, macd_hist, atr, adx, bb_upper, bb_mid, bb_lower, vol_avg)."""
    df = df.copy()
    # ensure required columns present
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = np.nan

    # EMAs & MAs
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ma_short"] = df["close"].rolling(params["short_ma"], min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(params["long_ma"], min_periods=1).mean()

    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / params.get("rsi_period", 14), adjust=False).mean()
    ma_down = down.ewm(alpha=1 / params.get("rsi_period", 14), adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(50, inplace=True)

    # MACD & hist
    macd_fast = df["close"].ewm(span=12, adjust=False).mean()
    macd_slow = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = macd_fast - macd_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(params.get("atr_period", 14), min_periods=1).mean()

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

    # Bollinger Bands
    bb_mid = df["close"].rolling(params.get("bb_period", 20), min_periods=1).mean()
    bb_std = df["close"].rolling(params.get("bb_period", 20), min_periods=1).std().fillna(0)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + params.get("bb_k", 2) * bb_std
    df["bb_lower"] = bb_mid - params.get("bb_k", 2) * bb_std

    # Volume moving average
    df["vol_avg"] = df["volume"].rolling(params.get("vol_period", 20), min_periods=1).mean()

    return df


def evaluate_confluences(row, params) -> Dict[str, bool]:
    """Given a single row (with indicators), evaluate confluence booleans."""
    confs = {}
    # MA crossover
    confs["ma_cross"] = (row["ma_short"] > row["ma_long"])
    # EMA trend: ema9 above ema21
    confs["ema_trend"] = (row["ema9"] > row["ema21"])
    # Bollinger breakout (close above upper band)
    confs["bb_breakout"] = (row["close"] > row["bb_upper"])
    # Volume spike
    confs["vol_spike"] = (row["volume"] >= params["vol_spike_mult"] * row["vol_avg"]) if (not np.isnan(row.get("vol_avg", np.nan))) else False
    # MACD hist rising (positive and increasing)
    confs["macd_hist_rising"] = (row["macd_hist"] > 0 and row["macd_hist"] > row.get("_macd_hist_prev", 0))
    # ATR volatility threshold
    confs["atr_ok"] = (row["atr"] >= params.get("atr_min", 0))
    return confs


# -----------------------------
# Backtester with confluence tagging
# -----------------------------
def backtest_with_confluences(df_raw: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, dict]:
    prepared, meta = detect_and_prepare(df_raw)
    if prepared is None:
        raise ValueError(f"Failed to detect OHLC columns. Mapping attempt: {meta['col_map']}. Original columns: {meta['original_columns']}")

    df = add_indicators(prepared, params)
    trades = []
    position = None

    # store previous macd_hist for "rising" detection
    df["_macd_hist_prev"] = df["macd_hist"].shift(1).fillna(0)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is None:
            confs = evaluate_confluences(row, params)
            # decide based on rule mode
            if params["rule_mode"] == "strict":
                do_enter = all([confs.get(k, False) for k in params["confluence_list"]])
            else:  # any_k
                hits = sum([1 for k in params["confluence_list"] if confs.get(k, False)])
                do_enter = hits >= params["any_k"]

            if do_enter:
                entry_price = row["open"]
                sl_price = (row["close"] - row["atr"] * params["atr_mult"]) if params["use_atr_sl"] else entry_price * (1 - params["sl_pct"] / 100)
                target_price = entry_price * (1 + params["target_pct"] / 100) if not params.get("target_atr", False) else entry_price + row["atr"] * params.get("target_atr_mult", 1.5)

                position = {
                    "entry_index": row.name,
                    "entry_price": float(entry_price),
                    "sl": float(sl_price),
                    "target": float(target_price),
                    "hold_days": 0,
                    "confluences": confs
                }
        else:
            position["hold_days"] += 1
            exit_price = None
            reason = None
            # check exits
            if row["high"] >= position["target"]:
                exit_price = position["target"]; reason = "Target"
            elif row["low"] <= position["sl"]:
                exit_price = position["sl"]; reason = "StopLoss"
            elif position["hold_days"] >= params["max_hold"]:
                exit_price = row["close"]; reason = "MaxHold"

            if exit_price is not None:
                pnl = exit_price - position["entry_price"]
                trades.append({
                    "entry_date": position["entry_index"],
                    "exit_date": row.name,
                    "entry_price": position["entry_price"],
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "hold_days": position["hold_days"],
                    "reason": reason,
                    "confluences": position["confluences"]
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
            "exit_price": float(last["close"]),
            "pnl": float(pnl),
            "hold_days": position["hold_days"],
            "reason": "EOD",
            "confluences": position["confluences"]
        })

    trades_df = pd.DataFrame(trades)
    metrics = {"net_pnl": 0.0, "n_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "avg_hold": 0.0, "trades_per_100bars": 0.0}
    if not trades_df.empty:
        metrics["net_pnl"] = trades_df["pnl"].sum()
        metrics["n_trades"] = len(trades_df)
        metrics["win_rate"] = (trades_df["pnl"] > 0).mean()
        metrics["avg_win"] = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if (trades_df["pnl"] > 0).any() else 0.0
        metrics["avg_loss"] = trades_df[trades_df["pnl"] <= 0]["pnl"].mean() if (trades_df["pnl"] <= 0).any() else 0.0
        metrics["avg_hold"] = trades_df["hold_days"].mean()
        metrics["trades_per_100bars"] = metrics["n_trades"] / max(1, len(prepared)) * 100

    return trades_df, metrics


# -----------------------------
# Optimizer (2-stage random + hill-climb)
# -----------------------------
def scoring_fn(metrics: dict, min_trades: int = 4) -> float:
    """Scoring: prefer positive net_pnl, decent win_rate, penalize very low trades."""
    score = float(metrics.get("net_pnl", 0.0))
    win = float(metrics.get("win_rate", 0.0))
    trades = int(metrics.get("n_trades", 0))
    # penalize tiny trade counts
    if trades < min_trades:
        score *= 0.4
    # combine with win-rate
    score *= (1.0 + win)
    # small penalty for tiny avg pnl magnitude (avoid noise)
    avg_abs = abs(metrics.get("avg_win", 0.0)) + abs(metrics.get("avg_loss", 0.0))
    if avg_abs < 0.01:
        score *= 0.8
    return score


def random_search_stage(df_raw, search_space: dict, evals: int, seed: int = 42) -> List[dict]:
    random.seed(seed)
    results = []
    for _ in range(evals):
        params = {k: random.choice(v) for k, v in search_space.items() if isinstance(v, list)}
        # attach other basic params
        params["rsi_period"] = search_space.get("rsi_period", [14])[0]
        params["confluence_list"] = search_space.get("confluence_list", [["ma_cross", "ema_trend", "macd_hist_rising", "vol_spike", "bb_breakout", "atr_ok"]])[0] if isinstance(search_space.get("confluence_list"), list) else search_space.get("confluence_list")
        params["rule_mode"] = random.choice(search_space.get("rule_mode", ["strict", "any_k"]))
        if params["rule_mode"] == "any_k":
            # pick any_k between 2 and number of confluences
            params["any_k"] = random.randint(2, max(2, len(params["confluence_list"])))
        # fixed defaults
        params.setdefault("use_atr_sl", False)
        params.setdefault("atr_mult", 1.5)
        params.setdefault("target_atr", False)
        params.setdefault("target_atr_mult", 1.5)
        params.setdefault("max_hold", 5)
        # run backtest
        try:
            trades_df, metrics = backtest_with_confluences(df_raw, params)
            score = scoring_fn(metrics, min_trades=search_space.get("min_trades", 4))
            results.append({"params": params, "metrics": metrics, "score": score, "trades_df": trades_df})
        except Exception:
            continue
    return results


def hill_climb_neighbors(base_params: dict, search_space: dict) -> List[dict]:
    """Generate neighbor parameter sets by small tweaks around base_params."""
    neighbors = []
    # tweak numeric params by +/- one step where applicable
    def pick_close(option_list, value):
        if value not in option_list:
            return random.choice(option_list)
        idx = option_list.index(value)
        choices = [option_list[max(0, idx - 1)], option_list[min(len(option_list) - 1, idx + 1)]]
        return random.choice(choices)

    for _ in range(15):  # produce 15 neighbors
        neighbor = base_params.copy()
        for key, space in search_space.items():
            if not isinstance(space, list):
                continue
            if key in neighbor:
                # small tweak for numeric lists
                neighbor[key] = pick_close(space, neighbor[key])
            else:
                neighbor[key] = random.choice(space)
        # adjust any_k if needed
        if neighbor.get("rule_mode") == "any_k" and "any_k" not in neighbor:
            neighbor["any_k"] = random.randint(2, max(2, len(neighbor.get("confluence_list", []))))
        neighbors.append(neighbor)
    return neighbors


def two_stage_optimize(df_raw: pd.DataFrame, search_space: dict, stage_a_eval: int = 120, seed: int = 42) -> Tuple[dict, List[dict]]:
    # Stage A
    stage_a_results = random_search_stage(df_raw, search_space, evals=stage_a_eval, seed=seed)
    if not stage_a_results:
        raise RuntimeError("Stage A found no valid parameter sets.")
    # pick top K
    top_k = sorted(stage_a_results, key=lambda x: x["score"], reverse=True)[:8]
    # Stage B: hill-climb around top_k
    best = top_k[0]
    all_candidates = stage_a_results.copy()
    for entry in top_k:
        base_params = entry["params"]
        neighbors = hill_climb_neighbors(base_params, search_space)
        for n in neighbors:
            try:
                trades_df, metrics = backtest_with_confluences(df_raw, n)
                score = scoring_fn(metrics, min_trades=search_space.get("min_trades", 4))
                all_candidates.append({"params": n, "metrics": metrics, "score": score, "trades_df": trades_df})
                if score > best["score"]:
                    best = {"params": n, "metrics": metrics, "score": score, "trades_df": trades_df}
            except Exception:
                continue
    return best, all_candidates


# -----------------------------
# Plot helpers
# -----------------------------
def plot_price_with_trades(prepared_df: pd.DataFrame, trades_df: pd.DataFrame, params: dict):
    df = add_indicators(prepared_df, params)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode="lines", name="Close", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_short"], mode="lines", name=f"MA{params['short_ma']}", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_long"], mode="lines", name=f"MA{params['long_ma']}", line=dict(width=1)))

    if trades_df is not None and not trades_df.empty:
        buys = trades_df.copy()
        buys["entry_date"] = pd.to_datetime(buys["entry_date"])
        buys["exit_date"] = pd.to_datetime(buys["exit_date"])
        fig.add_trace(go.Scatter(x=buys["entry_date"], y=buys["entry_price"], mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"), name="Entry"))
        colors = ["green" if p > 0 else "red" for p in buys["pnl"]]
        fig.add_trace(go.Scatter(x=buys["exit_date"], y=buys["exit_price"], mode="markers", marker=dict(symbol="x", size=10, color=colors), name="Exit"))

    fig.update_layout(template="plotly_dark", height=600, title="Price & Trades")
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚀 Auto-Optimized Swing Trading Dashboard — Advanced")
st.markdown("Upload an OHLC CSV/XLSX. This version finds more quality trades using multiple confluences and a two-stage optimizer.")

uploaded = st.file_uploader("Upload OHLC (CSV or XLSX)", type=["csv", "xlsx"])
if not uploaded:
    st.info("Upload a CSV/XLSX file with OHLC data to continue.")
    st.stop()

# read file
try:
    if uploaded.name.lower().endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed reading file: {e}")
    st.stop()

st.markdown("**Raw columns detected:**")
st.write(list(df_raw.columns))

prepared, meta = detect_and_prepare(df_raw)
if prepared is None:
    st.error("Uploaded file doesn't contain usable OHLC data after normalization.")
    st.markdown("Mapping attempted:")
    st.json(meta["col_map"])
    st.markdown("Original columns:")
    st.write(meta["original_columns"])
    st.stop()

st.success(f"Loaded {len(prepared)} rows. Date index: {isinstance(prepared.index, pd.DatetimeIndex)}")
st.markdown("### Column mapping (canonical -> original)")
st.json(meta)

# default search space (expanded)
default_search_space = {
    "short_ma": list(range(5, 31, 1)),
    "long_ma": list(range(20, 121, 5)),
    "rsi_entry": [20, 25, 30, 35, 40],
    "target_pct": [0.8, 1.0, 1.5, 2.0, 3.0],
    "sl_pct": [0.5, 1.0, 1.5, 2.0, 3.0],
    "atr_mult": [0.8, 1.0, 1.5, 2.0],
    "use_atr_sl": [True, False],
    "target_atr": [False, True],
    "target_atr_mult": [1.0, 1.5, 2.0],
    "max_hold": [3, 5, 7, 10],
    "rsi_period": [14],
    "bb_period": [14, 20],
    "bb_k": [2],
    "vol_period": [10, 20],
    "vol_spike_mult": [1.2, 1.5, 2.0],
    "atr_period": [14],
    "adx_min": [10, 12, 15, 20],
    "confluence_list": [["ma_cross", "ema_trend", "macd_hist_rising", "vol_spike", "bb_breakout", "atr_ok"]],
    "rule_mode": ["strict", "any_k"],
    "min_trades": 4
}

colA, colB = st.columns([2, 1])
with colB:
    stage_a_eval = st.number_input("Stage A random evaluations", min_value=20, max_value=1000, value=180, step=20)
    run_opt = st.button("Run Two-Stage Optimize & Backtest")

if not run_opt:
    st.info("Adjust evaluations or press 'Run Two-Stage Optimize & Backtest' to begin optimization.")
    st.stop()

with st.spinner("Running Stage A (random search) — this may take time..."):
    try:
        best_candidate, all_candidates = two_stage_optimize(df_raw, default_search_space, stage_a_eval, seed=42)
    except Exception as e:
        st.error(f"Optimizer failed: {e}")
        st.stop()

st.markdown("## ✅ Best candidate (auto-selected)")
st.json({"params": best_candidate["params"], "metrics": best_candidate["metrics"], "score": best_candidate["score"]})

# produce backtest/trades for best params
best_params = best_candidate["params"]
trades_df, metrics = backtest_with_confluences(df_raw, best_params)

# Top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net PnL", f"{metrics['net_pnl']:.2f}")
c2.metric("Trades", f"{metrics['n_trades']}")
c3.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
c4.metric("Avg Hold (bars)", f"{metrics['avg_hold']:.2f}")
c5.metric("Trades / 100 bars", f"{metrics['trades_per_100bars']:.2f}")

# Tabs
tab1, tab2, tab3 = st.tabs(["Chart", "Trades & Confluences", "Optimizer Top Picks"])

with tab1:
    st.markdown("### Price chart with entries/exits")
    plot_fig = plot_price_with_trades(prepared, trades_df, best_params)
    st.plotly_chart(plot_fig, use_container_width=True)

    # Live rec based on last bar using best params
    st.markdown("### Live recommendation (based on optimized params)")
    df_ind = add_indicators(prepared, best_params)
    latest = df_ind.iloc[-1]
    prev = df_ind.iloc[-2]
    # update prev macd_hist for rising check
    latest["_macd_hist_prev"] = df_ind["macd_hist"].iloc[-2]
    confs_latest = evaluate_confluences(latest, best_params)
    if best_params["rule_mode"] == "strict":
        enter_now = all([confs_latest.get(k, False) for k in best_params["confluence_list"]])
    else:
        hits = sum([1 for k in best_params["confluence_list"] if confs_latest.get(k, False)])
        enter_now = hits >= best_params.get("any_k", 2)

    if enter_now:
        entry_price = float(latest["open"])
        sl_price = float((latest["close"] - latest["atr"] * best_params["atr_mult"]) if best_params["use_atr_sl"] else latest["close"] * (1 - best_params["sl_pct"] / 100))
        target_price = float(latest["close"] * (1 + best_params["target_pct"] / 100)) if not best_params.get("target_atr", False) else float(latest["close"] + latest["atr"] * best_params.get("target_atr_mult", 1.5))
        confidence_pct = metrics["win_rate"] * 100 if metrics["n_trades"] > 0 else 0.0
        st.success(f"Buy signal TODAY — Confidence: {confidence_pct:.1f}%")
        st.json({"date": str(latest.name), "entry": entry_price, "sl": sl_price, "target": target_price, "confluences": confs_latest})
    else:
        st.info("No buy signal on the last bar with optimized params.")
        st.json({"last_bar_confluences": confs_latest})

with tab2:
    st.markdown("### Trades (descending) with confluence tags")
    if trades_df.empty:
        st.write("No trades generated with optimized params.")
    else:
        display = trades_df.sort_values("entry_date", ascending=False).reset_index(drop=True)
        # expand confluence dicts into readable columns
        def confluence_summary(cdict):
            return ", ".join([k for k, v in cdict.items() if v])

        if "confluences" in display.columns:
            display["confs"] = display["confluences"].apply(lambda x: confluence_summary(x) if isinstance(x, dict) else "")
            display.drop(columns=["confluences"], inplace=True)
        st.dataframe(display)

with tab3:
    st.markdown("### Top optimizer picks (by score) — top 8")
    top_sorted = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:8]
    for i, t in enumerate(top_sorted, 1):
        st.markdown(f"**#{i} — score {t['score']:.2f}**")
        st.json({"params": t["params"], "metrics": t["metrics"]})

st.balloons()
