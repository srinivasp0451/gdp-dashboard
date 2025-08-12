# streamlit_swing_final_with_confidence.py
"""
Auto-Optimized Swing Trading Dashboard — Final (with readable logic & per-trade confidence)

- Keeps the same two-stage optimized backtest engine (confluences) you approved.
- Adds sidebar dropdown (All/Long/Short) and file upload.
- Backtest trades table + Live recommendation table (newest-first) with:
    - readable logic text containing actual indicator values,
    - confluence count (matched / total),
    - per-trade confidence % computed from historical trades with same confluence count,
    - BUY rows green / SELL rows red (pandas Styler).
"""

import re
import random
from typing import Tuple, Dict, Any, List
from math import isnan

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Auto-Optimized Swing — Final (Confidence)", initial_sidebar_state="expanded")

# -----------------------------
# Robust loader / normalization
# -----------------------------
def _clean_name(s: str) -> str:
    s = str(s).lower().strip()
    return re.sub(r'[^a-z0-9]', '', s)


def detect_and_prepare(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    orig_cols = list(df.columns)
    cleaned_map = {_clean_name(c): c for c in orig_cols}

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
        for k, v in cleaned_map.items():
            for cand in orig_candidate_list:
                if _clean_name(cand) in k:
                    return v
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

    # set date index if available
    if col_map["date"] and col_map["date"] in df.columns:
        try:
            df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors="coerce")
            if df[col_map["date"]].notna().sum() > 0:
                df = df.set_index(col_map["date"])
        except Exception:
            pass
    else:
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

    prepared.dropna(subset=["open", "high", "low", "close"], inplace=True)

    if not isinstance(prepared.index, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(prepared.index, errors="coerce")
            if idx.notna().sum() > 0:
                prepared.index = idx
                prepared = prepared[~prepared.index.isna()]
        except Exception:
            pass

    if prepared["close"].dropna().empty:
        return None, {"col_map": col_map, "original_columns": orig_cols}

    prepared.sort_index(inplace=True)
    return prepared, col_map


# -----------------------------
# Indicators & Confluences
# -----------------------------
def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = np.nan

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ma_short"] = df["close"].rolling(params["short_ma"], min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(params["long_ma"], min_periods=1).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / params.get("rsi_period", 14), adjust=False).mean()
    ma_down = down.ewm(alpha=1 / params.get("rsi_period", 14), adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(50, inplace=True)

    macd_fast = df["close"].ewm(span=12, adjust=False).mean()
    macd_slow = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = macd_fast - macd_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(params.get("atr_period", 14), min_periods=1).mean()

    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr14 = tr.rolling(14, min_periods=1).sum()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / tr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / tr14.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.rolling(14, min_periods=1).mean()

    bb_mid = df["close"].rolling(params.get("bb_period", 20), min_periods=1).mean()
    bb_std = df["close"].rolling(params.get("bb_period", 20), min_periods=1).std().fillna(0)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + params.get("bb_k", 2) * bb_std
    df["bb_lower"] = bb_mid - params.get("bb_k", 2) * bb_std

    df["vol_avg"] = df["volume"].rolling(params.get("vol_period", 20), min_periods=1).mean()

    return df


def evaluate_confluences(row, params, direction="long") -> Dict[str, bool]:
    confs = {}
    confs["ma_cross_long"] = (row["ma_short"] > row["ma_long"])
    confs["ema_trend_long"] = (row["ema9"] > row["ema21"])
    confs["bb_breakout_long"] = (row["close"] > row["bb_upper"])
    confs["macd_hist_rising_long"] = (row["macd_hist"] > 0 and row["macd_hist"] > row.get("_macd_hist_prev", 0))
    confs["vol_spike"] = (row["volume"] >= params["vol_spike_mult"] * row["vol_avg"]) if (not np.isnan(row.get("vol_avg", np.nan))) else False
    confs["atr_ok"] = (row["atr"] >= params.get("atr_min", 0))

    if direction == "short":
        confs = {
            "ma_cross_short": (row["ma_short"] < row["ma_long"]),
            "ema_trend_short": (row["ema9"] < row["ema21"]),
            "bb_breakout_short": (row["close"] < row["bb_lower"]),
            "macd_hist_rising_short": (row["macd_hist"] < 0 and row["macd_hist"] < row.get("_macd_hist_prev", 0)),
            "vol_spike": confs["vol_spike"],
            "atr_ok": confs["atr_ok"]
        }
    return confs


# -----------------------------
# Backtester with confluence tagging (supports long/short/all)
# -----------------------------
def backtest_with_confluences(df_raw: pd.DataFrame, params: dict, trade_type: str = "Long") -> Tuple[pd.DataFrame, dict]:
    prepared, meta = detect_and_prepare(df_raw)
    if prepared is None:
        raise ValueError(f"Failed to detect OHLC columns. Mapping attempt: {meta['col_map']}. Original columns: {meta['original_columns']}")

    df = add_indicators(prepared, params)
    trades = []
    position = None

    df["_macd_hist_prev"] = df["macd_hist"].shift(1).fillna(0)

    for i in range(1, len(df)):
        row = df.iloc[i]

        long_allowed = trade_type in ("Long", "All")
        short_allowed = trade_type in ("Short", "All")

        confs_long = evaluate_confluences(row, params, direction="long") if long_allowed else {}
        confs_short = evaluate_confluences(row, params, direction="short") if short_allowed else {}

        if position is None:
            enter_long = False
            enter_short = False

            if long_allowed:
                if params["rule_mode"] == "strict":
                    # require all defined confluences to be True (match keys from confluence_list with _long suffix)
                    enter_long = all(confs_long.get(k if k.endswith("_long") else k + "_long", False) for k in params["confluence_list"])
                else:
                    hits = sum(1 for k in params["confluence_list"] if confs_long.get(k if k.endswith("_long") else k + "_long", False))
                    enter_long = hits >= params.get("any_k", 2)

            if short_allowed:
                if params["rule_mode"] == "strict":
                    enter_short = all(confs_short.get(k if k.endswith("_short") else k + "_short", False) for k in params["confluence_list"])
                else:
                    hits_s = sum(1 for k in params["confluence_list"] if confs_short.get(k if k.endswith("_short") else k + "_short", False))
                    enter_short = hits_s >= params.get("any_k", 2)

            if enter_long or enter_short:
                long_hits = sum(1 for v in confs_long.values() if v) if long_allowed else 0
                short_hits = sum(1 for v in confs_short.values() if v) if short_allowed else 0

                if enter_long and (not enter_short or long_hits >= short_hits):
                    entry_price = row["open"]
                    sl_price = (row["close"] - row["atr"] * params["atr_mult"]) if params["use_atr_sl"] else entry_price * (1 - params["sl_pct"] / 100)
                    target_price = entry_price * (1 + params["target_pct"] / 100) if not params.get("target_atr", False) else entry_price + row["atr"] * params.get("target_atr_mult", 1.5)
                    position = {
                        "direction": "BUY",
                        "entry_index": row.name,
                        "entry_price": float(entry_price),
                        "sl": float(sl_price),
                        "target": float(target_price),
                        "hold_days": 0,
                        "confluences": confs_long,
                        "confluence_count": long_hits,
                        "total_confluences": len([k for k in confs_long.keys() if not k.startswith("vol_spike") or True])
                    }
                else:
                    entry_price = row["open"]
                    sl_price = (row["close"] + row["atr"] * params["atr_mult"]) if params["use_atr_sl"] else entry_price * (1 + params["sl_pct"] / 100)
                    target_price = entry_price * (1 - params["target_pct"] / 100) if not params.get("target_atr", False) else entry_price - row["atr"] * params.get("target_atr_mult", 1.5)
                    position = {
                        "direction": "SELL",
                        "entry_index": row.name,
                        "entry_price": float(entry_price),
                        "sl": float(sl_price),
                        "target": float(target_price),
                        "hold_days": 0,
                        "confluences": confs_short,
                        "confluence_count": short_hits,
                        "total_confluences": len([k for k in confs_short.keys() if not k.startswith("vol_spike") or True])
                    }

        else:
            position["hold_days"] += 1
            exit_price = None
            reason = None
            if position["direction"] == "BUY":
                if row["high"] >= position["target"]:
                    exit_price = position["target"]; reason = "Target"
                elif row["low"] <= position["sl"]:
                    exit_price = position["sl"]; reason = "StopLoss"
            else:
                if row["low"] <= position["target"]:
                    exit_price = position["target"]; reason = "Target"
                elif row["high"] >= position["sl"]:
                    exit_price = position["sl"]; reason = "StopLoss"

            if position["hold_days"] >= params["max_hold"]:
                exit_price = row["close"]; reason = "MaxHold"

            if exit_price is not None:
                pnl = (exit_price - position["entry_price"]) if position["direction"] == "BUY" else (position["entry_price"] - exit_price)
                trades.append({
                    "entry_date": position["entry_index"],
                    "exit_date": row.name,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "hold_days": position["hold_days"],
                    "reason": reason,
                    "confluences": position["confluences"],
                    "confluence_count": position.get("confluence_count", 0),
                    "total_confluences": position.get("total_confluences", 0)
                })
                position = None

    if position is not None:
        last = df.iloc[-1]
        pnl = (last["close"] - position["entry_price"]) if position["direction"] == "BUY" else (position["entry_price"] - last["close"])
        trades.append({
            "entry_date": position["entry_index"],
            "exit_date": last.name,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": float(last["close"]),
            "pnl": float(pnl),
            "hold_days": position["hold_days"],
            "reason": "EOD",
            "confluences": position["confluences"],
            "confluence_count": position.get("confluence_count", 0),
            "total_confluences": position.get("total_confluences", 0)
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
# Optimizer (2-stage)
# -----------------------------
def scoring_fn(metrics: dict, min_trades: int = 4) -> float:
    score = float(metrics.get("net_pnl", 0.0))
    win = float(metrics.get("win_rate", 0.0))
    trades = int(metrics.get("n_trades", 0))
    if trades < min_trades:
        score *= 0.4
    score *= (1.0 + win)
    avg_abs = abs(metrics.get("avg_win", 0.0)) + abs(metrics.get("avg_loss", 0.0))
    if avg_abs < 0.01:
        score *= 0.8
    return score


def random_search_stage(df_raw, search_space: dict, evals: int, seed: int = 42, trade_type: str = "Long") -> List[dict]:
    random.seed(seed)
    results = []
    for _ in range(evals):
        params = {k: random.choice(v) for k, v in search_space.items() if isinstance(v, list)}
        params["rsi_period"] = search_space.get("rsi_period", [14])[0]
        params["confluence_list"] = search_space.get("confluence_list", [["ma_cross", "ema_trend", "macd_hist_rising", "vol_spike", "bb_breakout", "atr_ok"]])[0]
        params["rule_mode"] = random.choice(search_space.get("rule_mode", ["strict", "any_k"]))
        if params["rule_mode"] == "any_k":
            params["any_k"] = random.randint(2, max(2, len(params["confluence_list"])))
        try:
            trades_df, metrics = backtest_with_confluences(df_raw, params, trade_type=trade_type)
            score = scoring_fn(metrics, min_trades=search_space.get("min_trades", 4))
            results.append({"params": params, "metrics": metrics, "score": score, "trades_df": trades_df})
        except Exception:
            continue
    return results


def hill_climb_neighbors(base_params: dict, search_space: dict) -> List[dict]:
    neighbors = []
    def pick_close(option_list, value):
        if value not in option_list:
            return random.choice(option_list)
        idx = option_list.index(value)
        choices = [option_list[max(0, idx - 1)], option_list[min(len(option_list) - 1, idx + 1)]]
        return random.choice(choices)

    for _ in range(15):
        neighbor = base_params.copy()
        for key, space in search_space.items():
            if not isinstance(space, list):
                continue
            if key in neighbor:
                neighbor[key] = pick_close(space, neighbor[key])
            else:
                neighbor[key] = random.choice(space)
        if neighbor.get("rule_mode") == "any_k" and "any_k" not in neighbor:
            neighbor["any_k"] = random.randint(2, max(2, len(neighbor.get("confluence_list", []))))
        neighbors.append(neighbor)
    return neighbors


def two_stage_optimize(df_raw: pd.DataFrame, search_space: dict, stage_a_eval: int = 120, seed: int = 42, trade_type: str = "Long") -> Tuple[dict, List[dict]]:
    stage_a_results = random_search_stage(df_raw, search_space, evals=stage_a_eval, seed=seed, trade_type=trade_type)
    if not stage_a_results:
        raise RuntimeError("Stage A found no valid parameter sets.")
    top_k = sorted(stage_a_results, key=lambda x: x["score"], reverse=True)[:8]
    best = top_k[0]
    all_candidates = stage_a_results.copy()
    for entry in top_k:
        base_params = entry["params"]
        neighbors = hill_climb_neighbors(base_params, search_space)
        for n in neighbors:
            try:
                trades_df, metrics = backtest_with_confluences(df_raw, n, trade_type=trade_type)
                score = scoring_fn(metrics, min_trades=search_space.get("min_trades", 4))
                all_candidates.append({"params": n, "metrics": metrics, "score": score, "trades_df": trades_df})
                if score > best["score"]:
                    best = {"params": n, "metrics": metrics, "score": score, "trades_df": trades_df}
            except Exception:
                continue
    return best, all_candidates


# -----------------------------
# Plot helpers & UI helpers
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
        fig.add_trace(go.Scatter(x=buys[buys["direction"]=="BUY"]["entry_date"], y=buys[buys["direction"]=="BUY"]["entry_price"],
                                 mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"), name="Entry BUY"))
        fig.add_trace(go.Scatter(x=buys[buys["direction"]=="BUY"]["exit_date"], y=buys[buys["direction"]=="BUY"]["exit_price"],
                                 mode="markers", marker=dict(symbol="x", size=10, color="green"), name="Exit BUY"))
        fig.add_trace(go.Scatter(x=buys[buys["direction"]=="SELL"]["entry_date"], y=buys[buys["direction"]=="SELL"]["entry_price"],
                                 mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"), name="Entry SELL"))
        fig.add_trace(go.Scatter(x=buys[buys["direction"]=="SELL"]["exit_date"], y=buys[buys["direction"]=="SELL"]["exit_price"],
                                 mode="markers", marker=dict(symbol="x", size=10, color="red"), name="Exit SELL"))

    fig.update_layout(template="plotly_dark", height=650, title="Price & Trades")
    return fig


def build_readable_logic(row, params, direction):
    """
    Create a human-readable logic string containing actual indicator values for the given row.
    Example: 'EMA9(1520.4)>EMA21(1508.3); MA20(1510.2)>MA50(1500.6); ATR=12.4; RSI=56.2; MACD_hist=0.45'
    """
    parts = []
    try:
        if direction == "BUY":
            parts.append(f"EMA9={row['ema9']:.2f} > EMA21={row['ema21']:.2f}" if not np.isnan(row['ema9']) and not np.isnan(row['ema21']) else "")
            parts.append(f"MA{params['short_ma']}={row['ma_short']:.2f} > MA{params['long_ma']}={row['ma_long']:.2f}" if not np.isnan(row['ma_short']) and not np.isnan(row['ma_long']) else "")
            parts.append(f"ATR={row['atr']:.2f}")
            parts.append(f"RSI={row['rsi']:.1f}")
            parts.append(f"MACD_hist={row['macd_hist']:.4f}")
            parts.append(f"Close={row['close']:.2f}")
        else:
            parts.append(f"EMA9={row['ema9']:.2f} < EMA21={row['ema21']:.2f}" if not np.isnan(row['ema9']) and not np.isnan(row['ema21']) else "")
            parts.append(f"MA{params['short_ma']}={row['ma_short']:.2f} < MA{params['long_ma']}={row['ma_long']:.2f}" if not np.isnan(row['ma_short']) and not np.isnan(row['ma_long']) else "")
            parts.append(f"ATR={row['atr']:.2f}")
            parts.append(f"RSI={row['rsi']:.1f}")
            parts.append(f"MACD_hist={row['macd_hist']:.4f}")
            parts.append(f"Close={row['close']:.2f}")
    except Exception:
        pass
    parts = [p for p in parts if p]
    return "; ".join(parts)


def compute_confidence_for_live(trades_df: pd.DataFrame, matched_count: int):
    """
    Confidence: proportion of historical trades (from trades_df) that had exactly matched_count confluence matches and were winners.
    If insufficient historical samples (<3), fallback to overall win_rate.
    Returns percentage (0-100).
    """
    if trades_df is None or trades_df.empty:
        return 0.0
    # use confluence_count field from historical trades
    if "confluence_count" in trades_df.columns:
        same = trades_df[trades_df["confluence_count"] == matched_count]
        if len(same) >= 3:
            return float((same["pnl"] > 0).mean() * 100)
    # fallback to overall win rate
    return float((trades_df["pnl"] > 0).mean() * 100) if not trades_df.empty else 0.0


def style_trade_rows(df_display: pd.DataFrame):
    """
    Return a pandas Styler that colors rows green for BUY and red for SELL.
    """
    def row_color(row):
        return ["background-color: #093; color: white; font-weight:600" if row["Signal"] == "BUY"
                else "background-color: #900; color:white; font-weight:600" if row["Signal"] == "SELL"
                else "" for _ in row]
    return df_display.style.apply(row_color, axis=1)


# -----------------------------
# Streamlit UI (sidebar + main)
# -----------------------------
st.title("🚀 Auto-Optimized Swing Trading Dashboard — Final (Per-Trade Confidence & Readable Logic)")

with st.sidebar:
    st.header("Upload & Controls")
    uploaded = st.file_uploader("Upload OHLC CSV / XLSX", type=["csv", "xlsx"])
    trade_type = st.selectbox("Trade type", ["All", "Long", "Short"])
    stage_a_eval = st.number_input("Stage A evaluations", min_value=20, max_value=1000, value=180, step=20)
    run_opt = st.button("Run Two-Stage Optimize & Backtest")

if uploaded is None:
    st.info("Upload your OHLC CSV/XLSX in the sidebar to begin.")
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

st.success(f"Loaded {len(prepared)} rows — date index? {isinstance(prepared.index, pd.DatetimeIndex)}")
st.markdown("### Column mapping (canonical -> original)")
st.json(meta)

# default search space (same as before)
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

if not run_opt:
    st.info("Set evaluations and click 'Run Two-Stage Optimize & Backtest' in the sidebar when ready.")
    st.stop()

with st.spinner("Running optimizer (Stage A random search + Stage B hill-climb)..."):
    try:
        best_candidate, all_candidates = two_stage_optimize(df_raw, default_search_space, stage_a_eval, seed=42, trade_type=trade_type)
    except Exception as e:
        st.error(f"Optimizer failed: {e}")
        st.stop()

st.markdown("## ✅ Best candidate (auto-selected)")
st.json({"params": best_candidate["params"], "metrics": best_candidate["metrics"], "score": best_candidate["score"]})

best_params = best_candidate["params"]
trades_df, metrics = backtest_with_confluences(df_raw, best_params, trade_type=trade_type)

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

    st.markdown("### Latest Recommendations (newest first)")

    # Build live rec based on last bar and calculate readable logic / confluences matched
    df_ind = add_indicators(prepared, best_params)
    df_ind["_macd_hist_prev"] = df_ind["macd_hist"].shift(1).fillna(0)
    latest = df_ind.iloc[-1]

    # Evaluate both directions
    confs_long = evaluate_confluences(latest, best_params, direction="long")
    confs_short = evaluate_confluences(latest, best_params, direction="short")
    # Count matched confluences based on confluence_list
    total_confs = len(best_params.get("confluence_list", []))
    matched_long = sum(1 for k in best_params["confluence_list"] if confs_long.get(k if k.endswith("_long") else k + "_long", False))
    matched_short = sum(1 for k in best_params["confluence_list"] if confs_short.get(k if k.endswith("_short") else k + "_short", False))

    # Decide present live signals
    live_signals = []
    if trade_type in ("Long", "All"):
        if (best_params["rule_mode"] == "strict" and matched_long == total_confs) or (best_params["rule_mode"] == "any_k" and matched_long >= best_params.get("any_k", 2)):
            # build a live-buy recommendation
            entry = float(latest["open"])
            sl = float((latest["close"] - latest["atr"] * best_params["atr_mult"]) if best_params["use_atr_sl"] else latest["close"] * (1 - best_params["sl_pct"] / 100))
            target = float(latest["close"] * (1 + best_params["target_pct"] / 100)) if not best_params.get("target_atr", False) else float(latest["close"] + latest["atr"] * best_params.get("target_atr_mult", 1.5))
            # compute per-trade confidence: historical trades with same confluence_count
            conf_pct = compute_confidence_for_live(trades_df, matched_long)
            logic_text = build_readable_logic(latest, best_params, "BUY")
            live_signals.append({
                "Date": latest.name,
                "Signal": "BUY",
                "Entry": entry,
                "Target": target,
                "SL": sl,
                "ConfluencesMatched": f"{matched_long}/{total_confs}",
                "ConfidencePct": round(conf_pct, 1),
                "Logic": logic_text
            })

    if trade_type in ("Short", "All"):
        if (best_params["rule_mode"] == "strict" and matched_short == total_confs) or (best_params["rule_mode"] == "any_k" and matched_short >= best_params.get("any_k", 2)):
            entry = float(latest["open"])
            sl = float((latest["close"] + latest["atr"] * best_params["atr_mult"]) if best_params["use_atr_sl"] else latest["close"] * (1 + best_params["sl_pct"] / 100))
            target = float(latest["close"] * (1 - best_params["target_pct"] / 100)) if not best_params.get("target_atr", False) else float(latest["close"] - latest["atr"] * best_params.get("target_atr_mult", 1.5))
            conf_pct = compute_confidence_for_live(trades_df, matched_short)
            logic_text = build_readable_logic(latest, best_params, "SELL")
            live_signals.append({
                "Date": latest.name,
                "Signal": "SELL",
                "Entry": entry,
                "Target": target,
                "SL": sl,
                "ConfluencesMatched": f"{matched_short}/{total_confs}",
                "ConfidencePct": round(conf_pct, 1),
                "Logic": logic_text
            })

    # Display live signals as a styled DataFrame
    if not live_signals:
        st.info("No live recommendation on the last bar with optimized params.")
    else:
        live_df = pd.DataFrame(live_signals).sort_values("Date", ascending=False).reset_index(drop=True)
        # style rows
        st.subheader("Live Recommendation")
        st.write("Confidence is computed from historical trades with the same confluence count; fallback to overall win-rate if insufficient history.")
        sty = style_trade_rows(live_df.rename(columns={"Signal": "Signal"}))
        st.dataframe(sty, use_container_width=True)

with tab2:
    st.markdown("### Trades (descending) with confluence tags")
    if trades_df.empty:
        st.write("No trades generated with optimized params.")
    else:
        display = trades_df.sort_values("entry_date", ascending=False).reset_index(drop=True)
        def confluence_summary(cdict):
            return ", ".join([k for k, v in cdict.items() if v]) if isinstance(cdict, dict) else ""
        if "confluences" in display.columns:
            display["confs"] = display["confluences"].apply(lambda x: confluence_summary(x) if isinstance(x, dict) else "")
            display = display.rename(columns={"direction": "Signal", "entry_price": "Entry", "exit_price": "Exit", "pnl": "PnL", "hold_days": "Hold"})
            # keep columns order
            display = display[["entry_date", "Signal", "Entry", "Exit", "PnL", "Hold", "confluence_count", "confs", "reason"]]
            display = display.rename(columns={"entry_date": "EntryDate", "confluence_count": "ConfluenceCount", "confs": "Confluences", "reason": "ExitReason"})
        st.dataframe(display)

with tab3:
    st.markdown("### Top optimizer picks (by score) — top 8")
    top_sorted = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:8]
    for i, t in enumerate(top_sorted, 1):
        st.markdown(f"**#{i} — score {t['score']:.2f}**")
        st.json({"params": t["params"], "metrics": t["metrics"]})

st.balloons()
