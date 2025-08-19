                                                "target_atr": True,
   # app.py
"""
Final Streamlit trading dashboard (auto-optimized, confluences, long/short/all)
- Sidebar: Upload + Trade type (All/Long/Short) + Evaluations
- Tabs: Backtest Results | Live Recommendations
- Backtest uses the advanced two-stage optimizer + confluence rules (same engine you approved)
- Both Backtest and Live recommendations include human readable logic, SL/Target reason,
  confluence counts, dynamic confidence, and color-coded BUY/SELL rows.
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
import io

st.set_page_config(layout="wide", page_title="Swing Trading Dashboard (Final)", initial_sidebar_state="expanded")

# -------------------------
# Utility: robust loader
# -------------------------
def _clean_name(s: str) -> str:
    s = str(s or "").lower().strip()
    return re.sub(r'[^a-z0-9]', '', s)

def detect_and_prepare(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Detect OHLCV/date columns (fuzzy), coerce numeric, return prepared df and mapping."""
    df = df_raw.copy()
    orig_cols = list(df.columns)
    cleaned_map = {_clean_name(c): c for c in orig_cols}

    open_cands = ["open", "openprice", "o"]
    high_cands = ["high", "highprice", "h"]
    low_cands = ["low", "lowprice", "l"]
    close_cands = ["close", "closeprice", "ltp", "last", "price", "adjclose"]
    vol_cands = ["volume", "vol", "quantity", "tradedqty"]
    date_cands = ["date", "datetime", "timestamp", "time"]

    def find(cands):
        for cand in cands:
            key = _clean_name(cand)
            if key in cleaned_map:
                return cleaned_map[key]
        for k, v in cleaned_map.items():
            for cand in cands:
                if _clean_name(cand) in k:
                    return v
        for orig in orig_cols:
            low = str(orig).lower().replace(' ', '')
            for cand in cands:
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

    # try set date index
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
        if orig and orig in df.columns:
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

# -------------------------
# Indicators & confluences
# -------------------------
def add_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for c in ["open","high","low","close","volume"]:
        if c not in df.columns:
            df[c] = np.nan

    # EMAs / MAs
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ma_short"] = df["close"].rolling(params["short_ma"], min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(params["long_ma"], min_periods=1).mean()

    # RSI
    per = params.get("rsi_period", 14)
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/per, adjust=False).mean()
    ma_down = down.ewm(alpha=1/per, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(50, inplace=True)

    # MACD + hist
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

    df["vol_avg"] = df["volume"].rolling(params.get("vol_period", 20), min_periods=1).mean()
    return df

def evaluate_confluences_for_direction(row: pd.Series, params: Dict[str,Any], direction: str="long") -> Dict[str,bool]:
    # returns dictionary of confluence boolean flags specific to direction
    if direction == "long":
        confs = {
            "ma_cross": row["ma_short"] > row["ma_long"],
            "ema_trend": row["ema9"] > row["ema21"],
            "bb_breakout": row["close"] > row["bb_upper"],
            "macd_hist_rising": (row["macd_hist"] > 0 and row["macd_hist"] > row.get("_macd_hist_prev", 0)),
            "vol_spike": (row["volume"] >= params.get("vol_spike_mult", 1.5) * row.get("vol_avg", 0)) if not isnan(row.get("vol_avg", np.nan)) else False,
            "atr_ok": row.get("atr", 0) >= params.get("atr_min", 0)
        }
    else:
        confs = {
            "ma_cross": row["ma_short"] < row["ma_long"],
            "ema_trend": row["ema9"] < row["ema21"],
            "bb_breakout": row["close"] < row["bb_lower"],
            "macd_hist_rising": (row["macd_hist"] < 0 and row["macd_hist"] < row.get("_macd_hist_prev", 0)),
            "vol_spike": (row["volume"] >= params.get("vol_spike_mult", 1.5) * row.get("vol_avg", 0)) if not isnan(row.get("vol_avg", np.nan)) else False,
            "atr_ok": row.get("atr", 0) >= params.get("atr_min", 0)
        }
    return confs

# -------------------------
# Backtester (long/short/all)
# -------------------------
def compute_sl_target_reasons(entry_price: float, row: pd.Series, params: Dict[str,Any], direction: str):
    """Return (sl_value, sl_reason_text, target_value, target_reason_text, adjusted_flag)"""
    adjusted = False
    # default use ATR-based SL if configured, else percentage
    if params.get("use_atr_sl", False):
        atr = float(row.get("atr", 0.0) or 0.0)
        sl = (entry_price - atr * params.get("atr_mult",1.5)) if direction=="long" else (entry_price + atr * params.get("atr_mult",1.5))
        sl_reason = f"SL = Entry {'−' if direction=='long' else '+'} ATR×{params.get('atr_mult',1.5)} = {entry_price:.2f} {'−' if direction=='long' else '+'} ({atr:.3f}×{params.get('atr_mult',1.5)})"
    else:
        sl = (entry_price * (1 - params.get("sl_pct",1.5)/100)) if direction=="long" else (entry_price * (1 + params.get("sl_pct",1.5)/100))
        sl_reason = f"SL = Entry {'−' if direction=='long' else '+'} {params.get('sl_pct',1.5)}% = {sl:.2f}"

    # target: either fixed percent or ATR-mult
    if params.get("target_atr", False):
        atr = float(row.get("atr", 0.0) or 0.0)
        target = (entry_price + atr * params.get("target_atr_mult",1.5)) if direction=="long" else (entry_price - atr * params.get("target_atr_mult",1.5))
        target_reason = f"Target = Entry {'+' if direction=='long' else '−'} ATR×{params.get('target_atr_mult',1.5)} = {entry_price:.2f} {'+' if direction=='long' else '−'} ({atr:.3f}×{params.get('target_atr_mult',1.5)})"
    else:
        target = (entry_price * (1 + params.get("target_pct",2.0)/100)) if direction=="long" else (entry_price * (1 - params.get("target_pct",2.0)/100))
        target_reason = f"Target = Entry {'+' if direction=='long' else '−'} {params.get('target_pct',2.0)}% = {target:.2f}"

    # Safety checks: ensure SL & target are on correct sides of entry; if not, fix and mark adjusted
    if direction == "long":
        if sl >= entry_price:
            # force below
            atr = float(row.get("atr", 0.0) or 0.0)
            sl = entry_price - abs(atr * params.get("atr_mult",1.5))
            sl_reason += " (adjusted)"
            adjusted = True
        if target <= entry_price:
            atr = float(row.get("atr", 0.0) or 0.0)
            target = entry_price + abs(atr * params.get("target_atr_mult", params.get("atr_mult",1.5)))
            target_reason += " (adjusted)"
            adjusted = True
    else:
        if sl <= entry_price:
            atr = float(row.get("atr", 0.0) or 0.0)
            sl = entry_price + abs(atr * params.get("atr_mult",1.5))
            sl_reason += " (adjusted)"
            adjusted = True
        if target >= entry_price:
            atr = float(row.get("atr", 0.0) or 0.0)
            target = entry_price - abs(atr * params.get("target_atr_mult", params.get("atr_mult",1.5)))
            target_reason += " (adjusted)"
            adjusted = True

    return float(sl), sl_reason, float(target), target_reason, adjusted

def backtest_with_confluences(df_raw: pd.DataFrame, params: Dict[str,Any], trade_type: str="Long") -> Tuple[pd.DataFrame, dict]:
    prepared, meta = detect_and_prepare(df_raw)
    if prepared is None:
        raise ValueError(f"Failed to detect OHLC columns. Mapping attempt: {meta['col_map']}. Original columns: {meta['original_columns']}")

    df = add_indicators(prepared, params)
    trades = []
    position = None
    df["_macd_hist_prev"] = df["macd_hist"].shift(1).fillna(0)

    for i in range(1, len(df)):
        row = df.iloc[i]

        long_allowed = trade_type in ("Long","All")
        short_allowed = trade_type in ("Short","All")

        if position is None:
            enter_long=False; enter_short=False
            confs_long = evaluate_confluences_for_direction(row, params, "long") if long_allowed else {}
            confs_short = evaluate_confluences_for_direction(row, params, "short") if short_allowed else {}

            # evaluate rule_mode
            def decide(confs):
                if params["rule_mode"] == "strict":
                    return all([confs.get(k, False) for k in params["confluence_list"]])
                else:
                    hits = sum([1 for k in params["confluence_list"] if confs.get(k, False)])
                    return hits >= params.get("any_k", 2)

            if long_allowed:
                enter_long = decide(confs_long)
            if short_allowed:
                enter_short = decide(confs_short)

            # if both, choose by number of confluences
            if enter_long or enter_short:
                long_hits = sum(1 for v in confs_long.values() if v) if long_allowed else 0
                short_hits = sum(1 for v in confs_short.values() if v) if short_allowed else 0

                if enter_long and (not enter_short or long_hits >= short_hits):
                    direction = "BUY"
                    entry_price = float(row["open"])
                    sl, sl_reason, target, target_reason, adjusted = compute_sl_target_reasons(entry_price, row, params, "long")
                    position = {
                        "direction":"BUY","entry_index":row.name,"entry_price":entry_price,
                        "sl":sl,"target":target,"hold_days":0,"confluences":confs_long,
                        "sl_reason":sl_reason,"target_reason":target_reason,"adjusted":adjusted
                    }
                else:
                    direction = "SELL"
                    entry_price = float(row["open"])
                    sl, sl_reason, target, target_reason, adjusted = compute_sl_target_reasons(entry_price, row, params, "short")
                    position = {
                        "direction":"SELL","entry_index":row.name,"entry_price":entry_price,
                        "sl":sl,"target":target,"hold_days":0,"confluences":confs_short,
                        "sl_reason":sl_reason,"target_reason":target_reason,"adjusted":adjusted
                    }

        else:
            position["hold_days"] += 1
            exit_price=None; reason=None
            if position["direction"]=="BUY":
                if row["high"] >= position["target"]:
                    exit_price = position["target"]; reason="Target"
                elif row["low"] <= position["sl"]:
                    exit_price = position["sl"]; reason="StopLoss"
            else:
                if row["low"] <= position["target"]:
                    exit_price = position["target"]; reason="Target"
                elif row["high"] >= position["sl"]:
                    exit_price = position["sl"]; reason="StopLoss"

            if position["hold_days"] >= params.get("max_hold",5) and exit_price is None:
                exit_price = row["close"]; reason="MaxHold"

            if exit_price is not None:
                pnl = (exit_price - position["entry_price"]) if position["direction"]=="BUY" else (position["entry_price"] - exit_price)
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
                    "sl_reason": position.get("sl_reason",""),
                    "target_reason": position.get("target_reason",""),
                    "adjusted": position.get("adjusted", False)
                })
                position = None

    # close open
    if position is not None:
        last = df.iloc[-1]
        pnl = (last["close"] - position["entry_price"]) if position["direction"]=="BUY" else (position["entry_price"] - last["close"])
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
            "sl_reason": position.get("sl_reason",""),
            "target_reason": position.get("target_reason",""),
            "adjusted": position.get("adjusted", False)
        })

    trades_df = pd.DataFrame(trades)
    metrics = {"net_pnl":0.0,"n_trades":0,"win_rate":0.0,"avg_win":0.0,"avg_loss":0.0,"avg_hold":0.0,"trades_per_100bars":0.0}
    if not trades_df.empty:
        metrics["net_pnl"] = trades_df["pnl"].sum()
        metrics["n_trades"] = len(trades_df)
        metrics["win_rate"] = (trades_df["pnl"]>0).mean()
        metrics["avg_win"] = trades_df[trades_df["pnl"]>0]["pnl"].mean() if (trades_df["pnl"]>0).any() else 0.0
        metrics["avg_loss"] = trades_df[trades_df["pnl"]<=0]["pnl"].mean() if (trades_df["pnl"]<=0).any() else 0.0
        metrics["avg_hold"] = trades_df["hold_days"].mean()
        metrics["trades_per_100bars"] = metrics["n_trades"]/max(1,len(prepared))*100

    return trades_df, metrics

# -------------------------
# Optimizer (two-stage random + hill-climb)
# -------------------------
def scoring_fn(metrics: dict, min_trades: int=4) -> float:
    score = float(metrics.get("net_pnl",0.0))
    win = float(metrics.get("win_rate",0.0))
    trades = int(metrics.get("n_trades",0))
    if trades < min_trades:
        score *= 0.4
    score *= (1.0 + win)
    avg_abs = abs(metrics.get("avg_win",0.0)) + abs(metrics.get("avg_loss",0.0))
    if avg_abs < 0.01:
        score *= 0.8
    return score

def random_search_stage(df_raw: pd.DataFrame, search_space: Dict[str,Any], evals:int, seed:int=42, trade_type:str="Long") -> List[dict]:
    random.seed(seed)
    results=[]
    for _ in range(evals):
        params = {k: random.choice(v) for k,v in search_space.items() if isinstance(v,list)}
        params["rsi_period"] = search_space.get("rsi_period",[14])[0]
        params["confluence_list"] = search_space.get("confluence_list", [["ma_cross","ema_trend","macd_hist_rising","vol_spike","bb_breakout","atr_ok"]])[0]
        params["rule_mode"] = random.choice(search_space.get("rule_mode",["strict","any_k"]))
        if params["rule_mode"]=="any_k":
            params["any_k"] = random.randint(2,max(2,len(params["confluence_list"])))
        try:
            trades_df, metrics = backtest_with_confluences(df_raw, params, trade_type=trade_type)
            score = scoring_fn(metrics, min_trades=search_space.get("min_trades",4))
            results.append({"params":params,"metrics":metrics,"score":score,"trades_df":trades_df})
        except Exception:
            continue
    return results

def hill_climb_neighbors(base_params:dict, search_space:dict) -> List[dict]:
    neighbors=[]
    def pick_close(opts, val):
        if val not in opts:
            return random.choice(opts)
        idx = opts.index(val)
        return opts[max(0, idx-1)] if random.random()<0.5 else opts[min(len(opts)-1, idx+1)]
    for _ in range(15):
        n=base_params.copy()
        for k,space in search_space.items():
            if not isinstance(space,list): continue
            if k in n:
                n[k] = pick_close(space, n[k])
            else:
                n[k] = random.choice(space)
        if n.get("rule_mode")=="any_k" and "any_k" not in n:
            n["any_k"] = random.randint(2, max(2, len(n.get("confluence_list",[]))))
        neighbors.append(n)
    return neighbors

def two_stage_optimize(df_raw:pd.DataFrame, search_space:dict, stage_a_eval:int=120, seed:int=42, trade_type:str="Long"):
    stage_a_results = random_search_stage(df_raw, search_space, evals=stage_a_eval, seed=seed, trade_type=trade_type)
    if not stage_a_results:
        raise RuntimeError("Stage A found no valid parameter sets.")
    top_k = sorted(stage_a_results, key=lambda x: x["score"], reverse=True)[:8]
    best = top_k[0]
    all_candidates = stage_a_results.copy()
    for entry in top_k:
        base = entry["params"]
        neighbors = hill_climb_neighbors(base, search_space)
        for n in neighbors:
            try:
                trades_df, metrics = backtest_with_confluences(df_raw, n, trade_type=trade_type)
                score = scoring_fn(metrics, min_trades=search_space.get("min_trades",4))
                all_candidates.append({"params":n,"metrics":metrics,"score":score,"trades_df":trades_df})
                if score > best["score"]:
                    best = {"params":n,"metrics":metrics,"score":score,"trades_df":trades_df}
            except Exception:
                continue
    return best, all_candidates

# -------------------------
# Plot & rendering helpers
# -------------------------
def plot_price_with_trades(prepared_df:pd.DataFrame, trades_df:pd.DataFrame, params:dict):
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

def trades_to_display_df(trades_df:pd.DataFrame):
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    # create readable logic & confluence count & confidence placeholder (filled later)
    def confluence_summary(cdict):
        if not isinstance(cdict, dict): return ""
        return ", ".join([k for k,v in cdict.items() if v])
    df["confluence_count"] = df["confluences"].apply(lambda x: sum(1 for v in x.values()) if isinstance(x, dict) else 0)
    df["confluence_total"] = df["confluences"].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    df["confluences_text"] = df["confluences"].apply(confluence_summary)
    df["logic"] = df.apply(lambda r: readable_logic_from_row(r), axis=1)
    # add target/sl reasons columns already present
    return df

def readable_logic_from_row(row):
    # Build a human-friendly logic string with key indicator values
    # We may not have indicator snapshot in the trade row, so produce best-effort
    confs = row.get("confluences", {})
    parts=[]
    if isinstance(confs, dict):
        if row["direction"]=="BUY":
            # attempt to include values from confluences if included
            for k in ["ma_cross","ema_trend","macd_hist_rising","bb_breakout","vol_spike","atr_ok"]:
                if confs.get(k):
                    parts.append(k.replace("_"," ").title())
        else:
            for k in ["ma_cross","ema_trend","macd_hist_rising","bb_breakout","vol_spike","atr_ok"]:
                if confs.get(k):
                    parts.append(k.replace("_"," ").title())
    return "; ".join(parts)

def style_trades_df_for_display(df_display:pd.DataFrame):
    if df_display.empty:
        return df_display
    # select and rename columns for display
    disp = df_display.copy()
    disp["Date"] = disp["entry_date"].dt.strftime("%Y-%m-%d %H:%M")
    disp["Direction"] = disp["direction"]
    disp["Entry"] = disp["entry_price"].map(lambda x: f"{x:.2f}")
    disp["Target"] = disp["exit_price"].map(lambda x: f"{x:.2f}")
    disp["SL"] = (disp["entry_price"] - disp["pnl"]).map(lambda x: f"{x:.2f}")  # approximate original sl (entry - pnl for BUY) - keep original sl_reason column too
    disp["PnL"] = disp["pnl"].map(lambda x: f"{x:.2f}")
    disp["Hold"] = disp["hold_days"]
    disp["Confs"] = disp["confluence_count"].astype(str) + "/" + disp["confluence_total"].astype(str)
    disp["Logic"] = disp["logic"]
    disp["Target Reason"] = disp["target_reason"]
    disp["SL Reason"] = disp["sl_reason"]
    disp["Confidence"] = disp.get("confidence_pct", None)
    # order columns
    order = ["Date","Direction","Entry","Target","SL","Target Reason","SL Reason","Confs","Logic","Confidence","PnL","Hold"]
    disp = disp[order]
    # styling via pandas Styler -> green/red rows
    def highlight_row(row):
        color = ""
        if row["Direction"] == "BUY":
            color = "background-color: #0a6b2f; color: white;"
        else:
            color = "background-color: #8b0b0b; color: white;"
        return [color if col in ["Direction","Entry","Target","SL","Target Reason","SL Reason","Confs","Logic","Confidence","PnL","Hold","Date"] else "" for col in row.index]
    sty = disp.style
    # apply per-row coloring on Direction column; use apply to whole
    def row_style(r):
        return ["background-color: #0a6b2f; color: white;" if r["Direction"]=="BUY" and c!='Date' else
                "background-color: #8b0b0b; color: white;" if r["Direction"]=="SELL" and c!='Date' else ""
                for c in r.index]
    try:
        sty = sty.apply(lambda r: row_style(r), axis=1)
    except Exception:
        pass
    return sty

# -------------------------
# Confidence helper
# -------------------------
def dynamic_confidence_for_live(trades_df:pd.DataFrame, live_direction:str, confluence_count:int):
    """Compute confidence% as win rate of past trades that match direction and same confluence_count.
    Fallback to overall win rate if no match."""
    if trades_df is None or trades_df.empty:
        return 0.0
    df = trades_df.copy()
    df["con_count"] = df["confluences"].apply(lambda x: sum(1 for v in x.values()) if isinstance(x, dict) else 0)
    subset = df[(df["direction"]==live_direction) & (df["con_count"]==confluence_count)]
    if not subset.empty:
        return float((subset["pnl"]>0).mean() * 100)
    else:
        # fallback to overall direction win rate
        subs2 = df[df["direction"]==live_direction]
        if not subs2.empty:
            return float((subs2["pnl"]>0).mean() * 100)
    # final fallback
    return float((df["pnl"]>0).mean() * 100) if not df.empty else 0.0

# -------------------------
# Sidebar and main UI
# -------------------------
st.title("🚀 Swing Trading Dashboard — Final (Auto-optimize & Confluences)")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload OHLC CSV or XLSX", type=["csv","xlsx"])
    trade_type = st.selectbox("Trade Type", ["All","Long","Short"], index=1)
    stage_a_eval = st.number_input("Stage A evaluations (random search)", min_value=20, max_value=1000, value=250, step=20)
    run_opt = st.button("Run Optimize & Backtest")
    st.markdown("---")
    st.caption("Notes: Optimizer auto-selects EMA/ATR/etc. parameters. Live confidence uses similar historical trades.")

if uploaded is None:
    st.info("Upload a file to start (use your previously working CSV).")
    st.stop()

# read file
try:
    if uploaded.name.lower().endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.markdown("**Detected columns (raw):**")
st.write(list(df_raw.columns))

prepared, meta = detect_and_prepare(df_raw)
if prepared is None:
    st.error("Uploaded file doesn't contain usable OHLC data after normalization.")
    st.json(meta)
    st.stop()

st.success(f"Loaded {len(prepared)} rows — Date index? {isinstance(prepared.index, pd.DatetimeIndex)}")
st.markdown("### Column mapping")
st.json(meta)

# search space (same as before)
default_search_space = {
    "short_ma": list(range(5,31,1)),
    "long_ma": list(range(20,121,5)),
    "rsi_entry": [20,25,30,35,40],
    "target_pct": [0.8,1.0,1.5,2.0,3.0],
    "sl_pct": [0.5,1.0,1.5,2.0,3.0],
    "atr_mult": [0.8,1.0,1.5,2.0],
    "use_atr_sl": [True, False],
    "target_atr": [False, True],
    "target_atr_mult": [1.0,1.5,2.0],
    "max_hold": [3,5,7,10],
    "rsi_period": [14],
    "bb_period": [14,20],
    "bb_k": [2],
    "vol_period": [10,20],
    "vol_spike_mult": [1.2,1.5,2.0],
    "atr_period": [14],
    "adx_min": [10,12,15,20],
    "confluence_list": [["ma_cross","ema_trend","macd_hist_rising","vol_spike","bb_breakout","atr_ok"]],
    "rule_mode": ["strict","any_k"],
    "min_trades": 4
}

if not run_opt:
    st.info("Set evaluation count and click 'Run Optimize & Backtest' in the sidebar.")
    st.stop()

with st.spinner("Running optimizer (Stage A random search + Stage B hill-climb)..."):
    try:
        best_candidate, all_candidates = two_stage_optimize(df_raw, default_search_space, stage_a_eval, seed=42, trade_type=trade_type)
    except Exception as e:
        st.error(f"Optimizer failed: {e}")
        st.stop()

st.markdown("## ✅ Best candidate & its metrics")
st.json({"params": best_candidate["params"], "metrics": best_candidate["metrics"], "score": round(best_candidate["score"],3)})

best_params = best_candidate["params"]
trades_df, metrics = backtest_with_confluences(df_raw, best_params, trade_type=trade_type)

# fill trades DataFrame with readable fields & confidence placeholders
display_df = trades_to_display_df(trades_df)
# compute confidence for each historical trade (confidence not used for backtest rows, but show same metric)
if not trades_df.empty:
    display_df["confidence_pct"] = display_df.apply(lambda r: dynamic_confidence_for_live(trades_df, r["direction"], r["confluence_count"]) , axis=1)
else:
    display_df["confidence_pct"] = 0.0

# Metrics cards (cleaner look)
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Net PnL", f"{metrics['net_pnl']:.2f}")
c2.metric("Trades", f"{metrics['n_trades']}")
c3.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
c4.metric("Avg Hold (bars)", f"{metrics['avg_hold']:.2f}")
c5.metric("Trades / 100 bars", f"{metrics['trades_per_100bars']:.2f}")

# Tabs: Backtest Results & Live Recommendations
tab1, tab2 = st.tabs(["Backtest Results", "Live Recommendations"])

with tab1:
    st.markdown("### Backtest trades (newest first)")
    if trades_df is None or trades_df.empty:
        st.info("No trades generated with optimized params.")
    else:
        # prepare display table, include confluence count, reason columns
        df_disp = display_df.copy().sort_values("entry_date", ascending=False)
        # show columns of interest and add confidence column
        df_for_table = df_disp[[
            "entry_date","direction","entry_price","exit_price","pnl","hold_days",
            "confluence_count","confluence_total","confluences_text","target_reason","sl_reason","adjusted","confidence_pct"
        ]].rename(columns={
            "entry_date":"Date","direction":"Direction","entry_price":"Entry",
            "exit_price":"Exit","pnl":"PnL","hold_days":"Hold",
            "confluence_count":"ConfsMatched","confluence_total":"ConfsTotal",
            "confluences_text":"Confluences","target_reason":"Target Reason","sl_reason":"SL Reason",
            "adjusted":"Adjusted","confidence_pct":"ConfidencePct"
        })
        # make readable full reason text already included
        # CSV download
        csv_bytes = trades_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download backtest trades (CSV)", data=csv_bytes, file_name="backtest_trades.csv", mime="text/csv")

        # style & display using pandas Styler and HTML (to colour rows)
        def highlight_dir(row):
            if row["Direction"]=="BUY":
                return ["background-color: #0a6b2f; color: white;" for _ in row]
            else:
                return ["background-color: #8b0b0b; color: white;" for _ in row]
        try:
            styled = df_for_table.style.format({
                "Entry":"{:.2f}","Exit":"{:.2f}","PnL":"{:.2f}","ConfidencePct":"{:.1f}"
            }).apply(highlight_dir, axis=1)
            st.dataframe(df_for_table)  # show plain table so it's interactive
            st.markdown("**Detailed (styled) view:**")
            st.write(styled.to_html(), unsafe_allow_html=True)
        except Exception:
            st.dataframe(df_for_table)

with tab2:
    st.markdown("### Live recommendation(s) (using optimized params)")
    # Generate live recommendation by checking last bar's confluences and selecting a trade if rules match
    prepared_df, _ = detect_and_prepare(df_raw)
    df_ind = add_indicators(prepared_df, best_params)
    last = df_ind.iloc[-1]
    prev = df_ind.iloc[-2]
    last["_macd_hist_prev"] = prev["macd_hist"]
    confs_long = evaluate_confluences_for_direction(last, best_params, "long")
    confs_short = evaluate_confluences_for_direction(last, best_params, "short")

    # evaluate entry conditions
    def decide_now(confs, params):
        if params["rule_mode"]=="strict":
            return all([confs.get(k,False) for k in params["confluence_list"]])
        else:
            hits = sum([1 for k in params["confluence_list"] if confs.get(k,False)])
            return hits >= params.get("any_k",2)

    now_long = decide_now(confs_long, best_params) if trade_type in ("Long","All") else False
    now_short = decide_now(confs_short, best_params) if trade_type in ("Short","All") else False

    recs = []
    if now_long or now_short:
        # choose stronger (more confluence hits)
        long_hits = sum(1 for v in confs_long.values() if v)
        short_hits = sum(1 for v in confs_short.values() if v)
        if now_long and (not now_short or long_hits>=short_hits):
            direction="BUY"; confs=confs_long
            entry_price = float(last["open"])
            sl, sl_reason, target, target_reason, adjusted = compute_sl_target_reasons(entry_price, last, best_params, "long")
        else:
            direction="SELL"; confs=confs_short
            entry_price = float(last["open"])
            sl, sl_reason, target, target_reason, adjusted = compute_sl_target_reasons(entry_price, last, best_params, "short")

        con_count = sum(1 for v in confs.values() if v)
        con_total = len(confs)
        # dynamic confidence based on similar past trades
        conf_pct = dynamic_confidence_for_live(trades_df, direction, con_count)
        logic_text = []
        if "ma_cross" in confs and confs.get("ma_cross"): logic_text.append(f"MA{best_params['short_ma']}>{best_params['long_ma']}" if direction=="BUY" else f"MA{best_params['short_ma']}<{best_params['long_ma']}")
        if "ema_trend" in confs and confs.get("ema_trend"): logic_text.append(f"EMA9>{'EMA21' if direction=='BUY' else 'EMA21'}")
        if "macd_hist_rising" in confs and confs.get("macd_hist_rising"): logic_text.append(f"MACD hist {'>0 rising' if direction=='BUY' else '<0 falling'}")
        if "bb_breakout" in confs and confs.get("bb_breakout"): logic_text.append("BB breakout")
        if "vol_spike" in confs and confs.get("vol_spike"): logic_text.append(f"Vol >= {best_params.get('vol_spike_mult',1.5)}×avg")
        if "atr_ok" in confs and confs.get("atr_ok"): logic_text.append(f"ATR {last.get('atr',np.nan):.3f}>=min")

        rec = {
            "date": last.name,
            "direction": direction,
            "entry": round(entry_price,2),
            "target": round(target,2),
            "sl": round(sl,2),
            "target_reason": target_reason,
            "sl_reason": sl_reason,
            "confs_matched": f"{con_count}/{con_total}",
            "logic": "; ".join(logic_text),
            "confidence_pct": round(conf_pct,1),
            "adjusted": adjusted
        }
        recs.append(rec)

    if not recs:
        st.info("No live recommendation on the latest bar with optimized params.")
    else:
        # show recommendation in a nice table (newest first)
        rec_df = pd.DataFrame(recs).sort_values("date", ascending=False)
        # add styling and display
        def color_row(r):
            return ["background-color: #0a6b2f; color:white;" if r["direction"]=="BUY" else "background-color: #8b0b0b; color:white;" for _ in r.index]
        try:
            rec_df_display = rec_df[["date","direction","entry","target","sl","target_reason","sl_reason","confs_matched","logic","confidence_pct","adjusted"]].rename(columns={
                "date":"Date","direction":"Signal","entry":"Entry","target":"Target","sl":"SL","target_reason":"Target Reason","sl_reason":"SL Reason",
                "confs_matched":"Confluences","logic":"Logic","confidence_pct":"ConfidencePct","adjusted":"Adjusted"
            })
            st.dataframe(rec_df_display)
            # styled HTML view
            styled = rec_df_display.style.apply(lambda r: ["background-color: #0a6b2f; color:white;" if r["Signal"]=="BUY" else "background-color: #8b0b0b; color:white;" for _ in r.index], axis=1)
            st.markdown("**Styled recommendation view:**")
            st.write(styled.to_html(), unsafe_allow_html=True)
        except Exception:
            st.dataframe(rec_df)

st.success("Done — optimizer finished and recommendations generated.")                                             "confluence_list": ["ma_cross", "ema_trend", "macd_hist_rising", "vol_spike", "bb_breakout", "atr_ok"],
                                                "rule_mode": rm,
                                                "any_k": ak,
                                            })
    return grid


def optimize_params(df_raw: pd.DataFrame, trade_type: str, last_n: int) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, float]]:
    best_params = None
    best_score = -1e18
    best_trades = pd.DataFrame()
    best_metrics = {}

    grid = generate_param_grid()
    total = len(grid)

    # Light progress in sidebar
    prog = st.sidebar.empty()
    for idx, params in enumerate(grid, 1):
        trades_df, metrics, _ = backtest(df_raw, params, trade_type=trade_type, last_n=last_n)
        score = metrics["Net PnL"]  # objective: maximize Net PnL (can mix with Win Rate)
        # tie-breakers
        score += 0.01 * metrics["Win Rate"]

        if score > best_score:
            best_score = score
            best_params = params
            best_trades = trades_df
            best_metrics = metrics

        if idx % 50 == 0 or idx == total:
            prog.write(f"Optimizing… {idx}/{total}")

    prog.write("Optimization completed.")
    return best_params, best_trades, best_metrics


# ========================
# UI — Sidebar
# ========================
st.title("📈 Swing (EOD) Strategy — Auto Optimized, Entries at Next Open (t+1)")

with st.sidebar:
    st.header("Controls")
    upl = st.file_uploader("Upload Daily OHLC (CSV/XLSX)", type=["csv", "xlsx"])
    trade_type = st.selectbox("Trade Type", ["All", "Long", "Short"], index=1)
    lookback = st.number_input("Bars to use for Optimization", 60, 1000, 180, 10)
    run = st.button("Run Optimization + Backtest")

if upl is None:
    st.info("Upload a CSV/XLSX with Date, Open, High, Low, Close (Volume optional).")
    st.stop()

# load
try:
    if upl.name.lower().endswith(".xlsx"):
        raw = pd.read_excel(upl)
    else:
        raw = pd.read_csv(upl)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# ========================
# Optimize & Backtest
# ========================
if not run:
    st.info("Press **Run Optimization + Backtest** to start.")
    st.stop()

best_params, best_trades, best_metrics = optimize_params(raw, trade_type, last_n=int(lookback))

# re-run backtest with best params over the same lookback (for reporting)
trades_df, metrics, ind_df = backtest(raw, best_params, trade_type=trade_type, last_n=int(lookback))

# ========================
# Summary Cards
# ========================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net PnL", f"{metrics['Net PnL']:.2f}")
c2.metric("Trades", f"{metrics['Trades']}")
c3.metric("Win Rate", f"{metrics['Win Rate']:.1f}%")
c4.metric("Avg Hold (bars)", f"{metrics['Avg Hold (bars)']:.2f}")
c5.metric("Trades / 100 bars", f"{metrics['Trades / 100 bars']:.2f}")

# ========================
# Show Best Parameters
# ========================
st.subheader("Best Optimized Parameters (used for backtest & live plan)")
bp = best_params.copy()
param_df = pd.DataFrame([
    ["MA Short", f"MA{bp['ma_short']}"],
    ["MA Long", f"MA{bp['ma_long']}"],
    ["EMA Fast", f"EMA{bp['ema_fast']}"],
    ["EMA Slow", f"EMA{bp['ema_slow']}"],
    ["BB Period", f"{bp['bb_period']} (k={bp['bb_k']})"],
    ["Vol Spike ×Avg", f"{bp['vol_spike_mult']}"],
    ["ATR Period", f"{bp['atr_period']}"],
    ["ATR SL Mult", f"{bp['atr_mult_sl']}"],
    ["ATR Target Mult", f"{bp['atr_mult_tg']}"],
    ["Rule", f"{bp['rule_mode']} (any_k={bp['any_k']})"],
], columns=["Parameter", "Value"])
st.dataframe(param_df, use_container_width=True, hide_index=True)
st.download_button(
    "Download Best Parameters (CSV)",
    data=param_df.to_csv(index=False).encode("utf-8"),
    file_name="best_parameters.csv",
    mime="text/csv",
)

tab_bt, tab_live = st.tabs(["Backtest Results", "Live Next-Day Plan"])

# ========================
# Backtest Tab
# ========================
with tab_bt:
    st.subheader("Backtest Trades (optimized params)")
    if trades_df.empty:
        st.info("No trades generated with current optimized parameters.")
    else:
        show = trades_df.copy().sort_values("Entry Date", ascending=False)
        # confidence column (based on historical similar setups)
        show["Confidence (%)"] = show.apply(
            lambda r: round(dynamic_confidence(trades_df, r["Direction"], int(r["Confluences"])), 1), axis=1
        )
        for col in ["Signal Date", "Entry Date", "Exit Date"]:
            show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d")
        st.dataframe(
            show[
                [
                    "Signal Date",
                    "Entry Date",
                    "Exit Date",
                    "Direction",
                    "Entry",
                    "SL",
                    "Target",
                    "Exit",
                    "PnL",
                    "Confluences",
                    "ConfsTotal",
                    "Confidence (%)",
                    "Logic (with values)",
                    "SL Reason",
                    "Target Reason",
                    "Exit Reason",
                ]
            ],
            use_container_width=True,
            height=440,
        )
        st.download_button(
            "Download Trades CSV",
            data=trades_df.to_csv(index=False).encode("utf-8"),
            file_name="backtest_trades.csv",
            mime="text/csv",
        )

# ========================
# Live Tab (Next-Day Plan using Best Params)
# ========================
with tab_live:
    st.subheader("Next-Day Recommendation (EOD generated with optimized params)")

    # Use last completed bar within the same lookback slice used for optimization
    prepared, _ = detect_and_prepare(raw)
    latest_slice = prepared.iloc[-int(lookback):].copy()
    ind_live = add_indicators(latest_slice, best_params)

    if len(ind_live) < 3:
        st.info("Not enough data to form a next-day plan.")
    else:
        # Last completed bar (t) and previous (t-1)
        sig = ind_live.iloc[-1]
        prev = ind_live.iloc[-2]

        allow_long = trade_type in ("Long", "All")
        allow_short = trade_type in ("Short", "All")

        plan_rows = []

        def add_plan(direction: str, flags, texts):
            keys = best_params["confluence_list"]
            ok, hits = decide_confluence(flags, best_params)
            if not ok:
                return
            # Entry will be at next day's open (unknown now); estimate with last close for planning only
            est_entry = float(sig["close"])
            sl, slr, tgt, tgtr = calc_levels(est_entry, sig, best_params, "long" if direction == "BUY" else "short")
            conf = dynamic_confidence(trades_df, direction, hits)
            plan_rows.append(
                {
                    "Signal Date (EOD)": pd.to_datetime(sig.name).strftime("%Y-%m-%d"),
                    "Trade On (Next Open)": "(t+1 open — actual price unknown)",
                    "Direction": direction,
                    "Est Entry (≈ last close)": round(est_entry, 2),
                    "SL (planned)": round(sl, 2),
                    "Target (planned)": round(tgt, 2),
                    "Confluences": f"{hits}/{len(keys)}",
                    "Confidence (%)": round(conf, 1),
                    "Logic (with values)": " | ".join([texts[k] for k in keys if flags.get(k, False)]),
                    "SL Reason": slr,
                    "Target Reason": tgtr,
                }
            )

        if allow_long:
            fl, tl = eval_confluences(sig, prev, best_params, "long")
            add_plan("BUY", fl, tl)

        if allow_short:
            fs, ts = eval_confluences(sig, prev, best_params, "short")
            add_plan("SELL", fs, ts)

        if not plan_rows:
            st.info("No next-day recommendation based on the latest EOD candle with optimized params.")
        else:
            live_df = pd.DataFrame(plan_rows)
            st.dataframe(live_df, use_container_width=True)
            st.download_button(
                "Download Live Plan (CSV)",
                data=live_df.to_csv(index=False).encode("utf-8"),
                file_name="live_nextday_plan.csv",
                mime="text/csv",
            )

# ========================
# Quick Chart
# ========================
with st.expander("Chart (Close, MAs, Bollinger)"):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["close"], name="Close", mode="lines"))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["ma_short"], name=f"MA{best_params['ma_short']}", mode="lines"))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["ma_long"], name=f"MA{best_params['ma_long']}", mode="lines"))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["bb_upper"], name="BB Upper", mode="lines"))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["bb_lower"], name="BB Lower", mode="lines"))
        fig.update_layout(height=460, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
