# app.py
import re
import random
from typing import Tuple, Dict, Any, List
from math import isnan

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Swing Trading Dashboard (No Lookahead • Exact Logic)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Utilities
# ========================
def _clean_name(s: str) -> str:
    s = str(s or "").lower().strip()
    return re.sub(r'[^a-z0-9]', '', s)

def detect_and_prepare(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Detect OHLCV/Date columns (fuzzy), coerce numeric, return prepared df + mapping."""
    df = df_raw.copy()
    orig_cols = list(df.columns)
    cleaned_map = {_clean_name(c): c for c in orig_cols}

    open_cands = ["open","openprice","o"]
    high_cands = ["high","highprice","h"]
    low_cands  = ["low","lowprice","l"]
    close_cands= ["close","closeprice","ltp","last","price","adjclose","adjcloseprice"]
    vol_cands  = ["volume","vol","quantity","tradedqty","qty"]
    date_cands = ["date","datetime","timestamp","time"]

    def find(cands):
        # exact-ish
        for cand in cands:
            key = _clean_name(cand)
            if key in cleaned_map:
                return cleaned_map[key]
        # contains
        for k, v in cleaned_map.items():
            for cand in cands:
                if _clean_name(cand) in k:
                    return v
        # fallback
        for orig in orig_cols:
            low = str(orig).lower().replace(" ", "")
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
        "date": find(date_cands),
    }

    # index to datetime if possible
    if col_map["date"] and col_map["date"] in df.columns:
        df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors="coerce")
        if df[col_map["date"]].notna().sum() > 0:
            df = df.set_index(col_map["date"])

    # numeric coercion with cleaning
    def to_numeric_series(s):
        s = s.astype(str).fillna("")
        s = s.str.replace(r'[^0-9\.\-]', '', regex=True)
        return pd.to_numeric(s, errors="coerce")

    prepared = pd.DataFrame(index=df.index)
    for canon in ["open","high","low","close","volume"]:
        orig = col_map.get(canon)
        if orig and orig in df.columns:
            prepared[canon] = to_numeric_series(df[orig])
        else:
            prepared[canon] = np.nan

    prepared.dropna(subset=["open","high","low","close"], inplace=True)

    if not isinstance(prepared.index, pd.DatetimeIndex):
        try:
            prepared.index = pd.to_datetime(prepared.index, errors="coerce")
            prepared = prepared[~prepared.index.isna()]
        except Exception:
            pass

    prepared.sort_index(inplace=True)
    if prepared["close"].dropna().empty:
        return None, {"col_map": col_map, "original_columns": orig_cols}
    return prepared, {"col_map": col_map, "original_columns": orig_cols}

# ========================
# Indicators
# ========================
def add_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    d = df.copy()

    # EMAs & simple MAs
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ma_short"] = d["close"].rolling(params["short_ma"], min_periods=1).mean()
    d["ma_long"]  = d["close"].rolling(params["long_ma"],  min_periods=1).mean()

    # RSI (EMA-based)
    per = params.get("rsi_period", 14)
    delta = d["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/per, adjust=False).mean()
    ma_down = down.ewm(alpha=1/per, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    d["rsi"].fillna(50, inplace=True)

    # MACD + signal + hist
    fast = d["close"].ewm(span=12, adjust=False).mean()
    slow = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = fast - slow
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # ATR (simple TR rolling mean)
    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"]  - d["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr"] = tr.rolling(params.get("atr_period", 14), min_periods=1).mean()

    # Bollinger Bands
    bb_p = params.get("bb_period", 20)
    bb_k = params.get("bb_k", 2)
    mid = d["close"].rolling(bb_p, min_periods=1).mean()
    std = d["close"].rolling(bb_p, min_periods=1).std().fillna(0)
    d["bb_mid"]   = mid
    d["bb_upper"] = mid + bb_k * std
    d["bb_lower"] = mid - bb_k * std

    # Volume avg
    d["vol_avg"] = d["volume"].rolling(params.get("vol_period", 20), min_periods=1).mean()

    return d

# ========================
# Confluences with values
# ========================
def eval_confluences(row_sig: pd.Series, row_prev: pd.Series, params: Dict[str,Any], direction:str) -> Tuple[Dict[str,bool], Dict[str,str]]:
    """Return (flags, pretty_text_parts_with_values) for each confluence on the SIGNAL bar."""
    flags = {}
    texts = {}

    # MA cross
    if direction == "long":
        flags["ma_cross"] = bool(row_sig["ma_short"] > row_sig["ma_long"])
        texts["ma_cross"] = f"ma_cross(MA{params['short_ma']}>{params['long_ma']}: {row_sig['ma_short']:.2f} > {row_sig['ma_long']:.2f})"
    else:
        flags["ma_cross"] = bool(row_sig["ma_short"] < row_sig["ma_long"])
        texts["ma_cross"] = f"ma_cross(MA{params['short_ma']}<{params['long_ma']}: {row_sig['ma_short']:.2f} < {row_sig['ma_long']:.2f})"

    # EMA trend
    if direction == "long":
        flags["ema_trend"] = bool(row_sig["ema9"] > row_sig["ema21"])
        texts["ema_trend"] = f"ema_trend(EMA9>EMA21: {row_sig['ema9']:.2f} > {row_sig['ema21']:.2f})"
    else:
        flags["ema_trend"] = bool(row_sig["ema9"] < row_sig["ema21"])
        texts["ema_trend"] = f"ema_trend(EMA9<EMA21: {row_sig['ema9']:.2f} < {row_sig['ema21']:.2f})"

    # BB breakout
    if direction == "long":
        flags["bb_breakout"] = bool(row_sig["close"] > row_sig["bb_upper"])
        texts["bb_breakout"] = f"bb_breakout(Close {row_sig['close']:.2f} > BBupper {row_sig['bb_upper']:.2f})"
    else:
        flags["bb_breakout"] = bool(row_sig["close"] < row_sig["bb_lower"])
        texts["bb_breakout"] = f"bb_breakout(Close {row_sig['close']:.2f} < BBlower {row_sig['bb_lower']:.2f})"

    # MACD hist rising/falling
    macd_prev = float(row_prev["macd_hist"]) if row_prev is not None and not isnan(row_prev["macd_hist"]) else 0.0
    macd_curr = float(row_sig["macd_hist"])
    if direction == "long":
        flags["macd_hist_rising"] = bool(macd_curr > 0 and macd_curr > macd_prev)
        texts["macd_hist_rising"] = f"macd_hist_rising({macd_prev:.4f} → {macd_curr:.4f})"
    else:
        flags["macd_hist_rising"] = bool(macd_curr < 0 and macd_curr < macd_prev)
        texts["macd_hist_rising"] = f"macd_hist_falling({macd_prev:.4f} → {macd_curr:.4f})"

    # Volume spike
    vol_mult = params.get("vol_spike_mult", 1.5)
    vol_avg = float(row_sig.get("vol_avg", 0.0) or 0.0)
    flags["vol_spike"] = bool(row_sig["volume"] >= vol_mult * vol_avg) if vol_avg > 0 else False
    texts["vol_spike"] = f"vol_spike(Vol {row_sig['volume']:.0f} ≥ {vol_mult}×Avg {vol_avg:.0f})"

    # ATR min
    atr_min = params.get("atr_min", 0.0)
    atr_val = float(row_sig.get("atr", 0.0) or 0.0)
    flags["atr_ok"] = bool(atr_val >= atr_min)
    texts["atr_ok"] = f"atr_ok(ATR {atr_val:.2f} ≥ min {atr_min:.2f})"

    return flags, texts

# ========================
# SL/Target (original formula; no "auto-fix")
# ========================
def calc_levels_original(entry_price: float, row_sig: pd.Series, params: Dict[str,Any], direction: str):
    atr = float(row_sig.get("atr", 0.0) or 0.0)
    if params.get("use_atr_sl", False):
        if direction == "long":
            sl = entry_price - atr * params.get("atr_mult", 1.5)
            sl_reason = f"SL = Entry − ATR×{params.get('atr_mult',1.5)} = {entry_price:.2f} − ({atr:.3f}×{params.get('atr_mult',1.5)})"
        else:
            sl = entry_price + atr * params.get("atr_mult", 1.5)
            sl_reason = f"SL = Entry + ATR×{params.get('atr_mult',1.5)} = {entry_price:.2f} + ({atr:.3f}×{params.get('atr_mult',1.5)})"
    else:
        pct = params.get("sl_pct", 1.5)
        if direction == "long":
            sl = entry_price * (1 - pct/100)
            sl_reason = f"SL = Entry − {pct}% = {sl:.2f}"
        else:
            sl = entry_price * (1 + pct/100)
            sl_reason = f"SL = Entry + {pct}% = {sl:.2f}"

    if params.get("target_atr", False):
        mult = params.get("target_atr_mult", 1.5)
        if direction == "long":
            target = entry_price + atr * mult
            target_reason = f"Target = Entry + ATR×{mult} = {entry_price:.2f} + ({atr:.3f}×{mult})"
        else:
            target = entry_price - atr * mult
            target_reason = f"Target = Entry − ATR×{mult} = {entry_price:.2f} − ({atr:.3f}×{mult})"
    else:
        pct = params.get("target_pct", 2.0)
        if direction == "long":
            target = entry_price * (1 + pct/100)
            target_reason = f"Target = Entry + {pct}% = {target:.2f}"
        else:
            target = entry_price * (1 - pct/100)
            target_reason = f"Target = Entry − {pct}% = {target:.2f}"

    return float(sl), sl_reason, float(target), target_reason

# ========================
# Backtester (no lookahead)
# ========================
def backtest(df_raw: pd.DataFrame, params: Dict[str,Any], trade_type: str="Long") -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    prepared, meta = detect_and_prepare(df_raw)
    if prepared is None:
        raise ValueError(f"Could not detect OHLC columns. Mapping: {meta['col_map']} | Original: {meta['original_columns']}")
    d = add_indicators(prepared, params)

    trades = []
    position = None

    # Evaluate signals at bar i-1; enter at bar i open (NO LOOKAHEAD)
    for i in range(2, len(d)):
        sig = d.iloc[i-1]      # signal bar (indicators known up to here)
        prev = d.iloc[i-2]     # previous bar for MACD-hist trend
        cur = d.iloc[i]        # entry/next bar (we use its OPEN for entry)

        allow_long  = trade_type in ("Long","All")
        allow_short = trade_type in ("Short","All")

        if position is None:
            enter_long = enter_short = False
            confs_long = confs_short = {}
            texts_long = texts_short = {}

            if allow_long:
                confs_long, texts_long = eval_confluences(sig, prev, params, "long")
            if allow_short:
                confs_short, texts_short = eval_confluences(sig, prev, params, "short")

            def decide(confs):
                if not confs: return False
                clist = params["confluence_list"]
                if params["rule_mode"] == "strict":
                    return all(confs.get(k, False) for k in clist)
                else:
                    hits = sum(1 for k in clist if confs.get(k, False))
                    return hits >= params.get("any_k", 2)

            if allow_long:  enter_long  = decide(confs_long)
            if allow_short: enter_short = decide(confs_short)

            if enter_long or enter_short:
                # choose by hits
                long_hits  = sum(1 for v in confs_long.values() if v) if allow_long else -1
                short_hits = sum(1 for v in confs_short.values() if v) if allow_short else -1

                if enter_long and (not enter_short or long_hits >= short_hits):
                    direction = "BUY"
                    entry_price = float(cur["open"])
                    sl, sl_reason, target, target_reason = calc_levels_original(entry_price, sig, params, "long")
                    position = {
                        "direction":"BUY",
                        "entry_index": cur.name,
                        "entry_price": entry_price,
                        "sl": sl,
                        "target": target,
                        "hold": 0,
                        "sig_texts": texts_long,
                        "sig_flags": confs_long
                    }
                else:
                    direction = "SELL"
                    entry_price = float(cur["open"])
                    sl, sl_reason, target, target_reason = calc_levels_original(entry_price, sig, params, "short")
                    position = {
                        "direction":"SELL",
                        "entry_index": cur.name,
                        "entry_price": entry_price,
                        "sl": sl,
                        "target": target,
                        "hold": 0,
                        "sig_texts": texts_short,
                        "sig_flags": confs_short
                    }
                position["sl_reason"] = sl_reason
                position["target_reason"] = target_reason

        else:
            # manage position on bar i (same bar as we used for potential entry if we entered above)
            curbar = d.iloc[i]
            position["hold"] += 1
            exit_price = None
            exit_reason = None

            if position["direction"] == "BUY":
                if curbar["high"] >= position["target"]:
                    exit_price = position["target"]; exit_reason = "Target"
                elif curbar["low"] <= position["sl"]:
                    exit_price = position["sl"]; exit_reason = "StopLoss"
            else:
                if curbar["low"] <= position["target"]:
                    exit_price = position["target"]; exit_reason = "Target"
                elif curbar["high"] >= position["sl"]:
                    exit_price = position["sl"]; exit_reason = "StopLoss"

            if exit_price is None and position["hold"] >= params.get("max_hold",5):
                exit_price = float(curbar["close"]); exit_reason = "MaxHold"

            if exit_price is not None:
                pnl = (exit_price - position["entry_price"]) if position["direction"]=="BUY" else (position["entry_price"] - exit_price)
                conf_flags = position["sig_flags"]
                conf_texts = position["sig_texts"]
                confs_matched = sum(1 for k in params["confluence_list"] if conf_flags.get(k, False))
                trades.append({
                    "entry_date": position["entry_index"],
                    "exit_date": curbar.name,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "hold_days": position["hold"],
                    "exit_reason": exit_reason,
                    "confs_matched": confs_matched,
                    "confs_total": len(params["confluence_list"]),
                    "confs_text": " | ".join([conf_texts[k] for k in params["confluence_list"] if conf_flags.get(k, False)]),
                    "target_reason": position["target_reason"],
                    "sl_reason": position["sl_reason"],
                })
                position = None

    # Close at last bar if still open
    if position is not None:
        lastbar = d.iloc[-1]
        pnl = (lastbar["close"] - position["entry_price"]) if position["direction"]=="BUY" else (position["entry_price"] - lastbar["close"])
        conf_flags = position["sig_flags"]
        conf_texts = position["sig_texts"]
        confs_matched = sum(1 for k in params["confluence_list"] if conf_flags.get(k, False))
        trades.append({
            "entry_date": position["entry_index"],
            "exit_date": lastbar.name,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": float(lastbar["close"]),
            "pnl": float(pnl),
            "hold_days": position["hold"],
            "exit_reason": "EOD",
            "confs_matched": confs_matched,
            "confs_total": len(params["confluence_list"]),
            "confs_text": " | ".join([conf_texts[k] for k in params["confluence_list"] if conf_flags.get(k, False)]),
            "target_reason": position["target_reason"],
            "sl_reason": position["sl_reason"],
        })

    trades_df = pd.DataFrame(trades)
    metrics = {"net_pnl":0.0,"n_trades":0,"win_rate":0.0,"avg_hold":0.0,"avg_win":0.0,"avg_loss":0.0,"trades_per_100bars":0.0}
    if not trades_df.empty:
        metrics["net_pnl"] = trades_df["pnl"].sum()
        metrics["n_trades"] = len(trades_df)
        metrics["win_rate"] = float((trades_df["pnl"]>0).mean())
        metrics["avg_hold"] = float(trades_df["hold_days"].mean())
        metrics["avg_win"] = float(trades_df.loc[trades_df["pnl"]>0, "pnl"].mean()) if (trades_df["pnl"]>0).any() else 0.0
        metrics["avg_loss"] = float(trades_df.loc[trades_df["pnl"]<=0, "pnl"].mean()) if (trades_df["pnl"]<=0).any() else 0.0
        metrics["trades_per_100bars"] = metrics["n_trades"]/max(1,len(d))*100.0

    return trades_df, metrics, d

# ========================
# Optimizer (random + hill)
# ========================
def score_metrics(m: dict, min_trades:int=4) -> float:
    score = float(m.get("net_pnl", 0.0))
    wr = float(m.get("win_rate", 0.0))
    n  = int(m.get("n_trades", 0))
    if n < min_trades:
        score *= 0.4
    score *= (1.0 + wr)
    return score

def random_search(df_raw, space, evals:int, seed:int, trade_type:str):
    random.seed(seed)
    out = []
    for _ in range(evals):
        params = {k: random.choice(v) for k, v in space.items() if isinstance(v, list)}
        params["confluence_list"] = space.get("confluence_list", [["ma_cross","ema_trend","macd_hist_rising","vol_spike","bb_breakout","atr_ok"]])[0]
        params["rsi_period"] = space.get("rsi_period", [14])[0]
        if params.get("rule_mode") == "any_k" and "any_k" not in params:
            params["any_k"] = random.randint(2, max(2, len(params["confluence_list"])))
        try:
            tdf, met, _ = backtest(df_raw, params, trade_type)
            sc = score_metrics(met, space.get("min_trades",4))
            out.append({"params":params, "metrics":met, "score":sc, "trades_df":tdf})
        except Exception:
            continue
    return out

def neighbors(base, space):
    nb=[]
    def close_opt(opts, val):
        if val not in opts: return random.choice(opts)
        idx = opts.index(val)
        if len(opts)==1: return opts[0]
        return opts[max(0, idx-1)] if random.random()<0.5 else opts[min(len(opts)-1, idx+1)]
    for _ in range(15):
        n = base.copy()
        for k, v in space.items():
            if isinstance(v, list):
                n[k] = close_opt(v, n.get(k, random.choice(v)))
        if n.get("rule_mode")=="any_k" and "any_k" not in n:
            n["any_k"] = random.randint(2, max(2, len(n.get("confluence_list",[]))))
        nb.append(n)
    return nb

def optimize(df_raw, space, stage_a:int=150, seed:int=42, trade_type:str="Long"):
    a = random_search(df_raw, space, stage_a, seed, trade_type)
    if not a: raise RuntimeError("No valid parameter sets in Stage A.")
    top = sorted(a, key=lambda x: x["score"], reverse=True)[:8]
    best = top[0]
    for cand in top:
        for n in neighbors(cand["params"], space):
            try:
                tdf, met, _ = backtest(df_raw, n, trade_type)
                sc = score_metrics(met, space.get("min_trades",4))
                if sc > best["score"]:
                    best = {"params":n, "metrics":met, "score":sc, "trades_df":tdf}
            except Exception:
                continue
    return best

# ========================
# Confidence (live)
# ========================
def dynamic_confidence(trades_df: pd.DataFrame, direction: str, confs_matched:int) -> float:
    if trades_df is None or trades_df.empty: return 0.0
    tmp = trades_df.copy()
    tmp["cm"] = tmp["confs_matched"]
    sub = tmp[(tmp["direction"]==direction) & (tmp["cm"]==confs_matched)]
    if not sub.empty:
        return float((sub["pnl"]>0).mean()*100)
    sub2 = tmp[tmp["direction"]==direction]
    if not sub2.empty:
        return float((sub2["pnl"]>0).mean()*100)
    return float((tmp["pnl"]>0).mean()*100)

# ========================
# UI
# ========================
st.title("🚀 Swing Trading Dashboard (Exact Logic • No Lookahead)")

with st.sidebar:
    st.header("Controls")
    upl = st.file_uploader("Upload OHLC (CSV/XLSX)", type=["csv","xlsx"])
    trade_type = st.selectbox("Trade Type", ["All","Long","Short"], index=1)
    stage_a = st.number_input("Stage-A evaluations", min_value=50, max_value=1000, step=50, value=200)
    run = st.button("Run Optimize & Backtest")

if upl is None:
    st.info("Upload a CSV/XLSX to begin.")
    st.stop()

# Load file
try:
    if upl.name.lower().endswith(".xlsx"):
        raw = pd.read_excel(upl)
    else:
        raw = pd.read_csv(upl)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

prepared, mapping = detect_and_prepare(raw)
if prepared is None:
    st.error("Uploaded file doesn't contain usable OHLC after normalization.")
    st.json(mapping)
    st.stop()

st.success(f"Loaded {len(prepared)} rows.")
with st.expander("Column mapping"):
    st.json(mapping)

if not run:
    st.info("Set parameters and click **Run Optimize & Backtest**.")
    st.stop()

# Search space (same spirit as your previous working version)
space = {
    "short_ma": list(range(5, 31, 1)),
    "long_ma":  list(range(20, 121, 5)),
    "rsi_period": [14],
    "rsi_entry": [20,25,30,35,40],
    "bb_period": [14,20],
    "bb_k": [2],
    "vol_period": [10,20],
    "vol_spike_mult": [1.2,1.5,2.0],
    "atr_period": [14],
    "atr_min": [0.0, 5.0, 10.0, 15.0],
    "use_atr_sl": [True, False],
    "atr_mult": [1.0, 1.5, 2.0],
    "target_atr": [False, True],
    "target_atr_mult": [1.0, 1.5, 2.0],
    "target_pct": [1.0, 1.5, 2.0, 3.0],
    "sl_pct": [0.5, 1.0, 1.5, 2.0],
    "max_hold": [3,5,7,10],
    "confluence_list": [["ma_cross","ema_trend","macd_hist_rising","vol_spike","bb_breakout","atr_ok"]],
    "rule_mode": ["strict","any_k"],
    "min_trades": 4,
}

with st.spinner("Optimizing (random search + hill-climb)…"):
    best = optimize(raw, space, stage_a, seed=42, trade_type=trade_type)

best_params = best["params"]
trades_df, metrics, ind_df = backtest(raw, best_params, trade_type)

# Summary cards
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Net PnL", f"{metrics['net_pnl']:.2f}")
c2.metric("Trades", f"{metrics['n_trades']}")
c3.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
c4.metric("Avg Hold (bars)", f"{metrics['avg_hold']:.2f}")
c5.metric("Trades / 100 bars", f"{metrics['trades_per_100bars']:.2f}")

# Tabs
tab_backtest, tab_live = st.tabs(["Backtest Results", "Live Recommendations"])

with tab_backtest:
    st.subheader("Backtest Trades (newest first)")
    if trades_df.empty:
        st.info("No trades for the optimized parameters.")
    else:
        show = trades_df.copy().sort_values("entry_date", ascending=False)
        # Format columns
        show["entry_date"] = pd.to_datetime(show["entry_date"]).dt.strftime("%Y-%m-%d %H:%M")
        show["exit_date"]  = pd.to_datetime(show["exit_date"]).dt.strftime("%Y-%m-%d %H:%M")
        show = show.rename(columns={
            "entry_date":"Entry Date","exit_date":"Exit Date","direction":"Direction",
            "entry_price":"Entry","exit_price":"Exit","hold_days":"Hold",
            "exit_reason":"Exit Reason","confs_matched":"Confluences","confs_total":"ConfsTotal",
            "confs_text":"Logic (with values)","target_reason":"Target Reason","sl_reason":"SL Reason"
        })
        # Add per-trade confidence (same method as live)
        show["Confidence (%)"] = show.apply(lambda r: dynamic_confidence(trades_df, r["Direction"], int(r["Confluences"])), axis=1).round(1)
        # Order columns
        cols = ["Entry Date","Exit Date","Direction","Entry","Exit","SL Reason","Target Reason",
                "Confluences","ConfsTotal","Logic (with values)","Confidence (%)","Hold","Exit Reason","pnl"]
        cols = [c for c in cols if c in show.columns]
        st.dataframe(show[cols], use_container_width=True)
        st.download_button(
            "Download Backtest Trades (CSV)",
            data=trades_df.to_csv(index=False).encode("utf-8"),
            file_name="backtest_trades.csv",
            mime="text/csv"
        )

with tab_live:
    st.subheader("Live Recommendation (latest bar)")
    # Signals on the last COMPLETED bar; entry would be at current/next bar open
    if len(ind_df) < 3:
        st.info("Not enough data for live signal.")
    else:
        sig = ind_df.iloc[-2]   # last completed bar (signal)
        prev = ind_df.iloc[-3]
        cur  = ind_df.iloc[-1]  # next bar where we could enter at OPEN

        allow_long  = trade_type in ("Long","All")
        allow_short = trade_type in ("Short","All")

        def decide(flags):
            if not flags: return False
            cl = best_params["confluence_list"]
            if best_params["rule_mode"] == "strict":
                return all(flags.get(k, False) for k in cl)
            else:
                hits = sum(1 for k in cl if flags.get(k, False))
                return hits >= best_params.get("any_k", 2)

        recs = []
        if allow_long:
            f_long, t_long = eval_confluences(sig, prev, best_params, "long")
            if decide(f_long):
                entry = float(cur["open"])
                sl, slr, tgt, tgtr = calc_levels_original(entry, sig, best_params, "long")
                cm = sum(1 for k in best_params["confluence_list"] if f_long.get(k, False))
                conf = dynamic_confidence(trades_df, "BUY", cm)
                recs.append({
                    "Date": cur.name, "Signal": "BUY", "Entry": round(entry,2),
                    "SL": round(sl,2), "Target": round(tgt,2),
                    "SL Reason": slr, "Target Reason": tgtr,
                    "Confluences": f"{cm}/{len(best_params['confluence_list'])}",
                    "Logic (with values)": " | ".join([t_long[k] for k in best_params["confluence_list"] if f_long.get(k, False)]),
                    "Confidence (%)": round(conf,1)
                })

        if allow_short:
            f_short, t_short = eval_confluences(sig, prev, best_params, "short")
            if decide(f_short):
                entry = float(cur["open"])
                sl, slr, tgt, tgtr = calc_levels_original(entry, sig, best_params, "short")
                cm = sum(1 for k in best_params["confluence_list"] if f_short.get(k, False))
                conf = dynamic_confidence(trades_df, "SELL", cm)
                recs.append({
                    "Date": cur.name, "Signal": "SELL", "Entry": round(entry,2),
                    "SL": round(sl,2), "Target": round(tgt,2),
                    "SL Reason": slr, "Target Reason": tgtr,
                    "Confluences": f"{cm}/{len(best_params['confluence_list'])}",
                    "Logic (with values)": " | ".join([t_short[k] for k in best_params["confluence_list"] if f_short.get(k, False)]),
                    "Confidence (%)": round(conf,1)
                })

        if not recs:
            st.info("No live recommendation on the latest bar.")
        else:
            live_df = pd.DataFrame(recs).sort_values("Date", ascending=False)
            st.dataframe(live_df, use_container_width=True)

# Optional: quick chart
with st.expander("Chart (Close, MA Short/Long)"):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["close"], mode="lines", name="Close", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["ma_short"], mode="lines", name=f"MA{best_params['short_ma']}", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["ma_long"],  mode="lines", name=f"MA{best_params['long_ma']}",  line=dict(width=1)))
        fig.update_layout(height=450, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
