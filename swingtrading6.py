# app.py
import re
from math import isnan
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(
    page_title="Swing (EOD) — Auto Optimized, No Lookahead",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ========================
# Utilities
# ========================
def _clean_name(s: str) -> str:
    s = str(s or "").lower().strip()
    return re.sub(r"[^a-z0-9]", "", s)


def detect_and_prepare(df_raw: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], dict]:
    """Detect OHLCV + Date columns (fuzzy), coerce numeric, set Date index, sort."""
    df = df_raw.copy()
    orig_cols = list(df.columns)
    cleaned_map = {_clean_name(c): c for c in orig_cols}

    def find(cands: List[str]):
        for cand in cands:
            key = _clean_name(cand)
            if key in cleaned_map:
                return cleaned_map[key]
        # partial contains
        for k, v in cleaned_map.items():
            for cand in cands:
                if _clean_name(cand) in k:
                    return v
        for orig in orig_cols:
            low = str(orig).lower().replace(" ", "")
            for cand in cands:
                if cand in low:
                    return orig
        return None

    col_map = {
        "date": find(["date", "datetime", "timestamp", "time"]),
        "open": find(["open", "o", "openprice"]),
        "high": find(["high", "h", "highprice"]),
        "low": find(["low", "l", "lowprice"]),
        "close": find(["close", "c", "closeprice", "ltp", "last", "price", "adjclose"]),
        "volume": find(["volume", "vol", "quantity", "tradedqty", "qty"]),
    }

    # index to datetime
    if col_map["date"] and col_map["date"] in df.columns:
        df[col_map["date"]] = pd.to_datetime(df[col_map["date"]], errors="coerce")
        if df[col_map["date"]].notna().sum() > 0:
            df = df.set_index(col_map["date"])

    # numeric coercion (strip commas/text)
    def to_num(s: pd.Series) -> pd.Series:
        s = s.astype(str).fillna("")
        s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
        return pd.to_numeric(s, errors="coerce")

    prepared = pd.DataFrame(index=df.index)
    for canon in ["open", "high", "low", "close", "volume"]:
        orig = col_map.get(canon)
        if orig and orig in df.columns:
            prepared[canon] = to_num(df[orig])
        else:
            prepared[canon] = np.nan

    prepared.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # ensure DateTime index
    if not isinstance(prepared.index, pd.DatetimeIndex):
        prepared.index = pd.to_datetime(prepared.index, errors="coerce")
        prepared = prepared[~prepared.index.isna()]

    prepared.sort_index(inplace=True)

    if prepared["close"].dropna().empty:
        return None, {"col_map": col_map, "original_columns": orig_cols}

    return prepared, {"col_map": col_map, "original_columns": orig_cols}


# ========================
# Indicators
# ========================
def add_indicators(df: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    d = df.copy()

    # EMAs & MAs
    d["ema_fast"] = d["close"].ewm(span=p["ema_fast"], adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=p["ema_slow"], adjust=False).mean()
    d["ma_short"] = d["close"].rolling(p["ma_short"], min_periods=1).mean()
    d["ma_long"] = d["close"].rolling(p["ma_long"], min_periods=1).mean()

    # MACD
    fast = d["close"].ewm(span=12, adjust=False).mean()
    slow = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = fast - slow
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # ATR
    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"] - d["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr"] = tr.rolling(p.get("atr_period", 14), min_periods=1).mean()

    # Bollinger
    bb_p = p.get("bb_period", 20)
    bb_k = p.get("bb_k", 2)
    mid = d["close"].rolling(bb_p, min_periods=1).mean()
    std = d["close"].rolling(bb_p, min_periods=1).std().fillna(0)
    d["bb_mid"] = mid
    d["bb_upper"] = mid + bb_k * std
    d["bb_lower"] = mid - bb_k * std

    # Volume avg
    d["vol_avg"] = d["volume"].rolling(p.get("vol_period", 20), min_periods=1).mean()

    return d


# ========================
# Confluences with values
# ========================
def eval_confluences(row_sig: pd.Series, row_prev: pd.Series, p: Dict[str, Any], direction: str):
    flags, texts = {}, {}

    # MA cross (simple)
    if direction == "long":
        flags["ma_cross"] = bool(row_sig["ma_short"] > row_sig["ma_long"])
        texts["ma_cross"] = (
            f"ma_cross(MA{p['ma_short']}={row_sig['ma_short']:.2f} > "
            f"MA{p['ma_long']}={row_sig['ma_long']:.2f})"
        )
    else:
        flags["ma_cross"] = bool(row_sig["ma_short"] < row_sig["ma_long"])
        texts["ma_cross"] = (
            f"ma_cross(MA{p['ma_short']}={row_sig['ma_short']:.2f} < "
            f"MA{p['ma_long']}={row_sig['ma_long']:.2f})"
        )

    # EMA trend
    if direction == "long":
        flags["ema_trend"] = bool(row_sig["ema_fast"] > row_sig["ema_slow"])
        texts["ema_trend"] = (
            f"ema_trend(EMA{p['ema_fast']}={row_sig['ema_fast']:.2f} > "
            f"EMA{p['ema_slow']}={row_sig['ema_slow']:.2f})"
        )
    else:
        flags["ema_trend"] = bool(row_sig["ema_fast"] < row_sig["ema_slow"])
        texts["ema_trend"] = (
            f"ema_trend(EMA{p['ema_fast']}={row_sig['ema_fast']:.2f} < "
            f"EMA{p['ema_slow']}={row_sig['ema_slow']:.2f})"
        )

    # BB breakout
    if direction == "long":
        flags["bb_breakout"] = bool(row_sig["close"] > row_sig["bb_upper"])
        texts["bb_breakout"] = f"bb_breakout(Close {row_sig['close']:.2f} > BBupper {row_sig['bb_upper']:.2f})"
    else:
        flags["bb_breakout"] = bool(row_sig["close"] < row_sig["bb_lower"])
        texts["bb_breakout"] = f"bb_breakout(Close {row_sig['close']:.2f} < BBlower {row_sig['bb_lower']:.2f})"

    # MACD hist trend
    macd_prev = float(row_prev["macd_hist"]) if row_prev is not None and not isnan(row_prev["macd_hist"]) else 0.0
    macd_curr = float(row_sig["macd_hist"])
    if direction == "long":
        flags["macd_hist_rising"] = bool(macd_curr > 0 and macd_curr > macd_prev)
        texts["macd_hist_rising"] = f"macd_hist_rising({macd_prev:.4f} → {macd_curr:.4f})"
    else:
        flags["macd_hist_rising"] = bool(macd_curr < 0 and macd_curr < macd_prev)
        texts["macd_hist_rising"] = f"macd_hist_falling({macd_prev:.4f} → {macd_curr:.4f})"

    # Volume spike
    vol_mult = p.get("vol_spike_mult", 1.5)
    vol_avg = float(row_sig.get("vol_avg", 0.0) or 0.0)
    vol = float(row_sig.get("volume", 0.0) or 0.0)
    flags["vol_spike"] = bool(vol_avg > 0 and vol >= vol_mult * vol_avg)
    texts["vol_spike"] = f"vol_spike({vol:.0f} ≥ {vol_mult:.2f}×{vol_avg:.0f})"

    # ATR min
    atr_min = p.get("atr_min", 0.0)
    atr_val = float(row_sig.get("atr", 0.0) or 0.0)
    flags["atr_ok"] = bool(atr_val >= atr_min)
    texts["atr_ok"] = f"atr_ok(ATR {atr_val:.3f} ≥ min {atr_min:.3f})"

    return flags, texts


def decide_confluence(flags: Dict[str, bool], p: Dict[str, Any]) -> Tuple[bool, int]:
    keys = p["confluence_list"]
    hits = sum(1 for k in keys if flags.get(k, False))
    if p["rule_mode"] == "strict":
        ok = all(flags.get(k, False) for k in keys)
    else:
        ok = hits >= p.get("any_k", 3)
    return ok, hits


# ========================
# Levels (SL / Target)
# ========================
def calc_levels(entry: float, row_sig: pd.Series, p: Dict[str, Any], direction: str):
    atr = float(row_sig.get("atr", 0.0) or 0.0)
    mult_sl = p.get("atr_mult_sl", 1.5)
    mult_tg = p.get("atr_mult_tg", 1.5)

    if direction == "long":
        sl = entry - atr * mult_sl
        tgt = entry + atr * mult_tg
        sl_reason = f"SL = Entry − ATR×{mult_sl} = {entry:.2f} − ({atr:.3f}×{mult_sl})"
        tgt_reason = f"Target = Entry + ATR×{mult_tg} = {entry:.2f} + ({atr:.3f}×{mult_tg})"
    else:
        sl = entry + atr * mult_sl
        tgt = entry - atr * mult_tg
        sl_reason = f"SL = Entry + ATR×{mult_sl} = {entry:.2f} + ({atr:.3f}×{mult_sl})"
        tgt_reason = f"Target = Entry − ATR×{mult_tg} = {entry:.2f} − ({atr:.3f}×{mult_tg})"
    return float(sl), sl_reason, float(tgt), tgt_reason


# ========================
# Backtest (signals at t, entries at t+1 open)
# ========================
def backtest(df_raw: pd.DataFrame, p: Dict[str, Any], trade_type: str = "Long", last_n: Optional[int] = None):
    prepared, meta = detect_and_prepare(df_raw)
    if prepared is None:
        raise ValueError(
            f"Could not detect OHLC columns. Mapping: {meta['col_map']} | Original: {meta['original_columns']}"
        )

    if last_n is not None and last_n > 0:
        prepared = prepared.iloc[-last_n:].copy()

    d = add_indicators(prepared, p)
    trades = []
    position = None

    # iterate to second-last bar (need t+1 open for entry/exit)
    for i in range(2, len(d) - 1):
        sig = d.iloc[i]      # signal bar t (we know close & indicators)
        prev = d.iloc[i - 1] # t-1 for MACD trend
        nxt = d.iloc[i + 1]  # t+1 bar (entry/exit at OPEN or intraday SL/TGT)

        allow_long = trade_type in ("Long", "All")
        allow_short = trade_type in ("Short", "All")

        if position is None:
            enter_long = enter_short = False
            flags_long = flags_short = {}
            texts_long = texts_short = {}
            hits_long = hits_short = 0

            if allow_long:
                flags_long, texts_long = eval_confluences(sig, prev, p, "long")
                enter_long, hits_long = decide_confluence(flags_long, p)

            if allow_short:
                flags_short, texts_short = eval_confluences(sig, prev, p, "short")
                enter_short, hits_short = decide_confluence(flags_short, p)

            direction = None
            if enter_long or enter_short:
                if enter_long and (not enter_short or hits_long >= hits_short):
                    direction = "BUY"
                    entry = float(nxt["open"])  # t+1 open
                    sl, slr, tgt, tgtr = calc_levels(entry, sig, p, "long")
                    position = {
                        "dir": "BUY",
                        "entry_date": nxt.name,
                        "entry": entry,
                        "sl": sl,
                        "tgt": tgt,
                        "sig_texts": texts_long,
                        "sig_flags": flags_long,
                        "sig_hits": hits_long,
                    }
                elif enter_short:
                    direction = "SELL"
                    entry = float(nxt["open"])
                    sl, slr, tgt, tgtr = calc_levels(entry, sig, p, "short")
                    position = {
                        "dir": "SELL",
                        "entry_date": nxt.name,
                        "entry": entry,
                        "sl": sl,
                        "tgt": tgt,
                        "sig_texts": texts_short,
                        "sig_flags": flags_short,
                        "sig_hits": hits_short,
                    }
                if position:
                    position["sl_reason"] = slr
                    position["tgt_reason"] = tgtr
                    position["sig_date"] = sig.name
        else:
            # manage open position using t+1 bar (nxt)
            exit_price = None
            exit_reason = None

            if position["dir"] == "BUY":
                if nxt["low"] <= position["sl"]:
                    exit_price = position["sl"]; exit_reason = "StopLoss"
                elif nxt["high"] >= position["tgt"]:
                    exit_price = position["tgt"]; exit_reason = "Target"
                else:
                    flags_short, _ = eval_confluences(sig, prev, p, "short")
                    rev_ok, _ = decide_confluence(flags_short, p)
                    if rev_ok:
                        exit_price = float(nxt["open"]); exit_reason = "Reversal@Open"

            else:  # SELL
                if nxt["high"] >= position["sl"]:
                    exit_price = position["sl"]; exit_reason = "StopLoss"
                elif nxt["low"] <= position["tgt"]:
                    exit_price = position["tgt"]; exit_reason = "Target"
                else:
                    flags_long, _ = eval_confluences(sig, prev, p, "long")
                    rev_ok, _ = decide_confluence(flags_long, p)
                    if rev_ok:
                        exit_price = float(nxt["open"]); exit_reason = "Reversal@Open"

            if exit_price is not None:
                pnl = (exit_price - position["entry"]) if position["dir"] == "BUY" else (position["entry"] - exit_price)
                confs_keys = p["confluence_list"]
                confs_matched = sum(1 for k in confs_keys if position["sig_flags"].get(k, False))
                trades.append({
                    "Signal Date": position["sig_date"],
                    "Entry Date": position["entry_date"],
                    "Exit Date": nxt.name,
                    "Direction": position["dir"],
                    "Entry": position["entry"],
                    "SL": position["sl"],
                    "Target": position["tgt"],
                    "Exit": float(exit_price),
                    "PnL": float(pnl),
                    "Confluences": confs_matched,
                    "ConfsTotal": len(confs_keys),
                    "Logic (with values)": " | ".join([position["sig_texts"][k] for k in confs_keys if position["sig_flags"].get(k, False)]),
                    "SL Reason": position["sl_reason"],
                    "Target Reason": position["tgt_reason"],
                    "Exit Reason": exit_reason,
                })
                position = None

    # If still open, close EOD at last close (optional)
    if position is not None:
        lastbar = d.iloc[-1]
        pnl = (lastbar["close"] - position["entry"]) if position["dir"] == "BUY" else (position["entry"] - lastbar["close"])
        confs_keys = p["confluence_list"]
        confs_matched = sum(1 for k in confs_keys if position["sig_flags"].get(k, False))
        trades.append({
            "Signal Date": position["sig_date"],
            "Entry Date": position["entry_date"],
            "Exit Date": lastbar.name,
            "Direction": position["dir"],
            "Entry": position["entry"],
            "SL": position["sl"],
            "Target": position["tgt"],
            "Exit": float(lastbar["close"]),
            "PnL": float(pnl),
            "Confluences": confs_matched,
            "ConfsTotal": len(confs_keys),
            "Logic (with values)": " | ".join([position["sig_texts"][k] for k in confs_keys if position["sig_flags"].get(k, False)]),
            "SL Reason": position["sl_reason"],
            "Target Reason": position["tgt_reason"],
            "Exit Reason": "EOD Close",
        })

    trades_df = pd.DataFrame(trades)

    metrics = {
        "Net PnL": 0.0,
        "Trades": 0,
        "Win Rate": 0.0,
        "Avg Hold (bars)": 0.0,
        "Trades / 100 bars": 0.0,
    }
    if not trades_df.empty:
        metrics["Net PnL"] = float(trades_df["PnL"].sum())
        metrics["Trades"] = int(len(trades_df))
        metrics["Win Rate"] = float((trades_df["PnL"] > 0).mean() * 100)
        # crude hold proxy from dates (daily): difference in days
        try:
            ed = pd.to_datetime(trades_df["Entry Date"])
            xd = pd.to_datetime(trades_df["Exit Date"])
            holds = (xd - ed).dt.days.clip(lower=0)
            metrics["Avg Hold (bars)"] = float(holds.mean()) if len(holds) else 0.0
        except Exception:
            metrics["Avg Hold (bars)"] = 0.0
        metrics["Trades / 100 bars"] = round(100.0 * len(d) and 100.0 * len(trades_df) / len(d), 2)

    return trades_df, metrics, d


def dynamic_confidence(trades_df: pd.DataFrame, direction: str, confs_matched: int) -> float:
    if trades_df is None or trades_df.empty:
        return 0.0
    subset = trades_df[(trades_df["Direction"] == direction) & (trades_df["Confluences"] == confs_matched)]
    if not subset.empty:
        return float((subset["PnL"] > 0).mean() * 100)
    subset2 = trades_df[trades_df["Direction"] == direction]
    if not subset2.empty:
        return float((subset2["PnL"] > 0).mean() * 100)
    return float((trades_df["PnL"] > 0).mean() * 100)


# ========================
# Optimization (Grid Search)
# ========================
def generate_param_grid() -> List[Dict[str, Any]]:
    grid = []
    ma_short_opts = [7, 10, 12, 15]
    ma_long_opts = [20, 30, 50, 100]
    ema_fast_opts = [9, 10, 12]
    ema_slow_opts = [20, 21, 26]
    bb_period_opts = [18, 20, 22]
    vol_mult_opts = [1.2, 1.5, 1.8, 2.0]
    atr_mult_sl_opts = [1.0, 1.25, 1.5, 2.0]
    atr_mult_tg_opts = [1.0, 1.5, 2.0, 2.5]
    any_k_opts = [2, 3, 4]
    rule_modes = ["any_k"]  # keep your layout/logic; can add "strict" if needed

    for ms in ma_short_opts:
        for ml in ma_long_opts:
            if ms >= ml:
                continue
            for ef in ema_fast_opts:
                for es in ema_slow_opts:
                    if ef >= es:
                        continue
                    for bbp in bb_period_opts:
                        for vm in vol_mult_opts:
                            for slm in atr_mult_sl_opts:
                                for tgm in atr_mult_tg_opts:
                                    for ak in any_k_opts:
                                        for rm in rule_modes:
                                            grid.append({
                                                "ma_short": ms,
                                                "ma_long": ml,
                                                "ema_fast": ef,
                                                "ema_slow": es,
                                                "bb_period": bbp,
                                                "bb_k": 2,
                                                "vol_period": 20,
                                                "vol_spike_mult": vm,
                                                "atr_period": 14,
                                                "atr_min": 0.0,
                                                "atr_mult_sl": slm,
                                                "atr_mult_tg": tgm,
                                                "target_atr": True,
                                                "confluence_list": ["ma_cross", "ema_trend", "macd_hist_rising", "vol_spike", "bb_breakout", "atr_ok"],
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
