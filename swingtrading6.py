# app.py
import re
from math import isnan
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(
    page_title="Swing (EOD) — No Lookahead",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ========================
# Helpers
# ========================
def _clean_name(s: str) -> str:
    s = str(s or "").lower().strip()
    return re.sub(r"[^a-z0-9]", "", s)


def detect_and_prepare(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Detect OHLCV + Date columns (fuzzy), coerce numeric, set Date index, sort."""
    df = df_raw.copy()
    orig_cols = list(df.columns)
    cleaned_map = {_clean_name(c): c for c in orig_cols}

    def find(cands: List[str]):
        for cand in cands:
            key = _clean_name(cand)
            if key in cleaned_map:
                return cleaned_map[key]
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
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ma_short"] = d["close"].rolling(p["short_ma"], min_periods=1).mean()
    d["ma_long"] = d["close"].rolling(p["long_ma"], min_periods=1).mean()

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

    # MA cross
    if direction == "long":
        flags["ma_cross"] = bool(row_sig["ma_short"] > row_sig["ma_long"])
        texts["ma_cross"] = f"ma_cross(MA{p['short_ma']}>{p['long_ma']}: {row_sig['ma_short']:.2f} > {row_sig['ma_long']:.2f})"
    else:
        flags["ma_cross"] = bool(row_sig["ma_short"] < row_sig["ma_long"])
        texts["ma_cross"] = f"ma_cross(MA{p['short_ma']}<{p['long_ma']}: {row_sig['ma_short']:.2f} < {row_sig['ma_long']:.2f})"

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
    texts["vol_spike"] = f"vol_spike(Vol {vol:.0f} ≥ {vol_mult}×Avg {vol_avg:.0f})"

    # ATR min
    atr_min = p.get("atr_min", 0.0)
    atr_val = float(row_sig.get("atr", 0.0) or 0.0)
    flags["atr_ok"] = bool(atr_val >= atr_min)
    texts["atr_ok"] = f"atr_ok(ATR {atr_val:.2f} ≥ min {atr_min:.2f})"

    return flags, texts


def decide_confluence(flags: Dict[str, bool], p: Dict[str, Any]) -> Tuple[bool, int]:
    keys = p["confluence_list"]
    hits = sum(1 for k in keys if flags.get(k, False))
    if p["rule_mode"] == "strict":
        ok = all(flags.get(k, False) for k in keys)
    else:
        ok = hits >= p.get("any_k", 2)
    return ok, hits


# ========================
# Levels (SL / Target) — original style but t+1 entry aware
# ========================
def calc_levels(entry: float, row_sig: pd.Series, p: Dict[str, Any], direction: str):
    atr = float(row_sig.get("atr", 0.0) or 0.0)
    # SL
    if p.get("use_atr_sl", True):
        mult = p.get("atr_mult", 1.5)
        if direction == "long":
            sl = entry - atr * mult
            sl_reason = f"SL = Entry − ATR×{mult} = {entry:.2f} − ({atr:.3f}×{mult})"
        else:
            sl = entry + atr * mult
            sl_reason = f"SL = Entry + ATR×{mult} = {entry:.2f} + ({atr:.3f}×{mult})"
    else:
        pct = p.get("sl_pct", 1.5)
        if direction == "long":
            sl = entry * (1 - pct / 100)
            sl_reason = f"SL = Entry − {pct}% = {sl:.2f}"
        else:
            sl = entry * (1 + pct / 100)
            sl_reason = f"SL = Entry + {pct}% = {sl:.2f}"

    # Target
    if p.get("target_atr", True):
        mult = p.get("target_atr_mult", 1.5)
        if direction == "long":
            tgt = entry + atr * mult
            tgt_reason = f"Target = Entry + ATR×{mult} = {entry:.2f} + ({atr:.3f}×{mult})"
        else:
            tgt = entry - atr * mult
            tgt_reason = f"Target = Entry − ATR×{mult} = {entry:.2f} − ({atr:.3f}×{mult})"
    else:
        pct = p.get("target_pct", 2.0)
        if direction == "long":
            tgt = entry * (1 + pct / 100)
            tgt_reason = f"Target = Entry + {pct}% = {tgt:.2f}"
        else:
            tgt = entry * (1 - pct / 100)
            tgt_reason = f"Target = Entry − {pct}% = {tgt:.2f}"

    return float(sl), sl_reason, float(tgt), tgt_reason


# ========================
# Backtest (signals at t, entries at t+1 open)
# ========================
def backtest(df_raw: pd.DataFrame, p: Dict[str, Any], trade_type: str = "Long"):
    prepared, meta = detect_and_prepare(df_raw)
    if prepared is None:
        raise ValueError(
            f"Could not detect OHLC columns. Mapping: {meta['col_map']} | Original: {meta['original_columns']}"
        )

    d = add_indicators(prepared, p)
    trades = []
    position = None

    # iterate to second-last bar (need t+1 open for entry/exit)
    for i in range(2, len(d) - 1):
        sig = d.iloc[i]      # signal bar t (we know close & indicators)
        prev = d.iloc[i - 1] # t-1 for MACD trend
        nxt = d.iloc[i + 1]  # t+1 bar (entry/exit at OPEN or intraday SL/TGT)

        # Evaluate long & short signals on t close
        allow_long = trade_type in ("Long", "All")
        allow_short = trade_type in ("Short", "All")

        if position is None:
            enter_long = enter_short = False
            flags_long = flags_short = {}
            texts_long = texts_short = {}

            if allow_long:
                flags_long, texts_long = eval_confluences(sig, prev, p, "long")
                enter_long, hits_long = decide_confluence(flags_long, p)
            else:
                hits_long = 0

            if allow_short:
                flags_short, texts_short = eval_confluences(sig, prev, p, "short")
                enter_short, hits_short = decide_confluence(flags_short, p)
            else:
                hits_short = 0

            # If both true, pick higher confluence hits
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
                    # check reversal at t close (sig); if reversal, exit at next open (which is nxt.open we already used)
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
        metrics["Trades / 100 bars"] = round(100.0 * len(trades_df) / max(1, len(d)), 2)

    return trades_df, metrics, d


def dynamic_confidence(trades_df: pd.DataFrame, direction: str, confs_matched: int) -> float:
    if trades_df is None or trades_df.empty:
        return 0.0
    subset = trades_df[(trades_df["Direction"] == direction) & (trades_df["Confluences"] == confs_matched)]
    if not subset.empty:
        return float((subset["PnL"] > 0).mean() * 100)
    # fallback to direction-only stats
    subset2 = trades_df[trades_df["Direction"] == direction]
    if not subset2.empty:
        return float((subset2["PnL"] > 0).mean() * 100)
    return float((trades_df["PnL"] > 0).mean() * 100)


# ========================
# UI — Sidebar
# ========================
st.title("📈 Swing (EOD) Strategy — Entries at Next Open (t+1)")

with st.sidebar:
    st.header("Controls")
    upl = st.file_uploader("Upload Daily OHLC (CSV/XLSX)", type=["csv", "xlsx"])
    trade_type = st.selectbox("Trade Type", ["All", "Long", "Short"], index=1)

    st.markdown("**Confluences & Risk**")
    short_ma = st.number_input("Short MA", 5, 50, 10, 1)
    long_ma = st.number_input("Long MA", 10, 200, 50, 5)
    bb_period = st.number_input("BB Period", 10, 50, 20, 1)
    vol_mult = st.slider("Volume Spike ×Avg", 1.0, 3.0, 1.5, 0.1)
    atr_min = st.number_input("Min ATR", 0.0, 1000.0, 0.0, 0.5)
    atr_mult = st.slider("ATR SL/Target Mult", 0.5, 3.0, 1.5, 0.1)
    rule_mode = st.selectbox("Signal Rule", ["any_k", "strict"], index=0)
    any_k = st.number_input("any_k (if rule=any_k)", 1, 6, 3, 1)

    run = st.button("Run Backtest")

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

params = {
    "short_ma": int(short_ma),
    "long_ma": int(long_ma),
    "bb_period": int(bb_period),
    "bb_k": 2,
    "vol_period": 20,
    "vol_spike_mult": float(vol_mult),
    "atr_period": 14,
    "atr_min": float(atr_min),
    "use_atr_sl": True,
    "atr_mult": float(atr_mult),
    "target_atr": True,
    "target_atr_mult": float(atr_mult),
    "confluence_list": ["ma_cross", "ema_trend", "macd_hist_rising", "vol_spike", "bb_breakout", "atr_ok"],
    "rule_mode": rule_mode,
    "any_k": int(any_k),
}

trades_df, metrics, ind_df = backtest(raw, params, trade_type=trade_type)

# ========================
# Summary
# ========================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net PnL", f"{metrics['Net PnL']:.2f}")
c2.metric("Trades", f"{metrics['Trades']}")
c3.metric("Win Rate", f"{metrics['Win Rate']:.1f}%")
c4.metric("Avg Hold (bars)", f"{metrics['Avg Hold (bars)']:.2f}")
c5.metric("Trades / 100 bars", f"{metrics['Trades / 100 bars']:.2f}")

tab_bt, tab_live = st.tabs(["Backtest Results", "Live Next-Day Plan"])

# ========================
# Backtest Tab
# ========================
with tab_bt:
    st.subheader("Backtest Trades")
    if trades_df.empty:
        st.info("No trades generated with current parameters.")
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
            height=420,
        )
        st.download_button(
            "Download Trades CSV",
            data=trades_df.to_csv(index=False).encode("utf-8"),
            file_name="backtest_trades.csv",
            mime="text/csv",
        )

# ========================
# Live Tab (Next-Day Plan)
# ========================
with tab_live:
    st.subheader("Next-Day Recommendation (EOD generated)")

    if len(ind_df) < 3:
        st.info("Not enough data to form a next-day plan.")
    else:
        # Use last completed bar as signal day (t)
        sig = ind_df.iloc[-1]
        prev = ind_df.iloc[-2]

        allow_long = trade_type in ("Long", "All")
        allow_short = trade_type in ("Short", "All")

        plan_rows = []

        def add_plan(direction: str, flags, texts):
            keys = params["confluence_list"]
            ok, hits = decide_confluence(flags, params)
            if not ok:
                return
            # Entry will be at next day's open (unknown now); estimate with last close for planning
            est_entry = float(sig["close"])
            sl, slr, tgt, tgtr = calc_levels(est_entry, sig, params, "long" if direction == "BUY" else "short")
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
            fl, tl = eval_confluences(sig, prev, params, "long")
            add_plan("BUY", fl, tl)

        if allow_short:
            fs, ts = eval_confluences(sig, prev, params, "short")
            add_plan("SELL", fs, ts)

        if not plan_rows:
            st.info("No next-day recommendation based on the latest EOD candle.")
        else:
            live_df = pd.DataFrame(plan_rows)
            st.dataframe(live_df, use_container_width=True)

# ========================
# Quick Chart
# ========================
with st.expander("Chart (Close, MAs, Bollinger)"):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["close"], name="Close", mode="lines"))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["ma_short"], name=f"MA{params['short_ma']}", mode="lines"))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["ma_long"], name=f"MA{params['long_ma']}", mode="lines"))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["bb_upper"], name="BB Upper", mode="lines"))
        fig.add_trace(go.Scatter(x=ind_df.index, y=ind_df["bb_lower"], name="BB Lower", mode="lines"))
        fig.update_layout(height=460, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
