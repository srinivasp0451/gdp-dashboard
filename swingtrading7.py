# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools, random, math
from datetime import timedelta

# =========================================
# ---------- Data Cleaning Layer ----------
# =========================================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # unify names: strip, lower, replace non-alnum with underscore
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    # drop exact duplicate column names (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def coerce_numeric(series: pd.Series) -> pd.Series:
    # remove Indian commas, stray spaces/dashes
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")

def load_any(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def clean_ohlcv(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df_raw.copy())

    # map likely date fields to "date"
    for candidate in ["date", "datetime", "timestamp", "time", "dt"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "date"})
            break

    if "date" not in df.columns:
        raise ValueError("❌ No date-like column found. Columns: " + ", ".join(df.columns))

    # coerce date (try dayfirst too)
    date_parsed = pd.to_datetime(df["date"], errors="coerce", utc=False)
    if date_parsed.isna().all():
        date_parsed = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, utc=False)
    df["date"] = date_parsed

    # map common OHLCV aliases
    alias_map = {
        "open": ["open", "o", "op"],
        "high": ["high", "h", "hi"],
        "low": ["low", "l", "lo"],
        "close": ["close", "c", "cl", "last", "settle", "adj_close", "adjusted_close"],
        "volume": ["volume", "vol", "qty", "traded_qty", "no_of_trades", "shares"]
    }
    for tgt, aliases in alias_map.items():
        if tgt not in df.columns:
            for a in aliases:
                if a in df.columns:
                    df = df.rename(columns={a: tgt})
                    break

    # ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])

    # final filtering
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols].copy()
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)
    else:
        df["volume"] = 0

    # sort ascending for proper backtest (no leakage)
    df = df.sort_values("date").reset_index(drop=True)

    # drop duplicates by date keeping first
    df = df.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    return df

# =========================================
# ---------- Indicators (manual) ----------
# =========================================
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    m_fast = ema(close, fast)
    m_slow = ema(close, slow)
    macd_line = m_fast - m_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, n=20, k=2.0):
    mid = sma(close, n)
    std = close.rolling(n).std()
    upper = mid + k*std
    lower = mid - k*std
    return mid, upper, lower

def true_range(df):
    prev_close = df["close"].shift()
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df, n=14): return true_range(df).rolling(n).mean()

def obv(close, volume):
    obv_vals = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv_vals.append(obv_vals[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv_vals.append(obv_vals[-1] - volume.iloc[i])
        else:
            obv_vals.append(obv_vals[-1])
    return pd.Series(obv_vals, index=close.index)

def cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md)

def stochastic(df, k=14, d=3):
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    kperc = 100 * (df["close"] - ll) / (hh - ll)
    dperc = kperc.rolling(d).mean()
    return kperc, dperc

def momentum(close, n=10):
    return close / close.shift(n) - 1.0

def adx(df, n=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_n = tr.rolling(n).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).sum() / atr_n)
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).sum() / atr_n)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(n).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["sma_fast"] = sma(c, 10)
    df["sma_slow"] = sma(c, 50)
    df["ema_fast"] = ema(c, 12)
    df["ema_slow"] = ema(c, 26)
    df["rsi14"] = rsi(c, 14)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(c, 12, 26, 9)
    df["bb_mid"], df["bb_up"], df["bb_lo"] = bollinger(c, 20, 2.0)
    df["atr14"] = atr(df, 14)
    df["obv"] = obv(df["close"], df["volume"])
    df["cci20"] = cci(df, 20)
    df["stoch_k"], df["stoch_d"] = stochastic(df, 14, 3)
    df["mom10"] = momentum(c, 10)
    df["adx14"] = adx(df, 14)
    return df

# =========================================
# ---------- Strategy & Backtest ----------
# =========================================
def entry_conditions(df, i, side, p):
    """
    Build both a boolean and a human-readable reason string (no leakage).
    Uses current-index i only.
    """
    row = df.iloc[i]
    reasons = []
    ok = True

    # SMA/EMA trend filter
    bull = (row["sma_fast"] > row["sma_slow"]) and (row["ema_fast"] > row["ema_slow"])
    bear = (row["sma_fast"] < row["sma_slow"]) and (row["ema_fast"] < row["ema_slow"])
    if side == "long":
        ok &= bull
        if bull: reasons.append("Uptrend: SMA/EMA fast above slow")
    else:
        ok &= bear
        if bear: reasons.append("Downtrend: SMA/EMA fast below slow")

    # RSI gate
    if side == "long":
        rsi_ok = row["rsi14"] <= p["rsi_buy_max"]
        ok &= rsi_ok
        reasons.append(f"RSI≤{p['rsi_buy_max']}" if rsi_ok else f"RSI>{p['rsi_buy_max']}")
    else:
        rsi_ok = row["rsi14"] >= p["rsi_sell_min"]
        ok &= rsi_ok
        reasons.append(f"RSI≥{p['rsi_sell_min']}" if rsi_ok else f"RSI<{p['rsi_sell_min']}")

    # MACD confirmation (optional)
    if p["macd_confirm"]:
        if side == "long":
            macd_ok = row["macd"] > row["macd_sig"]
        else:
            macd_ok = row["macd"] < row["macd_sig"]
        ok &= macd_ok
        reasons.append("MACD confirm" if macd_ok else "MACD reject")

    # ADX strength
    adx_ok = row["adx14"] >= p["adx_min"]
    ok &= adx_ok
    reasons.append(f"ADX≥{p['adx_min']}" if adx_ok else f"ADX<{p['adx_min']}")

    # Bollinger optional gate (band squeeze breakout bias)
    if p["use_bb"]:
        if side == "long":
            bb_ok = row["close"] >= row["bb_mid"]
        else:
            bb_ok = row["close"] <= row["bb_mid"]
        ok &= bb_ok
        reasons.append("BB-side OK" if bb_ok else "BB-side reject")

    return ok, "; ".join(reasons)

def exit_by_opposite(df, i, side, p):
    """Opposite signal exit condition using same gates."""
    opposite = "short" if side == "long" else "long"
    ok, _ = entry_conditions(df, i, opposite, p)
    return ok

def backtest_with_trailing(df_in: pd.DataFrame, p: dict, trade_side: str):
    """
    Discrete bar-by-bar backtest with next-bar execution, ATR SL/TP & trailing.
    Produces detailed trade log without leakage.
    """
    df = df_in.copy()
    n = len(df)
    position = 0  # +1 long, -1 short, 0 flat
    entry_price = None
    entry_i = None
    side = None
    sl = tp = trail = None
    trades = []
    wins = 0
    losses = 0

    for i in range(1, n):  # start from 1 to allow prev bar for execution
        # compute probability-of-profit at *entry time* from past trades only
        def current_pop():
            total = wins + losses
            return (wins / total) if total > 0 else 0.5

        if position == 0:
            # evaluate entries on bar i-1, execute at open/close of bar i?
            # We'll execute at bar i close = conservative for daily swing
            if trade_side in ["long", "long_short"]:
                ok, reason = entry_conditions(df, i-1, "long", p)
                if ok:
                    position = +1
                    side = "long"
                    entry_i = i
                    entry_price = df["close"].iloc[i]  # execute at current close
                    atr = df["atr14"].iloc[i]
                    sl = entry_price - p["sl_mult"] * atr
                    tp = entry_price + p["tp_mult"] * atr
                    trail = entry_price - p["trail_mult"] * atr if p["trail_mult"] > 0 else None
                    trades.append({
                        "entry_time": df["date"].iloc[i],
                        "side": "LONG",
                        "entry_price": entry_price,
                        "initial_sl": sl,
                        "target": tp,
                        "trail": trail,
                        "entry_reason": reason,
                        "pop_at_entry": round(current_pop(), 3)
                    })
                    continue

            if trade_side in ["short", "long_short"]:
                ok, reason = entry_conditions(df, i-1, "short", p)
                if ok:
                    position = -1
                    side = "short"
                    entry_i = i
                    entry_price = df["close"].iloc[i]
                    atr = df["atr14"].iloc[i]
                    sl = entry_price + p["sl_mult"] * atr
                    tp = entry_price - p["tp_mult"] * atr
                    trail = entry_price + p["trail_mult"] * atr if p["trail_mult"] > 0 else None
                    trades.append({
                        "entry_time": df["date"].iloc[i],
                        "side": "SHORT",
                        "entry_price": entry_price,
                        "initial_sl": sl,
                        "target": tp,
                        "trail": trail,
                        "entry_reason": reason,
                        "pop_at_entry": round(current_pop(), 3)
                    })
                    continue
        else:
            # manage open position (on bar i)
            price = df["close"].iloc[i]
            atr = df["atr14"].iloc[i]

            # trailing SL update
            if p["trail_mult"] > 0:
                if side == "long":
                    trail_candidate = price - p["trail_mult"] * atr
                    if trail is not None:
                        trail = max(trail, trail_candidate)
                    else:
                        trail = trail_candidate
                else:
                    trail_candidate = price + p["trail_mult"] * atr
                    if trail is not None:
                        trail = min(trail, trail_candidate)
                    else:
                        trail = trail_candidate

            hit = None
            exit_reason = None

            # check TP / SL (bar close approximation)
            if side == "long":
                if price >= tp:
                    hit, exit_reason = "Target", "TP hit"
                elif price <= (trail if trail is not None else sl):
                    hit, exit_reason = "Stop", "SL/Trail hit"
                elif exit_by_opposite(df, i-1, "long", p):
                    hit, exit_reason = "Opposite", "Opposite signal"
            else:
                if price <= tp:
                    hit, exit_reason = "Target", "TP hit"
                elif price >= (trail if trail is not None else sl):
                    hit, exit_reason = "Stop", "SL/Trail hit"
                elif exit_by_opposite(df, i-1, "short", p):
                    hit, exit_reason = "Opposite", "Opposite signal"

            # optional max bars in trade
            if hit is None and p["max_bars"] > 0 and (i - entry_i) >= p["max_bars"]:
                hit, exit_reason = "Time", f"Timed exit {p['max_bars']} bars"

            if hit is not None:
                # exit at current close
                exit_price = price
                pnl = (exit_price - entry_price) * (1 if side == "long" else -1)
                pnl_pct = pnl / entry_price
                if pnl > 0: wins += 1
                else: losses += 1

                trades[-1].update({
                    "exit_time": df["date"].iloc[i],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl_pct, 4)
                })

                # flatten
                position = 0
                entry_i = None
                entry_price = None
                side = None
                sl = tp = trail = None

    # build equity from trades (flat periods carry unchanged equity)
    equity = pd.Series(1.0, index=df.index)
    eq = 1.0
    t_idx = 0
    for i in range(len(df)):
        equity.iloc[i] = eq
        while t_idx < len(trades) and "exit_time" in trades[t_idx] and df["date"].iloc[i] == trades[t_idx]["exit_time"]:
            eq *= (1 + trades[t_idx]["pnl_pct"])
            equity.iloc[i] = eq
            t_idx += 1

    # daily returns & summary
    strat_ret = equity.pct_change().fillna(0.0)
    buy_hold_ret = df["close"].pct_change().fillna(0.0)

    return trades, strat_ret, buy_hold_ret

def evaluate(trades, strat_ret, buy_hold_ret):
    def cumprod(x): return (1 + x).cumprod()
    strat_curve = cumprod(strat_ret)
    bh_curve = cumprod(buy_hold_ret)
    strat_total = strat_curve.iloc[-1] - 1
    bh_total = bh_curve.iloc[-1] - 1
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades if t.get("pnl", 0) <= 0)
    accuracy = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
    # simple drawdown proxy
    peak = strat_curve.cummax()
    dd = (strat_curve / peak - 1).min()
    return {
        "strat_total": float(strat_total),
        "bh_total": float(bh_total),
        "accuracy": float(accuracy),
        "max_dd": float(dd),
        "num_trades": int(wins + losses),
        "wins": int(wins),
        "losses": int(losses)
    }

# =========================================
# ---------- Optimization Engine ----------
# =========================================
def parameter_space(wide=False):
    # base space
    space = {
        "rsi_buy_max": [55, 60, 65, 70],
        "rsi_sell_min": [30, 35, 40, 45],
        "macd_confirm": [True, False],
        "adx_min": [15, 20, 25],
        "sl_mult": [1.0, 1.5, 2.0],
        "tp_mult": [1.5, 2.0, 2.5, 3.0],
        "trail_mult": [0.0, 0.5, 1.0, 1.5],
        "max_bars": [0, 10, 20],  # 0 = disabled
        "use_bb": [True, False]
    }
    if wide:
        space.update({
            # widen RSIs and risk multiples
            "rsi_buy_max": [50, 55, 60, 65, 70, 75],
            "rsi_sell_min": [25, 30, 35, 40, 45, 50],
            "sl_mult": [0.8, 1.0, 1.5, 2.0, 2.5],
            "tp_mult": [1.2, 1.5, 2.0, 2.5, 3.0, 3.5],
            "trail_mult": [0.0, 0.5, 1.0, 1.5, 2.0],
            "adx_min": [10, 15, 20, 25, 30],
        })
    return space

def iterate_params(space, method, n_iter=150, seed=42):
    rng = random.Random(seed)
    keys = list(space.keys())
    if method == "Exhaustive Grid":
        # sorted for determinism
        lists = [space[k] for k in keys]
        for combo in itertools.product(*lists):
            yield dict(zip(keys, combo))
    else:
        for _ in range(n_iter):
            yield {k: rng.choice(space[k]) for k in keys}

def optimize(df_features: pd.DataFrame, trade_side: str, method: str, base_iters=200):
    # initial (narrow) search
    best = None
    space = parameter_space(wide=False)
    for params in iterate_params(space, method, n_iter=base_iters):
        trades, strat_ret, bh_ret = backtest_with_trailing(df_features, params, trade_side)
        summary = evaluate(trades, strat_ret, bh_ret)
        score = summary["strat_total"]
        if (best is None) or (score > best["summary"]["strat_total"]):
            best = {"params": params, "trades": trades, "strat_ret": strat_ret, "bh_ret": bh_ret, "summary": summary}

    # if not beating B&H, expand space and try more
    if best and best["summary"]["strat_total"] <= best["summary"]["bh_total"]:
        space = parameter_space(wide=True)
        bump_iters = base_iters * (3 if method == "Randomized Search" else 1)  # give randomized more tries
        for params in iterate_params(space, method, n_iter=bump_iters):
            trades, strat_ret, bh_ret = backtest_with_trailing(df_features, params, trade_side)
            summary = evaluate(trades, strat_ret, bh_ret)
            if summary["strat_total"] > best["summary"]["strat_total"]:
                best = {"params": params, "trades": trades, "strat_ret": strat_ret, "bh_ret": bh_ret, "summary": summary}

    return best

# =========================================
# ---------- Live Recommendation ----------
# =========================================
def live_recommendation(df_feat: pd.DataFrame, params: dict, trade_side: str, backtest_trades: list):
    """
    Use the very last bar (close) to produce a recommendation using current gates.
    """
    last_i = len(df_feat) - 1
    suggestions = []
    def pop_from_backtest():
        # overall historical win rate
        wins = sum(1 for t in backtest_trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in backtest_trades if t.get("pnl", 0) <= 0)
        return (wins / (wins + losses)) if (wins + losses) > 0 else 0.5

    def build(side):
        ok, reason = entry_conditions(df_feat, last_i, side, params)
        if not ok:
            return None
        price = df_feat["close"].iloc[last_i]
        a = df_feat["atr14"].iloc[last_i]
        if side == "long":
            sl = price - params["sl_mult"] * a
            tp = price + params["tp_mult"] * a
            trail = price - params["trail_mult"] * a if params["trail_mult"] > 0 else None
            logic_exit = "Target hit, ATR-trailing SL hit, or opposite short signal."
        else:
            sl = price + params["sl_mult"] * a
            tp = price - params["tp_mult"] * a
            trail = price + params["trail_mult"] * a if params["trail_mult"] > 0 else None
            logic_exit = "Target hit, ATR-trailing SL hit, or opposite long signal."
        return {
            "side": side.upper(),
            "entry_price": round(price, 2),
            "stop_loss": round(sl, 2),
            "target": round(tp, 2),
            "trail": (round(trail, 2) if trail is not None else None),
            "entry_logic": reason,
            "exit_logic": logic_exit,
            "probability_of_profit": round(pop_from_backtest(), 3)
        }

    if trade_side in ["long", "long_short"]:
        s = build("long")
        if s: suggestions.append(s)
    if trade_side in ["short", "long_short"]:
        s = build("short")
        if s: suggestions.append(s)

    if not suggestions:
        return {"message": "No fresh entry according to current filters.", "params_used": params}
    # If both possible, pick the one whose entry is closer to band side (small heuristic)
    if len(suggestions) == 2:
        return {"message": "Two-sided opportunity (pick one).", "ideas": suggestions, "params_used": params}
    return {"message": "Opportunity", "idea": suggestions[0], "params_used": params}

# =========================================
# ----------------- UI --------------------
# =========================================
st.set_page_config(page_title="Ultra-Robust Swing Strategy Optimizer", layout="wide")
st.title("📈 Ultra-Robust Swing Strategy (Manual Indicators, No Leakage)")

uploaded = st.file_uploader("Upload CSV/Excel with Date, Open, High, Low, Close, Volume", type=["csv", "xlsx"])
col1, col2, col3 = st.columns(3)
with col1:
    trade_side = st.selectbox("Trade Direction", ["long", "short", "long_short"])
with col2:
    opt_method = st.selectbox("Optimization Method", ["Exhaustive Grid", "Randomized Search"],index=1)
with col3:
    iters = st.number_input("Randomized iterations (if selected)", min_value=50, max_value=2000, value=400, step=50)

if uploaded:
    try:
        raw = load_any(uploaded)
        st.caption("Raw preview (first 5 rows)")
        st.dataframe(raw.head(5), use_container_width=True)

        df = clean_ohlcv(raw)
        st.caption("Cleaned OHLCV (first 5 rows)")
        st.dataframe(df.head(5), use_container_width=True)

        # Indicators
        df_feat = add_indicators(df).dropna().reset_index(drop=True)
        if len(df_feat) < 100:
            st.warning("Data after indicator warmup is short; results may be unstable.")

        # Optimize
        st.info("Running optimization…")
        best = optimize(df_feat, trade_side, opt_method, base_iters=(iters if opt_method=="Randomized Search" else 0))
        if best is None:
            st.error("No viable strategy found.")
            st.stop()

        params = best["params"]
        trades = best["trades"]
        summary = best["summary"]
        strat_ret = best["strat_ret"]
        bh_ret = best["bh_ret"]

        # ---------- Results ----------
        st.subheader("🔧 Best Parameters (used for backtest & live)")
        st.json(params)

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Strategy Return %", f"{summary['strat_total']*100:.2f}")
        colB.metric("Buy & Hold %", f"{summary['bh_total']*100:.2f}")
        colC.metric("Win Rate %", f"{summary['accuracy']*100:.2f}")
        colD.metric("Max Drawdown %", f"{summary['max_dd']*100:.2f}")

        # ---------- Trade Log ----------
        if len(trades) == 0 or "exit_price" not in trades[-1]:
            st.warning("No completed trades in backtest window. (Signals may exist but didn’t exit within the data.)")

        log_cols = ["entry_time","side","entry_price","initial_sl","target","trail",
                    "entry_reason","pop_at_entry","exit_time","exit_price","exit_reason","pnl","pnl_pct"]
        trade_log = pd.DataFrame(trades, columns=log_cols)
        st.subheader("📑 Detailed Trade Log")
        st.dataframe(trade_log, use_container_width=True)

        # ---------- Curves ----------
        st.subheader("📉 Price & Equity Curves")
        fig, ax = plt.subplots(figsize=(11,5))
        ax.plot(df_feat["date"], df_feat["close"], label="Close")
        ax.plot(df_feat["date"], df_feat["sma_fast"], label="SMA fast")
        ax.plot(df_feat["date"], df_feat["sma_slow"], label="SMA slow")
        ax.legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(11,4))
        eq = (1 + strat_ret).cumprod()
        bh = (1 + bh_ret).cumprod()
        ax2.plot(df_feat["date"], eq, label="Strategy Equity")
        ax2.plot(df_feat["date"], bh, label="Buy & Hold")
        ax2.legend()
        st.pyplot(fig2)

        # ---------- Live Recommendation ----------
        st.subheader("🔔 Live Recommendation (based on last candle close)")
        live = live_recommendation(df_feat, params, trade_side, trades)
        st.json(live)

        # ---------- Summary (≈100 words) ----------
        st.subheader("📝 Human-Readable Summary")
        msg = []
        msg.append(
            f"We optimized a swing strategy using manual indicators (SMA/EMA, RSI, MACD, Bollinger, ATR, OBV, CCI, "
            f"Stochastic, Momentum, ADX) without future data leakage. The best setup returned "
            f"{summary['strat_total']*100:.2f}% versus Buy & Hold {summary['bh_total']*100:.2f}%, with "
            f"{summary['accuracy']*100:.2f}% win rate. Entries require trend agreement, RSI bounds, optional MACD "
            f"confirmation, and ADX strength. Risk management uses ATR-based stop, target, and optional trailing stop; "
            f"time-based exits are optional. The live suggestion above is computed from the latest candle with the same "
            f"parameters, including entry price, SL, target, trailing plan, and estimated probability of profit derived "
            f"from historical wins observed in backtesting."
        )
        st.write(" ".join(msg))

    except Exception as e:
        st.error(f"Error: {e}")
