# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import itertools
import random

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Swing Trading Strategy (Close-Only)", layout="wide")
st.title("Swing Trading Strategy — Close-Only Backtest & Live Signals")

# -----------------------------
# Helpers: IO / cleaning
# -----------------------------
def read_any(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace(".", "", regex=False)
    )
    return df

def remove_commas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
    return df

def find_col(cols, patterns):
    for c in cols:
        for p in patterns:
            if p in c:
                return c
    return None

# -----------------------------
# Upload & detect columns
# -----------------------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Please upload a file to continue.")
    st.stop()

try:
    df_raw = read_any(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

df = clean_columns(df_raw)
df = remove_commas(df)

st.success(f"Loaded: {uploaded.name} • shape: {df.shape}")
st.write("Normalized columns:", list(df.columns))

cols = list(df.columns)
date_col   = find_col(cols, ["date", "datetime", "timestamp", "time"])
open_col   = find_col(cols, ["open", "openprice", "o"])
high_col   = find_col(cols, ["high", "h"])
low_col    = find_col(cols, ["low", "l"])
close_col  = find_col(cols, ["close", "adjclose", "closeprice", "last", "ltp", "settle"])
volume_col = find_col(cols, ["volume", "vol", "qty", "turnover"])

st.subheader("Detected columns")
st.write({
    "date": date_col,
    "open": open_col,
    "high": high_col,
    "low": low_col,
    "close": close_col,
    "volume": volume_col,
})

if date_col is None or close_col is None:
    st.error("Must detect at least a date column and a close column.")
    st.stop()

# -----------------------------
# Parse date + filter by end-date (inclusive)
# -----------------------------
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df = df.sort_values(by=date_col).reset_index(drop=True)

start_date = df[date_col].min().date()
today = datetime.now().date()  # default end date = today; no disabled dates
st.markdown(f"**Start date (from data):** {start_date}")
end_date = st.date_input("Select end date (inclusive)", value=today)
mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
df = df.loc[mask].sort_values(by=date_col).reset_index(drop=True)

# -----------------------------
# Convert numeric OHLCV properly
# -----------------------------
for c in [open_col, high_col, low_col, close_col, volume_col]:
    if c and c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[close_col])  # require close

# -----------------------------
# Basic views & stats
# -----------------------------
st.markdown("### Data Preview")
c1, c2, c3 = st.columns(3)
with c1:
    st.write("Top 5 rows")
    st.dataframe(df.head())
with c2:
    st.write("Bottom 5 rows")
    st.dataframe(df.tail())
with c3:
    st.write("Shape:", df.shape)
    st.write("Min Close:", float(df[close_col].min()))
    st.write("Max Close:", float(df[close_col].max()))
    st.write("Min Date:", df[date_col].min())
    st.write("Max Date:", df[date_col].max())

# Close line chart (cast to int for plotting request)
df_plot = df.copy()
df_plot[close_col + "_int"] = df_plot[close_col].astype(float).round(0).astype("Int64")

st.subheader("Close Price (line)")
st.line_chart(df_plot.set_index(date_col)[close_col + "_int"])

# -----------------------------
# Correct Year × Month heatmap of returns (only if > 1 year)
# end-of-month close → monthly % change → pivot Year×Month
# -----------------------------
def monthly_return_heatmap(df_prices: pd.DataFrame, date_col: str, close_col: str):
    dfi = df_prices[[date_col, close_col]].dropna().copy()
    dfi = dfi.sort_values(by=date_col).reset_index(drop=True)
    dfi = dfi.set_index(date_col)
    monthly_close = dfi[close_col].resample("M").last().dropna()
    if len(monthly_close) < 2:
        return None, None
    monthly_ret = monthly_close.pct_change().dropna()
    mdf = monthly_ret.to_frame("ret")
    mdf["Year"] = mdf.index.year
    mdf["Month"] = mdf.index.month
    pivot = mdf.pivot(index="Year", columns="Month", values="ret").sort_index()
    return pivot, monthly_ret

data_span_days = (df[date_col].max() - df[date_col].min()).days
if data_span_days > 365:
    st.subheader("Monthly Returns Heatmap (Year × Month)")
    pivot, monthly_ret_series = monthly_return_heatmap(df, date_col, close_col)
    if pivot is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        # ensure 1..12 columns
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[sorted(pivot.columns)]
        sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", linewidths=0.2, linecolor="white", ax=ax)
        ax.set_xlabel("Month"); ax.set_ylabel("Year"); ax.set_title("Monthly % Return")
        st.pyplot(fig)

        st.subheader("Monthly Returns (bar)")
        monthly_ret_df = monthly_ret_series.to_frame("ret").reset_index()
        monthly_ret_df["label"] = monthly_ret_df[date_col].dt.strftime("%Y-%m")
        fig2, ax2 = plt.subplots(figsize=(12, 3.5))
        ax2.bar(monthly_ret_df["label"], monthly_ret_df["ret"])
        ax2.set_title("Monthly % Return (EOM-to-EOM)")
        ax2.set_ylabel("%")
        ax2.set_xticklabels(monthly_ret_df["label"], rotation=90)
        st.pyplot(fig2)
    else:
        st.info("Not enough monthly data to compute heatmap.")

# -----------------------------
# Volatility graph (rolling close-to-close)
# -----------------------------
st.subheader("Volatility (rolling close-to-close σ)")
dfi = df.set_index(date_col)
daily_ret = dfi[close_col].pct_change()
win = st.slider("Rolling window (days)", 10, 100, 20, step=5)
vol = daily_ret.rolling(win).std() * np.sqrt(252)
figv, axv = plt.subplots(figsize=(10, 3))
axv.plot(vol.index, vol.values)
axv.set_title(f"Annualized Volatility (window={win})")
axv.set_ylabel("σ (annualized)")
st.pyplot(figv)

# -----------------------------
# Strategy: indicators (no talib/pandas-ta)
# "Close"=close_col, "Open"=open_col, etc.
# -----------------------------
O, H, L, C, V = open_col, high_col, low_col, close_col, volume_col
dfi = df.set_index(date_col).copy()

def SMA(series, n): return series.rolling(n, min_periods=1).mean()
def EMA(series, n): return series.ewm(span=n, adjust=False).mean()

def RSI(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ma_up = up.rolling(n).mean()
    ma_down = down.rolling(n).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    fe = EMA(series, fast); se = EMA(series, slow)
    macd = fe - se
    sig = EMA(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def Bollinger(series, n=20, k=2):
    mid = SMA(series, n); std = series.rolling(n).std()
    return mid + k*std, mid, mid - k*std

def ATR(local, n=14):
    tr = pd.concat([
        (local[H] - local[L]).abs(),
        (local[H] - local[C].shift()).abs(),
        (local[L] - local[C].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def Momentum(series, n=10): return series.diff(n)

def Stochastic(local, k=14, d=3):
    low_min = local[L].rolling(k).min(); high_max = local[H].rolling(k).max()
    kline = 100*(local[C]-low_min)/(high_max-low_min+1e-10)
    dline = kline.rolling(d).mean()
    return kline, dline

def ADX(local, n=14):
    up_move = local[H].diff(); down_move = -local[L].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        local[H]-local[L],
        (local[H]-local[C].shift()).abs(),
        (local[L]-local[C].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100*(plus_dm.rolling(n).mean()/(atr+1e-10))
    minus_di = 100*(minus_dm.rolling(n).mean()/(atr+1e-10))
    dx = 100*(plus_di - minus_di).abs()/(plus_di+minus_di+1e-10)
    return dx.rolling(n).mean(), plus_di, minus_di

def OBV(local):
    obv = [0]
    for i in range(1, len(local)):
        if local[C].iloc[i] > local[C].iloc[i-1]:
            obv.append(obv[-1] + (local[V].iloc[i] if V in local else 0))
        elif local[C].iloc[i] < local[C].iloc[i-1]:
            obv.append(obv[-1] - (local[V].iloc[i] if V in local else 0))
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=local.index)

def VWAP(local):
    if V not in local: return pd.Series(np.nan, index=local.index)
    tp = (local[H] + local[L] + local[C]) / 3
    pv = tp * local[V]
    return pv.cumsum() / local[V].cumsum().replace(0, np.nan)

def CCI(local, n=20):
    tp = (local[H] + local[L] + local[C]) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True) + 1e-10
    return (tp - sma) / (0.015 * mad)

default_params = {
    "sma_fast": 9, "sma_slow": 21,
    "ema_fast": 8, "ema_slow": 34,
    "rsi_period": 14,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "bb_period": 20, "bb_k": 2.0,
    "atr_period": 14,
    "mom_period": 10,
    "stoch_k": 14, "stoch_d": 3,
    "adx_period": 14,
    "cci_period": 20,
    "min_confluence": 3,
    "target_atr_mult": 2.0,
    "sl_atr_mult": 1.0,
}

def compute_indicators(local, p):
    s = local[C]
    res = local.copy()
    res[f"sma_f"] = SMA(s, p["sma_fast"])
    res[f"sma_s"] = SMA(s, p["sma_slow"])
    res[f"ema_f"] = EMA(s, p["ema_fast"])
    res[f"ema_s"] = EMA(s, p["ema_slow"])
    r = RSI(s, p["rsi_period"]); res["rsi"] = r
    macd, sig, hist = MACD(s, p["macd_fast"], p["macd_slow"], p["macd_signal"])
    res["macd"]=macd; res["macd_sig"]=sig; res["macd_hist"]=hist
    up, mid, low = Bollinger(s, p["bb_period"], p["bb_k"])
    res["bb_up"]=up; res["bb_mid"]=mid; res["bb_lo"]=low
    res["atr"] = ATR(res, p["atr_period"])
    res["mom"] = Momentum(s, p["mom_period"])
    k, d = Stochastic(res, p["stoch_k"], p["stoch_d"]); res["stoch_k"]=k; res["stoch_d"]=d
    adx, pdi, mdi = ADX(res, p["adx_period"]); res["adx"]=adx; res["pdi"]=pdi; res["mdi"]=mdi
    res["obv"] = OBV(res)
    res["vwap"] = VWAP(res)
    res["cci"] = CCI(res, p["cci_period"])
    return res

def confluence_signals(local, p, side="Both"):
    x = compute_indicators(local, p)
    out = x.copy()
    longs = []; shorts = []; votes = []; sigs = []

    idx_list = list(x.index)
    for i, idx in enumerate(idx_list):
        row = x.iloc[i]
        lset, sset = set(), set()

        # SMA / EMA
        if not np.isnan(row["sma_f"]) and not np.isnan(row["sma_s"]):
            (lset if row["sma_f"] > row["sma_s"] else sset).add("SMA")
        if not np.isnan(row["ema_f"]) and not np.isnan(row["ema_s"]):
            (lset if row["ema_f"] > row["ema_s"] else sset).add("EMA")

        # MACD histogram
        if not np.isnan(row["macd_hist"]):
            (lset if row["macd_hist"] > 0 else sset).add("MACD")

        # RSI extremes
        if not np.isnan(row["rsi"]):
            if row["rsi"] < 35: lset.add("RSI")
            elif row["rsi"] > 65: sset.add("RSI")

        # Bollinger bounces
        if not np.isnan(row["bb_up"]) and not np.isnan(row["bb_lo"]):
            if row[C] < row["bb_lo"]: lset.add("BB_Lower")
            elif row[C] > row["bb_up"]: sset.add("BB_Upper")

        # Momentum
        if not np.isnan(row["mom"]):
            (lset if row["mom"] > 0 else sset).add("MOM")

        # Stochastic
        if not (np.isnan(row["stoch_k"]) or np.isnan(row["stoch_d"])):
            if row["stoch_k"] > row["stoch_d"] and row["stoch_k"] < 30: lset.add("STOCH")
            elif row["stoch_k"] < row["stoch_d"] and row["stoch_k"] > 70: sset.add("STOCH")

        # ADX direction if trending
        if not (np.isnan(row["adx"]) or np.isnan(row["pdi"]) or np.isnan(row["mdi"])):
            if row["adx"] > 20 and row["pdi"] > row["mdi"]: lset.add("ADX+")
            elif row["adx"] > 20 and row["mdi"] > row["pdi"]: sset.add("ADX-")

        # OBV trend (3-bar mean vs prior 3-bar mean)
        if i >= 5:
            recent = x["obv"].iloc[i-2:i+1].mean()
            prev = x["obv"].iloc[i-5:i-2].mean()
            if recent > prev: lset.add("OBV")
            elif recent < prev: sset.add("OBV")

        # VWAP
        if not np.isnan(row["vwap"]):
            (lset if row[C] > row["vwap"] else sset).add("VWAP")

        # CCI extremes
        if not np.isnan(row["cci"]):
            if row["cci"] < -100: lset.add("CCI")
            elif row["cci"] > 100: sset.add("CCI")

        lvotes, svotes = len(lset), len(sset)
        direction = 1 if lvotes > svotes else (-1 if svotes > lvotes else 0)

        # side filter
        if side == "Long" and direction == -1: direction = 0
        if side == "Short" and direction == 1: direction = 0

        final = direction if max(lvotes, svotes) >= p["min_confluence"] else 0
        longs.append(sorted(list(lset)))
        shorts.append(sorted(list(sset)))
        votes.append((lvotes, svotes, max(lvotes, svotes)))
        sigs.append(final)

    out["indicators_long"] = longs
    out["indicators_short"] = shorts
    lv, sv, tv = zip(*votes)
    out["long_votes"] = lv; out["short_votes"] = sv; out["total_votes"] = tv
    out["Signal"] = sigs
    return out

# -----------------------------
# Backtest: CLOSE-ONLY
# - Entries at signal bar CLOSE
# - Exits at CLOSE by target/SL (checked on close only) or opposite-signal confluence
# -----------------------------
def backtest_close_only(local_with_signals: pd.DataFrame, p: dict):
    trades = []
    in_pos = False
    side = 0
    entry_price = None
    entry_date = None
    entry_inds = None
    confl = 0
    target = None
    stop = None

    first_close = local_with_signals[C].iloc[0]
    last_close = local_with_signals[C].iloc[-1]
    buy_hold_points = last_close - first_close

    for i in range(len(local_with_signals)):
        row = local_with_signals.iloc[i]
        ts = local_with_signals.index[i]
        sig = row["Signal"]

        # Entry at CLOSE of signal bar
        if (not in_pos) and sig != 0:
            in_pos = True
            side = sig
            entry_date = ts
            entry_price = row[C]  # CLOSE
            atrv = row["atr"]
            if np.isnan(atrv) or atrv == 0:
                atrv = local_with_signals["atr"].median() or 1.0
            if side == 1:
                target = entry_price + p["target_atr_mult"] * atrv
                stop = entry_price - p["sl_atr_mult"] * atrv
                entry_inds = row["indicators_long"]
                confl = row["total_votes"]
            else:
                target = entry_price - p["target_atr_mult"] * atrv
                stop = entry_price + p["sl_atr_mult"] * atrv
                entry_inds = row["indicators_short"]
                confl = row["total_votes"]
            continue

        # Manage position at each subsequent CLOSE only
        if in_pos:
            # Check target/stop on close only
            price = row[C]
            exit_reason = None
            exit_flag = False

            if side == 1:
                if price >= target:
                    exit_reason = "Target (close)"
                    exit_flag = True
                elif price <= stop:
                    exit_reason = "Stop (close)"
                    exit_flag = True
                elif row["Signal"] == -1 and row["total_votes"] >= p["min_confluence"]:
                    exit_reason = "Opposite signal (close)"
                    exit_flag = True
            else:  # short
                if price <= target:
                    exit_reason = "Target (close)"
                    exit_flag = True
                elif price >= stop:
                    exit_reason = "Stop (close)"
                    exit_flag = True
                elif row["Signal"] == 1 and row["total_votes"] >= p["min_confluence"]:
                    exit_reason = "Opposite signal (close)"
                    exit_flag = True

            # End of data: always flatten on last close
            if (i == len(local_with_signals)-1) and in_pos and not exit_flag:
                exit_reason = "End of data (close)"
                exit_flag = True

            if exit_flag:
                exit_price = price
                points = (exit_price - entry_price) if side == 1 else (entry_price - exit_price)
                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": ts,
                    "Side": "Long" if side == 1 else "Short",
                    "Entry Price": float(entry_price),
                    "Exit Price": float(exit_price),
                    "Target": float(target),
                    "Stop": float(stop),
                    "Points": float(points),
                    "Hold Days": (ts.date() - entry_date.date()).days,
                    "Confluences": int(confl),
                    "Indicators": entry_inds,
                    "Reason Exit": exit_reason
                })
                in_pos = False
                side = 0
                entry_price = None
                entry_date = None
                entry_inds = None
                target = None
                stop = None

    trades_df = pd.DataFrame(trades)
    total_points = trades_df["Points"].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = (trades_df["Points"] > 0).sum() if num_trades > 0 else 0
    prob = (wins / num_trades) if num_trades > 0 else 0.0

    summary = {
        "total_points": float(total_points),
        "num_trades": int(num_trades),
        "wins": int(wins),
        "prob_of_profit": float(prob),
        "buy_hold_points": float(buy_hold_points),
        "beat_buy_hold": bool(total_points > buy_hold_points),
        ">=70%_of_abs_buy_hold": bool(total_points >= 0.7 * abs(buy_hold_points)),
    }
    return summary, trades_df

# -----------------------------
# Optimization (Random / Exhaustive with cap)
# -----------------------------
def search_space():
    return {
        "sma_fast": list(range(5, 16)),
        "sma_slow": list(range(18, 51, 2)),
        "ema_fast": list(range(5, 16)),
        "ema_slow": list(range(20, 61, 5)),
        "rsi_period": list(range(8, 21)),
        "macd_fast": [8, 10, 12, 16],
        "macd_slow": [20, 26, 34],
        "macd_signal": [7, 9, 12],
        "bb_period": [14, 20, 25],
        "bb_k": [1.5, 2.0, 2.5],
        "atr_period": [10, 14, 20],
        "mom_period": [5, 10, 20],
        "stoch_k": [10, 14, 20],
        "stoch_d": [3, 5],
        "adx_period": [14, 20],
        "cci_period": [14, 20],
        "min_confluence": [2, 3, 4, 5],
        "target_atr_mult": [1.0, 1.5, 2.0, 3.0],
        "sl_atr_mult": [0.5, 1.0, 1.5],
    }

def make_candidates(grid, mode="Random Search", n_iter=200, exhaustive_cap=4000):
    keys = list(grid.keys())
    if mode == "Random Search":
        return [{k: random.choice(grid[k]) for k in keys} for _ in range(n_iter)]
    # Exhaustive (but cap size)
    all_vals = [grid[k] for k in keys]
    total = np.prod([len(v) for v in all_vals])
    if total > exhaustive_cap:
        st.warning(f"Full exhaustive size {total:,} > cap {exhaustive_cap:,}. Sampling {exhaustive_cap:,} random combos.")
        return [{k: random.choice(grid[k]) for k in keys} for _ in range(exhaustive_cap)]
    prod = list(itertools.product(*all_vals))
    return [dict(zip(keys, tup)) for tup in prod]

def optimize(local_prices, side="Both", mode="Random Search", n_iter=200):
    grid = search_space()
    cands = make_candidates(grid, mode=mode, n_iter=n_iter)
    best = {"score": -1e18, "params": None, "summary": None, "trades": None}
    progress = st.progress(0)
    for i, cand in enumerate(cands):
        p = default_params.copy(); p.update(cand)
        sigs = confluence_signals(local_prices, p, side=side)
        summary, trades = backtest_close_only(sigs, p)
        # score: prioritize points, then prob, mild penalty for too many trades
        score = summary["total_points"] + 10*summary["prob_of_profit"] - 0.01*summary["num_trades"]
        if score > best["score"]:
            best.update({"score": score, "params": p, "summary": summary, "trades": trades})
        progress.progress(int((i+1)/len(cands)*100))
    progress.empty()
    return best

# -----------------------------
# UI: Side & Optimization & Run
# -----------------------------
st.sidebar.header("Strategy Controls")
trade_side = st.sidebar.selectbox("Trade Side", ["Both", "Long", "Short"], index=0)
search_mode = st.sidebar.selectbox("Optimization Mode", ["Random Search", "Exhaustive Grid Search"], index=0)
n_iter = st.sidebar.number_input("Random Search iterations", min_value=50, max_value=5000, value=200, step=50)

if st.sidebar.button("Run Backtest & Optimize (Close-Only)"):
    st.info("Running optimization… entries/exits strictly at candle CLOSE.")
    best = optimize(dfi, side=trade_side, mode=search_mode, n_iter=n_iter)

    st.subheader("Best Parameters")
    st.json(best["params"])

    st.subheader("Backtest Summary (Close-Only)")
    st.write(best["summary"])

    st.subheader("Trade Log")
    if best["trades"] is None or best["trades"].empty:
        st.write("No trades generated by the best strategy.")
    else:
        # augment with logic text & P(Profit)
        opp = best["summary"]["prob_of_profit"]
        tdf = best["trades"].copy()
        tdf["Prob. Profit (historical)"] = f"{opp:.2%}"
        tdf["Reason/Logic"] = tdf.apply(
            lambda r: f"Confluences={r['Confluences']} | Indicators={', '.join(r['Indicators']) if isinstance(r['Indicators'], list) else r['Indicators']}",
            axis=1,
        )
        st.dataframe(tdf[
            ["Entry Date","Exit Date","Side","Entry Price","Exit Price","Target","Stop","Points",
             "Hold Days","Confluences","Prob. Profit (historical)","Reason/Logic","Reason Exit"]
        ])

    # Show whether it meets your requirement
    meets = best["summary"]["beat_buy_hold"] or best["summary"][">=70%_of_abs_buy_hold"]
    st.markdown(f"**Beats Buy & Hold or ≥70% of |Buy&Hold|?**  {'✅ Yes' if meets else '❌ No'}")

    # -----------------------------
    # Live Recommendation — strictly at last candle close
    # -----------------------------
    st.subheader("Live Recommendation (evaluated at LAST candle CLOSE)")
    latest_index = dfi.index[-1]
    df_signals = confluence_signals(dfi, best["params"], side=trade_side)
    last_row = df_signals.iloc[-1]
    last_close = float(last_row[C])
    sig = int(last_row["Signal"])
    conf = int(last_row["total_votes"])
    if sig == 0:
        st.info("No signal at the last candle close.")
        st.write({
            "Date (last close)": str(latest_index),
            "Close": last_close,
            "Confluence votes": conf,
            "Long vs Short votes": f"{int(last_row['long_votes'])}/{int(last_row['short_votes'])}",
        })
    else:
        side_txt = "Long" if sig == 1 else "Short"
        atrv = float(last_row["atr"]) if not np.isnan(last_row["atr"]) else float(df_signals["atr"].median())
        tgt = last_close + best["params"]["target_atr_mult"]*atrv if sig==1 else last_close - best["params"]["target_atr_mult"]*atrv
        sl  = last_close - best["params"]["sl_atr_mult"]*atrv    if sig==1 else last_close + best["params"]["sl_atr_mult"]*atrv
        st.write({
            "Entry Date/Time": str(latest_index),
            "Side": side_txt,
            "Entry Level (Close)": last_close,
            "Target (Close-based)": float(tgt),
            "Stop (Close-based)": float(sl),
            "Confluence Count": conf,
            "Indicators": (last_row["indicators_long"] if sig==1 else last_row["indicators_short"]),
            "Estimated Prob. Profit (from backtest)": f"{best['summary']['prob_of_profit']:.2%}",
            "No. of days old": 0,
            "Points gained/lost so far": 0,
        })

# -----------------------------
# END
# -----------------------------
