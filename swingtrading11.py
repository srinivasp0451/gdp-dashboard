# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="Swing Trading: Close-Only + Optimize to Target", layout="wide")
st.title("Swing Trading — Close-only confluence strategy (optimize to target)")

# -----------------------
# Upload + basic cleaning (keep your working logic intact)
# -----------------------
def read_any(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)

def clean_columns(df):
    df = df.copy()

    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "", regex=False)
                  .str.replace("_", "", regex=False)
                  .str.replace(".", "", regex=False))
    return df

def remove_commas(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
    return df

uploaded = st.file_uploader("Upload CSV or Excel (OHLCV)", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df_raw = read_any(uploaded)
    st.write(df_raw.columns)
    if "prev. close" in df_raw.columns:
        df_raw = df_raw.drop('prev. close',axis=1)

    if "prev.close" in df_raw.columns:
        df_raw = df_raw.drop('prev.close',axis=1)
    if "prevclose" in df_raw.columns:
        df_raw = df_raw.drop('prevclose',axis=1)
        
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df = clean_columns(df_raw)
df = remove_commas(df)
st.success(f"Loaded {uploaded.name} — shape {df.shape}")
st.write("Normalized columns:", list(df.columns))

cols = list(df.columns)
def find_col(cols, patterns):
    for c in cols:
        for p in patterns:
            if p in c:
                return c
    return None

date_col   = find_col(cols, ["date", "datetime", "timestamp", "time"])
open_col   = find_col(cols, ["open", "openprice", "o"])
high_col   = find_col(cols, ["high", "h"])
low_col    = find_col(cols, ["low", "l"])
close_col  = find_col(cols, ["close", "adjclose", "last", "ltp", "settle"])
volume_col = find_col(cols, ["volume", "vol", "qty", "turnover"])

st.markdown("**Detected columns**")
st.write({
    "date": date_col,
    "open": open_col,
    "high": high_col,
    "low": low_col,
    "close": close_col,
    "volume": volume_col
})

if date_col is None or close_col is None:
    st.error("Need at least Date and Close columns.")
    st.stop()

# parse date, sort, filter by end-date (start = min_date)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
start_date = df[date_col].min().date()
today = datetime.now().date()
st.markdown(f"Start date (from data): **{start_date}**")
end_date = st.date_input("Select end date (inclusive)", value=today)
mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
df = df.loc[mask].sort_values(by=date_col).reset_index(drop=True)

# coerce OHLCV to numeric
for c in [open_col, high_col, low_col, close_col, volume_col]:
    if c and c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[close_col]).reset_index(drop=True)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
df.index = pd.DatetimeIndex(df[date_col])

# Basic stats and close plot (int conversion for plot as you requested previously)
st.subheader("Data summary")
c1, c2, c3 = st.columns(3)
c1.write("Rows × Cols")
c1.write(df.shape)
c2.write("First / Last date")
c2.write(f"{df[date_col].min()} → {df[date_col].max()}")
c3.write("Close min / max")
c3.write(f"{float(df[close_col].min()):.4f} / {float(df[close_col].max()):.4f}")

st.subheader("Close (rounded int for view)")
df_plot = df.copy()
df_plot["close_int"] = df_plot[close_col].astype(float).round(0).astype("Int64")
st.line_chart(df_plot.set_index(date_col)["close_int"])

# -----------------------
# Indicators implementations
# -----------------------
# column alias for brevity:
O, H, L, C, V = open_col, high_col, low_col, close_col, volume_col

def SMA(s, n): return s.rolling(n, min_periods=1).mean()
def EMA(s, n): return s.ewm(span=n, adjust=False).mean()

def RSI(s, n=14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(n).mean()
    ma_down = down.rolling(n).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def MACD(s, fast=12, slow=26, sig=9):
    f = EMA(s, fast); l = EMA(s, slow)
    macd = f - l
    macd_sig = EMA(macd, sig)
    hist = macd - macd_sig
    return macd, macd_sig, hist

def Bollinger(s, n=20, k=2.0):
    mid = SMA(s, n); sd = s.rolling(n).std()
    return mid + k*sd, mid, mid - k*sd

def ATR(df_local, n=14):
    tr1 = (df_local[H] - df_local[L]).abs()
    tr2 = (df_local[H] - df_local[C].shift()).abs()
    tr3 = (df_local[L] - df_local[C].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def Momentum(s, n=10): return s.diff(n)

def Stochastic(df_local, k=14, d=3):
    low_k = df_local[L].rolling(k).min()
    high_k = df_local[H].rolling(k).max()
    kline = 100 * (df_local[C] - low_k) / (high_k - low_k + 1e-12)
    dline = kline.rolling(d).mean()
    return kline, dline

def ADX(df_local, n=14):
    up = df_local[H].diff()
    down = -df_local[L].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(df_local[H] - df_local[L]).abs(),
                    (df_local[H] - df_local[C].shift()).abs(),
                    (df_local[L] - df_local[C].shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).mean() / (atr_ + 1e-12))
    minus_di = 100 * (minus_dm.rolling(n).mean() / (atr_ + 1e-12))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.rolling(n).mean(), plus_di, minus_di

def OBV(df_local):
    if V is None or V not in df_local:
        return pd.Series(0.0, index=df_local.index)
    obv = [0.0]
    for i in range(1, len(df_local)):
        if df_local[C].iat[i] > df_local[C].iat[i-1]:
            obv.append(obv[-1] + (df_local[V].iat[i] or 0))
        elif df_local[C].iat[i] < df_local[C].iat[i-1]:
            obv.append(obv[-1] - (df_local[V].iat[i] or 0))
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df_local.index)

def VWAP(df_local):
    if V is None or V not in df_local:
        return pd.Series(np.nan, index=df_local.index)
    tp = (df_local[H] + df_local[L] + df_local[C]) / 3
    pv = tp * df_local[V]
    return pv.cumsum() / df_local[V].cumsum().replace(0, np.nan)

def CCI(df_local, n=20):
    tp = (df_local[H] + df_local[L] + df_local[C]) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True) + 1e-12
    return (tp - sma) / (0.015 * mad)

# default params
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
    "sl_atr_mult": 1.0
}

# compute all indicators into a DataFrame
def compute_indicators(df_local, p):
    s = df_local[C]
    r = df_local.copy()
    r["sma_f"] = SMA(s, p["sma_fast"])
    r["sma_s"] = SMA(s, p["sma_slow"])
    r["ema_f"] = EMA(s, p["ema_fast"])
    r["ema_s"] = EMA(s, p["ema_slow"])
    r["rsi"] = RSI(s, p["rsi_period"])
    macd, macd_sig, macd_hist = MACD(s, p["macd_fast"], p["macd_slow"], p["macd_signal"])
    r["macd"] = macd; r["macd_sig"] = macd_sig; r["macd_hist"] = macd_hist
    up, mid, lo = Bollinger(s, p["bb_period"], p["bb_k"])
    r["bb_up"] = up; r["bb_mid"]=mid; r["bb_lo"]=lo
    r["atr"] = ATR(r, p["atr_period"])
    r["mom"] = Momentum(s, p["mom_period"])
    k, d = Stochastic(r, p["stoch_k"], p["stoch_d"]); r["stoch_k"]=k; r["stoch_d"]=d
    adx, pdi, mdi = ADX(r, p["adx_period"]); r["adx"]=adx; r["pdi"]=pdi; r["mdi"]=mdi
    r["obv"] = OBV(r)
    r["vwap"] = VWAP(r)
    r["cci"] = CCI(r, p["cci_period"])
    return r

# confluence signal generation — returns DataFrame with signals, indicator lists and indicator values
def confluence_signals(df_local, p, side="Both"):
    X = compute_indicators(df_local, p)
    X = X.copy()
    long_list = []; short_list = []; total_votes=[]; sigs=[]
    # we'll also collect the actual numeric values we want at the row for logs
    for i in range(len(X)):
        row = X.iloc[i]
        lset, sset = set(), set()

        # SMA & EMA
        if not pd.isna(row["sma_f"]) and not pd.isna(row["sma_s"]):
            (lset if row["sma_f"] > row["sma_s"] else sset).add("SMA")
        if not pd.isna(row["ema_f"]) and not pd.isna(row["ema_s"]):
            (lset if row["ema_f"] > row["ema_s"] else sset).add("EMA")

        # MACD histogram
        if not pd.isna(row["macd_hist"]):
            (lset if row["macd_hist"] > 0 else sset).add("MACD")

        # RSI
        if not pd.isna(row["rsi"]):
            if row["rsi"] < 35: lset.add("RSI")
            elif row["rsi"] > 65: sset.add("RSI")

        # Bollinger
        if not pd.isna(row["bb_up"]) and not pd.isna(row["bb_lo"]):
            if row[C] < row["bb_lo"]: lset.add("BB_L")
            elif row[C] > row["bb_up"]: sset.add("BB_U")

        # Momentum
        if not pd.isna(row["mom"]):
            (lset if row["mom"] > 0 else sset).add("MOM")

        # Stochastic (K/D)
        if not (pd.isna(row["stoch_k"]) or pd.isna(row["stoch_d"])):
            if row["stoch_k"] > row["stoch_d"] and row["stoch_k"] < 30: lset.add("STOCH")
            elif row["stoch_k"] < row["stoch_d"] and row["stoch_k"] > 70: sset.add("STOCH")

        # ADX direction
        if not (pd.isna(row["adx"]) or pd.isna(row["pdi"]) or pd.isna(row["mdi"])):
            if row["adx"] > 20 and row["pdi"] > row["mdi"]: lset.add("ADX+")
            elif row["adx"] > 20 and row["mdi"] > row["pdi"]: sset.add("ADX-")

        # OBV trend (3 vs prior-3)
        if i >= 5:
            recent = X["obv"].iloc[i-2:i+1].mean()
            prev = X["obv"].iloc[i-5:i-2].mean()
            if recent > prev: lset.add("OBV")
            elif recent < prev: sset.add("OBV")

        # VWAP
        if not pd.isna(row["vwap"]):
            (lset if row[C] > row["vwap"] else sset).add("VWAP")

        # CCI extremes
        if not pd.isna(row["cci"]):
            if row["cci"] < -100: lset.add("CCI")
            elif row["cci"] > 100: sset.add("CCI")

        lv, sv = len(lset), len(sset)
        direction = 1 if lv > sv else (-1 if sv > lv else 0)

        if side == "Long" and direction == -1:
            direction = 0
            lv = sv = 0
        if side == "Short" and direction == 1:
            direction = 0
            lv = sv = 0

        final = direction if max(lv, sv) >= p["min_confluence"] else 0

        long_list.append(sorted(list(lset)))
        short_list.append(sorted(list(sset)))
        total_votes.append(max(lv, sv))
        sigs.append(final)

    X["indicators_long"] = long_list
    X["indicators_short"] = short_list
    X["long_votes"] = [len(x) for x in long_list]
    X["short_votes"] = [len(x) for x in short_list]
    X["total_votes"] = total_votes
    X["Signal"] = sigs
    return X

# -----------------------
# Backtest: CLOSE-only (entries and exits executed at CLOSE)
# capture indicator numeric values at entry and show formulas used for Target/SL
# -----------------------
def backtest_close_only(local_with_signals, p):
    trades = []
    in_pos = False
    entry_row = None
    entry_ts = None
    entry_price = None
    entry_side = 0
    entry_ind_vals = None
    first_close = float(local_with_signals[C].iloc[0])
    last_close = float(local_with_signals[C].iloc[-1])
    buy_hold_points = last_close - first_close
    # iterate by index
    for i in range(len(local_with_signals)):
        row = local_with_signals.iloc[i]
        ts = local_with_signals.index[i]
        sig = int(row["Signal"])
        # ENTRY at close of signal bar
        if (not in_pos) and sig != 0:
            in_pos = True
            entry_side = sig
            entry_ts = ts
            entry_price = float(row[C])
            # ATR to use
            atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else float(local_with_signals["atr"].median() if not local_with_signals["atr"].isna().all() else 1.0)
            target = entry_price + p["target_atr_mult"]*atr_val if entry_side==1 else entry_price - p["target_atr_mult"]*atr_val
            stop   = entry_price - p["sl_atr_mult"]*atr_val if entry_side==1 else entry_price + p["sl_atr_mult"]*atr_val
            entry_ind_vals = {
                "sma_f": safe_float(row.get("sma_f")),
                "sma_s": safe_float(row.get("sma_s")),
                "ema_f": safe_float(row.get("ema_f")),
                "ema_s": safe_float(row.get("ema_s")),
                "macd_hist": safe_float(row.get("macd_hist")),
                "rsi": safe_float(row.get("rsi")),
                "bb_up": safe_float(row.get("bb_up")),
                "bb_lo": safe_float(row.get("bb_lo")),
                "atr": atr_val,
                "mom": safe_float(row.get("mom")),
                "stoch_k": safe_float(row.get("stoch_k")),
                "stoch_d": safe_float(row.get("stoch_d")),
                "adx": safe_float(row.get("adx")),
                "pdi": safe_float(row.get("pdi")),
                "mdi": safe_float(row.get("mdi")),
                "obv": safe_float(row.get("obv")),
                "vwap": safe_float(row.get("vwap")),
                "cci": safe_float(row.get("cci"))
            }
            entry_indicators_list = (row["indicators_long"] if entry_side==1 else row["indicators_short"])
            confl = int(row["total_votes"])
            # store entry snapshot in temp object and continue
            entry_info = {
                "Entry Date": entry_ts,
                "Entry Price": entry_price,
                "Side": "Long" if entry_side==1 else "Short",
                "IndicatorsList": entry_indicators_list,
                "IndicatorValues": entry_ind_vals,
                "Confluences": confl,
                "Target": float(target),
                "Stop": float(stop),
                "Target Formula": f"{entry_price:.4f} {'+' if entry_side==1 else '-'} {p['target_atr_mult']} * ATR({p['atr_period']}) = {target:.4f}",
                "Stop Formula": f"{entry_price:.4f} {'-' if entry_side==1 else '+'} {p['sl_atr_mult']} * ATR({p['atr_period']}) = {stop:.4f}"
            }
            continue

        # if in position, check exit on CLOSE only
        if in_pos:
            price_close = float(row[C])
            exit_reason = None
            exit_flag = False
            # long
            if entry_side == 1:
                if price_close >= entry_info["Target"]:
                    exit_reason = "Target hit (close)"
                    exit_flag = True
                elif price_close <= entry_info["Stop"]:
                    exit_reason = "Stop hit (close)"
                    exit_flag = True
                elif row["Signal"] == -1 and int(row["total_votes"]) >= p["min_confluence"]:
                    exit_reason = "Opposite confluence (close)"
                    exit_flag = True
            else:
                if price_close <= entry_info["Target"]:
                    exit_reason = "Target hit (close)"
                    exit_flag = True
                elif price_close >= entry_info["Stop"]:
                    exit_reason = "Stop hit (close)"
                    exit_flag = True
                elif row["Signal"] == 1 and int(row["total_votes"]) >= p["min_confluence"]:
                    exit_reason = "Opposite confluence (close)"
                    exit_flag = True

            # final flatten at last row
            if (i == len(local_with_signals)-1) and in_pos and not exit_flag:
                exit_flag = True
                exit_reason = "End of data (close)"

            if exit_flag:
                exit_price = price_close
                points = (exit_price - entry_info["Entry Price"]) if entry_side==1 else (entry_info["Entry Price"] - exit_price)
                trades.append({
                    "Entry Date": entry_info["Entry Date"],
                    "Exit Date": ts,
                    "Side": entry_info["Side"],
                    "Entry Price": entry_info["Entry Price"],
                    "Exit Price": float(exit_price),
                    "Target": entry_info["Target"],
                    "Stop": entry_info["Stop"],
                    "Target Formula": entry_info["Target Formula"],
                    "Stop Formula": entry_info["Stop Formula"],
                    "ATR Used": entry_info["IndicatorValues"]["atr"],
                    "Points": float(points),
                    "Hold Days": (ts.date() - entry_info["Entry Date"].date()).days,
                    "Confluences": entry_info["Confluences"],
                    "IndicatorsList": entry_info["IndicatorsList"],
                    # include numeric indicator columns as separate fields for readability
                    **{f"iv_{k}": v for k, v in entry_info["IndicatorValues"].items()},
                    "Reason Exit": exit_reason
                })
                # reset
                in_pos = False
                entry_info = None
                entry_price = None
                entry_side = 0

    trades_df = pd.DataFrame(trades)
    total_points = trades_df["Points"].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = int((trades_df["Points"] > 0).sum()) if not trades_df.empty else 0
    prob_of_profit = wins / num_trades if num_trades>0 else 0.0
    strategy_pct = (total_points / abs(first_close) * 100) if abs(first_close) > 1e-12 else np.nan
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if abs(first_close) > 1e-12 else np.nan

    summary = {
        "total_points": float(total_points),
        "num_trades": int(num_trades),
        "wins": int(wins),
        "prob_of_profit": float(prob_of_profit),
        "buy_hold_points": float(buy_hold_points),
        "buy_hold_pct": float(buy_hold_pct),
        "strategy_pct": float(strategy_pct),
        "beat_buy_hold": float(total_points) > float(buy_hold_points)
    }
    return summary, trades_df

# helper to convert possible NaNs to floats
def safe_float(x):
    try:
        if pd.isna(x): return float("nan")
        return float(x)
    except:
        return float("nan")

# -----------------------
# Search/Optimization
# -----------------------
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
        "sl_atr_mult": [0.5, 1.0, 1.5]
    }

def make_candidates(grid, mode="Random Search", n_iter=200, exhaustive_cap=3000):
    keys = list(grid.keys())
    if mode == "Random Search":
        return [{k: random.choice(grid[k]) for k in keys} for _ in range(n_iter)]
    # exhaustive with cap sampling
    values = [grid[k] for k in keys]
    total = np.prod([len(v) for v in values])
    if total > exhaustive_cap:
        st.warning(f"Full grid size {total:,} > cap {exhaustive_cap:,}. Sampling {exhaustive_cap} combos.")
        return [{k: random.choice(grid[k]) for k in keys} for _ in range(exhaustive_cap)]
    prods = list(itertools.product(*values))
    return [dict(zip(keys, p)) for p in prods]

def optimize(df_prices, side="Both", mode="Random Search", n_iter=200, expected_return_pct=150):
    grid = search_space()
    candidates = make_candidates(grid, mode=mode, n_iter=n_iter)
    best_overall = {"score": -1e18, "params": None, "summary": None, "trades": None}
    best_meets_target = None
    progress = st.progress(0)
    total = len(candidates)
    for i, cand in enumerate(candidates):
        params = default_params.copy(); params.update(cand)
        X = confluence_signals(df_prices, params, side=side)
        summary, trades = backtest_close_only(X, params)
        # scoring function
        score = summary["total_points"] + 10*summary["prob_of_profit"] - 0.01*summary["num_trades"]
        # track overall best
        if score > best_overall["score"]:
            best_overall.update({"score": score, "params": params, "summary": summary, "trades": trades})
        # check expected return criterion (strategy_pct)
        strat_pct = summary.get("strategy_pct", -9999)
        if not np.isnan(strat_pct) and strat_pct >= expected_return_pct:
            # prefer best among those meeting expected target, maximizing strategy_pct then score
            if best_meets_target is None or (strat_pct > best_meets_target["summary"]["strategy_pct"]) or (strat_pct == best_meets_target["summary"]["strategy_pct"] and score > best_meets_target["score"]):
                best_meets_target = {"score": score, "params": params, "summary": summary, "trades": trades}
        if total>0 and (i % 5 == 0 or i==total-1):
            progress.progress(int((i+1)/total * 100))
    progress.empty()
    # choose result: if exists candidate meeting expected return, return that. else return best overall
    if best_meets_target is not None:
        return best_meets_target, True
    return best_overall, False

# -----------------------
# UI Controls for optimization / expected return
# -----------------------
st.sidebar.header("Strategy controls")
trade_side = st.sidebar.selectbox("Trade Side", ["Both", "Long", "Short"], index=0)
search_mode = st.sidebar.selectbox("Search Mode", ["Random Search", "Exhaustive Grid Search"], index=0)
n_iter = st.sidebar.number_input("Random Search iterations", min_value=50, max_value=5000, value=300, step=50)
expected_return_pct = st.sidebar.number_input("Expected strategy return (%) — optimizer will prioritise this", min_value=1, max_value=5000, value=150, step=10)

if st.sidebar.button("Run Optimize & Backtest (Close-only)"):
    st.info("Optimization started. This may take time; progress shown.")
    # run optimization
    best_result, met_target = optimize(df, side=trade_side, mode=search_mode, n_iter=n_iter, expected_return_pct=expected_return_pct)
    best_params = best_result["params"]
    best_summary = best_result["summary"]
    best_trades = best_result["trades"]

    st.subheader("Best parameters (selected)")
    st.json(best_params)

    st.subheader("Backtest summary (selected)")
    # show both strategy and buy & hold metrics
    st.write({
        "Strategy total points": best_summary["total_points"],
        "Strategy percent (sum(points)/first_close *100)": f"{best_summary['strategy_pct']:.2f}%",
        "Buy & Hold points (last_close - first_close)": best_summary["buy_hold_points"],
        "Buy & Hold percent": f"{best_summary['buy_hold_pct']:.2f}%",
        "Number of trades": best_summary["num_trades"],
        "Wins": best_summary["wins"],
        "Probability of profit (historical)": f"{best_summary['prob_of_profit']:.2%}",
        "Beat buy & hold (points)": best_summary["beat_buy_hold"],
        "Met expected return target": bool(met_target)
    })

    st.markdown("#### Backtest trade log (detailed)")
    if best_trades is None or best_trades.empty:
        st.write("No trades generated by this parameter set.")
    else:
        tdf = best_trades.copy()
        # make indicator columns human friendly (already named iv_*)
        # also include Target/Stop formula as separate columns (already present)
        tdf_display = tdf[[
            "Entry Date","Exit Date","Side","Entry Price","Exit Price","Target","Stop",
            "Target Formula","Stop Formula","ATR Used","Points","Hold Days","Confluences","Reason Exit"
        ] + [c for c in tdf.columns if c.startswith("iv_")]].sort_values("Entry Date").reset_index(drop=True)
        # format small floats
        for c in ["Entry Price","Exit Price","Target","Stop","ATR Used","Points"]:
            if c in tdf_display.columns:
                tdf_display[c] = tdf_display[c].apply(lambda x: float(x) if pd.notna(x) else x)
        st.dataframe(tdf_display)

    # show whether meets the expected return explicitly
    meets_text = "✅ Met expected return target (selected candidate)" if met_target else "❌ Did NOT meet expected target — best candidate shown"
    st.markdown(f"**Expected return target**: {expected_return_pct:.1f}% → {meets_text}")

    # -----------------------
    # Live recommendation evaluated strictly at last candle CLOSE
    # -----------------------
    st.subheader("Live recommendation (evaluated at LAST candle CLOSE)")
    df_with = confluence_signals(df, best_params, side=trade_side)
    last_row = df_with.iloc[-1]
    last_close = float(last_row[C])
    sig = int(last_row["Signal"])
    conf = int(last_row["total_votes"])
    # compute ATR used
    atr_used = float(last_row["atr"]) if not pd.isna(last_row["atr"]) else float(df_with["atr"].median() if not df_with["atr"].isna().all() else 1.0)
    if sig == 0:
        st.info("No confluence signal at last candle close.")
        st.write({
            "Date (last close)": str(df.index[-1]),
            "Close": last_close,
            "Long/Short votes": f"{int(last_row['long_votes'])}/{int(last_row['short_votes'])}",
            "Total confluence": conf
        })
    else:
        side_txt = "Long" if sig==1 else "Short"
        # entry at last close (as requested)
        entry_level = last_close
        target = entry_level + best_params["target_atr_mult"]*atr_used if sig==1 else entry_level - best_params["target_atr_mult"]*atr_used
        stop   = entry_level - best_params["sl_atr_mult"]*atr_used if sig==1 else entry_level + best_params["sl_atr_mult"]*atr_used

        # assemble indicator values for display
        ind_values = {k.replace("iv_",""): safe_float(last_row[k]) for k in last_row.index if str(k).startswith("iv_")}
        st.write({
            "Evaluated At": str(df.index[-1]),
            "Side": side_txt,
            "Entry (Close)": float(entry_level),
            "Target (formula)": f"{entry_level:.4f} {'+' if sig==1 else '-'} {best_params['target_atr_mult']} * ATR({best_params['atr_period']}) = {target:.4f}",
            "Stop (formula)": f"{entry_level:.4f} {'-' if sig==1 else '+'} {best_params['sl_atr_mult']} * ATR({best_params['atr_period']}) = {stop:.4f}",
            "ATR used (value)": float(atr_used),
            "Confluence count": conf,
            "Indicators supporting (names)": (last_row["indicators_long"] if sig==1 else last_row["indicators_short"]),
            "Indicator values (snapshot at entry)": ind_values,
            "Estimated probability of profit (from backtest)": f"{best_summary['prob_of_profit']:.2%}"
        })

    st.success("Optimization & analysis complete.")
