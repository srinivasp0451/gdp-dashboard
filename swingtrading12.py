# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
from datetime import datetime

st.set_page_config(page_title="Swing Trading — Close-only + Expectancy Target", layout="wide")
st.title("Swing Trading — Close-only confluence (optimize to expected return & expectancy)")

# =========================
# File I/O + cleaning (kept intact in spirit)
# =========================
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
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df = clean_columns(df_raw)
df = remove_commas(df)
st.success(f"Loaded {uploaded.name} — shape {df.shape}")
st.write("Normalized columns:", list(df.columns))

# auto-detect columns (case/space tolerant)
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
    "date": date_col, "open": open_col, "high": high_col,
    "low": low_col, "close": close_col, "volume": volume_col
})

if date_col is None or close_col is None:
    st.error("Need at least Date and Close columns.")
    st.stop()

# parse date, sort, end-date filter (start = min date, end default today)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
start_date = df[date_col].min().date()
today = datetime.now().date()
st.markdown(f"Start date (from data): **{start_date}**")
end_date = st.date_input("Select end date (inclusive)", value=today)
mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
df = df.loc[mask].sort_values(by=date_col).reset_index(drop=True)

# numeric coercion
for c in [open_col, high_col, low_col, close_col, volume_col]:
    if c and c in df.columns:
        # keeping your earlier request: comment pd.to_numeric if it failed for you previously
        # df[c] = pd.to_numeric(df[c], errors="coerce")
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except:
            pass

df = df.dropna(subset=[close_col]).reset_index(drop=True)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
df.index = pd.DatetimeIndex(df[date_col])

# quick summary and close plot (int-rounded for view as you wanted)
st.subheader("Data summary")
c1, c2, c3 = st.columns(3)
c1.write("Rows × Cols"); c1.write(df.shape)
c2.write("First / Last date"); c2.write(f"{df[date_col].min()} → {df[date_col].max()}")
c3.write("Close min / max"); c3.write(f"{float(df[close_col].min()):.4f} / {float(df[close_col].max()):.4f}")

st.subheader("Close (rounded int for view)")
df_plot = df.copy()
df_plot["close_int"] = df_plot[close_col].astype(float).round(0).astype("Int64")
st.line_chart(df_plot.set_index(date_col)["close_int"])

# =========================
# Indicators (no TA-Lib)
# =========================
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

# =========================
# Parameters (base + optional filters)
# =========================
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

    # NEW (optional, default off / neutral)
    "use_htf_confirm": False,  # require HTF trend to agree
    "htf_multiplier": 3,       # how “higher” the HTF proxy is (multiplies periods)
    "trend_adx_min": 0         # 0 = disabled; else require ADX >= threshold
}

def compute_indicators(df_local, p):
    s = df_local[C]
    r = df_local.copy()

    # base indicators
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

    # HTF proxy (same series, longer periods as a lightweight higher timeframe confirmation)
    if p.get("use_htf_confirm", False):
        m = max(2, int(p.get("htf_multiplier", 3)))
        r["htf_ema_f"] = EMA(s, max(3, p["ema_fast"] * m))
        r["htf_ema_s"] = EMA(s, max(5, p["ema_slow"] * m))
        r["htf_trend_up"] = (r["htf_ema_f"] > r["htf_ema_s"]).astype(int)
        r["htf_trend_dn"] = (r["htf_ema_f"] < r["htf_ema_s"]).astype(int)
    else:
        r["htf_trend_up"] = 1  # neutral (no restriction)
        r["htf_trend_dn"] = 1

    return r

def confluence_signals(df_local, p, side="Both"):
    X = compute_indicators(df_local, p)
    X = X.copy()
    long_list = []; short_list = []; total_votes=[]; sigs=[]

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

        # Bollinger extremes
        if not pd.isna(row["bb_up"]) and not pd.isna(row["bb_lo"]):
            if row[C] < row["bb_lo"]: lset.add("BB_L")
            elif row[C] > row["bb_up"]: sset.add("BB_U")

        # Momentum
        if not pd.isna(row["mom"]):
            (lset if row["mom"] > 0 else sset).add("MOM")

        # Stochastic
        if not (pd.isna(row["stoch_k"]) or pd.isna(row["stoch_d"])):
            if row["stoch_k"] > row["stoch_d"] and row["stoch_k"] < 30: lset.add("STOCH")
            elif row["stoch_k"] < row["stoch_d"] and row["stoch_k"] > 70: sset.add("STOCH")

        # ADX direction
        if not (pd.isna(row["adx"]) or pd.isna(row["pdi"]) or pd.isna(row["mdi"])):
            if row["adx"] > 20 and row["pdi"] > row["mdi"]: lset.add("ADX+")
            elif row["adx"] > 20 and row["mdi"] > row["pdi"]: sset.add("ADX-")

        # OBV short-term slope
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

        # Count votes and decide raw direction
        lv, sv = len(lset), len(sset)
        direction = 1 if lv > sv else (-1 if sv > lv else 0)

        # Optional trend regime filter: require ADX >= threshold if enabled
        adx_min = p.get("trend_adx_min", 0)
        if adx_min and not pd.isna(row["adx"]):
            if row["adx"] < adx_min:
                direction = 0
                lv = sv = 0

        # Optional HTF confirmation (proxy): require HTF trend to agree
        if p.get("use_htf_confirm", False):
            if direction == 1 and int(row["htf_trend_up"]) != 1:
                direction = 0; lv = sv = 0
            if direction == -1 and int(row["htf_trend_dn"]) != 1:
                direction = 0; lv = sv = 0

        # Honor side selection
        if side == "Long" and direction == -1:
            direction = 0; lv = sv = 0
        if side == "Short" and direction == 1:
            direction = 0; lv = sv = 0

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

def safe_float(x):
    try:
        if pd.isna(x): return float("nan")
        return float(x)
    except:
        return float("nan")

# =========================
# Backtest — close-only entries & exits, with detailed logs
# =========================
def backtest_close_only(local_with_signals, p):
    trades = []
    in_pos = False
    entry_info = None
    first_close = float(local_with_signals[C].iloc[0])
    last_close = float(local_with_signals[C].iloc[-1])
    buy_hold_points = last_close - first_close

    for i in range(len(local_with_signals)):
        row = local_with_signals.iloc[i]
        ts = local_with_signals.index[i]
        sig = int(row["Signal"])

        # ENTRY: strictly at close of signal bar
        if (not in_pos) and sig != 0:
            in_pos = True
            entry_side = sig
            entry_price = float(row[C])
            atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else float(local_with_signals["atr"].median() if not local_with_signals["atr"].isna().all() else 1.0)
            target = entry_price + p["target_atr_mult"]*atr_val if entry_side==1 else entry_price - p["target_atr_mult"]*atr_val
            stop   = entry_price - p["sl_atr_mult"]*atr_val if entry_side==1 else entry_price + p["sl_atr_mult"]*atr_val

            entry_info = {
                "Entry Date": ts,
                "Entry Price": entry_price,
                "Side": "Long" if entry_side==1 else "Short",
                "Confluences": int(row["total_votes"]),
                "IndicatorsList": (row["indicators_long"] if entry_side==1 else row["indicators_short"]),
                "IndicatorValues": {
                    "sma_f": safe_float(row.get("sma_f")), "sma_s": safe_float(row.get("sma_s")),
                    "ema_f": safe_float(row.get("ema_f")), "ema_s": safe_float(row.get("ema_s")),
                    "macd_hist": safe_float(row.get("macd_hist")), "rsi": safe_float(row.get("rsi")),
                    "bb_up": safe_float(row.get("bb_up")), "bb_lo": safe_float(row.get("bb_lo")),
                    "atr": atr_val, "mom": safe_float(row.get("mom")),
                    "stoch_k": safe_float(row.get("stoch_k")), "stoch_d": safe_float(row.get("stoch_d")),
                    "adx": safe_float(row.get("adx")), "pdi": safe_float(row.get("pdi")), "mdi": safe_float(row.get("mdi")),
                    "obv": safe_float(row.get("obv")), "vwap": safe_float(row.get("vwap")), "cci": safe_float(row.get("cci"))
                },
                "Target": float(target),
                "Stop": float(stop),
                "Target Formula": f"{entry_price:.4f} {'+' if entry_side==1 else '-'} {p['target_atr_mult']} * ATR({p['atr_period']}) = {target:.4f}",
                "Stop Formula": f"{entry_price:.4f} {'-' if entry_side==1 else '+'} {p['sl_atr_mult']} * ATR({p['atr_period']}) = {stop:.4f}",
                "SideInt": entry_side
            }
            continue

        # EXIT: strictly at close (target/stop/flip/end)
        if in_pos:
            price_close = float(row[C])
            exit_reason = None
            exit_flag = False
            if entry_info["SideInt"] == 1:
                if price_close >= entry_info["Target"]:
                    exit_reason = "Target hit (close)"; exit_flag = True
                elif price_close <= entry_info["Stop"]:
                    exit_reason = "Stop hit (close)"; exit_flag = True
                elif row["Signal"] == -1 and int(row["total_votes"]) >= p["min_confluence"]:
                    exit_reason = "Opposite confluence (close)"; exit_flag = True
            else:
                if price_close <= entry_info["Target"]:
                    exit_reason = "Target hit (close)"; exit_flag = True
                elif price_close >= entry_info["Stop"]:
                    exit_reason = "Stop hit (close)"; exit_flag = True
                elif row["Signal"] == 1 and int(row["total_votes"]) >= p["min_confluence"]:
                    exit_reason = "Opposite confluence (close)"; exit_flag = True

            if (i == len(local_with_signals)-1) and in_pos and not exit_flag:
                exit_flag = True; exit_reason = "End of data (close)"

            if exit_flag:
                exit_price = price_close
                points = (exit_price - entry_info["Entry Price"]) if entry_info["SideInt"]==1 else (entry_info["Entry Price"] - exit_price)
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
                    **{f"iv_{k}": v for k, v in entry_info["IndicatorValues"].items()},
                    "Reason Exit": exit_reason
                })
                in_pos = False
                entry_info = None

    trades_df = pd.DataFrame(trades)
    total_points = trades_df["Points"].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = int((trades_df["Points"] > 0).sum()) if num_trades>0 else 0
    losses = num_trades - wins
    prob_of_profit = wins / num_trades if num_trades>0 else 0.0
    avg_win = trades_df.loc[trades_df["Points"] > 0, "Points"].mean() if wins>0 else 0.0
    avg_loss = abs(trades_df.loc[trades_df["Points"] < 0, "Points"].mean()) if losses>0 else 0.0
    expectancy_points = prob_of_profit * avg_win - (1 - prob_of_profit) * avg_loss

    strategy_pct = (total_points / abs(first_close) * 100) if abs(first_close) > 1e-12 else np.nan
    buy_hold_pct = ((last_close - first_close) / first_close * 100) if abs(first_close) > 1e-12 else np.nan

    summary = {
        "total_points": float(total_points),
        "num_trades": int(num_trades),
        "wins": int(wins),
        "losses": int(losses),
        "prob_of_profit": float(prob_of_profit),   # accuracy
        "avg_win": float(0.0 if np.isnan(avg_win) else avg_win),
        "avg_loss": float(0.0 if np.isnan(avg_loss) else avg_loss),
        "expectancy_points": float(0.0 if np.isnan(expectancy_points) else expectancy_points),
        "buy_hold_points": float(last_close - first_close),
        "buy_hold_pct": float(buy_hold_pct),
        "strategy_pct": float(strategy_pct),
        "beat_buy_hold": float(total_points) > float(last_close - first_close)
    }
    return summary, trades_df

# =========================
# Optimization (meet Expected Return % and Expected Expectancy)
# =========================
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

        # NEW knobs (optional filters)
        "use_htf_confirm": [False, True],
        "htf_multiplier": [2, 3, 4],
        "trend_adx_min": [0, 15, 20, 25]
    }

def make_candidates(grid, mode="Random Search", n_iter=200, exhaustive_cap=3000):
    keys = list(grid.keys())
    if mode == "Random Search":
        seen = set()
        out = []
        attempts = 0
        while len(out) < n_iter and attempts < n_iter * 10:
            cand = tuple((k, random.choice(grid[k])) for k in keys)
            if cand not in seen:
                seen.add(cand)
                out.append({k: v for k, v in cand})
            attempts += 1
        return out
    # Exhaustive (capped)
    values = [grid[k] for k in keys]
    total = np.prod([len(v) for v in values])
    if total > exhaustive_cap:
        st.warning(f"Full grid {total:,} > cap {exhaustive_cap:,}. Sampling {exhaustive_cap:,} combos.")
        # sample without replacement where possible
        seen = set()
        out = []
        attempts = 0
        while len(out) < exhaustive_cap and attempts < exhaustive_cap * 10:
            cand = tuple((k, random.choice(grid[k])) for k in keys)
            if cand not in seen:
                seen.add(cand)
                out.append({k: v for k, v in cand})
            attempts += 1
        return out
    prods = list(itertools.product(*values))
    return [dict(zip(keys, p)) for p in prods]

def optimize(df_prices, side="Both", mode="Random Search", n_iter=200,
             expected_return_pct=150.0, expected_expectancy=1.0):
    grid = search_space()
    candidates = make_candidates(grid, mode=mode, n_iter=n_iter)
    best_overall = {"score": -1e18, "params": None, "summary": None, "trades": None}
    best_meet_both = None
    best_meet_return = None
    best_meet_expect = None
    progress = st.progress(0)
    total = len(candidates)

    for i, cand in enumerate(candidates):
        params = default_params.copy(); params.update(cand)
        X = confluence_signals(df_prices, params, side=side)
        summary, trades = backtest_close_only(X, params)

        # Expectancy-aware scoring (robust blend)
        # weight returns, winrate, expectancy; penalize overtrading mildly
        score = (summary["strategy_pct"]
                 + 60 * summary["prob_of_profit"]
                 + 4 * summary["expectancy_points"]
                 - 0.02 * summary["num_trades"])

        cur = {"score": score, "params": params, "summary": summary, "trades": trades}

        # overall best
        if score > best_overall["score"]:
            best_overall = cur

        # target checks
        meets_return = (not np.isnan(summary["strategy_pct"])) and summary["strategy_pct"] >= expected_return_pct
        meets_expect = summary["expectancy_points"] >= expected_expectancy

        if meets_return and meets_expect:
            if (best_meet_both is None) or (score > best_meet_both["score"]):
                best_meet_both = cur
        elif meets_return:
            if (best_meet_return is None) or (score > best_meet_return["score"]):
                best_meet_return = cur
        elif meets_expect:
            if (best_meet_expect is None) or (score > best_meet_expect["score"]):
                best_meet_expect = cur

        if total>0 and (i % 5 == 0 or i==total-1):
            progress.progress(int((i+1)/total * 100))
    progress.empty()

    if best_meet_both is not None:
        return best_meet_both, "Met Return & Expectancy"
    if best_meet_return is not None:
        return best_meet_return, "Met Return only"
    if best_meet_expect is not None:
        return best_meet_expect, "Met Expectancy only"
    return best_overall, "Best overall (targets not met)"

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Strategy controls")
trade_side = st.sidebar.selectbox("Trade Side", ["Both", "Long", "Short"], index=0)
search_mode = st.sidebar.selectbox("Search Mode", ["Random Search", "Exhaustive Grid Search"], index=0)
n_iter = st.sidebar.number_input("Random Search iterations", min_value=50, max_value=8000, value=300, step=50)
expected_return_pct = st.sidebar.number_input("Expected strategy return (%)", min_value=1, max_value=10000, value=150, step=10)

# NEW: Expected Expectancy (points per trade)
expected_expectancy = st.sidebar.number_input("Expected expectancy (points per trade)", min_value=0.0, value=1.0, step=0.5, format="%.2f")

if st.sidebar.button("Run Optimize & Backtest (Close-only)"):
    st.info("Optimization started…")
    best_result, target_status = optimize(
        df, side=trade_side, mode=search_mode, n_iter=n_iter,
        expected_return_pct=expected_return_pct,
        expected_expectancy=expected_expectancy
    )
    best_params = best_result["params"]
    best_summary = best_result["summary"]
    best_trades = best_result["trades"]

    st.subheader("Selected parameters")
    st.json(best_params)

    st.subheader("Backtest summary (selected)")
    st.write({
        "Strategy total points": best_summary["total_points"],
        "Strategy percent": f"{best_summary['strategy_pct']:.2f}%",
        "Buy & Hold points": best_summary["buy_hold_points"],
        "Buy & Hold percent": f"{best_summary['buy_hold_pct']:.2f}%",
        "Number of trades": best_summary["num_trades"],
        "Wins": best_summary["wins"],
        "Losses": best_summary["losses"],
        "Win rate (prob of profit)": f"{best_summary['prob_of_profit']:.2%}",
        "Avg win (points)": round(best_summary["avg_win"], 4),
        "Avg loss (points)": round(best_summary["avg_loss"], 4),
        "Expectancy (points/trade)": round(best_summary["expectancy_points"], 4),
        "Beat buy & hold (points)": best_summary["beat_buy_hold"],
        "Target status": target_status,
        "Target: expected return (%)": expected_return_pct,
        "Target: expected expectancy (pts/trade)": expected_expectancy
    })

    st.markdown("#### Backtest trade log (detailed)")
    if best_trades is None or best_trades.empty:
        st.write("No trades generated by this parameter set.")
    else:
        tdf = best_trades.copy()
        cols_disp = [
            "Entry Date","Exit Date","Side","Entry Price","Exit Price","Target","Stop",
            "Target Formula","Stop Formula","ATR Used","Points","Hold Days","Confluences","Reason Exit"
        ] + [c for c in tdf.columns if c.startswith("iv_")]
        tdf = tdf[cols_disp].sort_values("Entry Date").reset_index(drop=True)
        st.dataframe(tdf)

    # =========================
    # Live recommendation at LAST candle close (same params & logic)
    # =========================
    st.subheader("Live recommendation (LAST candle close)")
    df_with = confluence_signals(df, best_params, side=trade_side)
    last_row = df_with.iloc[-1]
    last_close = float(last_row[C])
    sig = int(last_row["Signal"])
    conf = int(last_row["total_votes"])
    atr_used = float(last_row["atr"]) if not pd.isna(last_row["atr"]) else float(df_with["atr"].median() if not df_with["atr"].isna().all() else 1.0)

    if sig == 0:
        st.info("No confluence signal at last candle close.")
        st.write({
            "Date (last close)": str(df.index[-1]),
            "Close": last_close,
            "Long/Short votes": f"{int(last_row['long_votes'])}/{int(last_row['short_votes'])}",
            "Total confluence": conf,
            "HTF trend filter active": bool(best_params.get("use_htf_confirm", False)),
            "Trend ADX min": int(best_params.get("trend_adx_min", 0))
        })
    else:
        side_txt = "Long" if sig==1 else "Short"
        entry_level = last_close
        target = entry_level + best_params["target_atr_mult"]*atr_used if sig==1 else entry_level - best_params["target_atr_mult"]*atr_used
        stop   = entry_level - best_params["sl_atr_mult"]*atr_used if sig==1 else entry_level + best_params["sl_atr_mult"]*atr_used

        # indicator snapshot on the last candle
        ind_values = {
            "sma_f": safe_float(last_row.get("sma_f")), "sma_s": safe_float(last_row.get("sma_s")),
            "ema_f": safe_float(last_row.get("ema_f")), "ema_s": safe_float(last_row.get("ema_s")),
            "macd_hist": safe_float(last_row.get("macd_hist")), "rsi": safe_float(last_row.get("rsi")),
            "bb_up": safe_float(last_row.get("bb_up")), "bb_lo": safe_float(last_row.get("bb_lo")),
            "atr": safe_float(last_row.get("atr")), "mom": safe_float(last_row.get("mom")),
            "stoch_k": safe_float(last_row.get("stoch_k")), "stoch_d": safe_float(last_row.get("stoch_d")),
            "adx": safe_float(last_row.get("adx")), "pdi": safe_float(last_row.get("pdi")), "mdi": safe_float(last_row.get("mdi")),
            "obv": safe_float(last_row.get("obv")), "vwap": safe_float(last_row.get("vwap")), "cci": safe_float(last_row.get("cci"))
        }
        st.write({
            "Evaluated At": str(df.index[-1]),
            "Side": side_txt,
            "Entry (Close)": float(entry_level),
            "Target (formula)": f"{entry_level:.4f} {'+' if sig==1 else '-'} {best_params['target_atr_mult']} * ATR({best_params['atr_period']}) = {target:.4f}",
            "Stop (formula)": f"{entry_level:.4f} {'-' if sig==1 else '+'} {best_params['sl_atr_mult']} * ATR({best_params['atr_period']}) = {stop:.4f}",
            "ATR used (value)": float(atr_used),
            "Confluence count": conf,
            "Indicators supporting (names)": (last_row["indicators_long"] if sig==1 else last_row["indicators_short"]),
            "Indicator values (snapshot)": ind_values,
            "HTF trend filter active": bool(best_params.get("use_htf_confirm", False)),
            "Trend ADX min": int(best_params.get("trend_adx_min", 0)),
            "Estimated probability of profit (from backtest)": f"{best_summary['prob_of_profit']:.2%}",
            "Expectancy from backtest (pts/trade)": round(best_summary["expectancy_points"], 4)
        })

    st.success("Optimization & analysis complete.")
