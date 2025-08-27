import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Strip spaces from column names
    df.columns = df.columns.str.strip().str.lower()

    if "prev. close" in df.columns:
        df = df.drop('prev. close',axis=1)

    if "prev.close" in df.columns:
        df = df.drop('prev.close',axis=1)

    st.write(f'columns:: {df.columns}')

    # Identify columns
    date_col = [c for c in df.columns if "date" in c][0]
    open_col = [c for c in df.columns if "open" in c][0]
    high_col = [c for c in df.columns if "high" in c][0]
    low_col = [c for c in df.columns if "low" in c][0]
    close_col = [c for c in df.columns if "close" in c][0]
    volume_col = [c for c in df.columns if "volume" in c][0]

    # Convert date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col, ascending=True)

    # Remove commas from numeric columns
    for col in [open_col, high_col, low_col, close_col, volume_col]:
        df[col] = df[col].astype(str).str.replace(",", "")
        # pd.to_numeric(df[col], errors="coerce")  # commented as per request

    # Date filter
    start_date = df[date_col].min().date()
    default_end = datetime.now().date()
    end_date = st.date_input("Select End Date", value=default_end)

    df_filtered = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]

    st.write("### Top 5 Rows", df_filtered.head())
    st.write("### Bottom 5 Rows", df_filtered.tail())
    st.write("### Shape:", df_filtered.shape)
    st.write("### Min Close Price:", df_filtered[close_col].min())
    st.write("### Max Close Price:", df_filtered[close_col].max())
    st.write("### Min Date:", df_filtered[date_col].min())
    st.write("### Max Date:", df_filtered[date_col].max())

    # df_filtered[close_col] = df_filtered[close_col].astype(str).str.replace(",", "").astype(float)
    st.write(df_filtered[close_col])
    # Convert Close to numeric (int)
    df_filtered[close_col] = df_filtered[close_col].astype(float).fillna(0).astype(int)
   

    # Plot Close Price line chart
    st.subheader("Close Price Over Time")
    st.line_chart(df_filtered.set_index(date_col)[close_col])

    # Returns calculation
    df_filtered["returns"] = df_filtered[close_col].pct_change()

    # If data > 1 year, plot heatmap (Year vs Month)
    if (df_filtered[date_col].max() - df_filtered[date_col].min()).days > 365:
        df_filtered["year"] = df_filtered[date_col].dt.year
        df_filtered["month"] = df_filtered[date_col].dt.month

        monthly_returns = df_filtered.groupby(["year", "month"])["returns"].mean().unstack()

        st.subheader("Monthly Returns Heatmap (Year vs Month)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(monthly_returns, cmap="RdYlGn", center=0, annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Statistics of Close price
    st.subheader("Statistics of Close Price")
    st.write(df_filtered[close_col].describe())


# st.write(f'Shape of data: {df.shape}')

# st.write('Top 5 rows')
# st.dataframe(df.head(5))
# st.write('Bottom 5 rows')
# st.dataframe(df.tail(5))

# st.write(f'columns:: {df.columns}')
# st.write(f'Min of date: {min(df["date"])}')
# st.write(f'max of date: {max(df["date"])}')


# ---------------------------
# STRATEGY EXTENSION: indicators, confluence signals, backtest, optimization, live signals
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
import itertools
import random
from datetime import timedelta

# ---------- SAFETY / ASSUMPTIONS ----------
# This block assumes `df` already exists and is filtered/sorted ascending by date,
# and the variables date_col, open_col, high_col, low_col, close_col, volume_col
# contain the exact column names in df (strings).
#
# If you want to run this snippet standalone, uncomment the wrapper at the bottom
# and set `uploaded_file` etc like your earlier working app.

# ---------- 1) Ensure numeric types for price/volume ----------
for c in [open_col, high_col, low_col, close_col, volume_col]:
    # remove commas if any remain and convert to numeric (coerce errors)
    df[c] = df[c].astype(str).str.replace(",", "", regex=True)
    df[c] = pd.to_numeric(df[c], errors="coerce")

# drop rows which lost close/open/others after coercion
df = df.dropna(subset=[date_col, close_col]).reset_index(drop=True)
# ensure datetime index for convenience
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
df.index = pd.DatetimeIndex(df[date_col])

# Helper short names for readability
O = open_col; H = high_col; L = low_col; C = close_col; V = volume_col
# ------------ Indicator implementations (no external libs) ------------
def SMA(series, n):
    return series.rolling(n, min_periods=1).mean()

def EMA(series, n):
    return series.ewm(span=n, adjust=False).mean()

def RSI(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(n).mean()
    ma_down = down.rolling(n).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def MACD(series, fast=12, slow=26, signal=9):
    fast_ema = EMA(series, fast)
    slow_ema = EMA(series, slow)
    macd = fast_ema - slow_ema
    sig = EMA(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def Bollinger(series, n=20, k=2):
    mid = SMA(series, n)
    std = series.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower

def ATR(df_local, n=14):
    high = df_local[H]; low = df_local[L]; close = df_local[C]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def Momentum(series, n=10):
    return series.diff(n)

def Stochastic(df_local, k=14, d=3):
    low_min = df_local[L].rolling(k).min()
    high_max = df_local[H].rolling(k).max()
    k_line = 100 * (df_local[C] - low_min) / (high_max - low_min + 1e-10)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def ADX(df_local, n=14):
    up_move = df_local[H].diff()
    down_move = -df_local[L].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr1 = df_local[H] - df_local[L]
    tr2 = (df_local[H] - df_local[C].shift()).abs()
    tr3 = (df_local[L] - df_local[C].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(n).mean() / (atr + 1e-10))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(n).mean()
    return adx, plus_di, minus_di

def OBV(df_local):
    obv = [0]
    for i in range(1, len(df_local)):
        if df_local[C].iat[i] > df_local[C].iat[i-1]:
            obv.append(obv[-1] + df_local[V].iat[i])
        elif df_local[C].iat[i] < df_local[C].iat[i-1]:
            obv.append(obv[-1] - df_local[V].iat[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df_local.index)

def VWAP(df_local):
    tp = (df_local[H] + df_local[L] + df_local[C]) / 3
    pv = tp * df_local[V]
    cum_pv = pv.cumsum()
    cum_vol = df_local[V].cumsum().replace(0, np.nan)
    return cum_pv / cum_vol

def CCI(df_local, n=20):
    tp = (df_local[H] + df_local[L] + df_local[C]) / 3
    sma_tp = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True) + 1e-10
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

# ---------- 2) Compute a set of indicators and store in df ----------
# default parameter values (used as baseline and in optimization search)
default_params = {
    "sma_fast": 9, "sma_slow": 21,
    "ema_fast": 8, "ema_slow": 34,
    "rsi_period": 14,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "bb_period": 20, "bb_k": 2,
    "atr_period": 14,
    "mom_period": 10,
    "stoch_k": 14, "stoch_d": 3,
    "adx_period": 14,
    "cci_period": 20,
    # trade sizing / exit
    "min_confluence": 3,
    "target_atr_mult": 2.0,
    "sl_atr_mult": 1.0
}

def compute_indicators(df_local, params):
    s = df_local[C]
    df_local = df_local.copy()
    # moving averages
    df_local[f"sma_{params['sma_fast']}"] = SMA(s, params['sma_fast'])
    df_local[f"sma_{params['sma_slow']}"] = SMA(s, params['sma_slow'])
    df_local[f"ema_{params['ema_fast']}"] = EMA(s, params['ema_fast'])
    df_local[f"ema_{params['ema_slow']}"] = EMA(s, params['ema_slow'])
    # rsi
    df_local[f"rsi_{params['rsi_period']}"] = RSI(s, params['rsi_period'])
    # macd
    macd_line, macd_sig, macd_hist = MACD(s, params['macd_fast'], params['macd_slow'], params['macd_signal'])
    df_local["macd"] = macd_line; df_local["macd_sig"] = macd_sig; df_local["macd_hist"] = macd_hist
    # bollinger
    upper, mid, lower = Bollinger(s, params['bb_period'], params['bb_k'])
    df_local["bb_upper"] = upper; df_local["bb_mid"] = mid; df_local["bb_lower"] = lower
    # atr
    df_local[f"atr_{params['atr_period']}"] = ATR(df_local, params['atr_period'])
    # momentum
    df_local[f"mom_{params['mom_period']}"] = Momentum(s, params['mom_period'])
    # stochastic
    k_line, d_line = Stochastic(df_local, params['stoch_k'], params['stoch_d'])
    df_local["stoch_k"] = k_line; df_local["stoch_d"] = d_line
    # adx
    adx, pdi, mdi = ADX(df_local, params['adx_period'])
    df_local["adx"] = adx; df_local["pdi"]=pdi; df_local["mdi"]=mdi
    # obv
    df_local["obv"] = OBV(df_local)
    # vwap
    df_local["vwap"] = VWAP(df_local)
    # cci
    df_local["cci"] = CCI(df_local, params['cci_period'])
    return df_local

# ---------- 3) Signal generation using confluence voting ----------
def generate_confluence_signals(df_local, params, side="Both"):
    """
    side: "Long", "Short", "Both"
    Returns a DataFrame with columns:
      - vote_count (how many indicators signal same direction)
      - long_vote (bool), short_vote (bool)
      - details: list of indicators that voted
      - Signal: 1 (enter long), -1 (enter short), 0 (no entry)
    """
    df_calc = compute_indicators(df_local, params)
    votes = []
    sig_series = pd.Series(0, index=df_calc.index)
    details = []

    for idx in df_calc.index:
        row = df_calc.loc[idx]
        indicators_that_long = []
        indicators_that_short = []

        # 1) SMA crossover (fast > slow -> long)
        sma_fast = row[f"sma_{params['sma_fast']}"]
        sma_slow = row[f"sma_{params['sma_slow']}"]
        if not np.isnan(sma_fast) and not np.isnan(sma_slow):
            if sma_fast > sma_slow:
                indicators_that_long.append("SMA")
            elif sma_fast < sma_slow:
                indicators_that_short.append("SMA")

        # 2) EMA crossover
        ema_f = row[f"ema_{params['ema_fast']}"]
        ema_s = row[f"ema_{params['ema_slow']}"]
        if not np.isnan(ema_f) and not np.isnan(ema_s):
            if ema_f > ema_s:
                indicators_that_long.append("EMA")
            elif ema_f < ema_s:
                indicators_that_short.append("EMA")

        # 3) MACD histogram positive -> long
        if not np.isnan(row["macd_hist"]):
            if row["macd_hist"] > 0:
                indicators_that_long.append("MACD")
            elif row["macd_hist"] < 0:
                indicators_that_short.append("MACD")

        # 4) RSI strength: <30 long (oversold bounce), >70 short (overbought)
        rsi_val = row[f"rsi_{params['rsi_period']}"]
        if not np.isnan(rsi_val):
            if rsi_val < 35:
                indicators_that_long.append("RSI")
            elif rsi_val > 65:
                indicators_that_short.append("RSI")

        # 5) Bollinger: price crossing upper/lower band
        if not np.isnan(row["bb_upper"]) and not np.isnan(row["bb_lower"]):
            price = row[C]
            if price < row["bb_lower"]:
                indicators_that_long.append("BB_Lower")
            elif price > row["bb_upper"]:
                indicators_that_short.append("BB_Upper")

        # 6) Momentum positive -> long
        mom = row[f"mom_{params['mom_period']}"]
        if not np.isnan(mom):
            if mom > 0:
                indicators_that_long.append("MOM")
            elif mom < 0:
                indicators_that_short.append("MOM")

        # 7) Stochastic K/D
        if not np.isnan(row["stoch_k"]) and not np.isnan(row["stoch_d"]):
            if row["stoch_k"] > row["stoch_d"] and row["stoch_k"] < 30:
                indicators_that_long.append("STOCH")
            elif row["stoch_k"] < row["stoch_d"] and row["stoch_k"] > 70:
                indicators_that_short.append("STOCH")

        # 8) ADX strength — direction via PDI > MDI
        if not np.isnan(row["adx"]) and not np.isnan(row["pdi"]) and not np.isnan(row["mdi"]):
            if row["adx"] > 20 and row["pdi"] > row["mdi"]:
                indicators_that_long.append("ADX+")
            elif row["adx"] > 20 and row["mdi"] > row["pdi"]:
                indicators_that_short.append("ADX-")

        # 9) OBV rising -> long
        # compare last 3-obv average vs previous 3
        i = df_calc.index.get_indexer([idx])[0]
        obv_vote_long = False; obv_vote_short = False
        if i >= 3:
            recent = df_calc["obv"].iloc[i-2:i+1].mean()
            prev = df_calc["obv"].iloc[i-5:i-2].mean() if i>=5 else df_calc["obv"].iloc[max(0,i-5):i-2].mean()
            if recent > prev:
                obv_vote_long = True
            elif recent < prev:
                obv_vote_short = True
            if obv_vote_long: indicators_that_long.append("OBV")
            if obv_vote_short: indicators_that_short.append("OBV")

        # 10) VWAP: price above VWAP -> long
        if not np.isnan(row["vwap"]):
            if row[C] > row["vwap"]:
                indicators_that_long.append("VWAP")
            elif row[C] < row["vwap"]:
                indicators_that_short.append("VWAP")

        # 11) CCI extremes
        if not np.isnan(row["cci"]):
            if row["cci"] < -100:
                indicators_that_long.append("CCI")
            elif row["cci"] > 100:
                indicators_that_short.append("CCI")

        # count confluences
        long_votes = len(set(indicators_that_long))
        short_votes = len(set(indicators_that_short))
        total_votes = max(long_votes, short_votes)

        vote_direction = 1 if long_votes > short_votes else (-1 if short_votes > long_votes else 0)
        # apply side filter
        if side == "Long" and vote_direction == -1:
            vote_direction = 0
            total_votes = 0
        if side == "Short" and vote_direction == 1:
            vote_direction = 0
            total_votes = 0

        # produce final signal only if votes >= threshold
        final_sig = vote_direction if total_votes >= params['min_confluence'] else 0

        votes.append({
            "index": idx,
            "long_votes": long_votes,
            "short_votes": short_votes,
            "total_votes": total_votes,
            "direction": vote_direction,
            "signal": final_sig,
            "indicators_long": indicators_that_long,
            "indicators_short": indicators_that_short
        })
        sig_series.loc[idx] = final_sig

    votes_df = pd.DataFrame(votes).set_index("index")
    result = df_calc.join(votes_df)
    result["Signal"] = sig_series
    return result

# ---------- 4) Backtester ----------
def backtest_point_strategy(df_signals, params):
    """
    Simple daily backtester working with signals column.
    Entry executed at next day's Open if available, otherwise at same day's Close.
    Exit occurs when Close hits Target or SL; if neither, exit at last price (mark-to-market).
    Returns: (total_points, trades_df)
    """
    trades = []
    in_pos = False
    pos_side = 0
    entry_price = None
    entry_date = None
    entry_details = None
    target = None
    sl = None

    # buy & hold baseline (points)
    first_price = df_signals[C].iloc[0]
    last_price = df_signals[C].iloc[-1]
    buy_hold_points = last_price - first_price  # positive if ended higher

    # iterate rows in order
    for i in range(len(df_signals)-1):
        row = df_signals.iloc[i]
        next_row = df_signals.iloc[i+1]
        sig = row["Signal"]

        # If not in position and there's a signal -> enter next day's open
        if (not in_pos) and sig != 0:
            entry_date = next_row.name  # next day's timestamp (index)
            # entry price prefer Open of next day, fallback to Close next day
            entry_price = next_row[O] if not pd.isna(next_row[O]) else next_row[C]
            pos_side = sig  # 1 long, -1 short
            # set sl/target based on ATR
            atr_val = row.get(f"atr_{params['atr_period']}", np.nan)
            if np.isnan(atr_val) or atr_val == 0:
                atr_val = df_signals[f"atr_{params['atr_period']}"].median() or 1.0
            if pos_side == 1:
                target = entry_price + params['target_atr_mult'] * atr_val
                sl = entry_price - params['sl_atr_mult'] * atr_val
            else:
                target = entry_price - params['target_atr_mult'] * atr_val
                sl = entry_price + params['sl_atr_mult'] * atr_val

            entry_details = {
                "Entry Date": entry_date, "Entry Price": entry_price,
                "Side": "Long" if pos_side==1 else "Short",
                "Indicators": (row["indicators_long"] if pos_side==1 else row["indicators_short"]),
                "Confluences": row["total_votes"]
            }
            in_pos = True
            continue

        # If in position, check exit conditions for next_row (we check next_row price range)
        if in_pos:
            # check if next_row intraday min/max hit target/sl - we only have daily OHLC so do heuristic:
            # For long: if next_row[H] >= target -> target hit that day; elif next_row[L] <= sl -> stopped
            h = next_row[H]; l = next_row[L]; closep = next_row[C]
            exit_price = None; exit_date = None; reason = None

            if pos_side == 1:
                if not pd.isna(h) and h >= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(l) and l <= sl:
                    exit_price = sl
                    reason = "Stopped"
                # if neither, optionally close if opposite signal appears with sufficient confluence
                elif next_row["Signal"] == -1 and next_row["total_votes"] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"
            else:  # short
                if not pd.isna(l) and l <= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(h) and h >= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif next_row["Signal"] == 1 and next_row["total_votes"] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"

            # final day fallback: if last day and still in_pos, exit at close
            if (i+1) == (len(df_signals)-1) and in_pos and exit_price is None:
                exit_price = closep
                reason = "End of data"

            if exit_price is not None:
                exit_date = next_row.name
                points = (exit_price - entry_price) if pos_side == 1 else (entry_price - exit_price)
                trades.append({
                    **entry_details,
                    "Exit Date": exit_date,
                    "Exit Price": exit_price,
                    "Reason Exit": reason,
                    "Points": points,
                    "Hold Days": (exit_date.date() - entry_details["Entry Date"].date()).days
                })
                # reset
                in_pos = False
                pos_side = 0
                entry_price = None
                entry_details = None
                target = None
                sl = None

    # Aggregate results
    trades_df = pd.DataFrame(trades)
    total_points = trades_df["Points"].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = (trades_df["Points"] > 0).sum() if not trades_df.empty else 0
    prob_of_profit = (wins / num_trades) if num_trades>0 else 0.0

    # percent_return relative to buy-hold points (guard divide by zero)
    buy_hold_points = last_price - first_price if 'first_price' in locals() else (df_signals[C].iloc[-1] - df_signals[C].iloc[0])
    percent_vs_buyhold = (total_points / (abs(buy_hold_points)+1e-9)) * 100 if abs(buy_hold_points) > 0 else np.nan

    summary = {
        "total_points": total_points,
        "num_trades": num_trades,
        "wins": wins,
        "prob_of_profit": prob_of_profit,
        "buy_hold_points": buy_hold_points,
        "pct_vs_buyhold": percent_vs_buyhold
    }
    return summary, trades_df

# ---------- 5) Optimization routine ----------
def search_space_generator():
    # ranges used for searching - exhaustive grid can be large; random search samples combinations
    grid = {
        "sma_fast": list(range(5, 16)),            # 5..15
        "sma_slow": list(range(18, 51, 2)),        # 18..50
        "ema_fast": list(range(5, 16)),
        "ema_slow": list(range(20, 61, 5)),
        "rsi_period": list(range(8, 21)),
        "macd_fast": [8, 10, 12, 16],
        "macd_slow": [20, 26, 34],
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
    return grid

def generate_candidates(grid, mode="random", n_iter=200, max_exhaustive=2000):
    keys = list(grid.keys())
    if mode == "random":
        candidates = []
        for _ in range(n_iter):
            cand = {k: random.choice(grid[k]) for k in keys}
            candidates.append(cand)
        return candidates
    else:
        # exhaustive cartesian product (but cap sample size if too big)
        all_values = [grid[k] for k in keys]
        total = np.prod([len(v) for v in all_values])
        # if total is huge, sample combinations to a manageable number
        if total > max_exhaustive:
            st.warning(f"Full exhaustive combinations = {total:,} > {max_exhaustive:,}. Sampling {max_exhaustive} combinations uniformly at random.")
            candidates = []
            for _ in range(max_exhaustive):
                cand = {k: random.choice(grid[k]) for k in keys}
                candidates.append(cand)
            return candidates
        else:
            products = list(itertools.product(*all_values))
            candidates = [dict(zip(keys, p)) for p in products]
            return candidates

def optimize(df_local, side="Both", mode="random", n_iter=200):
    grid = search_space_generator()
    candidates = generate_candidates(grid, mode=("random" if mode=="Random Search" else "exhaustive"), n_iter=n_iter, max_exhaustive=2000)

    best = {"score": -1e9, "params": None, "summary": None, "trades": None}
    progress_bar = st.progress(0)
    total = len(candidates)
    for i, params in enumerate(candidates):
        # merge default params with candidate
        working_params = default_params.copy()
        working_params.update(params)
        # compute signals
        df_signals = generate_confluence_signals(df_local, working_params, side=side)
        # backtest
        summary, trades_df = backtest_point_strategy(df_signals, working_params)
        # scoring: primary = total_points; secondary = prob_of_profit; tertiary = num_trades small penalty
        score = summary["total_points"] + (summary["prob_of_profit"]*10) - (summary["num_trades"]*0.01)
        # require beating buy&hold OR >=70% improvement relative criterion
        beat_buyhold = False
        # compute percentage improvement vs buy_hold in absolute terms if defined
        try:
            buy_hold = summary["buy_hold_points"]
            if abs(buy_hold) > 1e-9:
                # percent improvement
                improvement_pct = summary["total_points"] / (abs(buy_hold)+1e-9) * 100
                beat_buyhold = improvement_pct > 0 or summary["total_points"] > abs(buy_hold)
            else:
                improvement_pct = np.nan
        except:
            improvement_pct = np.nan

        # keep candidate if better by score
        if score > best["score"]:
            best.update({
                "score": score,
                "params": working_params,
                "summary": summary,
                "trades": trades_df,
                "improvement_pct": improvement_pct
            })
        # progress
        if total>0:
            progress_bar.progress(int((i+1)/total * 100))
    progress_bar.empty()
    return best

# ---------- 6) Streamlit interface to run optimization & show results ----------
st.sidebar.markdown("## Strategy & Optimization")
side_choice = st.sidebar.selectbox("Trade Side", ["Both", "Long", "Short"], index=0)
opt_mode = st.sidebar.selectbox("Search Mode", ["Random Search", "Exhaustive Grid Search"], index=0)
n_iter = st.sidebar.number_input("Random Search iterations", min_value=50, max_value=2000, value=200, step=50)

if st.sidebar.button("Run backtest & optimize"):
    st.info("Optimization started — this may take time depending on iterations and grid size. Progress shown below.")
    best_result = optimize(df, side=side_choice, mode=opt_mode, n_iter=n_iter)
    st.success("Optimization completed.")

    if best_result["params"] is None:
        st.warning("No strategy found.")
    else:
        st.subheader("Best parameters found")
        st.json(best_result["params"])
        st.subheader("Backtest summary (best)")
        st.write(best_result["summary"])
        st.write(f"Improvement vs buy & hold (pct approx): {best_result.get('improvement_pct'):.2f}")

        st.subheader("Trades (detailed log)")
        trades_df = best_result["trades"]
        if trades_df is None or trades_df.empty:
            st.write("No trades were generated by this strategy.")
        else:
            # present a clean table
            st.dataframe(trades_df.sort_values("Entry Date").reset_index(drop=True))

        # Display metrics / interpretation
        st.markdown("### Interpretation / Data-backed reason summaries")
        # show top 5 trade rows with reason strings
        if trades_df is not None and not trades_df.empty:
            def reason_from_row(r):
                inds = r["Indicators"] if isinstance(r.get("Indicators"), (list, tuple)) else []
                return f"Confluences: {r.get('Confluences', 0)} — Indicators: {', '.join(map(str, inds))}"
            trades_df["Reason Summary"] = trades_df.apply(reason_from_row, axis=1)
            st.dataframe(trades_df[["Entry Date","Exit Date","Side","Entry Price","Exit Price","Points","Hold Days","Confluences","Reason Summary"]].head(50))

        # Live recommendation based on last row
        st.subheader("Live Recommendation (next trading day)")
        last_idx = df.index[-1]
        next_day = last_idx + timedelta(days=1)
        # compute indicators on whole df with best params
        df_with_signals = generate_confluence_signals(df, best_result["params"], side=side_choice)
        last_row = df_with_signals.iloc[-1]
        sig = last_row["Signal"]
        long_votes = last_row["long_votes"] if "long_votes" in last_row else 0
        short_votes = last_row["short_votes"] if "short_votes" in last_row else 0
        total_votes = last_row["total_votes"] if "total_votes" in last_row else 0
        if sig == 0:
            st.info("No immediate signal by best-strategy confluence at close of last candle.")
            st.write({
                "Next Day": str(next_day.date()),
                "Last Close": float(df[C].iloc[-1]),
                "Confluence (long_votes/short_votes/total)": f"{long_votes}/{short_votes}/{total_votes}"
            })
        else:
            entry_side = "Long" if sig==1 else "Short"
            entry_price = df[O].iloc[-1] if not pd.isna(df[O].iloc[-1]) else df[C].iloc[-1]
            atrv = df_with_signals[f"atr_{best_result['params']['atr_period']}"].iloc[-1]
            if np.isnan(atrv) or atrv==0: atrv = df_with_signals[f"atr_{best_result['params']['atr_period']}"].median()
            if sig == 1:
                tgt = entry_price + best_result["params"]["target_atr_mult"] * atrv
                stop = entry_price - best_result["params"]["sl_atr_mult"] * atrv
            else:
                tgt = entry_price - best_result["params"]["target_atr_mult"] * atrv
                stop = entry_price + best_result["params"]["sl_atr_mult"] * atrv

            # approximate probability of profit from backtest
            prob = best_result["summary"]["prob_of_profit"] if best_result["summary"] else 0.0
            st.write({
                "Entry Date": str(next_day.date()),
                "Side": entry_side,
                "Entry Level (approx)": float(entry_price),
                "Target (approx)": float(tgt),
                "Stop Loss (approx)": float(stop),
                "Estimated Probability of Profit": f"{prob:.2%}",
                "Confluence Count": int(total_votes),
                "Indicators supporting this": (last_row["indicators_long"] if sig==1 else last_row["indicators_short"])
            })

    # show which parameters produced greatest returns
    st.markdown("---")
    st.subheader("Which parameters gave greatest returns?")
    if best_result.get("params"):
        st.json(best_result["params"])

# End of strategy extension
