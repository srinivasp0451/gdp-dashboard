# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
import pytz
import random
from itertools import product
from math import isnan

st.set_page_config(page_title="Swing Trading Recommender", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Utilities: column mapping, timezone, basic indicators
# ---------------------------
def map_columns(df):
    # Lowercase columns for matching
    cols = {c: c.lower() for c in df.columns}
    df_col_lower = {c.lower(): c for c in df.columns}
    mapping = {}
    # heuristics
    def find_candidate(key_words):
        for k, orig in df_col_lower.items():
            for kw in key_words:
                if kw in k:
                    return orig
        return None

    mapping['date'] = find_candidate(['date', 'time', 'timestamp', 'dt'])
    mapping['open'] = find_candidate(['open', 'o', 'openprice', 'open_price'])
    mapping['high'] = find_candidate(['high', 'h', 'highprice', 'high_price'])
    mapping['low']  = find_candidate(['low', 'l', 'lowprice', 'low_price'])
    mapping['close'] = find_candidate(['close', 'c', 'closeprice', 'close_price', 'last'])
    mapping['volume'] = find_candidate(['volume', 'vol', 'quantity', 'qty', 'shares', 'turnover'])

    # If any missing, try substring fuzzy approach
    for k in ['date','open','high','low','close','volume']:
        if mapping.get(k) is None:
            for orig in df.columns:
                if k in orig.lower():
                    mapping[k] = orig
                    break

    # as last resort try first 6 columns by position
    cols_list = list(df.columns)
    fallback = {'date':0, 'open':1, 'high':2, 'low':3, 'close':4}
    for k,v in fallback.items():
        if mapping.get(k) is None and len(cols_list) > v:
            mapping[k] = cols_list[v]

    # return mapping and mapped df
    mapped = pd.DataFrame()
    for k in ['date','open','high','low','close','volume']:
        if mapping.get(k) is not None:
            mapped[k] = df[mapping[k]]
        else:
            mapped[k] = np.nan
    return mapped, mapping

def convert_to_datetime_index(df, date_col='date', ist=True):
    # Try parse date, handle tz-aware and naive
    s = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    # if all NaT, try other parsing
    if s.isna().all():
        s = pd.to_datetime(df[date_col].astype(str), errors='coerce')
    # if tz-aware, convert to IST
    try:
        if s.dt.tz is None:
            # naive -> assume it's in local timezone or UTC? we'll treat naive as UTC then convert to IST
            s = s.dt.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')
        s = s.dt.tz_convert('Asia/Kolkata')
    except Exception:
        # If still fails, attempt to localize to Asia/Kolkata
        try:
            s = s.dt.tz_localize('Asia/Kolkata')
        except Exception:
            # fallback: leave naive
            s = pd.to_datetime(df[date_col], errors='coerce')
    df = df.copy()
    df['date_ist'] = s
    df = df.set_index('date_ist')
    return df

def clean_numeric_columns(df):
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def add_basic_indicators(df):
    # Simple moving average and returns for features
    df['ret'] = df['close'].pct_change().fillna(0)
    df['rsi14'] = compute_rsi(df['close'], 14)
    df['sma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma50'] = df['close'].rolling(50, min_periods=1).mean()
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=1).mean()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# ---------------------------
# Pattern detectors (heuristic)
# ---------------------------
def detect_double_top_bottom(df, lookback=30, tolerance=0.02):
    signals = []
    # Search windows for double tops/bottoms
    for i in range(lookback, len(df)-1):
        window = df['close'].iloc[i-lookback:i]
        peak = window.max()
        trough = window.min()
        cur = df['close'].iloc[i]
        # double top: two peaks similar and drop
        # heuristic: if recent two peaks within tolerance and subsequent drop > x
        peaks = window[window > (peak*(1-tolerance))]
        if len(peaks) >= 2 and cur < peak*(1 - 0.01):
            signals.append(('double_top', df.index[i]))
        bottoms = window[window < (trough*(1+tolerance))]
        if len(bottoms) >= 2 and cur > trough*(1 + 0.01):
            signals.append(('double_bottom', df.index[i]))
    return signals

def detect_head_shoulders(df, lookback=60):
    signals = []
    # Very heuristic: look for left shoulder, head (higher), right shoulder (approx equal to left)
    for i in range(lookback, len(df)-1):
        window = df['close'].iloc[i-lookback:i]
        if len(window) < 10:
            continue
        peaks = window[window==window.max()]
        if peaks.empty:
            continue
        head_idx = peaks.index[0]
        # approximate shoulders
        left = window.loc[:head_idx]
        right = window.loc[head_idx:]
        if len(left) < 3 or len(right) < 3:
            continue
        leftpeak = left.max()
        rightpeak = right.max()
        head = window.max()
        # check shoulders are lower than head and similar to each other
        if leftpeak < head and rightpeak < head and abs(leftpeak - rightpeak)/max(leftpeak, rightpeak+1e-9) < 0.08:
            signals.append(('head_shoulder', df.index[i]))
    return signals

def detect_triangle(df, lookback=50, thresh=0.02):
    signals = []
    for i in range(lookback, len(df)-1):
        window = df['close'].iloc[i-lookback:i]
        x = np.arange(len(window))
        # fit lines to highs and lows
        highs = window.rolling(5).max().dropna()
        lows = window.rolling(5).min().dropna()
        if len(highs) < 10 or len(lows) < 10:
            continue
        # linear fit
        try:
            m_high, b_high = np.polyfit(x[-len(highs):], highs.values, 1)
            m_low, b_low = np.polyfit(x[-len(lows):], lows.values, 1)
            # if slopes converge (one negative, one positive or different signs) -> triangle
            if m_high * m_low < 0 or (abs(m_high) < 0.001 and abs(m_low) > 0.001):
                signals.append(('triangle_or_wedge', df.index[i]))
        except Exception:
            continue
    return signals

def detect_cup_handle(df, lookback=200):
    signals = []
    # Very rough detect: a rounded bottom followed by small pullback
    for i in range(lookback, len(df)-1):
        window = df['close'].iloc[i-lookback:i]
        left = window[:len(window)//2]
        right = window[len(window)//2:]
        if left.mean() > left.min() and right.mean() > right.min():
            # curvature check
            if (left.max() > left.median()) and (right.max() > right.median()):
                # detect small handle formation at end: slight dip
                if window.iloc[-1] < window.iloc[-5] * 1.03:
                    signals.append(('cup_handle', df.index[i]))
    return signals

# You can add more detectors similarly...
def aggregate_pattern_signals(df):
    sigs = []
    sigs += detect_double_top_bottom(df)
    sigs += detect_head_shoulders(df)
    sigs += detect_triangle(df)
    sigs += detect_cup_handle(df)
    # Deduplicate by index
    result = {}
    for name, idx in sigs:
        result.setdefault(idx, []).append(name)
    # convert to list of tuples
    agg = [(idx, list(set(names))) for idx, names in result.items()]
    return agg

# ---------------------------
# Price action helpers
# ---------------------------
def detect_support_resistance(df, n=10):
    # Simple pivot-based support/resistance
    supports = []
    resistances = []
    for i in range(n, len(df)-n):
        window = df['low'].iloc[i-n:i+n+1]
        if df['low'].iloc[i] == window.min():
            supports.append((df.index[i], df['low'].iloc[i]))
        wh = df['high'].iloc[i]
        window_h = df['high'].iloc[i-n:i+n+1]
        if wh == window_h.max():
            resistances.append((df.index[i], wh))
    return supports, resistances

def detect_volume_spikes(df, mult=2):
    spikes = df['volume'] > df['vol_ma20'] * mult
    return df[spikes]

def is_fake_breakout(df, idx, lookback=10):
    # heuristic: breakout above resistance but volume not supportive and quickly reverses
    i = df.index.get_loc(idx)
    if i+1 >= len(df): return False
    # compare close of next 3 candles
    window = df['close'].iloc[max(0, i-lookback):i+4]
    # fake if after breakout, price re-enters previous range within 3 candles
    if window.max() - window.min() == 0:
        return False
    recent = df['close'].iloc[i+1:i+4]
    if recent.empty:
        return False
    # if price falls back > 0.5% immediately -> fake
    if (recent < df['close'].iloc[i] * 0.995).any():
        return True
    return False

# ---------------------------
# Backtester
# ---------------------------
def generate_signals(df, params):
    """
    params: dict of parameters used to generate signals.
    We'll use pattern detections, trend confluence and volume to create long/short signals.
    """
    signals = []  # each: dict with idx, signal_type, reason, prob
    pattern_list = aggregate_pattern_signals(df)
    pattern_dict = {idx: pats for idx, pats in pattern_list}
    supports, resistances = detect_support_resistance(df, n=params.get('sr_lookback',10))
    vol_spikes = detect_volume_spikes(df, mult=params.get('vol_mult',2))
    # trend
    df['trend'] = np.where(df['sma20'] > df['sma50'], 1, -1)

    for i in range(len(df)-1):
        dt = df.index[i]
        price = df['close'].iloc[i]
        reason_parts = []
        prob = 0.5
        side = None
        # pattern vote
        if dt in pattern_dict:
            pats = pattern_dict[dt]
            if any('double_top' in p or 'head_shoulder' in p for p in pats):
                side = 'short'
                reason_parts.append("Reversal pattern detected: " + ",".join(pats))
                prob += 0.15
            elif any('double_bottom' in p or 'cup_handle' in p or 'triangle' in p or 'w' in ",".join(pats) for p in pats):
                side = 'long'
                reason_parts.append("Continuation/reversal pattern: " + ",".join(pats))
                prob += 0.12
            else:
                # triangle/wedge ambiguous -> trend decides
                if df['trend'].iloc[i] > 0:
                    side = 'long'
                    prob += 0.07
                    reason_parts.append("Triangle/Wedge in uptrend")
                else:
                    side = 'short'
                    prob += 0.07
                    reason_parts.append("Triangle/Wedge in downtrend")

        # Volume confluence
        if dt in vol_spikes.index:
            prob += 0.08
            reason_parts.append("Volume spike")

        # candlestick psychology: long if bullish engulfing or hammer
        if is_bullish_bearish_candle(df, i):
            typ = is_bullish_bearish_candle(df, i)
            if typ == 'bull':
                side = side or 'long'
                prob += 0.05
                reason_parts.append("Bullish candle pattern")
            elif typ == 'bear':
                side = side or 'short'
                prob += 0.05
                reason_parts.append("Bearish candle pattern")

        # trend confluence
        if df['trend'].iloc[i] == 1:
            prob += 0.03
            reason_parts.append("Trend SMA20>50")
        else:
            prob -= 0.01

        # Final threshold
        if prob >= params.get('min_prob', 0.55):
            signals.append({
                'index': dt,
                'pos': i,
                'side': side if side else 'long',
                'price': price,
                'reason': "; ".join(reason_parts) if reason_parts else "Pattern/price action",
                'prob': min(prob, 0.95)
            })
    return signals

def is_bullish_bearish_candle(df, i):
    # Simple candlestick psychology detection
    o = df['open'].iloc[i]
    c = df['close'].iloc[i]
    h = df['high'].iloc[i]
    l = df['low'].iloc[i]
    body = abs(c-o)
    total = h-l + 1e-9
    # hammer
    if (body/total) < 0.3 and (c - l) / total > 0.5 and c > o:
        return 'bull'
    # bearish hammer (inverse)
    if (body/total) < 0.3 and (h - c) / total > 0.5 and c < o:
        return 'bear'
    # engulfing
    if i > 0:
        prev_o = df['open'].iloc[i-1]
        prev_c = df['close'].iloc[i-1]
        if c > o and prev_c < prev_o and (c - o) > (prev_o - prev_c):
            return 'bull'
        if c < o and prev_c > prev_o and (o - c) > (prev_c - prev_o):
            return 'bear'
    return None

def backtest_with_params(df, params, mode='both', end_date=None):
    """
    Strict no-future leakage: iterate row by row; when signal occurs at index i,
    entry = close of i; exit as per TP/SL or hold days/close rules.
    """
    df = df.copy()
    n = len(df)
    trades = []
    signals = generate_signals(df, params)
    taken_until = -1
    for sig in signals:
        i = sig['pos']
        if i <= taken_until:
            continue
        if end_date is not None and df.index[i] > end_date:
            continue
        side = sig['side']
        if mode == 'long' and side != 'long':
            continue
        if mode == 'short' and side != 'short':
            continue
        entry_price = df['close'].iloc[i]  # entry on close of signal candle
        entry_time = df.index[i]
        # SL and TP logic: use ATR-like width or percent
        atr = df['high'].rolling(14).max().iloc[i] - df['low'].rolling(14).min().iloc[i]
        sl_dist = max(params.get('sl_pct', 0.02) * entry_price, params.get('sl_atr_mult', 0.5) * (atr if not np.isnan(atr) else entry_price*0.02))
        tp_dist = max(params.get('tp_ratio', 2.0) * sl_dist, params.get('min_points', 0.01)*entry_price)
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        # Walk forward to find exit (no future leak): check next candles
        exit_price = None
        exit_time = None
        exit_reason = None
        # max hold days in candles
        max_hold = params.get('max_hold', 20)
        for j in range(i+1, min(n, i+1+max_hold)):
            high = df['high'].iloc[j]
            low = df['low'].iloc[j]
            close = df['close'].iloc[j]
            t = df.index[j]
            # check TP/SL
            if side == 'long':
                if low <= sl:
                    exit_price = sl
                    exit_time = t
                    exit_reason = 'SL'
                    break
                if high >= tp:
                    exit_price = tp
                    exit_time = t
                    exit_reason = 'TP'
                    break
            else:
                if high >= sl:
                    exit_price = sl
                    exit_time = t
                    exit_reason = 'SL'
                    break
                if low <= tp:
                    exit_price = tp
                    exit_time = t
                    exit_reason = 'TP'
                    break
        if exit_price is None:
            # exit at close of last available or at end of backtest horizon
            exit_price = df['close'].iloc[min(i+max_hold, n-1)]
            exit_time = df.index[min(i+max_hold, n-1)]
            exit_reason = 'TIMEOUT'

        pnl = (exit_price - entry_price) if side == 'long' else (entry_price - exit_price)
        points = pnl
        trades.append({
            'entry_time': entry_time, 'exit_time': exit_time,
            'side': side, 'entry': entry_price, 'exit': exit_price,
            'tp': tp, 'sl': sl, 'reason': sig['reason'],
            'exit_reason': exit_reason, 'prob': sig['prob'],
            'pnl': pnl, 'points': points, 'duration': (exit_time - entry_time)
        })
        # avoid overlapping trades: set taken_until to j
        taken_until = df.index.get_loc(exit_time) if exit_time in df.index else i+1
    # compute metrics
    df_tr = pd.DataFrame(trades)
    if df_tr.empty:
        metrics = {
            'net_pnl': 0, 'total_trades': 0, 'win_rate':0, 'avg_points':0,
            'positive_trades':0, 'negative_trades':0
        }
    else:
        net = df_tr['pnl'].sum()
        total = len(df_tr)
        pos = (df_tr['pnl'] > 0).sum()
        neg = (df_tr['pnl'] <= 0).sum()
        win_rate = pos/total if total>0 else 0
        avg = df_tr['pnl'].mean()
        metrics = {'net_pnl': net, 'total_trades': total, 'win_rate': win_rate,
                   'avg_points': avg, 'positive_trades': pos, 'negative_trades': neg}
    return trades, metrics

# ---------------------------
# Optimization engines: random search & grid search
# ---------------------------
def random_search(df, param_space, n_iter, mode, end_date, target_accuracy=0.6, min_points=0.0, progress_bar=None):
    best = None
    best_score = -1e9
    results = []
    for it in range(n_iter):
        params = {}
        for k, v in param_space.items():
            if isinstance(v, list):
                params[k] = random.choice(v)
            elif isinstance(v, tuple) and len(v) == 2:
                params[k] = random.uniform(v[0], v[1])
            else:
                params[k] = v
        trades, metrics = backtest_with_params(df, params, mode=mode, end_date=end_date)
        # scoring: prefer higher net_pnl but penalize low win rate below target_accuracy
        score = metrics.get('net_pnl',0) + 1000 * (metrics.get('win_rate',0) - target_accuracy)
        # penalize too few points
        if metrics.get('avg_points',0) < min_points:
            score -= abs(min_points - metrics.get('avg_points',0)) * 100
        results.append((params, metrics, trades, score))
        if score > best_score:
            best_score = score
            best = (params, metrics, trades)
        if progress_bar:
            progress_bar.progress(int((it+1)/n_iter*100))
    return best, results

def grid_search(df, param_grid, mode, end_date, target_accuracy=0.6, min_points=0.0, progress_bar=None):
    keys = list(param_grid.keys())
    combos = list(product(*(param_grid[k] for k in keys)))
    best = None
    best_score = -1e9
    results = []
    for idx, combo in enumerate(combos):
        params = {k: combo[i] for i,k in enumerate(keys)}
        trades, metrics = backtest_with_params(df, params, mode=mode, end_date=end_date)
        score = metrics.get('net_pnl',0) + 1000 * (metrics.get('win_rate',0) - target_accuracy)
        if metrics.get('avg_points',0) < min_points:
            score -= abs(min_points - metrics.get('avg_points',0)) * 100
        results.append((params, metrics, trades, score))
        if score > best_score:
            best_score = score
            best = (params, metrics, trades)
        if progress_bar:
            progress_bar.progress(int((idx+1)/len(combos)*100))
    return best, results

# ---------------------------
# Presentation / Summaries
# ---------------------------
def human_summary(df):
    # 100-word approx summary of the stock data
    period_days = (df.index.max() - df.index.min()).days if not df.empty else 0
    mean_return = df['ret'].mean() * 252 if 'ret' in df else 0
    vol = df['ret'].std() * np.sqrt(252) if 'ret' in df else 0
    up = (df['ret']>0).mean() if 'ret' in df else 0
    s = f"This dataset spans {period_days} days from {df.index.min()} to {df.index.max()}. The annualized mean return is approximately {mean_return:.2%} with volatility around {vol:.2%}. Positive-return days account for {up:.1%} of the sample. Price structure shows SMA20 vs SMA50 trend: currently {'bullish' if df['sma20'].iloc[-1] > df['sma50'].iloc[-1] else 'bearish'}. Volume patterns and price action will be used to detect supply/demand, breakouts and traps. The system will search for tradeable chart patterns and optimize entry/SL/targets to aim for higher returns than buy-and-hold while controlling risk."
    return s

def trade_reason_text(trade):
    # Compose human readable reason
    txt = f"{trade['side'].upper()} entry at {trade['entry']:.2f} on {trade['entry_time']}.\n"
    txt += f"Exit at {trade['exit']:.2f} on {trade['exit_time']} due to {trade['exit_reason']}. "
    txt += f"TP {trade['tp']:.2f}, SL {trade['sl']:.2f}. P&L: {trade['pnl']:.2f}.\n"
    txt += f"Logic: {trade['reason']}. Estimated probability: {trade['prob']:.2f}."
    return txt

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Swing Trading Recommender (Automated Backtest + Live Recommendation)")
st.sidebar.header("Upload & Settings")

uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV or Excel", type=['csv','xlsx','xls'])
mode = st.sidebar.selectbox("Direction", ['both','long','short'])
search_method = st.sidebar.selectbox("Optimization Method", ['random_search','grid_search'])
n_iter = st.sidebar.number_input("Random search iterations", min_value=5, max_value=2000, value=50, step=5)
grid_iters = st.sidebar.number_input("Grid search max combos (for safety)", min_value=1, max_value=1000, value=200)
desired_accuracy = st.sidebar.slider("Desired minimum win rate (0-1)", 0.0, 1.0, 0.65)
min_points = st.sidebar.number_input("Minimum average points per trade (fraction of price, e.g., 0.01)", value=0.005, step=0.001, format="%.4f")
progress_bar = st.sidebar.empty()

if uploaded_file is None:
    st.info("Upload your historical OHLCV file (CSV or Excel). Column names can be anything — the app will map common names.")
    st.stop()

# read the file
try:
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# Map columns
mapped_df, mapping = map_columns(df_raw)
mapped_df = clean_numeric_columns(mapped_df)
# copy original date column to maintain for timezone parsing
mapped_df['date_orig'] = df_raw[mapping.get('date')] if mapping.get('date') in df_raw.columns else df_raw.iloc[:,0]
# convert date and set IST index
try:
    df_timeindexed = convert_to_datetime_index(mapped_df, date_col='date_orig')
except Exception as e:
    st.warning("Date parsing had issues - attempting simple parse.")
    mapped_df['date_orig'] = pd.to_datetime(mapped_df['date_orig'], errors='coerce')
    df_timeindexed = mapped_df.set_index('date_orig')

df_timeindexed = df_timeindexed.sort_index(ascending=True)
df_timeindexed = clean_numeric_columns(df_timeindexed)
df_timeindexed = add_basic_indicators(df_timeindexed)

# top/bottom rows, min/max dates and prices
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Data Preview")
    st.write("Detected mapping:", mapping)
    st.write("Top 5 rows")
    st.write(df_timeindexed.head(5))
    st.write("Bottom 5 rows")
    st.write(df_timeindexed.tail(5))
with col2:
    st.subheader("Summary stats")
    st.write("Date range:", df_timeindexed.index.min(), "to", df_timeindexed.index.max())
    st.write("Price min/max:", df_timeindexed['close'].min(), df_timeindexed['close'].max())

# End date selection for backtest horizon
default_end = df_timeindexed.index.max()
end_date = st.sidebar.date_input("Select simulation end date (inclusive)", value=default_end.date(), min_value=df_timeindexed.index.min().date(), max_value=df_timeindexed.index.max().date())
# convert chosen date to timestamp at end of day IST
end_dt = pd.to_datetime(str(end_date)).tz_localize('Asia/Kolkata') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# EDA: raw plot and heatmap of returns Year vs Month
st.subheader("Raw Candlestick Chart (last 300 candles)")
last_n = st.slider("Candles to show", min_value=50, max_value=min(2000, len(df_timeindexed)), value=min(300, len(df_timeindexed)))
plot_df = df_timeindexed.iloc[-last_n:].reset_index()
fig = go.Figure(data=[go.Candlestick(x=plot_df['date_ist'],
                open=plot_df['open'], high=plot_df['high'],
                low=plot_df['low'], close=plot_df['close'])])
fig.update_layout(height=450, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Returns Heatmap (Year vs Month)")
df_timeindexed['year'] = df_timeindexed.index.year
df_timeindexed['month'] = df_timeindexed.index.month
monthly = df_timeindexed.groupby(['year','month'])['ret'].sum().unstack(level=0).fillna(0).T
fig2, ax2 = plt.subplots(figsize=(10,3))
sns.heatmap(monthly, annot=True, fmt=".2f", cmap='RdYlGn', ax=ax2)
st.pyplot(fig2)

st.subheader("100-word Summary of Data")
st.write(human_summary(df_timeindexed))

# Prepare parameter space for optimization (user could later expand or we can auto tune)
st.sidebar.header("Optimization Parameter Ranges")
# defaults
param_space = {
    'sl_pct': (0.005, 0.05),           # percent SL of price when random; for grid we'll sample set
    'sl_atr_mult': (0.2, 1.5),
    'tp_ratio': (1.0, 4.0),
    'min_prob': (0.52, 0.7),
    'vol_mult': [1.5, 2.0, 2.5, 3.0],
    'sr_lookback': [5,10,20],
    'max_hold': [5, 10, 20, 40],
    'min_points': (min_points, min_points+0.02)
}

st.info("Default parameter ranges are set. The optimizer will sample these ranges. Grid search will enumerate small grids; random search samples.")

# Create progress bar
global_progress = st.progress(0)

# Run optimization
st.subheader("Optimizer & Backtest")
run_opt = st.button("Run Optimization & Backtest (this may take time)")
best_result = None

if run_opt:
    progress_bar = st.sidebar.progress(0)
    if search_method == 'random_search':
        # random search
        best, results = random_search(df_timeindexed, param_space, n_iter=n_iter, mode=mode, end_date=end_dt, target_accuracy=desired_accuracy, min_points=min_points, progress_bar=progress_bar)
    else:
        # build grid (coerce continuous ranges into small discrete for safety)
        param_grid = {}
        for k,v in param_space.items():
            if isinstance(v, tuple) and len(v)==2:
                # sample 3 points between
                param_grid[k] = np.linspace(v[0], v[1], num=3).tolist()
            elif isinstance(v, list):
                param_grid[k] = v
            else:
                param_grid[k] = [v]
        # safety: limit combos
        combos = 1
        for k in param_grid:
            combos *= len(param_grid[k])
        if combos > grid_iters:
            st.warning(f"Grid combos ({combos}) exceed your limit ({grid_iters}). Reduce grid size or increase limit.")
            progress_bar.empty()
            st.stop()
        best, results = grid_search(df_timeindexed, param_grid, mode=mode, end_date=end_dt, target_accuracy=desired_accuracy, min_points=min_points, progress_bar=progress_bar)

    progress_bar.empty()
    if best is None:
        st.warning("No strategy found with given constraints.")
    else:
        best_params, best_metrics, best_trades = best
        st.success("Found best strategy!")
        st.write("Best Parameters:")
        st.json(best_params)
        st.write("Best Metrics:")
        st.json(best_metrics)
        st.subheader("Backtest Trade Log (most recent 200)")
        if isinstance(best_trades, list) and len(best_trades)>0:
            df_tr = pd.DataFrame(best_trades)
            # show details for each trade including reasons
            st.dataframe(df_tr[['entry_time','exit_time','side','entry','exit','tp','sl','pnl','exit_reason','reason','prob']].sort_values('entry_time').tail(200))
            # show human readable reasons for last 20 trades
            st.subheader("Trade Explanations (last 20)")
            for idx, row in df_tr.tail(20).iterrows():
                st.markdown(f"**Trade {idx+1}** — {row['side'].upper()} | Entry: {row['entry']:.2f} | Exit: {row['exit']:.2f} | PnL: {row['pnl']:.2f}")
                st.write(trade_reason_text(row))
        else:
            st.write("No trades executed with the best parameters.")

        # Aggregate metrics
        st.subheader("Aggregate Backtest Metrics")
        st.metric("Net P&L (points)", value=f"{best_metrics.get('net_pnl',0):.2f}")
        st.metric("Total Trades", value=best_metrics.get('total_trades',0))
        st.metric("Win Rate", value=f"{best_metrics.get('win_rate',0):.2%}")
        st.metric("Avg Points/Trade", value=f"{best_metrics.get('avg_points',0):.4f}")

        st.subheader("Backtest Written Summary")
        summary_lines = []
        summary_lines.append(f"Backtest ran until {end_dt}. Net PnL: {best_metrics.get('net_pnl',0):.2f} points across {best_metrics.get('total_trades',0)} trades with a win rate of {best_metrics.get('win_rate',0):.2%}.")
        summary_lines.append(f"Strategy used parameters: {best_params}. Average points per trade: {best_metrics.get('avg_points',0):.4f}.")
        summary_lines.append("Recommendations for live: follow same rules. Enter on close of signal candle. Use SL and TP as optimized. Monitor volume spikes & pattern confluence for higher probability.")
        st.write(" ".join(summary_lines))

        # Live recommendation using best params and last candle close
        st.subheader("Live Recommendation (based on last available candle close)")
        live_params = best_params
        live_signals = generate_signals(df_timeindexed.tail(200), live_params)
        # we want signals that occur at the last candle index
        if len(live_signals) == 0:
            st.info("No live signal in recent candles using best strategy.")
        else:
            # find the latest signal at last candle index
            last_idx = df_timeindexed.index[-1]
            candidate = None
            # prefer signal exactly on last candle (entry on its close)
            for s in live_signals[::-1]:
                if s['index'] == last_idx:
                    candidate = s
                    break
            if candidate is None:
                # pick the latest signal before last candle but not after end_dt
                candidate = live_signals[-1]
            # Compose live trade structure using same exit rules as backtest but applied forward
            fake_trades, _ = backtest_with_params(df_timeindexed.tail(500), live_params, mode=mode, end_date=None)
            # We want the entry that matches candidate
            # Build live recommendation from candidate heuristics:
            entry_price = df_timeindexed['close'].iloc[-1]
            # Compute SL/TP like backtester
            atr = df_timeindexed['high'].rolling(14).max().iloc[-1] - df_timeindexed['low'].rolling(14).min().iloc[-1]
            sl_dist = max(live_params.get('sl_pct', 0.02) * entry_price, live_params.get('sl_atr_mult', 0.5) * (atr if not np.isnan(atr) else entry_price*0.02))
            tp_dist = max(live_params.get('tp_ratio', 2.0) * sl_dist, live_params.get('min_points', 0.01)*entry_price)
            side = candidate['side']
            if side == 'long':
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist
            prob = candidate.get('prob', 0.55)
            st.markdown(f"**{side.upper()}** entry at **{entry_price:.2f}** (on close of last candle {last_idx}).")
            st.markdown(f"- Target (TP): **{tp:.2f}**")
            st.markdown(f"- Stop Loss (SL): **{sl:.2f}**")
            st.markdown(f"- Estimated Probability of Profit: **{prob:.2%}** (heuristic from pattern/volume/trend confluence)")
            st.markdown(f"- Reason/Logic: {candidate.get('reason')}")
            st.markdown("**Note**: Enter on close of the last candle. The system follows strict no-future leakage rules (entry price is candle close).")
            # Add psychology summary
            st.subheader("Candlestick & Market Psychology (Live)")
            psych = []
            if df_timeindexed['sma20'].iloc[-1] > df_timeindexed['sma50'].iloc[-1]:
                psych.append("Short-term trend is bullish (SMA20 > SMA50). Buyers are in control on moving averages.")
            else:
                psych.append("Short-term trend is bearish (SMA20 < SMA50). Sellers are dominant on moving averages.")
            if df_timeindexed['volume'].iloc[-1] > df_timeindexed['vol_ma20'].iloc[-1]*1.5:
                psych.append("Recent volume is higher than 20-period average — higher participation (confirming move).")
            if is_bullish_bearish_candle(df_timeindexed, len(df_timeindexed)-1) == 'bull':
                psych.append("Last candle shows bullish psychology (hammer/engulfing).")
            elif is_bullish_bearish_candle(df_timeindexed, len(df_timeindexed)-1) == 'bear':
                psych.append("Last candle shows bearish psychology (inversion/engulfing).")
            st.write("\n".join(psych))

st.sidebar.markdown("---")
st.sidebar.write("Developed: Automated swing trading framework. Entries happen on close candle; optimization supports random/grid search. Extend detectors as needed.")
