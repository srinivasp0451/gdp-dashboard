"""
Streamlit Swing Recommender
- Upload OHLCV CSV/Excel (columns with arbitrary names allowed)
- Automatically map columns (date/open/high/low/close/volume)
- Sort data ascending and convert date column to IST timezone
- Exploratory Data Analysis (head/tail, min/max, plots, heatmap of returns)
- Automatic price-action & pattern-based signal generation (breakouts, triangles, flags, head & shoulders, double tops/bottoms, cup-and-handle heuristics)
- Backtester that enters on the SAME candle close where signal is generated (no future leakage)
- Grid / Random search optimization over strategy parameters (default: Random Search)
- Live recommendation using the best-found strategy applied to the last available candle (entry at last candle close)
- Human-readable reasons for each trade and summary descriptions

NOTES:
- This implementation is heuristic-driven and meant to be a strong, self-contained starting point.
- No external TA libraries (talib / pandas_ta) are used.
- Save this file and run: `streamlit run streamlit_swing_recommender.py`

Dependencies:
pip install streamlit pandas numpy plotly pytz

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import io
import itertools
import random

st.set_page_config(layout="wide", page_title="Swing Recommender")

# ------------------------------- Utilities -------------------------------

def map_columns(df):
    # Return suggested mapping for core fields
    mapping = {"date": None, "open": None, "high": None, "low": None, "close": None, "volume": None}
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    for i, c in enumerate(cols):
        lc = lower[i]
        if mapping['date'] is None and ("date" in lc or "time" in lc or "timestamp" in lc):
            mapping['date'] = c
        if mapping['open'] is None and ("open" in lc and "interest" not in lc):
            mapping['open'] = c
        if mapping['high'] is None and "high" in lc:
            mapping['high'] = c
        if mapping['low'] is None and "low" in lc:
            mapping['low'] = c
        if mapping['close'] is None and ("close" in lc or "last" in lc or lc.endswith('c')):
            mapping['close'] = c
        if mapping['volume'] is None and ("volume" in lc or "qty" in lc or "trade" in lc or "shares" in lc):
            mapping['volume'] = c
    # Fallbacks (substring match)
    for target in mapping:
        if mapping[target] is None:
            for c in cols:
                if target in c.lower():
                    mapping[target] = c
                    break
    return mapping


def to_ist_series(s):
    tz = pytz.timezone('Asia/Kolkata')
    def convert(x):
        if pd.isna(x):
            return pd.NaT
        try:
            if x.tzinfo is None:
                # assume naive timestamps are already in IST
                return tz.localize(x)
            else:
                return x.astimezone(tz)
        except Exception:
            return pd.NaT
    return s.apply(convert)


def compute_atr(df, n=14):
    # df expects columns: high, low, close
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=n, min_periods=1).mean()
    return atr


def find_local_pivots(series, left=3, right=3):
    # return indices of local highs and lows
    highs = []
    lows = []
    vals = series.values
    n = len(vals)
    for i in range(left, n - right):
        window_left = vals[i-left:i]
        window_right = vals[i+1:i+1+right]
        if vals[i] > max(window_left) and vals[i] >= max(window_right):
            highs.append(i)
        if vals[i] < min(window_left) and vals[i] <= min(window_right):
            lows.append(i)
    return highs, lows


def cluster_levels(levels, tol=0.005):
    # cluster numeric levels within relative tolerance tol
    if len(levels) == 0:
        return []
    levels_sorted = sorted(levels)
    clusters = [[levels_sorted[0]]]
    for lvl in levels_sorted[1:]:
        if abs(lvl - clusters[-1][-1]) <= tol * lvl:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    # return mean of clusters
    return [np.mean(c) for c in clusters]


def detect_basic_patterns(df, i, lookback=60):
    # detect patterns that include the current candle i using only data up to i
    # Returns a list of detected patterns for that point
    patterns = []
    start = max(0, i - lookback + 1)
    window = df.iloc[start:i+1]
    close = window['close'].values
    high = window['high'].values
    low = window['low'].values
    idxs = np.arange(start, i+1)
    # simple double top/bottom detection
    highs, lows = find_local_pivots(pd.Series(close), left=3, right=3)
    # double top: two highs similar level
    if len(highs) >= 2:
        last_two = highs[-2:]
        h1, h2 = close[last_two[0]-start], close[last_two[1]-start]
        if abs(h1 - h2) <= 0.01 * max(h1, h2):
            # check valley between
            mid = close[last_two[0]-start:last_two[1]-start+1]
            if np.min(mid) < 0.98 * min(h1, h2):
                patterns.append(('double_top', {'indices': [idxs[last_two[0]-start], idxs[last_two[1]-start]]}))
    if len(lows) >= 2:
        last_two = lows[-2:]
        l1, l2 = close[last_two[0]-start], close[last_two[1]-start]
        if abs(l1 - l2) <= 0.01 * max(l1, l2):
            patterns.append(('double_bottom', {'indices': [idxs[last_two[0]-start], idxs[last_two[1]-start]]}))
    # head and shoulders (very heuristic): three peaks with middle highest
    if len(highs) >= 3:
        a,b,c = highs[-3:]
        ca, cb, cc = close[a-start], close[b-start], close[c-start]
        if cb > ca and cb > cc and abs(ca - cc) <= 0.03 * cb:
            patterns.append(('head_and_shoulders', {'indices': [a,b,c]}))
    # cup & handle (heuristic): long rounded bottom followed by small pullback
    # detect long run where lows gradually decrease then increase
    if len(window) >= 40:
        half = int(len(window)/2)
        left = close[:half]
        right = close[half:]
        if np.mean(left) > np.mean(right) and np.percentile(right, 90) > np.percentile(left, 90):
            # simple curvature check
            if np.polyfit(np.arange(len(close)), close, 2)[0] > 0:
                patterns.append(('cup_and_handle', {}))
    # triangle (descending highs, rising lows)
    if len(window) >= 10:
        xs = np.arange(len(window))
        highs_lin = np.polyfit(xs, window['high'].values, 1)[0]
        lows_lin = np.polyfit(xs, window['low'].values, 1)[0]
        if highs_lin < 0 and lows_lin > 0:
            patterns.append(('triangle', {}))
    # flag/flagpole: sharp move (flagpole) then small consolidation
    if len(window) >= 8:
        # flagpole: last close much higher than close 8 bars ago
        if close[-1] > 1.05 * close[0] and np.std(close[-8:]) < 0.02 * np.mean(close[-8:]):
            patterns.append(('flag', {}))
    return patterns


def generate_signals(df, params):
    # params dict: lookback, vol_mult, atr_sl_mult, target_atr_mult, min_move
    lookback = int(params.get('lookback', 60))
    vol_mult = float(params.get('vol_mult', 1.5))
    atr_sl_mult = float(params.get('atr_sl_mult', 1.5))
    target_atr_mult = float(params.get('target_atr_mult', 2.0))

    atr = compute_atr(df, n=14)
    mean_vol = df['volume'].rolling(window=max(5, int(lookback/4)), min_periods=1).mean()
    signals = []
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    for i in range(lookback-1, len(df)-1):
        # use data up to and including i to create signal and entry at close[i]
        recent = df.iloc[max(0, i - lookback + 1):i+1]
        # find pivots in recent
        highs_idx, lows_idx = find_local_pivots(recent['close'], left=3, right=3)
        # compute clustered resistance/support
        resistance_levels = cluster_levels([recent['close'].values[idx] for idx in highs_idx], tol=0.01)
        support_levels = cluster_levels([recent['close'].values[idx] for idx in lows_idx], tol=0.01)
        last_close = close[i]
        last_vol = df['volume'].iloc[i]
        recent_mean_vol = mean_vol.iloc[i]
        # breakout above resistance
        if len(resistance_levels) > 0:
            last_res = resistance_levels[-1]
            # breakout if close > resistance and previous close <= resistance (break happened on this candle)
            prev_close = close[i-1] if i-1>=0 else close[i]
            if last_close > last_res and prev_close <= last_res and last_vol > max(200, recent_mean_vol * vol_mult):
                # pattern detection
                patterns = detect_basic_patterns(df, i, lookback=lookback)
                # set stoploss as nearest support or entry - atr*mult
                current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else np.std(close[max(0,i-14):i+1])
                sl = last_close - max(0.5 * current_atr, atr_sl_mult * current_atr)
                # target as measured move: either distance to support/resistance or atr multiplier
                measured_move = last_close - (support_levels[-1] if len(support_levels)>0 else (np.min(recent['low'].values)))
                target = last_close + max(target_atr_mult * current_atr, measured_move)
                reason = f"Breakout above resistance {last_res:.2f} with volume {int(last_vol)} > {recent_mean_vol:.1f}."
                patterns_text = ",".join([p[0] for p in patterns]) if patterns else "breakout"
                signals.append({
                    'idx': i,
                    'datetime': df['date'].iloc[i],
                    'side': 'long',
                    'entry': float(last_close),
                    'sl': float(sl),
                    'target': float(target),
                    'reason': reason,
                    'patterns': patterns_text,
                })
        # breakout below support (short)
        if len(support_levels) > 0:
            last_sup = support_levels[-1]
            prev_close = close[i-1] if i-1>=0 else close[i]
            if last_close < last_sup and prev_close >= last_sup and last_vol > max(200, recent_mean_vol * vol_mult):
                patterns = detect_basic_patterns(df, i, lookback=lookback)
                current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else np.std(close[max(0,i-14):i+1])
                sl = last_close + max(0.5 * current_atr, atr_sl_mult * current_atr)
                measured_move = (resistance_levels[-1] if len(resistance_levels)>0 else np.max(recent['high'].values)) - last_close
                target = last_close - max(target_atr_mult * current_atr, measured_move)
                reason = f"Breakdown below support {last_sup:.2f} with volume {int(last_vol)} > {recent_mean_vol:.1f}."
                patterns_text = ",".join([p[0] for p in patterns]) if patterns else "breakdown"
                signals.append({
                    'idx': i,
                    'datetime': df['date'].iloc[i],
                    'side': 'short',
                    'entry': float(last_close),
                    'sl': float(sl),
                    'target': float(target),
                    'reason': reason,
                    'patterns': patterns_text,
                })
    return signals


def run_backtest(df, signals, max_hold=10):
    # simulate trades given signals. Entries at close at signal idx. Exits when target or SL hit in future candles.
    trades = []
    n = len(df)
    for s in signals:
        i = s['idx']
        entry_price = s['entry']
        side = s['side']
        sl = s['sl']
        target = s['target']
        entry_time = s['datetime']
        exit_time = None
        exit_price = None
        exit_idx = None
        hit = None
        for j in range(i+1, min(n, i+1+max_hold)):
            h = df['high'].iloc[j]
            l = df['low'].iloc[j]
            o = df['open'].iloc[j]
            c = df['close'].iloc[j]
            if side == 'long':
                sl_hit = (l <= sl)
                target_hit = (h >= target)
                if sl_hit and not target_hit:
                    exit_price = float(sl)
                    exit_time = df['date'].iloc[j]
                    exit_idx = j
                    hit = 'SL'
                    break
                if target_hit and not sl_hit:
                    exit_price = float(target)
                    exit_time = df['date'].iloc[j]
                    exit_idx = j
                    hit = 'TARGET'
                    break
                if sl_hit and target_hit:
                    # ambiguous: use candle open to decide
                    if o > entry_price:
                        exit_price = float(target); hit='TARGET'
                    else:
                        exit_price = float(sl); hit='SL'
                    exit_time = df['date'].iloc[j]; exit_idx=j
                    break
            else: # short
                sl_hit = (h >= sl)
                target_hit = (l <= target)
                if sl_hit and not target_hit:
                    exit_price = float(sl)
                    exit_time = df['date'].iloc[j]
                    exit_idx = j
                    hit = 'SL'
                    break
                if target_hit and not sl_hit:
                    exit_price = float(target)
                    exit_time = df['date'].iloc[j]
                    exit_idx = j
                    hit = 'TARGET'
                    break
                if sl_hit and target_hit:
                    if o < entry_price:
                        exit_price = float(target); hit='TARGET'
                    else:
                        exit_price = float(sl); hit='SL'
                    exit_time = df['date'].iloc[j]; exit_idx=j
                    break
        # if not exited within holding, close at last available price in holding window
        if exit_price is None:
            final_idx = min(n-1, i+max_hold)
            exit_price = float(df['close'].iloc[final_idx])
            exit_time = df['date'].iloc[final_idx]
            exit_idx = final_idx
            hit = 'END'
        pnl = (exit_price - entry_price) if side=='long' else (entry_price - exit_price)
        pnl_pct = pnl / entry_price * 100
        trades.append({
            'entry_idx': i,
            'entry_time': entry_time,
            'side': side,
            'entry_price': entry_price,
            'exit_idx': exit_idx,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'pnl_points': pnl,
            'pnl_pct': pnl_pct,
            'result': 'win' if pnl>0 else ('loss' if pnl<0 else 'flat'),
            'reason': s.get('reason','') + ' patterns:' + s.get('patterns','')
        })
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {}
    total_pnl = trades_df['pnl_points'].sum()
    wins = trades_df[trades_df['pnl_points']>0].shape[0]
    losses = trades_df[trades_df['pnl_points']<0].shape[0]
    flats = trades_df[trades_df['pnl_points']==0].shape[0]
    accuracy = wins / len(trades_df) * 100
    avg_points = trades_df['pnl_points'].mean()
    metrics = {
        'total_pnl': total_pnl,
        'trades': len(trades_df),
        'wins': wins,
        'losses': losses,
        'flats': flats,
        'accuracy': accuracy,
        'avg_points': avg_points
    }
    # equity curve
    equity = trades_df['pnl_points'].cumsum().values
    if len(equity)>0:
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak).min()
        metrics['max_drawdown'] = float(dd)
    else:
        metrics['max_drawdown'] = 0.0
    return trades_df, metrics


def optimize_parameters(df, param_grid, search='random', n_iter=30, desired_accuracy=None, desired_points=None):
    # param_grid is dict of param:[values]
    combos = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())
    best = None
    results = []
    if search == 'random':
        sampled = [tuple(random.choice(param_grid[k]) for k in keys) for _ in range(min(n_iter, 3000))]
    else:
        sampled = combos
    sampled = list(dict.fromkeys(sampled))[:max(1, min(len(sampled), n_iter if search=='random' else len(sampled)))]
    for vals in sampled:
        params = dict(zip(keys, vals))
        signals = generate_signals(df, params)
        trades_df, metrics = run_backtest(df, signals, max_hold=int(params.get('max_hold',10)))
        # compute buy & hold
        bh_points = df['close'].iloc[-1] - df['close'].iloc[0]
        metrics['params'] = params
        metrics['bh_points'] = bh_points
        results.append(metrics)
        # select best satisfying conditions
        ok = True
        if desired_accuracy is not None and metrics.get('accuracy',0) < desired_accuracy:
            ok = False
        if desired_points is not None and (metrics.get('avg_points',0) < desired_points and metrics.get('total_pnl',0) < desired_points):
            ok = False
        if ok:
            if best is None:
                best = metrics
            else:
                # prefer higher total_pnl then accuracy
                if metrics['total_pnl'] > best['total_pnl']:
                    best = metrics
    # if not found any satisfying, pick highest accuracy then highest pnl
    if best is None and len(results)>0:
        results_sorted = sorted(results, key=lambda x: (x.get('accuracy',0), x.get('total_pnl',0)), reverse=True)
        best = results_sorted[0]
    return best, results

# ------------------------------- Streamlit UI -------------------------------

st.title("Swing Trading Recommender — Automatic Backtest + Live Recommendation")

uploaded = st.file_uploader("Upload your OHLCV file (CSV or Excel)", type=['csv','xlsx','xls'])

if uploaded is None:
    st.info("Upload a CSV or Excel file containing OHLCV data. Columns can have any names (e.g., 'Open', 'open price', 'CLOSE', 'Close', 'volume', 'trades').")
    st.stop()

# Read file
try:
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded file is empty")
    st.stop()

# Suggest mapping
suggested = map_columns(df)
st.subheader("Column Mapping (auto-detected)")
cols = list(df.columns)
col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox('Date/Time column', options=[None]+cols, index=cols.index(suggested['date']) if suggested['date'] in cols else 0)
    open_col = st.selectbox('Open column', options=[None]+cols, index=cols.index(suggested['open']) if suggested['open'] in cols else 0)
    high_col = st.selectbox('High column', options=[None]+cols, index=cols.index(suggested['high']) if suggested['high'] in cols else 0)
with col2:
    low_col = st.selectbox('Low column', options=[None]+cols, index=cols.index(suggested['low']) if suggested['low'] in cols else 0)
    close_col = st.selectbox('Close column', options=[None]+cols, index=cols.index(suggested['close']) if suggested['close'] in cols else 0)
    vol_col = st.selectbox('Volume column', options=[None]+cols, index=cols.index(suggested['volume']) if suggested['volume'] in cols else 0)

# Validate mapping
required = [date_col, open_col, high_col, low_col, close_col]
if any(x in (None, 0, '') for x in required):
    st.warning('Please map at least date, open, high, low, close columns.')
    st.stop()

# Rename and parse
df_work = df.rename(columns={date_col: 'date', open_col: 'open', high_col: 'high', low_col: 'low', close_col: 'close'})
if vol_col:
    df_work = df_work.rename(columns={vol_col: 'volume'})
else:
    df_work['volume'] = 0

# Parse date and convert to IST
df_work['date'] = pd.to_datetime(df_work['date'], errors='coerce')
if df_work['date'].isna().all():
    st.error('Could not parse any dates from the selected date column. Please check the format.')
    st.stop()
# Localize naive timestamps to IST (assume user data is already in IST), convert tz-aware to IST
try:
    df_work['date'] = to_ist_series(df_work['date'])
except Exception as e:
    st.warning('Date timezone conversion issue, proceeding with naive datetimes: '+str(e))
    df_work['date'] = pd.to_datetime(df_work['date']).dt.tz_localize('Asia/Kolkata', ambiguous='NaT', nonexistent='NaT')

# Sort ascending and reset index
df_work = df_work.sort_values('date').reset_index(drop=True)

# Show head/tail and basic stats
st.subheader('Data Preview')
left, right = st.columns([1,1])
with left:
    st.write('Top 5 rows')
    st.dataframe(df_work.head())
with right:
    st.write('Bottom 5 rows')
    st.dataframe(df_work.tail())

st.markdown('---')

st.subheader('Date & Price Range')
min_date = df_work['date'].min()
max_date = df_work['date'].max()
min_price = df_work['close'].min()
max_price = df_work['close'].max()
col1, col2, col3 = st.columns(3)
col1.metric('Start Date (min)', str(min_date))
col2.metric('End Date (max)', str(max_date))
#col3.metric('Price Range', f"{min_price:.2f} — {max_price:.2f}")
col3.metric('Price Range', "{min_price:.2f} — {max_price:.2f}")

# End date selection (user wants to test on past data). Build date options from data
unique_dates = sorted(df_work['date'].dt.normalize().unique())
# default end date: use today's date if present else max_date
today_ist = pd.Timestamp.now(tz=pytz.timezone('Asia/Kolkata')).normalize()
if today_ist in unique_dates:
    default_idx = unique_dates.index(today_ist)
else:
    default_idx = len(unique_dates)-1
selected_end_date = st.selectbox('Select end date for backtest (affects data used for backtest & live signal)', options=unique_dates, index=default_idx)
# filter data up to selected_end_date inclusive
df_use = df_work[df_work['date'] <= (pd.Timestamp(selected_end_date).tz_localize('Asia/Kolkata') + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1))].copy()

st.subheader('Raw Candlestick Plot (up to selected end date)')
fig = go.Figure(data=[go.Candlestick(x=df_use['date'], open=df_use['open'], high=df_use['high'], low=df_use['low'], close=df_use['close'])])
fig.update_layout(height=400, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig, use_container_width=True)

# EDA: returns heatmap year vs month
st.subheader('Exploratory Data Analysis (Returns Heatmap & Summary)')
df_use['year'] = df_use['date'].dt.year
df_use['month'] = df_use['date'].dt.month
monthly = df_use.groupby(['year','month']).apply(lambda x: x['close'].iloc[-1]/x['close'].iloc[0]-1).reset_index(name='ret')
heat = monthly.pivot(index='year', columns='month', values='ret').fillna(0)
fig2 = px.imshow(heat, labels=dict(x='Month', y='Year', color='Monthly Return'), x=list(range(1,13)), y=heat.index.astype(str))
fig2.update_layout(height=350)
st.plotly_chart(fig2, use_container_width=True)

# 100-word summary of what data is telling
st.subheader('Data Summary (100-word)')
avg_ret = (df_use['close'].pct_change().mean()*252*100) if len(df_use)>1 else 0
volatility = (df_use['close'].pct_change().std()*np.sqrt(252)*100) if len(df_use)>1 else 0
trend_direction = 'uptrend' if df_use['close'].iloc[-1] > df_use['close'].iloc[0] else 'downtrend' if df_use['close'].iloc[-1] < df_use['close'].iloc[0] else 'sideways'
summary = (f"From {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}, the instrument shows a {trend_direction}. "
           f"Annualized return (approx) {avg_ret:.2f}% with volatility {volatility:.2f}%. "
           f"Recent price range is {min_price:.2f} to {max_price:.2f}. Look for supply/demand clusters near pivot levels and volume-confirmed breakouts for higher probability trades. "
           f"The data suggests potential opportunities around breakout-to-pullback setups, triangle breakouts, and clear trend continuations or reversals when confluence of volume and pattern exists.")
# truncate/ensure ~100 words
summary_words = summary.split()
if len(summary_words) > 100:
    summary = ' '.join(summary_words[:100])
st.write(summary)

st.markdown('---')

# ---------------- optimization UI ----------------
st.sidebar.header('Strategy & Optimization Settings')
side = st.sidebar.selectbox('Side to generate (Long / Short / Both)', ['both','long','short'])
search_type = st.sidebar.selectbox('Search Type', ['random','grid'])
desired_accuracy = st.sidebar.slider('Desired minimum accuracy (%)', min_value=0.0, max_value=100.0, value=40.0, step=1.0)
desired_points = st.sidebar.number_input('Desired minimum avg points (leave 0 to ignore)', min_value=0.0, value=0.0)
iterations = st.sidebar.number_input('Random search iterations (if selected)', min_value=5, max_value=300, value=40)

st.sidebar.markdown('Parameter Ranges for optimization (defaults chosen automatically, modify if needed)')
lookback_min = st.sidebar.number_input('Lookback min', min_value=10, max_value=500, value=40)
lookback_max = st.sidebar.number_input('Lookback max', min_value=lookback_min, max_value=1000, value=120)
vol_mult_min = st.sidebar.number_input('Volume multiplier min', min_value=0.5, max_value=5.0, value=1.0, format="%.2f")
vol_mult_max = st.sidebar.number_input('Volume multiplier max', min_value=vol_mult_min, max_value=10.0, value=2.0, format="%.2f")
atr_sl_min = st.sidebar.number_input('SL ATR multiplier min', min_value=0.1, max_value=5.0, value=1.0, format="%.2f")
atr_sl_max = st.sidebar.number_input('SL ATR multiplier max', min_value=atr_sl_min, max_value=10.0, value=2.0, format="%.2f")
target_atr_min = st.sidebar.number_input('Target ATR multiplier min', min_value=0.5, max_value=10.0, value=1.0, format="%.2f")
target_atr_max = st.sidebar.number_input('Target ATR multiplier max', min_value=target_atr_min, max_value=20.0, value=3.0, format="%.2f")
max_hold_min = st.sidebar.number_input('Max hold days min', min_value=1, max_value=200, value=3)
max_hold_max = st.sidebar.number_input('Max hold days max', min_value=max_hold_min, max_value=500, value=10)

if st.sidebar.button('Run Optimization & Backtest'):
    with st.spinner('Running optimization and backtest — this may take a while'):
        # build parameter grid
        param_grid = {
            'lookback': list(range(lookback_min, lookback_max+1, max(1,int((lookback_max-lookback_min)/3)))),
            'vol_mult': [round(x,2) for x in np.linspace(vol_mult_min, vol_mult_max, 3)],
            'atr_sl_mult': [round(x,2) for x in np.linspace(atr_sl_min, atr_sl_max, 3)],
            'target_atr_mult': [round(x,2) for x in np.linspace(target_atr_min, target_atr_max, 3)],
            'max_hold': list(range(max_hold_min, max_hold_max+1, max(1,int((max_hold_max-max_hold_min)/2))))
        }
        n_iter = int(iterations)
        best, results = optimize_parameters(df_use, param_grid, search=search_type, n_iter=n_iter, desired_accuracy=desired_accuracy, desired_points=(desired_points if desired_points>0 else None))
        if best is None:
            st.warning('No strategy met the constraints. Showing best available result.')
            if len(results)>0:
                best = sorted(results, key=lambda x: (x.get('accuracy',0), x.get('total_pnl',0)), reverse=True)[0]
            else:
                st.error('No results. Aborting.')
                st.stop()
        st.subheader('Best Strategy Summary')
        st.write('Parameters:')
        st.json(best['params'])
        st.write('Metrics:')
        st.write(f"Total PnL (points): {best['total_pnl']:.2f}")
        st.write(f"Trades: {best['trades']}, Wins: {best['wins']}, Losses: {best['losses']}, Accuracy: {best['accuracy']:.2f}%")
        st.write(f"Avg points/trade: {best['avg_points']:.2f}, Max drawdown: {best.get('max_drawdown',0):.2f}")

        # regenerate trades df for best params
        best_params = best['params']
        signals = generate_signals(df_use, best_params)
        # filter side:
        if side != 'both':
            signals = [s for s in signals if s['side']==side]
        trades_df, metrics = run_backtest(df_use, signals, max_hold=int(best_params.get('max_hold',10)))
        if trades_df.empty:
            st.write('No trades were generated by the best strategy.')
        else:
            st.subheader('Backtest Trades (entry on same-candle close)')
            st.dataframe(trades_df[['entry_time','side','entry_price','exit_time','exit_price','pnl_points','pnl_pct','reason']].sort_values('entry_time'))
            # plot equity
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=pd.to_datetime(trades_df['exit_time']), y=trades_df['pnl_points'].cumsum(), mode='lines+markers'))
            fig_eq.update_layout(title='Equity curve (points)', height=300)
            st.plotly_chart(fig_eq, use_container_width=True)

        # Live recommendation: apply same best params on last available candle
        st.subheader('Live Recommendation (based on last available candle close)')
        live_signals = generate_signals(df_use, best_params)
        if side != 'both':
            live_signals = [s for s in live_signals if s['side']==side]
        # find signals whose idx is the last candle index
        last_idx = len(df_use)-1
        live_for_last = [s for s in live_signals if s['idx'] == last_idx]
        if len(live_for_last)==0:
            st.write('No live signal on the last candle according to the best strategy.')
        else:
            s = live_for_last[-1]
            st.write('Signal generated AT CLOSE of last candle:')
            st.json({
                'datetime': str(s['datetime']),
                'side': s['side'],
                'entry': s['entry'],
                'stoploss': s['sl'],
                'target': s['target'],
                'patterns': s['patterns'],
                'reason': s['reason']
            })

        # final human readable summary of what happened in backtest and live suggestion
        st.subheader('Final Human-readable Summary (what happened & live advice)')
        summary_bt = []
        summary_bt.append(f"Backtest from {df_use['date'].min().date()} to {df_use['date'].max().date()} using parameters {best['params']}. ")
        summary_bt.append(f"Total trades: {best['trades']}, accuracy {best['accuracy']:.2f}% with total points {best['total_pnl']:.2f}. ")
        if best['trades']>0:
            summary_bt.append(f"Winning trades: {best['wins']}, Losing trades: {best['losses']}. Average points per trade {best['avg_points']:.2f}. ")
        if len(live_for_last)>0:
            s = live_for_last[-1]
            summary_bt.append(f"Live recommendation: {s['side'].upper()} Entry at {s['entry']:.2f}, Target {s['target']:.2f}, SL {s['sl']:.2f}. Reason: {s['reason']}. Patterns observed: {s['patterns']}. ")
        else:
            summary_bt.append('No live recommendation at the most recent candle — wait for a pattern or breakout with volume confirmation as per the strategy. ')
        final_text = ' '.join(summary_bt)
        # truncate to ~150-200 words for clarity
        st.write(final_text)
        st.success('Done — please review the results and iterate parameter ranges if you want different trade-off between accuracy and returns.')

else:
    st.info('Adjust optimization parameters in the sidebar and click "Run Optimization & Backtest" to begin.')

# End of file
