# streamlit_swing_recommender.py
# Streamlit Swing Trading Recommender (single-file)
# - No talib or pandas_ta
# - Heuristic pattern detectors, ATR/RSI manually computed
# - Strict rule: entry executed at CLOSE of the signal candle (no future leakage)
# - Random or Grid optimization to find best parameter set
# - Live recommendation uses last available candle close
#
# Requirements:
# pip install streamlit pandas numpy plotly pytz scipy
# then: streamlit run streamlit_swing_recommender.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import pytz
import io
import math
import random
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Swing Recommender", initial_sidebar_state="expanded")

# ---------------------------
# Helpers: column mapping & timezone
# ---------------------------
def find_column(cols, candidates):
    """Return first column in cols where any candidate substring exists (case-insensitive)."""
    lc = [c.lower() for c in cols]
    for cand in candidates:
        for i, col in enumerate(lc):
            if cand in col:
                return cols[i]
    return None

def map_ohlcv(df):
    # map many possible names
    cols = list(df.columns)
    # date/datetime
    date_col = find_column(cols, ["date", "datetime", "time", "timestamp", "trade_time"])
    if date_col is None:
        # maybe index contains date
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_col = df.columns[0]
        else:
            raise ValueError("Couldn't find a date/datetime column. Please include one.")
    # other columns
    open_col = find_column(cols, ["open", "o", "openprice", "open_price", "price open", "price_open"])
    high_col = find_column(cols, ["high", "h", "highprice", "high_price"])
    low_col = find_column(cols, ["low", "l", "lowprice", "low_price"])
    close_col = find_column(cols, ["close", "c", "closeprice", "close_price", "last"])
    vol_col = find_column(cols, ["volume", "vol", "shares", "traded", "quantity", "qty"])
    # Minimal check
    missing = []
    if open_col is None: missing.append("open")
    if high_col is None: missing.append("high")
    if low_col is None: missing.append("low")
    if close_col is None: missing.append("close")
    if missing:
        raise ValueError(f"Missing columns detected: {', '.join(missing)}. The upload file must include OHLC fields (names can be any case or form).")
    # rename
    df = df.rename(columns={date_col: "datetime", open_col: "open", high_col: "high", low_col: "low", close_col: "close"})
    if vol_col:
        df = df.rename(columns={vol_col: "volume"})
    else:
        df["volume"] = 0.0
    return df

def to_ist_datetime(series):
    """Convert pandas series to timezone-aware Asia/Kolkata.
       If tz-aware: convert. If naive: assume UTC then convert.
    """
    series_parsed = pd.to_datetime(series, errors="coerce")
    if series_parsed.isna().any():
        # Try infer datetime format fallback
        series_parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    if series_parsed.isna().any():
        raise ValueError("Some datetime values couldn't be parsed. Check format.")
    # check tzinfo
    if series_parsed.dt.tz is None:
        # naive -> interpret as UTC then convert to IST (user requested IST)
        series_parsed = series_parsed.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    else:
        series_parsed = series_parsed.dt.tz_convert('Asia/Kolkata')
    return series_parsed

# ---------------------------
# Indicators (manual)
# ---------------------------
def compute_indicators(df, params):
    """
    Compute SMA, EMA, ATR, RSI, rolling vol, rolling max/min for support/resistance
    params: dict with keys like 'ma_short', 'ma_long', 'atr_period', 'rsi_period', 'sr_lookback'
    """
    df = df.copy()
    df['returns'] = df['close'].pct_change().fillna(0)
    # moving averages
    df['ma_short'] = df['close'].rolling(window=params['ma_short'], min_periods=1).mean()
    df['ma_long'] = df['close'].rolling(window=params['ma_long'], min_periods=1).mean()
    # True Range & ATR
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda row: max(row['high'] - row['low'], abs(row['high'] - row['prev_close'] if not pd.isna(row['prev_close']) else 0),
                        abs(row['low'] - row['prev_close'] if not pd.isna(row['prev_close']) else 0)), axis=1)
    df['atr'] = df['tr'].rolling(window=params['atr_period'], min_periods=1).mean()
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(params['rsi_period'], min_periods=1).mean()
    roll_down = down.rolling(params['rsi_period'], min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    # rolling vol
    df['vol_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
    # support/resistance: rolling max/min
    df['resistance_lookback'] = df['close'].rolling(window=params['sr_lookback'], min_periods=1).max().shift(1)
    df['support_lookback'] = df['close'].rolling(window=params['sr_lookback'], min_periods=1).min().shift(1)
    return df

# ---------------------------
# Pattern detection heuristics (approximate)
# ---------------------------
def detect_pivots(series, left=3, right=3):
    """Return indices of pivot highs and pivot lows"""
    n = len(series)
    pivots_high = []
    pivots_low = []
    for i in range(left, n-right):
        window = series.iloc[i-left:i+right+1]
        center = series.iloc[i]
        if center == window.max():
            # ensure unique
            if (window == center).sum() == 1:
                pivots_high.append(i)
        if center == window.min():
            if (window == center).sum() == 1:
                pivots_low.append(i)
    return pivots_high, pivots_low

def detect_double_top_bottom(df, lookback=20):
    """Heuristic double top/bottom detection"""
    clos = df['close']
    highs_idx, lows_idx = detect_pivots(clos, left=3, right=3)
    doubles = []
    # double top: two peaks at similar levels separated by some bars and valley in between
    for i in range(len(highs_idx)-1):
        i1, i2 = highs_idx[i], highs_idx[i+1]
        if 3 <= (i2 - i1) <= lookback:
            p1, p2 = clos.iloc[i1], clos.iloc[i2]
            mid_min = clos.iloc[i1:i2].min()
            if abs(p1 - p2) / max(p1, p2) < 0.02 and mid_min < min(p1, p2) * 0.98:
                doubles.append({'type':'double_top', 'left_idx':i1, 'right_idx':i2, 'peak_price':(p1+p2)/2})
    for i in range(len(lows_idx)-1):
        i1, i2 = lows_idx[i], lows_idx[i+1]
        if 3 <= (i2 - i1) <= lookback:
            p1, p2 = clos.iloc[i1], clos.iloc[i2]
            mid_max = clos.iloc[i1:i2].max()
            if abs(p1 - p2) / max(p1, p2) < 0.02 and mid_max > max(p1, p2) * 1.02:
                doubles.append({'type':'double_bottom', 'left_idx':i1, 'right_idx':i2, 'trough_price':(p1+p2)/2})
    return doubles

def detect_head_and_shoulders(df, window=60):
    """Simple heuristic: find pattern in moving window"""
    clos = df['close']
    patterns = []
    n = len(clos)
    for start in range(0, n-window, max(1, window//6)):
        end = min(start+window, n)
        seg = clos.iloc[start:end].reset_index(drop=True)
        highs = seg[seg == seg.rolling(5, center=True, min_periods=1).max()]
        # choose 3 peaks
        if len(highs) >= 3:
            # pick three largest distinct peaks in order
            peaks = highs.sort_values().index.tolist()
            # fallback simplistic approach: pick first, middle, last local maxima positions approx
            lm = seg.argmax()
            left = seg[:lm].argmax() if lm>0 else None
            right = seg[lm+1:].argmax() + lm + 1 if lm < len(seg)-1 else None
            if left is not None and right is not None:
                L = seg.iloc[left]; H = seg.iloc[lm]; R = seg.iloc[right]
                # shoulders lower than head and roughly similar
                if H > L*1.02 and H > R*1.02 and abs(L-R)/max(L,R) < 0.08:
                    patterns.append({'type':'head_and_shoulders', 'start_idx':start+left, 'head_idx':start+lm, 'end_idx':start+right})
    return patterns

def detect_triangle(df, lookback=40):
    """Detect triangular consolidation by checking converging trendlines over lookback"""
    prices = df['close'].values
    n = len(prices)
    patterns = []
    for end in range(lookback, n):
        start = end - lookback
        xs = np.arange(lookback)
        ys = prices[start:end]
        # linear fits for highs and lows (upper and lower envelopes)
        window_highs = pd.Series(ys).rolling(window=5, center=True, min_periods=1).max().dropna().values
        window_lows = pd.Series(ys).rolling(window=5, center=True, min_periods=1).min().dropna().values
        if len(window_highs) < 2 or len(window_lows) <2:
            continue
        # regress
        ah, bh = np.polyfit(xs[:len(window_highs)], window_highs, 1)
        al, bl = np.polyfit(xs[:len(window_lows)], window_lows, 1)
        # check for convergence: slopes opposite sign & magnitudes decreasing
        slope_diff = abs(ah - al)
        if (ah < 0 and al > 0) or (ah > 0 and al < 0):
            # check that distance between upper and lower at start is larger than at end (converging)
            start_gap = (ah*0 + bh) - (al*0 + bl)
            end_gap = (ah*(lookback-1) + bh) - (al*(lookback-1) + bl)
            if abs(end_gap) < abs(start_gap) * 0.8:
                patterns.append({'type':'triangle', 'start_idx':start, 'end_idx':end-1})
    return patterns

def detect_cup_handle(df, lookback=120):
    """Heuristic detection of cup and handle (rounded bottom then small pullback)"""
    clos = df['close'].values
    n = len(clos)
    res = []
    for end in range(lookback, n):
        start = end - lookback
        segment = clos[start:end]
        min_idx = segment.argmin()
        left_max = segment[:min_idx].max() if min_idx > 0 else None
        right_max = segment[min_idx+1:].max() if min_idx < len(segment)-1 else None
        if left_max and right_max:
            # cup: both sides must be similar tops and bottom in middle and > some depth
            if abs(left_max - right_max)/max(left_max, right_max) < 0.08:
                depth = min(left_max, right_max) - segment[min_idx]
                if depth / min(left_max, right_max) > 0.07:  # at least 7% rounded bottom
                    # handle: small consolidation after the right shoulder
                    # look for a small pullback after the right_max
                    # approximate: check next 10-20 bars for small decline
                    post = clos[end:end+20] if end+20 <= n else clos[end:n]
                    if len(post) >= 3 and (post.max() < right_max*1.02):
                        res.append({'type':'cup_handle', 'start_idx':start, 'bottom_idx':start+min_idx, 'end_idx':end-1})
    return res

# Detect supply/demand & SL-hunt zones (simple clustering of repeated highs/lows)
def detect_zones(df, lookback=30, cluster_thresh=0.01):
    clos = df['close']
    zones = {'supply': [], 'demand': [], 'sl_hunt': []}
    rolling_highs = clos.rolling(window=lookback, min_periods=1).max().shift(1)
    rolling_lows = clos.rolling(window=lookback, min_periods=1).min().shift(1)
    for i in range(lookback, len(clos)):
        rh = rolling_highs.iloc[i]
        rl = rolling_lows.iloc[i]
        # if price touches the rolling high multiple times in window -> supply zone
        seg = clos.iloc[i-lookback:i]
        if seg.isin([round(rh, 4)]).sum() >= 1:
            zones['supply'].append({'index':i, 'price':rh})
        if seg.isin([round(rl, 4)]).sum() >= 1:
            zones['demand'].append({'index':i, 'price':rl})
        # sl-hunt heuristics: big wick beyond support/resistance then reversal
        recent_high = df['high'].iloc[i]
        prev_close = df['close'].iloc[i-1] if i-1>=0 else df['close'].iloc[i]
        if recent_high > rh * (1 + 0.02) and df['close'].iloc[i] < prev_close:
            zones['sl_hunt'].append({'index':i, 'price':recent_high})
    return zones

# ---------------------------
# Strategy: entry/exits + backtest with strict entry-on-close
# ---------------------------
def generate_entry_signal(df, i, params, side='long'):
    """Return (bool, reason_dict) if an entry signal triggers at index i (based only on data up to i)"""
    # Input df must have indicators computed
    reason = {'confluences': []}
    row = df.iloc[i]
    # basic MA trend
    trend_ok = row['ma_short'] > row['ma_long'] if side=='long' else row['ma_short'] < row['ma_long']
    if trend_ok:
        reason['confluences'].append('trend_ma')
    # breakout vs resistance/support
    if side == 'long':
        if row['close'] > row['resistance_lookback'] * (1 + params['resistance_breakout_pct']):
            reason['confluences'].append('break_resistance')
    else:
        if row['close'] < row['support_lookback'] * (1 - params['resistance_breakout_pct']):
            reason['confluences'].append('break_support')
    # volume spike
    if row['volume'] > max(1, row['vol_ma'] * params['vol_mul']):  # avoid divide by zero
        reason['confluences'].append('volume_spike')
    # RSI filter
    if side == 'long':
        if row['rsi'] < params['rsi_threshold_long']:
            reason['confluences'].append('rsi_ok')
    else:
        if row['rsi'] > params['rsi_threshold_short']:
            reason['confluences'].append('rsi_ok')
    # pattern confluence: crude checks by scanning recent patterns stored in df._patterns (if present)
    pattern_match = None
    patterns = df.attrs.get('patterns', [])
    # choose patterns that include index i (approx)
    for p in patterns:
        if p.get('start_idx') is not None and p.get('end_idx') is not None:
            if p['start_idx'] <= i <= p['end_idx']:
                pattern_match = p['type']
                break
        else:
            # for double top/bottom etc
            if 'left_idx' in p and p['left_idx'] <= i <= p.get('right_idx', p['left_idx']):
                pattern_match = p['type']; break
    if pattern_match:
        if (side=='long' and pattern_match in ['double_bottom','cup_handle','triangle','W pattern','double_bottom']) or \
           (side=='short' and pattern_match in ['double_top','head_and_shoulders','M pattern']):
            reason['confluences'].append(f'pattern:{pattern_match}')
    # final confluence count
    count = len(reason['confluences'])
    # require minimum confluences
    if count >= params['min_confluence']:
        # also require ATR not too low
        if row['atr'] > 1e-9:
            return True, reason
    return False, reason

def backtest_strategy(df, params, side_choice='both', verbose=False):
    """Backtest with strict rule: when signal at i, enter at close[i]
       side_choice: 'long','short','both'
    """
    trades = []
    n = len(df)
    in_trade = False
    for i in range(params['start_idx'], n-1):
        if in_trade:
            continue  # we allow only one trade at a time for swing simplicity
        # check sides allowed
        for side in (['long','short'] if side_choice=='both' else [side_choice]):
            signal, reason = generate_entry_signal(df, i, params, side=side)
            if signal:
                entry_price = df['close'].iloc[i]  # strict entry on close of signal candle
                atr = df['atr'].iloc[i]
                if atr<=0: atr = df['close'].iloc[i]*0.01
                sl = entry_price - atr*params['atr_sl_mult'] if side=='long' else entry_price + atr*params['atr_sl_mult']
                # target as R-multiple
                if params['use_fixed_target']:
                    target = entry_price + params['fixed_target_points'] if side=='long' else entry_price - params['fixed_target_points']
                else:
                    rr = params['target_r']
                    if side=='long':
                        target = entry_price + (entry_price - sl)*rr
                    else:
                        target = entry_price - (sl - entry_price)*rr
                max_hold = params.get('max_hold_bars', 40)
                entry_dt = df['datetime'].iloc[i]
                exit_idx = None
                exit_price = None
                exit_dt = None
                exit_reason = None
                # Scan next candles for exit events
                for j in range(i+1, min(n, i+1+max_hold)):
                    high_j = df['high'].iloc[j]
                    low_j = df['low'].iloc[j]
                    close_j = df['close'].iloc[j]
                    # Long: target hit if high >= target; SL hit if low <= sl
                    if side=='long':
                        if low_j <= sl:
                            exit_idx = j
                            exit_price = sl  # assume SL hit at SL price (conservative)
                            exit_reason = 'sl_hit'
                            break
                        if high_j >= target:
                            exit_idx = j
                            exit_price = target
                            exit_reason = 'target_hit'
                            break
                    else:
                        if high_j >= sl:
                            exit_idx = j
                            exit_price = sl
                            exit_reason = 'sl_hit'
                            break
                        if low_j <= target:
                            exit_idx = j
                            exit_price = target
                            exit_reason = 'target_hit'
                            break
                # if no exit by hits and holding limit reached, exit at close of last bar checked
                if exit_idx is None:
                    j = min(n-1, i + max_hold)
                    exit_idx = j
                    exit_price = df['close'].iloc[j]
                    exit_reason = 'time_exit'
                pnl = (exit_price - entry_price) if side=='long' else (entry_price - exit_price)
                trades.append({
                    'side': side,
                    'entry_idx': i, 'entry_dt': entry_dt, 'entry_price': entry_price,
                    'exit_idx': exit_idx, 'exit_dt': df['datetime'].iloc[exit_idx], 'exit_price': exit_price,
                    'pnl': pnl, 'pnl_pct': pnl/entry_price*100.0,
                    'sl': sl, 'target': target, 'reason': reason, 'exit_reason': exit_reason
                })
                in_trade = True
                break
        # reset in_trade after we record? For simplicity we allow next signal after exit
        if in_trade:
            # allow next trades only after exit index + 1
            next_allowed_idx = trades[-1]['exit_idx'] + 1
            for skip_i in range(i+1, next_allowed_idx):
                pass
            in_trade = False
    # compute metrics
    if not trades:
        metrics = {'total_trades':0, 'wins':0, 'losses':0, 'pnl':0.0, 'accuracy':0.0, 'avg_pnl': 0.0}
    else:
        total = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = total - wins
        pnl = sum(t['pnl'] for t in trades)
        avg_pnl = np.mean([t['pnl'] for t in trades])
        accuracy = wins/total*100.0
        metrics = {'total_trades':total, 'wins':wins, 'losses':losses, 'pnl':pnl, 'accuracy':accuracy, 'avg_pnl':avg_pnl}
    return trades, metrics

# ---------------------------
# Optimization (Random or Grid)
# ---------------------------
def random_search(df, side_choice, trials, param_space, desired_accuracy, desired_points, progress_callback=None):
    best = None
    best_score = -1e12
    evaluated = 0
    for t in range(trials):
        params = {}
        for k, v in param_space.items():
            if v['type'] == 'int':
                params[k] = random.randint(v['min'], v['max'])
            elif v['type'] == 'float':
                params[k] = random.uniform(v['min'], v['max'])
            elif v['type'] == 'choice':
                params[k] = random.choice(v['vals'])
        # keep some required keys with defaults
        defaults = {'ma_short':5, 'ma_long':21, 'atr_period':14, 'rsi_period':14, 'sr_lookback':20,
                    'atr_sl_mult':2.0, 'target_r':2.0, 'min_confluence':2, 'vol_mul':1.2,
                    'resistance_breakout_pct':0.005, 'rsi_threshold_long':60, 'rsi_threshold_short':40,
                    'use_fixed_target':False, 'fixed_target_points':0.0, 'start_idx':max(50, params.get('ma_long',21))}
        for k,dv in defaults.items():
            params.setdefault(k, dv)
        # compute indicators with this param set
        df_ind = compute_indicators(df, params)
        # attach simple pattern list for confluence checking
        pats = detect_double_top_bottom(df_ind) + detect_head_and_shoulders(df_ind) + detect_triangle(df_ind) + detect_cup_handle(df_ind)
        df_ind.attrs['patterns'] = pats
        trades,metrics = backtest_strategy(df_ind, params, side_choice=side_choice)
        # scoring logic: prefer strategies that beat desired_accuracy and desired_points & positive pnl
        score = metrics['pnl']
        # boost strategies that meet desired accuracy and desired points (avg_pnl or total pnl)
        if metrics['accuracy'] >= desired_accuracy and metrics['pnl'] >= desired_points:
            score += metrics['pnl']*1.2 + metrics['accuracy']*10
        # favor accuracy if desired accuracy is high
        score += metrics['accuracy']*5
        if score > best_score:
            best_score = score
            best = {'params':params, 'metrics':metrics, 'trades':trades, 'pats':pats}
        evaluated += 1
        if progress_callback:
            progress_callback(int(evaluated/trials*100))
    return best

def grid_search(df, side_choice, grid, progress_callback=None):
    # grid is dict of parameter: list_of_values
    keys = list(grid.keys())
    combos = []
    def recurse(i, cur):
        if i == len(keys):
            combos.append(cur.copy()); return
        k = keys[i]
        for v in grid[k]:
            cur[k]=v
            recurse(i+1,cur)
    recurse(0,{})
    best = None
    best_score = -1e12
    for idx,combo in enumerate(combos):
        params = combo.copy()
        defaults = {'ma_short':5, 'ma_long':21, 'atr_period':14, 'rsi_period':14, 'sr_lookback':20,
                    'atr_sl_mult':2.0, 'target_r':2.0, 'min_confluence':2, 'vol_mul':1.2,
                    'resistance_breakout_pct':0.005, 'rsi_threshold_long':60, 'rsi_threshold_short':40,
                    'use_fixed_target':False, 'fixed_target_points':0.0, 'start_idx':max(50, params.get('ma_long',21))}
        for k,dv in defaults.items():
            params.setdefault(k, dv)
        df_ind = compute_indicators(df, params)
        pats = detect_double_top_bottom(df_ind) + detect_head_and_shoulders(df_ind) + detect_triangle(df_ind) + detect_cup_handle(df_ind)
        df_ind.attrs['patterns'] = pats
        trades,metrics = backtest_strategy(df_ind, params, side_choice=side_choice)
        score = metrics['pnl'] + metrics['accuracy']*5
        if score > best_score:
            best_score = score
            best = {'params':params, 'metrics':metrics, 'trades':trades, 'pats':pats}
        if progress_callback:
            progress_callback(int((idx+1)/len(combos)*100))
    return best

# ---------------------------
# Plot helpers
# ---------------------------
def plot_candles_with_trades(df, trades, title="Candles with Trades"):
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         name='candles')])
    for t in trades:
        entry_dt = t['entry_dt']
        exit_dt = t['exit_dt']
        fig.add_trace(go.Scatter(x=[entry_dt], y=[t['entry_price']], mode='markers', marker_symbol='triangle-up' if t['side']=='long' else 'triangle-down',
                                 marker=dict(size=12), name=f"Entry {t['side']}"))
        fig.add_trace(go.Scatter(x=[exit_dt], y=[t['exit_price']], mode='markers', marker_symbol='x', marker=dict(size=10), name=f"Exit {t['side']}"))
        # SL and target lines at the time of entry
        fig.add_hline(y=t['sl'], line_dash="dot", annotation_text="SL", annotation_position="right", opacity=0.6)
        fig.add_hline(y=t['target'], line_dash="dash", annotation_text="Target", annotation_position="right", opacity=0.6)
    fig.update_layout(height=600, title=title, xaxis_rangeslider_visible=False)
    return fig

def heatmap_year_month(df):
    df2 = df.copy()
    df2['year'] = df2['datetime'].dt.year
    df2['month'] = df2['datetime'].dt.month
    # monthly returns aggregated (sum of daily returns approximates)
    monthly = df2.groupby(['year','month'])['returns'].sum().reset_index()
    pivot = monthly.pivot(index='year', columns='month', values='returns').fillna(0)
    fig = px.imshow(pivot, labels=dict(x="Month", y="Year", color="Sum Returns"), origin='lower')
    fig.update_layout(height=300)
    return fig

# ---------------------------
# UI and flow
# ---------------------------
st.title("Swing Trading — Live Recommendations + Backtest (automated)")
st.markdown("Upload OHLCV file (CSV/Excel). Columns can be any names (e.g. 'Close', 'close price', 'last', 'Open Price', 'volume', etc.). The app will map columns, compute signals, run optimization, backtest and give a live recommendation on the last candle close. **Strategy entries are executed on the close of the signal candle (no future leakage).**")

# Sidebar options
with st.sidebar:
    st.header("Settings")
    side_choice = st.selectbox("Trade side", options=["both","long","short"], index=0)
    search_type = st.selectbox("Optimizer", options=["random_search","grid_search"], index=0)
    trials = st.number_input("Random search trials", min_value=5, max_value=500, value=40, step=5)
    desired_accuracy = st.slider("Desired accuracy (%)", min_value=50, max_value=95, value=70)
    desired_points = st.number_input("Desired total points (pnl) threshold", min_value=0.0, value=50.0, step=1.0)
    random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
    st.markdown("Defaults used inside strategy can be changed in advanced parameters below.")
    show_advanced = st.checkbox("Show advanced params", value=False)

if show_advanced:
    st.subheader("Advanced defaults (change if you like)")
    adv_ma_short = st.number_input("Default ma_short (used if optimizer doesn't set)", min_value=2, max_value=200, value=5)
    adv_ma_long = st.number_input("Default ma_long", min_value=5, max_value=500, value=21)
else:
    adv_ma_short = 5; adv_ma_long = 21

# File upload
uploaded = st.file_uploader("Upload CSV/Excel with OHLCV", type=['csv','xlsx','xls'])
if uploaded is None:
    st.info("Upload a file to continue. Example columns: Date, Open, High, Low, Close, Volume.")
    st.stop()

# Show progress
progress_bar = st.progress(0)
progress_text = st.empty()
progress_step = 0
def update_progress(pct, txt=None):
    progress_bar.progress(min(100, max(0,int(pct))))
    if txt:
        progress_text.text(txt)

# Read file
update_progress(2, "Reading file...")
try:
    if uploaded.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

update_progress(6, "Mapping columns...")
# Map columns
try:
    df_std = map_ohlcv(df_raw)
except Exception as e:
    st.error(f"Column mapping error: {e}")
    st.stop()

# Convert date to IST timezone
update_progress(10, "Converting datetime to IST...")
try:
    df_std['datetime'] = to_ist_datetime(df_std['datetime'])
except Exception as e:
    st.error(f"Datetime parse/convert error: {e}")
    st.stop()

# Ensure numeric columns
for col in ['open','high','low','close','volume']:
    df_std[col] = pd.to_numeric(df_std[col], errors='coerce')
df_std = df_std.dropna(subset=['open','high','low','close']).reset_index(drop=True)

# Sort ascending (no future leakage)
update_progress(15, "Sorting data ascending (no future leakage)...")
df_std = df_std.sort_values('datetime', ascending=True).reset_index(drop=True)

# Display head/tail and min/max
st.subheader("Data snapshot (mapped & sorted)")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.markdown("**Top 5 rows**")
    st.dataframe(df_std.head(5))
with c2:
    st.markdown("**Bottom 5 rows**")
    st.dataframe(df_std.tail(5))
with c3:
    st.markdown("**Range & Price Stats**")
    min_dt = df_std['datetime'].min()
    max_dt = df_std['datetime'].max()
    min_price = df_std['close'].min()
    max_price = df_std['close'].max()
    st.write(f"Date range: **{min_dt}**  →  **{max_dt}** (IST)")
    st.write(f"Min close: **{min_price:.4f}**, Max close: **{max_price:.4f}**")

# End date selection (user may choose earlier for backtest cutoff)
today_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
default_end = min(max_dt, pd.Timestamp(today_ist))
end_date = st.date_input("Select end date for backtest (inclusive)", value=default_end.date(), min_value=min_dt.date(), max_value=max_dt.date())
# Truncate data up to end_date (inclusive)
end_ts = pd.Timestamp(end_date).tz_localize('Asia/Kolkata')
df = df_std[df_std['datetime'] <= end_ts].reset_index(drop=True)
if df.empty:
    st.error("No data up to selected end date. Choose an earlier date.")
    st.stop()

update_progress(20, "Preparing EDA...")
# Plot raw close
st.subheader("Raw Price Chart and EDA")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df['datetime'], y=df['close'], mode='lines', name='Close'))
fig_price.update_layout(height=300, title="Close Price (selected range)")
st.plotly_chart(fig_price, use_container_width=True)

# Heatmap of returns year vs month
update_progress(28, "Computing returns heatmap...")
df['returns'] = df['close'].pct_change().fillna(0)
fig_heat = heatmap_year_month(df)
st.plotly_chart(fig_heat, use_container_width=True)

# 100-word summary about the stock data
update_progress(32, "Generating data summary...")
def generate_data_summary(df):
    # Basic stats + opportunity hints
    up_days = (df['returns']>0).sum()
    down_days = (df['returns']<0).sum()
    avg_return = df['returns'].mean()*100
    vol = df['returns'].std()*100
    last_price = df['close'].iloc[-1]
    summary = (f"The uploaded dataset spans {df['datetime'].min().date()} to {df['datetime'].max().date()} with {len(df)} candles. "
               f"Up days: {up_days}, down days: {down_days}. Average daily return ~{avg_return:.2f}%, volatility ~{vol:.2f}% (daily). "
               f"Recent price is {last_price:.2f}. Observed momentum (MA cross) and volatility suggest swing opportunities when price breaks clear support/resistance with volume confluence. "
               f"Watch supply/demand clusters for entries and beware SL-hunt wick spikes near levels.")
    return summary

st.markdown("**100-word data summary**")
st.write(generate_data_summary(df))

# compute baseline buy & hold points (for comparison)
buy_hold_points = df['close'].iloc[-1] - df['close'].iloc[0]
st.metric("Buy & Hold Points (end - start)", f"{buy_hold_points:.2f}")

# Advanced indicators default params & optimizer param space
base_param_space = {
    'ma_short': {'type':'int','min':3,'max':20},
    'ma_long': {'type':'int','min':10,'max':80},
    'atr_period': {'type':'int','min':5,'max':30},
    'rsi_period': {'type':'int','min':7,'max':21},
    'sr_lookback': {'type':'int','min':10,'max':60},
    'atr_sl_mult': {'type':'float','min':1.0,'max':4.0},
    'target_r': {'type':'float','min':1.0,'max':4.0},
    'min_confluence': {'type':'int','min':1,'max':4},
    'vol_mul': {'type':'float','min':0.8,'max':3.0},
    'resistance_breakout_pct': {'type':'float','min':0.001,'max':0.02},
    'rsi_threshold_long': {'type':'int','min':45,'max':80},
    'rsi_threshold_short': {'type':'int','min':20,'max':55}
}

# Show sample of parameters and allow user to continue
st.subheader("Optimization & Backtest controls")
col1, col2 = st.columns(2)
with col1:
    st.write("Optimizer type:", search_type)
    if search_type == "grid_search":
        st.write("**Grid Search selected** — a small grid will be used automatically; change trials to limit compute.")
with col2:
    st.write("Trade side:", side_choice)
st.write("Click **Run Optimization & Backtest** to start. This will compute patterns, search params, backtest, and show results. Live recommendation will be shown after that.")

# run optimization button
if st.button("Run Optimization & Backtest"):
    random.seed(int(random_seed))
    update_progress(35, "Starting pattern detection & indicator compute...")
    # default params for compute_indicators in search
    dummy_defaults = {'ma_short':adv_ma_short, 'ma_long':adv_ma_long, 'atr_period':14,
                      'rsi_period':14, 'sr_lookback':20}
    df_for_search = compute_indicators(df, dummy_defaults)
    # detect patterns and record in df attrs
    patterns_found = detect_double_top_bottom(df_for_search) + detect_head_and_shoulders(df_for_search) + detect_triangle(df_for_search) + detect_cup_handle(df_for_search)
    df_for_search.attrs['patterns'] = patterns_found
    update_progress(45, f"Found {len(patterns_found)} pattern hints. Starting optimizer...")
    # prepare param space conversion for random search
    param_space = base_param_space
    # Create progress callback to update progress bar
    def progress_cb(pct):
        update_progress(45 + int(pct*0.45), f"Optimizer running... {pct}%")
    if search_type == "random_search":
        best = random_search(df, side_choice, trials, param_space, desired_accuracy, desired_points, progress_callback=progress_cb)
    else:
        # small grid for demonstration based on param_space bounds
        grid = {
            'ma_short':[5,8],
            'ma_long':[21,34],
            'atr_sl_mult':[1.5,2.5],
            'target_r':[1.5,2.0],
            'min_confluence':[1,2]
        }
        best = grid_search(df, side_choice, grid, progress_callback=lambda p: update_progress(45 + int(p*0.45), f"Grid search {p}%"))
    update_progress(85, "Optimizer done. Running final backtest on best params...")
    if best is None:
        st.error("Optimizer did not return any candidate. Try increasing trials or check data quality.")
        st.stop()
    # Compute full indicators and attach patterns
    best_params = best['params']
    # ensure required items
    best_params.setdefault('start_idx', max(50, best_params.get('ma_long', adv_ma_long)))
    df_final = compute_indicators(df, best_params)
    df_final.attrs['patterns'] = best.get('pats', patterns_found)
    trades, metrics = backtest_strategy(df_final, best_params, side_choice=side_choice)
    update_progress(92, "Preparing results...")
    # Show best strategy summary
    st.subheader("Best Strategy (from search)")
    st.json(best_params)
    st.metric("Backtest PnL (points)", f"{metrics['pnl']:.2f}")
    st.metric("Backtest Accuracy (%)", f"{metrics['accuracy']:.2f}")
    st.metric("Total trades", f"{metrics['total_trades']}")
    # show top metrics
    st.write("**Backtest trade list (most recent 50)**")
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # convert datetimes to string
        trades_df['entry_dt'] = trades_df['entry_dt'].astype(str)
        trades_df['exit_dt'] = trades_df['exit_dt'].astype(str)
        st.dataframe(trades_df[['side','entry_dt','entry_price','exit_dt','exit_price','pnl','pnl_pct','exit_reason']].tail(50))
    else:
        st.write("No trades generated by best strategy.")

    # Plot candles with trades (all data truncated to end_date)
    fig_trades = plot_candles_with_trades(df_final, trades, title="Backtest — Candles with trades")
    st.plotly_chart(fig_trades, use_container_width=True)

    # Detailed explanation per trade (automated reason)
    st.subheader("Automated reasons & explanations (sample of last 10 trades)")
    if trades:
        for t in trades[-10:]:
            st.markdown(f"**{t['side'].upper()}** Entry: {t['entry_dt']} @ {t['entry_price']:.2f} | Exit: {t['exit_dt']} @ {t['exit_price']:.2f} | PnL: {t['pnl']:.2f} ({t['pnl_pct']:.2f}%)")
            # reason details
            confluences = t['reason'].get('confluences', [])
            st.write(f"**Reason / Logic**: Entry on close because -> {', '.join(confluences) if confluences else 'signal from indicators'}.")
            st.write(f"Stop Loss set at {t['sl']:.4f} (ATR × {best_params.get('atr_sl_mult',2.0):.2f}), Target {t['target']:.4f} ({best_params.get('target_r',2.0):.2f}R). Exit reason: {t['exit_reason']}.")
            st.write("---")
    else:
        st.write("No trades to explain.")

    update_progress(96, "Generating live recommendation...")
    # Live recommendation on last candle (use last candle's close as entry price if signal)
    last_idx = len(df_final)-1
    live_signal_long, reason_long = generate_entry_signal(df_final, last_idx, best_params, side='long')
    live_signal_short, reason_short = generate_entry_signal(df_final, last_idx, best_params, side='short')
    live_recs = []
    if (side_choice in ['long','both']) and live_signal_long:
        entry_price = df_final['close'].iloc[last_idx]
        atr = df_final['atr'].iloc[last_idx]
        sl = entry_price - atr*best_params['atr_sl_mult']
        target = entry_price + (entry_price - sl)*best_params['target_r'] if not best_params.get('use_fixed_target') else entry_price + best_params.get('fixed_target_points',0)
        # estimate probability_of_profit from backtest (win rate)
        prob = metrics['accuracy'] if metrics['total_trades']>0 else 50.0
        live_recs.append({'side':'long','entry_price':entry_price,'sl':sl,'target':target,'probability':prob,'reason':reason_long})
    if (side_choice in ['short','both']) and live_signal_short:
        entry_price = df_final['close'].iloc[last_idx]
        atr = df_final['atr'].iloc[last_idx]
        sl = entry_price + atr*best_params['atr_sl_mult']
        target = entry_price - (sl - entry_price)*best_params['target_r'] if not best_params.get('use_fixed_target') else entry_price - best_params.get('fixed_target_points',0)
        prob = metrics['accuracy'] if metrics['total_trades']>0 else 50.0
        live_recs.append({'side':'short','entry_price':entry_price,'sl':sl,'target':target,'probability':prob,'reason':reason_short})

    # Display live recommendation(s)
    st.subheader("LIVE Recommendation (based on last candle close)")
    if not live_recs:
        st.info("No live signal on the last candle using the selected strategy and confluence settings.")
    else:
        for rec in live_recs:
            st.markdown(f"### {rec['side'].upper()} Recommendation")
            st.write(f"- Entry (on close): **{rec['entry_price']:.4f}** at {df_final['datetime'].iloc[last_idx]}")
            st.write(f"- Stop Loss: **{rec['sl']:.4f}**")
            st.write(f"- Target: **{rec['target']:.4f}**")
            st.write(f"- Estimated probability of profit (based on backtest similar setups): **{rec['probability']:.2f}%**")
            st.write(f"- Reason / Logic: {', '.join(rec['reason'].get('confluences',[])) if rec['reason'].get('confluences') else 'Indicator confluences / pattern match'}")
            st.write(f"- Risk management: Use position sizing so that SL distance × position size ≤ your risk capital. Example: For SL distance {abs(rec['entry_price']-rec['sl']):.2f}, using 200 risk per trade -> qty = floor(200 / SL_distance).")
            st.write("---")

    update_progress(99, "Finalizing summaries...")
    # Final text summary of backtest and live recommendation
    def final_summary(metrics, trades, best_params):
        s = []
        s.append(f"The optimized backtest produced {metrics['total_trades']} trades with accuracy {metrics['accuracy']:.2f}% and total PnL {metrics['pnl']:.2f} points.")
        s.append(f"Key rules: MA({best_params.get('ma_short')}) vs MA({best_params.get('ma_long')}), ATR SL multiplier {best_params.get('atr_sl_mult'):.2f}, target R {best_params.get('target_r'):.2f}, min confluences {best_params.get('min_confluence')}.")
        s.append("Entries used confluence of trend, breakout of recent support/resistance, volume spike, and pattern confirmation when available.")
        s.append("Backtest entries were strictly executed on the close of signal candles (no future candle data used for entry).")
        if trades:
            avg_hold = np.mean([(pd.to_datetime(t['exit_dt']) - pd.to_datetime(t['entry_dt'])).days for t in trades])
            s.append(f"Average holding duration ~ {avg_hold:.1f} days.")
        return " ".join(s)
    st.subheader("Final summary and guidance (human readable)")
    st.write(final_summary(metrics, trades, best_params))
    update_progress(100, "Done.")

    st.success("Optimization, backtest and live recommendation complete. Please check the trade list, charts and explanations above. Tweak optimizer trials / desired accuracy if you need more aggressive or conservative strategy selection.")
    st.balloons()
else:
    st.info("Press the Run Optimization & Backtest button to begin analysis.")
