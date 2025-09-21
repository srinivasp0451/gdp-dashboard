"""
Streamlit Swing Trading Recommender
- Upload CSV/Excel OHLCV data (flexible column names)
- Maps columns automatically (open/high/low/close/volume/date/time)
- Sorts data ascending and converts dates to IST (Asia/Kolkata)
- Allows choosing an `end date` inside the dataset for backtesting
- Performs EDA (top/bottom rows, min/max, plots, year-month returns heatmap)
- Implements price-action based signal generation (patterns, support/resistance, volume/ATR confluences)
- Runs hyperparameter optimization (random search or grid search)
- Backtests with no-future-leakage: entry on the close of the candle that generated the signal
- Shows backtest trade-level table and summary stats
- Produces live recommendation on the last available candle using the best-found strategy

Dependencies:
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn

Note: This is a complex, best-effort implementation of many advanced price-action ideas. It is intended as a strong starting point you can iterate on. It avoids talib/pandas_ta as requested.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime, timedelta
import itertools
import random
from sklearn.model_selection import ParameterGrid

# -------------------- Utilities --------------------
st.set_page_config(layout="wide", page_title="Swing Trading Recommender")

@st.cache_data
def map_columns(df):
    """Map flexible column names to standard 'datetime','open','high','low','close','volume'"""
    col_map = {"datetime": None, "open": None, "high": None, "low": None, "close": None, "volume": None}
    cols = list(df.columns)
    lowered = [c.lower() for c in cols]
    for i, c in enumerate(lowered):
        if any(x in c for x in ["date", "time", "timestamp"]):
            if col_map['datetime'] is None:
                col_map['datetime'] = cols[i]
        if any(x in c for x in ["open", "o", "open price"]):
            if col_map['open'] is None:
                col_map['open'] = cols[i]
        if any(x in c for x in ["high", "h"]):
            if col_map['high'] is None:
                col_map['high'] = cols[i]
        if any(x in c for x in ["low", "l"]):
            if col_map['low'] is None:
                col_map['low'] = cols[i]
        if any(x in c for x in ["close", "c", "close price", "last", "price"]):
            if col_map['close'] is None:
                col_map['close'] = cols[i]
        if any(x in c for x in ["volume", "qty", "shares", "turnover"]):
            if col_map['volume'] is None:
                col_map['volume'] = cols[i]
    # Fallback heuristics: if any price column missing, try positional guesses
    if col_map['close'] is None:
        # try last column or look for 'adj close'
        for alt in ['adj close', 'adjclose', 'lastprice']:
            for c in cols:
                if alt in c.lower():
                    col_map['close'] = c
    return col_map


def ensure_datetime_ist(df, datetime_col):
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    # if any NaT, try to parse combining date+time columns (skipped here for brevity)
    # Localize or convert to Asia/Kolkata
    try:
        if df[datetime_col].dt.tz is None:
            df[datetime_col] = df[datetime_col].dt.tz_localize('Asia/Kolkata')
        else:
            df[datetime_col] = df[datetime_col].dt.tz_convert('Asia/Kolkata')
    except Exception:
        # fallback: remove tz aware then localize
        df[datetime_col] = pd.to_datetime(df[datetime_col].dt.tz_convert('UTC').dt.tz_localize(None))
        df[datetime_col] = df[datetime_col].dt.tz_localize('Asia/Kolkata')
    return df


def compute_indicators(df):
    df = df.copy()
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()))
    df['atr_14'] = df['tr'].rolling(14, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['return'] = df['close'].pct_change()
    return df


def peaks_troughs(df, window=5):
    """Find local highs and lows indexes using simple rolling window."""
    highs = []
    lows = []
    for i in range(window, len(df) - window):
        seg_high = df['high'].iloc[i - window: i + window + 1]
        seg_low = df['low'].iloc[i - window: i + window + 1]
        if df['high'].iloc[i] == seg_high.max():
            highs.append((i, df['high'].iloc[i]))
        if df['low'].iloc[i] == seg_low.min():
            lows.append((i, df['low'].iloc[i]))
    return highs, lows


def cluster_levels(levels, price_tolerance=0.01):
    """Cluster levels that are close to each other. tolerance as fraction of price"""
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lv in levels[1:]:
        if abs(lv - clusters[-1][-1]) <= price_tolerance * lv:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    # return representative (median)
    return [np.median(c) for c in clusters]


def detect_patterns(df, lookback=100):
    """Detect a few patterns using heuristic rules and return list of detected patterns with details."""
    res = []
    if len(df) < 30:
        return res
    tail = df.iloc[-lookback:]
    highs, lows = peaks_troughs(tail, window=3)
    # Double bottom / W
    if len(lows) >= 2:
        # take two deepest troughs separated by 3-30 candles with a middle peak higher than troughs
        lows_sorted = sorted(lows, key=lambda x: x[1])
        if len(lows_sorted) >= 2:
            i1, v1 = lows_sorted[0]
            i2, v2 = lows_sorted[1]
            if abs(v1 - v2) / max(v1, v2) < 0.08 and 3 <= abs(i2 - i1) <= 40:
                # find middle peak
                mid_idx = min(i1, i2) + abs(i2 - i1) // 2
                if tail['high'].iloc[mid_idx] > v1 * 1.02:
                    res.append({'pattern': 'W/Double Bottom', 'idxs': (tail.index[i1], tail.index[i2])})
    # Head and Shoulders (simplified)
    if len(highs) >= 3:
        # pick last three peaks
        h = highs[-3:]
        ph = [x[1] for x in h]
        if ph[1] > ph[0] and ph[1] > ph[2] and abs(ph[0] - ph[2]) / max(ph[0], ph[2]) < 0.08:
            res.append({'pattern': 'Head and Shoulders', 'idxs': (tail.index[h[0][0]], tail.index[h[1][0]], tail.index[h[2][0]])})
    # Cup & Handle (very rough): long rounded bottom then small pullback
    # check whether series had decline followed by recovery approx symmetric and handle as small pullback
    prices = tail['close'].values
    mid = len(prices) // 2
    left = prices[:mid]
    right = prices[mid:]
    if len(left) > 5 and len(right) > 5:
        if np.nanmin(left) < np.percentile(prices, 30) and np.nanmax(right) > np.percentile(prices, 70):
            # crude curvature check
            if np.mean(np.diff(left)) > -0.001 and np.mean(np.diff(right)) > 0.001:
                res.append({'pattern': 'Cup and Handle (approx)', 'idxs': (tail.index[0], tail.index[-1])})
    # Triangles / Wedges (look for converging highs/lows)
    # compute linear fits on highs and lows
    try:
        recent = df.iloc[-lookback:]
        x = np.arange(len(recent))
        hi = recent['high'].values
        lo = recent['low'].values
        # slopes
        slope_hi = np.polyfit(x, hi, 1)[0]
        slope_lo = np.polyfit(x, lo, 1)[0]
        if slope_hi * slope_lo < 0 and abs(slope_hi) < 0.5 * abs(slope_lo):
            res.append({'pattern': 'Symmetric Triangle', 'idxs': (recent.index[0], recent.index[-1])})
    except Exception:
        pass
    return res


def signal_from_price_action(df, params):
    """Given df up to time t (last row is current candle), return a signal dict or None.
       Signal includes side, entry, target, sl, reason"""
    # Evaluate conditions on last available candle
    last = df.iloc[-1]
    entry_price = last['close']
    atr = max(last.get('atr_14', 0.0001), 1e-6)
    sma_fast = last.get(f'sma_{params["fast_sma"]}', None)
    sma_slow = last.get(f'sma_{params["slow_sma"]}', None)
    patterns = detect_patterns(df, lookback=params['pattern_lookback'])
    volume_spike = False
    if 'volume' in df.columns:
        avg_vol = df['volume'].rolling(params.get('vol_lookback', 20), min_periods=1).mean().iloc[-1]
        volume_spike = last['volume'] > avg_vol * params.get('vol_mult', 2.0)

    reason_parts = []
    side = None
    # Simple trend filter
    sma_short = df['close'].rolling(params['fast_sma']).mean().iloc[-1] if params['fast_sma'] else None
    sma_long = df['close'].rolling(params['slow_sma']).mean().iloc[-1] if params['slow_sma'] else None
    trend = None
    if sma_short is not None and sma_long is not None:
        if sma_short > sma_long:
            trend = 'up'
        elif sma_short < sma_long:
            trend = 'down'
    # Pattern confluence
    pat_names = [p['pattern'] for p in patterns]
    if any('W/Double' in p for p in pat_names) or any('Cup' in p for p in pat_names):
        # bullish patterns
        reason_parts.append('Bullish pattern detected: ' + ','.join([p for p in pat_names if 'W' in p or 'Cup' in p]))
        side = 'long'
    if any('Head and Shoulders' in p for p in pat_names):
        reason_parts.append('Bearish pattern Head & Shoulders detected')
        side = 'short' if side is None else side
    if 'Symmetric Triangle' in pat_names:
        # breakout direction via last candle
        if df['close'].iloc[-1] > df['close'].iloc[-2]:
            reason_parts.append('Triangle breakout to upside')
            side = 'long'
        else:
            reason_parts.append('Triangle breakout to downside')
            side = 'short'
    # Volume confluence
    if volume_spike:
        reason_parts.append('Volume spike')
    # Trend confluence
    if trend == 'up':
        reason_parts.append('Price above short SMA; trend up')
    elif trend == 'down':
        reason_parts.append('Price below short SMA; trend down')

    # If no clear pattern but price made strong bullish candle with wicks
    body = abs(last['close'] - last['open'])
    candle_strength = body / max(last['high'] - last['low'], 1e-9)
    if candle_strength > params.get('candle_strength', 0.6) and last['close'] > last['open']:
        reason_parts.append('Strong bullish candle')
        if side is None:
            side = 'long'
    if candle_strength > params.get('candle_strength', 0.6) and last['close'] < last['open']:
        reason_parts.append('Strong bearish candle')
        if side is None:
            side = 'short'

    # If user selected side preference, apply later in UI wrapper
    if side is None:
        return None

    # compute sl and target
    sl_buffer = params.get('atr_mult', 1.0) * atr
    if side == 'long':
        sl = entry_price - sl_buffer
        target = entry_price + params.get('target_rr', 1.5) * (entry_price - sl)
    else:
        sl = entry_price + sl_buffer
        target = entry_price - params.get('target_rr', 1.5) * (sl - entry_price)

    # probability heuristic: combine factors
    prob = 0.5
    if 'W/Double' in ','.join(pat_names) or 'Cup' in ','.join(pat_names):
        prob += 0.2
    if volume_spike:
        prob += 0.1
    if trend == 'up' and side == 'long':
        prob += 0.1
    if trend == 'down' and side == 'short':
        prob += 0.1
    prob = min(0.95, prob)

    return {
        'side': side,
        'entry': entry_price,
        'sl': float(sl),
        'target': float(target),
        'probability': prob,
        'reason': '; '.join(reason_parts) if reason_parts else 'Price-action signal'
    }


def backtest_engine(df, params):
    """Run walk-forward backtest on df using the signal generator (no future leakage). Returns trades list and metrics."""
    trades = []
    df = df.copy().reset_index(drop=True)
    for t in range(params.get('min_lookback', 50), len(df) - 1):
        # create view up to t (inclusive)
        view = df.iloc[:t + 1].copy()
        sig = signal_from_price_action(view, params)
        if sig is None:
            continue
        entry_idx = t
        entry_time = view['datetime'].iloc[-1]
        entry_price = sig['entry']
        sl = sig['sl']
        target = sig['target']
        side = sig['side']
        # simulate forward until target or sl hit
        exit_idx = None
        exit_price = None
        exit_time = None
        for f in range(t + 1, len(df)):
            high = df['high'].iloc[f]
            low = df['low'].iloc[f]
            time = df['datetime'].iloc[f]
            # if both sl and target hit in same candle, determine which touched first using proximity to open
            op = df['open'].iloc[f]
            if side == 'long':
                hit_target = high >= target
                hit_sl = low <= sl
                if hit_target and not hit_sl:
                    exit_idx = f
                    exit_price = target
                    exit_time = time
                    break
                if hit_sl and not hit_target:
                    exit_idx = f
                    exit_price = sl
                    exit_time = time
                    break
                if hit_target and hit_sl:
                    # determine which comes first: compare distances from open
                    if abs(op - target) < abs(op - sl):
                        exit_price = target
                    else:
                        exit_price = sl
                    exit_idx = f
                    exit_time = time
                    break
            else:
                hit_target = low <= target
                hit_sl = high >= sl
                if hit_target and not hit_sl:
                    exit_idx = f
                    exit_price = target
                    exit_time = time
                    break
                if hit_sl and not hit_target:
                    exit_idx = f
                    exit_price = sl
                    exit_time = time
                    break
                if hit_target and hit_sl:
                    if abs(op - target) < abs(op - sl):
                        exit_price = target
                    else:
                        exit_price = sl
                    exit_idx = f
                    exit_time = time
                    break
        if exit_idx is None:
            # position still open at end of data: close at last close
            exit_idx = len(df) - 1
            exit_price = df['close'].iloc[-1]
            exit_time = df['datetime'].iloc[-1]
        # record trade
        pnl = (exit_price - entry_price) if side == 'long' else (entry_price - exit_price)
        pnl_percent = pnl / entry_price * 100.0
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'target': target,
            'sl': sl,
            'pnl_points': pnl,
            'pnl_pct': pnl_percent,
            'duration': (exit_time - entry_time) / np.timedelta64(1, 'm'),
            'reason': sig['reason'],
            'prob_est': sig['probability']
        })
    trades_df = pd.DataFrame(trades)
    # metrics
    if len(trades_df) > 0:
        total_pnl = trades_df['pnl_points'].sum()
        win_rate = (trades_df['pnl_points'] > 0).mean()
        total_trades = len(trades_df)
        avg_pnl = trades_df['pnl_points'].mean()
        pos_trades = (trades_df['pnl_points'] > 0).sum()
        neg_trades = (trades_df['pnl_points'] <= 0).sum()
    else:
        total_pnl = win_rate = total_trades = avg_pnl = pos_trades = neg_trades = 0
    metrics = {
        'total_pnl': float(total_pnl),
        'win_rate': float(win_rate) if isinstance(win_rate, (float, np.floating)) else 0.0,
        'total_trades': int(total_trades),
        'avg_pnl': float(avg_pnl),
        'positive_trades': int(pos_trades),
        'negative_trades': int(neg_trades)
    }
    return trades_df, metrics


def objective_for_params(df_train, params):
    trades_df, metrics = backtest_engine(df_train, params)
    # objective: maximize total_pnl and win_rate
    score = metrics['total_pnl'] + metrics['win_rate'] * 100.0
    # penalize too few trades
    if metrics['total_trades'] < 3:
        score -= 1000
    return score, trades_df, metrics

# -------------------- Streamlit App Layout --------------------

st.title("📈 Swing Trading Recommender — Automated Backtest & Live Signal")
st.markdown("Upload your historical OHLCV file (CSV or Excel). Columns can be any names — code will map them.")

uploaded_file = st.file_uploader("Upload CSV / Excel with OHLCV (or OHLC)", type=["csv", "xlsx", "xls"]) 

if uploaded_file is not None:
    # Read file robustly
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.sidebar.header("Backtest / Live Options")
    # Column mapping
    col_map = map_columns(df_raw)
    st.sidebar.subheader("Detected column mapping (you can override)")
    # allow user to override
    cols = list(df_raw.columns)
    for key in col_map:
        selected = st.sidebar.selectbox(f"{key}", options=[None] + cols, index=cols.index(col_map[key]) + 1 if col_map[key] in cols else 0)
        if selected:
            col_map[key] = selected

    # require at least datetime and close
    if col_map['datetime'] is None or col_map['close'] is None:
        st.error("Couldn't detect datetime or close column. Please select them in the sidebar.")
        st.stop()

    # Build working df
    df = df_raw.copy()
    # rename columns for standardization if found
    rename_map = {}
    for k, v in col_map.items():
        if v is not None:
            rename_map[v] = k
    df = df.rename(columns=rename_map)

    # ensure numeric
    for p in ['open', 'high', 'low', 'close', 'volume']:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors='coerce')

    # datetime handling
    df = ensure_datetime_ist(df, 'datetime')
    df = df.sort_values('datetime', ascending=True).reset_index(drop=True)

    # show top and bottom
    st.subheader("Data preview")
    st.write("Top 5 rows")
    st.dataframe(df.head(5))
    st.write("Bottom 5 rows")
    st.dataframe(df.tail(5))

    start_date = df['datetime'].dt.date.min()
    default_end = datetime.now().astimezone().date()
    end_date_input = st.sidebar.date_input("Select end date for backtest (results will be generated using data up to this date)", value=default_end, min_value=start_date, max_value=df['datetime'].dt.date.max())
    # ensure end_date not after last date in data
    if end_date_input > df['datetime'].dt.date.max():
        st.warning("Selected end date is after the available data. Using last available data date instead for backtest.")
        end_date_input = df['datetime'].dt.date.max()

    # filter for backtest
    df_backtest = df[df['datetime'].dt.date <= end_date_input].copy()
    if df_backtest.empty:
        st.error("No data available up to selected end date. Choose earlier date.")
        st.stop()

    # compute indicators
    df = compute_indicators(df)
    df_backtest = compute_indicators(df_backtest)

    # summary stats top
    st.subheader("Basic stats")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Start date", str(df['datetime'].dt.date.min()))
    with c2:
        st.metric("End date", str(df['datetime'].dt.date.max()))
    with c3:
        st.metric("Min price", float(df['close'].min()))
    with c4:
        st.metric("Max price", float(df['close'].max()))

    # Raw candlestick plot for entire dataset
    st.subheader("Candlestick — full data")
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Year-Month returns heatmap
    st.subheader("Year vs Month returns heatmap")
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    monthly = df.groupby(['year', 'month'])['return'].sum().reset_index()
    pivot = monthly.pivot(index='year', columns='month', values='return').fillna(0)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt='.2f', ax=ax2)
    ax2.set_title('Sum of returns by year and month')
    st.pyplot(fig2)

    # 100-word summary about data
    def human_summary(df):
        returns = df['return'].dropna()
        avg = returns.mean()
        vol = returns.std()
        direction = 'neutral'
        if avg > 0.0005:
            direction = 'slightly bullish'
        elif avg < -0.0005:
            direction = 'slightly bearish'
        summary = f"The provided dataset spans {df['datetime'].dt.date.min()} to {df['datetime'].dt.date.max()}. Average daily return is {avg:.4f} with volatility {vol:.4f}. Overall short-term bias appears {direction}. Notable price swings exist (look at months with darker heatmap cells). Potential opportunities include trading clear price-action patterns (breakouts from triangles, W bottoms, head & shoulders failures) and using ATR-based stops to manage risk. Ensure trades align with trend and volume spikes for confluence. This summary is a concise starting point for swing setups."
        return summary

    st.subheader("Data summary (100 words approx)")
    st.write(human_summary(df))

    # Strategy selection UI
    st.sidebar.header("Strategy & Optimization")
    side_choice = st.sidebar.selectbox("Which side to consider?", options=['both', 'long', 'short'])
    search_type = st.sidebar.selectbox("Search type", options=['random_search', 'grid_search'])
    iterations = st.sidebar.number_input("Number of evaluations (random or grid choices)", min_value=5, max_value=500, value=60)
    desired_accuracy = st.sidebar.slider("Desired minimum accuracy (win rate) %", min_value=10, max_value=95, value=50)
    # hyperparameter space (user can tune if needed)
    st.sidebar.subheader("Hyperparameter ranges (advanced)")
    fast_choices = st.sidebar.multiselect("fast SMA (choices)", options=[5,8,10,12,15,20], default=[8, 10, 15])
    slow_choices = st.sidebar.multiselect("slow SMA (choices)", options=[20,30,50,80,100], default=[30,50])
    atr_mult_choices = st.sidebar.multiselect("ATR mult (choices)", options=[0.5,1.0,1.5,2.0,3.0], default=[1.0,1.5])
    target_rr_choices = st.sidebar.multiselect("Target R:R (choices)", options=[1.0,1.5,2.0,3.0], default=[1.5,2.0])

    # prepare parameter grid or random
    base_param_space = {
        'fast_sma': fast_choices or [8, 10],
        'slow_sma': slow_choices or [30,50],
        'atr_mult': atr_mult_choices or [1.0,1.5],
        'target_rr': target_rr_choices or [1.5,2.0],
        'pattern_lookback': [30, 60, 120],
        'vol_lookback': [10, 20],
        'vol_mult': [1.5, 2.0, 3.0],
        'candle_strength': [0.5, 0.6, 0.7],
        'min_lookback': [20, 40, 60]
    }

    # Build list of candidate parameter dicts
    candidates = []
    if search_type == 'grid_search':
        grid = list(ParameterGrid(base_param_space))
        if len(grid) > iterations:
            grid = random.sample(grid, iterations)
        candidates = grid
    else:
        # random search: sample randomly
        keys = list(base_param_space.keys())
        for _ in range(iterations):
            p = {k: random.choice(base_param_space[k]) for k in keys}
            candidates.append(p)

    # Run optimization
    st.header("Optimization & Backtest")
    if st.button("Run optimization & backtest"):
        progress = st.progress(0)
        best_score = -1e9
        best = None
        best_trades = None
        best_metrics = None
        results = []
        n = len(candidates)
        for i, params in enumerate(candidates):
            # merge default risk params
            params_full = params.copy()
            params_full.update({'atr_mult': params.get('atr_mult', 1.0), 'target_rr': params.get('target_rr', 1.5), 'fast_sma': params.get('fast_sma', 8), 'slow_sma': params.get('slow_sma', 30)})
            # ensure sma columns presence in df (we compute rolling inside indicators)
            score, trades_df, metrics = objective_for_params(df_backtest, params_full)
            results.append({**params_full, 'score': score, **metrics})
            # choose candidate if it meets desired_accuracy else try maximize composite
            if metrics['win_rate'] * 100.0 >= desired_accuracy and metrics['total_trades'] >= 3:
                # prefer the highest total_pnl among those
                if metrics['total_pnl'] > best_score:
                    best_score = metrics['total_pnl']
                    best = params_full
                    best_trades = trades_df
                    best_metrics = metrics
            else:
                # fallback composite
                if score > best_score:
                    best_score = score
                    best = params_full
                    best_trades = trades_df
                    best_metrics = metrics
            progress.progress(int((i + 1) / n * 100))
        st.success("Optimization completed")
        st.subheader("Best strategy summary")
        st.json(best)
        st.write("Backtest metrics for best strategy:")
        st.json(best_metrics)

        # show trades
        if best_trades is not None and not best_trades.empty:
            st.subheader("Backtest trades (entry/exit table)")
            display_trades = best_trades.copy()
            display_trades['entry_time'] = display_trades['entry_time'].astype(str)
            display_trades['exit_time'] = display_trades['exit_time'].astype(str)
            st.dataframe(display_trades)
        else:
            st.info("No trades generated by best strategy on the chosen backtest period.")

        # Summarize backtest in human readable words (last summary)
        def backtest_human_summary(metrics, trades_df):
            s = f"Backtest result: {metrics['total_trades']} trades, win rate {metrics['win_rate']*100:.1f}%, total points {metrics['total_pnl']:.2f}. "
            if metrics['total_trades'] > 0:
                s += f"Average per trade {metrics['avg_pnl']:.2f}. Positive trades {metrics['positive_trades']}, negative trades {metrics['negative_trades']}."
            s += " Strategy logic: price-action patterns + SMA trend + ATR-based SL + volume confluence."
            return s

        st.subheader("Backtest summary (human readable)")
        st.write(backtest_human_summary(best_metrics, best_trades))

        # Live recommendation on last candle using best strategy
        st.header("Live Recommendation (last candle)")
        live_params = best.copy() if best else candidates[0]
        # apply side filter
        live_sig = signal_from_price_action(df.copy(), live_params)
        if live_sig is None:
            st.info("No live signal on last candle based on best strategy.")
        else:
            if side_choice != 'both' and live_sig['side'] != side_choice:
                st.info(f"Signal generated is {live_sig['side']} but you selected side '{side_choice}'. No recommendation shown.")
            else:
                st.metric("Signal", live_sig['side'].upper())
                st.write(f"Entry (close of last candle): {live_sig['entry']}")
                st.write(f"Target: {live_sig['target']}   Stop Loss: {live_sig['sl']}")
                st.write(f"Estimated probability of profit: {live_sig['probability']*100:.1f}%")
                st.write("Reason / logic:")
                st.write(live_sig['reason'])

        st.balloons()

    st.sidebar.write("Notes: The system enforces entry at the close of the signal candle (no future leakage). Backtest uses forward simulation to determine exits. The optimization is a heuristic search—tune param ranges to explore more outcomes.")

else:
    st.info("Upload a CSV or Excel file to get started.")

# EOF
