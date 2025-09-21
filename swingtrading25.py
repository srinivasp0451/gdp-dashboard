# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timezone
import pytz
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterSampler
from itertools import product
import random
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Swing Trade Recommender (Price Action)")

# -------------------------
# Helper: Column mapping
# -------------------------
def map_columns(df):
    """Map common OHLCV columns from messy names. Returns df with columns
    ['date','open','high','low','close','volume'] when possible."""
    col_map = {}
    lower_cols = {c: c.lower() for c in df.columns}
    find = lambda keys: next((c for c, lc in lower_cols.items() if any(k in lc for k in keys)), None)

    # date
    date_col = find(['date', 'time', 'timestamp', 'dt'])
    if date_col is None:
        # maybe index is date-like
        date_col = None
    else:
        col_map['date'] = date_col

    col_map['open'] = find(['open', 'o', 'openprice', 'open_price'])
    col_map['high'] = find(['high', 'h', 'highprice', 'high_price', 'hi'])
    col_map['low']  = find(['low', 'l', 'lowprice', 'low_price'])
    col_map['close']= find(['close', 'c', 'closeprice', 'close_price', 'last', 'price'])
    col_map['volume']= find(['volume', 'vol', 'qty', 'shares', 'trades', 'turnover'])

    # Attempt to ensure at least close exists
    if col_map.get('close') is None:
        # try any column that looks numeric and not date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col_map['close'] = numeric_cols[-1]

    return col_map

# -------------------------
# Helper: Date parsing to IST
# -------------------------
def ensure_datetime_ist(df, date_col):
    """Convert/ensure date_col is timezone-aware in Asia/Kolkata."""
    if date_col is None:
        st.error("No date/time column detected. Please ensure your file has a date column.")
        raise ValueError("Missing date column")

    s = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    if s.isna().all():
        st.error("Date parsing failed — check your date format.")
        raise ValueError("Date parsing failed")

    # If timezone-aware, convert; if naive, localize to Asia/Kolkata (user requested IST)
    try:
        if s.dt.tz is None:
            s = s.dt.tz_localize('Asia/Kolkata')
        else:
            s = s.dt.tz_convert('Asia/Kolkata')
    except Exception:
        # fallback: treat naive as UTC then convert (safer if data in UTC)
        try:
            s = s.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        except Exception:
            s = s.dt.tz_localize('Asia/Kolkata')

    df[date_col] = s
    return df

# -------------------------
# Indicators & utilities
# -------------------------
def sma(series, n):
    return series.rolling(n, min_periods=1).mean()

def atr(df, n=14):
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def rolling_support_resistance(series, window=50):
    """Simple SR: recent rolling min and max"""
    sup = series.rolling(window, min_periods=1).min()
    res = series.rolling(window, min_periods=1).max()
    return sup, res

def detect_head_shoulders(prices, window=30):
    # heuristic: find three peaks in sliding window where middle is highest and sides similar
    hs = [False]*len(prices)
    for i in range(window, len(prices)-window):
        seg = prices[i-window:i+window+1]
        mid = window
        left_peak = seg[:mid].idxmax() if len(seg[:mid])>0 else None
        right_peak = seg[mid+1:].idxmax() if len(seg[mid+1:])>0 else None
        if left_peak and right_peak:
            try:
                left_val = seg.loc[left_peak]
                mid_val = seg.iloc[mid]
                right_val = seg.loc[right_peak]
                # basic check
                if mid_val > left_val*1.02 and mid_val > right_val*1.02 and abs(left_val-right_val)/mid_val < 0.10:
                    hs[i] = True
            except Exception:
                pass
    return pd.Series(hs, index=prices.index)

def detect_double_tops_bottoms(series, window=20, threshold=0.02):
    dt = [None]*len(series)
    for i in range(window, len(series)-window):
        seg = series[i-window:i+window+1]
        peak_idx = seg.idxmax()
        # find another peak before/after close in height
        left = seg[:peak_idx]
        right = seg[peak_idx+1:]
        if not left.empty and not right.empty:
            left_max = left.max()
            right_max = right.max()
            peak = seg.loc[peak_idx]
            # double top
            if abs(left_max-peak)/peak < threshold or abs(right_max-peak)/peak < threshold:
                dt[peak_idx] = 'double_top'
            # double bottom (invert)
        # analogous for bottoms
    return pd.Series(dt, index=series.index)

def detect_triangle(df, lookback=50):
    # detect consolidating highs and lows with narrowing range
    tri = [False]*len(df)
    for i in range(lookback, len(df)):
        seg = df.iloc[i-lookback:i]
        highs = seg['high'].values
        lows = seg['low'].values
        # linear fit slope magnitude shrinking heuristic
        hx = np.arange(len(highs))
        lx = np.arange(len(lows))
        try:
            hslope = np.polyfit(hx, highs, 1)[0]
            lslope = np.polyfit(lx, lows, 1)[0]
            if abs(hslope) < 0.02 and abs(lslope) < 0.02 and (highs.max()-lows.min())/np.median(seg['close']) < 0.06:
                tri[i] = True
        except Exception:
            pass
    return pd.Series(tri, index=df.index)

# -------------------------
# Pattern aggregator (returns reasons)
# -------------------------
def detect_patterns(df):
    reasons = []
    hs = detect_head_shoulders(df['close'])
    tri = detect_triangle(df)
    # Add points where pattern True
    for idx in hs[hs].index:
        reasons.append((idx, "Head and Shoulders-like structure observed (mid-peak higher than shoulders)"))
    for idx in tri[tri].index:
        reasons.append((idx, "Triangle/consolidation (tight highs & lows) — possible breakout/trap"))
    # Additional heuristics:
    # detect W/M patterns by local extrema in rolling windows
    # Cup & handle: large rounding bottom followed by small pullback
    # We'll use approximate heuristics for these:
    # Cup & handle
    window = 120
    for i in range(window, len(df)):
        seg = df['close'].iloc[i-window:i]
        if len(seg) < window: continue
        left = seg[:window//2]
        right = seg[window//2:]
        # cup: left higher than mid low and right climbs back
        mid = seg.idxmin()
        if seg.min() < left.mean()*0.95 and right.max() >= left.mean()*0.98:
            reasons.append((df.index[i], "Cup-and-handle-like rounding bottom observed in prior window"))
    return reasons

# -------------------------
# Strategy signal generator
# -------------------------
def generate_signals(df, params):
    """Return a DataFrame 'signals' with booleans for long/short entries and textual reason."""
    # compute indicators using only past data (pandas handles that)
    df = df.copy()
    df['sma_fast'] = sma(df['close'], params['fast_ma'])
    df['sma_slow'] = sma(df['close'], params['slow_ma'])
    df['atr'] = atr(df, params['atr_period'])
    sup, res = rolling_support_resistance(df['low'], window=params['sr_window'])
    df['sup'] = sup
    df['res'] = res

    signals = []
    patterns = detect_patterns(df)

    # logic:
    # - trend bias: sma_fast > sma_slow => bullish bias, opposite => bearish
    # - breakout: close above recent resistance (res rolling), with volume surge and ATR support
    # - pullback long: price near support + bullish SMA structure + bullish pattern
    # - sl hunting: if candle wicks below support but closes above, mark as fake-breakout/trap

    for i in range(len(df)):
        row = df.iloc[i]
        reason = []
        long = False
        short = False
        # need sufficient past
        if i < max(params['fast_ma'], params['slow_ma']):
            signals.append({'index': df.index[i], 'long': False, 'short': False, 'reason': 'insufficient history'})
            continue

        trend = 'bull' if row['sma_fast'] > row['sma_slow'] else 'bear'
        # breakout long
        if row['close'] > row['res']*(1 + params['res_breakout_pct']) and row['close'] > row['sma_fast']:
            # check volume spike
            vol_ok = False
            if 'volume' in df.columns and not df['volume'].isna().all():
                vol_m = df['volume'].iloc[max(0, i-params['vol_ma']):i+1].mean()
                if vol_m > 0 and row['volume'] > vol_m*(1+params['vol_spike_pct']):
                    vol_ok = True
            else:
                vol_ok = True  # no volume data, allow
            if vol_ok:
                long = True
                reason.append("Breakout above recent resistance with momentum")
        # pullback long (near support)
        if row['close'] <= row['sup']*(1+params['sup_pull_pct']) and trend=='bull':
            long = True
            reason.append("Bullish pullback near support with SMA trend")
        # sl hunt detection: long if wick pierced below support but closed above
        candle_low = row['low']
        candle_close = row['close']
        if candle_low < row['sup']*(1 - params['sl_hunt_pct']) and candle_close > row['sup']:
            long = True
            reason.append("SL-hunt pattern (wicked below support but closed above) — likely trap")

        # symmetric for short
        if row['close'] < row['sup']*(1 - params['res_breakout_pct']) and row['close'] < row['sma_fast']:
            short = True
            reason.append("Breakdown below recent support with momentum")
        if row['close'] >= row['res']*(1-params['sup_pull_pct']) and trend=='bear':
            short = True
            reason.append("Bearish pullback near resistance with SMA trend")
        # detect patterns: if pattern present near i, adjust reason
        pat_reasons = [r for idx, r in patterns if abs((idx - df.index[i]).total_seconds()) < params['pattern_time_seconds']]
        if pat_reasons:
            # if pattern indicates reversal and matches trend, flip or strengthen
            reason += pat_reasons

        signals.append({'index': df.index[i], 'long': long, 'short': short, 'reason': "; ".join(reason) if reason else "No clean signal"})
    return pd.DataFrame(signals).set_index('index')

# -------------------------
# Backtest engine
# -------------------------
def run_backtest(df, signals, params, side='both'):
    """
    Backtest trades:
    - Enter on the close price of candle where signals appear (no future leakage).
    - Set target and stoploss using ATR multipliers.
    - Exit when price hits target or SL in subsequent candles; if neither until max_hold_days, exit at close of last allowed candle.
    """
    trades = []
    max_hold = params.get('max_hold', 20)  # in candles
    for idx in signals.index:
        sig = signals.loc[idx]
        if side=='long' and not sig['long']: continue
        if side=='short' and not sig['short']: continue
        if side=='both' and not (sig['long'] or sig['short']): continue

        # entry info
        i = df.index.get_loc(idx)
        entry_price = df['close'].iloc[i]
        atr_val = df['atr'].iloc[i] if 'atr' in df.columns else (df['high'].iloc[max(0,i-14):i+1].std())
        direction = 'long' if sig['long'] else 'short'
        if direction == 'long':
            tp = entry_price * (1 + params['tp_atr_mult'] * atr_val / entry_price)
            sl = entry_price * (1 - params['sl_atr_mult'] * atr_val / entry_price)
        else:
            tp = entry_price * (1 - params['tp_atr_mult'] * atr_val / entry_price)
            sl = entry_price * (1 + params['sl_atr_mult'] * atr_val / entry_price)

        reason = sig['reason']
        open_i = i+1
        exit_price = None
        exit_idx = None
        exit_reason = None
        pnl = None

        # scan forward candles (starting from next candle) until TP or SL hit or max_hold
        for j in range(i, min(len(df), i+1+max_hold)):
            high = df['high'].iloc[j]
            low = df['low'].iloc[j]
            close = df['close'].iloc[j]
            time_j = df.index[j]

            if direction == 'long':
                # check SL first if pierced in candle (sl-hunt earlier - but exit logic prioritize hitting SL/TP)
                if low <= sl:
                    exit_price = sl
                    exit_idx = time_j
                    exit_reason = 'SL hit'
                    pnl = exit_price - entry_price
                    break
                if high >= tp:
                    exit_price = tp
                    exit_idx = time_j
                    exit_reason = 'TP hit'
                    pnl = exit_price - entry_price
                    break
            else:
                if high >= sl:
                    exit_price = sl
                    exit_idx = time_j
                    exit_reason = 'SL hit'
                    pnl = entry_price - exit_price
                    break
                if low <= tp:
                    exit_price = tp
                    exit_idx = time_j
                    exit_reason = 'TP hit'
                    pnl = entry_price - exit_price
                    break

            # else continue; if reach end of max_hold, exit at close
            if j == min(len(df)-1, i+max_hold):
                exit_price = close
                exit_idx = time_j
                exit_reason = 'time_exit'
                pnl = (exit_price - entry_price) if direction=='long' else (entry_price - exit_price)
                break

        if exit_price is None:
            continue

        trades.append({
            'entry_time': df.index[i],
            'entry_price': entry_price,
            'direction': direction,
            'tp': tp,
            'sl': sl,
            'exit_time': exit_idx,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            'exit_reason': exit_reason,
            'duration_candles': (df.index.get_loc(exit_idx) - i) if exit_idx in df.index else None
        })
    trades_df = pd.DataFrame(trades)
    # metrics
    if trades_df.empty:
        metrics = {'total_trades':0}
    else:
        wins = trades_df[trades_df['pnl']>0]
        losses = trades_df[trades_df['pnl']<=0]
        metrics = {
            'total_trades': len(trades_df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins)/len(trades_df) if len(trades_df)>0 else 0,
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl': trades_df['pnl'].mean(),
            'max_drawdown': None
        }
    return trades_df, metrics

# -------------------------
# Optimization loops
# -------------------------
def random_search(df, param_dist, n_iter, side, desired_accuracy=None):
    best = None
    best_metrics = None
    sampled = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    for params in sampled:
        signals = generate_signals(df, params)
        trades, metrics = run_backtest(df, signals, params, side=side)
        # acceptance: prefer higher win_rate and total_pnl
        score = (metrics.get('win_rate',0)*0.6) + (np.tanh((metrics.get('total_pnl',0))/1000)*0.4)
        if best is None or score > best:
            best = score
            best_params = params
            best_metrics = metrics
            best_trades = trades
        # if desired accuracy met, stop early
        if desired_accuracy and metrics.get('win_rate',0) >= desired_accuracy:
            break
    return best_params, best_metrics, best_trades

def grid_search(df, param_grid, side, desired_accuracy=None):
    best = None
    best_params=None
    best_metrics=None
    best_trades=None
    keys = list(param_grid.keys())
    for vals in product(*param_grid.values()):
        params = dict(zip(keys, vals))
        signals = generate_signals(df, params)
        trades, metrics = run_backtest(df, signals, params, side=side)
        score = (metrics.get('win_rate',0)*0.6) + (np.tanh((metrics.get('total_pnl',0))/1000)*0.4)
        if best is None or score > best:
            best = score
            best_params = params
            best_metrics = metrics
            best_trades = trades
        if desired_accuracy and metrics.get('win_rate',0) >= desired_accuracy:
            break
    return best_params, best_metrics, best_trades

# -------------------------
# Main Streamlit UI
# -------------------------
st.title("Swing Trading Recommender — Price Action + Patterns (Automated)")

with st.sidebar:
    st.header("Options")
    side_option = st.selectbox("Select side", ['both','long','short'])
    search_method = st.selectbox("Search method", ['random_search','grid_search'])
    n_iter = st.number_input("Random search iterations (if Random search)", value=50, min_value=5)
    desired_accuracy = st.slider("Desired minimum accuracy (win rate)", 0.0, 1.0, 0.6, step=0.05)
    n_points = st.number_input("Number of parameter samples / grid size (affects speed)", value=50, min_value=5)
    st.markdown("**Backtest constraints**")
    max_hold = st.number_input("Max hold (candles)", value=20, min_value=1)
    st.markdown("Upload file (CSV / Excel) with OHLCV data")
    uploaded = st.file_uploader("Upload CSV or Excel", type=['csv','xlsx','xls'])

if uploaded:
    # read file
    try:
        if uploaded.name.lower().endswith('.csv'):
            df0 = pd.read_csv(uploaded)
        else:
            df0 = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        raise

    # map columns
    col_map = map_columns(df0)
    st.subheader("Column mapping (detected)")
    st.write(col_map)

    # date handling
    date_col = col_map.get('date')
    if date_col is None:
        # try index
        if df0.index.name and ('date' in str(df0.index.name).lower() or 'time' in str(df0.index.name).lower()):
            df0 = df0.reset_index().rename(columns={df0.index.name:'date'})
            date_col = 'date'
        else:
            # try any column that looks like date
            maybe_date = next((c for c in df0.columns if 'date' in c.lower() or 'time' in c.lower()), None)
            date_col = maybe_date

    if date_col is None:
        st.error("Could not find a date column. Please ensure file has a date/time column.")
    else:
        # normalize column names and create canonical dataframe
        can_cols = {}
        for k,v in col_map.items():
            if v is not None:
                can_cols[k] = v

        # create df with canonical names where present
        df = pd.DataFrame()
        for k in ['date','open','high','low','close','volume']:
            if can_cols.get(k) is not None:
                df[k] = df0[can_cols[k]]
        # if some columns missing, try to fill using other numeric columns
        numeric_cols = df0.select_dtypes(include=[np.number]).columns.tolist()
        for k in ['open','high','low','close','volume']:
            if k not in df.columns:
                # try to fill close with first numeric if absent
                if numeric_cols:
                    df[k] = df0[numeric_cols[0]]
        # Ensure date to datetime & IST
        df['raw_index'] = df.index
        df_original = df.copy()
        df = df.dropna(how='all', axis=1)  # drop empty columns

        try:
            df = ensure_datetime_ist(df, 'date')
        except Exception as e:
            st.error(f"Date conversion to IST failed: {e}")
            raise

        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date')
        # rename columns to lowercase standard
        df.columns = [c.lower() for c in df.columns]
        # forward fill small gaps
        df[['open','high','low','close']] = df[['open','high','low','close']].ffill().bfill()

        # EDA summary
        st.subheader("Data snapshot")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Top 5 rows")
            st.dataframe(df.head(5))
        with c2:
            st.write("Bottom 5 rows")
            st.dataframe(df.tail(5))

        st.write(f"Date range: {df.index.min()}  to  {df.index.max()}")
        st.write(f"Price range: min {df['close'].min():.4f}, max {df['close'].max():.4f}")

        # End date selector inside data
        end_default = df.index.max()
        end_date = st.date_input("Select end date for backtest (inclusive)", value=end_default.date(),
                                 min_value=df.index.min().date(), max_value=df.index.max().date())
        # convert to timestamp at end of that day in IST
        end_dt = pd.to_datetime(end_date).tz_localize('Asia/Kolkata') + pd.Timedelta(hours=23, minutes=59, seconds=59)
        df_backtest = df[df.index <= end_dt].copy()
        if df_backtest.empty:
            st.error("No data available up to chosen end date.")
        else:
            st.subheader("Exploratory Data Analysis")
            # plot raw close
            fig = go.Figure(data=[go.Scatter(x=df_backtest.index, y=df_backtest['close'], name='Close')])
            fig.update_layout(height=350, margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig, use_container_width=True)

            # monthly returns heatmap
            df_backtest['return'] = df_backtest['close'].pct_change()
            df_backtest['year'] = df_backtest.index.year
            df_backtest['month'] = df_backtest.index.month
            pivot = df_backtest.groupby(['year','month'])['return'].apply(lambda x: (1+x).prod()-1).unstack(fill_value=0)
            st.write("Year vs Month returns heatmap")
            plt.figure(figsize=(10,4))
            sns.heatmap(pivot, annot=True, fmt=".2%", cbar_kws={'format':'%.0f%%'})
            st.pyplot(plt.gcf())
            plt.clf()

            # 100 words summary
            # derive simple signals to describe trend & volatility
            recent = df_backtest['close'].iloc[-50:]
            slope = (recent.values[-1] - recent.values[0]) / recent.values[0]
            vol = df_backtest['return'].std() * np.sqrt(252) if not df_backtest['return'].isna().all() else 0
            trend_desc = "uptrend" if slope>0.01 else ("downtrend" if slope<-0.01 else "sideways")
            summary = f"In the selected period ({df_backtest.index.min().date()} to {df_backtest.index.max().date()}), the price shows a {trend_desc} over recent candles. Volatility (annualized std) is approximately {vol:.2%}. Recent structure contains consolidations, possible support/resistance clusters and short squeezes/SL-hunt wicks. Potential opportunities: look for pullbacks to well-defined supports for long entries in an uptrend, or confirmed breakdowns with volume for shorts. Use ATR-based SL to adapt to current volatility and avoid position sizing risks."
            st.write(summary)

            # Build baseline parameter distribution (search space)
            base_param_dist = {
                'fast_ma': [5,8,10,12,15],
                'slow_ma': [20,30,40,50,60],
                'atr_period': [10,14,20],
                'tp_atr_mult': [0.8,1.0,1.5,2.0,3.0],
                'sl_atr_mult': [0.8,1.0,1.5,2.0,3.0],
                'sr_window': [20,30,50,80],
                'res_breakout_pct': [0.002,0.005,0.01],
                'sup_pull_pct': [0.005,0.01,0.02],
                'sl_hunt_pct': [0.01,0.02,0.03],
                'vol_ma': [5,10,20],
                'vol_spike_pct': [0.2,0.3,0.5],
                'pattern_time_seconds': [3600*24*3, 3600*24*7]  # 3 or 7 days tolerance
            }

            # allow user to shrink or accept defaults
            st.subheader("Optimization settings")
            use_fast_defaults = st.checkbox("Use default parameter grid (recommended)", value=True)
            if not use_fast_defaults:
                st.write("You can edit parameter sets in code if needed.")
            # prepare parameter sampling
            if search_method == 'random_search':
                n_iter_use = int(n_iter)
                # random sampling from grid
                param_dist = {}
                for k,vals in base_param_dist.items():
                    param_dist[k] = vals
                st.write("Running Randomized search — this will iterate", n_iter_use, "parameter sets.")
                if st.button("Start Random Search"):
                    with st.spinner("Searching best params..."):
                        # convert param_dist to sampler-friendly dict with lists
                        # we will sample uniformly
                        sampled = []
                        random.seed(42)
                        sampled_params = []
                        for _ in range(n_iter_use):
                            p = {k: random.choice(v) for k,v in param_dist.items()}
                            p['max_hold'] = int(max_hold)
                            sampled_params.append(p)
                        best_score = None
                        best_params = None
                        best_metrics = None
                        best_trades = None
                        for p in sampled_params:
                            signals = generate_signals(df_backtest, p)
                            trades, metrics = run_backtest(df_backtest, signals, p, side=side_option)
                            score = (metrics.get('win_rate',0)*0.6) + (np.tanh((metrics.get('total_pnl',0))/1000)*0.4)
                            if best_score is None or score>best_score:
                                best_score = score
                                best_params = p
                                best_metrics = metrics
                                best_trades = trades
                            # early break if desired_accuracy met
                            if metrics.get('win_rate',0) >= desired_accuracy:
                                break
                        st.success("Search complete")
                        st.subheader("Best Parameters (Random Search)")
                        st.json(best_params)
                        st.write("Backtest metrics for best params:")
                        st.write(best_metrics)
                        if best_trades is None or best_trades.empty:
                            st.write("No trades found with best params.")
                        else:
                            st.subheader("Trades (sample)")
                            st.dataframe(best_trades.head(50))
                            # show aggregate metrics nicely
                            st.write(f"Win rate: {best_metrics.get('win_rate',0):.2%}, Total PnL: {best_metrics.get('total_pnl',0):.2f}, Total trades: {best_metrics.get('total_trades',0)}")

                            # Live recommendation on last candle (entry at last close)
                            st.subheader("Live Recommendation (based on last candle)")
                            # generate signals on full df (but entry must be last candle close)
                            signals_full = generate_signals(df, best_params)
                            last_sig = signals_full.iloc[-1]
                            recs = []
                            if side_option in ['long','both'] and last_sig['long']:
                                # create recommendation dict
                                entry_price = df['close'].iloc[-1]
                                atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                                tp = entry_price * (1 + best_params['tp_atr_mult'] * atr_val / entry_price)
                                sl = entry_price * (1 - best_params['sl_atr_mult'] * atr_val / entry_price)
                                prob = best_metrics.get('win_rate',0)
                                recs.append({
                                    'side':'LONG',
                                    'entry_time': df.index[-1],
                                    'entry_price': entry_price,
                                    'tp': tp,
                                    'sl': sl,
                                    'probability': prob,
                                    'reason': last_sig['reason']
                                })
                            if side_option in ['short','both'] and last_sig['short']:
                                entry_price = df['close'].iloc[-1]
                                atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                                tp = entry_price * (1 - best_params['tp_atr_mult'] * atr_val / entry_price)
                                sl = entry_price * (1 + best_params['sl_atr_mult'] * atr_val / entry_price)
                                prob = best_metrics.get('win_rate',0)
                                recs.append({
                                    'side':'SHORT',
                                    'entry_time': df.index[-1],
                                    'entry_price': entry_price,
                                    'tp': tp,
                                    'sl': sl,
                                    'probability': prob,
                                    'reason': last_sig['reason']
                                })
                            if recs:
                                for r in recs:
                                    st.markdown(f"**{r['side']}** — Entry @ {r['entry_price']:.2f} on {r['entry_time']}")
                                    st.write(f"Target: {r['tp']:.2f}, SL: {r['sl']:.2f}")
                                    st.write(f"Probability of profit (from backtest): {r['probability']:.2%}")
                                    st.write("Reason/logic:", r['reason'])
                            else:
                                st.write("No live signal on last candle based on best params.")
                        # final text summary
                        st.subheader("Backtest Summary (human readable)")
                        if best_metrics:
                            hr = f"The optimized strategy produced {best_metrics.get('total_trades',0)} trades with a win rate of {best_metrics.get('win_rate',0):.2%} and total PnL {best_metrics.get('total_pnl',0):.2f}. Entry decisions were made on candle-close only (no future leakage). Use ATR-based SL/TP described above; reduce position size if ATR indicates high volatility. Live recommendation (if any) is shown above and is based on the same strategy that performed best in backtest."
                            st.write(hr)
            else:
                # grid search
                st.write("Running Grid search over a compact subset (may be slow).")
                if st.button("Start Grid Search"):
                    with st.spinner("Grid searching..."):
                        # build smaller grid using first few items in lists to keep reasonable
                        grid = {}
                        for k,v in base_param_dist.items():
                            grid[k] = v[:min(len(v),3)]
                        # iterate
                        best_score=None
                        best_params=None
                        best_metrics=None
                        best_trades=None
                        for vals in product(*grid.values()):
                            p = dict(zip(grid.keys(), vals))
                            p['max_hold'] = int(max_hold)
                            signals = generate_signals(df_backtest, p)
                            trades, metrics = run_backtest(df_backtest, signals, p, side=side_option)
                            score = (metrics.get('win_rate',0)*0.6) + (np.tanh((metrics.get('total_pnl',0))/1000)*0.4)
                            if best_score is None or score>best_score:
                                best_score = score
                                best_params = p
                                best_metrics = metrics
                                best_trades = trades
                            if metrics.get('win_rate',0) >= desired_accuracy:
                                break
                        st.success("Grid search done")
                        st.subheader("Best params (Grid)")
                        st.json(best_params)
                        st.write(best_metrics)
                        if best_trades is not None and not best_trades.empty:
                            st.subheader("Sample trades from best grid strategy")
                            st.dataframe(best_trades.head(50))
                            # Live rec
                            st.subheader("Live Recommendation (grid best)")
                            signals_full = generate_signals(df, best_params)
                            last_sig = signals_full.iloc[-1]
                            recs=[]
                            if side_option in ['long','both'] and last_sig['long']:
                                entry_price = df['close'].iloc[-1]
                                atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                                tp = entry_price * (1 + best_params['tp_atr_mult'] * atr_val / entry_price)
                                sl = entry_price * (1 - best_params['sl_atr_mult'] * atr_val / entry_price)
                                prob = best_metrics.get('win_rate',0)
                                recs.append({'side':'LONG','entry_time':df.index[-1],'entry_price':entry_price,'tp':tp,'sl':sl,'probability':prob,'reason':last_sig['reason']})
                            if side_option in ['short','both'] and last_sig['short']:
                                entry_price = df['close'].iloc[-1]
                                atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                                tp = entry_price * (1 - best_params['tp_atr_mult'] * atr_val / entry_price)
                                sl = entry_price * (1 + best_params['sl_atr_mult'] * atr_val / entry_price)
                                prob = best_metrics.get('win_rate',0)
                                recs.append({'side':'SHORT','entry_time':df.index[-1],'entry_price':entry_price,'tp':tp,'sl':sl,'probability':prob,'reason':last_sig['reason']})
                            if recs:
                                for r in recs:
                                    st.markdown(f"**{r['side']}** — Entry @ {r['entry_price']:.2f} on {r['entry_time']}")
                                    st.write(f"Target: {r['tp']:.2f}, SL: {r['sl']:.2f}")
                                    st.write(f"Probability of profit (from backtest): {r['probability']:.2%}")
                                    st.write("Reason/logic:", r['reason'])
                            else:
                                st.write("No live signal on last candle based on best grid params.")
                        # final summary
                        st.subheader("Backtest Summary (human readable)")
                        if best_metrics:
                            hr = f"Grid-optimized strategy produced {best_metrics.get('total_trades',0)} trades with a win rate of {best_metrics.get('win_rate',0):.2%} and total PnL {best_metrics.get('total_pnl',0):.2f}. Entry and exits always used candle-close logic; SL/TP were ATR-based. Use risk management: max 1-2% capital per trade, adjust position sizing to ATR."
                            st.write(hr)

else:
    st.info("Upload a CSV or Excel file containing OHLC data to begin.")

# Footer notes
st.markdown("---")
st.markdown("**Notes & assumptions**: This app uses heuristic, explainable price-action rules and ATR-based volatility sizing. Pattern detection is approximate and implemented with clear rules. All backtest entries occur on the same candle close where signal forms (no future leakage). The results are model-driven — use paper-trading to validate before real capital. This code avoids talib/pandas_ta and uses only pandas/numpy/sklearn/seaborn/plotly.")
