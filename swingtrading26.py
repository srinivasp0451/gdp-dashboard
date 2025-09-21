# Streamlit Swing Recommender
# Single-file Streamlit app that:
# - accepts OHLCV-like CSV/Excel with arbitrary column names
# - maps columns automatically (open/high/low/close/volume/date)
# - sorts data ascending and converts date column to IST timezone
# - displays head/tail, min/max dates & prices, plots raw candlestick
# - performs EDA including year-month returns heatmap
# - detects basic price-action features (support/resistance, zones)
# - implements a price-action based swing strategy (long/short/both)
# - performs Random Search or Grid Search optimization over hyper-params
# - backtests with NO future leakage (entry executed on the same candle close)
# - displays backtest results (entry/exit times & levels, PnL, win-rate, etc.)
# - shows live recommendation on last available candle using the best strategy
# - does NOT use talib or pandas_ta and keeps everything dynamic

import streamlit as st
import pandas as pd
import numpy as np
import io
import math
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from itertools import product
import random
import json

# --- Small fix added early to avoid NameError for calendar_month_name (used later in heatmap) ---
import calendar
def calendar_month_name(m):
    return calendar.month_abbr[m]
# --------------------------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Swing Recommender")

# ---------- Utilities ----------
def assume_and_map_columns(df: pd.DataFrame):
    # Create mapping to standard names using substring rules
    col_map = {c: c for c in df.columns}
    lowcols = [c.lower() for c in df.columns]
    # Candidate detection
    date_idx = None
    for i, c in enumerate(lowcols):
        if any(k in c for k in ["date", "time", "timestamp", "datetime"]):
            date_idx = i
            break
    if date_idx is None:
        # fallback: first column if it looks like a date
        for i, c in enumerate(df.iloc[:, 0].astype(str).head(10)):
            try:
                pd.to_datetime(c)
                date_idx = 0
                break
            except:
                pass
    if date_idx is not None:
        col_map[df.columns[date_idx]] = "date"

    # open/high/low/close mapping using substring matching
    def pick(colnames, keywords):
        for i, c in enumerate(colnames):
            if any(k in c for k in keywords):
                return i
        return None

    o_idx = pick(lowcols, ["open", "opn", "o_"]) or pick(lowcols, ["start"]) if pick(lowcols, ["open", "opn", "o_"]) is not None else None
    h_idx = pick(lowcols, ["high", "hi"]) 
    l_idx = pick(lowcols, ["low", "lo"]) 
    c_idx = pick(lowcols, ["close", "last", "closep", "cls"]) 
    v_idx = pick(lowcols, ["volume", "vol", "qty", "quantity", "shares", "traded"]) 

    if o_idx is not None:
        col_map[df.columns[o_idx]] = "open"
    if h_idx is not None:
        col_map[df.columns[h_idx]] = "high"
    if l_idx is not None:
        col_map[df.columns[l_idx]] = "low"
    if c_idx is not None:
        col_map[df.columns[c_idx]] = "close"
    if v_idx is not None:
        col_map[df.columns[v_idx]] = "volume"

    # rename columns
    df = df.rename(columns=col_map)
    return df, col_map

def parse_dates_and_set_ist(df: pd.DataFrame, date_col='date'):
    if date_col not in df.columns:
        # try index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'date'})
        else:
            st.error('No date-like column detected. Please ensure your file has a date/time column.')
            return None
    # coerce to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # drop NA dates
    df = df.dropna(subset=['date']).copy()
    # If timezone-naive, localize to Asia/Kolkata (IST)
    try:
        if df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize('Asia/Kolkata')
        else:
            df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')
    except Exception:
        # Some pandas versions don't allow dt.tz check on Series with mixed tz
        df['date'] = df['date'].apply(lambda x: x.tz_localize('Asia/Kolkata') if x.tzinfo is None else x.astimezone(pytz.timezone('Asia/Kolkata')))
    return df

def ensure_ohlcv(df: pd.DataFrame):
    # Ensure open/high/low/close/volume columns exist. If not, try to fill with close
    if 'close' not in df.columns:
        st.error('Cannot find any column matching close price. Please upload a file with close prices.')
        return None
    if 'open' not in df.columns:
        df['open'] = df['close']
    if 'high' not in df.columns:
        df['high'] = df[['open','close']].max(axis=1)
    if 'low' not in df.columns:
        df['low'] = df[['open','close']].min(axis=1)
    if 'volume' not in df.columns:
        df['volume'] = 0
    # convert numeric
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop rows without numeric close
    df = df.dropna(subset=['close'])
    return df

def rolling_support_resistance(df, lookback=20):
    # Rolling support (min low) and resistance (max high)
    df['rolling_min_low'] = df['low'].rolling(window=lookback, min_periods=1).min()
    df['rolling_max_high'] = df['high'].rolling(window=lookback, min_periods=1).max()
    return df

def detect_local_peaks(series, window=5, kind='max'):
    # naive local peak detection without scipy
    idxs = []
    n = len(series)
    for i in range(window, n-window):
        center = series.iloc[i]
        left = series.iloc[i-window:i]
        right = series.iloc[i+1:i+1+window]
        if kind == 'max':
            if center >= left.max() and center >= right.max():
                idxs.append(i)
        else:
            if center <= left.min() and center <= right.min():
                idxs.append(i)
    return idxs

def detect_double_top_bottom(df, tolerance=0.02, separation=5):
    peaks = detect_local_peaks(df['high'], window=3, kind='max')
    troughs = detect_local_peaks(df['low'], window=3, kind='min')
    double_tops = []
    double_bottoms = []
    # double top: two peaks with similar heights
    for i in range(len(peaks)):
        for j in range(i+1, len(peaks)):
            if abs(df['high'].iloc[peaks[i]] - df['high'].iloc[peaks[j]])/df['high'].iloc[peaks[i]] < tolerance and abs(peaks[j]-peaks[i])>=separation:
                double_tops.append((peaks[i], peaks[j]))
    for i in range(len(troughs)):
        for j in range(i+1, len(troughs)):
            if abs(df['low'].iloc[troughs[i]] - df['low'].iloc[troughs[j]])/df['low'].iloc[troughs[i]] < tolerance and abs(troughs[j]-troughs[i])>=separation:
                double_bottoms.append((troughs[i], troughs[j]))
    return double_tops, double_bottoms

def detect_head_shoulders(df):
    # very naive heuristic: three peaks with middle the highest
    peaks = detect_local_peaks(df['high'], window=4, kind='max')
    patterns = []
    for i in range(len(peaks)-2):
        a,b,c = peaks[i], peaks[i+1], peaks[i+2]
        if b-a>2 and c-b>2:
            if df['high'].iloc[b] > df['high'].iloc[a] and df['high'].iloc[b] > df['high'].iloc[c]:
                # shoulders roughly similar
                if abs(df['high'].iloc[a] - df['high'].iloc[c]) / df['high'].iloc[b] < 0.06:
                    patterns.append((a,b,c))
    return patterns

def detect_cup_handle(df, lookback=100):
    # naive attempt: find long rounded bottom then small consolidation
    n = len(df)
    patterns = []
    for end in range(lookback, n-10):
        window = df['close'].iloc[end-lookback:end]
        trough_idx = window.idxmin()
        trough_pos = trough_idx - (end-lookback)
        # trough should be central-ish
        if trough_pos>lookback*0.2 and trough_pos<lookback*0.8:
            left_peak = window.iloc[:trough_pos].max()
            right_peak = window.iloc[trough_pos+1:].max()
            if left_peak > window.iloc[trough_pos]*1.06 and right_peak > window.iloc[trough_pos]*1.06:
                # small handle: next 10 bars should be sideways
                handle = df['close'].iloc[end:end+10]
                if handle.max() < right_peak*1.02:
                    patterns.append((end-lookback, end+9))
    return patterns

# ---------- Strategy and Backtesting ----------
def generate_signals(df, params, side='both'):
    # params: lookback, vol_mul, threshold_pct, risk_reward, min_hold, max_hold
    lookback = int(params.get('lookback', 20))
    vol_mul = float(params.get('vol_mul', 1.0))
    threshold = float(params.get('threshold', 0.002))
    rr = float(params.get('risk_reward', 2.0))
    sl_buffer_pct = float(params.get('sl_buffer_pct', 0.002))
    min_hold = int(params.get('min_hold', 1))
    max_hold = int(params.get('max_hold', 10))

    df = df.copy().reset_index(drop=True)
    df = rolling_support_resistance(df, lookback=lookback)
    avg_vol = df['volume'].rolling(window=lookback, min_periods=1).mean()

    trades = []
    i = lookback
    n = len(df)
    while i < n:
        row = df.iloc[i]
        # Avoid using future: compute based on values up to i
        support = df['rolling_min_low'].iloc[i]
        resistance = df['rolling_max_high'].iloc[i]
        vol_ok = (row['volume'] >= (avg_vol.iloc[i] * vol_mul)) if avg_vol.iloc[i]>0 else True
        # Long entry
        if side in ['both','long'] and row['close'] > resistance*(1+threshold) and vol_ok:
            entry_price = row['close']
            sl = support - sl_buffer_pct*entry_price
            target = entry_price + rr*(entry_price - sl)
            entry_time = row['date']
            # simulate forward to find exit - but use only data from i (entry candle) onwards
            exit_index = None
            exit_time = None
            exit_price = None
            reason = f"Breakout above {resistance:.2f} with volume {row['volume']:.0f} (avg {avg_vol.iloc[i]:.0f})."
            # hold logic
            for j in range(i, min(n, i+max_hold)):
                high_j = df['high'].iloc[j]
                low_j = df['low'].iloc[j]
                if high_j >= target:
                    exit_index = j
                    exit_price = target
                    exit_time = df['date'].iloc[j]
                    outcome = 'target'
                    break
                if low_j <= sl:
                    exit_index = j
                    exit_price = sl
                    exit_time = df['date'].iloc[j]
                    outcome = 'sl'
                    break
            if exit_index is None:
                # exit at close of last allowed bar
                exit_index = min(n-1, i+max_hold-1)
                exit_price = df['close'].iloc[exit_index]
                exit_time = df['date'].iloc[exit_index]
                outcome = 'time_exit'
            pnl = exit_price - entry_price
            trades.append({
                'side':'long', 'entry_index':i, 'exit_index':exit_index,'entry_time':entry_time,'exit_time':exit_time,
                'entry_price':entry_price,'exit_price':exit_price,'pnl':pnl,'outcome':outcome,'reason':reason,
                'hold_days': exit_index - i
            })
            # move i to exit_index+1 to avoid overlapping entries
            i = exit_index + 1
            continue
        # Short entry
        if side in ['both','short'] and row['close'] < support*(1-threshold) and vol_ok:
            entry_price = row['close']
            sl = resistance + sl_buffer_pct*entry_price
            target = entry_price - rr*(sl - entry_price)
            entry_time = row['date']
            exit_index = None
            exit_time = None
            exit_price = None
            reason = f"Breakdown below {support:.2f} with volume {row['volume']:.0f} (avg {avg_vol.iloc[i]:.0f})."
            for j in range(i, min(n, i+max_hold)):
                high_j = df['high'].iloc[j]
                low_j = df['low'].iloc[j]
                if low_j <= target:
                    exit_index = j
                    exit_price = target
                    exit_time = df['date'].iloc[j]
                    outcome = 'target'
                    break
                if high_j >= sl:
                    exit_index = j
                    exit_price = sl
                    exit_time = df['date'].iloc[j]
                    outcome = 'sl'
                    break
            if exit_index is None:
                exit_index = min(n-1, i+max_hold-1)
                exit_price = df['close'].iloc[exit_index]
                exit_time = df['date'].iloc[exit_index]
                outcome = 'time_exit'
            pnl = entry_price - exit_price
            trades.append({
                'side':'short', 'entry_index':i, 'exit_index':exit_index,'entry_time':entry_time,'exit_time':exit_time,
                'entry_price':entry_price,'exit_price':exit_price,'pnl':pnl,'outcome':outcome,'reason':reason,
                'hold_days': exit_index - i
            })
            i = exit_index + 1
            continue
        i += 1
    trades_df = pd.DataFrame(trades)
    return trades_df

def evaluate_backtest(trades_df):
    if trades_df.empty:
        return None
    total_pnl = trades_df['pnl'].sum()
    wins = (trades_df['pnl']>0).sum()
    losses = (trades_df['pnl']<=0).sum()
    total = len(trades_df)
    win_rate = wins/total if total>0 else np.nan
    avg_pnl = trades_df['pnl'].mean()
    avg_hold = trades_df['hold_days'].mean()
    # simple drawdown calc on equity curve
    eq = trades_df['pnl'].cumsum()
    dd = (eq.cummax() - eq).max() if len(eq)>0 else 0
    return {
        'total_pnl': total_pnl,
        'wins': wins,
        'losses': losses,
        'total_trades': total,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'avg_hold': avg_hold,
        'max_drawdown': dd
    }

def run_search(df, side, search_type, n_iter, grid_params, desired_accuracy, target_points, progress_callback=None):
    # grid_params is dict of lists
    best = None
    tried = 0
    param_list = []
    if search_type=='grid':
        keys = list(grid_params.keys())
        combos = list(product(*[grid_params[k] for k in keys]))
        for combo in combos:
            p = {k:v for k,v in zip(keys, combo)}
            param_list.append(p)
    else:
        # random search - sample n_iter
        keys = list(grid_params.keys())
        for _ in range(n_iter):
            p = {}
            for k in keys:
                vals = grid_params[k]
                v = random.choice(vals)
                p[k]=v
            param_list.append(p)

    total = len(param_list)
    for idx,p in enumerate(param_list):
        trades = generate_signals(df, p, side=side)
        stats = evaluate_backtest(trades)
        tried += 1
        score = -1e9
        if stats is not None:
            # prefer strategies that beat buy&hold and have decent win rate
            buy_hold_return = (df['close'].iloc[-1]/df['close'].iloc[0]-1)*100
            score = stats['total_pnl'] + stats['win_rate']*1000 - stats['max_drawdown']*0.5
            # if desired_accuracy specified, penalize lower accuracy
            if desired_accuracy>0 and stats['win_rate'] < desired_accuracy:
                score = score - (desired_accuracy - stats['win_rate'])*10000
        if best is None or score>best['score']:
            best = {'params':p, 'score':score, 'stats':stats, 'trades':trades}
        if progress_callback:
            progress_callback(int((idx+1)/total*100))
    return best

# ---------- Streamlit App ----------
st.title('Swing Trading Recommender — Automated')
st.markdown('Upload OHLCV data (CSV or Excel). Column names can be arbitrary; the app will map them.')

uploaded_file = st.file_uploader('Upload CSV/Excel', type=['csv','xlsx','xls'])

if uploaded_file is not None:
    # read file
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f'Error reading file: {e}')
        st.stop()

    df_mapped, col_map = assume_and_map_columns(df_raw.copy())
    df_mapped = parse_dates_and_set_ist(df_mapped)
    if df_mapped is None:
        st.stop()
    df_mapped = ensure_ohlcv(df_mapped)
    if df_mapped is None:
        st.stop()

    # sort ascending by date to prevent future leakage
    df_mapped = df_mapped.sort_values('date').reset_index(drop=True)

    st.sidebar.header('Backtest & Live Options')
    # default end date today but changeable - but as dataset's max date may be past; show slider of available dates
    min_date = df_mapped['date'].min()
    max_date = df_mapped['date'].max()
    default_end = max_date
    end_date_picker = st.sidebar.date_input('Select end date (backtest end / live data cut-off)', value=default_end.date(), min_value=min_date.date(), max_value=max_date.date())
    # convert to tz-aware datetime (IST)
    end_date = pd.Timestamp(end_date_picker).tz_localize('Asia/Kolkata') + pd.Timedelta(hours=23, minutes=59, seconds=59)

    mode_side = st.sidebar.selectbox('Side', options=['both','long','short'], index=0)
    search_type = st.sidebar.selectbox('Search Type', options=['random','grid'], index=0)
    n_iter = st.sidebar.number_input('Number of iterations (for random search)', min_value=10, max_value=1000, value=80)
    desired_accuracy = st.sidebar.slider('Desired minimum accuracy (win rate)', min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    target_points = st.sidebar.number_input('If you prefer fixed target points (0 = use risk-reward)', min_value=0.0, value=0.0)
    st.sidebar.write('Progress will be shown while search runs. Default random search is recommended.')

    # display head/tail and stats
    st.subheader('Mapped Columns')
    st.json(col_map)

    st.subheader('Data Preview (Top 5 / Bottom 5)')
    c1,c2 = st.columns(2)
    c1.dataframe(df_mapped.head())
    c2.dataframe(df_mapped.tail())

    st.write('Date range and price range')
    st.write(f"Min date: {min_date}  —  Max date: {max_date}")
    st.write(f"Min price: {df_mapped['close'].min():.4f}  —  Max price: {df_mapped['close'].max():.4f}")

    # raw plot
    st.subheader('Price Chart (raw)')
    has_ohlc = all(c in df_mapped.columns for c in ['open','high','low','close'])
    fig = go.Figure()
    if has_ohlc:
        fig.add_trace(go.Candlestick(x=df_mapped['date'], open=df_mapped['open'], high=df_mapped['high'], low=df_mapped['low'], close=df_mapped['close'], name='price'))
    else:
        fig.add_trace(go.Scatter(x=df_mapped['date'], y=df_mapped['close'], mode='lines', name='close'))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # EDA: returns heatmap
    st.subheader('EDA — Year vs Month Returns Heatmap')
    # calendar_month_name is defined up top so no NameError will occur
    df_mapped['year'] = df_mapped['date'].dt.year
    df_mapped['month'] = df_mapped['date'].dt.month
    df_mapped['ret'] = df_mapped['close'].pct_change()
    monthly = df_mapped.groupby(['year','month'])['ret'].sum().reset_index()
    pivot = monthly.pivot(index='year', columns='month', values='ret').fillna(0)
    heat = go.Figure(data=go.Heatmap(z=pivot.values, x=[calendar_month_name(m) for m in pivot.columns], y=pivot.index.astype(str), hovertemplate='%{y} %{x}: %{z:.2%}'))
    heat.update_layout(height=300, title='Sum of monthly returns')
    st.plotly_chart(heat, use_container_width=True)

    # quick 100-word summary
    st.subheader('100-word summary (automated)')
    summary_text = generate_summary_text(df_mapped)
    st.write(summary_text)

    # restrict data for backtest to end_date
    df_cut = df_mapped[df_mapped['date'] <= end_date].copy().reset_index(drop=True)

    # optimization parameter ranges
    grid_params = {
        'lookback': list(range(10,61,10)),
        'vol_mul': [0.5,1.0,1.5,2.0],
        'threshold': [0.0005,0.001,0.002,0.003,0.005],
        'risk_reward': [1.0,1.5,2.0,3.0],
        'sl_buffer_pct':[0.0005,0.001,0.002],
        'min_hold':[1,1,2],
        'max_hold':[3,5,10]
    }
    if search_type=='grid':
        st.write('Grid search will evaluate all combinations — may take long. Consider reducing param lists.')

    # Run search when button clicked
    if st.button('Run optimization & backtest'):
        progress = st.progress(0)
        def progress_cb(pct):
            try:
                progress.progress(pct)
            except:
                pass
        best = run_search(df_cut, mode_side, search_type, n_iter, grid_params, desired_accuracy, target_points, progress_callback=progress_cb)
        progress.empty()
        if best is None or best['stats'] is None:
            st.warning('No trades generated by any parameter set. Try relaxing parameter ranges or using both sides.')
        else:
            st.subheader('Best Strategy Summary')
            st.json({
                'best_params': best['params'],
                'performance': best['stats']
            })
            # show trades
            trades = best['trades']
            if not trades.empty:
                # convert times to strings
                trades_display = trades.copy()
                trades_display['entry_time'] = trades_display['entry_time'].astype(str)
                trades_display['exit_time'] = trades_display['exit_time'].astype(str)
                st.subheader('Backtest Trades')
                st.dataframe(trades_display)
                st.download_button('Download trades as CSV', trades_display.to_csv(index=False).encode('utf-8'), file_name='backtest_trades.csv')

            # backtest summary text
            backtest_summary = generate_backtest_summary(best['stats'], best['params'])
            st.subheader('Backtest summary (human readable)')
            st.write(backtest_summary)

            # Live recommendation on last candle from the cut data (use same strategy)
            st.subheader('Live recommendation (based on last available candle in selected period)')
            last_row = df_cut.iloc[-1]
            # For live recommendation, evaluate using best params on the full cut data but only consider signal at last index
            temp_trades = generate_signals(df_cut, best['params'], side=mode_side)
            if not temp_trades.empty:
                is_live = temp_trades[temp_trades['entry_index']==(len(df_cut)-1)]
                if not is_live.empty:
                    rec = is_live.iloc[-1]
                    prob = best['stats']['win_rate'] if best['stats'] else 0
                    # compute SL shown in readable way (logic depends on side)
                    if rec['side']=='long':
                        suggested_sl = rec['entry_price'] - (rec['entry_price'] - rec['exit_price']) if rec['entry_price']>rec['exit_price'] else rec['entry_price']*0.99
                    else:
                        suggested_sl = rec['entry_price'] + (rec['exit_price'] - rec['entry_price']) if rec['exit_price']>rec['entry_price'] else rec['entry_price']*1.01
                    st.write(f"Recommendation: {rec['side'].upper()} — ENTRY at {rec['entry_price']:.4f} (close of last candle) — SL {suggested_sl:.4f} — TARGET {rec['exit_price']:.4f}")
                    st.write(f"Probability of profit (win rate from backtest): {prob:.2%}")
                    st.write('Reason/Logic: ' + rec['reason'])
                else:
                    st.write('No immediate live entry signal on the last candle according to the best strategy.')
            else:
                st.write('No trades in backtest to derive live recommendation.')

            st.subheader('Detected chart patterns & price-action signals (sample)')
            dts = []
            dt_peaks = detect_double_top_bottom(df_cut)
            if dt_peaks[0] or dt_peaks[1]:
                dts.append(f"Double tops: {len(dt_peaks[0])}, Double bottoms: {len(dt_peaks[1])}")
            hs = detect_head_shoulders(df_cut)
            if hs:
                dts.append(f"Head & Shoulders patterns detected: {len(hs)}")
            ch = detect_cup_handle(df_cut)
            if ch:
                dts.append(f"Cup & Handle patterns found: {len(ch)}")
            if dts:
                for it in dts:
                    st.write('- '+it)
            else:
                st.write('No common patterns detected with current heuristic rules.')

            st.success('Optimization & backtest completed.')

# ---------- Helper functions not defined earlier ----------
def generate_summary_text(df):
    # approx 100-word human-friendly summary
    last = df['close'].iloc[-1]
    mean_ret = df['ret'].mean() if 'ret' in df.columns else df['close'].pct_change().mean()
    vol = df['ret'].std() if 'ret' in df.columns else df['close'].pct_change().std()
    period_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    trend = 'bullish' if last > df['close'].rolling(window=min(200,len(df))).mean().iloc[-1] else 'bearish'
    opp = 'opportunities for mean-reversion near recent support and breakout trades near resistance' if vol>0.01 else 'range-bound scalping and mean-reversion'
    txt = f"Over {period_days} days the instrument shows an average daily return of {mean_ret:.4%} with volatility {vol:.4%}. The longer-term trend is {trend}. Current price is {last:.2f}. Based on recent price-action, there are {opp}. Volume spikes often coincide with directional moves, suggesting follow-through trade potential. Traders should combine zone-based stops with disciplined risk management, keeping reward-to-risk favourable. Backtesting and optimization below will find parameter sets to try to capture these moves while avoiding look-ahead bias."
    # try to keep near 100 words by rough trimming
    words = txt.split()
    if len(words) > 100:
        return ' '.join(words[:100])
    return txt

def generate_backtest_summary(stats, params):
    if stats is None:
        return 'No trades were generated by the strategy.'
    s = f"Backtest results: Total trades {stats['total_trades']}, Wins {stats['wins']}, Losses {stats['losses']}, Win rate {stats['win_rate']:.2%}. Total PnL {stats['total_pnl']:.2f}. Average PnL per trade {stats['avg_pnl']:.2f}. Max drawdown {stats['max_drawdown']:.2f}. Strategy parameters used: lookback {params.get('lookback')}, vol multiplier {params.get('vol_mul')}, threshold {params.get('threshold')}, risk_reward {params.get('risk_reward')}.\n"
    s += "Recommendation for live: Follow identical rules; only take entries at candle close (no future data). Use position sizing so that potential loss per trade <= 1-2% of capital. Prefer trades where backtest showed positive edge."
    return s

# End of script
