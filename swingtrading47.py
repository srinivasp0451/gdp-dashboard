# Streamlit Algorithmic Trading Application
# Professional-grade: multi-timeframe, indicators, support/resistance, ratio analysis,
# simple Elliott-wave heuristic, backtest & optimization, IST timezone handling.
# Requirements: pandas, numpy, yfinance, matplotlib, plotly, scipy
# Optional: pip install ta (pure-python) for extra indicators.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import random
from datetime import datetime, timedelta
import pytz
from scipy.signal import argrelextrema
from scipy.stats import zscore
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------------------- Helpers: Indicators -----------------------------
import traceback
import inspect

# --- Logging helpers (prints into Streamlit UI for quick debugging with line numbers)
import streamlit as st

def log(msg):
    try:
        st.write(f"🟦 LOG: {msg}")
    except Exception:
        print(f"LOG: {msg}")

def log_shape(df, name="DF"):
    try:
        st.write(f"📏 {name} shape → Rows: {df.shape[0]}, Cols: {df.shape[1]}")
        # limit columns shown to first 30 to avoid UI overload
        cols = list(df.columns)[:30]
        st.write(f"Columns (first up to 30): {cols}")
    except Exception:
        st.write(f"❌ Could not read shape for {name}")

def log_error(e, note=""):
    try:
        # pick the last trace that is not inside this helper
        tb = traceback.format_exc()
        frame = inspect.trace()[-1]
        line = frame.lineno
        file = frame.filename
        st.error(f"🔥 **ERROR* ➤ Message: `{str(e)}` ➤ Note: {note} ➤ File: {file} ➤ Line: {line}")
        st.code(tb)
    except Exception:
        print("Logging failed", e)

# end logging helpers


def to_ist(df):
    # Convert index (DatetimeIndex) to Asia/Kolkata timezone and return with timezone-naive localized to IST
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # if tz-aware convert to Asia/Kolkata
    try:
        if df.index.tz is None:
            df = df.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            df = df.tz_convert('Asia/Kolkata')
    except Exception:
        try:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        except Exception:
            df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('Asia/Kolkata')
    # drop tz info to make index naive but IST-based
    df.index = df.index.tz_localize(None)
    return df


def flatten_columns(df, log_name=None):
    """Safely flatten MultiIndex columns and log shapes.
    Use log_name for descriptive logging (e.g., ticker name).
    """
    try:
        if log_name:
            log(f"Flattening columns for {log_name}")
        log_shape(df, f"{log_name or 'DF'} BEFORE flatten")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(c) for c in col if c not in ("", None)]).strip("_")
                for col in df.columns.values
            ]
        else:
            df.columns = df.columns.astype(str)

        log_shape(df, f"{log_name or 'DF'} AFTER flatten")
    except Exception as e:
        log_error(e, f"Error inside flatten_columns for {log_name}")
    return df


def sma(series, window):
    return series.rolling(window).mean()


def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    r = 100 - (100 / (1 + rs))
    return r


def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series, window=20, numsd=2):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + numsd * sd
    lower = ma - numsd * sd
    return upper, ma, lower


def atr(df, n=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    return atr


def adx(df, n=14):
    # Basic ADX implementation
    high = df['High']
    low = df['Low']
    close = df['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/n, adjust=False).mean() / atr_val)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/n, adjust=False).mean() / atr_val)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = pd.Series(dx).ewm(alpha=1/n, adjust=False).mean()
    plus_di.index = df.index
    minus_di.index = df.index
    adx_val.index = df.index
    return plus_di, minus_di, adx_val

# ----------------------------- Support & Resistance -----------------------------

def detect_pivots(series, order=5):
    # local minima/maxima
    minima = argrelextrema(series.values, np.less_equal, order=order)[0]
    maxima = argrelextrema(series.values, np.greater_equal, order=order)[0]
    return minima, maxima


def cluster_levels(levels, threshold=0.5):
    # cluster nearby levels (threshold in price units)
    if len(levels) == 0:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lv in levels[1:]:
        if abs(lv - clusters[-1][-1]) <= threshold:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    return [np.mean(c) for c in clusters]


def support_resistance(df, hits_threshold=3, pivot_order=5, cluster_threshold=None):
    close = df['Close']
    minima, maxima = detect_pivots(close, order=pivot_order)
    supps = close.iloc[minima].values.tolist()
    resis = close.iloc[maxima].values.tolist()
    if cluster_threshold is None:
        # automatic threshold as 0.5% of price
        cluster_threshold = df['Close'].mean() * 0.005
    supp_levels = cluster_levels(supps, threshold=cluster_threshold)
    res_levels = cluster_levels(resis, threshold=cluster_threshold)
    # Count hits within tolerance
    def count_hits(levels):
        hits = {}
        for level in levels:
            tol = cluster_threshold
            hit_mask = (df['Low'] <= level + tol) & (df['High'] >= level - tol)
            hits[level] = hit_mask.sum()
        return hits
    supp_hits = count_hits(supp_levels)
    res_hits = count_hits(res_levels)
    # Filter by hits threshold
    strong_supports = {k: v for k, v in supp_hits.items() if v >= hits_threshold}
    strong_res = {k: v for k, v in res_hits.items() if v >= hits_threshold}
    return strong_supports, strong_res

# ----------------------------- Simple Elliott-wave heuristic -----------------------------

def swing_points(series, window=5):
    # detect swings
    minima, maxima = detect_pivots(series, order=window)
    points = []
    for i in maxima:
        points.append((i, series.iloc[i], 'peak'))
    for i in minima:
        points.append((i, series.iloc[i], 'trough'))
    points_sorted = sorted(points, key=lambda x: x[0])
    return points_sorted


def elliott_heuristic(df):
    # Very approximate: look for 5 alternating swings (peak/trough) forming impulse
    sp = swing_points(df['Close'], window=5)
    labels = []
    for i in range(len(sp) - 4):
        segment = sp[i:i+5]
        types = [s[2] for s in segment]
        # Expect pattern: peak, trough, peak, trough, peak or inverse
        if types == ['peak', 'trough', 'peak', 'trough', 'peak'] or types == ['trough', 'peak', 'trough', 'peak', 'trough']:
            indices = [s[0] for s in segment]
            prices = [s[1] for s in segment]
            labels.append({'start_idx': indices[0], 'end_idx': indices[-1], 'prices': prices, 'indices': indices})
    # Return the most recent label if exists
    if labels:
        last = labels[-1]
        return last
    return None

# ----------------------------- Data Fetching & Ratio handling -----------------------------

def fetch_yf(ticker, period, interval, log_name=None):
    # Use delays to respect rate-limits and robust logging
    try:
        log(f"Fetching {ticker} | period={period} | interval={interval}")
        time.sleep(random.uniform(1.5, 3.0))
        data = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
        if data is None or data.empty:
            log(f"No data returned for {ticker}")
            return pd.DataFrame()
        # Flatten safely and log
        data = flatten_columns(data, log_name=ticker)
        data.dropna(how='all', inplace=True)
        if data.empty:
            return data
        # Standardize column names if lowercase columns
        colmap = {c: c.title() for c in data.columns}
        data.rename(columns=colmap, inplace=True)
        # Ensure required cols
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c not in data.columns:
                data[c] = np.nan
        data = to_ist(data)
        return data
    except Exception as e:
        log_error(e, f"Fetching failed for {ticker}")
        return pd.DataFrame()
    # Standardize column names if lowercase columns
    colmap = {c: c.title() for c in data.columns}
    data.rename(columns=colmap, inplace=True)
    # Ensure required cols
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in data.columns:
            data[c] = np.nan
    data = to_ist(data)
    return data


def align_for_ratio(df1, df2, log_name1=None, log_name2=None):
    # Align two dataframes on index by inner join after resampling to common frequency if needed
    try:
        log(f"Aligning for ratio: {log_name1 or 'DF1'} vs {log_name2 or 'DF2'}")
        log_shape(df1, f"{log_name1 or 'DF1'} BEFORE align")
        log_shape(df2, f"{log_name2 or 'DF2'} BEFORE align")

        df1 = df1.sort_index()
        df2 = df2.sort_index()

        common_idx = df1.index.intersection(df2.index)
        if len(common_idx) == 0:
            # try resampling both to a coarser frequency (e.g., 1min -> 5min) as fallback
            log("No common index — attempting resample fallback to 1min frequency union")
            # build union and forward/backfill cautiously
            union_idx = df1.index.union(df2.index).sort_values()
            a = df1.reindex(union_idx).ffill().bfill()
            b = df2.reindex(union_idx).ffill().bfill()
            # after filling, intersect again
            common_idx = a.index.intersection(b.index)
            a = a.loc[common_idx]
            b = b.loc[common_idx]
            log_shape(a, "DF1 after resample-fallback")
            log_shape(b, "DF2 after resample-fallback")
            return a, b
        a = df1.reindex(common_idx)
        b = df2.reindex(common_idx)
        log_shape(a, f"{log_name1 or 'DF1'} AFTER align")
        log_shape(b, f"{log_name2 or 'DF2'} AFTER align")
        return a, b
    except Exception as e:
        log_error(e, "Ratio alignment failed")
        return pd.DataFrame(), pd.DataFrame()

# ----------------------------- Analysis Engine -----------------------------

def compute_indicators(df):
    df = df.copy()
    df['RSI'] = rsi(df['Close'], 14)
    df['EMA9'] = ema(df['Close'], 9)
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['SMA200'] = sma(df['Close'], 200)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(df['Close'])
    df['BB_up'], df['BB_mid'], df['BB_low'] = bollinger_bands(df['Close'], 20, 2)
    df['ATR'] = atr(df, 14)
    df['plusDI'], df['minusDI'], df['ADX'] = adx(df, 14)
    df['LogReturn'] = np.log(df['Close']).diff()
    df['Volatility'] = df['LogReturn'].rolling(window=20).std() * np.sqrt(252)
    return df


def generate_signals(df):
    # Multi-timeframe checklist style signals
    s = {}
    close = df['Close']
    s['above_20ema'] = close.iloc[-1] > df['EMA20'].iloc[-1]
    s['adx_trend'] = df['ADX'].iloc[-1] > 25
    s['rsi'] = df['RSI'].iloc[-1]
    s['macd_bull'] = df['MACD_Hist'].iloc[-1] > 0
    # Divergence (simple): price making lower low, RSI making higher low in last 5 points
    if len(close) >= 6:
        price_ll = close.iloc[-6:-1].min()
        rsi_section = df['RSI'].iloc[-6:-1]
        rsi_ll = rsi_section.min()
        s['rsi_bull_div'] = (close.iloc[-1] < price_ll) and (df['RSI'].iloc[-1] > rsi_ll)
    else:
        s['rsi_bull_div'] = False
    return s

# ----------------------------- Backtesting & Optimization -----------------------------

def backtest_simple_cross(df, short_ema=9, long_ema=20, sl_atr=1.5, tp_mul=2.0):
    data = df.copy()
    data['short'] = ema(data['Close'], short_ema)
    data['long'] = ema(data['Close'], long_ema)
    data['position'] = 0
    data['position'] = np.where(data['short'] > data['long'], 1, 0)
    data['signal'] = data['position'].diff()
    cash = 100000
    pos = 0
    entry_price = 0
    trades = []
    for idx, row in data.iloc[1:].iterrows():
        if row['signal'] == 1 and pos == 0:
            pos = cash / row['Close']
            entry_price = row['Close']
            stop = entry_price - sl_atr * row['ATR']
            target = entry_price + tp_mul * row['ATR']
            trades.append({'entry_dt': idx, 'entry': entry_price, 'stop': stop, 'target': target, 'exit_dt': None, 'exit': None})
        elif pos > 0:
            # check stop/target
            trade = trades[-1]
            if row['Low'] <= trade['stop']:
                trades[-1]['exit_dt'] = idx
                trades[-1]['exit'] = trade['stop']
                pos = 0
            elif row['High'] >= trade['target']:
                trades[-1]['exit_dt'] = idx
                trades[-1]['exit'] = trade['target']
                pos = 0
    # compute performance
    results = []
    for t in trades:
        if t['exit'] is None:
            continue
        pnl = (t['exit'] - t['entry']) * (cash / t['entry'])
        results.append(pnl)
    total = sum(results)
    return {'trades': trades, 'total_pnl': total, 'n_trades': len(results)}


def optimize_params(df, param_grid):
    best = None
    for short in param_grid['short']:
        for long in param_grid['long']:
            if short >= long: continue
            res = backtest_simple_cross(df, short, long, sl_atr=param_grid.get('sl_atr',1.5), tp_mul=param_grid.get('tp_mul',2.0))
            if best is None or res['total_pnl'] > best['total_pnl']:
                best = {'short': short, 'long': long, **res}
    return best

# ----------------------------- Summary Generator -----------------------------

def generate_summary(df, signals, supports, resistances, elliot, ratio_info=None):
    # Create about 100-word professional summary.
    parts = []
    price = df['Close'].iloc[-1]
    trend = 'bullish' if signals['above_20ema'] and signals['macd_bull'] and signals['adx_trend'] else ('bearish' if not signals['above_20ema'] and not signals['macd_bull'] else 'neutral')
    parts.append(f"Current price {price:.2f} - trend: {trend}.")
    if signals['rsi_bull_div']:
        parts.append("RSI bullish divergence detected on selected timeframe.")
    if supports:
        s_levels = sorted(supports.items(), key=lambda x: -x[1])[:2]
        parts.append(f"Strong support at {', '.join([f'{k:.0f}' for k,_ in s_levels])} (hits recorded).")
    if resistances:
        r_levels = sorted(resistances.items(), key=lambda x: -x[1])[:2]
        parts.append(f"Strong resistance at {', '.join([f'{k:.0f}' for k,_ in r_levels])}.")
    if elliot:
        start_price = elliot['prices'][0]
        end_price = elliot['prices'][-1]
        parts.append(f"Recent Elliott-like sequence from {start_price:.0f} to {end_price:.0f} observed.")
    # Concise recommendation
    rec = "Hold"
    if trend == 'bullish' and not signals.get('rsi_bull_div', False):
        rec = f"Buy (entry ~{price:.0f}, SL {price - df['ATR'].iloc[-1]:.0f}, target {price + 2*df['ATR'].iloc[-1]:.0f})"
    elif trend == 'bearish':
        rec = f"Sell (entry ~{price:.0f}, SL {price + df['ATR'].iloc[-1]:.0f}, target {price - 2*df['ATR'].iloc[-1]:.0f})"
    parts.append(f"Recommendation: {rec}.")
    summary = ' '.join(parts)
    # trim to ~100 words
    words = summary.split()
    if len(words) > 110:
        summary = ' '.join(words[:110]) + '...'
    return summary

# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title='Algo Trading Dashboard', layout='wide')
st.title('Professional Algorithmic Trading Dashboard')

# Sidebar controls
with st.sidebar:
    st.header('Data & Settings')
    ticker1 = st.text_input('Ticker 1 (yfinance)', value='^NSEI')
    ticker2 = st.text_input('Ticker 2 (optional)', value='')
    period = st.selectbox('Period', ['1d','5d','7d','1mo','3mo','6mo','1y','2y','3y','5y','6y','10y','15y','20y','25y','30y'], index=6)
    interval = st.selectbox('Interval', ['1m','3m','5m','10m','15m','30m','60m','120m','240m','1d'], index=9)
    enable_ratio = st.checkbox('Enable Ratio Analysis', value=False if ticker2=='' else True)
    hits_threshold = st.number_input('Support/Resistance hits threshold', min_value=1, max_value=50, value=3)
    pivot_order = st.slider('Pivot detection lookback (order)', 1, 20, 5)
    fetch_button = st.button('Fetch Data')
    st.markdown('---')
    st.write('Backtest & Optimization')
    run_backtest = st.button('Run Backtest')
    run_opt = st.button('Optimize Strategy')

# Persist data in session_state to prevent UI disappearing
if 'df1' not in st.session_state:
    st.session_state['df1'] = None
if 'df2' not in st.session_state:
    st.session_state['df2'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = ''

# Fetching
if fetch_button:
    try:
        with st.spinner('Fetching ticker 1...'):
            df1 = fetch_yf(ticker1, period=period, interval=interval)
            if df1.empty:
                st.error('No data for ticker 1. Check ticker symbol and interval/period combination.')
            else:
                df1 = compute_indicators(df1)
                st.session_state['df1'] = df1
        if enable_ratio and ticker2.strip() != '':
            with st.spinner('Fetching ticker 2...'):
                df2 = fetch_yf(ticker2, period=period, interval=interval)
                if df2.empty:
                    st.error('No data for ticker 2. Check ticker symbol.')
                else:
                    df2 = compute_indicators(df2)
                    st.session_state['df2'] = df2
        st.success('Data fetched and indicators calculated.')
    except Exception as e:
        st.error(f'Error fetching data: {e}')

# Show Data & Analysis if available
if st.session_state['df1'] is not None:
    df = st.session_state['df1']
    st.subheader(f'{ticker1} — Latest')
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        latest_price = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2] if len(df)>=2 else latest_price
        pct = (latest_price - prev)/prev * 100 if prev!=0 else 0
        st.metric('Price', f'{latest_price:.2f}', f'{pct:.2f}%')
    with col2:
        st.metric('RSI (14)', f"{df['RSI'].iloc[-1]:.2f}")
    with col3:
        st.write('Key indicators')
        st.write({'EMA9':f"{df['EMA9'].iloc[-1]:.2f}", 'EMA20':f"{df['EMA20'].iloc[-1]:.2f}", 'ADX':f"{df['ADX'].iloc[-1]:.2f}"})

    # Plot interactive OHLC
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Support/Resistance
    supports, resistances = support_resistance(df, hits_threshold=hits_threshold, pivot_order=pivot_order)
    st.subheader('Strong Support & Resistance (detected)')
    st.write('Supports (level: hits):', supports)
    st.write('Resistances (level: hits):', resistances)

    # Elliott
    ell = elliott_heuristic(df)
    if ell:
        st.write('Elliott-like structure detected between indices:', ell['indices'])
    else:
        st.write('No clear Elliott-like 5-wave structure detected (heuristic).')

    # Signals & summary
    signals = generate_signals(df)
    summary = generate_summary(df, signals, supports, resistances, ell)
    st.session_state['summary'] = summary
    st.subheader('100-word Professional Summary')
    st.info(summary)

    # Data table
    st.subheader('Data Table (last 200 rows)')
    st.dataframe(df.tail(200))

    # Export
    to_export = df.reset_index()
    csv_buf = to_export.to_csv(index=False).encode('utf-8')
    st.download_button('Export CSV', data=csv_buf, file_name=f'{ticker1}_{period}_{interval}.csv')

    # Ratio analysis
    if enable_ratio and st.session_state['df2'] is not None:
        df2 = st.session_state['df2']
        a, b = align_for_ratio(df, df2)
        ratio_series = a['Close'] / b['Close']
        ratio_df = pd.DataFrame({'Close1': a['Close'], 'Close2': b['Close'], 'Ratio': ratio_series})
        ratio_df['RSI1'] = rsi(a['Close'], 14)
        ratio_df['RSI2'] = rsi(b['Close'], 14)
        ratio_df['RSIRatio'] = rsi(ratio_df['Ratio'].fillna(method='ffill'), 14)
        ratio_df['Z1'] = zscore(a['Close'].pct_change().fillna(0))
        ratio_df['Z2'] = zscore(b['Close'].pct_change().fillna(0))
        st.subheader('Ratio Analysis (aligned timestamps)')
        st.dataframe(ratio_df.tail(200))
        csv_buf2 = ratio_df.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button('Export Ratio CSV', data=csv_buf2, file_name=f'ratio_{ticker1}_{ticker2}.csv')

# Backtesting
if run_backtest and st.session_state.get('df1') is not None:
    dfbt = st.session_state['df1']
    with st.spinner('Running backtest...'):
        bt_res = backtest_simple_cross(dfbt)
        st.write('Backtest result (simple EMA crossover):')
        st.write(bt_res)

# Optimization
if run_opt and st.session_state.get('df1') is not None:
    dfopt = st.session_state['df1']
    with st.spinner('Optimizing...'):
        grid = {'short':[5,8,9,10], 'long':[15,20,25,30], 'sl_atr':1.5, 'tp_mul':2.0}
        best = optimize_params(dfopt, grid)
        st.write('Best parameters found:', best)

st.markdown('---')
st.caption('Notes: This app implements many heuristics: Elliott detection is approximate; support/resistance is pivot-based and clustered. Indicators are computed in pure-Python. Use backtest & optimize as experimental starting points. Always paper-test before real trading.')
