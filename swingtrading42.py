# Professional Algo Trading Dashboard (Streamlit)
# Single-file Streamlit application implementing core features requested.
# Notes:
# - No external indicator libraries used (no talib, no pandas_ta). All indicators implemented manually.
# - Built to be button-driven (user clicks 'Fetch Data & Analyze').
# - Timezone handling: converts to IST.
# - Caching used for yfinance fetches to reduce repeated calls.
# - Progress indicators provided for multi-timeframe operations.
# - Exports to CSV/Excel supported.
# - Many advanced features are implemented with sensible defaults; users can configure thresholds.
#
# Requirements: pip install streamlit yfinance pandas numpy plotly openpyxl python-dateutil pytz scipy
# Run: streamlit run professional_algo_trading_dashboard.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import skew, kurtosis
from scipy import stats
import math

# -----------------------------
# Helpers & Utility functions
# -----------------------------

IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.utc

# Map friendly asset names to yfinance tickers (default suggestions)
DEFAULT_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "USD/INR": "INR=X",
}

# Supported intervals mapping for yfinance
SUPPORTED_INTERVALS = {
    '1m': '1m',
    '3m': '3m',
    '5m': '5m',
    '10m': '10m',
    '15m': '15m',
    '30m': '30m',
    '60m': '60m',
    '90m': '90m',
    '1h': '60m',
    '2h': '120m',
    '4h': '240m',
    '1d': '1d',
    '1wk': '1wk',
    '1mo': '1mo'
}

# Periods accepted by yfinance
SUPPORTED_PERIODS = [
    '1d','5d','7d','1mo','3mo','6mo','1y','2y','3y','5y','6y','10y','15y','20y','25y','30y'
]

# -----------------------------
# Indicators (manual implementations)
# -----------------------------

def sma(series: pd.Series, period: int):
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int):
    # Exponential moving average using pandas ewm
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14):
    # Classic RSI implementation
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def atr(df: pd.DataFrame, period: int = 14):
    # df must have columns: High, Low, Close
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def zscore(series: pd.Series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std

# Fibonacci levels around recent swing

def fibonacci_levels(high, low):
    diff = high - low
    levels = {
        '0.0': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100.0%': low
    }
    return levels

# -----------------------------
# Data fetching with caching and rate limiting
# -----------------------------

@st.cache_data(show_spinner=False)
def fetch_yf(ticker: str, period: str, interval: str):
    # yfinance requires certain interval/period combos; try-except to handle
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
    except Exception as e:
        st.warning(f"yfinance download fallback with Ticker: {ticker}. Error: {e}")
        data = pd.DataFrame()
    if data.empty:
        return data
    # yfinance returns multi-index sometimes; reset & standardize
    data = data.reset_index()
    # Rename columns to standard format
    data.columns = [str(c) for c in data.columns]
    # Ensure DateTime column named 'Datetime' and is timezone-aware UTC -> convert to IST
    if 'Datetime' in data.columns:
        dtcol = 'Datetime'
    elif 'Date' in data.columns:
        dtcol = 'Date'
    else:
        dtcol = data.columns[0]
    data = data.rename(columns={dtcol: 'Datetime'})
    # Ensure numeric columns exist
    for col in ['Open','High','Low','Close','Adj Close','Volume']:
        if col not in data.columns:
            if col == 'Adj Close' and 'Close' in data.columns:
                data['Adj Close'] = data['Close']
            else:
                data[col] = np.nan
    # Convert to timezone-aware UTC then to IST
    try:
        if not pd.api.types.is_datetime64_any_dtype(data['Datetime']):
            data['Datetime'] = pd.to_datetime(data['Datetime'])
        if data['Datetime'].dt.tz is None:
            data['Datetime'] = data['Datetime'].dt.tz_localize('UTC')
        data['Datetime_IST'] = data['Datetime'].dt.tz_convert(IST)
    except Exception:
        # Fallback: assume naive UTC
        data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize('UTC')
        data['Datetime_IST'] = data['Datetime'].dt.tz_convert(IST)

    # Rename columns to remove spaces for safety
    data = data.rename(columns={'Adj Close': 'Adj_Close'})
    # Keep essential columns
    out = data[['Datetime','Datetime_IST','Open','High','Low','Close','Adj_Close','Volume']].copy()
    return out

# Rate-limited wrapper
def rate_limited_fetch(ticker, period, interval, delay_seconds=2.0):
    data = fetch_yf(ticker, period, interval)
    time.sleep(delay_seconds)
    return data

# -----------------------------
# Analysis functions
# -----------------------------


def prepare_ohlcv(df: pd.DataFrame):
    # Ensure standard columns and sorted
    df = df.copy()
    df = df.sort_values('Datetime')
    df = df.reset_index(drop=True)
    # Ensure numeric
    for c in ['Open','High','Low','Close','Adj_Close','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def add_indicators(df: pd.DataFrame):
    df = df.copy()
    df['EMA9'] = ema(df['Close'], 9)
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA33'] = ema(df['Close'], 33)
    df['EMA50'] = ema(df['Close'], 50)
    df['EMA100'] = ema(df['Close'], 100)
    df['EMA150'] = ema(df['Close'], 150)
    df['EMA200'] = ema(df['Close'], 200)
    df['SMA20'] = sma(df['Close'], 20)
    df['SMA50'] = sma(df['Close'], 50)
    df['SMA100'] = sma(df['Close'], 100)
    df['SMA150'] = sma(df['Close'], 150)
    df['SMA200'] = sma(df['Close'], 200)
    df['RSI14'] = rsi(df['Close'], 14)
    df['ATR14'] = atr(df, 14)
    return df

# Trend detection

def detect_trend(df: pd.DataFrame, lookback=20):
    # Simple slope on close over lookback
    if len(df) < 2:
        return 'Neutral'
    series = df['Close'].dropna().astype(float)
    if len(series) < 2:
        return 'Neutral'
    y = series[-lookback:]
    x = np.arange(len(y))
    if len(x) < 2:
        return 'Neutral'
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # Simple thresholding
    if slope > 0:
        return 'Up'
    elif slope < 0:
        return 'Down'
    else:
        return 'Neutral'

# Support and resistance detection - naive extrema-based

def support_resistance_levels(df: pd.DataFrame, n_levels=3):
    # Using rolling local minima/maxima
    highs = df['High']
    lows = df['Low']
    support = sorted(list(set(lows.rolling(window=20, center=True).min().dropna().round(2))))[-n_levels:]
    resistance = sorted(list(set(highs.rolling(window=20, center=True).max().dropna().round(2))))[:n_levels]
    # Clean and return
    support = support if isinstance(support, list) else []
    resistance = resistance if isinstance(resistance, list) else []
    return support, resistance

# Volatility bins

def volatility_bins(df: pd.DataFrame, n_bins:int=5):
    # volatility measured as rolling std of returns (daily equivalent)
    df = df.copy()
    df['ret'] = df['Close'].pct_change()
    df['vol'] = df['ret'].rolling(window=14).std() * np.sqrt(252)
    # Remove NaNs
    vol_nonan = df['vol'].dropna()
    if vol_nonan.empty:
        return pd.DataFrame()
    # create bins
    bins = pd.qcut(vol_nonan, q=n_bins, duplicates='drop')
    return pd.DataFrame({'Datetime_IST': df['Datetime_IST'], 'vol': df['vol'], 'bin': bins, 'Close': df['Close'], 'ret': df['ret']})

# Pattern detection for large moves

def detect_large_moves(df: pd.DataFrame, threshold_points=30):
    # Identify candles where abs(close - open) >= threshold
    df = df.copy()
    df['move'] = (df['Close'] - df['Open']).abs()
    candidates = df[df['move'] >= threshold_points]
    rows = []
    for idx in candidates.index:
        window = df[max(0, idx-10):idx+1]
        # features
        vol_burst = window['Close'].pct_change().std()
        vol_spike = window['Volume'].max() if 'Volume' in window.columns else np.nan
        rsi_before = window['RSI14'].iloc[0] if 'RSI14' in window.columns else np.nan
        rsi_at = window['RSI14'].iloc[-1] if 'RSI14' in window.columns else np.nan
        ema20_cross = ((window['EMA20'].iloc[-2] < window['EMA50'].iloc[-2]) and (window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1])) if ('EMA20' in window.columns and 'EMA50' in window.columns and len(window) > 1) else False
        direction = 'Up' if (df.loc[idx,'Close'] - df.loc[idx,'Open']) > 0 else 'Down'
        rows.append({
            'Datetime_IST': df.loc[idx,'Datetime_IST'],
            'move_points': df.loc[idx,'Close'] - df.loc[idx,'Open'],
            'move_abs': df.loc[idx,'move'],
            'direction': direction,
            'volatility_burst': vol_burst,
            'volume_spike': vol_spike,
            'rsi_before': rsi_before,
            'rsi_at': rsi_at,
            'ema20_50_cross': ema20_cross
        })
    return pd.DataFrame(rows)

# Ratio and binning

def compute_ratio(df1, df2):
    # Align on Datetime_IST
    merged = pd.merge(df1[['Datetime_IST','Close']], df2[['Datetime_IST','Close']], on='Datetime_IST', how='inner', suffixes=('_1','_2'))
    merged['ratio'] = merged['Close_1'] / merged['Close_2']
    merged['ratio_ret'] = merged['ratio'].pct_change()
    merged['RSI_ratio'] = rsi(merged['ratio'].fillna(method='ffill'), 14)
    return merged

# Ratio bin creation with explicit ranges

def ratio_bins(merged: pd.DataFrame, n_bins=5):
    q = merged['ratio'].quantile(np.linspace(0,1,n_bins+1))
    bins = []
    for i in range(n_bins):
        bins.append((q.iloc[i], q.iloc[i+1]))
    # label each row
    def which_bin(x):
        for i,b in enumerate(bins):
            if x>=b[0] and x<=b[1]:
                return i
        return np.nan
    merged['ratio_bin'] = merged['ratio'].apply(which_bin)
    # Create human-readable labels
    labels = [f"Bin {i+1} ({bins[i][0]:.6f}-{bins[i][1]:.6f})" for i in range(len(bins))]
    merged['ratio_bin_label'] = merged['ratio_bin'].apply(lambda x: labels[int(x)] if not pd.isna(x) else 'NA')
    return merged, labels

# Multi-timeframe fetcher
def fetch_multi_timeframes(ticker, tf_map, delay_seconds=1.5):
    # tf_map: dict of {label: (interval, period)}
    out = {}
    total = len(tf_map)
    i = 0
    for label,(interval,period) in tf_map.items():
        i += 1
        out[label] = rate_limited_fetch(ticker, period, interval, delay_seconds)
        st.session_state['progress'] = int(i/total*100)
    return out

# -----------------------------
# UI Components (Streamlit)
# -----------------------------

# Initialize session state variables necessary to keep UI
if 'progress' not in st.session_state:
    st.session_state['progress'] = 0
if 'last_fetch' not in st.session_state:
    st.session_state['last_fetch'] = None
if 'data_cache' not in st.session_state:
    st.session_state['data_cache'] = {}


st.set_page_config(page_title='Professional Algo Trading Dashboard', layout='wide', initial_sidebar_state='expanded')

# Sidebar - configuration
with st.sidebar:
    st.header('Configuration')
    ticker1_input = st.text_input('Ticker 1 (yfinance)', value='^NSEI')
    ticker1_label = st.text_input('Ticker 1 Display Name', value='NIFTY 50')
    ratio_enable = st.checkbox('Enable Ratio Analysis (Ticker 2)', value=False)
    ticker2_input = st.text_input('Ticker 2 (yfinance)', value='^NSEBANK' if ratio_enable else '')
    ticker2_label = st.text_input('Ticker 2 Display Name', value='BANK NIFTY' if ratio_enable else '')
    interval = st.selectbox('Interval', options=['1m','3m','5m','10m','15m','30m','1h','2h','4h','1d'], index=4)
    period = st.selectbox('Period', options=SUPPORTED_PERIODS, index=3)
    delay_seconds = st.slider('API delay (seconds) between calls', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    large_move_threshold = st.number_input('Large move detection threshold (points)', min_value=1.0, value=30.0, step=1.0)
    position_risk_pct = st.slider('Position risk per trade (%)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    st.markdown('---')
    st.write('Style & Export')
    export_format = st.selectbox('Export format', ['csv','excel'])

# Main layout
st.title('Professional Algo Trading Dashboard')
st.caption('Button-based fetching. All datetimes converted to IST. Indicators computed without external TA libs.')

col1, col2 = st.columns([3,1])

with col1:
    st.subheader('Ticker Selection & Fetch')
    st.write(f"Ticker 1: {ticker1_input}  \t Label: {ticker1_label}")
    if ratio_enable:
        st.write(f"Ticker 2: {ticker2_input}  \t Label: {ticker2_label}")

    # Button to fetch data
    if st.button('Fetch Data & Analyze'):
        st.session_state['progress'] = 1
        # fetch primary
        with st.spinner('Fetching data for ticker 1...'):
            df1 = rate_limited_fetch(ticker1_input, period, SUPPORTED_INTERVALS.get(interval, interval), delay_seconds)
            if df1.empty:
                st.error(f'No data returned for {ticker1_input} - please check ticker and interval/period compatibility')
            else:
                df1 = prepare_ohlcv(df1)
                df1 = add_indicators(df1)
                st.session_state['data_cache']['df1'] = df1
        # fetch secondary if ratio enabled
        if ratio_enable and ticker2_input.strip() != '':
            with st.spinner('Fetching data for ticker 2...'):
                df2 = rate_limited_fetch(ticker2_input, period, SUPPORTED_INTERVALS.get(interval, interval), delay_seconds)
                if df2.empty:
                    st.error(f'No data returned for {ticker2_input} - please check ticker and interval/period compatibility')
                else:
                    df2 = prepare_ohlcv(df2)
                    df2 = add_indicators(df2)
                    st.session_state['data_cache']['df2'] = df2
        st.session_state['last_fetch'] = datetime.now(tz=IST)
        st.success('Data fetch complete')

    # Show progress
    st.progress(st.session_state['progress'])
    if st.session_state['last_fetch']:
        st.write('Last fetch (IST): ', st.session_state['last_fetch'].strftime('%Y-%m-%d %H:%M:%S %Z'))

with col2:
    st.subheader('Quick controls')
    st.write('Use these to manage cached data:')
    if st.button('Clear Cache'):
        st.session_state['data_cache'] = {}
        st.success('Cache cleared')
    if st.button('Reset Progress'):
        st.session_state['progress'] = 0
        st.success('Progress reset')

# If data present, show dashboard
if 'df1' in st.session_state['data_cache']:
    df1 = st.session_state['data_cache']['df1']
    st.markdown('---')
    st.header('Market Overview')
    latest = df1.iloc[-1]
    prev = df1['Close'].iloc[-2] if len(df1) > 1 else latest['Close']
    change_pct = (latest['Close'] - prev) / prev * 100 if prev != 0 else 0
    points = latest['Close'] - prev
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric(label=f"{ticker1_label} Price", value=f"{latest['Close']:.2f}", delta=f"{change_pct:.2f}%")
    with colB:
        st.metric(label='Points Change', value=f"{points:.2f}")
    with colC:
        st.metric(label='ATR(14)', value=f"{latest['ATR14']:.4f}")
    with colD:
        st.metric(label='RSI(14)', value=f"{latest['RSI14']:.2f}")

    # Data Table
    st.subheader('Data Table (Ticker 1)')
    display_cols = ['Datetime_IST','Open','High','Low','Close','Volume','RSI14','ATR14']
    st.dataframe(df1[display_cols].tail(200).assign(Datetime_IST=lambda d: d['Datetime_IST'].dt.strftime('%Y-%m-%d %H:%M:%S')))

    # Export
    if export_format == 'csv':
        csv = df1.to_csv(index=False)
        st.download_button('Export Ticker1 CSV', data=csv, file_name=f'{ticker1_label}_data.csv', mime='text/csv')
    else:
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            df1.to_excel(writer, index=False, sheet_name='ticker1')
        towrite.seek(0)
        st.download_button('Export Ticker1 Excel', data=towrite, file_name=f'{ticker1_label}_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Ratio analysis
    if ratio_enable and 'df2' in st.session_state['data_cache']:
        df2 = st.session_state['data_cache']['df2']
        st.markdown('---')
        st.header('Ratio Analysis')
        merged = compute_ratio(df1, df2)
        merged, labels = ratio_bins(merged)
        st.subheader('Ratio Head')
        st.dataframe(merged[['Datetime_IST','Close_1','Close_2','ratio','RSI_ratio','ratio_bin_label']].tail(200).assign(Datetime_IST=lambda d: d['Datetime_IST'].dt.strftime('%Y-%m-%d %H:%M:%S')))
        # Binned summary
        bin_summary = merged.groupby('ratio_bin_label').agg({
            'Close_1': ['mean','last'],
            'Close_2': ['mean','last'],
            'ratio': ['mean','std'],
            'ratio_ret': lambda x: (x+1).prod()-1
        })
        st.subheader('Ratio Bin Summary')
        st.dataframe(bin_summary)
        # Export merged
        if export_format == 'csv':
            csv = merged.to_csv(index=False)
            st.download_button('Export Ratio CSV', data=csv, file_name=f'{ticker1_label}_{ticker2_label}_ratio.csv', mime='text/csv')
        else:
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                merged.to_excel(writer, index=False, sheet_name='ratio')
            towrite.seek(0)
            st.download_button('Export Ratio Excel', data=towrite, file_name=f'{ticker1_label}_{ticker2_label}_ratio.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Multi-timeframe analysis
    st.markdown('---')
    st.header('Multi-Timeframe Analysis')
    # Define default mapping per requirements
    tf_map = {
        '1m/1d': (SUPPORTED_INTERVALS.get('1m','1m'), '1d'),
        '5m/5d': (SUPPORTED_INTERVALS.get('5m','5m'), '5d'),
        '15m/5d': (SUPPORTED_INTERVALS.get('15m','15m'), '5d'),
        '30m/1mo': (SUPPORTED_INTERVALS.get('30m','30m'), '1mo'),
        '1h/1mo': (SUPPORTED_INTERVALS.get('1h','60m'), '1mo'),
        '2h/3mo': (SUPPORTED_INTERVALS.get('2h','120m'), '3mo'),
        '4h/6mo': (SUPPORTED_INTERVALS.get('4h','240m'), '6mo'),
        '1d/1y': (SUPPORTED_INTERVALS.get('1d','1d'), '1y'),
        '1w/5y': (SUPPORTED_INTERVALS.get('1wk','1wk'), '5y'),
        '1mo/10y': (SUPPORTED_INTERVALS.get('1mo','1mo'), '10y')
    }

    # Fetch multiple timeframes for ticker1 only (configurable)
    st.info('Fetching multi-timeframe data for analysis. This will use the configured API delay between calls.')
    if st.button('Fetch Multi-Timeframe Analysis'):
        st.session_state['progress'] = 0
        mdata = fetch_multi_timeframes(ticker1_input, tf_map, delay_seconds=delay_seconds)
        # Process each timeframe
        rows = []
        total = len(mdata)
        count = 0
        for k,v in mdata.items():
            count += 1
            if v.empty:
                continue
            v = prepare_ohlcv(v)
            v = add_indicators(v)
            trend = detect_trend(v)
            max_close = v['Close'].max()
            min_close = v['Close'].min()
            fib = fibonacci_levels(max_close, min_close)
            vol = v['Close'].pct_change().std() * np.sqrt(252)
            pct_change = (v['Close'].iloc[-1] - v['Close'].iloc[0]) / v['Close'].iloc[0] * 100 if len(v)>1 else 0
            points = v['Close'].iloc[-1] - v['Close'].iloc[0]
            sup, res = support_resistance_levels(v)
            rsi_val = v['RSI14'].iloc[-1]
            row = {
                'Timeframe': k,
                'Trend': trend,
                'Max_Close': max_close,
                'Min_Close': min_close,
                'Fibonacci': str(fib),
                'Volatility': vol,
                '%Change': pct_change,
                'PointsChanged': points,
                'Supports': str(sup),
                'Resistances': str(res),
                'RSI14': rsi_val,
                'EMA9': v['EMA9'].iloc[-1],
                'EMA20': v['EMA20'].iloc[-1],
                'EMA33': v['EMA33'].iloc[-1],
                'EMA50': v['EMA50'].iloc[-1],
                'EMA100': v['EMA100'].iloc[-1],
                'EMA150': v['EMA150'].iloc[-1],
                'EMA200': v['EMA200'].iloc[-1],
                'SMA20': v['SMA20'].iloc[-1],
                'SMA50': v['SMA50'].iloc[-1],
                'SMA100': v['SMA100'].iloc[-1],
                'SMA150': v['SMA150'].iloc[-1],
                'SMA200': v['SMA200'].iloc[-1],
            }
            rows.append(row)
            st.session_state['progress'] = int(count/total*100)
        tf_df = pd.DataFrame(rows)
        st.dataframe(tf_df)
        # Insights below table
        st.subheader('Multi-Timeframe Insights')
        if not tf_df.empty:
            up_count = sum(tf_df['Trend']=='Up')
            down_count = sum(tf_df['Trend']=='Down')
            neutral_count = len(tf_df)-up_count-down_count
            st.write(f"Trends summary: Up: {up_count}, Down: {down_count}, Neutral: {neutral_count}")
            # Simple forecast logic: if majority Up -> Up else Down else Neutral
            if up_count > down_count and up_count > neutral_count:
                rec = 'Up'
            elif down_count > up_count and down_count > neutral_count:
                rec = 'Down'
            else:
                rec = 'Neutral'
            st.info(f'Recommended forecast across timeframes: {rec}')

    st.progress(st.session_state['progress'])

    # Volatility bins analysis
    st.markdown('---')
    st.header('Volatility Bins Analysis (Ticker1)')
    vb_df = volatility_bins(df1)
    if vb_df.empty:
        st.write('Not enough data for volatility bins')
    else:
        st.dataframe(vb_df.tail(200).assign(Datetime_IST=lambda d: d['Datetime_IST'].dt.strftime('%Y-%m-%d %H:%M:%S')))
        # Summary statistics
        st.subheader('Volatility Summary')
        st.write('Highest vol:', vb_df['vol'].max())
        st.write('Lowest vol:', vb_df['vol'].min())
        st.write('Mean vol:', vb_df['vol'].mean())

    # Pattern recognition
    st.markdown('---')
    st.header('Advanced Pattern Recognition')
    df_for_patterns = df1.copy()
    large_moves = detect_large_moves(df_for_patterns, threshold_points=large_move_threshold)
    st.write('Large Moves Detected:')
    if large_moves.empty:
        st.write('No large moves detected with current threshold')
    else:
        st.dataframe(large_moves.assign(Datetime_IST=lambda d: d['Datetime_IST'].dt.strftime('%Y-%m-%d %H:%M:%S')))
        st.subheader('Pattern Summary')
        st.write(f"Total patterns detected: {len(large_moves)}")
        # frequency
        move_freq = large_moves['direction'].value_counts().to_dict()
        st.write('Move frequency:', move_freq)

    # Interactive charts (candlestick + indicators)
    st.markdown('---')
    st.header('Interactive Charts')
    chart_df = df1.copy()
    chart_df = chart_df.tail(500)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75,0.25])
    fig.add_trace(go.Candlestick(x=chart_df['Datetime_IST'], open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df['Datetime_IST'], y=chart_df['EMA20'], name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df['Datetime_IST'], y=chart_df['EMA50'], name='EMA50'), row=1, col=1)
    fig.add_trace(go.Bar(x=chart_df['Datetime_IST'], y=chart_df['Volume'], name='Volume', showlegend=False), row=2, col=1)
    fig.update_layout(height=700, margin={'t':10,'b':10})
    st.plotly_chart(fig, use_container_width=True)

    # Statistical Distribution Analysis
    st.markdown('---')
    st.header('Statistical Distribution Analysis')
    ret = df1['Close'].pct_change().dropna()
    mean_ret = ret.mean()
    std_ret = ret.std()
    skew_ret = skew(ret.dropna()) if len(ret.dropna())>0 else np.nan
    kurt_ret = kurtosis(ret.dropna()) if len(ret.dropna())>0 else np.nan
    st.write('Mean:', mean_ret, 'Std:', std_ret, 'Skewness:', skew_ret, 'Kurtosis:', kurt_ret)

    # Histogram with normal overlay using plotly
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=ret, nbinsx=50, name='Returns'))
    # normal curve
    x_vals = np.linspace(ret.min(), ret.max(), 200)
    pdf = stats.norm.pdf(x_vals, mean_ret, std_ret)
    # scale pdf to histogram by area
    pdf_scaled = pdf * (ret.count() * (x_vals[1]-x_vals[0]))
    hist_fig.add_trace(go.Scatter(x=x_vals, y=pdf_scaled, mode='lines', name='Normal'))
    hist_fig.update_layout(height=400)
    st.plotly_chart(hist_fig, use_container_width=True)

    # Z-score table
    z = zscore(df1['Close'].pct_change().fillna(0))
    z_df = pd.DataFrame({'Datetime_IST': df1['Datetime_IST'], 'ret': df1['Close'].pct_change().fillna(0), 'zscore': z})
    st.subheader('Z-Score Analysis')
    st.dataframe(z_df.tail(200).assign(Datetime_IST=lambda d: d['Datetime_IST'].dt.strftime('%Y-%m-%d %H:%M:%S')))

    # Final Trading Recommendation
    st.markdown('---')
    st.header('Final Trading Recommendation')
    # Multi-factor scoring
    def compute_signal(df):
        # Use latest values
        latest = df.iloc[-1]
        # Trend score from multi-timeframe (simple single timeframe here)
        trend_score = 1 if detect_trend(df, lookback=20)=='Up' else -1 if detect_trend(df, lookback=20)=='Down' else 0
        # RSI
        rsi_score = -1 if latest['RSI14']>70 else 1 if latest['RSI14']<30 else 0
        # Z-score
        zsc = ((df['Close'].iloc[-5:].pct_change().mean()) - df['Close'].pct_change().mean())
        z_score_component = -1 if zsc>0 else 1
        # EMA alignment
        ema_align = 0
        if latest['EMA9'] > latest['EMA20'] > latest['EMA50']:
            ema_align = 1
        elif latest['EMA9'] < latest['EMA20'] < latest['EMA50']:
            ema_align = -1
        # Weighted sum
        total = 0.3*trend_score + 0.2*rsi_score + 0.2*z_score_component + 0.3*ema_align
        return total, {'trend_score':trend_score,'rsi_score':rsi_score,'z_score_component':z_score_component,'ema_align':ema_align}

    score, components = compute_signal(df1)
    # Interpret
    if score > 0.5:
        signal = 'Strong Buy'
    elif score > 0.1:
        signal = 'Buy'
    elif score < -0.5:
        signal = 'Strong Sell'
    elif score < -0.1:
        signal = 'Sell'
    else:
        signal = 'Neutral/Hold'

    st.metric(label='Signal', value=signal)
    st.write('Score breakdown:', components)

    # Trade sizing and targets using ATR
    latest_atr = df1['ATR14'].iloc[-1]
    entry_price = df1['Close'].iloc[-1]
    stop_loss = entry_price - 1.5*latest_atr if signal in ['Buy','Strong Buy'] else entry_price + 1.5*latest_atr if signal in ['Sell','Strong Sell'] else np.nan
    target = entry_price + 3*latest_atr if signal in ['Buy','Strong Buy'] else entry_price - 3*latest_atr if signal in ['Sell','Strong Sell'] else np.nan
    st.write('Entry Price:', round(entry_price,2))
    st.write('Stop Loss:', round(stop_loss,2) if not pd.isna(stop_loss) else 'N/A')
    st.write('Target:', round(target,2) if not pd.isna(target) else 'N/A')
    # Position size
    if not pd.isna(stop_loss):
        risk_per_share = abs(entry_price - stop_loss)
        # Assume user capital
        capital = 20000
        risk_amount = capital * (position_risk_pct/100)
        size = math.floor(risk_amount / risk_per_share) if risk_per_share>0 else 0
        st.write(f'Recommended position size (capital {capital}, risk {position_risk_pct}%): {size} units')
        rr = abs(target - entry_price)/risk_per_share if risk_per_share>0 else np.nan
        st.write(f'Risk/Reward ratio approx: {rr:.2f}')

    # Persist analysis (keep session state)
    st.session_state['last_analysis'] = {
        'signal': signal,
        'score': score,
        'components': components,
        'entry': entry_price,
        'stop_loss': stop_loss,
        'target': target
    }

    st.markdown('---')
    st.success('Analysis complete. Use export buttons to save data. All analysis is persisted in this session.')

else:
    st.info('No data cached. Please select ticker and click "Fetch Data & Analyze"')

# Footer with notes
st.markdown('---')
st.caption('This application is an example professional-grade dashboard. Indicators are computed without external TA libraries. Trade signals are for educational purposes only and not financial advice.')

# End of file
