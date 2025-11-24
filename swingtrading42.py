"""
Streamlit Algo Trading Dashboard
Single-file Streamlit app implementing: data fetching (yfinance), multi-timeframe analysis,
technical indicators (manually computed), ratio analysis, volatility bins, pattern detection,
statistical distribution, z-score analysis, and multi-factor trading recommendation.

Notes:
- No use of talib or pandas_ta
- All datetimes converted to IST
- Button-based fetching, configurable delays
- Session state persists results
- Exports to CSV/Excel

Run: streamlit run streamlit_algo_dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz
from scipy import stats
import io
import base64
import math
import plotly.graph_objs as go
import plotly.express as px

# -----------------------------
# Configuration / Utilities
# -----------------------------
st.set_page_config(page_title="Algo Trading Dashboard", layout="wide")

IST = pytz.timezone('Asia/Kolkata')

DEFAULT_DELAY = 1.8  # seconds between API calls

# Supported tickers (examples) - user can enter custom yfinance ticker
PRESET_TICKERS = {
    'NIFTY 50': '^NSEI',
    'BANK NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'USDINR': 'INR=X'
}

# Helper: ensure session state keys
if 'data' not in st.session_state:
    st.session_state['data'] = {}

# -----------------------------
# Helper functions: timezones
# -----------------------------

def to_ist(dt: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert timezone-aware or naive index to IST timezone."""
    if isinstance(dt, pd.DatetimeIndex):
        if dt.tz is None:
            # assume UTC coming from yfinance
            dt = dt.tz_localize('UTC')
        return dt.tz_convert(IST)
    else:
        return pd.to_datetime(dt).tz_localize('UTC').tz_convert(IST)

# -----------------------------
# Data fetching and normalization
# -----------------------------

def clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reset multiindex, keep Datetime, Open, High, Low, Close, Volume, adjust tz to IST."""
    # yfinance sometimes returns multi-index columns; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]
    df = df.rename(columns=lambda x: x.strip())

    # Common column names mapping
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if 'open' in lc and 'adj' not in lc:
            colmap[c] = 'Open'
        if 'high' in lc:
            colmap[c] = 'High'
        if 'low' in lc:
            colmap[c] = 'Low'
        if ('close' in lc and 'adj' not in lc) or lc == 'close':
            colmap[c] = 'Close'
        if 'volume' in lc:
            colmap[c] = 'Volume'
    df = df.rename(columns=colmap)

    # keep required columns
    keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    df = df[keep].copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.index = to_ist(df.index)
    df.index.name = 'DateTime'
    df = df.reset_index()
    return df

@st.cache_data(ttl=300)
def fetch_yf(ticker, interval, period, delay=DEFAULT_DELAY):
    """Fetch data using yfinance with a delay. Returns cleaned dataframe."""
    # yfinance accepts period like '1mo' and interval like '5m'
    time.sleep(delay)
    try:
        raw = yf.download(tickers=ticker, interval=interval, period=period, progress=False, threads=False)
    except Exception as e:
        st.error(f"yfinance download error: {e}")
        raise
    if raw is None or raw.empty:
        return pd.DataFrame()
    return clean_yf_df(raw)

# -----------------------------
# Indicators (manual implementations)
# -----------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    # manual EMA using pandas ewm
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

# -----------------------------
# Fibonacci levels
# -----------------------------

def fibonacci_levels(high, low):
    diff = high - low
    levels = {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100%': low
    }
    return levels

# -----------------------------
# Support/Resistance (simple pivot and rolling)
# -----------------------------

def support_resistance_levels(df: pd.DataFrame, lookback=20, num_levels=3):
    highs = df['High'].rolling(window=lookback).max()
    lows = df['Low'].rolling(window=lookback).min()
    latest_high = highs.iloc[-1]
    latest_low = lows.iloc[-1]
    # return equally spaced levels between low and high
    levels = {}
    for i in range(1, num_levels+1):
        levels[f'Resistance_{i}'] = latest_high - (i-1) * (latest_high - latest_low) / num_levels
        levels[f'Support_{i}'] = latest_low + (i-1) * (latest_high - latest_low) / num_levels
    return levels

# -----------------------------
# Volatility bins
# -----------------------------

def compute_volatility(df: pd.DataFrame, window=14):
    # volatility as rolling std of returns * sqrt(periodicity)
    returns = df['Close'].pct_change()
    vol = returns.rolling(window).std() * math.sqrt(window)
    return vol

def assign_bins(values, bins=5):
    # create equal-frequency bins
    try:
        labels = [f'Bin_{i+1}' for i in range(bins)]
        return pd.qcut(values, q=bins, labels=labels, duplicates='drop')
    except Exception:
        return pd.Series(['Bin_1'] * len(values), index=values.index)

# -----------------------------
# Pattern detection (simplified rules)
# -----------------------------

def detect_patterns(df: pd.DataFrame, move_threshold=30):
    """Detect simple patterns preceding large moves."""
    patterns = []
    df = df.copy()
    df['ReturnPts'] = df['Close'].diff()
    df['ReturnPct'] = df['Close'].pct_change() * 100
    for i in range(10, len(df)):
        move = df['ReturnPts'].iloc[i]
        if abs(move) >= move_threshold:
            window = df.iloc[i-10:i]
            pat = {
                'DateTime': df['DateTime'].iloc[i],
                'MovePts': move,
                'MovePct': df['ReturnPct'].iloc[i],
                'Direction': 'Up' if move>0 else 'Down',
                'VolBurst': window['High'].max() - window['Low'].min(),
                'VolSpike': window['Volume'].max() if 'Volume' in window else np.nan,
                'RSI_before': float(rsi(window['Close']).iloc[-2]),
                'RSI_at': float(rsi(df['Close']).iloc[i])
            }
            patterns.append(pat)
    return pd.DataFrame(patterns)

# -----------------------------
# Z-score and distribution
# -----------------------------

def zscore(series):
    return (series - series.mean()) / series.std()

# -----------------------------
# Multi-timeframe aggregation
# -----------------------------

def aggregate_timeframes(ticker, base_interval, timeframe_map, period_map, delay):
    """Fetch multiple timeframes and compute summary table."""
    rows = []
    total = len(timeframe_map)
    prog = st.progress(0)
    for idx, (label, interval) in enumerate(timeframe_map.items()):
        period = period_map.get(label)
        df = fetch_yf(ticker, interval=interval, period=period, delay=delay)
        if df.empty:
            rows.append({'Timeframe': label, 'Error': 'No data'})
            prog.progress((idx+1)/total)
            continue
        close = df['Close']
        trend = 'Up' if close.iloc[-1] > close.iloc[0] else 'Down'
        maxc = close.max()
        minc = close.min()
        fibs = fibonacci_levels(maxc, minc)
        vol = compute_volatility(df).iloc[-1]
        pctchg = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
        emalist = {f'EMA_{p}': float(ema(close, p).iloc[-1]) for p in [9,20,21,33,50,100,150,200]}
        smalist = {f'SMA_{p}': float(sma(close, p).iloc[-1]) for p in [20,50,100,150,200]}
        rsi_val = float(rsi(close).iloc[-1])
        row = {
            'Timeframe': label,
            'Trend': trend,
            'MaxClose': maxc,
            'MinClose': minc,
            'Volatility': vol,
            'PctChange': pctchg,
            'RSI': rsi_val,
            **fibs,
            **emalist,
            **smalist
        }
        rows.append(row)
        prog.progress((idx+1)/total)
    prog.empty()
    return pd.DataFrame(rows)

# -----------------------------
# Signal generation (weights)
# -----------------------------

def generate_signal(mtf_df, rsi_val, zscore_val):
    """Combine weights per spec: MTF trend 30%, RSI 20%, Zscore 20%, EMA alignment 30%"""
    score = 0.0
    # MTF trend: count Up vs Down
    up_count = (mtf_df['Trend'] == 'Up').sum()
    down_count = (mtf_df['Trend'] == 'Down').sum()
    mtf_score = (up_count - down_count) / max(1, len(mtf_df))  # -1..1
    score += 0.3 * mtf_score
    # RSI: neutral around 50; higher suggests overbought -> negative for buy
    rsi_score = (50 - rsi_val) / 50  # positive when RSI<50
    score += 0.2 * rsi_score
    # Z-score: negative means price below mean -> positive for buy
    z_score_component = (-zscore_val if not np.isnan(zscore_val) else 0)
    score += 0.2 * z_score_component
    # EMA alignment: measure how many EMAs are below price
    ema_cols = [c for c in mtf_df.columns if c.startswith('EMA_')]
    ema_align = 0
    if ema_cols:
        latest = mtf_df.iloc[0]  # choose highest timeframe row
        price = latest.get('MaxClose', None) or latest.get('MinClose', None)
        if price is not None:
            below = sum([1 for c in ema_cols if latest[c] < price])
            ema_align = (below / len(ema_cols)) * 2 - 1  # -1..1
    score += 0.3 * ema_align
    # Map score to recommendation
    if score > 0.5:
        rec = 'Strong Buy'
    elif score > 0.1:
        rec = 'Buy'
    elif score < -0.5:
        rec = 'Strong Sell'
    elif score < -0.1:
        rec = 'Sell'
    else:
        rec = 'Neutral'
    return {'score': score, 'recommendation': rec}

# -----------------------------
# CSV / Excel export
# -----------------------------

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
        writer.save()
    processed_data = output.getvalue()
    return processed_data

# -----------------------------
# UI: Inputs
# -----------------------------

st.title('Professional Algo Trading Dashboard')
st.markdown('''
This app fetches market data from yfinance, computes indicators manually, performs multi-timeframe
analysis, ratio analysis, volatility bins, pattern detection, distribution and Z-score analysis,
and produces a weighted trading recommendation.
''')

col1, col2, col3 = st.columns([3,2,2])
with col1:
    ticker1 = st.selectbox('Ticker 1 (or enter custom)', options=list(PRESET_TICKERS.keys()) + ['Custom'], index=0)
    if ticker1 == 'Custom':
        t1_input = st.text_input('Enter yfinance ticker for Ticker 1', value='^NSEI')
        ticker1_sym = t1_input.strip()
    else:
        ticker1_sym = PRESET_TICKERS[ticker1]
with col2:
    ratio_enabled = st.checkbox('Enable Ratio Analysis (Ticker 2)', value=False)
    if ratio_enabled:
        ticker2 = st.selectbox('Ticker 2 (or enter custom)', options=list(PRESET_TICKERS.keys()) + ['Custom'], index=1)
        if ticker2 == 'Custom':
            t2_input = st.text_input('Enter yfinance ticker for Ticker 2', value='INR=X')
            ticker2_sym = t2_input.strip()
        else:
            ticker2_sym = PRESET_TICKERS[ticker2]
    else:
        ticker2_sym = None
with col3:
    interval = st.selectbox('Interval', ['1m','3m','5m','10m','15m','30m','60m','120m','240m','1d'], index=4)
    period = st.selectbox('Period', ['1d','5d','7d','1mo','3mo','6mo','1y','2y','3y','5y'], index=3)
    delay = st.slider('API delay between calls (s)', min_value=0.5, max_value=5.0, value=DEFAULT_DELAY, step=0.1)

col4, col5 = st.columns([1,3])
with col4:
    fetch_btn = st.button('Fetch Data & Analyze')
    download_all_csv = st.button('Export Latest Data')
with col5:
    move_threshold = st.number_input('Pattern detect move threshold (points)', value=30)
    z_window = st.number_input('Z-score window (returns)', value=50)

# -----------------------------
# Main operation
# -----------------------------

if fetch_btn:
    # Fetch ticker1
    with st.spinner('Fetching Ticker 1 data...'):
        try:
            df1 = fetch_yf(ticker1_sym, interval=interval, period=period, delay=delay)
        except Exception as e:
            st.error(f'Error fetching ticker1: {e}')
            df1 = pd.DataFrame()
    if df1.empty:
        st.error('Ticker 1 returned no data. Check ticker or try different interval/period.')
    else:
        st.session_state['data']['ticker1'] = df1

    # Fetch ticker2 if enabled
    if ratio_enabled and ticker2_sym:
        with st.spinner('Fetching Ticker 2 data...'):
            try:
                df2 = fetch_yf(ticker2_sym, interval=interval, period=period, delay=delay)
            except Exception as e:
                st.error(f'Error fetching ticker2: {e}')
                df2 = pd.DataFrame()
        if df2.empty:
            st.warning('Ticker 2 returned no data. Ratio analysis disabled.')
            st.session_state['data'].pop('ticker2', None)
            ratio_enabled = False
            ticker2_sym = None
        else:
            st.session_state['data']['ticker2'] = df2

    # Now compute indicators for each
    if 'ticker1' in st.session_state['data']:
        df = st.session_state['data']['ticker1']
        df['RSI'] = rsi(df['Close'])
        for p in [9,20,21,33,50,100,150,200]:
            df[f'EMA_{p}'] = ema(df['Close'], p)
        for p in [20,50,100,150,200]:
            df[f'SMA_{p}'] = sma(df['Close'], p)
        df['ATR_14'] = atr(df, 14)
        df['Volatility'] = compute_volatility(df)
        st.session_state['data']['ticker1'] = df

    if ratio_enabled and 'ticker2' in st.session_state['data']:
        df = st.session_state['data']['ticker2']
        df['RSI'] = rsi(df['Close'])
        for p in [9,20,21,33,50,100,150,200]:
            df[f'EMA_{p}'] = ema(df['Close'], p)
        for p in [20,50,100,150,200]:
            df[f'SMA_{p}'] = sma(df['Close'], p)
        df['ATR_14'] = atr(df, 14)
        df['Volatility'] = compute_volatility(df)
        st.session_state['data']['ticker2'] = df

    st.success('Data fetched and indicators computed.')

# Export
if download_all_csv:
    if 'ticker1' not in st.session_state['data']:
        st.warning('No data to export. Fetch data first.')
    else:
        buf = io.BytesIO()
        writer = pd.ExcelWriter(buf, engine='xlsxwriter')
        st.session_state['data']['ticker1'].to_excel(writer, sheet_name='ticker1', index=False)
        if 'ticker2' in st.session_state['data']:
            st.session_state['data']['ticker2'].to_excel(writer, sheet_name='ticker2', index=False)
        writer.save()
        st.download_button('Download Excel', data=buf.getvalue(), file_name='market_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# -----------------------------
# Display: Basic metrics
# -----------------------------
if 'ticker1' in st.session_state['data']:
    df1 = st.session_state['data']['ticker1']
    latest = df1.iloc[-1]
    prev = df1.iloc[0]

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric('Ticker 1 Price', f"{latest['Close']:.2f}", delta=f"{((latest['Close']/prev['Close']-1)*100):.2f}%")
    if 'ticker2' in st.session_state['data']:
        df2 = st.session_state['data']['ticker2']
        latest2 = df2.iloc[-1]
        prev2 = df2.iloc[0]
        with colB:
            st.metric('Ticker 2 Price', f"{latest2['Close']:.2f}", delta=f"{((latest2['Close']/prev2['Close']-1)*100):.2f}%")
        with colC:
            ratio_val = latest['Close'] / latest2['Close'] if latest2['Close'] != 0 else np.nan
            st.metric('Ratio (T1/T2)', f"{ratio_val:.4f}")
    else:
        with colB:
            st.write('')
        with colC:
            st.write('')

# -----------------------------
# Ratio analysis
# -----------------------------
if ratio_enabled and 'ticker1' in st.session_state['data'] and 'ticker2' in st.session_state['data']:
    st.subheader('Ratio Analysis')
    df1 = st.session_state['data']['ticker1'].set_index('DateTime')
    df2 = st.session_state['data']['ticker2'].set_index('DateTime')
    # align by index
    merged = pd.concat([df1['Close'], df2['Close']], axis=1, join='inner')
    merged.columns = ['T1_Close','T2_Close']
    merged = merged.dropna()
    merged['Ratio'] = merged['T1_Close'] / merged['T2_Close']
    merged['RSI_T1'] = rsi(merged['T1_Close'])
    merged['RSI_T2'] = rsi(merged['T2_Close'])
    merged['RSI_Ratio'] = rsi(merged['Ratio'])
    merged = merged.reset_index()

    # Ratio bins
    merged['Ratio_Bin'] = assign_bins(merged['Ratio'], bins=5)
    st.dataframe(merged.head(200))

    # Binning summary
    bin_summary = merged.groupby('Ratio_Bin').agg(
        T1_Return_Pts=('T1_Close', lambda x: x.iloc[-1] - x.iloc[0]),
        T1_Return_Pct=('T1_Close', lambda x: (x.iloc[-1]/x.iloc[0]-1)*100),
        T2_Return_Pts=('T2_Close', lambda x: x.iloc[-1] - x.iloc[0]),
        T2_Return_Pct=('T2_Close', lambda x: (x.iloc[-1]/x.iloc[0]-1)*100),
        Count=('Ratio', 'count')
    ).reset_index()
    st.table(bin_summary)
    csv = merged.to_csv(index=False).encode('utf-8')
    st.download_button('Export Ratio CSV', data=csv, file_name='ratio_analysis.csv', mime='text/csv')

# -----------------------------
# Multi-timeframe analysis
# -----------------------------
if 'ticker1' in st.session_state['data']:
    st.subheader('Multi-Timeframe Analysis (Ticker 1)')
    # timeframe mapping: label -> interval
    timeframe_map = {
        '1m/1d': '1m',
        '5m/5d': '5m',
        '15m/5d': '15m',
        '30m/1mo': '30m',
        '1h/1mo': '60m',
        '2h/3mo': '120m',
        '4h/6mo': '240m',
        '1d/1y': '1d',
        '1w/5y': '1wk',
        '1mo/10y': '1mo'
    }
    period_map = {
        '1m/1d':'1d','5m/5d':'5d','15m/5d':'5d','30m/1mo':'1mo','1h/1mo':'1mo','2h/3mo':'3mo','4h/6mo':'6mo','1d/1y':'1y','1w/5y':'5y','1mo/10y':'10y'
    }
    if st.button('Run Multi-Timeframe Analysis for Ticker 1'):
        mtf_df = aggregate_timeframes(ticker1_sym, interval, timeframe_map, period_map, delay)
        st.dataframe(mtf_df)
        st.session_state['data']['mtf1'] = mtf_df

# -----------------------------
# Volatility bins
# -----------------------------
if 'ticker1' in st.session_state['data']:
    st.subheader('Volatility Bins (Ticker 1)')
    df = st.session_state['data']['ticker1']
    df = df.set_index('DateTime')
    df['VolatilityVal'] = compute_volatility(df)
    df['VolBin'] = assign_bins(df['VolatilityVal'], bins=5)
    vol_summary = df.groupby('VolBin').agg(
        VolMean=('VolatilityVal','mean'),
        MaxReturnPts=('Close', lambda x: x.diff().max()),
        MinReturnPts=('Close', lambda x: x.diff().min()),
        Count=('Close','count')
    ).reset_index()
    st.table(vol_summary)

# -----------------------------
# Pattern recognition
# -----------------------------
if 'ticker1' in st.session_state['data']:
    st.subheader('Pattern Recognition (Ticker 1)')
    df = st.session_state['data']['ticker1']
    pat_df = detect_patterns(df, move_threshold=move_threshold)
    if not pat_df.empty:
        st.write(f"Detected {len(pat_df)} moves above threshold")
        st.dataframe(pat_df)
    else:
        st.write('No significant patterns detected for the given threshold.')

# -----------------------------
# Charts
# -----------------------------
if 'ticker1' in st.session_state['data']:
    st.subheader('Interactive Charts')
    df = st.session_state['data']['ticker1']
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['DateTime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    # EMA overlays
    for p in [20,50,200]:
        if f'EMA_{p}' in df.columns:
            fig.add_trace(go.Scatter(x=df['DateTime'], y=df[f'EMA_{p}'], name=f'EMA {p}'))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    # RSI
    rfig = go.Figure()
    rfig.add_trace(go.Scatter(x=df['DateTime'], y=df['RSI'], name='RSI'))
    rfig.update_layout(height=200)
    st.plotly_chart(rfig, use_container_width=True)

# -----------------------------
# Distribution & Z-score
# -----------------------------
if 'ticker1' in st.session_state['data']:
    st.subheader('Statistical Distribution & Z-score')
    df = st.session_state['data']['ticker1']
    df['Returns'] = df['Close'].pct_change()
    returns = df['Returns'].dropna()
    mu = returns.mean()
    sd = returns.std()
    skew = stats.skew(returns.dropna())
    kurt = stats.kurtosis(returns.dropna())
    st.write(f'Mean: {mu:.6f}, Std: {sd:.6f}, Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}')
    # Histogram
    hist_fig = px.histogram(returns, nbins=80, marginal='box', title='Returns Distribution')
    st.plotly_chart(hist_fig, use_container_width=True)
    # Z-score table
    df['ZScore'] = zscore(df['Close'].pct_change().fillna(0)).rolling(z_window, min_periods=1).mean()
    ztable = df[['DateTime','Returns','ZScore']].tail(200)
    st.dataframe(ztable)

# -----------------------------
# Final recommendation
# -----------------------------
if 'mtf1' in st.session_state['data'] and 'ticker1' in st.session_state['data']:
    st.subheader('Final Trading Recommendation (Ticker 1)')
    mtf_df = st.session_state['data']['mtf1']
    df = st.session_state['data']['ticker1']
    rsi_val = float(df['RSI'].iloc[-1])
    recent_returns = df['Close'].pct_change().dropna().tail(z_window)
    z_val = float(zscore(recent_returns).iloc[-1]) if not recent_returns.empty else 0
    sig = generate_signal(mtf_df, rsi_val, z_val)
    st.metric('Signal', sig['recommendation'], delta=f"Score: {sig['score']:.3f}")
    # Position sizing example
    st.write('Position Sizing (example):')
    capital = st.number_input('Trading Capital (INR)', value=20000)
    risk_pct = st.number_input('Risk per trade (%)', value=1.0)
    atr_val = float(df['ATR_14'].iloc[-1]) if 'ATR_14' in df.columns else 0
    if atr_val>0:
        stop_loss = df['Close'].iloc[-1] - atr_val * 1.5 if sig['recommendation'] in ['Buy','Strong Buy'] else df['Close'].iloc[-1] + atr_val * 1.5
        rr = abs(( (df['Close'].iloc[-1] - stop_loss) / (atr_val if atr_val!=0 else 1)))
        size = (capital * (risk_pct/100)) / abs(df['Close'].iloc[-1] - stop_loss) if abs(df['Close'].iloc[-1] - stop_loss)>0 else 0
        st.write(f"Entry: {df['Close'].iloc[-1]:.2f}, Stop: {stop_loss:.2f}, Position size: {size:.2f} units, ATR: {atr_val:.4f}")

st.markdown('---')
st.caption('This is a comprehensive example app. For production use, further hardening, backtesting, and risk controls are required.')
