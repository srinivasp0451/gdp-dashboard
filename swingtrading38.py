# streamlit_algo_trader_app.py
# Streamlit app for multi-asset algorithmic trading analysis
# Features implemented:
# - Multi-asset support (common tickers + custom input)
# - Multiple timeframes & periods
# - Button-based fetching, persistent UI (session_state)
# - Timezone-aware IST conversion
# - Pattern recognition (multi-candle rallies, liquidity sweep, opening range, reversals)
# - Similarity search across historical windows
# - Forward forecast with point estimates and confidence
# - Heatmaps: day vs month volatility, month vs year volatility, returns, variance, median
# - Download CSV/Excel (OHLCV only) with timezone-aware datetimes
# - Graceful error handling, color-coded returns

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
from datetime import datetime, timedelta, time
import pytz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import base64

st.set_page_config(page_title="Algo Trader Analyzer", layout="wide")

# ----------------------------- Constants & Helpers -----------------------------
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

# Common tickers mapping (yfinance friendly)
COMMON_TICKERS = {
    'NIFTY 50': '^NSEI',
    'BANK NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'GOLD (MCX proxy)': 'GC=F',
    'SILVER (MCX proxy)': 'SI=F',
    'USDINR': 'INR=X',
}

# Supported intervals mapping to yfinance
ALLOWED_INTERVALS = ['1m','2m','3m','5m','10m','15m','30m','60m','90m','1h','2h','4h','1d']
# yfinance doesn't accept both '1h' and '60m' interchangeably; standardize
INTERVAL_MAP = {
    '1m':'1m','2m':'2m','3m':'3m','5m':'5m','10m':'10m','15m':'15m','30m':'30m',
    '60m':'60m','1h':'60m','2h':'120m','4h':'240m','90m':'90m','1d':'1d'
}

PERIODS = ['1d','5d','7d','1mo','3mo','6mo','1y','2y','3y','5y','6y','10y','15y','20y','25y','30y']

# Utility: convert index to IST timezone gracefully
def to_ist(df):
    if df is None or df.empty:
        return df
    idx = df.index
    try:
        # If tz-aware, convert to IST
        if idx.tz is not None:
            df.index = idx.tz_convert(IST)
        else:
            # Assume UTC if naive and localize then convert
            df.index = idx.tz_localize(UTC).tz_convert(IST)
    except Exception:
        # Last resort: treat as naive local time then localize to IST
        try:
            df.index = pd.DatetimeIndex(df.index).tz_localize(IST)
        except Exception:
            pass
    return df

# Safe fetch with caching and error handling
@st.cache_data(ttl=300)
def fetch_yf_data(ticker, period, interval):
    try:
        yf_ticker = yf.Ticker(ticker)
        # yfinance limitations: intraday max lookback depends on interval. user will be warned
        hist = yf_ticker.history(period=period, interval=interval, prepost=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        # Ensure columns: Open, High, Low, Close, Volume
        hist = hist[['Open','High','Low','Close','Volume']].copy()
        # some tickers return timezone aware index; convert to IST gracefully
        hist = to_ist(hist)
        hist.index.name = 'Datetime'
        return hist
    except Exception as e:
        st.session_state['last_error'] = str(e)
        return pd.DataFrame()

# ----------------------------- Pattern Detection -----------------------------

def candle_points(c):
    return (c['Close'] - c['Open']).astype(float)

# find multi-candle rallies: consecutive up or down candles that sum to threshold
def find_multi_candle_rallies(df, threshold_points=50, min_candles=2):
    sigs = []
    pts = candle_points(df)
    directions = np.sign(pts)
    # group by consecutive direction
    start = 0
    for i in range(1, len(pts)):
        if directions[i] != directions[i-1] or directions[i]==0:
            # evaluate segment
            seg = pts[start:i]
            if len(seg) >= min_candles and abs(seg.sum()) >= threshold_points:
                sigs.append({
                    'start': df.index[start], 'end': df.index[i-1], 'points': seg.sum(), 'candles': len(seg), 'direction': 'up' if seg.sum()>0 else 'down'
                })
            start = i
    # last segment
    seg = pts[start:len(pts)]
    if len(seg) >= min_candles and abs(seg.sum()) >= threshold_points:
        sigs.append({
            'start': df.index[start], 'end': df.index[-1], 'points': seg.sum(), 'candles': len(seg), 'direction': 'up' if seg.sum()>0 else 'down'
        })
    return sigs

# liquidity sweep: large single-bar range at a specific time
def find_liquidity_sweeps(df, sweep_points=100, hour=11, minute=0):
    sweeps = []
    for dt, row in df.iterrows():
        if dt.hour == hour and dt.minute == minute:
            rng = row['High'] - row['Low']
            if rng >= sweep_points:
                sweeps.append({'datetime': dt, 'range': rng, 'open': row['Open'], 'close': row['Close']})
    return sweeps

# opening range pattern
def opening_range(df, start_time=time(9,15), end_time=time(9,18)):
    mask = [(idx.time() >= start_time and idx.time() <= end_time) for idx in df.index]
    window = df.iloc[np.where(mask)[0]] if any(mask) else pd.DataFrame()
    if window.empty:
        return None
    direction = 'up' if window['Close'].iloc[-1] > window['Open'].iloc[0] else 'down'
    points = (window['Close'].iloc[-1] - window['Open'].iloc[0])
    return {'start': window.index[0], 'end': window.index[-1], 'direction': direction, 'points': points}

# reversal after rally: detect rally then opposite large candle
def find_reversals_after_rally(df, rally_candles=3, rally_points=50, reversal_points=30):
    pts = candle_points(df)
    results = []
    for i in range(len(pts)-rally_candles-1):
        seg = pts[i:i+rally_candles]
        if seg.sum() >= rally_points:
            # look next candle
            nxt = pts[i+rally_candles]
            if nxt < -reversal_points:
                results.append({'rally_start': df.index[i], 'rally_end': df.index[i+rally_candles-1], 'rally_points': seg.sum(), 'reversal_time': df.index[i+rally_candles], 'reversal_points': nxt})
        if seg.sum() <= -rally_points:
            nxt = pts[i+rally_candles]
            if nxt > reversal_points:
                results.append({'rally_start': df.index[i], 'rally_end': df.index[i+rally_candles-1], 'rally_points': seg.sum(), 'reversal_time': df.index[i+rally_candles], 'reversal_points': nxt})
    return results

# create a signature for windows for similarity search
def window_signature(df_window):
    # signature: normalized point changes per candle and direction vector
    pts = (df_window['Close'] - df_window['Open']).values
    if pts.std() == 0:
        return pts - pts.mean()
    return (pts - pts.mean()) / (pts.std()+1e-9)

# search historical windows similar to target signature
def find_similar_windows(df, target_window, window_size=5, top_n=10):
    sig = window_signature(target_window)
    hist = df
    candidates = []
    for i in range(len(hist)-window_size):
        win = hist.iloc[i:i+window_size]
        s = window_signature(win)
        if len(s) != len(sig):
            continue
        # distance
        dist = np.linalg.norm(sig - s)
        candidates.append((i, dist))
    candidates = sorted(candidates, key=lambda x: x[1])[:top_n]
    results = []
    for idx, d in candidates:
        forward_window = hist.iloc[idx+window_size: idx+window_size+3]  # look 3 candles forward
        fwd_points = (forward_window['Close'].iloc[-1] - forward_window['Close'].iloc[0]) if not forward_window.empty else np.nan
        results.append({'start': hist.index[idx], 'distance': d, 'forward_points': fwd_points})
    return results

# Forecast: aggregate forward_points from matches
def forecast_from_matches(matches):
    points = [m['forward_points'] for m in matches if not pd.isna(m['forward_points'])]
    if len(points) == 0:
        return {'expected_points': 0.0, 'confidence': 0.0, 'n_matches': 0}
    expected = np.mean(points)
    # Confidence: ratio of matches that moved in same direction as mean
    same_dir = sum(1 for p in points if np.sign(p)==np.sign(expected))
    confidence = same_dir / len(points)
    return {'expected_points': float(expected), 'confidence': float(confidence), 'n_matches': len(points)}

# ----------------------------- Heatmaps & Summaries -----------------------------

def compute_volatility(df, period='D'):
    # returns per period volatility (std of returns)
    ret = df['Close'].pct_change().dropna()
    vol = ret.resample(period).std()
    return vol

# generic heatmap maker using pivot table
def make_heatmap_data(df, freq_outer, freq_inner, agg='std'):
    # freq_outer: e.g. 'M' months or 'Y' years, freq_inner: day of month or month of year
    tmp = df['Close'].pct_change().dropna().to_frame('ret')
    tmp['outer'] = tmp.index.to_period(freq_outer).to_timestamp()
    tmp['inner'] = tmp.index.to_period(freq_inner).to_timestamp()
    pivot = tmp.groupby(['outer','inner']).agg({'ret': agg}).unstack(level=0)['ret']
    return pivot.fillna(0)

# ----------------------------- Export helpers -----------------------------

def df_to_csv_bytes(df):
    buf = io.BytesIO()
    df_to_save = df[['Open','High','Low','Close','Volume']].copy()
    # keep tz-aware index: convert to ISO string with timezone
    df_to_save = df_to_save.reset_index()
    df_to_save['Datetime'] = df_to_save['Datetime'].apply(lambda x: x.isoformat())
    df_to_save.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def df_to_excel_bytes(df):
    buf = io.BytesIO()
    df_to_save = df[['Open','High','Low','Close','Volume']].copy()
    df_to_save = df_to_save.reset_index()
    df_to_save['Datetime'] = df_to_save['Datetime'].apply(lambda x: x.isoformat())
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df_to_save.to_excel(writer, index=False, sheet_name='OHLCV')
    buf.seek(0)
    return buf

# ----------------------------- Streamlit UI -----------------------------

st.title('Algo Trader Analyzer — Multi-Asset Streamlit')

# Sidebar controls
with st.sidebar:
    st.header('Data & Options')
    asset_choice = st.selectbox('Pick asset (or choose Custom)', list(COMMON_TICKERS.keys())+['Custom'])
    if asset_choice == 'Custom':
        ticker_input = st.text_input('Enter yfinance ticker (eg: RELIANCE.NS or AAPL)', value='^NSEI')
    else:
        ticker_input = COMMON_TICKERS[asset_choice]

    interval = st.selectbox('Interval / Timeframe', ['1m','3m','5m','10m','15m','30m','60m','2h','4h','1d'])
    period = st.selectbox('Period', PERIODS, index=1)
    min_trend_points = st.number_input('Threshold points for multi-candle rally', value=50)
    sweep_points = st.number_input('Sweep points threshold', value=100)
    window_size = st.number_input('Similarity window size (candles)', value=5, min_value=2)
    top_n = st.number_input('Top similar matches to retrieve', value=10, min_value=1)
    run_fetch = st.button('Fetch & Analyze')
    st.markdown('---')
    st.markdown('Export data:')
    export_csv = st.button('Download CSV (OHLCV)')
    export_xlsx = st.button('Download Excel (OHLCV)')
    st.markdown('Notes: Data pulled from Yahoo Finance (yfinance). Intraday history availability depends on ticker & interval.')

# Persist ticker / settings
if 'settings' not in st.session_state:
    st.session_state['settings'] = {}
st.session_state['settings'].update({'ticker': ticker_input, 'interval': interval, 'period': period})

# Handle fetch button — do not fetch on refresh
if run_fetch:
    with st.spinner('Fetching data — please wait...'):
        interval_mapped = INTERVAL_MAP.get(interval, interval)
        df = fetch_yf_data(ticker_input, period, interval_mapped)
        if df.empty:
            st.warning('No data returned for this ticker/interval/period. Try a different combination.')
        else:
            st.session_state['df'] = df
            st.session_state['last_fetch_time'] = datetime.now(IST).isoformat()
            st.success(f'Fetched {len(df)} rows — last point {df.index[-1].isoformat()}')

# If data exists in session, use it
df = st.session_state.get('df', pd.DataFrame())

# Export handling — use the session df
if export_csv and not df.empty:
    buf = df_to_csv_bytes(df)
    st.download_button('Download CSV', data=buf, file_name=f'data_{ticker_input}.csv', mime='text/csv')

if export_xlsx and not df.empty:
    buf = df_to_excel_bytes(df)
    st.download_button('Download Excel', data=buf, file_name=f'data_{ticker_input}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Main area: show data, charts, patterns
if df.empty:
    st.info('No data in memory. Click *Fetch & Analyze* to load data. Components will persist after click.')
else:
    st.subheader(f'Data snapshot — {ticker_input} — last fetched: {st.session_state.get("last_fetch_time")}')
    # show head tail
    st.dataframe(df.tail(20))

    # Basic stats: points change, percent change since previous day
    df_points = df.copy()
    df_points['Points'] = df_points['Close'] - df_points['Open']
    last = df_points.iloc[-1]
    prev_day_close = df_points['Close'].resample('D').last().shift(1).dropna()
    prev_close = prev_day_close.iloc[-1] if len(prev_day_close)>0 else np.nan
    pct_change_from_prev_day = (last['Close'] - prev_close) / prev_close * 100 if not pd.isna(prev_close) and prev_close!=0 else np.nan

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Last Close', f"{last['Close']:.2f}")
    with col2:
        if not pd.isna(last['Points']):
            delta = last['Points']
            if delta >= 0:
                st.markdown(f"<div style='color:green; font-weight:bold;'>Points gained: {delta:.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:red; font-weight:bold;'>Points lost: {delta:.2f}</div>", unsafe_allow_html=True)
    with col3:
        if not pd.isna(pct_change_from_prev_day):
            color = 'green' if pct_change_from_prev_day>=0 else 'red'
            st.markdown(f"<div style='color:{color}; font-weight:bold;'>% from prev day: {pct_change_from_prev_day:.2f}%</div>", unsafe_allow_html=True)

    # Simple plot: OHLC range plot (matplotlib simple)
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(df.index, df['High'], label='High')
    ax.plot(df.index, df['Low'], label='Low')
    ax.set_title('Max/Min Ranges')
    ax.legend()
    st.pyplot(fig)

    # Patterns
    st.subheader('Detected Patterns')
    # Multi-candle rallies
    rallies = find_multi_candle_rallies(df, threshold_points=min_trend_points, min_candles=2)
    st.markdown('**Multi-candle rallies (threshold applied)**')
    if rallies:
        for r in rallies[-10:]:
            color = 'green' if r['direction']=='up' else 'red'
            st.markdown(f"- {r['start'].isoformat()} to {r['end'].isoformat()}: {r['points']:.1f} pts over {r['candles']} candles — <span style='color:{color}'>{r['direction']}</span>", unsafe_allow_html=True)
    else:
        st.markdown('No multi-candle rallies found matching threshold.')

    # Liquidity sweeps
    sweeps = find_liquidity_sweeps(df, sweep_points=sweep_points, hour=11, minute=0)
    st.markdown('**Liquidity sweeps around 11:00**')
    if sweeps:
        for s in sweeps:
            st.markdown(f"- {s['datetime'].isoformat()}: range {s['range']:.2f} pts (open {s['open']}, close {s['close']})")
    else:
        st.markdown('No liquidity sweeps at 11:00 matching threshold.')

    # Opening range
    orange = opening_range(df)
    st.markdown('**Opening range (9:15 - 9:18 IST)**')
    if orange:
        dirc = orange['direction']
        coltxt = 'green' if dirc=='up' else 'red'
        st.markdown(f"Direction: <span style='color:{coltxt}; font-weight:bold'>{dirc}</span>, Points: {orange['points']:.2f}", unsafe_allow_html=True)
    else:
        st.markdown('Opening range not present in this dataset/timeframe.')

    # Reversals
    reversals = find_reversals_after_rally(df)
    st.markdown('**Reversals after rally**')
    if reversals:
        for rv in reversals[-10:]:
            st.markdown(f"- Rally {rv['rally_start'].isoformat()} to {rv['rally_end'].isoformat()} ({rv['rally_points']:.1f} pts) -> reversal at {rv['reversal_time'].isoformat()} ({rv['reversal_points']:.1f} pts)")
    else:
        st.markdown('No reversal-after-rally patterns detected.')

    # Similarity search: take last window_size candles as target
    st.subheader('Similarity Search & Forecast')
    if len(df) >= window_size+3:
        target = df.iloc[-window_size:]
        matches = find_similar_windows(df, target, window_size=window_size, top_n=top_n)
        st.markdown(f'Found {len(matches)} similar windows (top {top_n}) — showing start, distance, forward points')
        if matches:
            match_df = pd.DataFrame(matches)
            st.dataframe(match_df)
            fc = forecast_from_matches(matches)
            direction = 'up' if fc['expected_points']>0 else ('down' if fc['expected_points']<0 else 'flat')
            conf_color = 'green' if fc['confidence']>=0.6 else ('orange' if fc['confidence']>=0.4 else 'red')
            st.markdown('**Forecast Summary**')
            st.markdown(f"Expected move: <b>{fc['expected_points']:.2f} points</b> — Direction: <span style='color:{conf_color}; font-weight:bold'>{direction}</span>", unsafe_allow_html=True)
            st.markdown(f"Confidence: {fc['confidence']*100:.1f}% based on {fc['n_matches']} historical matches")
        else:
            st.markdown('No similar historical windows found.')
    else:
        st.info('Not enough data for similarity search — increase period or reduce window size.')

    # Heatmaps
    st.subheader('Heatmaps & Statistics')
    # Day vs Month volatility heatmap: compute volatility per day-of-month vs month
    try:
        vol_day_month = make_heatmap_data(df, freq_outer='M', freq_inner='D', agg='std')
        if not vol_day_month.empty:
            fig2, ax2 = plt.subplots(figsize=(10,4))
            cax = ax2.imshow(vol_day_month.T, aspect='auto')
            ax2.set_title('Day vs Month volatility (std of returns)')
            ax2.set_yticks(range(vol_day_month.shape[1]))
            ax2.set_yticklabels([d.strftime('%Y-%m') for d in vol_day_month.columns])
            ax2.set_xticks(range(vol_day_month.shape[0]))
            ax2.set_xticklabels([d.strftime('%d') for d in vol_day_month.index], rotation=90)
            st.pyplot(fig2)
            st.markdown('Summary: Higher values indicate higher volatility days in that month. If recent column shows cooler colors, volatility decreased.')
        else:
            st.markdown('Insufficient data for Day vs Month volatility heatmap.')
    except Exception as e:
        st.warning('Heatmap generation failed gracefully.')

    # Month vs Year volatility heatmap
    try:
        vol_month_year = make_heatmap_data(df, freq_outer='Y', freq_inner='M', agg='std')
        if not vol_month_year.empty:
            fig3, ax3 = plt.subplots(figsize=(10,4))
            ax3.imshow(vol_month_year.T, aspect='auto')
            ax3.set_title('Month vs Year volatility (std of returns)')
            ax3.set_yticks(range(vol_month_year.shape[1]))
            ax3.set_yticklabels([d.strftime('%Y') for d in vol_month_year.columns])
            ax3.set_xticks(range(vol_month_year.shape[0]))
            ax3.set_xticklabels([d.strftime('%b') for d in vol_month_year.index], rotation=90)
            st.pyplot(fig3)
            st.markdown('Summary: Look for months consistently hotter across years for seasonal volatility.')
        else:
            st.markdown('Insufficient data for Month vs Year volatility heatmap.')
    except Exception:
        st.warning('Heatmap generation failed gracefully.')

    # Returns heatmap (median returns by day vs month)
    try:
        ret_pivot = make_heatmap_data(df, freq_outer='M', freq_inner='D', agg='median')
        if not ret_pivot.empty:
            fig4, ax4 = plt.subplots(figsize=(10,4))
            ax4.imshow(ret_pivot.T, aspect='auto')
            ax4.set_title('Median returns: Day vs Month')
            st.pyplot(fig4)
            st.markdown('Summary: Positive median indicates upward bias on those days historically.')
        else:
            st.markdown('Insufficient data for Returns heatmap.')
    except Exception:
        st.warning('Heatmap generation failed gracefully.')

    # Variance heatmap
    try:
        var_pivot = make_heatmap_data(df, freq_outer='M', freq_inner='D', agg='var')
        if not var_pivot.empty:
            fig5, ax5 = plt.subplots(figsize=(10,4))
            ax5.imshow(var_pivot.T, aspect='auto')
            ax5.set_title('Variance heatmap: Day vs Month')
            st.pyplot(fig5)
            st.markdown('Summary: High variance cells are riskier — expect larger swings.')
        else:
            st.markdown('Insufficient data for Variance heatmap.')
    except Exception:
        st.warning('Heatmap generation failed gracefully.')

    # Median heatmap is same as ret_pivot above

    # Final summary of key insights
    st.subheader('Key Insights Summary')
    insights = []
    if rallies:
        insights.append(f"Recent multi-candle rallies: {len(rallies)} (latest {rallies[-1]['points']:.1f} pts)")
    if sweeps:
        insights.append(f"Liquidity sweeps found: {len(sweeps)} (last at {sweeps[-1]['datetime'].isoformat()})")
    if orange:
        insights.append(f"Opening range direction: {orange['direction']} ({orange['points']:.1f} pts)")
    if not pd.isna(pct_change_from_prev_day):
        insights.append(f"% change from previous day: {pct_change_from_prev_day:.2f}%")
    if 'fc' in locals():
        insights.append(f"Forecast: {fc['expected_points']:.2f} pts, confidence {fc['confidence']*100:.1f}%")
    if len(insights)==0:
        st.write('Not enough signals to summarize. Try different thresholds or expand the period.')
    else:
        for it in insights:
            st.write('- '+it)

    st.success('Analysis complete — use the download buttons in sidebar to export OHLCV data. All datetimes preserved in ISO with timezone info.')

# Show errors gracefully if any
if st.session_state.get('last_error'):
    st.error('A non-fatal error occurred during fetch: ' + st.session_state.get('last_error'))

# Footer
st.markdown('---')
st.caption('Built for analysis and idea generation. Not financial advice. Validate signals with your risk management rules before trading.')
