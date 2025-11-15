# streamlit_algo_trader_app.py
# Streamlit app for multi-asset algorithmic trading analysis (UPDATED)
# Changes in this update (per user request):
# - Fixed timezone conversion to IST robustly for all yfinance outputs
# - Heatmaps now show annotated values in cells and use readable colormap + colorbar
# - Similarity search: returns exact timestamps/dates for matches, count, distribution of forward moves
# - Volatility-based analysis added: rolling volatility, volatility-percentile selector, and conditional pattern scans
# - Pattern scan: allows simple sequences (e.g., 2 red then 1 green) and reports historical outcomes
# - Added second ticker comparison option to compare patterns across two instruments
# - Improved error handling and clearer layman summaries for detected patterns

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
from datetime import datetime, timedelta, time
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import base64

st.set_page_config(page_title="Algo Trader Analyzer (Updated)", layout="wide")

# ----------------------------- Constants & Helpers -----------------------------
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

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

INTERVAL_MAP = {
    '1m':'1m','3m':'3m','5m':'5m','10m':'10m','15m':'15m','30m':'30m','60m':'60m','2h':'120m','4h':'240m','1d':'1d'
}
PERIODS = ['5d','7d','1mo','3mo','6mo','1y','2y','3y','5y','10y']

# Robust IST conversion
def to_ist(df):
    if df is None or df.empty:
        return df
    # ensure datetime index
    try:
        idx = pd.DatetimeIndex(df.index)
    except Exception:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        idx = pd.DatetimeIndex(df.index)
    # If tz-aware, convert. If naive, assume UTC then convert to IST
    if idx.tz is None:
        try:
            df.index = idx.tz_localize(UTC).tz_convert(IST)
        except Exception:
            try:
                df.index = idx.tz_localize(IST)
            except Exception:
                pass
    else:
        try:
            df.index = idx.tz_convert(IST)
        except Exception:
            pass
    return df

# Caching fetch
@st.cache_data(ttl=300)
def fetch_yf_data(ticker, period, interval):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, prepost=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        hist = hist[['Open','High','Low','Close','Volume']].copy()
        hist = to_ist(hist)
        hist.index.name = 'Datetime'
        return hist
    except Exception as e:
        st.session_state['last_error'] = str(e)
        return pd.DataFrame()

# small helpers
def candle_points(df):
    return (df['Close'] - df['Open']).astype(float)

def candle_signs(df):
    pts = candle_points(df)
    return ['G' if p>0 else ('R' if p<0 else 'N') for p in pts]

# find multi-candle rallies
def find_multi_candle_rallies(df, threshold_points=50, min_candles=2):
    pts = candle_points(df)
    signs = np.sign(pts)
    results = []
    start = 0
    for i in range(1, len(pts)):
        if signs[i] != signs[i-1] or signs[i]==0:
            seg = pts[start:i]
            if len(seg) >= min_candles and abs(seg.sum()) >= threshold_points:
                results.append({'start': df.index[start], 'end': df.index[i-1], 'points': seg.sum(), 'candles': len(seg), 'direction': 'up' if seg.sum()>0 else 'down'})
            start = i
    seg = pts[start:len(pts)]
    if len(seg) >= min_candles and abs(seg.sum()) >= threshold_points:
        results.append({'start': df.index[start], 'end': df.index[-1], 'points': seg.sum(), 'candles': len(seg), 'direction': 'up' if seg.sum()>0 else 'down'})
    return results

# liquidity sweeps (time-based)
def find_liquidity_sweeps(df, sweep_points=100, hour=11, minute=0):
    sweeps = []
    for dt, row in df.iterrows():
        if dt.hour == hour and dt.minute == minute:
            rng = row['High'] - row['Low']
            if rng >= sweep_points:
                sweeps.append({'datetime': dt, 'range': rng, 'open': row['Open'], 'close': row['Close']})
    return sweeps

# opening range
def opening_range(df, start_time=time(9,15), end_time=time(9,18)):
    mask = [(idx.time() >= start_time and idx.time() <= end_time) for idx in df.index]
    idxs = [i for i,m in enumerate(mask) if m]
    if not idxs:
        return None
    window = df.iloc[idxs[0]:idxs[-1]+1]
    direction = 'up' if window['Close'].iloc[-1] > window['Open'].iloc[0] else 'down'
    points = window['Close'].iloc[-1] - window['Open'].iloc[0]
    return {'start': window.index[0], 'end': window.index[-1], 'direction': direction, 'points': points}

# similarity signature
def window_signature(df_window):
    pts = (df_window['Close'] - df_window['Open']).values
    if pts.std() == 0:
        return pts - pts.mean()
    return (pts - pts.mean()) / (pts.std()+1e-9)

# find similar windows with full metadata
def find_similar_windows_with_details(df, target_window, window_size=5, top_n=20):
    sig = window_signature(target_window)
    hist = df.copy()
    candidates = []
    for i in range(len(hist)-window_size-3):
        win = hist.iloc[i:i+window_size]
        s = window_signature(win)
        if len(s) != len(sig):
            continue
        dist = np.linalg.norm(sig - s)
        forward_window = hist.iloc[i+window_size:i+window_size+3]
        fwd_points = (forward_window['Close'].iloc[-1] - forward_window['Close'].iloc[0]) if not forward_window.empty else np.nan
        candidates.append({'index': i, 'start': hist.index[i], 'end': hist.index[i+window_size-1], 'distance': dist, 'forward_points': fwd_points})
    candidates = sorted(candidates, key=lambda x: x['distance'])[:top_n]
    return candidates

# forecast aggregation with distribution
def forecast_stats(matches):
    pts = [m['forward_points'] for m in matches if not pd.isna(m['forward_points'])]
    if not pts:
        return {'mean':0,'median':0,'count':0,'pct_pos':0,'std':0}
    arr = np.array(pts)
    return {'mean':float(arr.mean()), 'median':float(np.median(arr)), 'count':len(arr), 'pct_pos':float((arr>0).sum()/len(arr)), 'std':float(arr.std())}

# volatility helpers
def rolling_volatility(df, window=20):
    ret = df['Close'].pct_change()
    vol = ret.rolling(window=window).std()*np.sqrt(window)
    return vol

# pattern scanning example: detect sequences like ['R','R','G'] in sliding windows
def scan_sequence_pattern(df, seq):
    signs = candle_signs(df)
    L = len(seq)
    matches = []
    for i in range(len(signs)-L-2):
        window = signs[i:i+L]
        if window == seq:
            forward_pts = df['Close'].iloc[i+L+2] - df['Close'].iloc[i+L] if (i+L+2) < len(df) else np.nan
            matches.append({'start': df.index[i], 'end': df.index[i+L-1], 'forward_points': forward_pts})
    return matches

# export
def df_to_csv_bytes(df):
    buf = io.BytesIO()
    save = df[['Open','High','Low','Close','Volume']].reset_index()
    save['Datetime'] = save['Datetime'].apply(lambda x: x.isoformat())
    save.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ----------------------------- UI -----------------------------
st.title('Algo Trader Analyzer — Updated')

with st.sidebar:
    st.header('Data & Analysis Options')
    asset_choice = st.selectbox('Primary asset', list(COMMON_TICKERS.keys())+['Custom'])
    if asset_choice == 'Custom':
        ticker = st.text_input('Primary ticker', value='^NSEI')
    else:
        ticker = COMMON_TICKERS[asset_choice]

    compare_choice = st.selectbox('Compare with (optional)', list(COMMON_TICKERS.keys())+['None','Custom'])
    if compare_choice == 'Custom':
        ticker2 = st.text_input('Secondary ticker', value='^NSEBANK')
    elif compare_choice == 'None':
        ticker2 = None
    else:
        ticker2 = COMMON_TICKERS.get(compare_choice)

    interval = st.selectbox('Interval', ['1m','3m','5m','10m','15m','30m','60m','2h','4h','1d'])
    period = st.selectbox('Period', PERIODS, index=2)
    run = st.button('Fetch & Analyze')
    st.markdown('---')
    st.subheader('Similarity / Volatility')
    window_size = st.number_input('Similarity window (candles)', min_value=2, value=5)
    top_n = st.number_input('Top matches to show', min_value=1, value=10)
    vol_window = st.number_input('Volatility rolling window (candles)', min_value=2, value=20)
    vol_percentile = st.slider('Volatility percentile threshold (show windows above this)', 50, 100, 75)
    seq_input = st.text_input('Sequence pattern (e.g., RR G as RRG -> enter RRG)', value='RRG')
    st.markdown('Sequence legend: R=red (down), G=green (up), N=neutral')
    st.markdown('---')
    st.markdown('Export:')
    export_csv = st.button('Export OHLCV CSV')

# Session persist
if 'settings' not in st.session_state:
    st.session_state.settings = {}
st.session_state.settings.update({'ticker':ticker,'ticker2':ticker2,'interval':interval,'period':period})

if run:
    with st.spinner('Fetching data...'):
        try:
            intv = INTERVAL_MAP.get(interval, interval)
            df = fetch_yf_data(ticker, period, intv)
            if ticker2:
                df2 = fetch_yf_data(ticker2, period, intv)
            else:
                df2 = pd.DataFrame()
            if df.empty:
                st.error('No data for primary ticker. Try different period/interval or ticker.')
            else:
                st.session_state.df = df
                st.session_state.df2 = df2
                st.session_state.last_fetch = datetime.now(IST).isoformat()
                st.success(f'Fetched {len(df)} rows. Time index (IST): {df.index[0].isoformat()} ... {df.index[-1].isoformat()}')
        except Exception as e:
            st.error('Fetch failed: '+str(e))

# load from session
df = st.session_state.get('df', pd.DataFrame())
df2 = st.session_state.get('df2', pd.DataFrame())

if export_csv and not df.empty:
    buf = df_to_csv_bytes(df)
    st.download_button('Download CSV', data=buf, file_name=f'{ticker}_ohlcv.csv', mime='text/csv')

if df.empty:
    st.info('No data loaded. Click Fetch & Analyze.')
else:
    st.subheader(f'Data snapshot — {ticker} (IST index)')
    st.write('Data index range:', df.index[0].isoformat(), 'to', df.index[-1].isoformat())
    st.dataframe(df.tail(20))

    # Basic info
    df_points = df.copy()
    df_points['Points'] = df_points['Close'] - df_points['Open']
    last = df_points.iloc[-1]
    st.metric('Last Close', f"{last['Close']:.2f}")
    st.markdown(f"**Last candle points:** <span style='color:{'green' if last['Points']>=0 else 'red'}'>{last['Points']:.2f}</span>", unsafe_allow_html=True)

    # Multi-candle rallies
    st.subheader('Multi-candle rallies')
    rallies = find_multi_candle_rallies(df, threshold_points=50, min_candles=2)
    if rallies:
        st.write(f'Found {len(rallies)} multi-candle rallies (threshold 50 pts). Showing last 10:')
        for r in rallies[-10:]:
            st.write(f"{r['start'].isoformat()} to {r['end'].isoformat()} — {r['points']:.1f} pts over {r['candles']} candles ({r['direction']})")
    else:
        st.write('No multi-candle rallies found matching threshold.')

    # Similarity search detailed
    st.subheader('Similarity search (detailed)')
    if len(df) >= window_size+3:
        target = df.iloc[-window_size:]
        matches = find_similar_windows_with_details(df, target, window_size=window_size, top_n=top_n)
        if matches:
            st.write(f'Top {len(matches)} matches: (start -> end) distance | forward_points (next 3 candles)')
            mm = pd.DataFrame(matches)
            mm_display = mm.copy()
            mm_display['start'] = mm_display['start'].dt.strftime('%Y-%m-%d %H:%M')
            mm_display['end'] = mm_display['end'].dt.strftime('%Y-%m-%d %H:%M')
            mm_display['forward_points'] = mm_display['forward_points'].round(2)
            st.dataframe(mm_display[['start','end','distance','forward_points']])
            stats = forecast_stats(matches)
            st.markdown('**Forecast statistics from matched windows**')
            st.write(f"Matches: {stats['count']}, Mean forward points: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std: {stats['std']:.2f}, % positive: {stats['pct_pos']*100:.1f}%")

            # show example matched charts (overlay target and best match)
            best = matches[0]
            example_win = df.iloc[best['index']: best['index']+window_size]
            fig, axs = plt.subplots(1,2,figsize=(10,3))
            axs[0].plot(target['Close'].values, marker='o'); axs[0].set_title('Target window (Close)')
            axs[1].plot(example_win['Close'].values, marker='o'); axs[1].set_title(f'Best match start {best["start"].strftime("%Y-%m-%d %H:%M")}')
            st.pyplot(fig)

            # show where matches occurred in calendar (list dates and counts)
            starts = [m['start'].date() for m in matches]
            dates_count = pd.Series(starts).value_counts().sort_index()
            st.write('Matches by date:')
            st.dataframe(dates_count.rename('count'))
        else:
            st.write('No matches found.')
    else:
        st.info('Not enough candles for similarity search. Increase period or reduce window size.')

    # Sequence-based pattern scan
    st.subheader('Sequence pattern scan')
    seq = list(seq_input.strip().upper())
    if any(c not in ['R','G','N'] for c in seq):
        st.error('Sequence must be composed of R, G, N only.')
    else:
        seq_matches = scan_sequence_pattern(df, seq)
        st.write(f"Found {len(seq_matches)} occurrences of sequence {''.join(seq)}")
        if seq_matches:
            seq_df = pd.DataFrame(seq_matches)
            seq_df['start'] = seq_df['start'].dt.strftime('%Y-%m-%d %H:%M')
            seq_df['end'] = seq_df['end'].dt.strftime('%Y-%m-%d %H:%M')
            seq_df['forward_points'] = seq_df['forward_points'].round(2)
            st.dataframe(seq_df)
            # summary
            arr = seq_df['forward_points'].dropna().values
            st.write(f"After this sequence, mean forward points: {arr.mean():.2f}, median: {np.median(arr):.2f}, % positive: {(arr>0).sum()/len(arr)*100:.1f}%")
        else:
            st.write('No sequence occurrences found in this dataset/timeframe.')

    # Volatility analysis
    st.subheader('Volatility-based analysis (rolling)')
    vol = rolling_volatility(df, window=vol_window)
    df_vol = df.copy(); df_vol['vol'] = vol
    # pick threshold as percentile
    thr = np.nanpercentile(df_vol['vol'].dropna(), vol_percentile)
    st.write(f'Rolling vol threshold (percentile {vol_percentile}) = {thr:.6f}')
    high_vol_windows = df_vol[df_vol['vol'] >= thr]
    st.write(f'Found {len(high_vol_windows)} candles above threshold')
    # For each high-vol candle examine following 3 candles move
    outcomes = []
    for idx in high_vol_windows.index:
        try:
            i = df.index.get_loc(idx)
            if i+3 < len(df):
                fwd = df['Close'].iloc[i+3] - df['Close'].iloc[i]
                outcomes.append(fwd)
        except Exception:
            continue
    if outcomes:
        arr = np.array(outcomes)
        st.write(f'After high-vol candles (above {vol_percentile}th pct), forward 3-candle mean: {arr.mean():.2f}, % positive: {(arr>0).sum()/len(arr)*100:.1f}%')
    else:
        st.write('No valid forward outcomes found for high volatility windows.')

    # Heatmaps with annotations
    st.subheader('Annotated Heatmaps')
    try:
        # Day vs Month volatility heatmap (median volatility)
        tmp = df['Close'].pct_change().dropna().to_frame('ret')
        tmp['month'] = tmp.index.to_period('M').to_timestamp()
        tmp['day'] = tmp.index.day
        pivot = tmp.groupby(['day','month']).ret.std().unstack(fill_value=0)
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(12,6))
            sns.heatmap(pivot, ax=ax, annot=True, fmt='.4f', cmap='viridis', cbar_kws={'label':'Std(ret)'} )
            ax.set_title('Day (rows) vs Month (cols) volatility (std of returns)')
            st.pyplot(fig)
            st.write('Explanation: cells with larger numbers indicate historically higher volatility on that day/month combination.')
        else:
            st.write('Insufficient data for heatmap.')
    except Exception as e:
        st.warning('Heatmap generation error: '+str(e))

    # Compare with secondary ticker if provided
    if ticker2 and not df2.empty:
        st.subheader(f'Comparison with {ticker2}')
        st.write('Showing last 20 rows of second ticker (IST):')
        st.dataframe(df2.tail(20))
        # cross-match: find dates where both had similar direction sequences
        # simple example: count days where both had opening range same direction
        or1 = opening_range(df)
        or2 = opening_range(df2)
        if or1 and or2:
            same = or1['direction'] == or2['direction']
            st.write(f'Opening range direction — primary: {or1["direction"]}, secondary: {or2["direction"]}. Same direction? {same}')

    # Final plain-language summary
    st.subheader('Plain-language Summary')
    summary = []
    if rallies:
        summary.append(f"There were {len(rallies)} historical multi-candle rallies (>=50 pts). Latest rally was {rallies[-1]['points']:.1f} pts ending at {rallies[-1]['end'].strftime('%Y-%m-%d %H:%M')}")
    if seq_matches:
        arr = np.array([m['forward_points'] for m in seq_matches if not pd.isna(m['forward_points'])])
        if len(arr)>0:
            summary.append(f"Sequence {''.join(seq)} occurred {len(seq_matches)} times. After it, mean forward {arr.mean():.2f} pts and {((arr>0).sum()/len(arr))*100:.1f}% of occurrences moved up.")
    if outcomes:
        summary.append(f"High-volatility candles (>{vol_percentile}th pct) historically led to mean {np.mean(outcomes):.2f} pts in the next 3 candles, positive {((np.array(outcomes)>0).sum()/len(outcomes))*100:.1f}% of times.")
    if not summary:
        st.write('No strong signals found with current parameters. Try adjusting window sizes, percentile, or period.')
    else:
        for s in summary:
            st.write('- '+s)

st.markdown('---')
st.caption('Updated: timezone fixed to IST, annotated heatmaps, sequence and volatility-based similarity scans, secondary ticker comparison. Not financial advice.')
