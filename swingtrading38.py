import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from io import BytesIO

st.set_page_config(layout="wide", page_title="Algo Trader Assistant (Corrected)")

# -------------------- Robust helpers & fixes --------------------
@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetches data from yfinance with caching to reduce rate-limit. Returns raw yfinance DataFrame (may be multiindex)."""
    try:
        raw = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=True)
    except Exception as e:
        st.error(f'yfinance error: {e}')
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw.index = pd.to_datetime(raw.index)
    return raw


def flatten_and_map(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """Flatten potential MultiIndex columns and map/standardize column names to Open/High/Low/Close/Volume.
    Returns (df, mapping) where mapping shows how original columns were mapped.
    """
    if df is None or df.empty:
        return df, {}
    df = df.copy()
    # Flatten multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = ["_".join([str(i) for i in col if i and str(i) != '']) for col in df.columns]
        df.columns = flat_cols
    # create mapping by searching keywords
    mapping = {}
    # possible keywords in many yfinance outputs
    candidates = df.columns.tolist()
    # map priorities
    for std in ['Open','High','Low','Close','Adj Close','Volume']:
        for c in candidates:
            if c.lower().endswith(std.lower()) or std.lower() in c.lower().split('_') or c.lower() == std.lower():
                mapping[std] = c
                break
    # if Adj Close present and Close not, map Adj Close -> Close
    if 'Close' not in mapping and 'Adj Close' in mapping:
        mapping['Close'] = mapping['Adj Close']
    # Build final df with OHLCV
    final = pd.DataFrame(index=df.index)
    for std in ['Open','High','Low','Close','Volume']:
        if std in mapping:
            final[std] = df[mapping[std]]
        else:
            # try common lowercase keys
            matches = [c for c in candidates if std.lower() in c.lower()]
            final[std] = df[matches[0]] if matches else np.nan
            mapping[std] = matches[0] if matches else None
    return final, mapping


def to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Convert index timestamps to IST (Asia/Kolkata) and return timezone-naive timestamps in IST.
    Handles tz-aware and tz-naive indices gracefully.
    """
    if df is None or df.empty:
        return df
    idx = pd.to_datetime(df.index)
    # if tz-aware, convert directly
    if idx.tz is not None:
        try:
            idx = idx.tz_convert('Asia/Kolkata')
            idx = idx.tz_localize(None)
            df = df.copy()
            df.index = idx
            return df
        except Exception:
            # fallback below
            pass
    # if tz-naive, assume UTC and convert to IST
    try:
        idx = idx.tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)
        df = df.copy()
        df.index = idx
    except Exception:
        # if localization fails, leave as is
        df.index = idx
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Close'] = df['Close'].astype(float)
    df['returns'] = df['Close'].pct_change()
    df['logret'] = np.log(df['Close'] / df['Close'].shift(1)).replace([np.inf, -np.inf], np.nan)
    df['vol_20'] = df['logret'].rolling(20).std() * np.sqrt(252)
    df['vol_10'] = df['logret'].rolling(10).std() * np.sqrt(252)
    df['rsi_14'] = compute_rsi(df['Close'], 14)
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    return df


def make_pattern_vector(subdf: pd.DataFrame, window: int):
    """Create a fixed-size feature vector for a window of candles.
    To avoid mismatched concatenation lengths, we ensure every component has same length = window.
    Features used:
      - close log returns for each candle (length = window) -- first entry is 0 (no prior)
      - normalized (High-Low)/Close for each candle (length = window)
    Returns 1D numpy array of length 2*window or None if insufficient data.
    """
    if len(subdf) < window:
        return None
    closes = subdf['Close'].values[-window:]
    # compute log returns with same length: first entry 0 then diffs
    logprice = np.log(closes)
    lr = np.concatenate([[0.0], np.diff(logprice)])  # length window
    # range ratio
    rng = ((subdf['High'].values[-window:] - subdf['Low'].values[-window:]) / (closes + 1e-12))
    # stack
    vec = np.concatenate([lr, rng])  # length 2*window
    # normalize
    vec = (vec - np.mean(vec)) / (np.std(vec) + 1e-9)
    return vec


def sliding_similarity_search(df: pd.DataFrame, window: int = 10, top_k: int = 10, distance_threshold_pct: float = 5.0):
    """Search historical series for windows similar to the most recent window.
    Returns DataFrame of matches with forward returns and volatility.
    """
    X = df[['Open','High','Low','Close']].dropna()
    N = len(X)
    if N < window + 1:
        return pd.DataFrame()
    recent_vec = make_pattern_vector(X.iloc[-window:], window)
    if recent_vec is None:
        return pd.DataFrame()
    distances = []
    for i in range(window, N - 1 - 1):
        hist_win = X.iloc[i - window:i]
        hist_vec = make_pattern_vector(hist_win, window)
        if hist_vec is None:
            continue
        d = np.linalg.norm(recent_vec - hist_vec)
        distances.append((i, d))
    if not distances:
        return pd.DataFrame()
    distances = sorted(distances, key=lambda x: x[1])
    # threshold based on percentile of all distances
    dvals = [d for (_, d) in distances]
    thresh = np.percentile(dvals, distance_threshold_pct)
    top = [t for t in distances if t[1] <= thresh]
    # if none below threshold, take top_k by distance
    if not top:
        top = distances[:top_k]
    else:
        top = top[:top_k]
    # build matches
    matches = []
    for idx, dist in top:
        # forward returns horizons
        horizons = [1,5,10]
        fr = {}
        for h in horizons:
            if idx + h < N:
                fr[f'ret_{h}'] = df['Close'].iloc[idx + h] / df['Close'].iloc[idx] - 1
            else:
                fr[f'ret_{h}'] = np.nan
        vol = df['logret'].iloc[max(0, idx - window):idx].std() * np.sqrt(252)
        matches.append({
            'idx': int(idx),
            'date': df.index[idx],
            'distance': float(dist),
            **fr,
            'vol': float(vol)
        })
    matches_df = pd.DataFrame(matches)
    return matches_df


def summarize_matches(matches_df: pd.DataFrame):
    if matches_df is None or matches_df.empty:
        return {}
    out = {}
    for h in [1,5,10]:
        col = f'ret_{h}'
        out[f'avg_ret_{h}'] = float(matches_df[col].mean()) if col in matches_df else np.nan
        out[f'median_ret_{h}'] = float(matches_df[col].median()) if col in matches_df else np.nan
    out['avg_vol'] = float(matches_df['vol'].mean()) if 'vol' in matches_df else np.nan
    out['n_matches'] = int(len(matches_df))
    return out


def recommend_from_summary(summary: dict, upside_threshold=0.01, downside_threshold=-0.01):
    if not summary or summary.get('n_matches',0) < 2:
        return 'No strong recommendation (insufficient historical matches)'
    exp1 = summary.get('avg_ret_1', 0)
    exp5 = summary.get('avg_ret_5', 0)
    if exp1 > upside_threshold and exp5 > upside_threshold:
        return f'BUY (historic avg next-1 & next-5 returns: {exp1:.2%}, {exp5:.2%})'
    if exp1 < downside_threshold and exp5 < downside_threshold:
        return f'SELL (historic avg next-1 & next-5 returns: {exp1:.2%}, {exp5:.2%})'
    return f'HOLD (mixed signals — next-1 {exp1:.2%}, next-5 {exp5:.2%})'

# -------------------- Streamlit UI --------------------
st.title("Algo Trader Assistant — Corrected & Robust")

with st.sidebar:
    st.header('Data Selection & Controls')
    ticker = st.text_input('Ticker (yfinance)', value='^NSEI')
    ticker2 = st.text_input('Comparison Ticker (optional)', value='')
    period = st.selectbox('Period', options=['1d','5d','7d','1mo','3mo','6mo','1y','2y','3y','5y','6y','10y','15y','20y','25y','30y'], index=6)
    interval = st.selectbox('Interval', options=['1m','3m','5m','10m','15m','30m','60m','120m','240m','1d'], index=9)
    window = st.number_input('Pattern window (candles)', value=10, min_value=2, max_value=200)
    distance_pct = st.slider('Distance percentile threshold', min_value=1, max_value=100, value=5)
    top_k = st.number_input('Top matches to analyze', value=10, min_value=1, max_value=200)
    future_h = st.selectbox('Future horizon to evaluate (candles)', options=[1,5,10], index=1)
    st.markdown('---')
    st.write('Controls')
    fetch = st.button('Fetch & Analyze Data')
    st.write('After clicking fetch, data is cached and UI components remain visible.')

# session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df_raw_map' not in st.session_state:
    st.session_state['df_raw_map'] = None
if 'df2' not in st.session_state:
    st.session_state['df2'] = None
if 'matches' not in st.session_state:
    st.session_state['matches'] = None

if fetch:
    with st.spinner('Fetching data — cached for 5 minutes to avoid rate-limits'):
        raw = fetch_data(ticker, period, interval)
        if raw.empty:
            st.error('No data returned. Check ticker or timeframe.')
        else:
            df_flat, mapping = flatten_and_map(raw)
            df_flat = to_ist(df_flat)
            df_flat = compute_indicators(df_flat)
            st.session_state['df'] = df_flat
            st.session_state['df_raw_map'] = mapping
            if ticker2:
                raw2 = fetch_data(ticker2, period, interval)
                if raw2.empty:
                    st.warning('Second ticker returned no data.')
                    st.session_state['df2'] = None
                else:
                    df2_flat, mapping2 = flatten_and_map(raw2)
                    df2_flat = to_ist(df2_flat)
                    df2_flat = compute_indicators(df2_flat)
                    st.session_state['df2'] = df2_flat
                    st.session_state['df2_map'] = mapping2
            # similarity search
            matches_df = sliding_similarity_search(st.session_state['df'], window=window, top_k=top_k, distance_threshold_pct=distance_pct)
            st.session_state['matches'] = matches_df

# layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader(f'{ticker} — Price & Indicators')
    if st.session_state['df'] is None:
        st.info('No data loaded. Click "Fetch & Analyze Data".')
    else:
        df = st.session_state['df']
        # Candlestick
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
        # moving averages
        if 'ma_20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ma_20'], name='MA20'))
        if 'ma_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['ma_50'], name='MA50'))
        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # RSI
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], name='RSI(14)'))
        rsi_fig.update_layout(height=200)
        st.plotly_chart(rsi_fig, use_container_width=True)

        # Volume
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
        vol_fig.update_layout(height=150)
        st.plotly_chart(vol_fig, use_container_width=True)

        # Display mapped columns
        st.subheader('Mapped columns (original -> used)')
        st.json(st.session_state.get('df_raw_map', {}))

        # summary table of points and %
        summary_df = df.copy()
        summary_df['pts_change'] = summary_df['Close'] - summary_df['Close'].shift(1)
        summary_df['pct_change'] = summary_df['Close'].pct_change()
        st.subheader('Recent OHLC with points & % change')
        st.dataframe(summary_df[['Open','High','Low','Close','Volume','pts_change','pct_change']].tail(200))

        # Heatmap month vs year
        if len(df) > 30:
            daily = df['Close'].resample('D').last().dropna()
            monthly = daily.resample('M').last().pct_change()
            monthly_df = monthly.to_frame('monthly_ret')
            monthly_df['month'] = monthly_df.index.month
            monthly_df['year'] = monthly_df.index.year
            pivot = monthly_df.pivot_table(index='month', columns='year', values='monthly_ret')
            heat_fig = px.imshow(pivot.T, labels=dict(x='Month', y='Year', color='Monthly Return'))
            st.subheader('Heatmap — Month vs Year returns')
            st.plotly_chart(heat_fig, use_container_width=True)

with col2:
    st.subheader('Pattern Similarity & Forecast')
    if st.session_state['matches'] is None:
        st.info('No similarity analysis yet. Fetch data first.')
    else:
        matches = st.session_state['matches']
        if matches.empty:
            st.write('No similar historical patterns found with the current sensitivity.')
        else:
            st.write('Top historical matches (closest distances)')
            st.dataframe(matches)
            summary = summarize_matches(matches)
            st.write('Summary statistics of matches:')
            st.json(summary)
            rec = recommend_from_summary(summary)
            st.markdown(f"### Recommendation: {rec}")

    st.markdown('---')
    st.subheader('Pair Ratio Analysis (Ticker1 / Ticker2)')
    if ticker2 and st.session_state.get('df2') is not None:
        df1 = st.session_state['df']
        df2 = st.session_state['df2']
        merged = pd.merge(df1[['Close']], df2[['Close']], left_index=True, right_index=True, how='inner', suffixes=('_1','_2'))
        if merged.empty:
            st.warning('No overlapping timestamps between ticker1 and ticker2 to compute ratio.')
        else:
            merged['ratio'] = merged['Close_1'] / (merged['Close_2'].replace(0, np.nan))
            merged = merged.dropna()
            if len(merged) > 10:
                merged['ratio_bin'] = pd.qcut(merged['ratio'], q=5, duplicates='drop')
                bin_summary = merged.groupby('ratio_bin')['Close_1'].pct_change().mean().dropna()
                st.write('Ratio bin summary (avg future returns):')
                st.dataframe(bin_summary.to_frame('avg_return'))
            st.line_chart(merged[['Close_1','Close_2','ratio']])
    else:
        st.info('Provide a second ticker and fetch data to analyze ratio.')

# Export
st.markdown('---')
st.subheader('Export & Download')
if st.session_state['df'] is not None:
    to_download = st.session_state['df'].copy()
    csv = to_download.to_csv().encode()
    st.download_button('Download OHLCV CSV', data=csv, file_name=f'{ticker}_ohlcv.csv', mime='text/csv')
    towrite = BytesIO()
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        to_download.to_excel(writer, sheet_name='data')
        if st.session_state.get('matches') is not None and not st.session_state['matches'].empty:
            st.session_state['matches'].to_excel(writer, sheet_name='matches')
    towrite.seek(0)
    st.download_button('Download Excel workbook', data=towrite, file_name=f'{ticker}_analysis.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Explanation & final summary
st.markdown('---')
st.header('Automated Explanation & Key Insights')
if st.session_state['df'] is None:
    st.write('No data yet. Click Fetch & Analyze Data.')
else:
    df = st.session_state['df']
    recent = df.tail(1)['Close'].iloc[0]
    highest = df['Close'].max()
    lowest = df['Close'].min()
    avg_vol = df['vol_20'].mean()
    st.write(f'Latest Close: {recent:.2f} | Highest (period): {highest:.2f} | Lowest (period): {lowest:.2f}')
    st.write(f'Average annualized volatility (20): {avg_vol:.2%}')
    if st.session_state.get('matches') is not None and not st.session_state['matches'].empty:
        summary = summarize_matches(st.session_state['matches'])
        st.write('Historic pattern summary:')
        st.write(summary)
        st.write('Human readable explanation:')
        st.write(f"Found {summary.get('n_matches')} historical occurrences where the recent {window} candle pattern was similar. On average the next 1 candle returned {summary.get('avg_ret_1',0):.2%} and next 5 candles returned {summary.get('avg_ret_5',0):.2%}. Volatility during those occurrences was {summary.get('avg_vol'):.2%}. Recommendation: {recommend_from_summary(summary)}")
    else:
        st.write('No pattern matches found to explain automatically.')

st.markdown('---')
st.caption('Notes: Pattern matching is heuristic. For better early-warning detection of large rallies, consider adding DTW or model-based embeddings (autoencoders) which we can add next.')
