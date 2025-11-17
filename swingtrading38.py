import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from io import BytesIO

st.set_page_config(layout="wide", page_title="Algo Trader Assistant")

# -------------------- Helpers --------------------
@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetches data from yfinance with caching to prevent rate-limit overfetching."""
    # yfinance sometimes returns multiindex columns for period/interval combos. We'll normalize.
    raw = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=True)
    if raw.empty:
        return raw
    # Reset index and ensure timezone conversion later
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    # if tz-aware, convert to UTC then to IST
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


def to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Convert naive or utc-indexed df to Asia/Kolkata timezone-aware index and return localized index as IST naive timestamps."""
    # treat index as UTC if hour aligns with UTC trading times; safest: make UTC then convert
    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize('UTC')
    except Exception:
        # already tz-aware or ambiguous, try localize only if naive
        if idx.tz is None:
            pass
    # convert to IST
    try:
        idx = idx.tz_convert('Asia/Kolkata')
        # make index timezone-naive but in IST for downstream calculations (so exports show local time)
        idx = idx.tz_localize(None)
        df = df.copy()
        df.index = idx
    except Exception:
        # fallback: do nothing
        df.index = idx
    return df


def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select open, high, low, close, volume and ensure column names are standardized."""
    cols = [c for c in df.columns]
    # common names
    for c in ['Open','High','Low','Close','Adj Close','Volume']:
        if c in cols:
            pass
    # Keep only OHLCV if present
    need = []
    for cand in ['Open','High','Low','Close','Volume']:
        if cand in df.columns:
            need.append(cand)
    if not need:
        # maybe multiindex columns -> flatten
        df.columns = ["_".join([str(i) for i in col if i]) if isinstance(col, (list,tuple)) else col for col in df.columns]
        for cand in ['Open','High','Low','Close','Volume']:
            if any(cand in x for x in df.columns):
                need = [x for x in df.columns if cand in x]
                break
    df = df[[c for c in df.columns if any(k in c for k in ['Open','High','Low','Close','Volume'])]]
    # rename if necessary
    rename = {}
    for c in df.columns:
        if 'Open' in c and c != 'Open':
            rename[c] = 'Open'
        if 'High' in c and c != 'High':
            rename[c] = 'High'
        if 'Low' in c and c != 'Low':
            rename[c] = 'Low'
        if 'Close' in c and c != 'Close':
            rename[c] = 'Close'
        if 'Volume' in c and c != 'Volume':
            rename[c] = 'Volume'
    df = df.rename(columns=rename)
    # if any missing columns, try fill
    for c in ['Open','High','Low','Close','Volume']:
        if c not in df.columns:
            df[c] = np.nan
    df = df[['Open','High','Low','Close','Volume']]
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['logret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol_20'] = df['logret'].rolling(20).std() * np.sqrt(252)
    df['vol_10'] = df['logret'].rolling(10).std() * np.sqrt(252)
    df['rsi_14'] = compute_rsi(df['Close'], 14)
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    return df


def sliding_similarity_search(df: pd.DataFrame, window: int = 10, candidates: list = None, top_k: int = 10, distance_threshold_pct: float = 5.0):
    """Search entire historical series for patterns similar to the most recent 'window' candles.
    Returns list of matches with their future returns and volatility context.
    Approach: compute normalized vector of OHLC returns for the recent window; for each historical position compute same vector and euclidean distance. Keep matches where distance below percentile threshold or top_k smallest distances.
    """
    X = df[['Open','High','Low','Close']].fillna(method='ffill').dropna()
    # feature vector: normalized close returns for last window
    def make_vector(subdf):
        # use log returns of close and normalized highs/lows range
        v = []
        close = subdf['Close'].values
        lr = np.diff(np.log(close))
        # pad if necessary
        if len(lr) < window-1:
            return None
        v = lr[-(window-1):]
        # include volatility and range
        rng = (subdf['High'] - subdf['Low']) / subdf['Close']
        rngv = rng.values[-window:]
        return np.concatenate([v, rngv])

    recent = make_vector(X.iloc[-window:])
    if recent is None:
        return []
    # normalize recent
    rec_norm = (recent - np.mean(recent)) / (np.std(recent) + 1e-9)
    distances = []
    for i in range(window, len(X) - 1 - 1):
        hist_vec = make_vector(X.iloc[i-window:i])
        if hist_vec is None:
            continue
        hist_norm = (hist_vec - np.mean(hist_vec)) / (np.std(hist_vec) + 1e-9)
        d = np.linalg.norm(rec_norm - hist_norm)
        distances.append((i, d))
    if not distances:
        return []
    distances = sorted(distances, key=lambda x: x[1])
    # pick top_k
    top = distances[:top_k]
    # filter by threshold percentile
    dvals = [d for (_, d) in distances]
    thresh = np.percentile(dvals, distance_threshold_pct)
    filtered = [t for t in top if t[1] <= thresh or True]

    matches = []
    for idx, dist in filtered:
        # look forward returns for horizons 1,5,10 periods
        future_horizons = [1,5,10]
        fr = {}
        for h in future_horizons:
            if idx + h < len(df):
                fr[f'ret_{h}'] = df['Close'].iloc[idx + h] / df['Close'].iloc[idx] - 1
            else:
                fr[f'ret_{h}'] = np.nan
        # record volatility at that time
        vol = df['logret'].iloc[idx-window:idx].std() * np.sqrt(252)
        matches.append({
            'idx': idx,
            'date': df.index[idx],
            'distance': dist,
            **fr,
            'vol': vol
        })
    matches_df = pd.DataFrame(matches)
    return matches_df


def summarize_matches(matches_df: pd.DataFrame):
    if matches_df.empty:
        return "No historical matches found."
    out = {}
    for h in [1,5,10]:
        col = f'ret_{h}'
        out[f'avg_ret_{h}'] = matches_df[col].mean()
        out[f'median_ret_{h}'] = matches_df[col].median()
    out['avg_vol'] = matches_df['vol'].mean()
    out['n_matches'] = len(matches_df)
    return out


def recommend_from_summary(summary: dict, upside_threshold=0.01, downside_threshold=-0.01):
    """Simple recommendation heuristics based on average next-period return."""
    if not summary or summary.get('n_matches',0) < 2:
        return 'No strong recommendation (insufficient historical matches)'
    exp1 = summary.get('avg_ret_1', 0)
    exp5 = summary.get('avg_ret_5', 0)
    if exp1 > upside_threshold and exp5 > upside_threshold:
        return f'BUY (historical average next-1 & next-5 returns positive: {exp1:.2%}, {exp5:.2%})'
    if exp1 < downside_threshold and exp5 < downside_threshold:
        return f'SELL (historical average next-1 & next-5 returns negative: {exp1:.2%}, {exp5:.2%})'
    return f'HOLD (mixed signals — next-1 {exp1:.2%}, next-5 {exp5:.2%})'

# -------------------- Streamlit UI --------------------
st.title("Algo Trader Assistant — Streamlit Edition")

with st.sidebar:
    st.header('Data Selection')
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
    st.write('After clicking fetch, the UI will not disappear. Use the download buttons below to export data.')

# Persist fetched data in session state to avoid re-fetch on rerun
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df2' not in st.session_state:
    st.session_state['df2'] = None
if 'matches' not in st.session_state:
    st.session_state['matches'] = None

if fetch:
    with st.spinner('Fetching data — this is cached for 5 minutes to avoid rate limits'):
        raw = fetch_data(ticker, period, interval)
        if raw.empty:
            st.error('No data returned for ticker. Check ticker name or timeframe.')
        else:
            df = flatten_df(raw)
            df = to_ist(df)
            df = compute_indicators(df)
            st.session_state['df'] = df
            if ticker2:
                raw2 = fetch_data(ticker2, period, interval)
                df2 = flatten_df(raw2)
                df2 = to_ist(df2)
                df2 = compute_indicators(df2)
                st.session_state['df2'] = df2
            # run similarity search
            matches_df = sliding_similarity_search(df, window=window, top_k=top_k, distance_threshold_pct=distance_pct)
            st.session_state['matches'] = matches_df

# Display main panels
col1, col2 = st.columns([2,1])

with col1:
    st.subheader(f'{ticker} — Price & Indicators')
    if st.session_state['df'] is None:
        st.info('No data loaded. Click "Fetch & Analyze Data" in the sidebar.')
    else:
        df = st.session_state['df']
        # Candlestick with RSI and volume stacked
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # RSI plot
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], name='RSI(14)'))
        rsi_fig.update_layout(height=200)
        st.plotly_chart(rsi_fig, use_container_width=True)

        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
        vol_fig.update_layout(height=150)
        st.plotly_chart(vol_fig, use_container_width=True)

        # Display summary table of returns (points change) and percent change
        summary_df = df.copy()
        summary_df['pts_change'] = summary_df['Close'] - summary_df['Close'].shift(1)
        summary_df['pct_change'] = summary_df['Close'].pct_change()
        st.subheader('OHLC table with points & % change')
        st.dataframe(summary_df[['Open','High','Low','Close','Volume','pts_change','pct_change']].tail(200))

        # Heatmap: month vs year returns
        if len(df) > 30:
            temp = df['Close'].resample('D').last().dropna()
            monthly = temp.resample('M').last().pct_change()
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
        st.info('No similarity analysis available. Fetch data first.')
    else:
        matches = st.session_state['matches']
        if matches.empty:
            st.write('No similar historical patterns found.')
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
        # align on index
        merged = pd.merge(df1[['Close']], df2[['Close']], left_index=True, right_index=True, how='inner', suffixes=('_1','_2'))
        merged['ratio'] = merged['Close_1'] / merged['Close_2']
        # bins
        merged['ratio_bin'] = pd.qcut(merged['ratio'], q=5, duplicates='drop')
        bin_summary = merged.groupby('ratio_bin')['Close_1'].pct_change().mean().dropna()
        st.write('Ratio bin summary (avg future returns):')
        st.dataframe(bin_summary.to_frame('avg_return'))
        st.line_chart(merged[['Close_1','Close_2','ratio']])
    else:
        st.info('Provide a second ticker and fetch data to analyze ratio.')

# Export options
st.markdown('---')
st.subheader('Export & Download')
if st.session_state['df'] is not None:
    to_download = st.session_state['df'].copy()
    csv = to_download.to_csv().encode()
    st.download_button('Download OHLCV CSV', data=csv, file_name=f'{ticker}_ohlcv.csv', mime='text/csv')
    # excel
    towrite = BytesIO()
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        to_download.to_excel(writer, sheet_name='data')
        if st.session_state['matches'] is not None:
            st.session_state['matches'].to_excel(writer, sheet_name='matches')
    towrite.seek(0)
    st.download_button('Download Excel workbook', data=towrite, file_name=f'{ticker}_analysis.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Final recommendations & explanation box
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
        st.write(f"Found {summary.get('n_matches')} historical occurrences where the recent {window} candle pattern was similar. On average the next {future_h} candle(s) returned {summary.get('avg_ret_1',0):.2%} for 1-candle horizon and {summary.get('avg_ret_5',0):.2%} for 5-candle horizon. Volatility during those occurrences was {summary.get('avg_vol'):.2%}. Recommendation: {recommend_from_summary(summary)}")
    else:
        st.write('No pattern matches found to explain automatically.')

st.markdown('---')
st.caption('Notes: This tool is a helper — backtest decisions before trading. Pattern matching is heuristic; tune window, top_k and thresholds for better behavior.')
