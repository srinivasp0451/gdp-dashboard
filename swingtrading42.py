"""
Professional Algo Trading Dashboard - Streamlit App
Single-file Streamlit application implementing:
- Multi-asset, multi-timeframe data fetching via yfinance
- Manual calculation of indicators (EMA, SMA, RSI, ATR, VWAP, Fibonacci)
- Ratio analysis, volatility bins, z-score analysis
- Pattern detection (basic set) and multi-timeframe summary
- Interactive Plotly charts stacked vertically
- CSV/Excel export, IST timezone handling
- Caching, progress bars, configurable API delay, button-based fetch

Notes:
- No talib or pandas_ta used. Indicators computed manually.
- This is a comprehensive, production-oriented starting point. You can extend pattern recognition rules and UI as needed.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import time
from scipy.stats import skew, kurtosis
import io
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------- Helper functions ---------------------

IST = pytz.timezone('Asia/Kolkata')

st.set_page_config(page_title="Professional Algo Trading Dashboard", layout="wide")

# ---------- Indicator implementations (manual) ----------

def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return np.nan


def sma(series, period):
    return series.rolling(window=period, min_periods=1).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def vwap(df):
    # vwap uses price * volume cumulatively / volume cumulatively
    pv = (df['Close'] * df['Volume']).cumsum()
    v = df['Volume'].cumsum()
    return pv / v


def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)


def fib_levels(high, low):
    diff = high - low
    return {
        '0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100%': low
    }

# ---------- Timeframe mapping and utilities ----------

TF_MAP = {
    '1m': {'interval': '1m', 'default_period': '7d'},
    '3m': {'interval': '3m', 'default_period': '7d'},
    '5m': {'interval': '5m', 'default_period': '1mo'},
    '10m': {'interval': '10m', 'default_period': '1mo'},
    '15m': {'interval': '15m', 'default_period': '1mo'},
    '30m': {'interval': '30m', 'default_period': '2mo'},
    '1h': {'interval': '60m', 'default_period': '3mo'},
    '2h': {'interval': '120m', 'default_period': '6mo'},
    '4h': {'interval': '240m', 'default_period': '1y'},
    '1d': {'interval': '1d', 'default_period': '5y'},
}

PERIOD_CHOICES = ['1d','5d','7d','1mo','3mo','6mo','1y','2y','3y','5y','6y','10y','15y','20y','25y','30y']

# ---------- Fetching function with caching ----------

@st.cache_data(show_spinner=False)
def fetch_yfinance(ticker, interval='1d', period='1mo'):
    """Fetch data from yfinance and normalize dataframe."""
    try:
        df = yf.download(tickers=ticker, interval=interval, period=period, progress=False, threads=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # yfinance sometimes returns MultiIndex columns for some tickers; keep only standard OHLCV
        # Ensure column names are normalized
        df.columns = [c if not isinstance(c, tuple) else c[1] for c in df.columns]
        expected = ['Datetime','Date','Open','High','Low','Close','Adj Close','Volume']
        # Prefer 'Datetime' or 'Date' column as datetime
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime':'DateTime'}, inplace=True)
        elif 'Date' in df.columns:
            df.rename(columns={'Date':'DateTime'}, inplace=True)
        else:
            # sometimes index was date
            df['DateTime'] = pd.to_datetime(df.iloc[:,0])
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        # convert to IST
        if df['DateTime'].dt.tz is None:
            df['DateTime'] = df['DateTime'].dt.tz_localize('UTC').dt.tz_convert(IST)
        else:
            df['DateTime'] = df['DateTime'].dt.tz_convert(IST)
        # Keep only OHLCV
        keep_cols = [c for c in ['DateTime','Open','High','Low','Close','Volume'] if c in df.columns]
        df = df[keep_cols]
        df = df.sort_values('DateTime').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# ---------- Utility for delay and progress ----------

def fetch_with_delay(ticker, interval, period, delay_seconds):
    time.sleep(delay_seconds)
    return fetch_yfinance(ticker, interval=interval, period=period)

# ---------- Ratio and binning helpers ----------

def create_ratio(df1, df2):
    df = pd.DataFrame()
    df['DateTime'] = df1['DateTime']
    df['Price1'] = df1['Close'].values
    df['Price2'] = df2['Close'].values
    df['Ratio'] = df['Price1'] / df['Price2']
    return df


def create_bins(series, n_bins=5):
    bins = pd.qcut(series, q=n_bins, duplicates='drop')
    bin_ranges = bins.unique().categories if hasattr(bins, 'unique') else pd.Series(bins).unique()
    return bins

# ---------- Pattern recognition (basic rules) ----------

def detect_patterns(df, move_threshold_points=30):
    results = []
    close = df['Close']
    rsi_series = rsi(close)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    for i in range(1, len(df)):
        move = (close.iloc[i] - close.iloc[i-1])
        move_pct = move / close.iloc[i-1] * 100
        entry = {
            'DateTime': df['DateTime'].iloc[i],
            'MovePoints': move,
            'MovePct': move_pct,
            'Direction': 'Up' if move>0 else 'Down',
            'LargeBody': abs((df['Close'].iloc[i] - df['Open'].iloc[i])) > (df['High'].iloc[i] - df['Low'].iloc[i]) * 0.6 if all(col in df.columns for col in ['Open','High','Low']) else False,
            'VolumeSpike': False,
            'RSI_before': np.nan,
            'RSI_now': rsi_series.iloc[i] if i < len(rsi_series) else np.nan,
            'EMA20_cross_EMA50': (ema20.iloc[i] > ema50.iloc[i]) and (ema20.iloc[i-1] <= ema50.iloc[i-1]) if i>0 else False
        }
        # detect volume spike if column exists
        if 'Volume' in df.columns and i>1:
            vol = df['Volume']
            if vol.iloc[i] > vol.iloc[max(0,i-10):i].mean() * 2.0:
                entry['VolumeSpike'] = True
        # RSI before
        if i-1 >= 0:
            entry['RSI_before'] = rsi_series.iloc[i-1] if i-1 < len(rsi_series) else np.nan
        # mark big move
        entry['IsSignificant'] = abs(move) >= move_threshold_points
        results.append(entry)
    return pd.DataFrame(results)

# ---------- Statistical summaries ----------

def stats_summary(series):
    return {
        'mean': series.mean(),
        'std': series.std(ddof=0),
        'skew': skew(series.dropna()),
        'kurtosis': kurtosis(series.dropna())
    }

# ---------- Exports ----------

def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    return output.getvalue()

# --------------------- Streamlit UI ---------------------

st.title('Professional Algo Trading Dashboard')

# Sidebar configuration
with st.sidebar:
    st.header('Data & Fetch Settings')
    ticker1 = st.text_input('Ticker 1 (yfinance)', value='^NSEI')
    enable_ratio = st.checkbox('Enable Ratio Analysis (Ticker 2)?')
    ticker2 = st.text_input('Ticker 2 (yfinance)', value='^NSEBANK' if enable_ratio else '') if enable_ratio else ''
    interval = st.selectbox('Interval', options=list(TF_MAP.keys()), index=2)
    period = st.selectbox('Period', options=PERIOD_CHOICES, index=3)
    api_delay = st.slider('API delay between calls (seconds)', min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    move_threshold = st.number_input('Significant move threshold (points)', value=30)
    st.markdown('---')
    st.markdown('Export / Display options')
    show_raw = st.checkbox('Show raw OHLCV data', value=False)
    st.markdown('---')

# session state for results
if 'results' not in st.session_state:
    st.session_state['results'] = {}

# Fetch button triggers heavy processing
if st.button('Fetch Data & Analyze'):
    st.info('Fetching data — please do not click again. Progress bars will show status.')
    progress = st.progress(0)
    try:
        # Fetch ticker1
        progress.progress(5)
        df1 = fetch_with_delay(ticker1, TF_MAP[interval]['interval'], period, api_delay)
        if df1.empty:
            st.error(f'No data for {ticker1}. Try different ticker or timeframe.')
            st.stop()
        progress.progress(20)

        df2 = None
        if enable_ratio and ticker2:
            progress.progress(25)
            df2 = fetch_with_delay(ticker2, TF_MAP[interval]['interval'], period, api_delay)
            if df2.empty:
                st.warning(f'No data for {ticker2}. Disabling ratio analysis.')
                enable_ratio = False
            progress.progress(40)

        # calculate indicators for df1
        progress.progress(45)
        df1['SMA20'] = sma(df1['Close'], 20)
        df1['SMA50'] = sma(df1['Close'], 50)
        df1['EMA9'] = ema(df1['Close'], 9)
        df1['EMA20'] = ema(df1['Close'], 20)
        df1['EMA50'] = ema(df1['Close'], 50)
        df1['EMA200'] = ema(df1['Close'], 200)
        df1['RSI14'] = rsi(df1['Close'], 14)
        df1['ATR14'] = atr(df1, 14)
        try:
            df1['VWAP'] = vwap(df1)
        except Exception:
            df1['VWAP'] = np.nan
        progress.progress(60)

        if enable_ratio and df2 is not None:
            df2['SMA20'] = sma(df2['Close'], 20)
            df2['SMA50'] = sma(df2['Close'], 50)
            df2['EMA9'] = ema(df2['Close'], 9)
            df2['EMA20'] = ema(df2['Close'], 20)
            df2['EMA50'] = ema(df2['Close'], 50)
            df2['EMA200'] = ema(df2['Close'], 200)
            df2['RSI14'] = rsi(df2['Close'], 14)
            df2['ATR14'] = atr(df2, 14)
            try:
                df2['VWAP'] = vwap(df2)
            except Exception:
                df2['VWAP'] = np.nan
            progress.progress(70)

        # Ratio dataframe
        ratio_df = None
        if enable_ratio and df2 is not None:
            # align lengths — resample or inner join on DateTime
            merged = pd.merge_asof(df1[['DateTime','Close']].rename(columns={'Close':'Close1'}),
                                   df2[['DateTime','Close']].rename(columns={'Close':'Close2'}),
                                   on='DateTime', direction='nearest', tolerance=pd.Timedelta("1h"))
            merged.dropna(inplace=True)
            ratio_df = pd.DataFrame()
            ratio_df['DateTime'] = merged['DateTime']
            ratio_df['Price1'] = merged['Close1']
            ratio_df['Price2'] = merged['Close2']
            ratio_df['Ratio'] = ratio_df['Price1'] / ratio_df['Price2']
            ratio_df['RatioReturns'] = ratio_df['Ratio'].pct_change()
            ratio_df['RSI_Ratio'] = rsi(ratio_df['Ratio'].fillna(method='ffill'), 14)
            progress.progress(78)

        # Multi-timeframe summary (quick approximation using current df1/df2)
        mt_cols = ['Timeframe','Trend','MaxClose','MinClose','Fib_Levels','Volatility','PctChange','PointsChange','RSI','RSI_status']
        mt_rows = []
        # We'll compute for a pre-defined set based on user request
        multi_tf_requested = [
            ('1m','7d'),('5m','1mo'),('15m','1mo'),('30m','2mo'),('1h','3mo'),('2h','6mo'),('4h','1y'),('1d','5y')
        ]
        ix = 80
        for tf, pr in multi_tf_requested:
            progress.progress(ix)
            raw = fetch_yfinance(ticker1, interval=TF_MAP.get(tf, {'interval':tf})['interval'], period=pr)
            if raw.empty:
                continue
            maxc = raw['Close'].max()
            minc = raw['Close'].min()
            fib = fib_levels(maxc, minc)
            vol = raw['Close'].pct_change().std() * 100
            pctchg = (raw['Close'].iloc[-1] - raw['Close'].iloc[0]) / raw['Close'].iloc[0] * 100
            points = raw['Close'].iloc[-1] - raw['Close'].iloc[0]
            rsi_val = rsi(raw['Close']).iloc[-1]
            rsi_status = 'Neutral'
            if rsi_val <= 30:
                rsi_status = 'Oversold'
            elif rsi_val >= 70:
                rsi_status = 'Overbought'
            trend = 'Up' if pctchg>0 else 'Down' if pctchg<0 else 'Neutral'
            mt_rows.append({
                'Timeframe': tf,
                'Trend': trend,
                'MaxClose': maxc,
                'MinClose': minc,
                'Fib_Levels': fib,
                'Volatility': vol,
                'PctChange': pctchg,
                'PointsChange': points,
                'RSI': rsi_val,
                'RSI_status': rsi_status
            })
            ix += 2
        progress.progress(92)

        # Volatility bins
        returns = df1['Close'].pct_change().fillna(0)
        vol_series = returns.rolling(window=20, min_periods=1).std() * 100
        vol_df = pd.DataFrame({'DateTime':df1['DateTime'], 'VolPct': vol_series, 'Price': df1['Close']})
        vol_df['Bin'] = pd.qcut(vol_df['VolPct'].fillna(0)+1e-9, 5, labels=False, duplicates='drop')
        progress.progress(95)

        # Pattern detection
        patterns = detect_patterns(df1, move_threshold_points=move_threshold)
        progress.progress(98)

        # Statistical distribution & z-score
        returns_points = df1['Close'].diff().fillna(0)
        stats = stats_summary(returns_points)
        zscores = zscore(returns_points.fillna(0))
        z_df = pd.DataFrame({'DateTime':df1['DateTime'], 'ReturnsPoints': returns_points, 'ZScore': zscores})

        # Save in session
        st.session_state['results'] = {
            'df1': df1,
            'df2': df2,
            'ratio_df': ratio_df,
            'mt_rows': mt_rows,
            'vol_df': vol_df,
            'patterns': patterns,
            'z_df': z_df,
            'stats': stats
        }
        progress.progress(100)
        st.success('Analysis complete! Scroll down to view results.')
    except Exception as e:
        st.error(f'Unexpected error during analysis: {e}')

# --------------------- Display Section ---------------------

res = st.session_state.get('results', None)
if res:
    df1 = res['df1']
    df2 = res.get('df2')
    ratio_df = res.get('ratio_df')

    st.header('Market Overview')
    c1, c2, c3 = st.columns([2,2,6])
    with c1:
        last_price = df1['Close'].iloc[-1]
        prev = df1['Close'].iloc[-2] if len(df1)>1 else last_price
        pct = (last_price-prev)/prev*100 if prev!=0 else 0
        st.metric(label=f'{ticker1} Last Price', value=round(last_price,2), delta=f"{round(pct,2)}%")
    with c2:
        if df2 is not None:
            last_price2 = df2['Close'].iloc[-1]
            prev2 = df2['Close'].iloc[-2] if len(df2)>1 else last_price2
            pct2 = (last_price2-prev2)/prev2*100 if prev2!=0 else 0
            st.metric(label=f'{ticker2} Last Price', value=round(last_price2,2), delta=f"{round(pct2,2)}%")
    with c3:
        st.write('Key Stats')
        st.write(res['stats'])

    if show_raw:
        st.subheader('Raw OHLCV - Ticker 1')
        st.dataframe(df1)
        if df2 is not None:
            st.subheader('Raw OHLCV - Ticker 2')
            st.dataframe(df2)

    # Ratio analysis
    if ratio_df is not None:
        st.header('Ratio Analysis')
        st.dataframe(ratio_df.head(200))
        # Binning explicit ranges
        n_bins = 5
        ratio_df['Bin'] = pd.qcut(ratio_df['Ratio'], q=n_bins, duplicates='drop')
        bin_summary = ratio_df.groupby('Bin').agg({'Price1':['mean','last'], 'Price2':['mean','last'], 'Ratio':['mean','count']})
        st.subheader('Ratio Bins Summary')
        st.dataframe(bin_summary)
        csv = ratio_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Ratio CSV', data=csv, file_name=f'ratio_{ticker1}_{ticker2}.csv')

        # Charts for ratio (stacked vertically)
        st.subheader('Ratio Charts')
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03)
        # Price1
        fig.add_trace(go.Candlestick(x=df1['DateTime'], open=df1['Open'], high=df1['High'], low=df1['Low'], close=df1['Close'], name=f'{ticker1}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df1['DateTime'], y=df1['EMA20'], name='EMA20'), row=1, col=1)
        # Price2
        fig.add_trace(go.Candlestick(x=df2['DateTime'], open=df2['Open'], high=df2['High'], low=df2['Low'], close=df2['Close'], name=f'{ticker2}'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df2['DateTime'], y=df2['EMA20'], name='EMA20'), row=2, col=1)
        # Ratio
        fig.add_trace(go.Scatter(x=ratio_df['DateTime'], y=ratio_df['Ratio'], name='Ratio'), row=3, col=1)
        fig.update_layout(height=900)
        st.plotly_chart(fig, use_container_width=True)

    # Single ticker charts
    st.header('Price & Indicators - Ticker 1')
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
    fig2.add_trace(go.Candlestick(x=df1['DateTime'], open=df1['Open'], high=df1['High'], low=df1['Low'], close=df1['Close'], name=f'{ticker1}'), row=1, col=1)
    # EMA overlays
    fig2.add_trace(go.Scatter(x=df1['DateTime'], y=df1['EMA20'], name='EMA20', opacity=0.8), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df1['DateTime'], y=df1['EMA50'], name='EMA50', opacity=0.8), row=1, col=1)
    # RSI
    fig2.add_trace(go.Scatter(x=df1['DateTime'], y=df1['RSI14'], name='RSI14'), row=2, col=1)
    fig2.update_layout(height=700)
    st.plotly_chart(fig2, use_container_width=True)

    # Multi-timeframe table
    st.header('Multi-timeframe Analysis (Ticker 1)')
    mt_df = pd.DataFrame(res['mt_rows'])
    st.dataframe(mt_df)

    # Volatility bins
    st.header('Volatility Bins')
    st.dataframe(res['vol_df'].head(200))

    # Pattern detection
    st.header('Pattern Detection Summary')
    st.write('Total significant patterns found:', len(res['patterns'][res['patterns']['IsSignificant']==True]))
    st.dataframe(res['patterns'].head(200))

    # Statistical distribution and Z-score
    st.header('Statistical Distribution & Z-Score')
    col1, col2 = st.columns(2)
    with col1:
        st.write('Return stats:')
        st.json(res['stats'])
    with col2:
        st.write('Z-score head:')
        st.dataframe(res['z_df'].head(200))

    # Final trading recommendation (basic multi-factor aggregator)
    st.header('Final Trading Recommendation')
    # multi-factor scoring based on latest values
    try:
        latest = df1.iloc[-1]
        tf_score = 1 if mt_df.iloc[0]['Trend']=='Up' else -1
        rsi_val = latest['RSI14']
        rsi_score = 1 if rsi_val < 40 else -1 if rsi_val>60 else 0
        z = res['z_df']['ZScore'].iloc[-1]
        z_score = 1 if abs(z)>1 else 0
        ema_align = 1 if (latest['EMA20']>latest['EMA50']>latest['EMA200']) else -1
        combined = 0.3*tf_score + 0.2*rsi_score + 0.2*z_score + 0.3*ema_align
        strength = 'Neutral'
        if combined>0.3:
            strength='Buy'
        elif combined<-0.3:
            strength='Sell'
        else:
            strength='Hold'

        st.markdown(f"**Signal:** {strength}  **Combined Score:** {round(combined,3)}")
        # Risk management
        atr_val = latest['ATR14'] if 'ATR14' in latest else df1['ATR14'].iloc[-1]
        entry = latest['Close']
        target = entry + 2*atr_val if strength=='Buy' else entry - 2*atr_val if strength=='Sell' else np.nan
        stop = entry - 1*atr_val if strength=='Buy' else entry + 1*atr_val if strength=='Sell' else np.nan
        rr = abs((target-entry)/ (entry-stop)) if (not np.isnan(target) and not np.isnan(stop) and (entry-stop)!=0) else np.nan
        st.write({'Entry':entry, 'Target':target, 'Stop':stop, 'RR':rr})
    except Exception as e:
        st.write('Could not compute final recommendation:', e)

    # Exports
    st.subheader('Export Data')
    if st.button('Export Ticker1 to Excel'):
        towrite = to_excel_bytes(df1)
        st.download_button('Download Excel', data=towrite, file_name=f'{ticker1}_data.xlsx')
    if df2 is not None and st.button('Export Ticker2 to Excel'):
        towrite = to_excel_bytes(df2)
        st.download_button('Download Excel', data=towrite, file_name=f'{ticker2}_data.xlsx')

    st.markdown('---')
    st.caption('End of analysis. Extend pattern rules and tweak weights for your strategy.')
else:
    st.info('Click "Fetch Data & Analyze" to run analysis. Results persist in session state.')

# --------------------- End of App ---------------------

# Developer notes (not displayed):
# - Extend pattern detection for divergence, liquidity sweeps, smart-money footprints.
# - Add hypothesis testing and p-value calculations where required.
# - Add more robust multi-timeframe aggregation using resampling and proper alignment.
# - Consider streaming websocket for live LTP but keep fetch button for API rate safety.
