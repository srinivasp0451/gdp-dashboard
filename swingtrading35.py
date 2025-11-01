import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import pytz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide", page_title="Algo Trader Streamlit", initial_sidebar_state="expanded")

# -------------------- Helper functions --------------------
TIMEZONE = pytz.timezone('Asia/Kolkata')

@st.cache_data(ttl=300)
def fetch_yfinance(ticker, period, interval):
    # Use yf.download to fetch
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=True)
    if df.empty:
        return df
    # flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]
    # ensure timezone-aware -> convert to IST and then make tz-naive for Excel export
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(TIMEZONE)
    else:
        df.index = df.index.tz_convert(TIMEZONE)
    df.index = df.index.tz_localize(None)  # make naive but in IST
    return df


def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=window-1, adjust=False).mean()
    ma_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_divergences(price, rsi, window=5):
    # Simple divergence detection: compare local highs/lows
    pivots = []
    for i in range(window, len(price)-window):
        is_high = price[i] > price[i-window:i].max() and price[i] > price[i+1:i+window+1].max()
        is_low = price[i] < price[i-window:i].min() and price[i] < price[i+1:i+window+1].min()
        if is_high or is_low:
            pivots.append((i, price.index[i], price.iloc[i], rsi.iloc[i], 'high' if is_high else 'low'))
    # pair price pivot direction with RSI opposite direction -> divergence
    divergences = []
    for i, idx, pval, rval, typ in pivots:
        # find previous pivot of same type
        prev = [x for x in pivots if x[0] < i and x[4]==typ]
        if prev:
            j, jdx, pval2, rval2, _ = prev[-1]
            if typ=='high' and pval>pval2 and rval<rval2:
                divergences.append((jdx, idx, 'bearish'))
            if typ=='low' and pval<pval2 and rval>rval2:
                divergences.append((jdx, idx, 'bullish'))
    return divergences


def fibonacci_levels(high, low):
    diff = high - low
    levels = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '1.0': low
    }
    return levels


def compute_returns(df, timeframe='1d'):
    # Accept timeframe as pandas offset alias for resampling
    resampled = df['Close'].resample(timeframe).last()
    returns = resampled.pct_change() * 100
    return returns


def heatmap_matrix(df, freq='M'):
    # freq: 'M' monthly, 'D' daily etc.
    series = df['Close'].resample(freq).last()
    dfm = series.to_frame('Close')
    dfm['year'] = dfm.index.year
    dfm['month'] = dfm.index.month
    table = dfm.pivot_table(values='Close', index='year', columns='month', aggfunc=lambda x: (x.last()/x.first()-1)*100)
    return table


def prepare_excel(df):
    out = io.BytesIO()
    # ensure tz-naive
    df2 = df.copy()
    if df2.index.tz is not None:
        df2.index = df2.index.tz_convert(TIMEZONE).tz_localize(None)
    df2.to_excel(out, index=True)
    out.seek(0)
    return out

# -------------------- UI --------------------
st.sidebar.title("Settings")
instrument = st.sidebar.selectbox("Instrument (default list)", [
    "^NSEI (NIFTY)", "^BSESN (SENSEX)", "^NSEBANK (BANKNIFTY)", "NIFTY_MID_SELECT.NS (MIDCAP)",
    "BTC-USD", "ETH-USD", "SOL-USD", "EURINR=X", "USDINR=X", "GC=F (Gold)", "SI=F (Silver)",
])
custom_ticker = st.sidebar.text_input("Or enter custom ticker (yfinance)")
if custom_ticker.strip():
    ticker = custom_ticker.strip()
else:
    ticker = instrument.split()[0]

interval_map = {
    '1m':'1m','2m':'2m','3m':'3m','5m':'5m','10m':'10m','15m':'15m','30m':'30m','60m':'60m','90m':'90m',
    '1h':'60m','2h':'120m','4h':'240m','1d':'1d','5d':'5d','1wk':'1wk','1mo':'1mo'
}

selected_interval = st.sidebar.selectbox('Interval', ['1m','3m','5m','10m','15m','30m','1h','2h','4h','1d','5d','1mo','3mo','6mo','1y','2y','5y','10y'])
selected_period = st.sidebar.selectbox('Period', ['7d','1mo','3mo','6mo','1y','2y','5y','10y','max'])

# Theme
theme_dark = st.sidebar.checkbox('Dark theme', value=True)
if theme_dark:
    bgcolor='#0e1117'
    fontcolor='white'
else:
    bgcolor='white'
    fontcolor='black'

st.title('Algo Trader — Streamlit')
col1, col2 = st.columns([1,3])
with col1:
    st.markdown(f"**Ticker:** {ticker}")
    fetch_btn = st.button('Fetch Data')
    st.markdown('---')
    st.write('Instructions: Click Fetch to load data. UI will persist.')

# Container to hold results so that UI elements remain visible
results_container = st.container()

if fetch_btn:
    with st.spinner('Fetching data — please wait'):
        try:
            df = fetch_yfinance(ticker, period=selected_period, interval=selected_interval)
        except Exception as e:
            st.error(f'Error fetching: {e}')
            df = pd.DataFrame()

    if df.empty:
        st.warning('No data returned. Try different interval/period or ticker.')
    else:
        # compute indicators
        df['RSI'] = compute_rsi(df['Close'])
        df['Change'] = df['Close'].pct_change()*100

        # Points up/down per timeframe (resample table)
        points_df = pd.DataFrame()
        for tf_label, tf in [('1D','1D'),('1H','1H'),('15min','15T'),('5min','5T')]:
            try:
                ret = compute_returns(df, timeframe=tf)
                points_df[tf_label+' Return %'] = ret
            except Exception:
                points_df[tf_label+' Return %'] = np.nan

        # Divergences
        divergences = detect_divergences(df['Close'], df['RSI'])

        # Fibonacci on visible range
        high = df['High'].max()
        low = df['Low'].min()
        fibs = fibonacci_levels(high, low)

        # Heatmap
        heat = heatmap_matrix(df, freq='M')

        # Ratio chart default compare to USDINR if available
        ratio_chart = None
        try:
            base = df['Close']
            if ticker.upper() not in ['USDINR=X']:
                other = fetch_yfinance('USDINR=X', period=selected_period, interval=selected_interval)
                if not other.empty:
                    ratio = base / other['Close'].reindex(base.index, method='ffill')
                    ratio_chart = ratio
        except Exception:
            ratio_chart = None

        # Predictions simple: logistic on recent returns sign
        X = df[['Close']].pct_change().fillna(0).values
        y = (df['Close'].shift(-1) > df['Close']).astype(int).fillna(0).values
        tscv = TimeSeriesSplit(n_splits=3)
        accs = []
        for train_idx, test_idx in tscv.split(X):
            model = LogisticRegression(max_iter=200)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], preds))
        predictive_accuracy = np.mean(accs)

        # Validation of predictions historically — simple backtest
        model = LogisticRegression(max_iter=200)
        Xf = X[:-1]
        yf_ = y[:-1]
        model.fit(Xf, yf_)
        preds_all = model.predict(Xf)
        val_acc = accuracy_score(yf_, preds_all)

        # UI Display
        with results_container:
            st.subheader('Price Chart')
            # make subplot with candlestick, RSI, ratio
            rows = 3 if ratio_chart is not None else 2
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.6,0.2]+([0.2] if ratio_chart is not None else []))
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            # add fib lines
            for k,v in fibs.items():
                fig.add_hline(y=v, line_dash='dash', annotation_text=f'Fib {k} {v:.2f}', opacity=0.6)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', yaxis='y2'), row=2, col=1)
            if ratio_chart is not None:
                fig.add_trace(go.Scatter(x=ratio_chart.index, y=ratio_chart.values, name='Ratio'), row=3, col=1)

            fig.update_layout(height=900, template='plotly_dark' if theme_dark else 'plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            # Divergence markers
            if divergences:
                st.markdown('**Divergences detected:**')
                for a,b,typ in divergences:
                    st.write(f'{typ} divergence between {a.date()} and {b.date()}')

            # Insights
            st.subheader('Short Insights (50 words)')
            insight = []
            insight.append(f'Instrument {ticker} shows recent RSI={df["RSI"].iloc[-1]:.1f}. Fib levels around {list(fibs.values())[:3]}. Prediction accuracy (simple logistic) {predictive_accuracy:.2f}.')
            st.write(' '.join(insight))

            # Dataframe display with export
            st.markdown('**Data (scrollable)**')
            st.dataframe(df.tail(500))
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv = df.to_csv().encode()
                st.download_button('Download CSV', data=csv, file_name=f'{ticker.replace("/","-")}_data.csv')
            with col_dl2:
                excel_io = prepare_excel(df)
                st.download_button('Download Excel', data=excel_io, file_name=f'{ticker.replace("/","-")}_data.xlsx')

            # Heatmap
            st.subheader('Monthly Heatmap (Year vs Month)')
            st.dataframe(heat.round(1))

            # Points table
            st.subheader('Points/Returns table')
            st.table(points_df.tail(20).style.format('{:.2f}'))

            # Validation metrics
            st.subheader('Prediction Validation')
            st.write(f'In-sample accuracy: {val_acc:.3f} | CV mean accuracy: {predictive_accuracy:.3f}')

            # Comparison tables for ratio
            if ratio_chart is not None:
                st.subheader('Ratio vs Close comparison (last 50 rows)')
                comp = pd.DataFrame({'Close': base.reindex(ratio_chart.index), 'OtherClose': other['Close'].reindex(ratio_chart.index), 'Ratio': ratio_chart.values})
                st.dataframe(comp.tail(50))

            # Summary
            st.subheader('Market View — Confluence')
            view = 'Bullish' if df['Close'].iloc[-1] > df['Close'].rolling(50).mean().iloc[-1] and df['RSI'].iloc[-1]>50 else 'Bearish'
            st.markdown(f'**Overall view:** {view} — based on 50-EMA and RSI confluence')

else:
    st.info('Click "Fetch Data" to load instrument data. UI controls will remain visible after fetch.')

# Footer
st.markdown('---')
st.caption('This tool is for educational/demo purposes. Backtesting and predictions are simplistic. Ensure you validate before trading.')
