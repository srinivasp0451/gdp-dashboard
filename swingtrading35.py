import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import io

st.set_page_config(layout="wide", page_title="Algo Trader Dashboard")

# ----------------------- Helper functions -----------------------
INTERVAL_MAP = {
    "1m": "1m","2m":"2m","3m":"3m","5m":"5m","10m":"10m","15m":"15m",
    "30m":"30m","60m":"60m","90m":"90m","120m":"120m","4h":"240m",
    "1d":"1d","5d":"5d","1wk":"1wk","1mo":"1mo","3mo":"3mo",
}

# Flatten multiindex columns from yfinance
def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_" .join([str(c) for c in col if str(c) != '']) for col in df.columns]
    return df

# Make datetimes timezone-naive in IST for Excel
IST = pytz.timezone('Asia/Kolkata')

def to_ist_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        localized = idx.tz_localize(pytz.UTC).tz_convert(IST)
    else:
        localized = idx.tz_convert(IST)
    return localized.tz_localize(None)

# RSI implementation
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# Simple moving average
def sma(series: pd.Series, window: int):
    return series.rolling(window).mean()

# MACD
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return macd_line, sig, hist

# Fibonacci levels
def fibonacci_levels(high, low):
    diff = high - low
    levels = {
        '0.0': high,
        '0.236': high - 0.236*diff,
        '0.382': high - 0.382*diff,
        '0.5': high - 0.5*diff,
        '0.618': high - 0.618*diff,
        '1.0': low
    }
    return levels

# Simple prediction engine: naive pattern-based
def simple_prediction(df: pd.DataFrame):
    # returns a short text prediction and confidence
    if df.empty:
        return "No data", 0.0
    rsi = compute_rsi(df['Close'])
    last_rsi = rsi.iloc[-1]
    ma50 = sma(df['Close'],50).iloc[-1] if len(df)>=50 else np.nan
    ma200 = sma(df['Close'],200).iloc[-1] if len(df)>=200 else np.nan
    price = df['Close'].iloc[-1]
    conf = 0.3
    text = "Neutral"
    if last_rsi < 30 and price > ma50:
        text = "Potential rebound (bullish)"
        conf = 0.5
    elif last_rsi > 70 and price < ma50:
        text = "Potential pullback (bearish)"
        conf = 0.5
    elif ma50>ma200:
        text = "Bullish trend (MA50>MA200)"
        conf = 0.6
    elif ma50<ma200:
        text = "Bearish trend (MA50<MA200)"
        conf = 0.6
    return text, conf

# Backtest simple predictions: check next N bars moved in predicted direction
def validate_predictions(df: pd.DataFrame, predictions: list, lookahead=5):
    # predictions: list of (index, direction) direction: 'bull'/'bear'
    results = []
    for idx, direction in predictions:
        try:
            pos = df.index.get_loc(idx)
        except KeyError:
            continue
        end_pos = min(pos+lookahead, len(df)-1)
        future_close = df['Close'].iloc[end_pos]
        start = df['Close'].iloc[pos]
        moved_up = future_close > start
        success = (moved_up and direction=='bull') or (not moved_up and direction=='bear')
        results.append(success)
    if len(results)==0:
        return 0,0
    return sum(results), len(results)

# Heatmap generation
def generate_heatmap(df: pd.DataFrame, freq='M'):
    if df.empty:
        return pd.DataFrame()
    df2 = df['Close'].copy()
    df2 = df2.resample('D').ffill()
    df2.index = to_ist_naive(df2.index)
    df2 = df2.to_frame()
    df2['year'] = df2.index.year
    df2['month'] = df2.index.month
    pivot = df2.groupby(['year','month']).last().unstack(level=0)['Close']
    # pivot columns are years
    return pivot

# Points change table
def points_table(df: pd.DataFrame, periods=['1d','5d','1mo']):
    out = {}
    now = df.index[-1]
    for p in periods:
        try:
            if p.endswith('d'):
                days = int(p[:-1])
                past = now - pd.Timedelta(days=days)
            elif p.endswith('mo'):
                months = int(p[:-2])
                past = now - pd.DateOffset(months=months)
            else:
                past = now - pd.Timedelta(days=1)
            past_idx = df.index.asof(past)
            if pd.isna(past_idx):
                continue
            change = df['Close'].iloc[-1] - df.loc[past_idx]['Close']
            pct = change / df.loc[past_idx]['Close'] * 100
            out[p] = (change, pct)
        except Exception:
            continue
    t = pd.DataFrame.from_dict({k: {'Points': v[0], 'Pct': v[1]} for k,v in out.items()}, orient='index')
    return t

# ----------------------- Streamlit UI -----------------------
st.title("Algo Trader — Multi-Asset Streamlit Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    theme = st.checkbox("Dark theme", value=False)
    symbol = st.text_input("Ticker or symbol (yfinance)", value="^NSEI")
    timeframe = st.selectbox("Interval", options=['1m','2m','3m','5m','10m','15m','30m','60m','90m','120m','1d','5d','1wk','1mo','3mo'], index=10)
    period = st.selectbox("Period (yfinance period)", options=['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','max'], index=1)
    bins = st.slider("Number of bins (ratio charts)", min_value=5, max_value=80, value=20)
    fetch = st.button("Fetch data")

# Keep UI visible after click by storing in session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if fetch:
    with st.spinner("Fetching data — respecting rate limits..."):
        try:
            # yfinance sometimes refuses 1m for long periods; use period mapping
            df = yf.download(tickers=symbol, interval=timeframe, period=period, progress=False, threads=False)
            df = flatten_df(df)
            # Ensure required columns exist
            for c in ['Open','High','Low','Close','Volume']:
                if c not in df.columns:
                    # try lower-case
                    if c.lower() in df.columns:
                        df[c] = df[c.lower()]
            df.dropna(subset=['Close'], inplace=True)
            st.session_state.df = df
            st.success("Data fetched — {} rows".format(len(df)))
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")

if not st.session_state.df.empty:
    df = st.session_state.df.copy()
    # Ensure datetime index timezone to IST naive
    df.index = pd.to_datetime(df.index)
    df.index = to_ist_naive(df.index)

    # Calculations
    df['RSI'] = compute_rsi(df['Close'])
    df['SMA50'] = sma(df['Close'],50)
    df['SMA200'] = sma(df['Close'],200)
    df['MACD'], df['MACD_SIG'], df['MACD_HIST'] = macd(df['Close'])

    # Points/returns table
    pts_tbl = points_table(df, periods=['1d','5d','1mo','3mo'])

    # Layout: main charts
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader(f"{symbol} — Candlestick")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.2,0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candles'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200'), row=1, col=1)
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash='dash', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash', row=2, col=1)
        # MACD
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='MACD Hist'), row=3, col=1)
        fig.update_layout(height=900, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Fibonacci on visible range (last 200 bars)
        window = df['Close'].iloc[-200:]
        fibs = fibonacci_levels(window.max(), window.min())
        st.markdown("**Fibonacci levels (last 200 bars)**")
        st.write(fibs)

        # Prediction and insight
        pred_text, conf = simple_prediction(df)
        st.markdown(f"**Prediction:** {pred_text} — confidence {conf:.0%}")
        insight = f"{symbol} last close {df['Close'].iloc[-1]:.2f}. RSI {df['RSI'].iloc[-1]:.1f}. SMA50 {df['SMA50'].iloc[-1]:.2f if not np.isnan(df['SMA50'].iloc[-1]) else 'n/a'}. {pred_text}."
        st.info(insight)

        # Dataframe view and export
        st.subheader("Chart Data")
        st.dataframe(df.tail(500))
        # Export buttons
        to_csv = df.copy()
        to_csv.index = to_ist_naive(to_csv.index)
        csv = to_csv.to_csv(index=True)
        st.download_button("Download CSV", csv, file_name=f"{symbol.replace('^','')}_{timeframe}.csv")

        # Excel export - remove timezone
        excel_buffer = io.BytesIO()
        to_xls = to_csv.copy()
        to_xls.index = pd.to_datetime(to_xls.index)
        try:
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                to_xls.to_excel(writer, sheet_name='data')
            st.download_button("Download Excel", data=excel_buffer.getvalue(), file_name=f"{symbol}_{timeframe}.xlsx")
        except Exception as e:
            st.warning(f"Excel export skipped: {e}")

    with col2:
        st.subheader("Quick Stats & Heatmaps")
        st.table(pts_tbl.style.format({"Points":"{:.2f}","Pct":"{:.2f}%"}))

        st.markdown("---")
        st.markdown("**Monthly Heatmap (Year vs Month)**")
        heat = generate_heatmap(df)
        if not heat.empty:
            st.dataframe(heat.round(1))
        else:
            st.write("No heatmap data")

    # Ratio chart example: ratio to USDINR if available
    st.subheader("Ratio Chart")
    base = df['Close']
    # allow user to pick comparison symbol
    comp_symbol = st.text_input("Comparison symbol (for ratio)", value="USDINR=X")
    if st.button("Fetch comparison and compute ratio"):
        comp = yf.download(tickers=comp_symbol, interval=timeframe, period=period, progress=False, threads=False)
        comp = flatten_df(comp)
        if 'Close' in comp.columns:
            comp.index = pd.to_datetime(comp.index)
            comp.index = to_ist_naive(comp.index)
            # align
            merged = pd.merge_asof(df[['Close']].reset_index().rename(columns={'index':'Datetime'}),
                                   comp[['Close']].reset_index().rename(columns={'index':'Datetime'}), on='Datetime')
            merged.set_index('Datetime', inplace=True)
            merged.dropna(inplace=True)
            merged['Ratio'] = merged['Close_x'] / merged['Close_y']
            # bin ratio
            merged['bin'] = pd.cut(merged['Ratio'], bins=bins)
            # show ratio chart
            figr = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
            figr.add_trace(go.Scatter(x=merged.index, y=merged['Ratio'], name='Ratio'), row=1, col=1)
            figr.add_trace(go.Scatter(x=merged.index, y=merged['Close_x'], name=f'{symbol} Close'), row=2, col=1)
            figr.add_trace(go.Scatter(x=merged.index, y=merged['Close_y'], name=f'{comp_symbol} Close'), row=2, col=1)
            st.plotly_chart(figr, use_container_width=True)
            # bin stats
            bin_stats = merged.groupby('bin').apply(lambda g: pd.Series({'mean_return': (g['Close_x'].pct_change().mean()*252), 'count': len(g)}))
            st.dataframe(bin_stats.sort_values('count', ascending=False).head(50))
        else:
            st.error("Comparison symbol did not return Close prices")

    # Validation/backtest area
    st.subheader("Prediction Validation")
    # simplistic validation: generate signals when RSI crosses extreme
    signals = []
    rsi = df['RSI']
    for i in range(1, len(rsi)):
        if rsi.iloc[i-1] < 30 and rsi.iloc[i] >=30:
            signals.append((df.index[i], 'bull'))
        if rsi.iloc[i-1] > 70 and rsi.iloc[i] <=70:
            signals.append((df.index[i], 'bear'))
    succ, total = validate_predictions(df, signals, lookahead=5)
    st.write(f"Validated signals success: {succ}/{total} — accuracy: {(succ/total*100) if total>0 else 0:.1f}%")

    # Heatmap options
    st.subheader("Advanced Heatmaps")
    hm_opt = st.radio("Heatmap type", options=['Year vs Month','Day vs Month','Quarter vs Year','Week vs Year'])
    if hm_opt == 'Year vs Month':
        hm = generate_heatmap(df)
        if not hm.empty:
            st.dataframe(hm.round(1))
    else:
        st.info("Other heatmap types are available in next versions")

    # Synchronized charts: place symbol, RSI, ratio one above another (already stacked in subplot)
    st.markdown("---")
    st.write("**Summary of insights (auto-generated)**")
    insights = []
    insights.append(f"Range (last): {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    insights.append(f"RSI last: {df['RSI'].iloc[-1]:.1f}")
    insights.append(f"Momentum: MACD hist last {df['MACD_HIST'].iloc[-1]:.4f}")
    st.write("\n".join(insights))

else:
    st.info("Click 'Fetch data' to get started. This avoids auto-fetch and reduces yfinance rate limits.")

# Footer
st.markdown("---")
st.caption("Notes: This app uses simple heuristics for predictions. Backtest/validate before trading. Timezones converted to IST for display and exports.")
