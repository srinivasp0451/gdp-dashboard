import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz

# Instrument configuration & mapping
default_tickers = {
    "Nifty50": "^NSEI",
    "Sensex": "^BSESN",
    "BankNifty": "^NSEBANK",
    "Midcap Nifty": "^NSEMDCP50",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "Solana": "SOL-USD",
    "USDINR": "USDINR=X",
    "EURINR": "EURINR=X",
    "GBPINR": "GBPINR=X",
    "JPYINR": "JPYINR=X",
    "GOLD (INR)": "GOLDINR=X",
    "GOLD (USD)": "GC=F",
    "SILVER (INR)": "SILVERINR=X",
    "SILVER (USD)": "SI=F",
    "COPPER (INR)": "COPPERINR=X",
    "COPPER (USD)": "HG=F"
}
# Add top Indian stocks if needed
top_indian_stocks = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "HDFC": "HDFCBANK.NS",
}

# Merge for UI
all_tickers = {**default_tickers, **top_indian_stocks}
period_options = [
    "1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h",
    "1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", 
    "6y", "10y", "15y", "20y", "25y", "30y"
]
interval_options = [
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]
# -------------------------------------------------------------------

st.set_page_config(page_title="Algo Trader Dashboard", layout="wide")
st.title('🦾 Pro-Grade Algo Trader Dashboard')

# Theme selector
dark_mode = st.sidebar.checkbox("Dark Theme", value=True)
if dark_mode:
    st.markdown("<style>body{background-color: #222;color: #fff;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background-color: #f9f9f9;color: #222;}</style>", unsafe_allow_html=True)

# Instrument selection UI
st.sidebar.header("Instrument & Timeframe Selection")
selected_symbols = st.sidebar.multiselect("Select Instruments", list(all_tickers.keys()), default=list(all_tickers.keys())[:5])
custom_ticker = st.sidebar.text_input("Custom Ticker (Yahoo format)", "")
period = st.sidebar.selectbox("Select Period", period_options, index=period_options.index("1mo"))
interval = st.sidebar.selectbox("Select Interval", interval_options, index=interval_options.index("5m"))
num_bins = st.sidebar.slider("Ratio Chart Bins", 5, 30, 10)
export_fmt = st.sidebar.radio("Export format", ["CSV", "Excel"])

# Fetch trigger button (no auto-fetch!)
fetch_button = st.button("Fetch & Analyze Data")

def fetch_yfinance(ticker, period, interval):
    # yfinance sometimes produces multiindex, timezone-aware datetimes, and rate limit errors. Handle them!
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"No data fetched for {ticker}")
            return None
        # Flatten multiindex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        df.reset_index(inplace=True)
        # Convert all dates to IST, make timezone unaware (for Excel export)
        if 'Datetime' in df.columns:
            df['Datetime'] = df['Datetime'].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize("Asia/Kolkata").dt.tz_localize(None)
        elif 'index' in df.columns:
            df['index'] = pd.to_datetime(df['index']).dt.tz_localize("Asia/Kolkata").dt.tz_localize(None)
        elif 'Open' in df.columns:  # last fallback
            if isinstance(df.iloc[0,0], pd.Timestamp):
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]]).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        df.dropna(axis=1, how='all', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

# Core analysis and charting logic starts below
def analyze_and_plot(df, symbol, ratio_df=None):
    # Calculate percentage change, color-code up/down
    df['Change'] = df['Close'].diff()
    df['Change %'] = df['Close'].pct_change() * 100
    df['Up/Down'] = np.where(df['Change'] > 0, 'Up', 'Down')

    # Candlestick and line chart
    fig_candlestick = go.Figure()
    fig_candlestick.add_trace(go.Candlestick(
        x=df[df.columns[0]], open=df['Open'],
        high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='green', decreasing_line_color='red'
    ))
    fig_candlestick.update_layout(
        title=f"{symbol} Candlestick ({period}, {interval})",
        xaxis_rangeslider_visible=True,
        template='plotly_dark' if dark_mode else 'plotly_white'
    )
    # Add trendlines for up/down
    min_idx, max_idx = df['Close'].idxmin(), df['Close'].idxmax()
    fig_candlestick.add_trace(go.Scatter(
        x=[df[df.columns[0]].iloc[min_idx], df[df.columns[0]].iloc[max_idx]],
        y=[df['Close'].iloc[min_idx], df['Close'].iloc[max_idx]],
        mode='lines',
        line=dict(color='blue', dash='dot'),
        name='Trendline'
    ))
    # Calculate RSI natively
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    # RSI divergence logic: bullish if RSI rising, close falling; bearish if opposite
    divergence = []
    for i in range(1, len(df)):
        if df['RSI'].iloc[i] > df['RSI'].iloc[i-1] and df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            divergence.append('Bullish')
        elif df['RSI'].iloc[i] < df['RSI'].iloc[i-1] and df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            divergence.append('Bearish')
        else:
            divergence.append('None')
    df['RSI Divergence'] = ['None'] + divergence

    # Ratio chart if ratio_df is provided
    if ratio_df is not None:
        ratio_series = df['Close'] / ratio_df['Close']
        bins = pd.cut(ratio_series, bins=num_bins)
        ratio_return = df['Close'].groupby(bins).apply(lambda x: x.pct_change().mean())
        # Plot ratio histogram
        fig_ratio = go.Figure([go.Bar(x=ratio_return.index.astype(str), y=ratio_return.values, marker_color='purple')])
        fig_ratio.update_layout(title=f"Ratio Chart: {symbol} / {ratio_df['Symbol'][0]}", template='plotly_dark' if dark_mode else 'plotly_white', xaxis_title="Ratio Bin", yaxis_title="Avg Return (%)")
        st.plotly_chart(fig_ratio, use_container_width=True)
        # Comparison table
        comp_df = pd.DataFrame({
            "Date": df[df.columns[0]],
            f"{symbol} Close": df['Close'],
            f"{ratio_df['Symbol'][0]} Close": ratio_df['Close'],
            "Ratio": ratio_series
        })
        st.dataframe(comp_df, height=200)
    # Fibonacci levels plotting
    min_price, max_price = df['Close'].min(), df['Close'].max()
    diff = max_price - min_price
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    fib_vals = [min_price + l * diff for l in fib_levels]
    for lvl, val in zip(fib_levels, fib_vals):
        fig_candlestick.add_hline(y=val, line_dash="dash", annotation_text=f"Fib {lvl:.3f} ({val:.2f})", annotation_position="top left")
    # RSI line chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df[df.columns[0]], y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
    fig_rsi.update_layout(title=f"{symbol} RSI", template='plotly_dark' if dark_mode else 'plotly_white')

    # Synchronized cursors: limitation in Streamlit, but present charts stacked for visual sync
    st.plotly_chart(fig_candlestick, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Points Up/Down summary (color table)
    summary_df = df[[df.columns[0], 'Close', 'Change', 'Change %', 'Up/Down', 'RSI', 'RSI Divergence']].copy()
    st.dataframe(summary_df.style.apply(lambda x: ['background-color: #2ecc40' if v == 'Up' else 'background-color: #ff4136' for v in x['Up/Down']], axis=1), height=300)

    # Heatmap of returns (day vs month, year vs month, quarter vs year)
    returns_df = df.copy()
    returns_df['Year'] = returns_df[df.columns[0]].dt.year
    returns_df['Month'] = returns_df[df.columns[0]].dt.month
    returns_df['Day'] = returns_df[df.columns[0]].dt.day
    returns_df['Quarter'] = returns_df[df.columns[0]].dt.quarter
    returns_df['Week'] = returns_df[df.columns[0]].dt.isocalendar().week
    pivot_month_day = returns_df.pivot_table(index='Day', columns='Month', values='Change %', aggfunc="mean").round(1)
    pivot_year_month = returns_df.pivot_table(index='Year', columns='Month', values='Change %', aggfunc="mean").round(1)
    pivot_quarter_year = returns_df.pivot_table(index='Quarter', columns='Year', values='Change %', aggfunc="mean").round(1)
    pivot_week_year = returns_df.pivot_table(index='Week', columns='Year', values='Change %', aggfunc="mean").round(1)
    for pivot, label in zip(
        [pivot_month_day, pivot_year_month, pivot_quarter_year, pivot_week_year],
        ["Day vs Month", "Year vs Month", "Quarter vs Year", "Week vs Year"]
    ):
        st.write(f"Heatmap: {label}")
        st.dataframe(pivot.style.background_gradient(cmap="coolwarm"), height=150)

    # Export data options
    if export_fmt == "CSV":
        st.download_button(f"Download {symbol} Data (CSV)", data=summary_df.to_csv(index=False), file_name=f"{symbol}_data.csv")
    else:
        summary_df.to_excel("temp.xlsx", index=False)  # Save to disk
        with open("temp.xlsx", "rb") as f:
            st.download_button(f"Download {symbol} Data (Excel)", data=f, file_name=f"{symbol}_data.xlsx")

    # Auto insights (layman, 50 words)
    last_change_pct = df['Change %'].iloc[-1]
    insight = f"{symbol}: Latest move is {'up' if last_change_pct > 0 else 'down'}, with {last_change_pct:.2f}% return in last period. RSI at {df['RSI'].iloc[-1]:.1f}. Fibonacci resistance at {max(fib_vals):.2f}, support at {min(fib_vals):.2f}. Observed {'bullish' if last_change_pct>0 else 'bearish'} confluence."
    st.info(insight)

    # Prediction based on RSI/Fibonacci/Trend
    pred = "bullish" if df['RSI'].iloc[-1] > 60 and last_change_pct > 0 else "bearish" if df['RSI'].iloc[-1] < 40 and last_change_pct < 0 else "sideways"
    st.markdown(f"Prediction: **{pred.upper()}** move expected based on last RSI, trend, and retracement.")

# ------------------------------------------------

# Prediction validation storage
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []
if "pred_accuracy" not in st.session_state:
    st.session_state["pred_accuracy"] = []

if fetch_button:
    # Aggregated fetch logic (all tickers)
    symbols_to_fetch = selected_symbols.copy()
    if custom_ticker:
        symbols_to_fetch.append(custom_ticker)
    for symbol in symbols_to_fetch:
        ticker = all_tickers.get(symbol, symbol)
        df = fetch_yfinance(ticker, period, interval)
        if df is not None and "Close" in df.columns:
            df['Symbol'] = symbol
            st.markdown(f"### {symbol} Analysis")
            # For ratio chart: use next ticker
            ratio_df = None
            if len(symbols_to_fetch) >= 2 and symbol == symbols_to_fetch[0]:
                next_ticker = all_tickers.get(symbols_to_fetch[1], symbols_to_fetch[1])
                ratio_df = fetch_yfinance(next_ticker, period, interval)
                if ratio_df is not None and "Close" in ratio_df.columns:
                    ratio_df['Symbol'] = symbols_to_fetch[1]
            analyze_and_plot(df, symbol, ratio_df=ratio_df)
            # Prediction accuracy logic
            pred = "bullish" if df['RSI'].iloc[-1] > 60 and df['Change %'].iloc[-1] > 0 else "bearish" if df['RSI'].iloc[-1] < 40 and df['Change %'].iloc[-1] < 0 else "sideways"
            actual_move = "bullish" if df['Change %'].iloc[-1] > 0 else "bearish" if df['Change %'].iloc[-1] < 0 else "sideways"
            st.session_state["prediction_history"].append((symbol, pred, actual_move, df[df.columns[0]].iloc[-1]))
    # Accuracy summary table
    pred_df = pd.DataFrame(st.session_state["prediction_history"], columns=["Symbol", "Predicted", "Actual", "Datetime"])
    pred_df['Hit/Miss'] = pred_df['Predicted'] == pred_df['Actual']
    acc = (pred_df['Hit/Miss'].sum() / len(pred_df)) * 100 if len(pred_df)>0 else 0
    st.markdown("### Prediction Accuracy Summary")
    st.dataframe(pred_df, height=200)
    st.success(f"Prediction accuracy to date: {acc:.2f}%")

    # Layman summary for all instruments
    instruments_summary = []
    for symbol in symbols_to_fetch:
        ticker = all_tickers.get(symbol, symbol)
        df = fetch_yfinance(ticker, period, interval)
        if df is not None and "Close" in df.columns:
            last_chg = df['Change %'].iloc[-1]
            min_r, max_r = df['Close'].min(), df['Close'].max()
            rsi = df['RSI'].iloc[-1]
            summary = f"{symbol}: Last close {df['Close'].iloc[-1]:.2f}, change {last_chg:.2f}%, RSI {rsi:.2f}, Range {min_r:.2f}-{max_r:.2f}"
            instruments_summary.append(summary)
    st.markdown("#### All Instruments Summary")
    for summ in instruments_summary:
        st.write(summ)
