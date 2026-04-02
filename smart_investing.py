import streamlit as st
import yfinance as yf
import numpy as np
import time
import datetime
import plotly.graph_objects as go

# ------------------------------
# EMA Calculation (manual, TradingView style)
# ------------------------------
def ema(series, period):
    alpha = 2 / (period + 1)
    ema_values = []
    for i, price in enumerate(series):
        if i == 0:
            ema_values.append(price)
        else:
            ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
    return ema_values

# ------------------------------
# Fetch Data
# ------------------------------
def fetch_data(ticker, interval, period):
    try:
        data = yf.download(ticker, interval=interval, period=period)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ------------------------------
# Strategy: EMA Crossover
# ------------------------------
def ema_crossover(data, fast=9, slow=15):
    data['ema_fast'] = ema(data['Close'].values, fast)
    data['ema_slow'] = ema(data['Close'].values, slow)
    signals = []
    for i in range(1, len(data)):
        if data['ema_fast'][i-1] < data['ema_slow'][i-1] and data['ema_fast'][i] > data['ema_slow'][i]:
            signals.append(('BUY', data.index[i], data['Open'][i]))
        elif data['ema_fast'][i-1] > data['ema_slow'][i-1] and data['ema_fast'][i] < data['ema_slow'][i]:
            signals.append(('SELL', data.index[i], data['Open'][i]))
    return signals

# ------------------------------
# Backtesting Logic
# ------------------------------
def backtest(data, signals, sl_points=10, target_points=20):
    results = []
    violations = 0
    for signal in signals:
        trade_type, entry_time, entry_price = signal
        candle = data.loc[entry_time]
        if trade_type == 'BUY':
            sl = entry_price - sl_points
            target = entry_price + target_points
            # Conservative: check SL first
            if candle['Low'] <= sl:
                exit_price = sl
                reason = "Stoploss hit"
            elif candle['High'] >= target:
                exit_price = target
                reason = "Target hit"
            else:
                exit_price = candle['Close']
                reason = "Exit at close"
            if candle['Low'] < sl and candle['High'] > target:
                violations += 1
        else:  # SELL
            sl = entry_price + sl_points
            target = entry_price - target_points
            if candle['High'] >= sl:
                exit_price = sl
                reason = "Stoploss hit"
            elif candle['Low'] <= target:
                exit_price = target
                reason = "Target hit"
            else:
                exit_price = candle['Close']
                reason = "Exit at close"
            if candle['High'] > sl and candle['Low'] < target:
                violations += 1
        pnl = exit_price - entry_price if trade_type == 'BUY' else entry_price - exit_price
        results.append({
            "Trade Type": trade_type,
            "Entry Time": entry_time,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "Reason": reason,
            "PnL": pnl
        })
    return results, violations

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Smart Investing", layout="wide")
st.title("📈 Smart Investing")

tabs = st.tabs(["Backtesting", "Live Trading", "Trade History"])

# ------------------------------
# Sidebar Config
# ------------------------------
ticker = st.sidebar.text_input("Ticker", "NSEI")  # Nifty50 default
interval = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1h","1d","1wk"])
period = st.sidebar.selectbox("Period", ["1d","5d","7d","1mo","3mo","6mo","1y","2y","5y","10y","20y"])
fast_ema = st.sidebar.number_input("Fast EMA", value=9)
slow_ema = st.sidebar.number_input("Slow EMA", value=15)
sl_points = st.sidebar.number_input("Stoploss Points", value=10)
target_points = st.sidebar.number_input("Target Points", value=20)
cooldown = st.sidebar.checkbox("Cooldown (5s)", value=True)
prevent_overlap = st.sidebar.checkbox("Prevent Overlap", value=True)

# ------------------------------
# Backtesting Tab
# ------------------------------
with tabs[0]:
    st.header("Backtesting")
    data = fetch_data(ticker, interval, period)
    if data is not None and not data.empty:
        signals = ema_crossover(data, fast_ema, slow_ema)
        results, violations = backtest(data, signals, sl_points, target_points)
        st.write("Results Table")
        st.dataframe(results)
        st.write(f"Violations of conservative SL-first rule: {violations}")
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name="Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data['ema_fast'], line=dict(color='blue'), name=f"EMA {fast_ema}"))
        fig.add_trace(go.Scatter(x=data.index, y=data['ema_slow'], line=dict(color='red'), name=f"EMA {slow_ema}"))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Live Trading Tab
# ------------------------------
with tabs[1]:
    st.header("Live Trading")
    start = st.button("Start")
    stop = st.button("Stop")
    squareoff = st.button("Square-off")
    if start:
        st.write("Live trading started with config:")
        st.json({
            "Ticker": ticker,
            "Interval": interval,
            "Period": period,
            "Fast EMA": fast_ema,
            "Slow EMA": slow_ema,
            "Stoploss": sl_points,
            "Target": target_points
        })
        # Simulate live data fetch
        data = fetch_data(ticker, interval, period)
        if data is not None and not data.empty:
            st.write("Last fetched candle:")
            st.write(data.tail(1))
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name="Price"))
            fig.add_trace(go.Scatter(x=data.index, y=ema(data['Close'].values, fast_ema), line=dict(color='blue'), name=f"EMA {fast_ema}"))
            fig.add_trace(go.Scatter(x=data.index, y=ema(data['Close'].values, slow_ema), line=dict(color='red'), name=f"EMA {slow_ema}"))
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Trade History Tab
# ------------------------------
with tabs[2]:
    st.header("Trade History")
    st.write("Completed trades will appear here (persist even during live trading).")
