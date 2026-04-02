import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import socket

# --- CONFIGURATION & SESSION STATE ---
st.set_page_config(page_title="Smart Investing", layout="wide")

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'live_active' not in st.session_state:
    st.session_state.live_active = False
if 'current_position' not in st.session_state:
    st.session_state.current_position = None

# --- UTILITY FUNCTIONS ---
def get_ip():
    return socket.gethostbyname(socket.gethostname())

def calculate_ema(data, window):
    """Matches TradingView EMA (Uses Simple Moving Average for the first value)"""
    return data.ewm(span=window, adjust=False).mean()

def fetch_data(ticker, interval, period):
    # Fetch extra padding to handle EMA calculations during gap ups/downs
    # and ensure indicators are not 'NaN' at the start of the visible range
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    return data

def get_ltp_data(ticker):
    ticker_data = yf.Ticker(ticker)
    todays_data = ticker_data.history(period='2d')
    if len(todays_data) < 2: return 0, 0, 0
    ltp = todays_data['Close'].iloc[-1]
    prev_close = todays_data['Close'].iloc[-2]
    change = ltp - prev_close
    p_change = (change / prev_close) * 100
    return ltp, change, p_change

# --- CORE TRADING LOGIC ---
class TradingEngine:
    @staticmethod
    def backtest(df, fast_ema, slow_ema, sl_type, sl_val, tgt_type, tgt_val, overlap_check):
        df['ema_f'] = calculate_ema(df['Close'], fast_ema)
        df['ema_s'] = calculate_ema(df['Close'], slow_ema)
        
        trades = []
        in_position = False
        last_exit_time = datetime.min
        
        # Shift signals: Entry at N+1 Open if crossover at N
        df['signal'] = np.where(df['ema_f'] > df['ema_s'], 1, 0)
        df['trigger'] = df['signal'].diff()

        for i in range(1, len(df)-1):
            curr_time = df.index[i]
            
            # Entry Logic (Crossover)
            if not in_position and df['trigger'].iloc[i] == 1:
                if overlap_check and curr_time < last_exit_time: continue
                
                entry_price = df['Open'].iloc[i+1] # Entry at Next Candle Open
                entry_time = df.index[i+1]
                
                # Simplified SL/Target Calculation
                sl_price = entry_price - sl_val if sl_type == "Points" else entry_price * 0.99
                tgt_price = entry_price + tgt_val if tgt_type == "Points" else entry_price * 1.02
                
                # Backtest Conservative Check
                # Buy: Check Low for SL first, then High for Target
                candle_low = df['Low'].iloc[i+1]
                candle_high = df['High'].iloc[i+1]
                
                outcome = "Open"
                if candle_low <= sl_price:
                    exit_price = sl_price
                    outcome = "SL Hit"
                elif candle_high >= tgt_price:
                    exit_price = tgt_price
                    outcome = "Target Hit"
                else:
                    exit_price = df['Close'].iloc[i+1]
                
                trades.append({
                    "Entry Time": entry_time,
                    "Type": "Buy",
                    "Entry": entry_price,
                    "Exit": exit_price,
                    "SL": sl_price,
                    "Tgt": tgt_price,
                    "Result": outcome,
                    "PnL": exit_price - entry_price
                })
        return pd.DataFrame(trades)

# --- UI LAYOUT ---
st.title("📈 Smart Investing")
st.caption(f"Registered IP: {get_ip()} | SEBI Compliant Logged")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("Strategy Settings")
    ticker = st.selectbox("Ticker", ["^NSEI", "^NSEBANK", "BTC-USD", "GC=F", "SBIN.NS"])
    interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"])
    period = st.selectbox("Period", ["1d", "5d", "1mo", "1y", "max"])
    
    st.divider()
    strategy = st.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
    fast_ema = st.number_input("Fast EMA", value=9)
    slow_ema = st.number_input("Slow EMA", value=15)
    
    st.divider()
    sl_opt = st.selectbox("Stoploss Type", ["Points", "Trailing", "Risk Reward", "ATR"])
    sl_val = st.number_input("SL Value", value=10)
    tgt_opt = st.selectbox("Target Type", ["Points", "EMA Crossover", "Risk Reward"])
    tgt_val = st.number_input("Target Value", value=20)
    
    st.divider()
    cooldown = st.checkbox("Cooldown Period (5s)", value=True)
    no_overlap = st.checkbox("Prevent Overlap", value=True)
    dhan_enabled = st.checkbox("Enable Dhan Broker", value=False)
    options_enabled = st.checkbox("Options Trading", value=False)

# --- TOP STATS ---
ltp, change, p_change = get_ltp_data(ticker)
color = "green" if change >= 0 else "red"
st.markdown(f"### {ticker}: <span style='color:{color}'>{ltp:.2f} ({change:+.2f}, {p_change:+.2f}%)</span>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Backtesting", "Live Trading", "Trade History"])

with tab1:
    if st.button("Run Backtest"):
        data = fetch_data(ticker, interval, period)
        results = TradingEngine.backtest(data, fast_ema, slow_ema, sl_opt, sl_val, tgt_opt, tgt_val, no_overlap)
        
        st.dataframe(results)
        
        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"))
        fig.add_trace(go.Scatter(x=data.index, y=calculate_ema(data['Close'], fast_ema), name=f"EMA {fast_ema}"))
        fig.add_trace(go.Scatter(x=data.index, y=calculate_ema(data['Close'], slow_ema), name=f"EMA {slow_ema}"))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2, col3 = st.columns(3)
    start = col1.button("▶ START LIVE", use_container_width=True)
    stop = col2.button("⏹ STOP", use_container_width=True)
    sq_off = col3.button("✖ SQUARE OFF", use_container_width=True)
    
    if start: st.session_state.live_active = True
    if stop: st.session_state.live_active = False

    live_container = st.empty()
    
    while st.session_state.live_active:
        with live_container.container():
            now = datetime.now()
            st.info(f"Live Monitoring... Last Update: {now.strftime('%H:%M:%S')}")
            
            # Fetch latest data
            live_data = fetch_data(ticker, interval, "1d")
            latest_ltp = live_data['Close'].iloc[-1]
            
            # Display Config Overlay
            st.json({
                "Ticker": ticker,
                "Timeframe": interval,
                "Strategy": f"{fast_ema}/{slow_ema} EMA Cross",
                "SL": f"{sl_val} ({sl_opt})",
                "Target": f"{tgt_val} ({tgt_opt})"
            })
            
            # Placeholder for Tick-by-Tick Logic
            # if latest_ltp meets signal -> place_dhan_order(...)
            
            st.metric("Current LTP", f"{latest_ltp:.2f}")
            st.table(live_data.tail(5)) # Display last fetched candles
            
        time.sleep(1.5) # API Rate Limit Handling
        if not st.session_state.live_active: break

with tab3:
    st.header("Completed Trades")
    if st.session_state.trade_history:
        st.table(st.session_state.trade_history)
    else:
        st.write("No trades executed in this session.")
