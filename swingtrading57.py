import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
from scipy.signal import argrelextrema

# ==========================================
# 1. CORE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Advanced Pro-Algo Terminal")
IST = pytz.timezone('Asia/Kolkata')

if 'trading_active' not in st.session_state:
    st.session_state.update({
        'trading_active': False,
        'current_position': None, # Dict: type, entry, sl, tp, time, trail_val
        'trade_history': [],
        'trade_log': [],
        'iteration_count': 0,
        'last_api_call': 0
    })

def add_log(msg):
    t = datetime.now(IST).strftime("%H:%M:%S")
    st.session_state.trade_log.insert(0, f"[{t}] {msg}")

# ==========================================
# 2. ADVANCED MATHEMATICAL INDICATORS
# ==========================================
def calculate_atr(df, period=14):
    h_l = df['High'] - df['Low']
    h_pc = (df['High'] - df['Close'].shift()).abs()
    l_pc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def get_ema_angle(series, lookback=1):
    """Calculates the angle of the EMA slope in degrees"""
    # Scaling factor (100) helps normalize the slope for different price ranges
    slope = (series - series.shift(lookback)) / series.shift(lookback) * 100
    angle = np.degrees(np.arctan(slope))
    return angle

# ==========================================
# 3. SIDEBAR: STRATEGY & RISK PARAMETERS
# ==========================================
with st.sidebar:
    st.header("⚙️ Strategy Config")
    ticker = st.text_input("Ticker (e.g., BTC-USD, RELIANCE.NS)", "BTC-USD")
    strat_choice = st.selectbox("Execution Mode", ["Simple Buy", "Simple Sell", "EMA Crossover"])
    
    # EMA Advanced Options
    if strat_choice == "EMA Crossover":
        ema_fast = st.number_input("Fast EMA Period", value=9)
        ema_slow = st.number_input("Slow EMA Period", value=20)
        crossover_mode = st.selectbox("Confirmation Type", 
                                    ["Simple Crossover", "Strong Candle (Points)", "Strong Candle (ATR)"])
        
        min_angle = st.slider("Minimum Crossover Angle (°)", 0, 90, 20)
        
        strong_pts = 0.0
        atr_mult = 0.0
        if crossover_mode == "Strong Candle (Points)":
            strong_pts = st.number_input("Min Candle Body (Points)", value=10.0)
        elif crossover_mode == "Strong Candle (ATR)":
            atr_mult = st.number_input("ATR Multiplier (e.g., 1.5)", value=1.5)

    st.divider()
    st.header("🛡️ Risk Management")
    sl_type = st.selectbox("Stop Loss Type", ["Signal Based", "Fixed Points", "Trailing SL"])
    sl_val = st.number_input("SL Value", value=50.0)
    tp_type = st.selectbox("Target Type", ["Signal Based", "Fixed Points"])
    tp_val = st.number_input("TP Value", value=100.0)

# ==========================================
# 4. TRADING ENGINE & SIGNAL LOGIC
# ==========================================
def process_trading():
    # Rate Limiting (1.8s delay)
    now_time = time.time()
    if now_time - st.session_state.last_api_call < 1.8:
        time.sleep(1.8 - (now_time - st.session_state.last_api_call))
    
    df = yf.download(ticker, period="1d", interval="1m", progress=False)
    st.session_state.last_api_call = time.time()
    
    if df.empty: return None, None
    
    # Clean data & calculate EMA
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_convert(IST)
    
    # Logic for EMA
    if strat_choice == "EMA Crossover":
        df['fast'] = df['Close'].ewm(span=ema_fast, adjust=False).mean()
        df['slow'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()
        df['angle'] = get_ema_angle(df['fast'])
        
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        # 1. Base Crossover
        bull_cross = (prev['fast'] <= prev['slow']) and (curr['fast'] > curr['slow'])
        bear_cross = (prev['fast'] >= prev['slow']) and (curr['fast'] < curr['slow'])
        
        # 2. Angle Filter
        angle_ok = abs(curr['angle']) >= min_angle
        
        # 3. Candle Confirmation
        candle_ok = True
        body = abs(curr['Close'] - curr['Open'])
        if crossover_mode == "Strong Candle (Points)":
            candle_ok = body >= strong_pts
        elif crossover_mode == "Strong Candle (ATR)":
            atr = calculate_atr(df).iloc[-1]
            candle_ok = body >= (atr * atr_mult)
            
        bull_signal = bull_cross and angle_ok and candle_ok
        bear_signal = bear_cross and angle_ok and candle_ok
    else:
        # Simple Buy/Sell executes immediately on start
        bull_signal = (strat_choice == "Simple Buy")
        bear_signal = (strat_choice == "Simple Sell")

    return df, (bull_signal, bear_signal)

# ==========================================
# 5. UI & EXECUTION TAB
# ==========================================
tab_live, tab_logs = st.tabs(["📺 Live Terminal", "📜 Trade Logs"])

with tab_live:
    c1, c2, c3 = st.columns([1,1,2])
    
    with c1:
        if not st.session_state.trading_active:
            if st.button("▶ START TRADING", type="primary", use_container_width=True):
                st.session_state.trading_active = True
                add_log(f"System Online: {ticker} using {strat_choice}")
                st.rerun()
        else:
            if st.button("⏹ STOP TRADING", type="secondary", use_container_width=True):
                st.session_state.trading_active = False
                st.session_state.current_position = None
                add_log("System Shutdown. Positions Liquidated.")
                st.rerun()

    if st.session_state.trading_active:
        df, (bull, bear) = process_trading()
        if df is not None:
            curr_price = df['Close'].iloc[-1]
            curr_time = df.index[-1]
            
            # --- POSITION MANAGEMENT ---
            pos = st.session_state.current_position
            
            if pos is None:
                if bull:
                    st.session_state.current_position = {
                        'type': 'LONG', 'entry': curr_price, 'time': curr_time,
                        'sl': curr_price - sl_val, 'tp': curr_price + tp_val
                    }
                    add_log(f"ENTRY LONG @ {curr_price:.2f}")
                elif bear:
                    st.session_state.current_position = {
                        'type': 'SHORT', 'entry': curr_price, 'time': curr_time,
                        'sl': curr_price + sl_val, 'tp': curr_price - tp_val
                    }
                    add_log(f"ENTRY SHORT @ {curr_price:.2f}")
            else:
                # EXIT LOGIC - THE "SIGNAL LOCK" FIX
                # Ensure we only check reversal signals on a NEW candle
                is_new_candle = curr_time > pos['time']
                exit_triggered = False
                reason = ""
                
                # Signal-Based Exit
                if (sl_type == "Signal Based" or tp_type == "Signal Based") and is_new_candle:
                    if pos['type'] == 'LONG' and bear:
                        exit_triggered, reason = True, "Strategy Reversal Signal (Bearish)"
                    if pos['type'] == 'SHORT' and bull:
                        exit_triggered, reason = True, "Strategy Reversal Signal (Bullish)"

                # Fixed SL/TP Exit
                if not exit_triggered:
                    if pos['type'] == 'LONG':
                        if curr_price <= pos['sl']: exit_triggered, reason = True, "Stop Loss Hit"
                        if curr_price >= pos['tp']: exit_triggered, reason = True, "Target Hit"
                    else:
                        if curr_price >= pos['sl']: exit_triggered, reason = True, "Stop Loss Hit"
                        if curr_price <= pos['tp']: exit_triggered, reason = True, "Target Hit"

                if exit_triggered:
                    pnl = curr_price - pos['entry'] if pos['type'] == 'LONG' else pos['entry'] - curr_price
                    st.session_state.trade_history.append({'pnl': pnl, 'reason': reason})
                    st.session_state.current_position = None
                    add_log(f"CLOSED @ {curr_price:.2f} | Reason: {reason} | P&L: {pnl:.2f}")

            # --- DISPLAY DASHBOARD ---
            with c2:
                st.metric("LTP", f"{curr_price:.2f}")
                st.write(f"Refreshes: {st.session_state.iteration_count}")
            
            with c3:
                if st.session_state.current_position:
                    p = st.session_state.current_position
                    pnl = curr_price - p['entry'] if p['type'] == 'LONG' else p['entry'] - curr_price
                    st.success(f"ACTIVE {p['type']} | P&L: {pnl:.2f}")
                    st.write(f"**Mentor:** Trade is breathing. Signal-based exit is locked until a fresh reversal candle forms.")
                else:
                    st.info("Scanning for next high-probability setup...")

            # --- CHARTING ---
            
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
            if strat_choice == "EMA Crossover":
                fig.add_trace(go.Scatter(x=df.index, y=df['fast'], name="Fast EMA", line=dict(color='yellow')))
                fig.add_trace(go.Scatter(x=df.index, y=df['slow'], name="Slow EMA", line=dict(color='cyan')))
            st.plotly_chart(fig, use_container_width=True)

with tab_logs:
    st.text_area("Live Events", value="\n".join(st.session_state.trade_log), height=500)

# ==========================================
# 6. AUTO-REFRESH TRIGGER
# ==========================================
if st.session_state.trading_active:
    st.session_state.iteration_count += 1
    time.sleep(1) # Extra buffer for app stability
    st.rerun()
