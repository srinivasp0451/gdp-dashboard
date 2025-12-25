import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import pytz

# --- CORE CONFIG ---
IST = pytz.timezone('Asia/Kolkata')
if 'trading_active' not in st.session_state:
    st.session_state.update({
        'trading_active': False,
        'current_position': None,
        'trade_history': [],
        'trade_log': [],
        'iteration_count': 0,
        'last_api_call': 0
    })

def add_log(msg):
    t = datetime.now(IST).strftime("%H:%M:%S")
    st.session_state.trade_log.insert(0, f"[{t}] {msg}")

# --- MATH CORE ---
def get_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def get_ema_angle(series):
    # Normalized slope calculation for consistent angle detection across assets
    slope = (series.diff() / series.shift(1)) * 100
    return np.degrees(np.arctan(slope))

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("📊 Market & Timeframe")
    ticker = st.text_input("Symbol", "BTC-USD")
    
    # Timeframe & Period Logic
    tf_options = ["1m", "5m", "15m", "30m", "1h", "1d"]
    selected_tf = st.selectbox("Timeframe", tf_options, index=0)
    
    # Auto-adjust available periods based on TF to prevent yfinance errors
    period_map = {"1m": ["1d", "5d"], "5m": ["1d", "5d", "1mo"], "15m": ["1d", "5d", "1mo"], 
                  "1h": ["1d", "5d", "1mo", "1y"], "1d": ["1mo", "1y", "5y", "max"]}
    selected_period = st.selectbox("Period", period_map.get(selected_tf, ["1d"]))

    st.divider()
    st.header("🚀 Strategy")
    strat_choice = st.selectbox("Mode", ["Simple Buy", "Simple Sell", "EMA Crossover"])
    
    if strat_choice == "EMA Crossover":
        e1_p = st.number_input("EMA 1 (Fast)", value=9)
        e2_p = st.number_input("EMA 2 (Slow)", value=20)
        min_angle = st.slider("Min Entry Angle (°)", 0, 90, 20)
        confirm_type = st.selectbox("Confirmation", ["Simple", "Points", "ATR"])
        conf_val = st.number_input("Value (Pts or ATR Mult)", value=1.0)

    st.divider()
    st.header("🛡️ Risk & Exit")
    sl_type = st.selectbox("Stop Loss", ["Signal Based", "Fixed Points", "Trailing"])
    sl_val = st.number_input("SL Value", value=50.0)
    tp_type = st.selectbox("Target", ["Signal Based", "Fixed Points"])
    tp_val = st.number_input("TP Value", value=100.0)

# --- TRADING LOGIC ---
tab_live, tab_hist, tab_logs = st.tabs(["📺 Live Terminal", "📊 History", "📜 Logs"])

with tab_live:
    # 1. Start/Stop Controls
    col_ctrl, col_status = st.columns([1, 3])
    with col_ctrl:
        if not st.session_state.trading_active:
            if st.button("▶ START TRADING", type="primary", use_container_width=True):
                st.session_state.trading_active = True
                st.rerun()
        else:
            if st.button("⏹ STOP TRADING", type="secondary", use_container_width=True):
                st.session_state.trading_active = False
                st.session_state.current_position = None
                st.rerun()

    # 2. Data Fetch & Signal Calculation
    if st.session_state.trading_active:
        # Rate Limiting
        time_since_last = time.time() - st.session_state.last_api_call
        if time_since_last < 1.8: time.sleep(1.8 - time_since_last)
        
        df = yf.download(ticker, period=selected_period, interval=selected_tf, progress=False)
        st.session_state.last_api_call = time.time()
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_convert(IST)
            
            curr_p = df['Close'].iloc[-1]
            curr_t = df.index[-1]
            
            # Indicator Processing
            bull, bear = False, False
            cur_e1, cur_e2, cur_angle = 0, 0, 0
            
            if strat_choice == "EMA Crossover":
                df['e1'] = get_ema(df['Close'], e1_p)
                df['e2'] = get_ema(df['Close'], e2_p)
                df['ang'] = get_ema_angle(df['e1'])
                
                cur_e1, cur_e2, cur_angle = df['e1'].iloc[-1], df['e2'].iloc[-1], df['ang'].iloc[-1]
                prev_e1, prev_e2 = df['e1'].iloc[-2], df['e2'].iloc[-2]
                
                # Crossover + Angle Logic
                cross_up = (prev_e1 <= prev_e2) and (cur_e1 > cur_e2)
                cross_down = (prev_e1 >= prev_e2) and (cur_e1 < cur_e2)
                angle_ok = abs(cur_angle) >= min_angle
                
                bull = cross_up and angle_ok
                bear = cross_down and angle_ok
            else:
                bull = (strat_choice == "Simple Buy")
                bear = (strat_choice == "Simple Sell")

            # 3. Execution Engine
            pos = st.session_state.current_position
            if pos is None:
                if bull:
                    st.session_state.current_position = {'type': 'LONG', 'entry': curr_p, 'time': curr_t, 'sl': curr_p - sl_val, 'tp': curr_p + tp_val}
                    add_log(f"LONG ENTRY @ {curr_p}")
                elif bear:
                    st.session_state.current_position = {'type': 'SHORT', 'entry': curr_p, 'time': curr_t, 'sl': curr_p + sl_val, 'tp': curr_p - tp_val}
                    add_log(f"SHORT ENTRY @ {curr_p}")
            else:
                # Signal-Based Exit Lock (Reversal only on NEW candle)
                is_new_candle = curr_t > pos['time']
                exit_now, reason = False, ""
                
                if (sl_type == "Signal Based" or tp_type == "Signal Based") and is_new_candle:
                    if pos['type'] == 'LONG' and bear: exit_now, reason = True, "Reversal Signal"
                    if pos['type'] == 'SHORT' and bull: exit_now, reason = True, "Reversal Signal"
                
                # Fixed SL/TP check
                if not exit_now:
                    if pos['type'] == 'LONG':
                        if curr_p <= pos['sl']: exit_now, reason = True, "SL Hit"
                        if curr_p >= pos['tp']: exit_now, reason = True, "Target Hit"
                    else:
                        if curr_p >= pos['sl']: exit_now, reason = True, "SL Hit"
                        if curr_p <= pos['tp']: exit_now, reason = True, "Target Hit"

                if exit_now:
                    st.session_state.trade_history.append({'pnl': curr_p - pos['entry'] if pos['type'] == 'LONG' else pos['entry'] - curr_p, 'reason': reason})
                    st.session_state.current_position = None
                    add_log(f"EXIT @ {curr_p} | {reason}")

            # 4. LIVE DASHBOARD (METRICS)
            st.divider()
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("LTP", f"{curr_p:.2f}")
            m2.metric("TF / Period", f"{selected_tf} / {selected_period}")
            
            if strat_choice == "EMA Crossover":
                m3.metric("EMA 1 / 2", f"{cur_e1:.1f} / {cur_e2:.1f}")
                m4.metric("Current Angle", f"{cur_angle:.1f}°", delta=f"{cur_angle - min_angle:.1f}°", delta_color="normal")
            
            if st.session_state.current_position:
                p = st.session_state.current_position
                pnl = curr_p - p['entry'] if p['type'] == 'LONG' else p['entry'] - curr_p
                m5.metric("Live P&L", f"{pnl:.2f}", delta=f"SL: {p['sl']:.1f}", delta_color="inverse")
            else:
                m5.info("Idle")

            # 5. CHARTING
            
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")])
            if strat_choice == "EMA Crossover":
                fig.add_trace(go.Scatter(x=df.index, y=df['e1'], line=dict(color='yellow', width=1.5), name=f"EMA {e1_p}"))
                fig.add_trace(go.Scatter(x=df.index, y=df['e2'], line=dict(color='cyan', width=1.5), name=f"EMA {e2_p}"))
            
            if st.session_state.current_position:
                p = st.session_state.current_position
                fig.add_hline(y=p['sl'], line_dash="dash", line_color="red")
                fig.add_hline(y=p['tp'], line_dash="dash", line_color="green")
            
            fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig, use_container_width=True)

# --- LOGS & REFRESH ---
with tab_logs:
    st.text_area("Live Log Feed", value="\n".join(st.session_state.trade_log), height=400)

if st.session_state.trading_active:
    st.session_state.iteration_count += 1
    time.sleep(1)
    st.rerun()
