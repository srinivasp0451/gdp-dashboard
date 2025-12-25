import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import pytz

# --- CONSTANTS & STATE ---
IST = pytz.timezone('Asia/Kolkata')
if 'trading_active' not in st.session_state:
    st.session_state.update({
        'trading_active': False,
        'current_position': None,
        'trade_history': [],
        'trade_log': [],
        'iteration': 0,
        'last_price': 0
    })

def add_log(msg):
    t = datetime.now(IST).strftime("%H:%M:%S")
    st.session_state.trade_log.insert(0, f"[{t}] {msg}")

# --- TECHNICAL CORE ---
def get_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def get_atr(df, p=14):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

# --- REFINED STRATEGY ENGINE ---
class AdvancedEMA:
    def __init__(self, p1, p2, mode, min_angle, candle_pts=0, atr_mult=0):
        self.p1, self.p2 = p1, p2
        self.mode = mode # simple, strong_candle_pts, strong_candle_atr
        self.min_angle = min_angle
        self.candle_pts = candle_pts
        self.atr_mult = atr_mult

    def analyze(self, df):
        df = df.copy()
        df['fast'] = get_ema(df['Close'], self.p1)
        df['slow'] = get_ema(df['Close'], self.p2)
        
        # Calculate Slope Angle in Degrees
        # Note: Scaling factor (100) used to make slope meaningful on price charts
        df['slope'] = np.degrees(np.arctan((df['fast'] - df['fast'].shift(1)) / df['fast'].shift(1) * 100))
        
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        # Core Crossover
        bull_cross = (prev['fast'] <= prev['slow']) and (curr['fast'] > curr['slow'])
        bear_cross = (prev['fast'] >= prev['slow']) and (curr['fast'] < curr['slow'])
        
        # Angle Filter
        angle_ok = abs(curr['slope']) >= self.min_angle
        
        # Candle Confirmation
        candle_ok = True
        body = abs(curr['Close'] - curr['Open'])
        if self.mode == "strong_candle_pts":
            candle_ok = body >= self.candle_pts
        elif self.mode == "strong_candle_atr":
            atr = get_atr(df).iloc[-1]
            candle_ok = body >= (atr * self.atr_mult)

        bull = bull_cross and angle_ok and candle_ok
        bear = bear_cross and angle_ok and candle_ok
        
        return bull, bear, {"fast": curr['fast'], "slow": curr['slow'], "angle": curr['slope']}

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("⚡ Live Execution")
    ticker = st.text_input("Ticker (yfinance)", "BTC-USD")
    strat_type = st.selectbox("Entry Strategy", ["Simple Buy", "Simple Sell", "EMA Crossover"])
    
    if strat_type == "EMA Crossover":
        c1, c2 = st.columns(2)
        p1 = c1.number_input("Fast EMA", 9)
        p2 = c2.number_input("Slow EMA", 20)
        mode = st.selectbox("Crossover Type", ["simple", "strong_candle_pts", "strong_candle_atr"])
        angle = st.slider("Min Crossover Angle", 0, 90, 20)
        pts = st.number_input("Min Body Points", 0.0) if mode == "strong_candle_pts" else 0
        atr_m = st.number_input("ATR Multiplier", 1.5) if mode == "strong_candle_atr" else 0
        engine = AdvancedEMA(p1, p2, mode, angle, pts, atr_m)
    
    st.divider()
    st.header("🛡️ Exit Logic")
    sl_type = st.selectbox("Stop Loss", ["Signal Based", "Fixed Points", "Trailing"])
    sl_val = st.number_input("SL Points", 50.0)
    tp_type = st.selectbox("Target", ["Signal Based", "Fixed Points"])
    tp_val = st.number_input("TP Points", 100.0)

# --- MAIN TERMINAL ---
tab_trade, tab_logs = st.tabs(["Dashboard", "Activity Log"])

with tab_trade:
    # Controls
    if not st.session_state.trading_active:
        if st.button("▶ START TRADING", type="primary", use_container_width=True):
            st.session_state.trading_active = True
            st.session_state.iteration = 0
            st.rerun()
    else:
        if st.button("⏹ STOP & LIQUIDATE", type="secondary", use_container_width=True):
            st.session_state.trading_active = False
            st.session_state.current_position = None
            st.rerun()

    # Data Engine
    df_raw = yf.download(ticker, period="1d", interval="1m", progress=False)
    if not df_raw.empty:
        df_raw.index = df_raw.index.tz_convert(IST)
        curr_price = df_raw['Close'].iloc[-1]
        curr_time = df_raw.index[-1]
        
        # Get Strategy Signals
        if strat_type == "EMA Crossover":
            bull, bear, meta = engine.analyze(df_raw)
        else:
            # "Simple" strategies don't wait for a signal
            bull = (strat_type == "Simple Buy")
            bear = (strat_type == "Simple Sell")
            meta = {}

        # POSITION MANAGEMENT
        if st.session_state.trading_active:
            pos = st.session_state.current_position
            
            # ENTRY LOGIC
            if pos is None:
                if bull:
                    st.session_state.current_position = {
                        'type': 'LONG', 'entry': curr_price, 'time': curr_time, 
                        'sl': curr_price - sl_val, 'tp': curr_price + tp_val
                    }
                    add_log(f"INSTANT LONG ENTRY @ {curr_price}")
                elif bear:
                    st.session_state.current_position = {
                        'type': 'SHORT', 'entry': curr_price, 'time': curr_time,
                        'sl': curr_price + sl_val, 'tp': curr_price - tp_val
                    }
                    add_log(f"INSTANT SHORT ENTRY @ {curr_price}")

            # EXIT LOGIC (LOCKED TO NEW CANDLE)
            else:
                is_new_candle = curr_time > pos['time']
                exit_now = False
                reason = ""
                pnl = (curr_price - pos['entry']) if pos['type'] == 'LONG' else (pos['entry'] - curr_price)

                # 1. FIXED SL/TP CHECK
                if sl_type == "Fixed Points":
                    if (pos['type'] == 'LONG' and curr_price <= pos['sl']) or (pos['type'] == 'SHORT' and curr_price >= pos['sl']):
                        exit_now, reason = True, "Stop Loss Hit"
                
                # 2. SIGNAL BASED EXIT (The Fix)
                # We only exit if there is a REVERSAL signal and it's NOT the entry candle
                if (sl_type == "Signal Based" or tp_type == "Signal Based") and is_new_candle:
                    if strat_type == "EMA Crossover":
                        # Exit LONG if EMA generates a BEARISH cross
                        if pos['type'] == 'LONG' and bear:
                            exit_now, reason = True, "Strategy Reversal (Bearish Cross)"
                        if pos['type'] == 'SHORT' and bull:
                            exit_now, reason = True, "Strategy Reversal (Bullish Cross)"

                if exit_now:
                    st.session_state.trade_history.append({'pnl': pnl, 'reason': reason, 'time': curr_time})
                    st.session_state.current_position = None
                    add_log(f"CLOSED @ {curr_price} | {reason} | P&L: {pnl:.2f}")

        # VISUALS
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"{curr_price:.2f}")
        if st.session_state.current_position:
            p = st.session_state.current_position
            pnl = (curr_price - p['entry']) if p['type'] == 'LONG' else (p['entry'] - curr_price)
            m2.metric("P&L Points", f"{pnl:.2f}", delta=f"{(pnl/p['entry'])*100:.2f}%")
            m3.write(f"**Mentor:** Trade is live. Strategy is monitoring for a reversal signal on the next candle. Do not touch the mouse.")
        
        # Chart
        fig = go.Figure(data=[go.Candlestick(x=df_raw.index, open=df_raw['Open'], high=df_raw['High'], low=df_raw['Low'], close=df_raw['Close'])])
        if strat_type == "EMA Crossover":
            df_plot = engine.analyze(df_raw)[2] # Get meta for plotting
            fig.add_trace(go.Scatter(x=df_raw.index, y=get_ema(df_raw['Close'], p1), line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=df_raw.index, y=get_ema(df_raw['Close'], p2), line=dict(color='orange')))
        
        st.plotly_chart(fig, use_container_width=True)

# --- LOGS ---
with tab_logs:
    st.text_area("Live Feed", value="\n".join(st.session_state.trade_log), height=400)

# --- AUTO-REFRESH ---
if st.session_state.trading_active:
    st.session_state.iteration += 1
    time.sleep(1.8)
    st.rerun()
