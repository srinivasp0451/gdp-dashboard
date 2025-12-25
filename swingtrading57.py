import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import pytz

# --- INITIALIZATION & IST SETUP ---
IST = pytz.timezone('Asia/Kolkata')
if 'trading_active' not in st.session_state:
    st.session_state.update({
        'trading_active': False,
        'current_position': None,
        'trade_history': [],
        'trade_log': [],
        'iteration': 0
    })

def add_log(msg):
    t = datetime.now(IST).strftime("%H:%M:%S")
    st.session_state.trade_log.insert(0, f"[{t}] {msg}")

# --- MATHEMATICAL ENGINE (MANUAL) ---
def get_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def get_atr(df, p=14):
    tr = pd.concat([
        df['High'] - df['Low'], 
        (df['High'] - df['Close'].shift()).abs(), 
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()

# --- ADVANCED EMA STRATEGY CLASS ---
class AdvancedEMA:
    def __init__(self, p1, p2, mode, min_angle, candle_pts=0, atr_mult=0):
        self.p1, self.p2 = p1, p2
        self.mode = mode  # simple, strong_candle_pts, strong_candle_atr
        self.min_angle = min_angle
        self.candle_pts = candle_pts
        self.atr_mult = atr_mult

    def analyze(self, df):
        df = df.copy()
        df['fast'] = get_ema(df['Close'], self.p1)
        df['slow'] = get_ema(df['Close'], self.p2)
        
        # Angle Calculation: Slope of Fast EMA converted to degrees
        # (Change in price / Previous Price) * 100 provides a scaled slope
        slope_raw = (df['fast'] - df['fast'].shift(1)) / df['fast'].shift(1) * 100
        df['angle'] = np.degrees(np.arctan(slope_raw))
        
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        # 1. Base Crossover Check
        bull_cross = (prev['fast'] <= prev['slow']) and (curr['fast'] > curr['slow'])
        bear_cross = (prev['fast'] >= prev['slow']) and (curr['fast'] < curr['slow'])
        
        # 2. Angle Filter (Ensures trend strength)
        angle_ok = abs(curr['angle']) >= self.min_angle
        
        # 3. Candle Confirmation (Momentum)
        candle_ok = True
        body_size = abs(curr['Close'] - curr['Open'])
        if self.mode == "strong_candle_pts":
            candle_ok = body_size >= self.candle_pts
        elif self.mode == "strong_candle_atr":
            atr = get_atr(df).iloc[-1]
            candle_ok = body_size >= (atr * self.atr_mult)

        bull_signal = bull_cross and angle_ok and candle_ok
        bear_signal = bear_cross and angle_ok and candle_ok
        
        return bull_signal, bear_signal, {
            "fast": curr['fast'], 
            "slow": curr['slow'], 
            "angle": curr['angle'],
            "body": body_size
        }

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚡ Live Strategy Config")
    ticker = st.text_input("Ticker Symbol", "BTC-USD")
    strat_type = st.selectbox("Entry Mode", ["Simple Buy", "Simple Sell", "EMA Crossover"])
    
    if strat_type == "EMA Crossover":
        p1 = st.number_input("Fast EMA Period", 9)
        p2 = st.number_input("Slow EMA Period", 20)
        ema_mode = st.selectbox("Crossover Confirmation", 
                                ["simple", "strong_candle_pts", "strong_candle_atr"])
        c_angle = st.slider("Min Slope Angle (Degrees)", 0, 90, 20)
        
        c_pts = 0.0
        c_atr = 0.0
        if ema_mode == "strong_candle_pts":
            c_pts = st.number_input("Min Candle Body (Points)", value=10.0)
        elif ema_mode == "strong_candle_atr":
            c_atr = st.number_input("ATR Multiplier", value=1.5)
            
        engine = AdvancedEMA(p1, p2, ema_mode, c_angle, c_pts, c_atr)
    
    st.divider()
    st.header("🛡️ Risk & Exit")
    sl_type = st.selectbox("Stop Loss Type", ["Signal Based", "Fixed Points"])
    sl_val = st.number_input("SL Points", 50.0)
    tp_type = st.selectbox("Target Type", ["Signal Based", "Fixed Points"])
    tp_val = st.number_input("TP Points", 100.0)

# --- MAIN DASHBOARD ---
tab_live, tab_logs = st.tabs(["Live Terminal", "Activity Log"])

with tab_live:
    # Action Buttons
    if not st.session_state.trading_active:
        if st.button("▶ START TRADING SESSION", type="primary", use_container_width=True):
            st.session_state.trading_active = True
            st.session_state.iteration = 0
            add_log(f"System Online: {ticker} | {strat_type}")
            st.rerun()
    else:
        if st.button("⏹ EMERGENCY STOP", type="secondary", use_container_width=True):
            st.session_state.trading_active = False
            st.session_state.current_position = None
            add_log("System Offline: All monitoring stopped.")
            st.rerun()

    # Data Acquisition
    df_raw = yf.download(ticker, period="1d", interval="1m", progress=False)
    
    if not df_raw.empty:
        # Timezone Handling
        if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw.index = df_raw.index.tz_convert(IST)
        
        curr_price = df_raw['Close'].iloc[-1]
        curr_time = df_raw.index[-1]
        
        # Signal Generation
        if strat_type == "EMA Crossover":
            bull, bear, meta = engine.analyze(df_raw)
        else:
            # Simple Buy/Sell triggers immediately on session start
            bull = (strat_type == "Simple Buy")
            bear = (strat_type == "Simple Sell")
            meta = {"angle": 0, "body": 0}

        # --- LIVE TRADING ENGINE ---
        if st.session_state.trading_active:
            pos = st.session_state.current_position
            
            # 1. ENTRY LOGIC
            if pos is None:
                if bull:
                    st.session_state.current_position = {
                        'type': 'LONG', 'entry': curr_price, 'time': curr_time, 
                        'sl': curr_price - sl_val, 'tp': curr_price + tp_val
                    }
                    add_log(f"ENTRY LONG @ {curr_price}")
                elif bear:
                    st.session_state.current_position = {
                        'type': 'SHORT', 'entry': curr_price, 'time': curr_time,
                        'sl': curr_price + sl_val, 'tp': curr_price - tp_val
                    }
                    add_log(f"ENTRY SHORT @ {curr_price}")

            # 2. EXIT LOGIC (LOCKED TO NEW CANDLES)
            else:
                is_new_candle = curr_time > pos['time']
                exit_now = False
                reason = ""
                pnl = (curr_price - pos['entry']) if pos['type'] == 'LONG' else (pos['entry'] - curr_price)

                # Fixed Point Check
                if sl_type == "Fixed Points":
                    if (pos['type'] == 'LONG' and curr_price <= pos['sl']) or \
                       (pos['type'] == 'SHORT' and curr_price >= pos['sl']):
                        exit_now, reason = True, "Stop Loss Triggered"
                
                # Signal Based Check (Requires New Candle & Strategy Reversal)
                if (sl_type == "Signal Based" or tp_type == "Signal Based") and is_new_candle:
                    # In EMA mode, we exit if the opposite signal appears
                    if strat_type == "EMA Crossover":
                        if pos['type'] == 'LONG' and bear:
                            exit_now, reason = True, "Strategy Reversal (Exit Long)"
                        elif pos['type'] == 'SHORT' and bull:
                            exit_now, reason = True, "Strategy Reversal (Exit Short)"

                if exit_now:
                    st.session_state.trade_history.append({'pnl': pnl, 'reason': reason, 'time': curr_time})
                    st.session_state.current_position = None
                    add_log(f"TRADE CLOSED @ {curr_price} | {reason} | P&L: {pnl:.2f}")

        # --- DASHBOARD UI ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Market Price", f"{curr_price:.2f}")
        
        if st.session_state.current_position:
            p = st.session_state.current_position
            pnl = (curr_price - p['entry']) if p['type'] == 'LONG' else (p['entry'] - curr_price)
            col2.metric("Open P&L", f"{pnl:.2f}", delta=f"{(pnl/p['entry'])*100:.2f}%")
            col3.info(f"💡 **Mentor:** Holding {p['type']}. Angle is {meta['angle']:.1f}°. Monitoring for reversal on new candles.")
        else:
            col2.write("🔭 Scanning for signals...")
            col3.write(f"Refreshes: {st.session_state.iteration}")

        # Charting with Indicators
        fig = go.Figure(data=[go.Candlestick(x=df_raw.index, open=df_raw['Open'], 
                        high=df_raw['High'], low=df_raw['Low'], close=df_raw['Close'], name="Price")])
        
        if strat_type == "EMA Crossover":
            fig.add_trace(go.Scatter(x=df_raw.index, y=get_ema(df_raw['Close'], p1), line=dict(color='cyan', width=1.5), name="Fast EMA"))
            fig.add_trace(go.Scatter(x=df_raw.index, y=get_ema(df_raw['Close'], p2), line=dict(color='orange', width=1.5), name="Slow EMA"))

        
        
        fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

with tab_logs:
    st.text_area("Live Feed", value="\n".join(st.session_state.trade_log), height=500)

# --- AUTO-REFRESH TRIGGER ---
if st.session_state.trading_active:
    st.session_state.iteration += 1
    time.sleep(1.8)  # Rate limiting
    st.rerun()
