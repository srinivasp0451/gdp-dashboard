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
    })

def add_log(msg):
    t = datetime.now(IST).strftime("%H:%M:%S")
    st.session_state.trade_log.insert(0, f"[{t}] {msg}")

# --- TECHNICAL CALCULATIONS ---
def get_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def get_atr(df, p=14):
    tr = pd.concat([
        df['High'] - df['Low'], 
        (df['High'] - df['Close'].shift()).abs(), 
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()

# --- ADVANCED EMA ENGINE ---
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
        
        # Calculate Slope Angle of the Fast EMA
        # We use a price-normalized slope to calculate degrees
        price_range = df['Close'].max() - df['Close'].min()
        slope_val = (df['fast'] - df['fast'].shift(1)) / (price_range if price_range != 0 else 1)
        df['angle'] = np.degrees(np.arctan(slope_val * 100)) # Scale for sensitivity
        
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        # 1. Check Crossover
        bull_cross = (prev['fast'] <= prev['slow']) and (curr['fast'] > curr['slow'])
        bear_cross = (prev['fast'] >= prev['slow']) and (curr['fast'] < curr['slow'])
        
        # 2. Check Angle Requirement
        angle_ok = abs(curr['angle']) >= self.min_angle
        
        # 3. Check Candle Strength
        candle_ok = True
        body = abs(curr['Close'] - curr['Open'])
        
        if self.mode == "strong_candle_pts":
            candle_ok = body >= self.candle_pts
        elif self.mode == "strong_candle_atr":
            atr = get_atr(df).iloc[-1]
            candle_ok = body >= (atr * self.atr_mult)

        bull = bull_cross and angle_ok and candle_ok
        bear = bear_cross and angle_ok and candle_ok
        
        return bull, bear, {"fast": curr['fast'], "slow": curr['slow'], "angle": curr['angle']}

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚡ Entry Parameters")
    ticker = st.text_input("Ticker", "BTC-USD")
    strat_type = st.selectbox("Execution Mode", ["Simple Buy", "Simple Sell", "EMA Crossover"])
    
    if strat_type == "EMA Crossover":
        c1, c2 = st.columns(2)
        p1 = c1.number_input("Fast Period", value=9)
        p2 = c2.number_input("Slow Period", value=20)
        
        ema_mode = st.selectbox("Crossover Confirmation", [
            "simple", 
            "strong_candle_pts", 
            "strong_candle_atr"
        ])
        
        min_ang = st.slider("Min Crossover Angle (Degrees)", 0, 90, 20)
        
        pts = 0.0
        atr_m = 0.0
        if ema_mode == "strong_candle_pts":
            pts = st.number_input("Min Candle Body Points", value=10.0)
        elif ema_mode == "strong_candle_atr":
            atr_m = st.number_input("ATR Multiplier (e.g. 1.5)", value=1.5)
            
        engine = AdvancedEMA(p1, p2, ema_mode, min_ang, pts, atr_m)
    
    st.divider()
    st.header("🛡️ Exit Strategy")
    sl_type = st.selectbox("Stop Loss", ["Signal Based", "Fixed Points"])
    sl_val = st.number_input("SL Points", value=50.0)
    tp_type = st.selectbox("Target", ["Signal Based", "Fixed Points"])
    tp_val = st.number_input("TP Points", value=100.0)

# --- TRADING INTERFACE ---
tab_live, tab_log = st.tabs(["📊 Live Terminal", "📜 Activity Log"])

with tab_live:
    # Action Buttons
    if not st.session_state.trading_active:
        if st.button("▶ START TRADING SESSION", type="primary", use_container_width=True):
            st.session_state.trading_active = True
            st.rerun()
    else:
        if st.button("⏹ STOP & EXIT POSITION", type="secondary", use_container_width=True):
            st.session_state.trading_active = False
            st.session_state.current_position = None
            st.rerun()

    # Data Fetching
    df_raw = yf.download(ticker, period="1d", interval="1m", progress=False)
    if not df_raw.empty:
        df_raw.index = df_raw.index.tz_convert(IST)
        curr_price = df_raw['Close'].iloc[-1]
        curr_time = df_raw.index[-1]

        # Determine Signals
        if strat_type == "EMA Crossover":
            bull, bear, meta = engine.analyze(df_raw)
        else:
            bull = (strat_type == "Simple Buy")
            bear = (strat_type == "Simple Sell")
            meta = {}

        # LIVE EXECUTION LOGIC
        if st.session_state.trading_active:
            pos = st.session_state.current_position
            
            # --- ENTRY ---
            if pos is None:
                if bull:
                    st.session_state.current_position = {
                        'type': 'LONG', 'entry': curr_price, 'time': curr_time, 
                        'sl': curr_price - sl_val, 'tp': curr_price + tp_val
                    }
                    add_log(f"LONG ENTRY @ {curr_price} (Angle: {meta.get('angle',0):.1f}°)")
                elif bear:
                    st.session_state.current_position = {
                        'type': 'SHORT', 'entry': curr_price, 'time': curr_time,
                        'sl': curr_price + sl_val, 'tp': curr_price - tp_val
                    }
                    add_log(f"SHORT ENTRY @ {curr_price} (Angle: {meta.get('angle',0):.1f}°)")

            # --- EXIT ---
            else:
                is_new_candle = curr_time > pos['time']
                pnl = (curr_price - pos['entry']) if pos['type'] == 'LONG' else (pos['entry'] - curr_price)
                exit_triggered = False
                exit_reason = ""

                # 1. Fixed Point Exit
                if sl_type == "Fixed Points":
                    if (pos['type'] == 'LONG' and curr_price <= pos['sl']) or (pos['type'] == 'SHORT' and curr_price >= pos['sl']):
                        exit_triggered, exit_reason = True, "Stop Loss Hit"
                
                if tp_type == "Fixed Points":
                    if (pos['type'] == 'LONG' and curr_price >= pos['tp']) or (pos['type'] == 'SHORT' and curr_price <= pos['tp']):
                        exit_triggered, exit_reason = True, "Target Hit"

                # 2. Signal Reversal Exit (Locked to New Candle)
                if (sl_type == "Signal Based" or tp_type == "Signal Based") and is_new_candle:
                    if strat_type == "EMA Crossover":
                        if pos['type'] == 'LONG' and bear:
                            exit_triggered, exit_reason = True, "Signal Reversal (Bearish Cross)"
                        elif pos['type'] == 'SHORT' and bull:
                            exit_triggered, exit_reason = True, "Signal Reversal (Bullish Cross)"

                if exit_triggered:
                    st.session_state.trade_history.append({'pnl': pnl, 'reason': exit_reason})
                    st.session_state.current_position = None
                    add_log(f"EXIT @ {curr_price} | {exit_reason} | P&L: {pnl:.2f}")

        # DASHBOARD METRICS
        c1, c2, c3 = st.columns(3)
        c1.metric("Market Price", f"{curr_price:.2f}")
        
        if st.session_state.current_position:
            p = st.session_state.current_position
            pnl = (curr_price - p['entry']) if p['type'] == 'LONG' else (p['entry'] - curr_price)
            c2.metric("P&L (Points)", f"{pnl:.2f}", delta=f"{(pnl/p['entry'])*100:.2f}%")
            
            # Mentor Friend Advice
            status_txt = "Momentum is holding." if abs(meta.get('angle', 0)) > 15 else "Price is flattening."
            c3.info(f"💡 **Mentor:** You are in a {p['type']} position. {status_txt} Reversal exit is active for next candle.")

        # VISUAL CHART
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_raw.index, open=df_raw['Open'], high=df_raw['High'], 
            low=df_raw['Low'], close=df_raw['Close'], name="Candles"
        )])
        
        if strat_type == "EMA Crossover":
            fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'].ewm(span=p1).mean(), line=dict(color='cyan', width=1.5), name="Fast EMA"))
            fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'].ewm(span=p2).mean(), line=dict(color='orange', width=1.5), name="Slow EMA"))

        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_log:
    st.text_area("Live Trading Logs", value="\n".join(st.session_state.trade_log), height=500)

# AUTO-REFRESH SCRIPT
if st.session_state.trading_active:
    time.sleep(1.8)
    st.rerun()
