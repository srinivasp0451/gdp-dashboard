import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import pytz

# --- CONSTANTS & CONFIG ---
IST = pytz.timezone('Asia/Kolkata')
TICKER_MAP = {
    "Indian Indices": ["^NSEI", "^NSEBANK", "^BSESN"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "Forex": ["USDINR=X", "EURUSD=X", "GBPUSD=X"],
    "Commodities": ["GC=F", "SI=F"]
}

ALLOWED_COMBINATIONS = {
    "1m": ["1d", "5m"],
    "5m": ["1d", "1mo"],
    "15m": ["1mo"],
    "1h": ["1mo", "1y"]
}

# --- CORE MATH ---
def get_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def get_atr(df, window=14):
    h_l, h_c, l_c = df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()
    return pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(window=window).mean()

def get_angle(ema_series, lookback=3):
    return np.arctan(ema_series.diff(lookback) / lookback) * (180 / np.pi)

# --- APP STATE ---
if 'live_active' not in st.session_state: st.session_state.live_active = False
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'logs' not in st.session_state: st.session_state.logs = []
if 'active_trade' not in st.session_state: st.session_state.active_trade = None

st.set_page_config(layout="wide", page_title="QuantPro Master")

# --- 1. STATIC SIDEBAR (Always stays visible) ---
with st.sidebar:
    st.header("🔍 Market Config")
    mkt_type = st.selectbox("Market Type", list(TICKER_MAP.keys()))
    ticker = st.selectbox("Ticker", TICKER_MAP[mkt_type])
    tf = st.selectbox("Interval", list(ALLOWED_COMBINATIONS.keys()), index=1)
    period = st.selectbox("Period", ALLOWED_COMBINATIONS[tf])
    
    st.divider()
    st.header("🛠 Strategy")
    ema_f = st.number_input("Fast EMA", 5, 50, 9)
    ema_s = st.number_input("Slow EMA", 10, 200, 15)
    min_ang = st.slider("Min Angle (°)", 0, 60, 15)
    
    st.divider()
    st.header("🛡 Risk Mgmt")
    sl_mode = st.selectbox("SL Mode", ["Points", "ATR", "Trailing", "Signal-based"])
    tp_mode = st.selectbox("TP Mode", ["Points", "ATR", "Risk-Reward", "Signal-based"])
    sl_val = st.number_input("SL Value (Pts/Mult)", 0.1, 1000.0, 20.0)
    rr_ratio = st.number_input("RR Ratio", 1.0, 10.0, 2.0) if tp_mode == "Risk-Reward" else 1.0
    tp_val = st.number_input("TP Value", 0.1, 5000.0, 40.0) if tp_mode != "Risk-Reward" else 0.0

# --- 2. PERMANENT UI STRUCTURE ---
tab_live, tab_hist, tab_log = st.tabs(["📺 Live Terminal", "📊 History", "📝 Logs"])

with tab_live:
    top_col1, top_col2 = st.columns([3, 1])
    
    with top_col2:
        st.subheader("Controls")
        c1, c2 = st.columns(2)
        start_btn = c1.button("START", use_container_width=True, type="primary")
        stop_btn = c2.button("STOP", use_container_width=True)
        
        if start_btn: st.session_state.live_active = True
        if stop_btn: st.session_state.live_active = False
        
        st.divider()
        # Stable placeholders that don't move
        monitor_container = st.empty() 
        
    with top_col1:
        # Placeholder for Chart to prevent jumping
        chart_container = st.empty()
        trade_status_container = st.empty()

# --- 3. LOGIC LOOP ---
if st.session_state.live_active:
    # Safe Fetching
    time.sleep(np.random.uniform(1.0, 1.2)) 
    df = yf.download(ticker, period=period, interval=tf, progress=False)
    
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize('UTC').tz_convert(IST)
        
        # Calculate Indicators
        df['EMA_F'] = get_ema(df['Close'], ema_f)
        df['EMA_S'] = get_ema(df['Close'], ema_s)
        df['Angle'] = get_angle(df['EMA_F'])
        df['ATR'] = get_atr(df)
        
        curr, prev = df.iloc[-1], df.iloc[-2]
        ltp = float(curr['Close'])
        
        # Signal Detection
        buy_trigger = (prev['EMA_F'] <= prev['EMA_S']) and (curr['EMA_F'] > curr['EMA_S']) and (curr['Angle'] >= min_ang)
        sell_trigger = (prev['EMA_F'] >= prev['EMA_S']) and (curr['EMA_F'] < curr['EMA_S']) and (curr['Angle'] <= -min_ang)

        # Trade Management
        if st.session_state.active_trade:
            t = st.session_state.active_trade
            pnl = (ltp - t['entry']) if t['type'] == "BUY" else (t['entry'] - ltp)
            
            # Trailing SL
            if sl_mode == "Trailing":
                if t['type'] == "BUY":
                    new_sl = ltp - sl_val
                    if new_sl > t['sl']: t['sl'] = new_sl
                else:
                    new_sl = ltp + sl_val
                    if new_sl < t['sl']: t['sl'] = new_sl

            # Check Exit
            hit_sl = (t['type']=="BUY" and ltp <= t['sl']) or (t['type']=="SELL" and ltp >= t['sl'])
            hit_tp = (t['type']=="BUY" and ltp >= t['tp']) or (t['type']=="SELL" and ltp <= t['tp'])
            rev_sig = (sl_mode == "Signal-based") and ((t['type']=="BUY" and curr['EMA_F'] < curr['EMA_S']) or (t['type']=="SELL" and curr['EMA_F'] > curr['EMA_S']))

            if hit_sl or hit_tp or rev_sig:
                t.update({'exit': ltp, 'pnl': pnl, 'reason': "SL" if hit_sl else "TP" if hit_tp else "Signal", 'exit_time': df.index[-1]})
                st.session_state.trade_history.append(t)
                st.session_state.logs.append(f"CLOSED {t['type']} @ {ltp:.2f}")
                st.session_state.active_trade = None
        
        elif buy_trigger or sell_trigger:
            side = "BUY" if buy_trigger else "SELL"
            dist = (curr['ATR'] * sl_val) if sl_mode == "ATR" else sl_val
            sl_p = (ltp - dist) if side == "BUY" else (ltp + dist)
            tp_p = ltp + (dist * rr_ratio) if tp_mode == "Risk-Reward" else (ltp + (curr['ATR']*tp_val if tp_mode=="ATR" else tp_val) * (1 if side=="BUY" else -1))
            
            st.session_state.active_trade = {'type': side, 'entry': ltp, 'sl': sl_p, 'tp': tp_p, 'time': df.index[-1]}
            st.session_state.logs.append(f"ENTRY {side} @ {ltp:.2f} | Angle: {curr['Angle']:.1f}")

        # --- REFRESH PLACEHOLDERS ONLY ---
        with monitor_container.container():
            st.write(f"**LTP:** `{ltp:.2f}`")
            st.write(f"**Fast EMA:** `{curr['EMA_F']:.2f}`")
            st.write(f"**Slow EMA:** `{curr['EMA_S']:.2f}`")
            st.write(f"**Angle:** `{curr['Angle']:.2f}°`")
            if st.session_state.active_trade:
                st.success(f"ACTIVE {st.session_state.active_trade['type']}")
            else:
                st.info("SCANNING...")

        with trade_status_container.container():
            if st.session_state.active_trade:
                tr = st.session_state.active_trade
                c1, c2, c3 = st.columns(3)
                c1.metric("Live PnL", f"{pnl:.2f}")
                c2.metric("SL Distance", f"{abs(ltp - tr['sl']):.2f}")
                c3.metric("TP Distance", f"{abs(ltp - tr['tp']):.2f}")

        with chart_container:
            fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")])
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_F'], line=dict(color='cyan', width=1), name="EMA 9"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_S'], line=dict(color='orange', width=1), name="EMA 15"))
            if st.session_state.active_trade:
                fig.add_hline(y=st.session_state.active_trade['sl'], line_dash="dash", line_color="red")
                fig.add_hline(y=st.session_state.active_trade['tp'], line_dash="dash", line_color="green")
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, use_container_width=True, key=f"c_{len(df)}")

    time.sleep(1.2)
    st.rerun()

# --- 4. DATA TABS (Always visible) ---
with tab_hist:
    if st.session_state.trade_history: st.table(pd.DataFrame(st.session_state.trade_history))
with tab_log:
    for l in reversed(st.session_state.logs): st.write(f"`{l}`")
