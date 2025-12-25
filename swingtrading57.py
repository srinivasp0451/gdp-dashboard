import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import pytz

# --- CONFIGURATION & CONSTANTS ---
IST = pytz.timezone('Asia/Kolkata')

ALLOWED_COMBINATIONS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "1mo"],
    "15m": ["1mo"],
    "30m": ["1mo"],
    "1h": ["1mo"],
    "1d": ["1mo", "1y", "2y", "5y"]
}

# --- INDICATOR ENGINE (MANUAL IMPLEMENTATION) ---

def get_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def get_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def get_angle(ema_series, lookback=3):
    """Calculates degrees of slope based on price change."""
    change = ema_series.diff(lookback)
    angle = np.arctan(change / lookback) * (180 / np.pi)
    return angle

# --- DATA HANDLER ---

def fetch_safe_data(ticker, interval, period):
    # Randomized delay to prevent YFinance Rate Limiting
    time.sleep(np.random.uniform(1.0, 1.5))
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        
        # Clean multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Timezone Normalization
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
            
        return df.ffill().dropna()
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None

# --- UI SETUP ---
st.set_page_config(layout="wide", page_title="AlgoTrade Pro")
st.markdown("""<style> .metric-box { border: 1px solid #333; padding: 10px; border-radius: 5px; background: #111; } </style>""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'live_active' not in st.session_state: st.session_state.live_active = False
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'system_logs' not in st.session_state: st.session_state.system_logs = []
if 'current_trade' not in st.session_state: st.session_state.current_trade = None

# --- SIDEBAR PARAMETERS ---
with st.sidebar:
    st.title("⚙️ Parameters")
    ticker = st.text_input("Ticker (yfinance)", "^NSEI")
    tf = st.selectbox("Interval", list(ALLOWED_COMBINATIONS.keys()), index=1)
    pr = st.selectbox("Period", ALLOWED_COMBINATIONS[tf])
    
    st.subheader("Strategy")
    ema_f = st.number_input("Fast EMA", 5, 50, 9)
    ema_s = st.number_input("Slow EMA", 10, 200, 15)
    min_angle = st.slider("Min Angle (°)", 0, 90, 15)
    
    st.subheader("Exit Logic")
    sl_type = st.selectbox("SL Type", ["Points", "ATR", "Signal-based", "Trailing"])
    tp_type = st.selectbox("Target Type", ["Points", "ATR", "Signal-based"])
    sl_val = st.number_input("SL Points / ATR Mult", 0.1, 1000.0, 20.0)
    tp_val = st.number_input("TP Points / ATR Mult", 0.1, 5000.0, 40.0)

# --- LOGIC FUNCTIONS ---

def process_signals(df):
    df['EMA_F'] = get_ema(df['Close'], ema_f)
    df['EMA_S'] = get_ema(df['Close'], ema_s)
    df['Angle'] = get_angle(df['EMA_F'])
    df['ATR'] = get_atr(df)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Buy: Cross over + Angle
    buy_sig = (prev['EMA_F'] <= prev['EMA_S']) and (curr['EMA_F'] > curr['EMA_S']) and (curr['Angle'] >= min_angle)
    # Sell: Cross under + Angle
    sell_sig = (prev['EMA_F'] >= prev['EMA_S']) and (curr['EMA_F'] < curr['EMA_S']) and (curr['Angle'] <= -min_angle)
    
    return curr, buy_sig, sell_sig

# --- MAIN UI ---
tab1, tab2, tab3 = st.tabs(["📺 Live Dashboard", "📊 Trade History", "📜 Logs"])

with tab1:
    c1, c2 = st.columns([3, 1])
    
    with c2:
        if not st.session_state.live_active:
            if st.button("▶️ START LIVE", use_container_width=True):
                st.session_state.live_active = True
                st.rerun()
        else:
            if st.button("🛑 STOP LIVE", use_container_width=True, type="primary"):
                st.session_state.live_active = False
                st.rerun()
        
        st.markdown("### Market Status")
        status_placeholder = st.empty()

    with c1:
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()

# --- LIVE EXECUTION LOOP ---
if st.session_state.live_active:
    df = fetch_safe_data(ticker, tf, pr)
    
    if df is not None:
        curr_row, buy_trigger, sell_trigger = process_signals(df)
        ltp = curr_row['Close']
        
        # 1. HANDLE CURRENT OPEN TRADE
        if st.session_state.current_trade:
            trade = st.session_state.current_trade
            exit_needed = False
            reason = ""
            
            # PnL Calculation
            pnl = (ltp - trade['entry']) if trade['type'] == "BUY" else (trade['entry'] - ltp)
            
            # Trailing Logic
            if sl_type == "Trailing":
                if trade['type'] == "BUY":
                    new_sl = ltp - sl_val
                    if new_sl > trade['sl']: trade['sl'] = new_sl
                else:
                    new_sl = ltp + sl_val
                    if new_sl < trade['sl']: trade['sl'] = new_sl

            # Check SL
            if (trade['type'] == "BUY" and ltp <= trade['sl']) or (trade['type'] == "SELL" and ltp >= trade['sl']):
                exit_needed, reason = True, "Stop Loss"
            # Check TP
            elif (trade['type'] == "BUY" and ltp >= trade['tp']) or (trade['type'] == "SELL" and ltp <= trade['tp']):
                exit_needed, reason = True, "Target Hit"
            # Check Signal-based Exit (Ensures at least 1 candle passed)
            elif (sl_type == "Signal-based" or tp_type == "Signal-based"):
                if (trade['type'] == "BUY" and curr_row['EMA_F'] < curr_row['EMA_S']) or \
                   (trade['type'] == "SELL" and curr_row['EMA_F'] > curr_row['EMA_S']):
                    exit_needed, reason = True, "Signal Reversal"

            if exit_needed:
                trade.update({'exit': ltp, 'exit_time': df.index[-1], 'reason': reason, 'pnl': pnl})
                st.session_state.trade_history.append(trade)
                st.session_state.system_logs.append(f"CLOSED {trade['type']} @ {ltp} | {reason}")
                st.session_state.current_trade = None

        # 2. HANDLE NEW ENTRIES
        elif buy_trigger or sell_trigger:
            side = "BUY" if buy_trigger else "SELL"
            # Calculate SL/TP
            dist = (curr_row['ATR'] * sl_val) if sl_type == "ATR" else sl_val
            tp_dist = (curr_row['ATR'] * tp_val) if tp_type == "ATR" else tp_val
            
            sl = (ltp - dist) if side == "BUY" else (ltp + dist)
            tp = (ltp + tp_dist) if side == "BUY" else (ltp - tp_dist)
            
            st.session_state.current_trade = {
                'type': side, 'entry': ltp, 'entry_time': df.index[-1],
                'sl': sl, 'tp': tp, 'index': len(df)
            }
            st.session_state.system_logs.append(f"OPENED {side} @ {ltp} | Angle: {curr_row['Angle']:.2f}")

        # 3. UPDATE UI COMPONENTS
        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LTP", f"{ltp:.2f}")
            m2.metric("EMA Fast/Slow", f"{curr_row['EMA_F']:.1f} / {curr_row['EMA_S']:.1f}")
            m3.metric("EMA Angle", f"{curr_row['Angle']:.2f}°")
            
            if st.session_state.current_trade:
                t = st.session_state.current_trade
                dist_sl = ltp - t['sl'] if t['type']=="BUY" else t['sl'] - ltp
                m4.metric("Live PnL", f"{pnl:.2f}", delta=f"SL Dist: {dist_sl:.1f}")
                st.info(f"⚡ ACTIVE {t['type']}: Entry {t['entry']} | SL: {t['sl']:.2f} | TP: {t['tp']:.2f}")
            else:
                m4.metric("Status", "WAITING")

        with chart_placeholder:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_F'], line=dict(color='yellow', width=1.5), name="Fast EMA"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_S'], line=dict(color='blue', width=1.5), name="Slow EMA"))
            fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{len(df)}")

    time.sleep(1.2)
    st.rerun()

# --- TABLES & LOGS ---
with tab2:
    if st.session_state.trade_history:
        st.table(pd.DataFrame(st.session_state.trade_history))
    else:
        st.info("No completed trades yet.")

with tab3:
    for log in reversed(st.session_state.system_logs):
        st.write(f"`{log}`")
