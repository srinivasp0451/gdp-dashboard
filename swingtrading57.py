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
    "Commodities": ["GC=F", "SI=F", "MZN=F"]
}

ALLOWED_COMBINATIONS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "1mo"],
    "15m": ["1mo"],
    "1h": ["1mo", "1y"],
    "1d": ["1y", "5y"]
}

# --- INDICATORS ---
def get_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def get_atr(df, window=14):
    h_l = df['High'] - df['Low']
    h_c = (df['High'] - df['Close'].shift()).abs()
    l_c = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def get_angle(ema_series, lookback=3):
    change = ema_series.diff(lookback)
    angle = np.arctan(change / lookback) * (180 / np.pi)
    return angle

# --- DATA FETCHING ---
def fetch_data(ticker, interval, period):
    time.sleep(np.random.uniform(1.0, 1.3)) # Rate limit protection
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_convert(IST) if df.index.tz else df.index.tz_localize('UTC').tz_convert(IST)
        return df.ffill().dropna()
    except Exception:
        return None

# --- UI STATE INITIALIZATION ---
if 'live_active' not in st.session_state: st.session_state.live_active = False
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'logs' not in st.session_state: st.session_state.logs = []
if 'active_trade' not in st.session_state: st.session_state.active_trade = None

st.set_page_config(layout="wide", page_title="Master Quant Algo")

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("🔍 Market Selection")
    mkt_type = st.selectbox("Market Type", list(TICKER_MAP.keys()))
    ticker = st.selectbox("Ticker", TICKER_MAP[mkt_type])
    tf = st.selectbox("Interval", list(ALLOWED_COMBINATIONS.keys()), index=1)
    period = st.selectbox("Period", ALLOWED_COMBINATIONS[tf])
    
    st.divider()
    st.header("🛠 Strategy Params")
    ema_f = st.number_input("Fast EMA", 5, 50, 9)
    ema_s = st.number_input("Slow EMA", 10, 200, 15)
    min_ang = st.slider("Min Angle (°)", 0, 60, 15)
    
    st.divider()
    st.header("🛡 Risk Mgmt")
    sl_mode = st.selectbox("SL Mode", ["Points", "ATR", "Signal-based", "Trailing"])
    tp_mode = st.selectbox("TP Mode", ["Points", "ATR", "Signal-based", "Risk-Reward"])
    sl_val = st.number_input("SL Value (Pts/Mult)", 0.1, 500.0, 20.0)
    rr_ratio = st.number_input("RR Ratio (if TP=RR)", 1.0, 10.0, 2.0) if tp_mode == "Risk-Reward" else 1.0
    tp_val = st.number_input("TP Value (Pts/Mult)", 0.1, 1000.0, 40.0) if tp_mode != "Risk-Reward" else 0.0

# --- MAIN DASHBOARD LAYOUT ---
tab_live, tab_hist, tab_log = st.tabs(["📺 Live Terminal", "📊 Trade History", "📝 System Logs"])

with tab_live:
    col_main, col_stats = st.columns([3, 1])
    
    with col_stats:
        st.subheader("Control Panel")
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("START", use_container_width=True, type="primary"):
            st.session_state.live_active = True
        if btn_col2.button("STOP", use_container_width=True):
            st.session_state.live_active = False
            
        st.divider()
        st.subheader("Live Strategy Monitor")
        monitor_placeholder = st.empty()

    with col_main:
        chart_placeholder = st.empty()
        trade_details_placeholder = st.empty()

# --- EXECUTION ENGINE ---
if st.session_state.live_active:
    df = fetch_data(ticker, tf, period)
    
    if df is not None:
        # Indicator calculation
        df['EMA_F'] = get_ema(df['Close'], ema_f)
        df['EMA_S'] = get_ema(df['Close'], ema_s)
        df['Angle'] = get_angle(df['EMA_F'])
        df['ATR'] = get_atr(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        ltp = curr['Close']
        curr_angle = curr['Angle']
        
        # 1. SIGNAL DETECTION
        buy_sig = (prev['EMA_F'] <= prev['EMA_S']) and (curr['EMA_F'] > curr['EMA_S']) and (curr_angle >= min_ang)
        sell_sig = (prev['EMA_F'] >= prev['EMA_S']) and (curr['EMA_F'] < curr['EMA_S']) and (curr_angle <= -min_ang)
        
        # 2. TRADE MANAGEMENT
        if st.session_state.active_trade:
            t = st.session_state.active_trade
            pnl = (ltp - t['entry']) if t['type'] == "BUY" else (t['entry'] - ltp)
            
            # Trailing Logic
            if sl_mode == "Trailing":
                if t['type'] == "BUY":
                    new_sl = ltp - sl_val
                    if new_sl > t['sl']: t['sl'] = new_sl
                else:
                    new_sl = ltp + sl_val
                    if new_sl < t['sl']: t['sl'] = new_sl

            # Exit Conditions
            exit_flag = False
            reason = ""
            if (t['type'] == "BUY" and ltp <= t['sl']) or (t['type'] == "SELL" and ltp >= t['sl']):
                exit_flag, reason = True, "SL Hit"
            elif (t['type'] == "BUY" and ltp >= t['tp']) or (t['type'] == "SELL" and ltp <= t['tp']):
                exit_flag, reason = True, "TP Hit"
            elif (sl_mode == "Signal-based" or tp_mode == "Signal-based"):
                if (t['type'] == "BUY" and curr['EMA_F'] < curr['EMA_S']) or (t['type'] == "SELL" and curr['EMA_F'] > curr['EMA_S']):
                    exit_flag, reason = True, "Reverse Signal"

            if exit_flag:
                t.update({'exit': ltp, 'pnl': pnl, 'reason': reason, 'exit_time': df.index[-1]})
                st.session_state.trade_history.append(t)
                st.session_state.logs.append(f"CLOSED {t['type']} @ {ltp} | PnL: {pnl:.2f}")
                st.session_state.active_trade = None
        
        # 3. NEW ENTRY
        elif buy_sig or sell_sig:
            side = "BUY" if buy_sig else "SELL"
            # Risk Calc
            dist = (curr['ATR'] * sl_val) if sl_mode == "ATR" else sl_val
            sl_price = (ltp - dist) if side == "BUY" else (ltp + dist)
            
            if tp_mode == "Risk-Reward":
                tp_price = ltp + (dist * rr_ratio) if side == "BUY" else ltp - (dist * rr_ratio)
            else:
                tp_dist = (curr['ATR'] * tp_val) if tp_mode == "ATR" else tp_val
                tp_price = (ltp + tp_dist) if side == "BUY" else (ltp - tp_dist)

            st.session_state.active_trade = {
                'type': side, 'entry': ltp, 'sl': sl_price, 'tp': tp_price, 
                'time': df.index[-1], 'entry_idx': len(df)
            }
            st.session_state.logs.append(f"ENTRY {side} @ {ltp} | Angle: {curr_angle:.1f}")

        # --- UPDATE UI CONTAINERS (NO WIPE) ---
        with monitor_placeholder.container():
            st.write(f"**LTP:** `{ltp:.2f}`")
            st.write(f"**Fast EMA:** `{curr['EMA_F']:.2f}`")
            st.write(f"**Slow EMA:** `{curr['EMA_S']:.2f}`")
            st.write(f"**Crossover Angle:** `{curr_angle:.2f}°` (Req: {min_ang}°)")
            
        with trade_details_placeholder.container():
            if st.session_state.active_trade:
                tr = st.session_state.active_trade
                dist_sl = abs(ltp - tr['sl'])
                dist_tp = abs(ltp - tr['tp'])
                live_pnl = (ltp - tr['entry']) if tr['type'] == "BUY" else (tr['entry'] - ltp)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current PnL", f"{live_pnl:.2f}")
                c2.metric("Dist to SL", f"{dist_sl:.2f}")
                c3.metric("Dist to TP", f"{dist_tp:.2f}")
                st.info(f"🟢 Active {tr['type']} | Entry: {tr['entry']} | SL: {tr['sl']:.2f} | TP: {tr['tp']:.2f}")

        with chart_placeholder:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_F'], line=dict(color='cyan', width=1), name="EMA Fast"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_S'], line=dict(color='orange', width=1), name="EMA Slow"))
            if st.session_state.active_trade:
                fig.add_hline(y=st.session_state.active_trade['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                fig.add_hline(y=st.session_state.active_trade['tp'], line_dash="dash", line_color="green", annotation_text="TP")
            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{len(df)}")

    time.sleep(1.2)
    st.rerun()

# --- HISTORY & LOGS ---
with tab_hist:
    if st.session_state.trade_history:
        st.dataframe(pd.DataFrame(st.session_state.trade_history), use_container_width=True)
with tab_log:
    for l in reversed(st.session_state.logs):
        st.write(f"`{l}`")
