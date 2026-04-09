import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import datetime
import socket

# --- PAGE CONFIG & STATE ---
st.set_page_config(page_title="Smart Investing", layout="wide")

# Persistent state to prevent flickering and maintain trades
if 'live_running' not in st.session_state: st.session_state.live_running = False
if 'trades' not in st.session_state: st.session_state.trades = []
if 'current_position' not in st.session_state: st.session_state.current_position = None
if 'last_fetch_time' not in st.session_state: st.session_state.last_fetch_time = 0

# --- MANUAL INDICATORS (Match TradingView) ---
def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# --- DATA ENGINE ---
def fetch_data(ticker, interval, period):
    now = time.time()
    # Handle yfinance rate limits (1.5s delay)
    if now - st.session_state.last_fetch_time < 1.5:
        time.sleep(1.5 - (now - st.session_state.last_fetch_time))
    
    try:
        # Fetch extra data for EMA warmup to match TV
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        st.session_state.last_fetch_time = time.time()
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        st.error(f"Fetch Error: {e}")
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.title("🚀 Smart Investing")
ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '1d'], index=1)
period = st.sidebar.selectbox("Period", ['5d', '1mo', '3mo', '1y', '2y', '5y'], index=1)

st.sidebar.subheader("Strategy Params")
strategy_mode = st.sidebar.selectbox("Mode", ["EMA Crossover", "Simple Buy", "Simple Sell"])
fast_ema_val = st.sidebar.number_input("Fast EMA", value=9)
slow_ema_val = st.sidebar.number_input("Slow EMA", value=15)

st.sidebar.subheader("Risk Management")
sl_val = st.sidebar.number_input("SL Points", value=10.0)
tgt_val = st.sidebar.number_input("Target Points", value=20.0)
qty = st.sidebar.number_input("Quantity", value=1)

# Broker Settings
dhan_enabled = st.sidebar.checkbox("Enable Dhan Broker")
options_enabled = st.sidebar.checkbox("Options Trading (Buying)")

# --- LOGIC: BACKTESTING ENGINE ---
def run_backtest(df, f_ema, s_ema, sl, tgt, q):
    df = df.copy()
    df['EMA_F'] = calc_ema(df['Close'], f_ema)
    df['EMA_S'] = calc_ema(df['Close'], s_ema)
    
    # Signal on Candle N
    df['Buy_Sig'] = (df['EMA_F'].shift(1) <= df['EMA_S'].shift(1)) & (df['EMA_F'] > df['EMA_S'])
    df['Sell_Sig'] = (df['EMA_F'].shift(1) >= df['EMA_S'].shift(1)) & (df['EMA_F'] < df['EMA_S'])
    
    trades = []
    in_pos = False
    pos = {}
    violations = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]

        if in_pos:
            exit_px = 0
            # Conservative Logic: Check SL first
            if pos['type'] == 'Buy':
                if curr['Low'] <= pos['sl'] and curr['High'] >= pos['tgt']: violations += 1
                if curr['Low'] <= pos['sl']: exit_px, reason = pos['sl'], "SL Hit"
                elif curr['High'] >= pos['tgt']: exit_px, reason = pos['tgt'], "Target Hit"
            else: # Sell
                if curr['High'] >= pos['sl'] and curr['Low'] <= pos['tgt']: violations += 1
                if curr['High'] >= pos['sl']: exit_px, reason = pos['sl'], "SL Hit"
                elif curr['Low'] <= pos['tgt']: exit_px, reason = pos['tgt'], "Target Hit"
            
            if exit_px != 0:
                pnl = (exit_px - pos['entry']) if pos['type'] == 'Buy' else (pos['entry'] - exit_px)
                trades.append({
                    "Entry Time": pos['time'], "Exit Time": df.index[i], "Type": pos['type'],
                    "Entry": pos['entry'], "Exit": exit_px, "PnL": pnl * q, "Reason": reason
                })
                in_pos = False

        if not in_pos:
            # Entry on Candle N+1 Open
            if prev['Buy_Sig']:
                in_pos, pos = True, {'type': 'Buy', 'entry': curr['Open'], 'sl': curr['Open'] - sl, 'tgt': curr['Open'] + tgt, 'time': df.index[i]}
            elif prev['Sell_Sig']:
                in_pos, pos = True, {'type': 'Sell', 'entry': curr['Open'], 'sl': curr['Open'] + sl, 'tgt': curr['Open'] - tgt, 'time': df.index[i]}
                
    return pd.DataFrame(trades), violations

# --- APP TABS ---
t1, t2, t3, t4 = st.tabs(["📊 Backtest", "⚡ Live", "📜 History", "⚙️ Optimizer"])

# LTP Header
header = st.empty()
df_main = fetch_data(ticker, interval, period)
if not df_main.empty:
    ltp = df_main['Close'].iloc[-1]
    prev_c = df_main['Close'].iloc[-2]
    chg = ltp - prev_c
    header.metric(f"{ticker} LTP", f"{ltp:.2f}", f"{chg:.2f} ({ (chg/prev_c)*100 :.2f}%)")

with t1:
    if st.button("Run Backtest"):
        res, v_count = run_backtest(df_main, fast_ema_val, slow_ema_val, sl_val, tgt_val, qty)
        if not res.empty:
            st.write(f"Violations (SL/Tgt same candle): {v_count}")
            # Fix for AttributeError: Use map instead of applymap
            st.dataframe(res.style.map(lambda x: 'color: green' if isinstance(x, float) and x > 0 else 'color: red', subset=['PnL']))
            
            fig = go.Figure(data=[go.Candlestick(x=df_main.index, open=df_main['Open'], high=df_main['High'], low=df_main['Low'], close=df_main['Close'])])
            fig.add_trace(go.Scatter(x=df_main.index, y=calc_ema(df_main['Close'], fast_ema_val), name="Fast EMA"))
            fig.add_trace(go.Scatter(x=df_main.index, y=calc_ema(df_main['Close'], slow_ema_val), name="Slow EMA"))
            st.plotly_chart(fig)

with t2:
    st.subheader("Live Trading")
    c1, c2 = st.columns(2)
    if c1.button("Start"): st.session_state.live_running = True
    if c2.button("Stop"): st.session_state.live_running = False

    if st.session_state.live_running:
        st.info("Bot is active. Polling every 1.5s...")
        # Live logic check against LTP
        current_ltp = df_main['Close'].iloc[-1]
        if st.session_state.current_position:
            cp = st.session_state.current_position
            # Tick-wise SL/Tgt check
            if (cp['type'] == 'Buy' and current_ltp <= cp['sl']) or (cp['type'] == 'Sell' and current_ltp >= cp['sl']):
                st.session_state.trades.append({"Time": datetime.datetime.now(), "Status": "SL HIT", "Price": current_ltp})
                st.session_state.current_position = None
        st.rerun()

with t3:
    st.dataframe(pd.DataFrame(st.session_state.trades))

with t4:
    st.subheader("Strategy Optimizer")
    st.write("Finds the best EMA combination for the current data.")
    if st.button("Optimize EMA Crossover"):
        opt_results = []
        with st.spinner("Testing 25 combinations..."):
            for f in [5, 9, 13, 20, 50]:
                for s in [15, 21, 34, 100, 200]:
                    if f >= s: continue
                    res, _ = run_backtest(df_main, f, s, sl_val, tgt_val, qty)
                    if not res.empty:
                        opt_results.append({"Fast": f, "Slow": s, "Total PnL": res['PnL'].sum(), "Win Rate": (len(res[res['PnL']>0])/len(res))*100})
        
        opt_df = pd.DataFrame(opt_results).sort_values(by="Total PnL", ascending=False)
        st.table(opt_df.head(10))
