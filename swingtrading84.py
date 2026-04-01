import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as plotly_go

# ==========================================
# PAGE CONFIGURATION & STATE INITIALIZATION
# ==========================================
st.set_page_config(page_title="Smart Investing", layout="wide")

# Persistent state
if "trade_history" not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=[
        "Entry Time", "Exit Time", "Type", "Entry Price", "Exit Price", 
        "SL", "Target", "High During Trade", "Low During Trade", 
        "Entry Reason", "Exit Reason", "PnL", "Result"
    ])
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "current_position" not in st.session_state:
    st.session_state.current_position = None
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = 0

# ==========================================
# CORE UTILITIES
# ==========================================
TICKER_MAP = {
    "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC": "BTC-USD", "ETH": "ETH-USD", "GOLD": "GC=F", "SILVER": "SI=F", "CUSTOM": "CUSTOM"
}

TIMEFRAME_PERIODS = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"]
}

def fetch_data(ticker, period, interval):
    yf_ticker = st.session_state.get('custom_ticker_input', 'RELIANCE.NS') if ticker == "CUSTOM" else TICKER_MAP[ticker]
    try:
        df = yf.download(yf_ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

def calculate_indicators(df, fast_ema, slow_ema):
    df['EMA_Fast'] = df['Close'].ewm(span=fast_ema, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_ema, adjust=False).mean()
    df['Buy_Signal'] = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
    df['Sell_Signal'] = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))
    
    # ATR 14
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.selectbox("Select Ticker", list(TICKER_MAP.keys()))
if ticker == "CUSTOM":
    st.session_state.custom_ticker_input = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")

interval = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()))
period = st.sidebar.selectbox("Period", TIMEFRAME_PERIODS[interval])
qty = st.sidebar.number_input("Quantity", value=1, min_value=1)
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
fast_ema_val = st.sidebar.number_input("Fast EMA", value=9)
slow_ema_val = st.sidebar.number_input("Slow EMA", value=15)

sl_type = st.sidebar.selectbox("Stoploss Type", ["Custom Points", "ATR Based", "Reverse EMA"])
sl_val = st.sidebar.number_input("SL Value/Points", value=10.0)
tgt_type = st.sidebar.selectbox("Target Type", ["Custom Points", "ATR Based", "EMA Crossover"])
tgt_val = st.sidebar.number_input("Target Value/Points", value=20.0)
atr_mult = st.sidebar.number_input("ATR Multiplier", value=1.5)

cooldown_enabled = st.sidebar.checkbox("Enable Cooldown (5s)", value=True)
prevent_overlap = st.sidebar.checkbox("Prevent Overlap", value=True)

# Broker Placeholder (Simplified for logic maintenance)
use_broker = st.sidebar.checkbox("Enable Dhan Broker", value=False)
broker_config = {'use_dhan': use_broker}

# ==========================================
# HEADER
# ==========================================
st.title("📈 Smart Investing")
df_h = fetch_data(ticker, "5d", "1d")
if df_h is not None and len(df_h) >= 2:
    ltp = float(df_h['Close'].iloc[-1])
    prev = float(df_h['Close'].iloc[-2])
    diff = ltp - prev
    pct = (diff/prev)*100
    color = "green" if diff >= 0 else "red"
    st.markdown(f"<h3 style='color:{color};'>{ticker}: {ltp:.2f} ({diff:+.2f}, {pct:+.2f}%)</h3>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Backtesting", "🔴 Live Trading", "📝 Trade History"])

# ------------------------------------------
# TAB 1: BACKTESTING
# ------------------------------------------
with tab1:
    if st.button("Run Backtest"):
        df_bt = fetch_data(ticker, period, interval)
        if df_bt is not None:
            df_bt = calculate_indicators(df_bt, fast_ema_val, slow_ema_val)
            trades, in_pos, pos_type = [], False, None
            entry_p, entry_t, sl_p, tgt_p = 0, None, 0, 0
            
            for idx, row in df_bt.iterrows():
                if in_pos:
                    exit_p = 0
                    if pos_type == 'Buy':
                        if row['Low'] <= sl_p: exit_p, reason = sl_p, "SL Hit"
                        elif row['High'] >= tgt_p: exit_p, reason = tgt_p, "Target Hit"
                    else:
                        if row['High'] >= sl_p: exit_p, reason = sl_p, "SL Hit"
                        elif row['Low'] <= tgt_p: exit_p, reason = tgt_p, "Target Hit"
                    
                    if exit_p != 0:
                        pnl = (exit_p - entry_p) if pos_type == 'Buy' else (entry_p - exit_p)
                        trades.append({"Type": pos_type, "Entry Price": entry_p, "Exit Price": exit_p, "PnL": pnl * qty, "Points": pnl, "Result": "Win" if pnl > 0 else "Loss", "Entry Time": entry_t, "Exit Time": idx})
                        in_pos = False

                if not in_pos:
                    sig = 'Buy' if row['Buy_Signal'] else ('Sell' if row['Sell_Signal'] else None)
                    if sig:
                        in_pos, pos_type, entry_p, entry_t = True, sig, row['Close'], idx
                        mult = 1 if sig == 'Buy' else -1
                        cur_sl = sl_val if sl_type == "Custom Points" else (row['ATR'] * atr_mult if not pd.isna(row['ATR']) else sl_val)
                        cur_tgt = tgt_val if tgt_type == "Custom Points" else (row['ATR'] * atr_mult if not pd.isna(row['ATR']) else tgt_val)
                        sl_p = entry_p - (cur_sl * mult)
                        tgt_p = entry_p + (cur_tgt * mult)

            # Summary Metrics
            if trades:
                tdf = pd.DataFrame(trades)
                pos_tdf = tdf[tdf['Points'] > 0]
                neg_tdf = tdf[tdf['Points'] <= 0]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Net PnL", f"{tdf['PnL'].sum():.2f}")
                c2.metric("Accuracy", f"{(len(pos_tdf)/len(tdf))*100:.1f}%")
                c3.metric("Profit Trades", len(pos_tdf))
                c4.metric("Loss Trades", len(neg_tdf))
                
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Avg Profit", f"{pos_tdf['Points'].mean():.2f}")
                c6.metric("Avg Loss", f"{neg_tdf['Points'].mean():.2f}")
                c7.metric("Max Profit (Pts)", f"{tdf['Points'].max():.2f}")
                c8.metric("Max Loss (Pts)", f"{tdf['Points'].min():.2f}")
                
                st.write("**Point Summary**")
                st.write(f"Total Points Gained: {pos_tdf['Points'].sum():.2f} | Total Points Lost: {abs(neg_tdf['Points'].sum()):.2f}")
                st.dataframe(tdf)

# ------------------------------------------
# TAB 2: LIVE TRADING
# ------------------------------------------
with tab2:
    c_btn1, c_btn2 = st.columns(2)
    if c_btn1.button("Start Live", use_container_width=True): st.session_state.live_running = True
    if c_btn2.button("Stop Live", use_container_width=True): st.session_state.live_running = False
    
    dashboard = st.empty()
    chart_spot = st.empty()
    
    if st.session_state.live_running:
        while st.session_state.live_running:
            df_l = fetch_data(ticker, "1d", interval)
            if df_l is not None:
                df_l = calculate_indicators(df_l, fast_ema_val, slow_ema_val)
                latest = df_l.iloc[-1]
                cur_p = float(latest['Close'])
                f_ema = float(latest['EMA_Fast'])
                s_ema = float(latest['EMA_Slow'])
                
                with dashboard.container():
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric(f"EMA {fast_ema_val}", f"{f_ema:.2f}")
                    m2.metric(f"EMA {slow_ema_val}", f"{s_ema:.2f}")
                    m3.metric("LTP", f"{cur_p:.2f}")
                    
                    if st.session_state.current_position:
                        pos = st.session_state.current_position
                        live_pnl = (cur_p - pos['price']) * qty if pos['type'] == 'Buy' else (pos['price'] - cur_p) * qty
                        
                        st.info(f"**Active Trade:** {pos['type']} | **Entry:** {pos['price']:.2f} ({pos['time'].strftime('%H:%M:%S')})")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Unrealized PnL", f"{live_pnl:.2f}", delta=f"{live_pnl:.2f}")
                        p2.metric("Stop Loss", f"{pos['sl']:.2f}")
                        p3.metric("Target", f"{pos['tgt']:.2f}")
                        
                        # Logic for Exit
                        if (pos['type'] == 'Buy' and (cur_p <= pos['sl'] or cur_p >= pos['tgt'])) or \
                           (pos['type'] == 'Sell' and (cur_p >= pos['sl'] or cur_p <= pos['tgt'])):
                            st.session_state.current_position = None
                            st.toast("Trade Exited!")
                    else:
                        st.write("Status: Scanning for signals...")
                        # Logic for Entry
                        if latest['Buy_Signal'] or latest['Sell_Signal']:
                            sig = 'Buy' if latest['Buy_Signal'] else 'Sell'
                            m = 1 if sig == 'Buy' else -1
                            sl_p = cur_p - (sl_val * m)
                            tgt_p = cur_p + (tgt_val * m)
                            st.session_state.current_position = {'type': sig, 'price': cur_p, 'time': datetime.now(), 'sl': sl_p, 'tgt': tgt_p}
                
                with chart_spot.container():
                    fig = plotly_go.Figure()
                    df_p = df_l.tail(50)
                    fig.add_trace(plotly_go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
                    fig.add_trace(plotly_go.Scatter(x=df_p.index, y=df_p['EMA_Fast'], name=f"EMA {fast_ema_val}", line=dict(color='orange')))
                    fig.add_trace(plotly_go.Scatter(x=df_p.index, y=df_p['EMA_Slow'], name=f"EMA {slow_ema_val}", line=dict(color='blue')))
                    fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{time.time()}")
                    
            time.sleep(2)

# ------------------------------------------
# TAB 3: TRADE HISTORY
# ------------------------------------------
with tab3:
    st.subheader("Trade Logs")
    if not st.session_state.trade_history.empty:
        th = st.session_state.trade_history
        # FIX: Handle pandas 2.1.0+ Styler.map vs older applymap
        def color_result(val):
            color = 'rgba(0, 255, 0, 0.1)' if val == 'Win' else 'rgba(255, 0, 0, 0.1)'
            return f'background-color: {color}'

        try:
            styled_df = th.style.map(color_result, subset=['Result'])
        except AttributeError:
            styled_df = th.style.applymap(color_result, subset=['Result'])
            
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No trades recorded yet.")
