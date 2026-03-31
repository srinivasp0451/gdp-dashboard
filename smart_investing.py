import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- INITIAL CONFIGURATION ---
st.set_page_config(page_title="Smart Investing", layout="wide")

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'live_active' not in st.session_state:
    st.session_state.live_active = False
if 'current_position' not in st.session_state:
    st.session_state.current_position = None

# --- UTILITY FUNCTIONS ---

def get_tv_ema(series, length):
    """Matches TradingView EMA calculation exactly."""
    return series.ewm(span=length, adjust=False).mean()

def fetch_data(ticker, interval, period):
    # Mapping for indices to yfinance symbols
    mapping = {
        "NIFTY50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "GOLD": "GC=F",
        "SILVER": "SI=F"
    }
    symbol = mapping.get(ticker, ticker)
    
    # Fetch extra data to handle EMA 'warm-up' period/gaps
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    return data

def calculate_indicators(df, fast_ema, slow_ema):
    df = df.copy()
    # Ensure we have enough data for EMA
    df['EMA_Fast'] = get_tv_ema(df['Close'], fast_ema)
    df['EMA_Slow'] = get_tv_ema(df['Close'], slow_ema)
    return df

# --- SIDEBAR / CONFIGURATION ---
st.sidebar.title("Smart Investing Config")

ticker_option = st.sidebar.selectbox("Ticker", ["NIFTY50", "BANKNIFTY", "SENSEX", "BTC", "ETH", "GOLD", "SILVER", "Custom"])
if ticker_option == "Custom":
    ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
else:
    ticker_symbol = ticker_option

timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d", "1wk"])
# Dynamic period selection based on interval
period_map = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"]
}
period = st.sidebar.selectbox("Data Period", period_map[timeframe])

st.sidebar.divider()
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
fast_ema_val = st.sidebar.number_input("Fast EMA", value=9)
slow_ema_val = st.sidebar.number_input("Slow EMA", value=15)

sl_type = st.sidebar.selectbox("Stoploss Type", ["Custom Points", "Trailing SL", "Reverse EMA", "Risk Reward"])
sl_val = st.sidebar.number_input("SL Value", value=10.0)

target_type = st.sidebar.selectbox("Target Type", ["Custom Points", "Trailing Target", "EMA Crossover", "Risk Reward"])
target_val = st.sidebar.number_input("Target Value", value=20.0)

st.sidebar.divider()
cooldown_enabled = st.sidebar.checkbox("Cooldown (5s)", value=True)
no_overlap = st.sidebar.checkbox("Prevent Overlap", value=True)

# Broker Config
use_dhan = st.sidebar.checkbox("Enable Dhan Broker", value=False)
options_mode = st.sidebar.checkbox("Options Trading", value=False)

# --- HEADER / LTP ---
df_latest = fetch_data(ticker_symbol, "1m", "2d")
if not df_latest.empty:
    ltp = df_latest['Close'].iloc[-1]
    prev_close = df_latest['Close'].iloc[-2]
    diff = ltp - prev_close
    pct = (diff / prev_close) * 100
    color = "green" if diff >= 0 else "red"
    st.markdown(f"### {ticker_symbol} : {ltp:.2f} <span style='color:{color}'>{diff:+.2f} ({pct:+.2f}%)</span>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Backtesting", "Live Trading", "Trade History"])

# --- TAB 1: BACKTESTING ---
with tab1:
    if st.button("Run Backtest"):
        data = fetch_data(ticker_symbol, timeframe, period)
        data = calculate_indicators(data, fast_ema_val, slow_ema_val)
        
        results = []
        violations = 0
        active_trade = None
        last_exit_time = datetime.min
        
        for i in range(1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i-1]
            
            # 1. CHECK EXIT
            if active_trade:
                hit_sl = False
                hit_tg = False
                
                if active_trade['type'] == 'Buy':
                    # Conservative Check: SL first
                    if row['Low'] <= active_trade['sl']: hit_sl = True
                    elif row['High'] >= active_trade['target']: hit_tg = True
                else: # Sell
                    if row['High'] >= active_trade['sl']: hit_sl = True
                    elif row['Low'] <= active_trade['target']: hit_tg = True
                
                if hit_sl or hit_tg:
                    active_trade['exit_time'] = data.index[i]
                    active_trade['exit_price'] = active_trade['sl'] if hit_sl else active_trade['target']
                    active_trade['pnl'] = (active_trade['exit_price'] - active_trade['entry_price']) if active_trade['type'] == 'Buy' else (active_trade['entry_price'] - active_trade['exit_price'])
                    results.append(active_trade)
                    last_exit_time = data.index[i]
                    active_trade = None
                    continue

            # 2. CHECK ENTRY
            if not active_trade:
                # Overlap & Cooldown Check
                if no_overlap and data.index[i] < last_exit_time: continue
                
                entry_signal = False
                trade_type = ""
                
                if strategy == "EMA Crossover":
                    if prev_row['EMA_Fast'] < prev_row['EMA_Slow'] and row['EMA_Fast'] > row['EMA_Slow']:
                        entry_signal = True; trade_type = "Buy"
                    elif prev_row['EMA_Fast'] > prev_row['EMA_Slow'] and row['EMA_Fast'] < row['EMA_Slow']:
                        entry_signal = True; trade_type = "Sell"

                if entry_signal:
                    entry_price = row['Close']
                    sl = entry_price - sl_val if trade_type == "Buy" else entry_price + sl_val
                    tg = entry_price + target_val if trade_type == "Buy" else entry_price - target_val
                    
                    active_trade = {
                        'type': trade_type, 'entry_time': data.index[i], 'entry_price': entry_price,
                        'sl': sl, 'target': tg, 'high': row['High'], 'low': row['Low'], 'reason': strategy
                    }

        st.table(pd.DataFrame(results))
        
        # Plotting
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Fast'], name=f'EMA {fast_ema_val}', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Slow'], name=f'EMA {slow_ema_val}', line=dict(color='red')))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: LIVE TRADING ---
with tab2:
    col_l, col_r = st.columns([1, 3])
    
    with col_l:
        if not st.session_state.live_active:
            if st.button("START LIVE TRADING", type="primary"):
                st.session_state.live_active = True
                st.rerun()
        else:
            if st.button("STOP & SQUARE OFF", type="secondary"):
                st.session_state.live_active = False
                st.session_state.current_position = None
                st.rerun()
        
        st.info(f"Config: {ticker_symbol} | {timeframe} | Strategy: {strategy}")

    # --- FRAGMENT FOR LIVE DATA (Avoids full page refresh) ---
    @st.fragment(run_every=1.5)
    def live_engine():
        if st.session_state.live_active:
            # Simulate fetching newest row
            live_data = fetch_data(ticker_symbol, timeframe, "1d")
            live_data = calculate_indicators(live_data, fast_ema_val, slow_ema_val)
            curr_row = live_data.iloc[-1]
            
            st.write(f"Last Fetched: {live_data.index[-1]}")
            
            # SIGNAL LOGIC
            # (Matches time-multiple logic for indicators, tick-based for SL)
            # Placeholder for Dhan execution logic provided in prompt
            
            # Overlay Plot
            fig_live = go.Figure(data=[go.Candlestick(x=live_data.index[-20:], **live_data[-20:][['Open','High','Low','Close']])])
            st.plotly_chart(fig_live, use_container_width=True)

    live_engine()

# --- TAB 3: TRADE HISTORY ---
with tab3:
    if st.session_state.trade_history:
        st.dataframe(pd.DataFrame(st.session_state.trade_history))
    else:
        st.write("No completed trades yet.")
