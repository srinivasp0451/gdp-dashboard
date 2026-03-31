import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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

# --- MANUAL INDICATOR CALCULATIONS (TradingView Match) ---
def calculate_ema_manual(series, length):
    """
    Manually calculates EMA to match TradingView.
    Formula: EMA = Price * Multiplier + Previous EMA * (1 - Multiplier)
    Multiplier = 2 / (length + 1)
    """
    return series.ewm(span=length, adjust=False).mean()

def get_ticker_symbol(selection):
    mapping = {
        "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
        "BTC": "BTC-USD", "ETH": "ETH-USD", "GOLD": "GC=F", "SILVER": "SI=F"
    }
    return mapping.get(selection, selection)

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.title("🚀 Smart Investing")

ticker_display = st.sidebar.selectbox("Ticker", ["NIFTY50", "BANKNIFTY", "SENSEX", "BTC", "ETH", "GOLD", "SILVER", "Custom"])
ticker_symbol = ticker_display if ticker_display != "Custom" else st.sidebar.text_input("Custom Ticker", "RELIANCE.NS")
symbol = get_ticker_symbol(ticker_symbol)

timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d", "1wk"])
period_options = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"]
}
period = st.sidebar.selectbox("Period", period_options[timeframe])

st.sidebar.divider()
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
fast_ema_len = st.sidebar.number_input("Fast EMA", value=9)
slow_ema_len = st.sidebar.number_input("Slow EMA", value=15)

sl_type = st.sidebar.selectbox("Stoploss", ["Custom Points", "Trailing SL", "Reverse EMA", "Risk Reward"])
sl_val = st.sidebar.number_input("SL Points", value=10.0)
target_type = st.sidebar.selectbox("Target", ["Custom Points", "Trailing Target", "EMA Crossover", "Risk Reward"])
target_val = st.sidebar.number_input("Target Points", value=20.0)

st.sidebar.divider()
cooldown_on = st.sidebar.checkbox("Cooldown (5s)", value=True)
no_overlap = st.sidebar.checkbox("Prevent Overlapping Trades", value=True)

# Broker & Options Settings
use_dhan = st.sidebar.checkbox("Enable Dhan Broker", value=False)
is_options = st.sidebar.checkbox("Options Trading", value=False)

# --- LTP HEADER ---
try:
    header_data = yf.download(symbol, period="2d", interval="1m", progress=False)
    if not header_data.empty:
        ltp = header_data['Close'].iloc[-1]
        prev_close = header_data['Close'].iloc[-2]
        change = ltp - prev_close
        pct = (change / prev_close) * 100
        color = "green" if change >= 0 else "red"
        st.markdown(f"## {ticker_symbol}: {ltp:.2f} <span style='color:{color}'>{change:+.2f} ({pct:+.2f}%)</span>", unsafe_allow_html=True)
except:
    st.error("Market Data Unavailable")

tab1, tab2, tab3 = st.tabs(["📊 Backtesting", "⚡ Live Trading", "📜 Trade History"])

# --- TAB 1: BACKTESTING ---
with tab1:
    if st.button("Run Backtest"):
        data = yf.download(symbol, period=period, interval=timeframe, progress=False)
        # Manual indicators
        data['EMA_Fast'] = calculate_ema_manual(data['Close'], fast_ema_len)
        data['EMA_Slow'] = calculate_ema_manual(data['Close'], slow_ema_len)
        
        trades = []
        violations = 0
        active = None
        last_exit = datetime.min

        for i in range(1, len(data)):
            row = data.iloc[i]
            prev = data.iloc[i-1]
            ts = data.index[i]

            if active:
                # CONSERVATIVE EXIT: Check SL FIRST for both Buy and Sell
                exit_triggered = False
                if active['type'] == 'Buy':
                    if row['Low'] <= active['sl']: 
                        exit_price, reason, exit_triggered = active['sl'], "Stoploss", True
                    elif row['High'] >= active['target']:
                        exit_price, reason, exit_triggered = active['target'], "Target", True
                    # Check Violation (Both hit in same candle)
                    if row['Low'] <= active['sl'] and row['High'] >= active['target']: violations += 1
                else: # Sell
                    if row['High'] >= active['sl']:
                        exit_price, reason, exit_triggered = active['sl'], "Stoploss", True
                    elif row['Low'] <= active['target']:
                        exit_price, reason, exit_triggered = active['target'], "Target", True
                    if row['High'] >= active['sl'] and row['Low'] <= active['target']: violations += 1

                if exit_triggered:
                    active.update({'exit_time': ts, 'exit_price': exit_price, 'reason_exit': reason,
                                   'pnl': (exit_price - active['entry_price']) if active['type'] == 'Buy' else (active['entry_price'] - exit_price)})
                    trades.append(active)
                    last_exit = ts
                    active = None
                    continue

            # ENTRY LOGIC
            if not active:
                if no_overlap and ts <= last_exit: continue
                
                sig = False
                t_type = ""
                if strategy == "EMA Crossover":
                    if prev['EMA_Fast'] < prev['EMA_Slow'] and row['EMA_Fast'] > row['EMA_Slow']:
                        sig, t_type = True, "Buy"
                    elif prev['EMA_Fast'] > prev['EMA_Slow'] and row['EMA_Fast'] < row['EMA_Slow']:
                        sig, t_type = True, "Sell"

                if sig:
                    ep = row['Close']
                    active = {'type': t_type, 'entry_time': ts, 'entry_price': ep,
                              'sl': ep - sl_val if t_type == "Buy" else ep + sl_val,
                              'target': ep + target_val if t_type == "Buy" else ep - target_val,
                              'High': row['High'], 'Low': row['Low'], 'reason_entry': strategy}

        st.metric("Strategy Violations", violations, help="Cases where SL and Target hit in the same candle. SL was chosen.")
        if trades:
            res_df = pd.DataFrame(trades)
            st.dataframe(res_df)
            
            # Plot
            fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Fast'], name="Fast EMA", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Slow'], name="Slow EMA", line=dict(color='red')))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: LIVE TRADING ---
with tab2:
    c1, c2 = st.columns([1, 2])
    with c1:
        if not st.session_state.live_active:
            if st.button("▶ START LIVE ALGO", use_container_width=True):
                st.session_state.live_active = True
                st.rerun()
        else:
            if st.button("🛑 STOP & SQUARE OFF", type="primary", use_container_width=True):
                st.session_state.live_active = False
                st.session_state.current_position = None
                st.rerun()

    # Fragment for non-flickering live updates
    @st.fragment(run_every=1.5)
    def live_monitoring():
        if st.session_state.live_active:
            # Fetch data for live (ensure sufficient history for EMAs)
            live_df = yf.download(symbol, period="5d", interval=timeframe, progress=False)
            live_df['EMA_Fast'] = calculate_ema_manual(live_df['Close'], fast_ema_len)
            live_df['EMA_Slow'] = calculate_ema_manual(live_df['Close'], slow_ema_len)
            
            curr_ltp = live_df['Close'].iloc[-1]
            st.write(f"**Last Tick Time:** {live_df.index[-1]} | **LTP:** {curr_ltp}")
            
            # Display configuration summary
            with st.expander("Active Configuration", expanded=False):
                st.json({"Ticker": ticker_symbol, "Timeframe": timeframe, "EMA": f"{fast_ema_len}/{slow_ema_len}", "SL": sl_val})
            
            # Position Display
            if st.session_state.current_position:
                pos = st.session_state.current_position
                pnl = (curr_ltp - pos['price']) if pos['type'] == 'Buy' else (pos['price'] - curr_ltp)
                st.success(f"Position: {pos['type']} | Entry: {pos['price']} | PnL: {pnl:.2f}")
            
            # Chart
            fig_live = go.Figure(data=[go.Candlestick(x=live_df.index[-30:], **live_df.iloc[-30:][['Open','High','Low','Close']])])
            st.plotly_chart(fig_live, use_container_width=True)

    live_monitoring()

# --- TAB 3: TRADE HISTORY ---
with tab3:
    if st.session_state.trade_history:
        st.table(pd.DataFrame(st.session_state.trade_history))
    else:
        st.info("No trades executed in this session.")
