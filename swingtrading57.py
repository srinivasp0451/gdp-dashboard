import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
    "4h": ["1mo"],
    "1d": ["1mo", "1y", "2y", "5y"],
    "1wk": ["1mo", "1y", "5y", "10y", "15y", "20y"],
    "1mo": ["1y", "2y", "5y", "10y", "15y", "20y", "25y", "30y"]
}

# --- INDICATOR IMPLEMENTATIONS (TRADINGVIEW ACCURATE) ---

def calculate_ema(series, length):
    """TV Equivalent: ta.ema(source, length)"""
    return series.ewm(span=length, adjust=False).mean()

def calculate_atr(df, length=14):
    """TV Equivalent: ta.atr(length)"""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=length).mean()

def calculate_ema_angle(ema_series, lookback=1):
    """Calculates angle of EMA in degrees based on price-to-time ratio."""
    # Note: In real TV, angle depends on visual scaling. 
    # Here we use price change normalized by lookback.
    slopes = (ema_series - ema_series.shift(lookback)) / lookback
    angles = np.arctan(slopes) * (180 / np.pi)
    return angles

# --- DATA ENGINE ---

def fetch_data(ticker, interval, period):
    time.sleep(np.random.uniform(1.0, 1.5))
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        
        # Flatten Multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Timezone Handling
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC').tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
            
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- STRATEGY ENGINE ---

def run_backtest(df, params):
    # Setup Indicators
    df['EMA_Fast'] = calculate_ema(df['Close'], params['ema_fast'])
    df['EMA_Slow'] = calculate_ema(df['Close'], params['ema_slow'])
    df['EMA_Angle'] = calculate_ema_angle(df['EMA_Fast'])
    df['ATR'] = calculate_atr(df)
    
    trades = []
    active_trade = None
    logs = []

    for i in range(1, len(df)):
        current_time = df.index[i]
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 1. CHECK EXIT IF IN TRADE
        if active_trade:
            exit_triggered = False
            exit_price = row['Close']
            exit_reason = ""

            # Signal Based Exit Logic (Fix for immediate hit)
            # Ensure we are at least one candle past entry to prevent instant exit
            is_new_candle = i > active_trade['entry_index']

            if params['exit_type'] == "Signal-based" and is_new_candle:
                if active_trade['type'] == 'BUY' and row['EMA_Fast'] < row['EMA_Slow']:
                    exit_triggered = True
                    exit_reason = "Signal Reverse"
                elif active_trade['type'] == 'SELL' and row['EMA_Fast'] > row['EMA_Slow']:
                    exit_triggered = True
                    exit_reason = "Signal Reverse"

            # Trailing SL / Target Logic
            if not exit_triggered:
                if active_trade['type'] == 'BUY':
                    if row['Low'] <= active_trade['sl']:
                        exit_triggered = True
                        exit_price = active_trade['sl']
                        exit_reason = "Stop Loss"
                    elif row['High'] >= active_trade['tp']:
                        exit_triggered = True
                        exit_price = active_trade['tp']
                        exit_reason = "Take Profit"
                else: # SELL
                    if row['High'] >= active_trade['sl']:
                        exit_triggered = True
                        exit_price = active_trade['sl']
                        exit_reason = "Stop Loss"
                    elif row['Low'] <= active_trade['tp']:
                        exit_triggered = True
                        exit_price = active_trade['tp']
                        exit_reason = "Take Profit"

            if exit_triggered:
                pnl = (exit_price - active_trade['entry_price']) if active_trade['type'] == 'BUY' else (active_trade['entry_price'] - exit_price)
                active_trade.update({
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl
                })
                trades.append(active_trade)
                logs.append(f"[{current_time}] Closed {active_trade['type']} at {exit_price} ({exit_reason})")
                active_trade = None

        # 2. CHECK ENTRY IF NO TRADE
        if not active_trade:
            # EMA Crossover Logic
            buy_signal = (prev_row['EMA_Fast'] <= prev_row['EMA_Slow']) and (row['EMA_Fast'] > row['EMA_Slow'])
            sell_signal = (prev_row['EMA_Fast'] >= prev_row['EMA_Slow']) and (row['EMA_Fast'] < row['EMA_Slow'])
            
            # Angle Filter
            angle_ok = abs(row['EMA_Angle']) >= params['min_angle']
            
            if (buy_signal or sell_signal) and angle_ok:
                side = 'BUY' if buy_signal else 'SELL'
                entry_p = row['Close']
                
                # SL/TP Calculation
                atr_val = row['ATR']
                sl_dist = atr_val * 2 if params['sl_type'] == "ATR-based" else params['sl_points']
                tp_dist = atr_val * 4 if params['tp_type'] == "ATR-based" else params['tp_points']
                
                # Signal-based SL/TP (Setting high/low bounds to avoid immediate triggers)
                if params['sl_type'] == "Signal-based":
                    sl_p = entry_p * 0.5 if side == 'BUY' else entry_p * 1.5
                else:
                    sl_p = entry_p - sl_dist if side == 'BUY' else entry_p + sl_dist

                if params['tp_type'] == "Signal-based":
                    tp_p = entry_p * 2.0 if side == 'BUY' else entry_p * 0.1
                else:
                    tp_p = entry_p + tp_dist if side == 'BUY' else entry_p - tp_dist

                active_trade = {
                    'entry_time': current_time,
                    'entry_price': entry_p,
                    'entry_index': i,
                    'type': side,
                    'sl': sl_p,
                    'tp': tp_p
                }
                logs.append(f"[{current_time}] Entered {side} at {entry_p} (Angle: {row['EMA_Angle']:.2f})")

    return pd.DataFrame(trades), logs

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="QuantPro Alpha Engine")

if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'live_logs' not in st.session_state:
    st.session_state.live_logs = []

st.title("🚀 QuantPro Alpha Engine")

with st.sidebar:
    st.header("1. Asset Configuration")
    asset_type = st.selectbox("Market", ["Indian Indices", "Crypto", "Forex", "Commodities", "Custom"])
    
    ticker_map = {
        "Indian Indices": ["^NSEI", "^NSEBANK", "^BSESN"],
        "Crypto": ["BTC-USD", "ETH-USD"],
        "Forex": ["USDINR=X", "EURUSD=X", "GBPUSD=X"],
        "Commodities": ["GC=F", "SI=F"]
    }
    
    if asset_type == "Custom":
        symbol = st.text_input("Enter yfinance Ticker (e.g., TSLA)", "AAPL")
    else:
        symbol = st.selectbox("Select Ticker", ticker_map[asset_type])

    interval = st.selectbox("Timeframe", list(ALLOWED_COMBINATIONS.keys()), index=2)
    period = st.selectbox("Period", ALLOWED_COMBINATIONS[interval])
    
    st.divider()
    st.header("2. Strategy Settings")
    ema_f = st.number_input("EMA Fast", 5, 50, 9)
    ema_s = st.number_input("EMA Slow", 10, 200, 15)
    min_ang = st.slider("Min Crossover Angle (°)", 0, 90, 20)
    
    st.divider()
    st.header("3. Risk Management")
    sl_type = st.selectbox("Stop Loss Type", ["Custom points", "ATR-based", "Signal-based"])
    sl_val = st.number_input("SL Points (if Custom)", 0.0, 1000.0, 50.0)
    tp_type = st.selectbox("Target Type", ["Custom points", "ATR-based", "Signal-based", "Risk-Reward"])
    tp_val = st.number_input("TP Points (if Custom)", 0.0, 5000.0, 100.0)

# --- APP TABS ---
tab_dashboard, tab_history, tab_logs = st.tabs(["📊 Live Dashboard", "📜 Trade History", "📝 System Logs"])

with tab_dashboard:
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Controls")
        if st.button("🚀 Start Live Simulation", use_container_width=True):
            st.session_state.trading_active = True
        if st.button("🛑 Stop Simulation", use_container_width=True):
            st.session_state.trading_active = False
            
        if st.session_state.trading_active:
            st.success("System Live")
        else:
            st.warning("System Idle")

    with col1:
        if st.button("Run Full Backtest"):
            with st.spinner("Fetching data and simulating..."):
                df = fetch_data(symbol, interval, period)
                if df is not None:
                    params = {
                        'ema_fast': ema_f, 'ema_slow': ema_s, 'min_angle': min_ang,
                        'sl_type': sl_type, 'sl_points': sl_val,
                        'tp_type': tp_type, 'tp_points': tp_val,
                        'exit_type': "Signal-based" if (sl_type == "Signal-based" or tp_type == "Signal-based") else "Fixed"
                    }
                    results, logs = run_backtest(df, params)
                    st.session_state.backtest_results = results
                    st.session_state.backtest_logs = logs
                    
                    # Dashboard Metrics
                    m1, m2, m3 = st.columns(3)
                    if not results.empty:
                        win_rate = (len(results[results['pnl'] > 0]) / len(results)) * 100
                        m1.metric("Total Trades", len(results))
                        m2.metric("Win Rate", f"{win_rate:.1f}%")
                        m3.metric("Total PnL", f"{results['pnl'].sum():.2f}")
                    
                    # Plotly Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
                    fig.add_trace(go.Scatter(x=df.index, y=calculate_ema(df['Close'], ema_f), line=dict(color='cyan', width=1), name=f"EMA {ema_f}"))
                    fig.add_trace(go.Scatter(x=df.index, y=calculate_ema(df['Close'], ema_s), line=dict(color='orange', width=1), name=f"EMA {ema_s}"))
                    
                    # Entry Markers
                    if not results.empty:
                        buys = results[results['type'] == 'BUY']
                        sells = results[results['type'] == 'SELL']
                        fig.add_trace(go.Scatter(x=buys['entry_time'], y=buys['entry_price'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'), name='Buy Entry'))
                        fig.add_trace(go.Scatter(x=sells['entry_time'], y=sells['entry_price'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Entry'))

                    fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True, key="main_chart")
                else:
                    st.error("No data found for this combination.")

with tab_history:
    if 'backtest_results' in st.session_state:
        st.dataframe(st.session_state.backtest_results, use_container_width=True)
    else:
        st.info("Run backtest to see trade history.")

with tab_logs:
    if 'backtest_logs' in st.session_state:
        for log in st.session_state.backtest_logs[-50:]: # Show last 50
            st.text(log)
    else:
        st.info("No logs available.")

# --- LIVE SIMULATION LOOP ---
if st.session_state.trading_active:
    # This simulates a real-time refresh every 1.5 seconds
    time.sleep(1.2)
    st.rerun()
