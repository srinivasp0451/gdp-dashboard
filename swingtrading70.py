import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz

# ==========================================
# 1. CONFIGURATION & STATE MANAGEMENT
# ==========================================

st.set_page_config(page_title="Pro Algo Trader v2", layout="wide", page_icon="⚡")

# IST Timezone
IST = pytz.timezone('Asia/Kolkata')

# Ticker Database
ASSETS = {
    "Indices": {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS"
    },
    "Crypto": {
        "BTC USD": "BTC-USD",
        "ETH USD": "ETH-USD",
        "SOL USD": "SOL-USD"
    },
    "Forex": {
        "USD/INR": "USDINR=X",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X"
    },
    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil": "CL=F"
    },
    "Custom": {"Custom": "CUSTOM"}
}

def init_session_state():
    defaults = {
        'trading_active': False,
        'position': None,  # {type: 1/-1, entry_price, quantity, sl, target, entry_time}
        'trade_history': [],
        'trade_logs': [],
        'highest_price': 0.0, # Track highest price during LONG trade
        'lowest_price': 0.0,  # Track lowest price during SHORT trade
        'custom_conditions': [],
        'stop_request': False # Flag to break the while loop
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

def add_log(message):
    timestamp = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    st.session_state['trade_logs'].append(log_entry)
    if len(st.session_state['trade_logs']) > 50:
        st.session_state['trade_logs'].pop(0)

def reset_position_state():
    st.session_state['position'] = None
    st.session_state['highest_price'] = 0.0
    st.session_state['lowest_price'] = 0.0

# ==========================================
# 2. INDICATOR LIBRARY
# ==========================================

class Indicators:
    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        u = delta.where(delta > 0, 0)
        d = -delta.where(delta < 0, 0)
        avg_u = u.rolling(window=period).mean()
        avg_d = d.rolling(window=period).mean()
        # Wilder's smoothing
        for i in range(period, len(series)):
            avg_u.iloc[i] = (avg_u.iloc[i-1] * (period - 1) + u.iloc[i]) / period
            avg_d.iloc[i] = (avg_d.iloc[i-1] * (period - 1) + d.iloc[i]) / period
        rs = avg_u / avg_d
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/period).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/period).mean()

    @staticmethod
    def ema_angle(ema_series):
        diff = ema_series.diff()
        return np.degrees(np.arctan(diff))

# ==========================================
# 3. DATA ENGINE
# ==========================================

def get_ticker_symbol(category, name, custom_input):
    if category == "Custom":
        return custom_input
    return ASSETS[category][name]

def fetch_data(ticker, interval, period):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        
        # Flatten and Cleanup
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
            
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols].copy()
        
        # Calculate Indicators
        df['EMA_Fast'] = Indicators.ema(df['Close'], 9)
        df['EMA_Slow'] = Indicators.ema(df['Close'], 15)
        df['RSI'] = Indicators.rsi(df['Close'])
        df['ATR'] = Indicators.atr(df['High'], df['Low'], df['Close'])
        df['ADX'] = Indicators.adx(df['High'], df['Low'], df['Close'])
        df['EMA_Angle'] = Indicators.ema_angle(df['EMA_Fast'])
        
        return df
    except Exception:
        return None

# ==========================================
# 4. SIGNAL LOGIC
# ==========================================

def check_signals(df, config):
    if len(df) < 20: return 0, "No Data"
    
    i = -1
    curr = df.iloc[i]
    prev = df.iloc[i-1]
    
    signal = 0
    reason = ""
    
    st_name = config['strategy']
    
    if st_name == "EMA Crossover":
        # Check Crossover
        bullish = (curr['EMA_Fast'] > curr['EMA_Slow']) and (prev['EMA_Fast'] <= prev['EMA_Slow'])
        bearish = (curr['EMA_Fast'] < curr['EMA_Slow']) and (prev['EMA_Fast'] >= prev['EMA_Slow'])
        
        # Angle Check
        angle = abs(curr['EMA_Angle'])
        angle_ok = angle >= config['min_angle']
        
        # ADX Check
        adx_ok = True
        if config['use_adx']:
            adx_ok = curr['ADX'] >= config['adx_threshold']
            
        if bullish and angle_ok and adx_ok:
            signal = 1
            reason = f"Bullish Cross (Angle: {angle:.1f})"
        elif bearish and angle_ok and adx_ok:
            signal = -1
            reason = f"Bearish Cross (Angle: {angle:.1f})"
            
    elif st_name == "RSI Strategy":
        if curr['RSI'] < 30:
            signal = 1
            reason = "RSI Oversold (<30)"
        elif curr['RSI'] > 70:
            signal = -1
            reason = "RSI Overbought (>70)"

    elif st_name == "Price Threshold":
        price = curr['Close']
        if config['threshold_mode'] == "Price >=" and price >= config['threshold_val']:
            signal = 1 if config['threshold_action'] == "BUY" else -1
            reason = f"Price {price:.2f} >= {config['threshold_val']}"
        elif config['threshold_mode'] == "Price <=" and price <= config['threshold_val']:
            signal = 1 if config['threshold_action'] == "BUY" else -1
            reason = f"Price {price:.2f} <= {config['threshold_val']}"

    return signal, reason

# ==========================================
# 5. UI CONFIGURATION
# ==========================================

def render_sidebar():
    st.sidebar.header("1. Asset Selection")
    
    cat = st.sidebar.selectbox("Category", list(ASSETS.keys()))
    
    if cat == "Custom":
        ticker_name = "Custom"
        ticker_symbol = st.sidebar.text_input("YFinance Ticker", value="RELIANCE.NS")
    else:
        ticker_name = st.sidebar.selectbox("Asset", list(ASSETS[cat].keys()))
        ticker_symbol = ASSETS[cat][ticker_name]

    col1, col2 = st.sidebar.columns(2)
    interval = col1.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
    period = col2.selectbox("Period", ["1d", "5d", "1mo", "1y"], index=2)
    qty = st.sidebar.number_input("Quantity", 1, 10000, 1)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Strategy")
    strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "RSI Strategy", "Price Threshold"])
    
    config = {'strategy': strategy, 'ticker': ticker_symbol, 'interval': interval, 'period': period, 'qty': qty}
    
    if strategy == "EMA Crossover":
        config['min_angle'] = st.sidebar.number_input("Min Angle", 0.0, 90.0, 0.5)
        config['use_adx'] = st.sidebar.checkbox("Use ADX Filter", value=False)
        if config['use_adx']:
            config['adx_threshold'] = st.sidebar.number_input("ADX Threshold", 10, 50, 25)
    
    elif strategy == "Price Threshold":
        config['threshold_mode'] = st.sidebar.selectbox("Condition", ["Price >=", "Price <="])
        config['threshold_val'] = st.sidebar.number_input("Value", value=20000.0)
        config['threshold_action'] = st.sidebar.selectbox("Action", ["BUY", "SELL"])

    st.sidebar.markdown("---")
    st.sidebar.header("3. Risk Management")
    
    sl_type = st.sidebar.selectbox("SL Type", ["Fixed Points", "Trailing SL (Points)", "ATR Based"])
    sl_points = st.sidebar.number_input("SL Points / Multiplier", 0.1, 1000.0, 20.0)
    
    target_type = st.sidebar.selectbox("Target Type", ["Fixed Points", "Open Target"])
    target_points = st.sidebar.number_input("Target Points", 0.1, 1000.0, 40.0)
    
    config.update({'sl_type': sl_type, 'sl_points': sl_points, 'target_type': target_type, 'target_points': target_points})
    
    return config

# ==========================================
# 6. MAIN APPLICATION
# ==========================================

def main():
    config = render_sidebar()
    
    st.title("⚡ Pro Algo Trader v2")
    
    tab1, tab2, tab3 = st.tabs(["Live Dashboard", "Backtest Analysis", "Trade History"])
    
    # --- TAB 1: LIVE TRADING ---
    with tab1:
        # Control Panel
        col1, col2, col3 = st.columns([1, 1, 4])
        
        # Start/Stop Logic
        if st.session_state['trading_active']:
            if col2.button("STOP TRADING", type="primary"):
                st.session_state['stop_request'] = True
                st.session_state['trading_active'] = False
                add_log("User requested stop.")
                st.rerun()
            st.success("System Running...")
        else:
            if col1.button("START TRADING"):
                st.session_state['trading_active'] = True
                st.session_state['stop_request'] = False
                add_log("Trading session started.")
                st.rerun()
            st.info("System Stopped.")

        # Placeholders for Flicker-Free Updates
        metrics_ph = st.empty()
        chart_ph = st.empty()
        
        # --- THE TRADING LOOP ---
        if st.session_state['trading_active']:
            while not st.session_state['stop_request']:
                
                # 1. Fetch Data
                df = fetch_data(config['ticker'], config['interval'], config['period'])
                if df is None:
                    time.sleep(2)
                    continue
                
                curr = df.iloc[-1]
                curr_price = curr['Close']
                
                # 2. Strategy Logic
                
                # A. Entry Logic
                if st.session_state['position'] is None:
                    sig, reason = check_signals(df, config)
                    if sig != 0:
                        # Calc SL/Target
                        sl_val = config['sl_points']
                        if config['sl_type'] == "ATR Based":
                             sl_val = curr['ATR'] * config['sl_points']
                             
                        sl_price = curr_price - sl_val if sig == 1 else curr_price + sl_val
                        tg_price = curr_price + config['target_points'] if sig == 1 else curr_price - config['target_points']
                        
                        st.session_state['position'] = {
                            'type': sig, 'entry_price': curr_price, 'quantity': config['qty'],
                            'sl': sl_price, 'target': tg_price, 
                            'entry_time': datetime.now(IST).strftime('%H:%M:%S')
                        }
                        
                        # Init High/Low tracking for Trailing
                        st.session_state['highest_price'] = curr_price
                        st.session_state['lowest_price'] = curr_price
                        
                        add_log(f"ENTRY {'LONG' if sig==1 else 'SHORT'} @ {curr_price:.2f} | {reason}")

                # B. Exit & Trailing Logic
                elif st.session_state['position'] is not None:
                    pos = st.session_state['position']
                    p_type = pos['type']
                    
                    # Update High/Low Watermarks for Trailing
                    if curr_price > st.session_state['highest_price']: st.session_state['highest_price'] = curr_price
                    if curr_price < st.session_state['lowest_price']: st.session_state['lowest_price'] = curr_price
                    
                    # Trailing SL Update Logic
                    if config['sl_type'] == "Trailing SL (Points)":
                        dist = config['sl_points']
                        if p_type == 1: # LONG
                            potential_sl = st.session_state['highest_price'] - dist
                            if potential_sl > pos['sl']: # Move SL Up
                                pos['sl'] = potential_sl
                        else: # SHORT
                            potential_sl = st.session_state['lowest_price'] + dist
                            if potential_sl < pos['sl']: # Move SL Down
                                pos['sl'] = potential_sl

                    # Check Exits
                    exit_reason = None
                    pnl = 0
                    
                    # SL Hit
                    if (p_type == 1 and curr_price <= pos['sl']) or (p_type == -1 and curr_price >= pos['sl']):
                        exit_reason = "SL Hit"
                    
                    # Target Hit
                    elif config['target_type'] == "Fixed Points":
                        if (p_type == 1 and curr_price >= pos['target']) or (p_type == -1 and curr_price <= pos['target']):
                            exit_reason = "Target Hit"
                            
                    if exit_reason:
                        exit_p = pos['sl'] if "SL" in exit_reason else pos['target']
                        pnl = (exit_p - pos['entry_price']) * pos['quantity'] if p_type == 1 else (pos['entry_price'] - exit_p) * pos['quantity']
                        
                        st.session_state['trade_history'].append({
                            'Time': datetime.now(IST).strftime('%H:%M:%S'),
                            'Symbol': config['ticker'],
                            'Type': "LONG" if p_type == 1 else "SHORT",
                            'Entry': pos['entry_price'], 'Exit': exit_p,
                            'PnL': pnl, 'Reason': exit_reason
                        })
                        add_log(f"EXIT {exit_reason} @ {exit_p:.2f} | PnL: {pnl:.2f}")
                        reset_position_state()
                
                # 3. Update UI (Flicker Free using Placeholders)
                
                with metrics_ph.container():
                    # Top Row Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Price", f"{curr_price:.2f}", f"{curr_price - df.iloc[-2]['Close']:.2f}")
                    m2.metric("RSI", f"{curr['RSI']:.1f}")
                    m3.metric("EMA 9", f"{curr['EMA_Fast']:.2f}")
                    m4.metric("EMA 15", f"{curr['EMA_Slow']:.2f}")
                    
                    # Position Status
                    if st.session_state['position']:
                        p = st.session_state['position']
                        type_str = "LONG" if p['type'] == 1 else "SHORT"
                        unrealized = (curr_price - p['entry_price']) * p['quantity'] if p['type'] == 1 else (p['entry_price'] - curr_price) * p['quantity']
                        
                        st.markdown("### 🟢 Active Position")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Type", type_str)
                        c2.metric("Entry", f"{p['entry_price']:.2f}")
                        c3.metric("Stop Loss", f"{p['sl']:.2f}", help="Updates if trailing")
                        c4.metric("Unrealized PnL", f"{unrealized:.2f}", delta=f"{unrealized:.2f}")
                    else:
                        st.markdown("### ⚪ Waiting for Signal")

                with chart_ph.container():
                    # Live Chart
                    fig = go.Figure()
                    # Only show last 50 candles for performance
                    d_plot = df.tail(50)
                    fig.add_trace(go.Candlestick(x=d_plot.index, open=d_plot['Open'], high=d_plot['High'], low=d_plot['Low'], close=d_plot['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=d_plot.index, y=d_plot['EMA_Fast'], line=dict(color='blue', width=1), name='EMA Fast'))
                    fig.add_trace(go.Scatter(x=d_plot.index, y=d_plot['EMA_Slow'], line=dict(color='orange', width=1), name='EMA Slow'))
                    
                    # Visualize SL if active
                    if st.session_state['position']:
                        sl_line = [st.session_state['position']['sl']] * len(d_plot)
                        fig.add_trace(go.Scatter(x=d_plot.index, y=sl_line, line=dict(color='red', dash='dot'), name='SL Level'))

                    fig.update_layout(
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=400,
                        xaxis_rangeslider_visible=False,
                        title=f"Live Feed: {config['ticker']}"
                    )
                    # Use a static key to prevent full DOM reload flicker, Plotly handles data updates internally
                    st.plotly_chart(fig, use_container_width=True, key="live_chart_widget")

                # Sleep to prevent API Rate Limits
                time.sleep(1.5)

    # --- TAB 2: BACKTEST ANALYSIS ---
    with tab2:
        st.subheader("Strategy Backtest")
        if st.button("Run Backtest"):
            with st.spinner("Simulating..."):
                df_bt = fetch_data(config['ticker'], config['interval'], "1mo") # Hardcoded larger period for BT
                if df_bt is not None:
                    # Simulation Logic (Simplified Loop)
                    trades = []
                    pos = None
                    
                    for i in range(20, len(df_bt)):
                        # Slice data up to i
                        # Optimization: Use vector calc in production, loop for logic clarity here
                        d_slice = df_bt.iloc[:i+1]
                        c_candle = df_bt.iloc[i]
                        
                        if pos is None:
                            s, _ = check_signals(d_slice, config)
                            if s != 0:
                                sl = config['sl_points']
                                if config['sl_type'] == "ATR Based": sl = c_candle['ATR'] * config['sl_points']
                                
                                entry = c_candle['Close']
                                sl_p = entry - sl if s == 1 else entry + sl
                                tp_p = entry + config['target_points'] if s == 1 else entry - config['target_points']
                                
                                pos = {'type': s, 'entry': entry, 'sl': sl_p, 'tp': tp_p, 'idx': i}
                        else:
                            # Exit Logic
                            exit_p = None
                            reason = ""
                            
                            # Check High/Low for hits within candle
                            if pos['type'] == 1:
                                if c_candle['Low'] <= pos['sl']:
                                    exit_p = pos['sl']; reason="SL"
                                elif c_candle['High'] >= pos['tp']:
                                    exit_p = pos['tp']; reason="TP"
                            else:
                                if c_candle['High'] >= pos['sl']:
                                    exit_p = pos['sl']; reason="SL"
                                elif c_candle['Low'] <= pos['tp']:
                                    exit_p = pos['tp']; reason="TP"
                                    
                            if exit_p:
                                pnl = (exit_p - pos['entry']) if pos['type'] == 1 else (pos['entry'] - exit_p)
                                trades.append({
                                    'Date': d_slice.index[-1],
                                    'Type': "LONG" if pos['type']==1 else "SHORT",
                                    'Entry': pos['entry'], 'Exit': exit_p,
                                    'PnL': pnl * config['qty'],
                                    'Result': "Win" if pnl > 0 else "Loss"
                                })
                                pos = None
                    
                    # RESULTS
                    if trades:
                        res_df = pd.DataFrame(trades)
                        
                        # Summary Metrics
                        tot_trades = len(res_df)
                        wins = len(res_df[res_df['PnL'] > 0])
                        win_rate = (wins/tot_trades)*100
                        total_pnl = res_df['PnL'].sum()
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Trades", tot_trades)
                        m2.metric("Accuracy", f"{win_rate:.1f}%")
                        m3.metric("Total PnL", f"{total_pnl:.2f}", delta=total_pnl)
                        m4.metric("Best Trade", f"{res_df['PnL'].max():.2f}")
                        
                        st.markdown("### Trade Log")
                        st.dataframe(res_df.style.applymap(lambda x: 'color: green' if x > 0 else 'color: red', subset=['PnL']), use_container_width=True)
                    else:
                        st.warning("No trades found in backtest period.")
                else:
                    st.error("Could not fetch data.")

    # --- TAB 3: HISTORY ---
    with tab3:
        if st.session_state['trade_history']:
            hist_df = pd.DataFrame(st.session_state['trade_history'])
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("No trades yet.")
        
        st.markdown("#### System Logs")
        for log in reversed(st.session_state['trade_logs']):
            st.text(log)

if __name__ == "__main__":
    main()
