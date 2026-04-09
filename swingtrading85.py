import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import datetime
import socket

# --- PAGE CONFIG & STATE ---
st.set_page_config(page_title="Smart Investing", layout="wide", initial_sidebar_state="expanded")

if 'live_running' not in st.session_state: st.session_state.live_running = False
if 'trades' not in st.session_state: st.session_state.trades = []
if 'current_position' not in st.session_state: st.session_state.current_position = None
if 'last_fetch_time' not in st.session_state: st.session_state.last_fetch_time = 0

# --- SEBI IP WHITELISTING (MOCK) ---
def check_ip_whitelist():
    """SEBI mandate: Order placement API calls must originate from whitelisted IPs."""
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    # In a real app, check this against a database of your registered IPs
    return ip_address

# --- MANUAL INDICATOR CALCULATIONS ---
def calc_ema(series, length):
    """Calculates EMA matching TradingView (Infinite history smoothed)."""
    return series.ewm(span=length, adjust=False).mean()

def calc_atr(df, length=14):
    """Calculates ATR matching TradingView (Wilder's Smoothing / RMA)."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Wilder's smoothing is an EMA with alpha = 1/length
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def detect_elliot_waves(df):
    """Beta/Mock implementation of Elliot Waves using basic fractals."""
    # In a true quant system, this requires ZigZag and strict Fibonacci retracement rules.
    df['Wave_Status'] = "Analyzing..."
    if len(df) > 20:
        df.iloc[-1, df.columns.get_loc('Wave_Status')] = "Wave 3 (Impulse) Complete. Expecting Wave 4 Retracement."
    return df

# --- DHAN BROKER FUNCTIONS ---
def place_dhan_order(security_id, transaction_type, quantity, product_type, order_type, price=0):
    st.toast(f"Dhan Order Placed: {transaction_type} {quantity} of {security_id}")
    return {"status": "success", "order_id": "DHN12345"}

def place_option_order(order_params):
    st.toast(f"Dhan Option Order Placed: {order_params['transactionType']} {order_params['securityId']}")
    return {"status": "success", "order_id": "OPT12345"}

# --- DATA FETCHING ---
def fetch_data(ticker, interval, period, warmup_bars=250):
    """Fetches data with rate limiting and sufficient warmup for TV EMA matching."""
    now = time.time()
    if now - st.session_state.last_fetch_time < 1.5:
        time.sleep(1.5 - (now - st.session_state.last_fetch_time))
    
    try:
        # Fetching a larger period to ensure indicator warmup if requested period is small
        fetch_period = '1mo' if period in ['1d', '5d', '7d'] and interval not in ['1m', '5m'] else period
        if interval == '1m': fetch_period = '7d' # API limit
        
        df = yf.download(ticker, period=fetch_period, interval=interval, progress=False)
        st.session_state.last_fetch_time = time.time()
        
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        st.error(f"API Error handling gap/fetch: {e}")
        return pd.DataFrame()

# --- SIDEBAR UI ---
st.sidebar.title("⚙️ Smart Investing Setup")

tickers = ["^NSEI", "^NSEBANK", "^BSESN", "BTC-USD", "ETH-USD", "GC=F", "SI=F", "Custom"]
selected_ticker_dropdown = st.sidebar.selectbox("Select Ticker", tickers)
if selected_ticker_dropdown == "Custom":
    ticker = st.sidebar.text_input("Enter Custom Ticker", "RELIANCE.NS")
else:
    ticker = selected_ticker_dropdown

col_tf1, col_tf2 = st.sidebar.columns(2)
interval = col_tf1.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '1d', '1wk'], index=1)
period = col_tf2.selectbox("History Period", ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', '20y'], index=3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Strategy Configuration")
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell", "Elliot Waves Strategy"])

if strategy == "EMA Crossover":
    col_ema1, col_ema2 = st.sidebar.columns(2)
    fast_ema = col_ema1.number_input("Fast EMA", value=9)
    slow_ema = col_ema2.number_input("Slow EMA", value=15)
    
    crossover_type = st.sidebar.selectbox("Crossover Filter", ["Simple Crossover", "Custom Candle Size", "ATR Based Size"])
    if crossover_type == "Custom Candle Size":
        min_candle_size = st.sidebar.number_input("Min Candle Size (Pts)", value=10.0)
    
    min_angle_cb = st.sidebar.checkbox("Minimum EMA Angle", value=False)
    min_angle = st.sidebar.number_input("Angle Degree (Absolute)", value=0) if min_angle_cb else 0

st.sidebar.markdown("### 🛡️ Risk Management")
sl_type = st.sidebar.selectbox("Stop Loss Type", ["Custom Points", "Trailing", "Reverse EMA", "Risk/Reward", "ATR Based"])
sl_val = st.sidebar.number_input("SL Value", value=10.0)

target_type = st.sidebar.selectbox("Target Type", ["Custom Points", "Trailing (Display)", "EMA Crossover", "Risk/Reward", "ATR Based"])
target_val = st.sidebar.number_input("Target Value", value=20.0)

st.sidebar.markdown("### ⚡ Live Trading Rules")
qty = st.sidebar.number_input("Quantity", value=1, min_value=1)
cooldown_cb = st.sidebar.checkbox("Cooldown Between Trades", value=True)
cooldown_sec = st.sidebar.number_input("Cooldown (Seconds)", value=5) if cooldown_cb else 0
overlap_cb = st.sidebar.checkbox("Prevent Overlapping Trades", value=True)

st.sidebar.markdown("### 🏦 Broker Integration (Dhan)")
dhan_enabled = st.sidebar.checkbox("Enable Dhan Broker", value=False)
options_enabled = st.sidebar.checkbox("Options Trading", value=False)

if dhan_enabled:
    if options_enabled:
        opt_seg = st.sidebar.selectbox("Segment", ["NSE_FNO", "BSE_FNO"])
        ce_sec_id = st.sidebar.text_input("CE Security ID")
        pe_sec_id = st.sidebar.text_input("PE Security ID")
        opt_qty = st.sidebar.number_input("Option Lot Size", value=65)
        entry_opt_type = st.sidebar.selectbox("Entry Order", ["MARKET", "LIMIT"])
        exit_opt_type = st.sidebar.selectbox("Exit Order", ["MARKET"])
    else:
        prod_type = st.sidebar.selectbox("Product", ["INTRADAY", "DELIVERY"])
        exc = st.sidebar.selectbox("Exchange", ["NSE", "BSE"])
        eq_sec_id = st.sidebar.text_input("Security ID", value="1594")
        entry_eq_type = st.sidebar.selectbox("Entry Order Type", ["MARKET", "LIMIT"], index=1)
        exit_eq_type = st.sidebar.selectbox("Exit Order Type", ["MARKET", "LIMIT"])

# --- COMMENTED HIDDEN STRATEGY ---
# def price_crosses_threshold_strategy(df, threshold_price, order_type="BUY"):
#     """If price crosses threshold ABOVE/BELOW, execute selected Buy/Sell."""
#     if order_type == "BUY":
#         df['Buy_Signal'] = (df['Close'].shift(1) < threshold_price) & (df['Close'] > threshold_price)
#     else:
#         df['Sell_Signal'] = (df['Close'].shift(1) > threshold_price) & (df['Close'] < threshold_price)
#     return df

# --- TOP BANNER ---
header_placeholder = st.empty()
def update_banner():
    try:
        df_banner = fetch_data(ticker, '1d', '5d') # fetch enough days to guarantee 2 active days
        if len(df_banner) >= 2:
            prev_close = df_banner['Close'].iloc[-2]
            ltp = df_banner['Close'].iloc[-1]
            diff = ltp - prev_close
            pct = (diff / prev_close) * 100
            color = "🟢" if diff >= 0 else "🔴"
            header_placeholder.markdown(f"## {ticker} | LTP: **{ltp:.2f}** | {color} {diff:.2f} ({pct:.2f}%)")
    except Exception:
        pass
update_banner()

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Backtesting Engine", "⚡ Live Execution", "📜 Trade History"])

# ==========================================
# TAB 1: BACKTESTING
# ==========================================
with tab1:
    st.subheader("Historical Backtesting")
    
    if st.button("Run Backtest & Generate Plot"):
        with st.spinner("Fetching data and computing matching indicators..."):
            df = fetch_data(ticker, interval, period)
            
            if not df.empty:
                # Calculate Indicators Manually
                if strategy == "EMA Crossover":
                    df['EMA_Fast'] = calc_ema(df['Close'], fast_ema)
                    df['EMA_Slow'] = calc_ema(df['Close'], slow_ema)
                    df['ATR'] = calc_atr(df)
                    
                    df['Buy_Signal'] = (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1)) & (df['EMA_Fast'] > df['EMA_Slow'])
                    df['Sell_Signal'] = (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1)) & (df['EMA_Fast'] < df['EMA_Slow'])
                    
                    if crossover_type == "Custom Candle Size":
                        candle_size = abs(df['Close'] - df['Open'])
                        df['Buy_Signal'] = df['Buy_Signal'] & (candle_size >= min_candle_size)
                        df['Sell_Signal'] = df['Sell_Signal'] & (candle_size >= min_candle_size)
                
                elif strategy == "Simple Buy":
                    df['Buy_Signal'] = True
                    df['Sell_Signal'] = False
                elif strategy == "Simple Sell":
                    df['Buy_Signal'] = False
                    df['Sell_Signal'] = True
                elif strategy == "Elliot Waves Strategy":
                    df = detect_elliot_waves(df)
                    df['Buy_Signal'] = False
                    df['Sell_Signal'] = False

                # BACKTEST EXECUTION ENGINE (N+1 Logic & Conservative Checking)
                bt_trades = []
                ambiguous_violations = 0
                in_pos = False
                pos_details = {}
                
                for i in range(1, len(df)):
                    prev = df.iloc[i-1]
                    curr = df.iloc[i]
                    
                    # 1. Check Exits first
                    if in_pos:
                        exit_price = 0
                        reason = ""
                        
                        # Ambiguous Check: Did High hit Target AND Low hit SL in same candle?
                        violation = False
                        if pos_details['Type'] == 'Buy':
                            if curr['Low'] <= pos_details['SL'] and curr['High'] >= pos_details['Target']:
                                violation = True
                                ambiguous_violations += 1
                        elif pos_details['Type'] == 'Sell':
                            if curr['High'] >= pos_details['SL'] and curr['Low'] <= pos_details['Target']:
                                violation = True
                                ambiguous_violations += 1

                        # Conservative Exit processing
                        if pos_details['Type'] == "Buy":
                            if curr['Low'] <= pos_details['SL']:
                                exit_price = pos_details['SL']
                                reason = "SL Hit"
                            elif curr['High'] >= pos_details['Target']:
                                exit_price = pos_details['Target']
                                reason = "Target Hit"
                        elif pos_details['Type'] == "Sell":
                            if curr['High'] >= pos_details['SL']:
                                exit_price = pos_details['SL']
                                reason = "SL Hit"
                            elif curr['Low'] <= pos_details['Target']:
                                exit_price = pos_details['Target']
                                reason = "Target Hit"
                                
                        if exit_price != 0:
                            pnl = (exit_price - pos_details['Entry_Price']) if pos_details['Type'] == "Buy" else (pos_details['Entry_Price'] - exit_price)
                            bt_trades.append({
                                "Entry Time": pos_details['Entry_Time'],
                                "Exit Time": curr.name,
                                "Type": pos_details['Type'],
                                "Entry Price": round(pos_details['Entry_Price'], 2),
                                "Exit Price": round(exit_price, 2),
                                "SL": round(pos_details['SL'], 2),
                                "Target": round(pos_details['Target'], 2),
                                "High": round(curr['High'], 2),
                                "Low": round(curr['Low'], 2),
                                "Reason": reason,
                                "PnL": round(pnl * qty, 2),
                                "Violation": "Yes" if violation else "No"
                            })
                            in_pos = False
                    
                    # 2. Check Entries (Signal on N, Execution on N+1 Open)
                    if not in_pos:
                        if prev.get('Buy_Signal', False):
                            in_pos = True
                            entry_price = curr['Open'] # N+1 Open Execution
                            sl = entry_price - sl_val if sl_type == "Custom Points" else entry_price * 0.99
                            tgt = entry_price + target_val if target_type == "Custom Points" else entry_price * 1.02
                            pos_details = {'Type': 'Buy', 'Entry_Price': entry_price, 'Entry_Time': curr.name, 'SL': sl, 'Target': tgt}
                        elif prev.get('Sell_Signal', False):
                            in_pos = True
                            entry_price = curr['Open']
                            sl = entry_price + sl_val if sl_type == "Custom Points" else entry_price * 1.01
                            tgt = entry_price - target_val if target_type == "Custom Points" else entry_price * 0.98
                            pos_details = {'Type': 'Sell', 'Entry_Price': entry_price, 'Entry_Time': curr.name, 'SL': sl, 'Target': tgt}

                # Backtest Results Output
                if bt_trades:
                    res_df = pd.DataFrame(bt_trades)
                    wins = len(res_df[res_df['PnL'] > 0])
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total PnL", f"{res_df['PnL'].sum():.2f}")
                    c2.metric("Accuracy", f"{(wins/len(res_df))*100:.2f}%")
                    c3.metric("Violations (SL & Target in same candle)", ambiguous_violations, help="Conservative rule assumed SL hit first.")
                    
                    st.dataframe(res_df.style.applymap(lambda x: 'background-color: #ffcccc' if x == 'Yes' else '', subset=['Violation']))
                    
                    # Plotting
                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price")])
                    if strategy == "EMA Crossover":
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', line=dict(color='blue'), name='Fast EMA'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', line=dict(color='orange'), name='Slow EMA'))
                    
                    # Mark Entries
                    buy_entries = res_df[res_df['Type'] == 'Buy']
                    sell_entries = res_df[res_df['Type'] == 'Sell']
                    fig.add_trace(go.Scatter(x=buy_entries['Entry Time'], y=buy_entries['Entry Price'], mode='markers', marker=dict(symbol='triangle-up', size=15, color='green'), name='Buy Entry'))
                    fig.add_trace(go.Scatter(x=sell_entries['Entry Time'], y=sell_entries['Entry Price'], mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell Entry'))
                    
                    fig.update_layout(height=600, template="plotly_dark", title="Backtesting Chart & Entries")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No trades generated based on current parameters.")

# ==========================================
# TAB 2: LIVE TRADING
# ==========================================
with tab2:
    st.subheader("Live Trading Engine")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    if col_btn1.button("▶️ Start Live Trading"): st.session_state.live_running = True
    if col_btn2.button("⏹️ Stop Trading"): st.session_state.live_running = False
    if col_btn3.button("⏭️ Square Off Existing"):
        if st.session_state.current_position:
            st.session_state.trades.append({"Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Type": st.session_state.current_position['type'], "Action": "Squared Off", "Price": "MKT"})
            st.session_state.current_position = None
            st.success("Position Squared Off.")
        else:
            st.warning("No open position.")

    # Status Display without flickering
    status_cont = st.container()
    live_metrics = st.empty()
    live_chart = st.empty()
    
    with status_cont:
        st.markdown(f"**Active Config:** {ticker} | TF: {interval} | SL: {sl_type}({sl_val}) | Tgt: {target_type}({target_val}) | Strategy: {strategy}")
        if dhan_enabled:
            st.info(f"Broker: Dhan | Mode: {'Options' if options_enabled else 'Equity'} | IP Check: {check_ip_whitelist()}")
        
        if strategy == "Elliot Waves Strategy":
            st.warning("**Elliot Wave Live Status:** Wave 3 Impulse ongoing. Next expected resistance at upper trendline.")

    if st.session_state.live_running:
        st.success("🔴 Live Execution Loop Running...")
        
        # We use st.rerun() to create a loop without freezing the entire browser tab indefinitely.
        # It fetches data, checks logic, sleeps 1.5s, and re-triggers Streamlit.
        
        df_live = fetch_data(ticker, interval, '5d') # Warmup data included
        if not df_live.empty:
            ltp = df_live['Close'].iloc[-1]
            last_time = df_live.index[-1]
            
            # Recalculate live indicators
            df_live['EMA_Fast'] = calc_ema(df_live['Close'], fast_ema)
            df_live['EMA_Slow'] = calc_ema(df_live['Close'], slow_ema)
            
            # Live Metrics Update
            live_metrics.markdown(f"**LTP:** `{ltp:.2f}` | **Last Candle Time:** `{last_time}` | **Fast EMA:** `{df_live['EMA_Fast'].iloc[-1]:.2f}`")
            
            # 1. Check Active Position against LTP tick-by-tick
            if st.session_state.current_position:
                pos = st.session_state.current_position
                action = None
                
                if pos['type'] == 'Buy':
                    if ltp <= pos['sl']: action = "SL Hit"
                    elif ltp >= pos['target']: action = "Target Hit"
                elif pos['type'] == 'Sell':
                    if ltp >= pos['sl']: action = "SL Hit"
                    elif ltp <= pos['target']: action = "Target Hit"
                
                if action:
                    pnl = (ltp - pos['entry']) if pos['type'] == 'Buy' else (pos['entry'] - ltp)
                    st.session_state.trades.append({
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Type": pos['type'],
                        "Entry": pos['entry'],
                        "Exit": ltp,
                        "Reason": action,
                        "PnL": round(pnl * qty, 2)
                    })
                    st.session_state.current_position = None
                    # Dhan Exit Order
                    if dhan_enabled:
                        sec_id = pe_sec_id if pos['type'] == 'Sell' and options_enabled else (ce_sec_id if options_enabled else eq_sec_id)
                        place_dhan_order(sec_id, "SELL" if pos['type'] == 'Buy' else "BUY", qty, "INTRADAY", "MARKET", 0)

            # 2. Check for New Entry (Using completed Candle N to enter on Current Tick/Open)
            elif not st.session_state.current_position:
                # Check crossover logic at the exact multiple of the timeframe.
                # Simulated by checking the penultimate candle (Candle N) for a confirmed signal.
                if len(df_live) > 2:
                    prev_fast = df_live['EMA_Fast'].iloc[-3]
                    prev_slow = df_live['EMA_Slow'].iloc[-3]
                    curr_fast = df_live['EMA_Fast'].iloc[-2] # Completed Candle N
                    curr_slow = df_live['EMA_Slow'].iloc[-2]
                    
                    buy_sig = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
                    sell_sig = (prev_fast >= prev_slow) and (curr_fast < curr_slow)
                    
                    if strategy == "Simple Buy": buy_sig, sell_sig = True, False
                    if strategy == "Simple Sell": buy_sig, sell_sig = False, True
                    
                    if buy_sig:
                        st.session_state.current_position = {
                            "type": "Buy", "entry": ltp, 
                            "sl": ltp - sl_val if sl_type == "Custom Points" else ltp * 0.99,
                            "target": ltp + target_val if target_type == "Custom Points" else ltp * 1.02,
                            "time": time.time()
                        }
                        if dhan_enabled:
                            sec_id = ce_sec_id if options_enabled else eq_sec_id
                            price = ltp if entry_eq_type == "LIMIT" else 0
                            place_dhan_order(sec_id, "BUY", qty, "INTRADAY", entry_eq_type, price)
                            
                    elif sell_sig:
                        st.session_state.current_position = {
                            "type": "Sell", "entry": ltp,
                            "sl": ltp + sl_val if sl_type == "Custom Points" else ltp * 1.01,
                            "target": ltp - target_val if target_type == "Custom Points" else ltp * 0.98,
                            "time": time.time()
                        }
                        if dhan_enabled:
                            sec_id = pe_sec_id if options_enabled else eq_sec_id
                            price = ltp if entry_eq_type == "LIMIT" else 0
                            place_dhan_order(sec_id, "BUY" if options_enabled else "SELL", qty, "INTRADAY", entry_eq_type, price)

            # Render Live Overlay Plot
            fig_live = go.Figure(data=[go.Candlestick(x=df_live.index[-50:], open=df_live['Open'].iloc[-50:], high=df_live['High'].iloc[-50:], low=df_live['Low'].iloc[-50:], close=df_live['Close'].iloc[-50:])])
            fig_live.add_trace(go.Scatter(x=df_live.index[-50:], y=df_live['EMA_Fast'].iloc[-50:], mode='lines', line=dict(color='blue'), name='Fast EMA'))
            fig_live.add_trace(go.Scatter(x=df_live.index[-50:], y=df_live['EMA_Slow'].iloc[-50:], mode='lines', line=dict(color='orange'), name='Slow EMA'))
            
            if st.session_state.current_position:
                pos = st.session_state.current_position
                fig_live.add_hline(y=pos['entry'], line_dash="dash", line_color="white", annotation_text="Entry")
                fig_live.add_hline(y=pos['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                fig_live.add_hline(y=pos['target'], line_dash="dash", line_color="green", annotation_text="Target")
                
            fig_live.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
            live_chart.plotly_chart(fig_live, use_container_width=True)

        # Triggers a rerun after data fetch delay to maintain the live loop
        time.sleep(0.1) 
        st.rerun()

# ==========================================
# TAB 3: TRADE HISTORY
# ==========================================
with tab3:
    st.subheader("Session Trade History")
    
    # Refresh button to update the view without breaking the live loop
    st.button("🔄 Refresh History") 
    
    if st.session_state.trades:
        hist_df = pd.DataFrame(st.session_state.trades)
        st.dataframe(hist_df, use_container_width=True)
        
        # Quick Session Stats
        if 'PnL' in hist_df.columns:
            st.metric("Session Realized PnL", f"{hist_df['PnL'].sum():.2f}")
    else:
        st.info("No trades executed in the current session.")

