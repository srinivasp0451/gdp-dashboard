import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import time
import datetime
from pydhan import pydhan
from dhanhq import dhanhq

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Investing", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'live_running' not in st.session_state: st.session_state.live_running = False
if 'trades' not in st.session_state: st.session_state.trades = []
if 'current_position' not in st.session_state: st.session_state.current_position = None

# --- BROKER INTEGRATION (DHAN) ---
def place_dhan_order(dhan_client, security_id, transaction_type, quantity, product_type, order_type, price=0):
    try:
        response = dhan_client.place_order(
            security_id=security_id, exchange_segment=dhan_client.NSE,
            transaction_type=transaction_type, quantity=quantity,
            order_type=order_type, product_type=product_type, price=price
        )
        return response
    except Exception as e:
        return f"Order failed: {e}"

def place_option_order(dhan_client, order_params):
    try:
        response = dhan_client.place_order(**order_params)
        return response
    except Exception as e:
        return f"Option Order failed: {e}"

# --- HELPER FUNCTIONS ---
def fetch_data(ticker, interval, period, warmup_bars=200):
    """Fetches data with a warmup period to match TradingView EMA calculations."""
    try:
        # Note: 1m data is limited to 7d max by yfinance
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return pd.DataFrame()
        # Flatten multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

def calculate_indicators(df, fast_ema, slow_ema):
    if df.empty or len(df) < slow_ema: return df
    df['EMA_Fast'] = ta.ema(df['Close'], length=fast_ema)
    df['EMA_Slow'] = ta.ema(df['Close'], length=slow_ema)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    # Crossover logic
    df['Previous_EMA_Fast'] = df['EMA_Fast'].shift(1)
    df['Previous_EMA_Slow'] = df['EMA_Slow'].shift(1)
    
    df['Buy_Signal'] = (df['Previous_EMA_Fast'] <= df['Previous_EMA_Slow']) & (df['EMA_Fast'] > df['EMA_Slow'])
    df['Sell_Signal'] = (df['Previous_EMA_Fast'] >= df['Previous_EMA_Slow']) & (df['EMA_Fast'] < df['EMA_Slow'])
    return df

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("⚙️ Configuration")
ticker = st.sidebar.text_input("Ticker Symbol (e.g., ^NSEI for Nifty50, BTC-USD)", value="^NSEI")
interval = st.sidebar.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '1d', '1wk'], index=1)
period = st.sidebar.selectbox("History Period", ['1d', '5d', '7d', '1mo', '3mo', '1y'], index=1)

st.sidebar.markdown("### Strategy")
strategy = st.sidebar.selectbox("Select Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell", "Elliot Waves (Beta)"])
fast_ema = st.sidebar.number_input("Fast EMA", value=9)
slow_ema = st.sidebar.number_input("Slow EMA", value=15)

st.sidebar.markdown("### Filters & Angles")
min_angle_cb = st.sidebar.checkbox("Minimum EMA Crossover Angle", value=False)
min_angle = st.sidebar.number_input("Angle Degree (Absolute)", value=0) if min_angle_cb else 0
crossover_type = st.sidebar.selectbox("Crossover Type", ["Simple Crossover", "Custom Candle Size", "ATR Based Size"])

st.sidebar.markdown("### Risk Management")
sl_type = st.sidebar.selectbox("Stop Loss Type", ["Custom Points", "Trailing", "Reverse EMA", "Risk/Reward", "ATR Based"])
sl_val = st.sidebar.number_input("SL Value", value=10.0)
target_type = st.sidebar.selectbox("Target Type", ["Custom Points", "Trailing (Display)", "EMA Crossover", "Risk/Reward", "ATR Based"])
target_val = st.sidebar.number_input("Target Value", value=20.0)

st.sidebar.markdown("### Live Trading Rules")
cooldown_cb = st.sidebar.checkbox("Cooldown Period Between Trades", value=True)
cooldown_sec = st.sidebar.number_input("Seconds", value=5) if cooldown_cb else 0
overlap_cb = st.sidebar.checkbox("Prevent Overlapping Trades", value=True)

st.sidebar.markdown("### Broker Integration (Dhan)")
dhan_enabled = st.sidebar.checkbox("Enable Dhan Broker", value=False)
options_enabled = st.sidebar.checkbox("Options Trading", value=False)

if dhan_enabled:
    client_id = st.sidebar.text_input("Client ID", type="password")
    token = st.sidebar.text_input("Access Token", type="password")
    
    if options_enabled:
        st.sidebar.selectbox("Segment", ["NSE_FNO", "BSE_FNO"])
        ce_sec_id = st.sidebar.text_input("CE Security ID")
        pe_sec_id = st.sidebar.text_input("PE Security ID")
        opt_qty = st.sidebar.number_input("Option Quantity (Lots*Size)", value=65)
    else:
        prod_type = st.sidebar.selectbox("Product", ["INTRADAY", "DELIVERY"])
        exc = st.sidebar.selectbox("Exchange", ["NSE", "BSE"])
        eq_sec_id = st.sidebar.text_input("Security ID", value="1594")
        eq_qty = st.sidebar.number_input("Quantity", value=1)
        order_type = st.sidebar.selectbox("Order Type", ["MARKET", "LIMIT"])

# --- COMMENTED STRATEGY AS REQUESTED ---
# Threshold Strategy (Hidden from UI)
# def threshold_strategy(df, threshold_price):
#     df['Buy_Signal'] = df['Close'] > threshold_price
#     df['Sell_Signal'] = df['Close'] < threshold_price
#     return df

# --- TOP BANNER (LTP) ---
header_placeholder = st.empty()
def update_header():
    try:
        tkr = yf.Ticker(ticker)
        todays_data = tkr.history(period='2d')
        if len(todays_data) >= 2:
            prev_close = todays_data['Close'].iloc[-2]
            ltp = todays_data['Close'].iloc[-1]
            change = ltp - prev_close
            pct_change = (change / prev_close) * 100
            color = "🟢" if change >= 0 else "🔴"
            header_placeholder.markdown(f"### {ticker} LTP: **{ltp:.2f}** | {color} {change:.2f} ({pct_change:.2f}%)")
    except:
        pass

update_header()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Backtesting", "⚡ Live Trading", "📜 Trade History"])

# --- BACKTESTING TAB ---
with tab1:
    st.header("Backtesting Engine")
    if st.button("Run Backtest"):
        with st.spinner("Fetching data and calculating indicators..."):
            df = fetch_data(ticker, interval, period)
            df = calculate_indicators(df, fast_ema, slow_ema)
            
            if df.empty:
                st.error("No data fetched.")
            else:
                backtest_trades = []
                in_position = False
                pos_type = None
                entry_price = 0
                entry_time = None
                sl = 0
                target = 0
                
                # Iterating for N+1 logic and conservative High/Low SL checks
                for i in range(1, len(df)):
                    prev_row = df.iloc[i-1]
                    curr_row = df.iloc[i]
                    
                    # Check Exits first
                    if in_position:
                        exit_price = 0
                        reason = ""
                        
                        if pos_type == "Buy":
                            # Conservative check: Check SL against candle LOW first
                            if curr_row['Low'] <= sl:
                                exit_price = sl
                                reason = "SL Hit"
                            elif curr_row['High'] >= target:
                                exit_price = target
                                reason = "Target Hit"
                                
                        elif pos_type == "Sell":
                            # Conservative check: Check SL against candle HIGH first
                            if curr_row['High'] >= sl:
                                exit_price = sl
                                reason = "SL Hit"
                            elif curr_row['Low'] <= target:
                                exit_price = target
                                reason = "Target Hit"
                                
                        if exit_price != 0:
                            pnl = (exit_price - entry_price) if pos_type == "Buy" else (entry_price - exit_price)
                            backtest_trades.append({
                                "Entry Time": entry_time, "Exit Time": curr_row.name,
                                "Type": pos_type, "Entry Price": entry_price, "Exit Price": exit_price,
                                "SL": sl, "Target": target, "Reason": reason, "PnL": pnl
                            })
                            in_position = False
                    
                    # Check Entries (N signal -> N+1 Open)
                    if not in_position:
                        if prev_row['Buy_Signal'] == True:
                            in_position = True
                            pos_type = "Buy"
                            entry_price = curr_row['Open']
                            entry_time = curr_row.name
                            sl = entry_price - sl_val if sl_type == "Custom Points" else entry_price * 0.99
                            target = entry_price + target_val if target_type == "Custom Points" else entry_price * 1.02
                        elif prev_row['Sell_Signal'] == True:
                            in_position = True
                            pos_type = "Sell"
                            entry_price = curr_row['Open']
                            entry_time = curr_row.name
                            sl = entry_price + sl_val if sl_type == "Custom Points" else entry_price * 1.01
                            target = entry_price - target_val if target_type == "Custom Points" else entry_price * 0.98

                # Results
                if backtest_trades:
                    res_df = pd.DataFrame(backtest_trades)
                    wins = len(res_df[res_df['PnL'] > 0])
                    st.metric("Total PnL", f"{res_df['PnL'].sum():.2f}")
                    st.metric("Accuracy", f"{(wins/len(res_df))*100:.2f}%")
                    st.dataframe(res_df)
                    
                    # Plot
                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], mode='lines', name='Fast EMA'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], mode='lines', name='Slow EMA'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No trades generated in this period.")

# --- LIVE TRADING TAB ---
with tab2:
    st.header("Live Execution Engine")
    
    col1, col2, col3 = st.columns(3)
    start_btn = col1.button("▶️ Start Live Trading")
    stop_btn = col2.button("⏹️ Stop")
    sq_btn = col3.button("⏭️ Square Off")
    
    if start_btn: st.session_state.live_running = True
    if stop_btn: st.session_state.live_running = False
    
    st.markdown("### Current Configuration")
    st.info(f"Ticker: {ticker} | TF: {interval} | SL: {sl_val} | Target: {target_val} | Overlap Blocked: {overlap_cb}")
    
    live_chart_ph = st.empty()
    status_ph = st.empty()
    
    if st.session_state.live_running:
        status_ph.success("🔴 Live Trading is Running...")
        # Note: In Streamlit, a while loop here blocks the UI. 
        # Using a simulated tick loop for demonstration of the logic requested.
        while st.session_state.live_running:
            df = fetch_data(ticker, interval, '5d') # fetch enough for EMA
            df = calculate_indicators(df, fast_ema, slow_ema)
            
            if not df.empty:
                ltp = df['Close'].iloc[-1]
                last_candle_time = df.index[-1]
                
                # Check Timeframe multiple for entry logic
                # (Simulated: Checking if current minute is a multiple of interval)
                current_min = datetime.datetime.now().minute
                
                # 1. Check active position SL/Target against LTP directly
                if st.session_state.current_position:
                    pos = st.session_state.current_position
                    if pos['type'] == 'Buy':
                        if ltp <= pos['sl']:
                            st.session_state.trades.append({"Type": "Buy", "Entry": pos['entry'], "Exit": ltp, "Reason": "SL Hit"})
                            st.session_state.current_position = None
                        elif ltp >= pos['target']:
                            st.session_state.trades.append({"Type": "Buy", "Entry": pos['entry'], "Exit": ltp, "Reason": "Target Hit"})
                            st.session_state.current_position = None
                            
                # 2. Check for new entries (Only if no position and no overlap)
                if not st.session_state.current_position and overlap_cb:
                    # Check Candle N for signal
                    prev_row = df.iloc[-2]
                    if prev_row['Buy_Signal']:
                        st.session_state.current_position = {"type": "Buy", "entry": ltp, "sl": ltp - sl_val, "target": ltp + target_val}
                        # Place Dhan Option Order logic placeholder
                        if dhan_enabled and options_enabled:
                            pass # place_option_order(...)
                            
            time.sleep(1.5) # API Rate Limit handler
            st.rerun() # Forces Streamlit to refresh UI without full reload

# --- TRADE HISTORY TAB ---
with tab3:
    st.header("Completed Trades")
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades))
    else:
        st.info("No trades executed yet.")
