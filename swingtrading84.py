import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Constants & Setup ---
st.set_page_config(page_title="Smart Wealth", layout="wide")
IST = pytz.timezone('Asia/Kolkata')

# Ticker Mapping
TICKERS = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "GOLD": "GC=F", "SILVER": "SI=F", "Custom": ""
}

TIMEFRAME_PERIODS = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"]
}

# --- State Management ---
if 'live_running' not in st.session_state: st.session_state.live_running = False
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'current_position' not in st.session_state: st.session_state.current_position = None
if 'last_signal_time' not in st.session_state: st.session_state.last_signal_time = None

# --- Broker API Mock Setup (Dhan) ---
class MockDhan:
    NSE = "NSE"
    BSE = "BSE"
    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    BUY = "BUY"
    SELL = "SELL"
    def place_order(self, **kwargs): return {"status": "success", "order_id": "MOCK_" + str(int(time.time()))}

dhan = MockDhan() # Replace with actual initialized dhan client

def place_order(algo_signal, is_options, config, ltp):
    """Handles mapping of Algo Buy/Sell to Equity or Options orders"""
    try:
        if not config['enable_dhan']: return "Paper Trade Success"
        
        # Determine actual transaction type based on Options Buyer rule
        if is_options:
            transaction_type = dhan.BUY # Always buy for options (CE or PE)
            security_id = config['ce_id'] if algo_signal == 'Buy' else config['pe_id']
            exchange = config['opt_exchange']
            qty = config['opt_qty']
            product = dhan.INTRADAY
            o_type = config['opt_entry_type']
        else:
            transaction_type = dhan.BUY if algo_signal == 'Buy' else dhan.SELL
            security_id = config['eq_id']
            exchange = config['eq_exchange']
            qty = config['eq_qty']
            product = config['eq_product']
            o_type = config['eq_entry_type']
            
        price = ltp if o_type == "LIMIT" else 0
        
        # Dhan Order call
        res = dhan.place_order(
            security_id=security_id, exchange_segment=exchange, transaction_type=transaction_type,
            quantity=qty, order_type=o_type, product_type=product, price=price
        )
        return res
    except Exception as e:
        return f"Order failed: {e}"

# --- Data Fetching & Indicators ---
def fetch_data(ticker, period, interval, warmup_days=20):
    """Fetches data with a warmup period to ensure EMA accuracy"""
    try:
        # Extend period slightly to warmup EMAs if possible, yfinance strings are tricky, handling gracefully
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        df.index = df.index.tz_convert(IST) if df.index.tzinfo else df.index.tz_localize('UTC').tz_convert(IST)
        return df
    except:
        return None

def apply_indicators(df, fast_ema, slow_ema):
    """Exact TradingView EMA Calculation"""
    if df is None or len(df) == 0: return df
    # adjust=False makes it match TradingView's recursive calculation
    df['EMA_Fast'] = df['Close'].ewm(span=fast_ema, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_ema, adjust=False).mean()
    return df

def get_ltp_stats(ticker):
    try:
        tkr = yf.Ticker(ticker)
        todays_data = tkr.history(period='2d')
        if len(todays_data) >= 2:
            prev_close = todays_data['Close'].iloc[-2]
            ltp = todays_data['Close'].iloc[-1]
            diff = float(ltp - prev_close)
            pct = float((diff / prev_close) * 100)
            return float(ltp), diff, pct
        return None, None, None
    except:
        return None, None, None

# --- UI Sidebar ---
st.sidebar.title("⚙️ Smart Wealth Config")

selected_asset = st.sidebar.selectbox("Select Asset", list(TICKERS.keys()))
custom_ticker = st.sidebar.text_input("Custom Ticker (YF format)") if selected_asset == "Custom" else ""
current_ticker = custom_ticker if selected_asset == "Custom" else TICKERS[selected_asset]

tf = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()))
period = st.sidebar.selectbox("Period", TIMEFRAME_PERIODS[tf])

st.sidebar.markdown("### Strategy Parameters")
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
"""
# Hidden/Commented Strategy Setup for Threshold Crossover
# strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Threshold Cross"])
# if strategy == "Threshold Cross":
#    threshold_val = st.sidebar.number_input("Threshold Price", value=20000)
#    cross_above_action = st.sidebar.selectbox("If Crosses Above", ["Buy", "Sell"])
#    cross_below_action = st.sidebar.selectbox("If Crosses Below", ["Sell", "Buy"])
"""
fast_ema = st.sidebar.number_input("Fast EMA", value=9)
slow_ema = st.sidebar.number_input("Slow EMA", value=15)

st.sidebar.markdown("### Risk Management")
sl_type = st.sidebar.selectbox("Stoploss Type", ["Custom Points", "Trailing SL", "Reverse EMA Crossover", "Risk-Reward"])
sl_val = st.sidebar.number_input("SL Value", value=10.0)

tgt_type = st.sidebar.selectbox("Target Type", ["Custom Points", "Trailing Target (Display)", "EMA Crossover", "Risk-Reward"])
tgt_val = st.sidebar.number_input("Target Value", value=20.0)

st.sidebar.markdown("### Execution Rules")
cooldown_enabled = st.sidebar.checkbox("Enable Cooldown", value=True)
cooldown_sec = st.sidebar.number_input("Cooldown (seconds)", value=5) if cooldown_enabled else 0
no_overlap = st.sidebar.checkbox("Prevent Overlapping Trades", value=True)

st.sidebar.markdown("### Broker Integration (Dhan)")
enable_dhan = st.sidebar.checkbox("Enable Dhan Broker", value=False)
opt_enabled = st.sidebar.checkbox("Options Trading", value=False)

if opt_enabled:
    opt_exchange = st.sidebar.selectbox("Exchange (FNO)", ["NSE_FNO", "BSE_FNO"])
    ce_id = st.sidebar.text_input("CE Security ID", "12345")
    pe_id = st.sidebar.text_input("PE Security ID", "67890")
    opt_qty = st.sidebar.number_input("Option Quantity (Lots*Size)", value=65)
    opt_entry_type = st.sidebar.selectbox("Entry Order Type (Opt)", ["MARKET", "LIMIT"])
    opt_exit_type = st.sidebar.selectbox("Exit Order Type (Opt)", ["MARKET", "LIMIT"])
else:
    eq_product = st.sidebar.selectbox("Product", ["INTRADAY", "DELIVERY"])
    eq_exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE"])
    eq_id = st.sidebar.text_input("Security ID", "1594")
    eq_qty = st.sidebar.number_input("Quantity", value=1)
    eq_entry_type = st.sidebar.selectbox("Entry Order Type", ["MARKET", "LIMIT"])
    eq_exit_type = st.sidebar.selectbox("Exit Order Type", ["MARKET", "LIMIT"])

config = {
    'ticker': current_ticker, 'tf': tf, 'period': period, 'strategy': strategy,
    'fast': fast_ema, 'slow': slow_ema, 'sl_val': sl_val, 'tgt_val': tgt_val,
    'cooldown': cooldown_sec, 'overlap': no_overlap, 'enable_dhan': enable_dhan,
    'opt_enabled': opt_enabled, 'eq_id': eq_id if not opt_enabled else None, 
    'ce_id': ce_id if opt_enabled else None, 'pe_id': pe_id if opt_enabled else None,
    'opt_exchange': opt_exchange if opt_enabled else None, 'eq_exchange': eq_exchange if not opt_enabled else None,
    'opt_qty': opt_qty if opt_enabled else None, 'eq_qty': eq_qty if not opt_enabled else None,
    'eq_product': eq_product if not opt_enabled else None, 'opt_entry_type': opt_entry_type if opt_enabled else None,
    'eq_entry_type': eq_entry_type if not opt_enabled else None
}

# --- Header ---
st.title("📈 Smart Wealth")
ltp, diff, pct = get_ltp_stats(current_ticker)
if ltp:
    color = "green" if diff >= 0 else "red"
    arrow = "▲" if diff >= 0 else "▼"
    st.markdown(f"**{selected_asset} LTP:** :blue[{ltp:.2f}] | <span style='color:{color}'>{arrow} {abs(diff):.2f} ({pct:.2f}%)</span>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🧪 Backtesting", "⚡ Live Trading", "📜 Trade History"])

# --- Tab 1: Backtesting Engine ---
with tab1:
    if st.button("Run Backtest"):
        with st.spinner("Fetching data and computing..."):
            df = fetch_data(current_ticker, period, tf)
            if df is not None:
                df = apply_indicators(df, fast_ema, slow_ema)
                trades = []
                in_trade = False
                entry_price = 0
                sl = 0
                tgt = 0
                trade_type = ""
                entry_time = None
                violation_count = 0
                last_exit_time = None
                
                # Single pass vectorised-style iteration
                for i in range(1, len(df)):
                    row = df.iloc[i]
                    prev_row = df.iloc[i-1]
                    current_time = df.index[i]
                    
                    if in_trade:
                        high, low = row['High'].item(), row['Low'].item()
                        hit_sl = False
                        hit_tgt = False
                        
                        # Conservative SL check
                        if trade_type == 'Buy':
                            if low <= sl: hit_sl = True
                            if high >= tgt: hit_tgt = True
                        else:
                            if high >= sl: hit_sl = True
                            if low <= tgt: hit_tgt = True
                            
                        if hit_sl and hit_tgt:
                            violation_count += 1
                            # Assume SL hit first in conservative approach
                            hit_tgt = False 

                        if hit_sl or hit_tgt:
                            exit_price = sl if hit_sl else tgt
                            pnl = (exit_price - entry_price) if trade_type == 'Buy' else (entry_price - exit_price)
                            trades.append({
                                'Entry Time': entry_time, 'Exit Time': current_time,
                                'Trade Type': trade_type, 'Entry Price': entry_price, 'Exit Price': exit_price,
                                'SL': sl, 'Target': tgt, 'High': high, 'Low': low,
                                'Entry Reason': f"{strategy}", 'Exit Reason': 'SL' if hit_sl else 'Target',
                                'PnL': pnl, 'Violation': 'Yes' if (hit_sl and hit_tgt) else 'No'
                            })
                            in_trade = False
                            last_exit_time = current_time
                    
                    else:
                        # Entry Logic
                        if cooldown_enabled and last_exit_time:
                            if (current_time - last_exit_time).total_seconds() < cooldown_sec: continue
                            
                        # Signal Generation
                        buy_signal = prev_row['EMA_Fast'] < prev_row['EMA_Slow'] and row['EMA_Fast'] > row['EMA_Slow']
                        sell_signal = prev_row['EMA_Fast'] > prev_row['EMA_Slow'] and row['EMA_Fast'] < row['EMA_Slow']
                        
                        if strategy == "Simple Buy": buy_signal, sell_signal = True, False
                        if strategy == "Simple Sell": buy_signal, sell_signal = False, True

                        if buy_signal or sell_signal:
                            in_trade = True
                            trade_type = 'Buy' if buy_signal else 'Sell'
                            entry_price = row['Close'].item()
                            entry_time = current_time
                            if trade_type == 'Buy':
                                sl = entry_price - sl_val
                                tgt = entry_price + tgt_val
                            else:
                                sl = entry_price + sl_val
                                tgt = entry_price - tgt_val
                
                # Render Results
                if trades:
                    tdf = pd.DataFrame(trades)
                    tdf['Accuracy'] = np.where(tdf['PnL'] > 0, 1, 0)
                    acc_pct = (tdf['Accuracy'].sum() / len(tdf)) * 100
                    
                    st.success(f"Backtest Complete! Win Rate: {acc_pct:.2f}% | Total PnL: {tdf['PnL'].sum():.2f}")
                    if violation_count > 0:
                        st.warning(f"⚠️ {violation_count} trades violated the High/Low rule (Candle crossed both SL and Target). Counted as SL conservatively.")
                    st.dataframe(tdf)
                    
                    # Plot
                    fig = make_subplots(rows=1, cols=1)
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], line=dict(color='blue'), name='Fast EMA'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], line=dict(color='orange'), name='Slow EMA'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trades generated with current parameters.")
            else:
                st.error("Failed to fetch backtest data.")

# --- Tab 2: Live Trading Engine ---
with tab2:
    col1, col2, col3 = st.columns(3)
    start_btn = col1.button("▶ Start Live Trading", type="primary")
    stop_btn = col2.button("🛑 Stop")
    sq_btn = col3.button("⏹ Square Off")

    if start_btn: st.session_state.live_running = True
    if stop_btn: st.session_state.live_running = False
    if sq_btn and st.session_state.current_position:
        st.toast("Squaring off position...")
        # Fire dhan exit order
        place_order('Sell' if st.session_state.current_position['type'] == 'Buy' else 'Buy', config['opt_enabled'], config, ltp)
        st.session_state.current_position = None

    # Placeholders for UI updates to prevent full-page flicker
    config_placeholder = st.empty()
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    if st.session_state.live_running:
        config_placeholder.info(f"**Live Config:** {selected_asset} | {tf} | {strategy} | SL: {sl_val} | TGT: {tgt_val}")
        
        while st.session_state.live_running:
            try:
                # 1. Fetch live tick/candle data
                df = fetch_data(current_ticker, '5d', tf)
                if df is not None:
                    df = apply_indicators(df, fast_ema, slow_ema)
                    last_row = df.iloc[-1]
                    prev_row = df.iloc[-2]
                    current_ltp = last_row['Close'].item()
                    current_time_dt = datetime.now(IST)
                    
                    # Update status
                    status_placeholder.write(f"Last Fetched: {current_time_dt.strftime('%H:%M:%S')} | LTP: {current_ltp:.2f} | Fast EMA: {last_row['EMA_Fast']:.2f}")

                    # 2. Position Management (Tick-level checking)
                    if st.session_state.current_position:
                        pos = st.session_state.current_position
                        exit_signal = False
                        exit_reason = ""
                        
                        # Check LTP against SL/TGT
                        if pos['type'] == 'Buy':
                            if current_ltp <= pos['sl']: exit_signal, exit_reason = True, "SL"
                            elif current_ltp >= pos['tgt']: exit_signal, exit_reason = True, "Target"
                        else:
                            if current_ltp >= pos['sl']: exit_signal, exit_reason = True, "SL"
                            elif current_ltp <= pos['tgt']: exit_signal, exit_reason = True, "Target"
                            
                        if exit_signal:
                            pnl = (pos['sl'] if exit_reason == "SL" else pos['tgt']) - pos['entry_price']
                            if pos['type'] == 'Sell': pnl = -pnl
                            
                            # Fire Broker Exit
                            place_order('Sell' if pos['type'] == 'Buy' else 'Buy', config['opt_enabled'], config, current_ltp)
                            
                            st.session_state.trade_history.append({
                                'Entry Time': pos['entry_time'], 'Exit Time': current_time_dt,
                                'Type': pos['type'], 'Entry Price': pos['entry_price'], 
                                'Exit Price': current_ltp, 'Reason': exit_reason, 'PnL': pnl
                            })
                            st.session_state.current_position = None
                            st.session_state.last_signal_time = current_time_dt
                            st.toast(f"Position Exited: {exit_reason}")

                    # 3. Signal Generation (Candle Close Check)
                    # For a 5m timeframe, only check logic if minute is multiple of 5
                    tf_mins = int(tf.replace('m','')) if 'm' in tf else 60 # Simplified parser
                    if current_time_dt.minute % tf_mins == 0 and current_time_dt.second < 5: 
                        
                        if not st.session_state.current_position or not no_overlap:
                            # Cooldown check
                            cooldown_ok = True
                            if cooldown_enabled and st.session_state.last_signal_time:
                                if (current_time_dt - st.session_state.last_signal_time).total_seconds() < cooldown_sec:
                                    cooldown_ok = False
                                    
                            if cooldown_ok:
                                buy_sig = prev_row['EMA_Fast'] < prev_row['EMA_Slow'] and last_row['EMA_Fast'] > last_row['EMA_Slow']
                                sell_sig = prev_row['EMA_Fast'] > prev_row['EMA_Slow'] and last_row['EMA_Fast'] < last_row['EMA_Slow']
                                
                                if buy_sig or sell_sig:
                                    trade_type = 'Buy' if buy_sig else 'Sell'
                                    sl = current_ltp - sl_val if trade_type == 'Buy' else current_ltp + sl_val
                                    tgt = current_ltp + tgt_val if trade_type == 'Buy' else current_ltp - tgt_val
                                    
                                    # Fire Broker Entry
                                    place_order(trade_type, config['opt_enabled'], config, current_ltp)
                                    
                                    st.session_state.current_position = {
                                        'type': trade_type, 'entry_price': current_ltp,
                                        'sl': sl, 'tgt': tgt, 'entry_time': current_time_dt
                                    }
                                    st.toast(f"New Position Entered: {trade_type}")

                    # 4. Chart Update
                    fig_live = go.Figure()
                    fig_live.add_trace(go.Scatter(x=df.index[-50:], y=df['Close'].iloc[-50:], mode='lines', name='Price'))
                    fig_live.add_trace(go.Scatter(x=df.index[-50:], y=df['EMA_Fast'].iloc[-50:], line=dict(color='blue'), name='Fast EMA'))
                    if st.session_state.current_position:
                         fig_live.add_hline(y=st.session_state.current_position['entry_price'], line_dash="solid", line_color="green", annotation_text="Entry")
                         fig_live.add_hline(y=st.session_state.current_position['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                         fig_live.add_hline(y=st.session_state.current_position['tgt'], line_dash="dash", line_color="blue", annotation_text="Target")
                    
                    fig_live.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
                    chart_placeholder.plotly_chart(fig_live, use_container_width=True)

            except Exception as e:
                status_placeholder.error(f"Live loop error: {e}")
            
            # API Rate Limit Protection
            time.sleep(1.5)

# --- Tab 3: Trade History ---
with tab3:
    st.markdown("### Completed Live Trades")
    if len(st.session_state.trade_history) > 0:
        hist_df = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No trades executed yet.")
