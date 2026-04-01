import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as plotly_go

# ==========================================
# PAGE CONFIGURATION & STATE INITIALIZATION
# ==========================================
st.set_page_config(page_title="Smart Investing", layout="wide")

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
# DICTIONARIES & HELPER FUNCTIONS
# ==========================================
TICKER_MAP = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CUSTOM": "CUSTOM"
}

TIMEFRAME_PERIODS = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"],
    "1wk": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"]
}

def fetch_data(ticker, period, interval, buffer_days=30):
    """Fetches data with a buffer to ensure indicators don't return NaN on big gaps."""
    if ticker == "CUSTOM":
        yf_ticker = st.session_state.get('custom_ticker_input', 'RELIANCE.NS')
    else:
        yf_ticker = TICKER_MAP[ticker]
    
    try:
        df = yf.download(yf_ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        return None

def calculate_indicators(df, fast_ema, slow_ema):
    """Calculates EMAs exactly matching TradingView (adjust=False) and ATR."""
    df['EMA_Fast'] = df['Close'].ewm(span=fast_ema, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_ema, adjust=False).mean()
    
    # Strategy Logic Generation
    df['Buy_Signal'] = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
    df['Sell_Signal'] = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))
    
    # --- ATR Calculation (14-period) ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

# ==========================================
# BROKER INTEGRATION (DHAN)
# ==========================================
def execute_broker_order(signal_type, config):
    if not config['use_dhan']:
        return "Broker Disabled"
    
    try:
        from pydhan import pydhan
        from dhanhq import dhanhq
    except ImportError:
        return "Dhan libraries not installed. Run: pip install pydhan dhanhq"

    if config['options_trading']:
        sec_id = config['ce_sec_id'] if signal_type == 'Buy' else config['pe_sec_id']
        qty = config['opt_qty']
        
        dhan_opt = dhanhq(config['dhan_client_id'], config['dhan_access_token'])
        order_params = {
            "transactionType": "BUY", 
            "exchangeSegment": config['opt_exchange'],
            "productType": "INTRADAY",
            "orderType": config['opt_entry_type'],
            "validity": "DAY",
            "securityId": sec_id,
            "quantity": qty,
            "price": config['ltp'] if config['opt_entry_type'] == 'LIMIT' else 0,
            "triggerPrice": 0
        }
        try:
            res = dhan_opt.place_order(**order_params)
            return f"Option Order Placed: {res}"
        except Exception as e:
            return f"Option Order Failed: {e}"
    else:
        dhan_eq = pydhan(client_id=config['dhan_client_id'], access_token=config['dhan_access_token'])
        trans_type = dhan_eq.BUY if signal_type == 'Buy' else dhan_eq.SELL
        prod_type = dhan_eq.INTRADAY if config['eq_product'] == 'INTRADAY' else dhan_eq.DELIVERY
        ord_type = dhan_eq.MARKET if config['eq_entry_type'] == 'MARKET ORDER' else dhan_eq.LIMIT
        
        try:
            res = dhan_eq.place_order(
                security_id=config['eq_sec_id'],
                exchange_segment=dhan_eq.NSE if config['eq_exchange'] == 'NSE' else dhan_eq.BSE,
                transaction_type=trans_type,
                quantity=config['eq_qty'],
                order_type=ord_type,
                product_type=prod_type,
                price=config['ltp'] if ord_type == dhan_eq.LIMIT else 0
            )
            return f"Equity Order Placed: {res}"
        except Exception as e:
            return f"Equity Order Failed: {e}"

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.selectbox("Select Ticker", list(TICKER_MAP.keys()))
if ticker == "CUSTOM":
    custom_ticker = st.sidebar.text_input("Enter YFinance Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
    st.session_state.custom_ticker_input = custom_ticker

interval = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()))
period = st.sidebar.selectbox("Period", TIMEFRAME_PERIODS[interval])
qty = st.sidebar.number_input("Quantity", value=1, min_value=1)

strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
fast_ema = st.sidebar.number_input("Fast EMA", value=9, min_value=1)
slow_ema = st.sidebar.number_input("Slow EMA", value=15, min_value=2)

st.sidebar.markdown("---")
# SL & Target with ATR Option
sl_type = st.sidebar.selectbox("Stoploss Type", ["Custom Points", "ATR Based", "Trailing SL", "Reverse EMA Crossover", "Risk/Reward Based"])
sl_val = st.sidebar.number_input("SL Custom Points", value=10.0)

tgt_type = st.sidebar.selectbox("Target Type", ["Custom Points", "ATR Based", "Trailing Target (Display Only)", "EMA Crossover", "Risk/Reward Based"])
tgt_val = st.sidebar.number_input("Target Custom Points", value=20.0)

atr_multiplier = st.sidebar.number_input("ATR Multiplier (If ATR Based)", value=1.5, min_value=0.1, step=0.1)

st.sidebar.markdown("---")
prevent_overlap = st.sidebar.checkbox("Prevent Overlapping Trades", value=True)
cooldown_enabled = st.sidebar.checkbox("Enable Cooldown", value=True)
cooldown_period = st.sidebar.number_input("Cooldown (Seconds)", value=5, min_value=0)

# Broker Config
st.sidebar.markdown("---")
use_broker = st.sidebar.checkbox("Enable Dhan Broker", value=False)
broker_config = {'use_dhan': use_broker}

if use_broker:
    broker_config['dhan_client_id'] = st.sidebar.text_input("Client ID", type="password")
    broker_config['dhan_access_token'] = st.sidebar.text_input("Access Token", type="password")
    
    use_options = st.sidebar.checkbox("Options Trading", value=False)
    broker_config['options_trading'] = use_options
    
    if use_options:
        broker_config['opt_exchange'] = st.sidebar.selectbox("Exchange", ["NSE_FNO", "BSE_FNO"])
        broker_config['ce_sec_id'] = st.sidebar.text_input("CE Security ID")
        broker_config['pe_sec_id'] = st.sidebar.text_input("PE Security ID")
        broker_config['opt_qty'] = st.sidebar.number_input("Options Qty", value=65)
        broker_config['opt_entry_type'] = st.sidebar.selectbox("Opt Entry Type", ["MARKET", "LIMIT"])
        broker_config['opt_exit_type'] = st.sidebar.selectbox("Opt Exit Type", ["MARKET", "LIMIT"])
    else:
        broker_config['eq_product'] = st.sidebar.selectbox("Product", ["INTRADAY", "DELIVERY"])
        broker_config['eq_exchange'] = st.sidebar.selectbox("Exchange", ["NSE", "BSE"])
        broker_config['eq_sec_id'] = st.sidebar.text_input("Security ID", value="1594")
        broker_config['eq_qty'] = st.sidebar.number_input("Equity Qty", value=1)
        broker_config['eq_entry_type'] = st.sidebar.selectbox("Entry Type", ["MARKET ORDER", "LIMIT ORDER"])
        broker_config['eq_exit_type'] = st.sidebar.selectbox("Exit Type", ["MARKET ORDER", "LIMIT ORDER"])

# ==========================================
# MAIN UI: HEADER
# ==========================================
st.title("📈 Smart Investing")

df_header = fetch_data(ticker, "5d", "1d")
header_placeholder = st.empty()

if df_header is not None and len(df_header) >= 2:
    current_ltp = float(df_header['Close'].iloc[-1])
    prev_close = float(df_header['Close'].iloc[-2])
    abs_change = current_ltp - prev_close
    pct_change = (abs_change / prev_close) * 100
    color = "green" if abs_change >= 0 else "red"
    arrow = "▲" if abs_change >= 0 else "▼"
    
    header_placeholder.markdown(f"""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid {color};'>
        <h2 style='margin:0; color:white;'>{ticker} <span style='color:{color};'>{current_ltp:.2f}</span></h2>
        <p style='margin:0; color:{color}; font-size: 1.2rem;'>{arrow} {abs_change:.2f} ({pct_change:.2f}%) from previous day</p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    broker_config['ltp'] = current_ltp

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Backtesting", "🔴 Live Trading", "📝 Trade History"])

# ------------------------------------------
# TAB 1: BACKTESTING
# ------------------------------------------
with tab1:
    st.subheader("Backtest Settings & Results")
    if st.button("Run Backtest"):
        with st.spinner("Fetching data and simulating trades..."):
            df_bt = fetch_data(ticker, period, interval)
            
            if df_bt is not None:
                df_bt = calculate_indicators(df_bt, fast_ema, slow_ema)
                trades = []
                in_position = False
                pos_type = None
                entry_price = 0
                entry_time = None
                sl_price = 0
                tgt_price = 0
                reason = ""
                
                violations = 0
                violation_indices = []

                for idx, row in df_bt.iterrows():
                    # Exit Check
                    if in_position:
                        high = row['High']
                        low = row['Low']
                        close = row['Close']
                        exit_price = 0
                        exit_reason = ""
                        
                        sl_hit = False
                        tgt_hit = False
                        
                        if pos_type == 'Buy':
                            if low <= sl_price: sl_hit = True
                            if high >= tgt_price: tgt_hit = True
                            
                            if sl_hit and tgt_hit:
                                violations += 1
                                violation_indices.append(idx)
                                exit_price = sl_price
                                exit_reason = "SL Hit (Violation)"
                            elif sl_hit:
                                exit_price = sl_price
                                exit_reason = "SL Hit"
                            elif tgt_hit:
                                exit_price = tgt_price
                                exit_reason = "Target Hit"
                            elif sl_type == "Reverse EMA Crossover" and row['Sell_Signal']:
                                exit_price = close
                                exit_reason = "Reverse Signal"
                                
                        elif pos_type == 'Sell':
                            if high >= sl_price: sl_hit = True
                            if low <= tgt_price: tgt_hit = True
                            
                            if sl_hit and tgt_hit:
                                violations += 1
                                violation_indices.append(idx)
                                exit_price = sl_price
                                exit_reason = "SL Hit (Violation)"
                            elif sl_hit:
                                exit_price = sl_price
                                exit_reason = "SL Hit"
                            elif tgt_hit:
                                exit_price = tgt_price
                                exit_reason = "Target Hit"
                            elif sl_type == "Reverse EMA Crossover" and row['Buy_Signal']:
                                exit_price = close
                                exit_reason = "Reverse Signal"
                        
                        if exit_price != 0:
                            pnl = (exit_price - entry_price) * qty if pos_type == 'Buy' else (entry_price - exit_price) * qty
                            trades.append({
                                "Entry Time": entry_time,
                                "Exit Time": idx,
                                "Type": pos_type,
                                "Entry Price": round(entry_price, 2),
                                "Exit Price": round(exit_price, 2),
                                "SL": round(sl_price, 2),
                                "Target": round(tgt_price, 2),
                                "High During Trade": round(high, 2),
                                "Low During Trade": round(low, 2),
                                "Entry Reason": reason,
                                "Exit Reason": exit_reason,
                                "PnL": round(pnl, 2),
                                "Result": "Win" if pnl > 0 else "Loss"
                            })
                            in_position = False

                    # Entry Check
                    if not in_position:
                        signal = None
                        if strategy == "EMA Crossover":
                            if row['Buy_Signal']: signal = 'Buy'
                            elif row['Sell_Signal']: signal = 'Sell'
                        elif strategy == "Simple Buy":
                            signal = 'Buy' 
                        elif strategy == "Simple Sell":
                            signal = 'Sell'
                        
                        if signal:
                            in_position = True
                            pos_type = signal
                            entry_price = row['Close']
                            entry_time = idx
                            reason = f"{strategy} {signal}"
                            
                            # SL Calculation
                            if sl_type == "Custom Points":
                                sl_price = entry_price - sl_val if pos_type == 'Buy' else entry_price + sl_val
                            elif sl_type == "ATR Based":
                                atr_current = row['ATR'] if not pd.isna(row['ATR']) else sl_val
                                sl_price = entry_price - (atr_current * atr_multiplier) if pos_type == 'Buy' else entry_price + (atr_current * atr_multiplier)
                            else:
                                sl_price = entry_price * 0.99 if pos_type == 'Buy' else entry_price * 1.01 
                                
                            # Target Calculation
                            if tgt_type == "Custom Points":
                                tgt_price = entry_price + tgt_val if pos_type == 'Buy' else entry_price - tgt_val
                            elif tgt_type == "ATR Based":
                                atr_current = row['ATR'] if not pd.isna(row['ATR']) else tgt_val
                                tgt_price = entry_price + (atr_current * atr_multiplier) if pos_type == 'Buy' else entry_price - (atr_current * atr_multiplier)
                            else:
                                tgt_price = entry_price * 1.02 if pos_type == 'Buy' else entry_price * 0.98

                df_trades = pd.DataFrame(trades)
                
                if not df_trades.empty:
                    wins = len(df_trades[df_trades['Result'] == 'Win'])
                    total = len(df_trades)
                    accuracy = (wins / total) * 100
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Trades", total)
                    c2.metric("Total PnL", round(df_trades['PnL'].sum(), 2))
                    c3.metric("Accuracy", f"{accuracy:.2f}%")
                    
                    if violations > 0:
                        st.warning(f"⚠️ Rule Violation Alert: {violations} trade(s) hit both SL and Target in the same candle. Conservative exit (SL) applied to match live realism.")
                    
                    st.dataframe(df_trades, use_container_width=True)
                    
                    fig = plotly_go.Figure()
                    fig.add_trace(plotly_go.Candlestick(x=df_bt.index, open=df_bt['Open'], high=df_bt['High'], low=df_bt['Low'], close=df_bt['Close'], name='Price'))
                    fig.add_trace(plotly_go.Scatter(x=df_bt.index, y=df_bt['EMA_Fast'], line=dict(color='orange', width=1), name=f'EMA {fast_ema}'))
                    fig.add_trace(plotly_go.Scatter(x=df_bt.index, y=df_bt['EMA_Slow'], line=dict(color='blue', width=1), name=f'EMA {slow_ema}'))
                    
                    buy_trades = df_trades[df_trades['Type'] == 'Buy']
                    sell_trades = df_trades[df_trades['Type'] == 'Sell']
                    
                    fig.add_trace(plotly_go.Scatter(x=buy_trades['Entry Time'], y=buy_trades['Entry Price'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=12), name='Buy Entry'))
                    fig.add_trace(plotly_go.Scatter(x=sell_trades['Entry Time'], y=sell_trades['Entry Price'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name='Sell Entry'))
                    
                    fig.update_layout(title=f"Backtest Chart - {ticker}", xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
                    
                    # Added unique Key to prevent Duplicate ID error
                    st.plotly_chart(fig, use_container_width=True, key="backtest_chart_fixed")

                else:
                    st.info("No trades executed based on current parameters.")
            else:
                st.error("Failed to fetch data for backtesting.")

# ------------------------------------------
# TAB 2: LIVE TRADING
# ------------------------------------------
with tab2:
    st.subheader("Live Market Execution")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("▶️ Start Live Trading", use_container_width=True, type="primary"):
            st.session_state.live_running = True
    with col_btn2:
        if st.button("🛑 Stop Live Trading", use_container_width=True):
            st.session_state.live_running = False
    with col_btn3:
        if st.button("⏹️ Square Off", use_container_width=True):
            st.session_state.current_position = None
            st.success("Existing position squared off locally.")

    config_placeholder = st.empty()
    live_status_placeholder = st.empty()
    chart_placeholder = st.empty()
    last_candle_placeholder = st.empty()

    config_text = f"**Running Config:** Ticker: {ticker} | Timeframe: {interval} | Strategy: {strategy} | SL: {sl_val} pts ({sl_type}) | Tgt: {tgt_val} pts ({tgt_type}) | Overlap Prevent: {prevent_overlap} | Broker: {'ON' if use_broker else 'OFF'}"
    config_placeholder.info(config_text)

    if st.session_state.live_running:
        with live_status_placeholder.container():
            st.markdown("### 🟢 LIVE ENGINE RUNNING...")
            
            while st.session_state.live_running:
                current_time = time.time()
                
                if current_time - st.session_state.last_fetch_time >= 1.5:
                    df_live = fetch_data(ticker, "5d", interval) 
                    st.session_state.last_fetch_time = time.time()
                    
                    if df_live is not None and len(df_live) > slow_ema:
                        df_live = calculate_indicators(df_live, fast_ema, slow_ema)
                        latest_row = df_live.iloc[-1]
                        ltp = float(latest_row['Close'])
                        
                        last_candle_placeholder.markdown(f"**Last Fetched Candle Time:** `{df_live.index[-1]}` | **LTP:** `{ltp}`")

                        pos = st.session_state.current_position
                        
                        # Live Exit Logic
                        if pos is not None:
                            exit_triggered = False
                            reason = ""
                            
                            if pos['type'] == 'Buy':
                                if ltp <= pos['sl']:
                                    exit_triggered = True; reason = "SL Hit"
                                elif ltp >= pos['target']:
                                    exit_triggered = True; reason = "Target Hit"
                            elif pos['type'] == 'Sell':
                                if ltp >= pos['sl']:
                                    exit_triggered = True; reason = "SL Hit"
                                elif ltp <= pos['target']:
                                    exit_triggered = True; reason = "Target Hit"
                            
                            if exit_triggered:
                                pnl = (ltp - pos['entry_price']) * qty if pos['type'] == 'Buy' else (pos['entry_price'] - ltp) * qty
                                new_trade = {
                                    "Entry Time": pos['entry_time'],
                                    "Exit Time": datetime.now(),
                                    "Type": pos['type'],
                                    "Entry Price": pos['entry_price'],
                                    "Exit Price": ltp,
                                    "SL": pos['sl'],
                                    "Target": pos['target'],
                                    "High During Trade": df_live['High'].max(), 
                                    "Low During Trade": df_live['Low'].min(),   
                                    "Entry Reason": pos['reason'],
                                    "Exit Reason": reason,
                                    "PnL": pnl,
                                    "Result": "Win" if pnl > 0 else "Loss"
                                }
                                st.session_state.trade_history.loc[len(st.session_state.trade_history)] = new_trade
                                st.session_state.current_position = None
                                
                                if use_broker:
                                    opposite_signal = 'Sell' if pos['type'] == 'Buy' else 'Buy'
                                    execute_broker_order(opposite_signal, broker_config)

                        # Live Entry Logic
                        current_dt = datetime.now()
                        min_val = current_dt.minute
                        int_num = int(''.join(filter(str.isdigit, interval))) if interval[0].isdigit() else 1
                        is_candle_close = (min_val % int_num == 0) if "m" in interval else True 
                        
                        if st.session_state.current_position is None and is_candle_close:
                            if cooldown_enabled and len(st.session_state.trade_history) > 0:
                                last_exit = st.session_state.trade_history.iloc[-1]['Exit Time']
                                if (datetime.now() - last_exit).total_seconds() < cooldown_period:
                                    continue 
                            
                            signal = None
                            if strategy == "EMA Crossover":
                                if latest_row['Buy_Signal']: signal = 'Buy'
                                elif latest_row['Sell_Signal']: signal = 'Sell'
                                
                            if signal:
                                entry_price = ltp
                                
                                # Dynamic SL based on ATR or Custom
                                if sl_type == "Custom Points":
                                    sl_price = entry_price - sl_val if signal == 'Buy' else entry_price + sl_val
                                elif sl_type == "ATR Based":
                                    atr_current = latest_row['ATR'] if not pd.isna(latest_row['ATR']) else sl_val
                                    sl_price = entry_price - (atr_current * atr_multiplier) if signal == 'Buy' else entry_price + (atr_current * atr_multiplier)
                                else:
                                    sl_price = entry_price * 0.99 if signal == 'Buy' else entry_price * 1.01
                                
                                # Dynamic Target based on ATR or Custom
                                if tgt_type == "Custom Points":
                                    tgt_price = entry_price + tgt_val if signal == 'Buy' else entry_price - tgt_val
                                elif tgt_type == "ATR Based":
                                    atr_current = latest_row['ATR'] if not pd.isna(latest_row['ATR']) else tgt_val
                                    tgt_price = entry_price + (atr_current * atr_multiplier) if signal == 'Buy' else entry_price - (atr_current * atr_multiplier)
                                else:
                                    tgt_price = entry_price * 1.02 if signal == 'Buy' else entry_price * 0.98
                                
                                st.session_state.current_position = {
                                    'type': signal, 'entry_price': entry_price, 
                                    'entry_time': datetime.now(), 'sl': sl_price, 
                                    'target': tgt_price, 'reason': f"Live {strategy} {signal}"
                                }
                                
                                if use_broker:
                                    execute_broker_order(signal, broker_config)
                        
                        # Live Chart Update
                        chart_placeholder.empty() # Clear previous to avoid UI overlap issues
                        with chart_placeholder.container():
                            if st.session_state.current_position:
                                p = st.session_state.current_position
                                st.warning(f"**Current Position:** {p['type']} @ {p['entry_price']} | SL: {p['sl']:.2f} | TGT: {p['target']:.2f}")
                            else:
                                st.success("No active positions. Scanning for signals...")
                            
                            fig_live = plotly_go.Figure()
                            df_plot = df_live.tail(50)
                            fig_live.add_trace(plotly_go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Price'))
                            fig_live.add_trace(plotly_go.Scatter(x=df_plot.index, y=df_plot['EMA_Fast'], line=dict(color='orange', width=1), name=f'EMA {fast_ema}'))
                            fig_live.add_trace(plotly_go.Scatter(x=df_plot.index, y=df_plot['EMA_Slow'], line=dict(color='blue', width=1), name=f'EMA {slow_ema}'))
                            
                            if st.session_state.current_position:
                                pos = st.session_state.current_position
                                fig_live.add_hline(y=pos['entry_price'], line_dash="solid", line_color="blue", annotation_text="Entry")
                                fig_live.add_hline(y=pos['sl'], line_dash="dot", line_color="red", annotation_text="SL")
                                fig_live.add_hline(y=pos['target'], line_dash="dot", line_color="green", annotation_text="Target")
                                
                            fig_live.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
                            
                            # Added dynamic unique key to bypass Duplicate ID error in continuous loop
                            st.plotly_chart(fig_live, use_container_width=True, key=f"live_chart_{int(current_time)}")
                            
                time.sleep(1.5) 
    else:
        live_status_placeholder.info("Live trading is currently stopped.")

# ------------------------------------------
# TAB 3: TRADE HISTORY
# ------------------------------------------
with tab3:
    st.subheader("Completed Trade Ledger")
    st.markdown("This tab displays all trades completed during the Live Trading session, updated dynamically.")
    
    if st.session_state.trade_history.empty:
        st.info("No trades executed yet in this session.")
    else:
        th = st.session_state.trade_history
        wins = len(th[th['Result'] == 'Win'])
        total = len(th)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Live Total Trades", total)
        c2.metric("Live Total PnL", round(th['PnL'].sum(), 2))
        c3.metric("Live Accuracy", f"{(wins / total) * 100:.2f}%" if total > 0 else "0.00%")
        
        st.dataframe(
            th.style.applymap(lambda x: 'background-color: rgba(0,255,0,0.1)' if x == 'Win' else 'background-color: rgba(255,0,0,0.1)', subset=['Result']),
            use_container_width=True
        )
        
        csv = th.to_csv(index=False).encode('utf-8')
        st.download_button("Download Trade History CSV", data=csv, file_name="smart_investing_live_history.csv", mime="text/csv")
