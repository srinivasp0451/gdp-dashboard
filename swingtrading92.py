import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Smart Money Institutional Trading System")

# --- INITIALIZE SESSION STATE ---
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "active_trades" not in st.session_state:
    st.session_state.active_trades = {}
if "live_tracking_active" not in st.session_state:
    st.session_state.live_tracking_active = False

# --- TICKER DICTIONARY MAP ---
TICKER_MAP = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "USD/INR": "INR=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
}

# --- ALL NIFTY 50 STOCKS WITH SECTOR MAPPING ---
NIFTY50_SECTORS = {
    "Financial Services": {
        "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "Axis Bank": "AXISBANK.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS", "State Bank of India": "SBIN.NS", "Bajaj Finance": "BAJFINANCE.NS",
        "Bajaj Finserv": "BAJAJFINSV.NS", "HDFC Life": "HDFCLIFE.NS", "SBI Life": "SBILIFE.NS",
        "Shriram Finance": "SHRIRAMFIN.NS", "Jio Financial Services": "JIOFIN.NS"
    },
    "Information Technology": {
        "TCS": "TCS.NS", "Infosys": "INFY.NS", "HCLTech": "HCLTECH.NS",
        "Tech Mahindra": "TECHM.NS", "Wipro": "WIPRO.NS"
    },
    "Oil, Gas & Consumable Fuels": {
        "Reliance Industries": "RELIANCE.NS", "ONGC": "ONGC.NS", "Coal India": "COALINDIA.NS",
        "BPCL": "BPCL.NS"
    },
    "Automobile and Auto Components": {
        "Tata Motors": "TATAMOTORS.NS", "Mahindra & Mahindra": "M&M.NS", "Maruti Suzuki": "MARUTI.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS", "Eicher Motors": "EICHERMOT.NS"
    },
    "Healthcare & Pharmaceuticals": {
        "Sun Pharma": "SUNPHARMA.NS", "Cipla": "CIPLA.NS", "Dr. Reddy's": "DRREDDY.NS",
        "Apollo Hospitals": "APOLLOHOSP.NS", "Max Healthcare": "MAXHEALTH.NS"
    },
    "Fast Moving Consumer Goods (FMCG)": {
        "Hindustan Unilever": "HINDUNILVR.NS", "ITC": "ITC.NS", "Nestle India": "NESTLEIND.NS",
        "Britannia": "BRITANNIA.NS", "Tata Consumer Products": "TATACONSUM.NS"
    },
    "Metals & Mining": {
        "Tata Steel": "TATASTEEL.NS", "JSW Steel": "JSWSTEEL.NS", "Hindalco": "HINDALCO.NS",
        "Adani Enterprises": "ADANIENT.NS"
    },
    "Power & Utilities": {
        "NTPC": "NTPC.NS", "Power Grid": "POWERGRID.NS"
    },
    "Consumer Durables & Services": {
        "Titan Company": "TITAN.NS", "Asian Paints": "ASIANPAINT.NS", "Trent": "TRENT.NS",
        "Eternal Ltd": "ETERNAL.NS"
    },
    "Construction & Materials": {
        "Larsen & Toubro": "LT.NS", "UltraTech Cement": "ULTRACEMCO.NS", "Grasim Industries": "GRASIM.NS"
    },
    "Services & Capital Goods": {
        "Adani Ports": "ADANIPORTS.NS", "Bharat Electronics": "BEL.NS", "InterGlobe Aviation (IndiGo)": "INDIGO.NS"
    }
}

# Flattend dictionary for universal access lookup
ALL_NIFTY_STOCKS = {}
for sector, stocks in NIFTY50_SECTORS.items():
    ALL_NIFTY_STOCKS.update(stocks)


# --- MANUAL TECHNICAL INDICATORS ---
def calculate_indicators(df, lookback=20):
    df = df.copy()
    if len(df) < lookback + 10:
        return df
    
    # 1. Moving Averages (EMA)
    def calc_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
        
    df['EMA_9'] = calc_ema(df['Close'], 9)
    df['EMA_15'] = calc_ema(df['Close'], 15)
    
    # 2. VWAP Calculation
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # 3. Rolling Highs/Lows & Ranges
    df['Rolling_High'] = df['High'].shift(1).rolling(window=lookback).max()
    df['Rolling_Low'] = df['Low'].shift(1).rolling(window=lookback).min()
    df['Candle_Range'] = (df['High'] - df['Low']).abs()
    df['Avg_Range_10'] = df['Candle_Range'].shift(1).rolling(window=10).mean()
    
    # 4. Volume Features
    df['Avg_Volume_10'] = df['Volume'].shift(1).rolling(window=10).mean()
    df['Volume_Per_Point'] = df['Volume'] / (df['Close'] - df['Open']).abs().replace(0, 0.0001)
    
    # 5. ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # 6. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 0.0001)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    return df

# --- DYNAMIC RISK MANAGEMENT ENGINE ---
def calculate_sl_tp(df, idx, sl_type, tp_type, custom_sl, custom_tp, direction):
    row = df.iloc[idx]
    close = row['Close']
    atr = row['ATR_14'] if 'ATR_14' in df.columns and not pd.isna(row['ATR_14']) else close * 0.01
    
    if sl_type in ["Custom SL", "Trailing SL"]:
        sl_val = close - custom_sl if direction == "BUY" else close + custom_sl
    elif sl_type == "ATR based SL":
        sl_val = close - (1.5 * atr) if direction == "BUY" else close + (1.5 * atr)
    elif sl_type in ["Signal based SL", "Logical SL", "Support Resistance based SL"]:
        if direction == "BUY":
            sl_val = row['Rolling_Low'] if not pd.isna(row['Rolling_Low']) else close * 0.99
        else:
            sl_val = row['Rolling_High'] if not pd.isna(row['Rolling_High']) else close * 1.01
    else:
        sl_val = close * 0.99 if direction == "BUY" else close * 1.01
        
    # Enforce minimum 1:2 Risk-to-Reward ratio
    risk_distance = abs(close - sl_val)
    if risk_distance == 0:
        risk_distance = close * 0.005
        
    if tp_type in ["Custom Target", "Trailing Target"]:
        tp1 = close + custom_tp if direction == "BUY" else close - custom_tp
    elif tp_type == "ATR based Target":
        tp1 = close + (3 * atr) if direction == "BUY" else close - (3 * atr)
    elif tp_type in ["Signal based Target", "Logical Target", "Support Resistance based Target"]:
        if direction == "BUY":
            tp1 = row['Rolling_High'] if not pd.isna(row['Rolling_High']) else close * 1.02
        else:
            tp1 = row['Rolling_Low'] if not pd.isna(row['Rolling_Low']) else close * 0.98
    else:
        tp1 = close + (2 * risk_distance) if direction == "BUY" else close - (2 * risk_distance)
        
    # Hard structural math override to protect risk matrix allocations (Ensuring at least 1:2)
    reward_distance = abs(tp1 - close)
    if reward_distance < (2 * risk_distance):
        tp1 = close + (2 * risk_distance) if direction == "BUY" else close - (2 * risk_distance)
        
    return round(float(sl_val), 2), round(float(tp1), 2)

# --- BACKTEST PNL SIMULATOR ---
def simulate_backtest_pnl(processed_df, found_signals, sl_mode, tp_mode, custom_sl_input, custom_tp_input, strategy_choice):
    records = []
    for sig_idx, (timestamp, sig_type, pattern, entry_p, conf, reason) in enumerate(found_signals):
        try:
            idx_pos = processed_df.index.get_loc(timestamp)
            if isinstance(idx_pos, slice):
                idx_pos = idx_pos.start
        except KeyError:
            continue
            
        sl, tp = calculate_sl_tp(processed_df, idx_pos, sl_mode, tp_mode, custom_sl_input, custom_tp_input, sig_type)
        
        pnl = 0.0
        status = "Closed (Target)"
        exit_price = tp
        
        for forward_idx in range(idx_pos + 1, len(processed_df)):
            f_row = processed_df.iloc[forward_idx]
            if sig_type == "BUY":
                if f_row['Low'] <= sl:
                    pnl = sl - entry_p
                    exit_price = sl
                    status = "Closed (Stop Loss)"
                    break
                elif f_row['High'] >= tp:
                    pnl = (tp - entry_p)
                    exit_price = tp
                    status = "Closed (Target)"
                    break
            elif sig_type == "SELL":
                if f_row['High'] >= sl:
                    pnl = entry_p - sl
                    exit_price = sl
                    status = "Closed (Stop Loss)"
                    break
                elif f_row['Low'] <= tp:
                    pnl = entry_p - tp
                    exit_price = tp
                    status = "Closed (Target)"
                    break
                    
        if pnl == 0.0:
            last_close = processed_df['Close'].iloc[-1]
            pnl = (last_close - entry_p) if sig_type == "BUY" else (entry_p - last_close)
            exit_price = last_close
            status = "End of Data Default"
            
        records.append({
            "Timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp),
            "Signal Type": sig_type,
            "Pattern Core": pattern,
            "Entry Reference": round(entry_p, 2),
            "Stop Loss Assignment": round(sl, 2),
            "Target (TP1 70%)": round(tp, 2),
            "Exit Price": round(exit_price, 2),
            "Trade Status": status,
            "Simulated PnL (Pts)": round(pnl, 2),
            "Confidence Score": f"{conf}%",
            "Reason of Entry": reason
        })
    return records

# --- ENGINE: TRAP STRATEGY SCORER ---
def run_trap_strategy(df, lookback=20):
    df = calculate_indicators(df, lookback)
    signals = []
    if len(df) < lookback + 5:
        return df, signals
    
    for i in range(lookback + 5, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        is_bull_breakout = prev_row['Close'] > prev_row['Rolling_High']
        is_bear_breakdown = prev_row['Close'] < prev_row['Rolling_Low']
        large_range = prev_row['Candle_Range'] > (1.2 * prev_row['Avg_Range_10'])
        weak_volume = prev_row['Volume'] <= prev_row['Avg_Volume_10']
        
        if is_bull_breakout and large_range and weak_volume:
            if row['Close'] < prev_row['Rolling_High']:
                score = 65
                if row['High'] > prev_row['High']: score += 15
                if row['Close'] < row['Open']: score += 10
                reason = "Breakout above resistance with weak volume failed and closed back below."
                signals.append((df.index[i], "SELL", "Bull Trap", row['Close'], score, reason))
                
        elif is_bear_breakdown and large_range and weak_volume:
            if row['Close'] > prev_row['Rolling_Low']:
                score = 65
                if row['Low'] < prev_row['Low']: score += 15
                if row['Close'] > row['Open']: score += 10
                reason = "Breakdown below support with weak volume failed and closed back above."
                signals.append((df.index[i], "BUY", "Bear Trap", row['Close'], score, reason))
                
    return df, signals

# --- ALTERNATIVE STRATEGY RUNNERS ---
def run_alternative_strategies(df, strategy_name):
    df = calculate_indicators(df)
    signals = []
    if len(df) < 20: return df, signals
    
    if strategy_name == "EMA Crossover Strategy":
        for i in range(1, len(df)):
            if df['EMA_9'].iloc[i-1] <= df['EMA_15'].iloc[i-1] and df['EMA_9'].iloc[i] > df['EMA_15'].iloc[i]:
                signals.append((df.index[i], "BUY", "EMA Gold Cross", df['Close'].iloc[i], 80, "9 EMA crossed above 15 EMA."))
            elif df['EMA_9'].iloc[i-1] >= df['EMA_15'].iloc[i-1] and df['EMA_9'].iloc[i] < df['EMA_15'].iloc[i]:
                signals.append((df.index[i], "SELL", "EMA Death Cross", df['Close'].iloc[i], 80, "9 EMA crossed below 15 EMA."))
                
    elif strategy_name == "RSI Overbought Oversold Strategy":
        for i in range(1, len(df)):
            if df['RSI_14'].iloc[i-1] >= 30 and df['RSI_14'].iloc[i] < 30:
                signals.append((df.index[i], "BUY", "RSI Oversold Pivot", df['Close'].iloc[i], 70, "RSI crossed below 30 threshold."))
            elif df['RSI_14'].iloc[i-1] <= 70 and df['RSI_14'].iloc[i] > 70:
                signals.append((df.index[i], "SELL", "RSI Overbought Pivot", df['Close'].iloc[i], 70, "RSI crossed above 70 threshold."))
                
    elif strategy_name == "VWAP Based Strategy":
        for i in range(1, len(df)):
            if df['Close'].iloc[i-1] <= df['VWAP'].iloc[i-1] and df['Close'].iloc[i] > df['VWAP'].iloc[i]:
                signals.append((df.index[i], "BUY", "VWAP Reclaim", df['Close'].iloc[i], 75, "Price crossed above historical anchor VWAP line."))
            elif df['Close'].iloc[i-1] >= df['VWAP'].iloc[i-1] and df['Close'].iloc[i] < df['VWAP'].iloc[i]:
                signals.append((df.index[i], "SELL", "VWAP Breakdown", df['Close'].iloc[i], 75, "Price crossed below historical anchor VWAP line."))
                
    return df, signals

# --- FETCH DATA WITH ROBUST DEFENSIVE COLUMN FLATTENER ---
def fetch_ticker_data(ticker, period, interval):
    time.sleep(0.1)  # Lower sleep delay for faster scanner iteration loops
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return pd.DataFrame()
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        data = data.loc[:, ~data.columns.duplicated()].copy()
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            else:
                alternate_col = col.lower()
                if alternate_col in data.columns:
                    data[col] = pd.to_numeric(data[alternate_col], errors='coerce')
                    
        if 'Close' not in data.columns or data['Close'].isnull().all():
            return pd.DataFrame()
            
        return data.dropna(subset=['Close'])
    except Exception:
        return pd.DataFrame()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.title("🛡️ Institutional Control")
ticker_selection = st.sidebar.selectbox("Select Core Asset Group", list(TICKER_MAP.keys()) + ["Custom Ticker"])

if ticker_selection == "Custom Ticker":
    ticker_symbol = st.sidebar.text_input("Enter Custom Symbol (e.g., AAPL, EURUSD=X)", "AAPL")
else:
    ticker_symbol = TICKER_MAP[ticker_selection]

strategy_choice = st.sidebar.selectbox("Market Strategy Engine", [
    "Institutional Trap Strategy", "EMA Crossover Strategy", 
    "RSI Overbought Oversold Strategy", "VWAP Based Strategy"
])

intervals_configs = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y"],
    "1d": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
}

interval_choice = st.sidebar.selectbox("Execution Candlestick TF", list(intervals_configs.keys()), index=2)
period_choice = st.sidebar.selectbox("Historical Framework Window", intervals_configs[interval_choice])

st.sidebar.subheader("Risk Matrix Framework")
sl_mode = st.sidebar.selectbox("Stop Loss Risk Rule", ["Custom SL", "Trailing SL", "Signal based SL", "ATR based SL", "Logical SL", "Support Resistance based SL"])
custom_sl_input = st.sidebar.number_input("Custom/Trailing SL Points Offset", min_value=0.01, max_value=5000.0, value=10.0, step=1.0)

tp_mode = st.sidebar.selectbox("Profit Execution Target Rule", ["Custom Target", "Trailing Target", "Signal based Target", "ATR based Target", "Logical Target", "Support Resistance based Target"])
custom_tp_input = st.sidebar.number_input("Custom Target Points Offset", min_value=0.01, max_value=5000.0, value=20.0, step=1.0)

# --- SYSTEM APP TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Backtesting & Signal History", 
    "⚡ Live Micro-Execution Framework", 
    "📜 System Trade Ledger",
    "🔍 Institutional Sector Screener"
])

# ==================== TAB 1: BACKTESTING ENGINE ====================
with tab1:
    st.header("Historical Engine Backtesting Performance")
    if st.button("Execute Backtest Simulation", key="bt_run"):
        with st.spinner("Processing architectural market arrays..."):
            raw_data = fetch_ticker_data(ticker_symbol, period_choice, interval_choice)
            
            if raw_data.empty:
                st.warning("No standard architectural frame found for data parsing.")
            else:
                if strategy_choice == "Institutional Trap Strategy":
                    processed_df, found_signals = run_trap_strategy(raw_data)
                else:
                    processed_df, found_signals = run_alternative_strategies(raw_data, strategy_choice)
                
                if not found_signals:
                    st.info("No system-aligned operational setups matched your confirmation definitions for this period.")
                else:
                    simulated_records = simulate_backtest_pnl(processed_df, found_signals, sl_mode, tp_mode, custom_sl_input, custom_tp_input, strategy_choice)
                    df_rec = pd.DataFrame(simulated_records)
                    
                    total_pnl = df_rec["Simulated PnL (Pts)"].sum()
                    win_trades = len(df_rec[df_rec["Simulated PnL (Pts)"] > 0])
                    win_rate = (win_trades / len(df_rec)) * 100 if len(df_rec) > 0 else 0
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Net Aggregated PnL", f"{total_pnl:.2f} Pts", delta=f"{total_pnl:.2f}")
                    c2.metric("Total Generated Trades", len(df_rec))
                    c3.metric("Win-Rate Ratio Profile", f"{win_rate:.1f}%")
                    
                    st.dataframe(df_rec, use_container_width=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=processed_df.index, open=processed_df['Open'], high=processed_df['High'],
                        low=processed_df['Low'], close=processed_df['Close'], name="Price Candle Action"
                    ))
                    if 'EMA_9' in processed_df.columns:
                        fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['EMA_9'], line=dict(color='orange', width=1), name='EMA 9'))
                    if 'EMA_15' in processed_df.columns:
                        fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['EMA_15'], line=dict(color='blue', width=1), name='EMA 15'))
                        
                    for rec in simulated_records:
                        color = 'green' if rec['Signal Type'] == 'BUY' else 'red'
                        fig.add_trace(go.Scatter(
                            x=[pd.to_datetime(rec['Timestamp'])], y=[rec['Entry Reference']], mode='markers',
                            marker=dict(color=color, size=12, symbol='triangle-up' if color=='green' else 'triangle-down'),
                            name=f"{rec['Signal Type']} Entry"
                        ))
                    fig.update_layout(title=f"{ticker_symbol} Historical Backtest Architecture Map", xaxis_rangeslider_visible=False, height=600)
                    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: LIVE MICRO-EXECUTION ====================
with tab2:
    st.header("Live Micro-Execution Tracker Framework")
    
    # --- HARDWARE STATE INTERACTION CONTROLS ---
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    if btn_col1.button("🟢 START LIVE STREAM", use_container_width=True):
        st.session_state.live_tracking_active = True
        st.toast("Telemetry Pipelines Initialized Safely.", icon="🛰️")
        
    if btn_col2.button("🔴 STOP LIVE STREAM", use_container_width=True):
        st.session_state.live_tracking_active = False
        st.toast("Telemetry Pipelines Suspended Gracefully.", icon="🛑")
        
    if btn_col3.button("💥 EMERGENCY SQUARE OFF ALL", use_container_width=True):
        st.session_state.live_tracking_active = False
        active_count = 0
        for trade in st.session_state.trade_history:
            if trade["System Status"] == "ACTIVE":
                trade["System Status"] = "MANUAL SQUARE OFF"
                active_count += 1
        st.session_state.active_trades.clear()
        st.success(f"Emergency Intercept Active. Closed {active_count} market positions.")

    st.write("---")
    
    # MASTER CONTENT CONTAINERS FOR AUTO-CLEAN REFRESH HOOKS
    col1, col2 = st.columns([1, 3])
    
    with col1:
        status_text = "🟢 ENGINE SCANNING" if st.session_state.live_tracking_active else "⚪ STANDBY"
        st.metric("System Core Engine State", status_text)
        st.metric("Monitoring Target Asset", ticker_symbol)
        st.metric("Assigned Logic Routine", strategy_choice)
        live_pnl_placeholder = st.empty()
        
        st.write("---")
        st.subheader("Nav Configurations")
        st.markdown(f"""
        - **Asset**: `{ticker_symbol}`
        - **Strategy**: `{strategy_choice}`
        - **Timeframe**: `{interval_choice}` (`{period_choice}`)
        - **SL Rule**: `{sl_mode}` (`{custom_sl_input} Pts`)
        - **Target Rule**: `{tp_mode}` (`{custom_tp_input} Pts`)
        """)
        
    # RIGHT SIDE BLOCKS OUTSIDE THE LOOP TO FIX ACCUMULATION BUG
    with col2:
        telemetry_display_box = st.empty()
        matrix_display_box = st.empty()
        chart_display_box = st.empty()

    # Isolated streaming fragment running every 2.0 seconds
    @st.fragment(run_every=2.0 if st.session_state.live_tracking_active else None)
    def live_streaming_fragment():
        if not st.session_state.live_tracking_active:
            with matrix_display_box.container():
                st.info("System is resting on standby mode. Launch live stream pipelines to initialize.")
            return

# ==================== TAB 3: SYSTEM LEDGER ====================
with tab3:
    st.header("Master Operations Ledger History")
    if not st.session_state.trade_history:
        st.info("No recorded context traces committed to permanent session storage arrays yet.")
    else:
        ledger_df = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(ledger_df, use_container_width=True)


# ==================== TAB 4: INSTITUTIONAL SECTOR SCREENER ====================
with tab4:
    st.header("🔍 Nifty 50 Real-Time Multi-Sector Market Screener")
    st.markdown("Scans Nifty 50 matrix arrays across structural sectors to identify dynamic asymmetric execution targets.")
    
    # Screener-specific configuration filters
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        selected_sector = st.selectbox("Filter by Industrial Sector", ["ALL Sectors"] + list(NIFTY50_SECTORS.keys()))
    with sc2:
        screener_strategy = st.selectbox("Screener Processing Engine", [
            "Institutional Trap Strategy", "EMA Crossover Strategy", 
            "RSI Overbought Oversold Strategy", "VWAP Based Strategy"
        ], key="screen_strat")
    with sc3:
        signal_filter = st.selectbox("Filter Current Signals", ["All Setups", "BUY Only", "SELL Only"])

    # Resolve target stocks to look up based on sector constraints
    stocks_to_scan = {}
    if selected_sector == "ALL Sectors":
        stocks_to_scan = ALL_NIFTY_STOCKS
    else:
        stocks_to_scan = NIFTY50_SECTORS[selected_sector]
        
    if st.button("🚀 Run Architecture Scan", use_container_width=True, key="screener_execution_trigger"):
        screen_results = []
        
        progress_bar = st.progress(0)
        status_text_area = st.empty()
        
        total_stocks = len(stocks_to_scan)
        
        for index, (display_name, yf_ticker) in enumerate(stocks_to_scan.items()):
            status_text_area.text(f"Scanning Array Segment [{index+1}/{total_stocks}]: Parsing {display_name} ({yf_ticker})...")
            progress_bar.progress((index + 1) / total_stocks)
            
            # Request historical dataframe blocks
            ticker_df = fetch_ticker_data(yf_ticker, period_choice, interval_choice)
            if ticker_df.empty or len(ticker_df) < 25:
                continue
                
            # Direct calculations to target selected engine structures
            if screener_strategy == "Institutional Trap Strategy":
                processed_df, found_signals = run_trap_strategy(ticker_df)
            else:
                processed_df, found_signals = run_alternative_strategies(ticker_df, screener_strategy)
                
            if found_signals:
                # Target the most recent structural signal setup matched
                latest_sig = found_signals[-1]
                timestamp, sig_type, pattern, entry_p, conf, reason = latest_sig
                
                try:
                    idx_pos = processed_df.index.get_loc(timestamp)
                    if isinstance(idx_pos, slice):
                        idx_pos = idx_pos.start
                except KeyError:
                    continue
                
                # Dynamic risk calculation with structural 1:2 matrix enforcement
                sl, tp = calculate_sl_tp(processed_df, idx_pos, sl_mode, tp_mode, custom_sl_input, custom_tp_input, sig_type)
                current_price = processed_df['Close'].iloc[-1]
                
                # Check filter matching criteria
                if signal_filter == "BUY Only" and sig_type != "BUY":
                    continue
                if signal_filter == "SELL Only" and sig_type != "SELL":
                    continue
                    
                # Find matching structural sector label for inverted lookup
                matched_sector = "Unknown"
                for sec_name, sec_dict in NIFTY50_SECTORS.items():
                    if display_name in sec_dict:
                        matched_sector = sec_name
                        break
                        
                risk_pts = abs(entry_p - sl)
                reward_pts = abs(tp - entry_p)
                actual_rr = round(reward_pts / risk_pts, 2) if risk_pts > 0 else 2.0
                
                screen_results.append({
                    "Stock": display_name,
                    "Ticker": yf_ticker,
                    "Sector": matched_sector,
                    "LTP": round(current_price, 2),
                    "Signal": sig_type,
                    "Pattern": pattern,
                    "Entry Ref": round(entry_p, 2),
                    "Stop Loss": round(sl, 2),
                    "Target (1:2 Min)": round(tp, 2),
                    "Risk-Reward Ratio": f"1:{actual_rr}",
                    "Signal Time": timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)
                })
                
        # Clear loading overlays
        progress_bar.empty()
        status_text_area.empty()
        
        if not screen_results:
            st.info("No stocks matched your active strategy conditions in this specific scanning sequence.")
        else:
            final_screener_df = pd.DataFrame(screen_results)
            
            # Display high-level metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Scanned Matrix Universe", f"{total_stocks} Tickers")
            m2.metric("Identified Setups", len(final_screener_df))
            m3.metric("Filtered Strategy Focus", screener_strategy)
            
            st.write("### 📈 Active Structural Setups Table Matrix")
            
            # Color-code setups for quick recognition
            def highlight_signals(val):
                if val == 'BUY':
                    return 'background-color: rgba(0, 128, 0, 0.2); color: green; font-weight: bold;'
                elif val == 'SELL':
                    return 'background-color: rgba(255, 0, 0, 0.2); color: red; font-weight: bold;'
                return ''
                
            styled_df = final_screener_df.style.applymap(highlight_signals, subset=['Signal'])
            st.dataframe(styled_df, use_container_width=True)
