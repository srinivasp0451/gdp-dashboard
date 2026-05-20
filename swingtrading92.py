import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Smart Money Institutional Trading System")

# --- INITIALIZE SESSION STATE ---
if "live_tracking_active" not in st.session_state:
    st.session_state.live_tracking_active = False
if "simulated_position" not in st.session_state:
    st.session_state.simulated_position = None
if "trade_ledger" not in st.session_state:
    st.session_state.trade_ledger = []

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
        "Kotak Mahindra Bank": "KOTAKBANK.NS", "State Bank of India": "SBIN.NS", "Bajaj Finance": "BAJFINANCE.NS"
    },
    "Information Technology": {
        "TCS": "TCS.NS", "Infosys": "INFY.NS", "HCLTech": "HCLTECH.NS", "Wipro": "WIPRO.NS"
    },
    "Oil, Gas & Consumable Fuels": {
        "Reliance Industries": "RELIANCE.NS", "ONGC": "ONGC.NS", "Coal India": "COALINDIA.NS"
    },
    "Automobile and Auto Components": {
        "Tata Motors": "TATAMOTORS.NS", "Mahindra & Mahindra": "M&M.NS", "Maruti Suzuki": "MARUTI.NS"
    },
    "Healthcare & Pharmaceuticals": {
        "Sun Pharma": "SUNPHARMA.NS", "Cipla": "CIPLA.NS", "Apollo Hospitals": "APOLLOHOSP.NS"
    },
    "Fast Moving Consumer Goods (FMCG)": {
        "Hindustan Unilever": "HINDUNILVR.NS", "ITC": "ITC.NS", "Nestle India": "NESTLEIND.NS"
    },
    "Metals & Mining": {
        "Tata Steel": "TATASTEEL.NS", "JSW Steel": "JSWSTEEL.NS", "Adani Enterprises": "ADANIENT.NS"
    }
}

ALL_NIFTY_STOCKS = {}
for sector, stocks in NIFTY50_SECTORS.items():
    ALL_NIFTY_STOCKS.update(stocks)

# --- MANUAL TECHNICAL INDICATORS ---
def calculate_indicators(df, lookback=20):
    df = df.copy()
    if len(df) < lookback:
        return df
    
    # 1. Moving Averages (EMA)
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_15'] = df['Close'].ewm(span=15, adjust=False).mean()
    
    # 2. VWAP Calculation
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, 1)
    
    # 3. Structural Highs/Lows
    df['Rolling_High'] = df['High'].shift(1).rolling(window=lookback).max()
    df['Rolling_Low'] = df['Low'].shift(1).rolling(window=lookback).min()
    df['Candle_Range'] = (df['High'] - df['Low']).abs()
    df['Avg_Range_10'] = df['Candle_Range'].shift(1).rolling(window=10).mean()
    df['Avg_Volume_10'] = df['Volume'].shift(1).rolling(window=10).mean()
    
    # 4. ATR & RSI
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 0.0001)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    return df

# --- STRATEGY ENGINE MATRIX ---
def run_trap_strategy(df, lookback=20):
    df = calculate_indicators(df, lookback)
    signals = []
    if len(df) < lookback + 5: return df, signals
    
    for i in range(lookback + 2, len(df)):
        prev_row = df.iloc[i-1]
        row = df.iloc[i]
        
        if prev_row['Close'] > prev_row['Rolling_High'] and prev_row['Volume'] <= prev_row['Avg_Volume_10']:
            if row['Close'] < prev_row['Rolling_High']:
                signals.append((df.index[i], "SELL", "Bull Trap", row['Close'], 85, "Institutional Trap Above Resistance"))
        elif prev_row['Close'] < prev_row['Rolling_Low'] and prev_row['Volume'] <= prev_row['Avg_Volume_10']:
            if row['Close'] > prev_row['Rolling_Low']:
                signals.append((df.index[i], "BUY", "Bear Trap", row['Close'], 85, "Institutional Trap Below Support"))
    return df, signals

def run_alternative_strategies(df, strategy_name):
    df = calculate_indicators(df)
    signals = []
    if len(df) < 20: return df, signals
    
    for i in range(1, len(df)):
        if strategy_name == "EMA Crossover Strategy":
            if df['EMA_9'].iloc[i-1] <= df['EMA_15'].iloc[i-1] and df['EMA_9'].iloc[i] > df['EMA_15'].iloc[i]:
                signals.append((df.index[i], "BUY", "EMA Cross", df['Close'].iloc[i], 80, "9 EMA Crossed Above 15 EMA"))
            elif df['EMA_9'].iloc[i-1] >= df['EMA_15'].iloc[i-1] and df['EMA_9'].iloc[i] < df['EMA_15'].iloc[i]:
                signals.append((df.index[i], "SELL", "EMA Cross", df['Close'].iloc[i], 80, "9 EMA Crossed Below 15 EMA"))
        elif strategy_name == "VWAP Based Strategy":
            if df['Close'].iloc[i-1] <= df['VWAP'].iloc[i-1] and df['Close'].iloc[i] > df['VWAP'].iloc[i]:
                signals.append((df.index[i], "BUY", "VWAP Reclaim", df['Close'].iloc[i], 75, "Price crossed above Anchor VWAP"))
            elif df['Close'].iloc[i-1] >= df['VWAP'].iloc[i-1] and df['Close'].iloc[i] < df['VWAP'].iloc[i]:
                signals.append((df.index[i], "SELL", "VWAP Breakdown", df['Close'].iloc[i], 75, "Price dropped below Anchor VWAP"))
    return df, signals

# --- FETCH DATA ---
def fetch_ticker_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data = data.loc[:, ~data.columns.duplicated()].copy()
        return data.dropna(subset=['Close'])
    except Exception:
        return pd.DataFrame()

# --- SIDEBAR MANUVER PANEL ---
st.sidebar.title("🛡️ Institutional Risk Desk")
ticker_selection = st.sidebar.selectbox("Select Active Asset Framework", list(TICKER_MAP.keys()) + ["Custom Ticker"])
ticker_symbol = st.sidebar.text_input("Symbol Code", "AAPL") if ticker_selection == "Custom Ticker" else TICKER_MAP[ticker_selection]

strategy_choice = st.sidebar.selectbox("Dashboard Analytics Strategy", [
    "Institutional Trap Strategy", "EMA Crossover Strategy", "VWAP Based Strategy"
])
interval_choice = st.sidebar.selectbox("Timeframe Frame", ["1m", "5m", "15m", "1h", "1d"], index=2)
period_choice = st.sidebar.selectbox("Lookback Limit", ["1d", "5d", "1mo", "3mo", "1y"], index=1)

st.sidebar.subheader("Risk Constraints")
sl_points = st.sidebar.number_input("Stop Loss points allocation", min_value=0.5, value=15.0)
target_multiplier = st.sidebar.selectbox("Enforced Risk-Reward Matrix", [2.0, 3.0, 4.0], index=0)

# --- CORE APP NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Backtesting Analytics", 
    "⚡ Live Micro-Execution Framework", 
    "📜 Structural History Ledger",
    "🔍 Institutional Sector Screener (ORB Setup)"
])

# ==================== TAB 1: HISTORICAL BACKTEST ENGINE ====================
with tab1:
    st.header("Historical Engine Diagnostic Board")
    if st.button("Execute System Backtest Analysis", use_container_width=True):
        df = fetch_ticker_data(ticker_symbol, period_choice, interval_choice)
        if not df.empty:
            df, signals = run_trap_strategy(df) if strategy_choice == "Institutional Trap Strategy" else run_alternative_strategies(df, strategy_choice)
            if signals:
                st.success(f"Successfully tracked {len(signals)} matching execution setups.")
                st.dataframe(pd.DataFrame(signals, columns=["Time", "Type", "Pattern", "Price", "Score", "Description"]), use_container_width=True)
            else:
                st.info("No patterns matched configuration metrics within this historical window block.")

# ==================== TAB 2: LIVE MICRO-EXECUTION CONTROLLER ====================
with tab2:
    st.header("Live Micro-Execution Matrix Stream")
    
    # Hardware Interface Control Switches
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    if c_btn1.button("🟢 START REAL-TIME STREAM PIPELINE", use_container_width=True):
        st.session_state.live_tracking_active = True
    if c_btn2.button("🔴 SHUT DOWN TELEMETRY STREAM", use_container_width=True):
        st.session_state.live_tracking_active = False
    if c_btn3.button("💥 EMERGENCY SQUARE-OFF ALL RUNS", use_container_width=True):
        st.session_state.live_tracking_active = False
        if st.session_state.simulated_position:
            st.session_state.simulated_position["Status"] = "SQUARED_OFF"
            st.session_state.trade_ledger.append(st.session_state.simulated_position)
            st.session_state.simulated_position = None
            st.toast("Positions liquidated safely.", icon="🚨")

    st.markdown("---")

    # Dynamic Fragment Update Container Block
    @st.fragment(run_every=1.0 if st.session_state.live_tracking_active else None)
    def render_live_telemetry():
        if not st.session_state.live_tracking_active:
            st.warning("Core engine pipelines running inside structural Standby Status. Initiate Stream to calculate logs.")
            return
            
        # Download data stream array blocks safely
        df = fetch_ticker_data(ticker_symbol, "5d", interval_choice)
        if df.empty:
            st.error("Telemetry Sync Error: No usable asset price rows parsed safely.")
            return
            
        df = calculate_indicators(df)
        ltp = round(float(df['Close'].iloc[-1]), 2)
        highest_p = round(float(df['High'].max()), 2)
        lowest_p = round(float(df['Low'].min()), 2)
        
        # Check active signal logic validation runs
        df, live_sigs = run_trap_strategy(df) if strategy_choice == "Institutional Trap Strategy" else run_alternative_strategies(df, strategy_choice)
        signal_found = "MATCHED SETUP" if live_sigs else "SCANNING FOR PATTERN"
        
        # Setup position simulator rules if active setup signals exist
        if live_sigs and st.session_state.simulated_position is None:
            latest = live_sigs[-1]
            direction = latest[1]
            sl_calc = ltp - sl_points if direction == "BUY" else ltp + sl_points
            tp_calc = ltp + (sl_points * target_multiplier) if direction == "BUY" else ltp - (sl_points * target_multiplier)
            
            st.session_state.simulated_position = {
                "Asset": ticker_symbol, "Time": datetime.now().strftime("%H:%M:%S"),
                "Direction": direction, "Entry": ltp, "SL": round(sl_calc, 2), "TP": round(tp_calc, 2),
                "Status": "OPEN"
            }
            st.toast(f"Executed live structural position logic framework: {direction}", icon="📈")

        # Position Monitoring Matrix and Real-time PnL Calculation Loops
        live_pnl = 0.0
        if st.session_state.simulated_position and st.session_state.simulated_position["Status"] == "OPEN":
            pos = st.session_state.simulated_position
            if pos["Direction"] == "BUY":
                live_pnl = ltp - pos["Entry"]
                if df['Low'].iloc[-1] <= pos["SL"]: pos["Status"] = "STOPPED_OUT"
                elif df['High'].iloc[-1] >= pos["TP"]: pos["Status"] = "TARGET_HIT"
            else:
                live_pnl = pos["Entry"] - ltp
                if df['High'].iloc[-1] <= pos["SL"]: pos["Status"] = "STOPPED_OUT"
                elif df['Low'].iloc[-1] >= pos["TP"]: pos["Status"] = "TARGET_HIT"
                
            if pos["Status"] in ["STOPPED_OUT", "TARGET_HIT"]:
                st.session_state.trade_ledger.append(pos)
                st.session_state.simulated_position = None
                st.toast("Position hit execution target boundary limit and moved cleanly to ledger records.", icon="📥")

        # Visual Dashboard layout layers
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("LTP (Last Traded Price)", f"{ltp:.2f}", delta=f"{df['Close'].diff().iloc[-1]:.2f}")
        m_col2.metric("Highest Trading High Window", f"{highest_p:.2f}")
        m_col3.metric("Lowest Trading Low Window", f"{lowest_p:.2f}")
        
        pnl_color = "inverse" if live_pnl < 0 else "normal"
        m_col4.metric("Real-Time Running PnL", f"{live_pnl:.2f} Pts", delta=f"{live_pnl:.2f} Pts", delta_color=pnl_color)

        st.markdown("---")
        
        layout_left, layout_right = st.columns([2, 3])
        with layout_left:
            st.subheader("Nav Control Matrix Allocations")
            st.markdown(f"""
            - **Monitored Asset Key**: `{ticker_symbol}`
            - **Strategic Rule Logic**: `{strategy_choice}`
            - **Assigned Frame Time**: `{interval_choice}`
            - **Calculated Engine Loop Status**: `{signal_found}`
            """)
            
            st.subheader("Active Position Parameters")
            if st.session_state.simulated_position:
                st.json(st.session_state.simulated_position)
            else:
                st.info("System currently scanning neutral. No open position traces verified.")
                
        with layout_right:
            st.subheader("Live Operational Overlay Canvas")
            fig = go.Figure()
            # Keep only the last 40 bars for scannable high-speed visualization
            plot_df = df.tail(40)
            fig.add_trace(go.Candlestick(
                x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'], name="Candles"
            ))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_9'], line=dict(color='orange', width=1), name='9 EMA'))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_15'], line=dict(color='cyan', width=1), name='15 EMA'))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    render_live_telemetry()

# ==================== TAB 3: SYSTEM AUDIT LEDGER ====================
with tab3:
    st.header("Master Operational Account Ledger Records")
    if st.session_state.trade_ledger:
        st.dataframe(pd.DataFrame(st.session_state.trade_ledger), use_container_width=True)
    else:
        st.info("No recorded real-time positions committed to memory records during this run.")

# ==================== TAB 4: INDEPENDENT SECTOR ORB SCREENER ====================
with tab4:
    st.header("🔍 Autonomic Intraday Opening Range Breakout (ORB) Screener")
    st.markdown("Independent institutional framework module scanning asset lists for 15-Minute range breakouts.")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        scr_sector = st.selectbox("Screener Segment Sector Filter", ["ALL Assets"] + list(NIFTY50_SECTORS.keys()))
    with sc2:
        orb_target_rr = st.selectbox("Execution Target Risk-Reward Metric", ["1:2 Target Layout", "1:4 Risk Matrix Acceleration"])
    with sc3:
        direction_filter = st.selectbox("Breakout Direction Lookups", ["All Breakouts", "Bullish Breakouts Only", "Bearish Breakouts Only"])

    # Map stock ticker scope allocation matrices
    scan_pool = ALL_NIFTY_STOCKS if scr_sector == "ALL Assets" else NIFTY50_SECTORS[scr_sector]
    rr_multiplier = 2.0 if "1:2" in orb_target_rr else 4.0

    if st.button("🚀 EXECUTE AUTONOMOUS STRATEGY MATRIX DISCOVERY", use_container_width=True):
        screener_records = []
        progress_slot = st.progress(0)
        status_slot = st.empty()
        
        pool_len = len(scan_pool)
        
        for idx, (name, symbol) in enumerate(scan_pool.items()):
            status_slot.text(f"Processing Array Node [{idx+1}/{pool_len}]: Querying Data Stream for {name} ({symbol})...")
            progress_slot.progress((idx + 1) / pool_len)
            
            # Fetch intraday data required for calculation mapping
            hist_df = fetch_ticker_data(symbol, "5d", "15m")
            if hist_df.empty or len(hist_df) < 5:
                continue
                
            # Filter rows for the most recent trading session to locate opening values cleanly
            last_date = hist_df.index[-1].date()
            day_df = hist_df[hist_df.index.date == last_date]
            if len(day_df) < 2: 
                continue
                
            # Define Opening Range boundaries based on the initial 15-minute bar
            first_candle = day_df.iloc[0]
            orb_high = float(first_candle['High'])
            orb_low = float(first_candle['Low'])
            
            current_row = day_df.iloc[-1]
            ltp = float(current_row['Close'])
            
            setup_status = "Neutral"
            entry_ref = 0.0
            sl_val = 0.0
            tp_val = 0.0
            
            # Evaluate range break rules across the remaining intraday matrix rows
            for bar_idx in range(1, len(day_df)):
                check_row = day_df.iloc[bar_idx]
                if check_row['Close'] > orb_high:
                    setup_status = "BULLISH BREAKOUT"
                    entry_ref = orb_high
                    sl_val = orb_low
                    tp_val = entry_ref + ((entry_ref - sl_val) * rr_multiplier)
                    break
                elif check_row['Close'] < orb_low:
                    setup_status = "BEARISH BREAKDOWN"
                    entry_ref = orb_low
                    sl_val = orb_high
                    tp_val = entry_ref - ((sl_val - entry_ref) * rr_multiplier)
                    break
            
            # Filter results based on selected direction criteria
            if direction_filter == "Bullish Breakouts Only" and setup_status != "BULLISH BREAKOUT": continue
            if direction_filter == "Bearish Breakouts Only" and setup_status != "BEARISH BREAKDOWN": continue
            if setup_status == "Neutral": continue
            
            screener_records.append({
                "Asset Name": name, "Ticker Symbol": symbol, "Intraday LTP": round(ltp, 2),
                "ORB Status": setup_status, "Opening High (15m)": round(orb_high, 2), "Opening Low (15m)": round(orb_low, 2),
                "Trigger Entry": round(entry_ref, 2), "Calculated SL": round(sl_val, 2), "Target Profit Line": round(tp_val, 2)
            })

        progress_slot.empty()
        status_slot.empty()

        if screener_records:
            res_df = pd.DataFrame(screener_records)
            
            # Modern element styling updates mapped via Styler.map() to fix the structural bug
            def highlight_orb_status(val):
                if val == "BULLISH BREAKOUT": return "background-color: rgba(0, 128, 0, 0.15); color: green; font-weight: bold;"
                if val == "BEARISH BREAKDOWN": return "background-color: rgba(255, 0, 0, 0.15); color: red; font-weight: bold;"
                return ""
                
            styled_output = res_df.style.map(highlight_orb_status, subset=["ORB Status"])
            st.dataframe(styled_output, use_container_width=True)
        else:
            st.info("No underlying sector assets triggered an internal 15-minute Opening Range break matrix verification during this tracking pass.")
