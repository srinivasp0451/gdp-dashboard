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
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_15'] = df['Close'].ewm(span=15, adjust=False).mean()
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, 1)
    df['Rolling_High'] = df['High'].shift(1).rolling(window=lookback).max()
    df['Rolling_Low'] = df['Low'].shift(1).rolling(window=lookback).min()
    df['Candle_Range'] = (df['High'] - df['Low']).abs()
    df['Avg_Range_10'] = df['Candle_Range'].shift(1).rolling(window=10).mean()
    df['Avg_Volume_10'] = df['Volume'].shift(1).rolling(window=10).mean()
    
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
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

st.sidebar.subheader("Risk & Target Framework Rules")
sl_mode = st.sidebar.selectbox("Stop Loss Risk Rule", [
    "Custom SL", "Trailing SL", "Signal based SL", "ATR based SL", "Logical SL", "Support Resistance based SL"
])
custom_sl_input = st.sidebar.number_input("Custom/Trailing SL Points Offset", min_value=0.01, max_value=5000.0, value=10.0, step=1.0)

tp_mode = st.sidebar.selectbox("Profit Execution Target Rule", [
    "Custom Target", "Trailing Target", "Signal based Target", "ATR based Target", "Logical Target", "Support Resistance based Target"
])
custom_tp_input = st.sidebar.number_input("Custom Target Points Offset", min_value=0.01, max_value=5000.0, value=20.0, step=1.0)
target_multiplier = st.sidebar.selectbox("Enforced Live Risk-Reward Matrix Multiplier", [1.0, 2.0, 3.0, 4.0, 5.0], index=1)

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

    @st.fragment(run_every=1.0 if st.session_state.live_tracking_active else None)
    def render_live_telemetry():
        if not st.session_state.live_tracking_active:
            st.warning("Core engine pipelines running inside structural Standby Status. Initiate Stream to calculate logs.")
            return
            
        df = fetch_ticker_data(ticker_symbol, "5d", interval_choice)
        if df.empty:
            st.error("Telemetry Sync Error: No usable asset price rows parsed safely.")
            return
            
        df = calculate_indicators(df)
        ltp = round(float(df['Close'].iloc[-1]), 2)
        highest_p = round(float(df['High'].max()), 2)
        lowest_p = round(float(df['Low'].min()), 2)
        
        df, live_sigs = run_trap_strategy(df) if strategy_choice == "Institutional Trap Strategy" else run_alternative_strategies(df, strategy_choice)
        signal_found = "MATCHED SETUP" if live_sigs else "SCANNING FOR PATTERN"
        
        if live_sigs and st.session_state.simulated_position is None:
            latest = live_sigs[-1]
            direction = latest[1]
            sl_calc = ltp - custom_sl_input if direction == "BUY" else ltp + custom_sl_input
            tp_calc = ltp + (custom_sl_input * target_multiplier) if direction == "BUY" else ltp - (custom_sl_input * target_multiplier)
            
            st.session_state.simulated_position = {
                "Asset": ticker_symbol, "Time": datetime.now().strftime("%H:%M:%S"),
                "Direction": direction, "Entry": ltp, "SL": round(sl_calc, 2), "TP": round(tp_calc, 2),
                "Status": "OPEN"
            }
            st.toast(f"Executed live structural position logic framework: {direction}", icon="📈")

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
                st.toast("Position hit execution target boundary limit cleanly.", icon="📥")

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
            - **SL Configuration Mode**: `{sl_mode}` (`{custom_sl_input} Pts`)
            - **TP Configuration Mode**: `{tp_mode}` (`{custom_tp_input} Pts`)
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
    
    # Human Readable Strategy Blueprint Explainer Block
    with st.expander("📖 System Blueprint: How the 15-Minute ORB Strategy Works", expanded=True):
        st.markdown("""
        The **Opening Range Breakout (ORB)** is an institutional high-momentum trading strategy designed to catch the intraday trend direction set during initial market volatility.
        
        **Execution Rules & Filtration Steps:**
        1. **Establish the Boundaries**: The engine isolates the **very first 15-minute candlestick** of the current market session to lock in the **Opening Range High** and **Opening Range Low**.
        2. **Scanning for Breakout/Breakdown**: 
           - **Bullish Setup**: If any subsequent intraday candlestick **closes completely above** the 15-minute Opening High, a buy trigger is flagged.
           - **Bearish Setup**: If any subsequent intraday candlestick **closes completely below** the 15-minute Opening Low, a short-sell trigger is flagged.
        3. **Risk & Target Structural Assignment**:
           - **Trigger Entry**: Set at the broken High or Low boundary lines.
           - **Stop Loss (SL)**: Set at the opposite side of the initial 15-minute range (Opening Low for Buys, Opening High for Sells).
           - **Target Profit Line**: Extrapolated mechanically out using your selected asymmetric Risk-to-Reward multiplier matrix configuration (`1:1` up to `1:5`).
        """)

    st.markdown("### 🎛️ Screener Engine Matrix Controls")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        scr_sector = st.selectbox("Screener Segment Sector Filter", ["ALL Nifty 50 Stocks", "Custom Asset Input"] + list(NIFTY50_SECTORS.keys()))
    with sc2:
        orb_target_rr = st.selectbox("Execution Target Risk-Reward Metric", [
            "1:1 Risk-Reward Balance", "1:2 Target Layout", "1:3 Matrix Risk Extension", 
            "1:4 Risk Matrix Acceleration", "1:5 Advanced Asymmetric Target"
        ], index=1)
    with sc3:
        direction_filter = st.selectbox("Breakout Direction Lookups", ["All Breakouts", "Bullish Breakouts Only", "Bearish Breakouts Only"])

    # Independent Timeframe and Period Config Layout Blocks for Screener Module Isolation
    sc4, sc5, sc6 = st.columns(3)
    with sc4:
        scr_interval = st.selectbox("Screener Candlestick Frame (ORB Base)", ["1m", "5m", "15m", "1h", "1d"], index=2, key="scr_tf")
    with sc5:
        scr_period = st.selectbox("Screener Data Fetch Lookback", ["1d", "5d", "1mo", "3mo"], index=1, key="scr_per")
    with sc6:
        custom_scr_ticker = st.text_input("Custom Ticker Code Entry (Active if Custom Asset Selected)", "RELIANCE.NS")

    # Resolve target ticker asset pools matching configuration selections
    if scr_sector == "ALL Nifty 50 Stocks":
        scan_pool = ALL_NIFTY_STOCKS
    elif scr_sector == "Custom Asset Input":
        scan_pool = {"Custom Asset": custom_scr_ticker.strip()}
    else:
        scan_pool = NIFTY50_SECTORS[scr_sector]

    # Map selected Risk-to-Reward matrix multiplier indexes
    rr_map = {"1:1": 1.0, "1:2": 2.0, "1:3": 3.0, "1:4": 4.0, "1:5": 5.0}
    rr_multiplier = 2.0  # Default fallback
    for key, val in rr_map.items():
        if key in orb_target_rr:
            rr_multiplier = val
            break

    if st.button("🚀 EXECUTE AUTONOMOUS STRATEGY MATRIX DISCOVERY", use_container_width=True):
        screener_records = []
        progress_slot = st.progress(0)
        status_slot = st.empty()
        pool_len = len(scan_pool)
        
        for idx, (name, symbol) in enumerate(scan_pool.items()):
            status_slot.text(f"Processing Array Node [{idx+1}/{pool_len}]: Querying Data Stream for {name} ({symbol})...")
            progress_slot.progress((idx + 1) / pool_len)
            
            # Request intraday history rows dynamically using decoupled parameters
            hist_df = fetch_ticker_data(symbol, scr_period, scr_interval)
            if hist_df.empty or len(hist_df) < 5:
                continue
                
            last_date = hist_df.index[-1].date()
            day_df = hist_df[hist_df.index.date == last_date]
            if len(day_df) < 2: 
                continue
                
            first_candle = day_df.iloc[0]
            orb_high = float(first_candle['High'])
            orb_low = float(first_candle['Low'])
            ltp = float(day_df['Close'].iloc[-1])
            
            setup_status = "Neutral"
            entry_ref = 0.0
            sl_val = 0.0
            tp_val = 0.0
            
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
            def highlight_orb_status(val):
                if val == "BULLISH BREAKOUT": return "background-color: rgba(0, 128, 0, 0.15); color: green; font-weight: bold;"
                if val == "BEARISH BREAKDOWN": return "background-color: rgba(255, 0, 0, 0.15); color: red; font-weight: bold;"
                return ""
            styled_output = res_df.style.map(highlight_orb_status, subset=["ORB Status"])
            st.dataframe(styled_output, use_container_width=True)
        else:
            st.info("No underlying assets triggered an internal opening range break condition during this historical tracking pass.")
