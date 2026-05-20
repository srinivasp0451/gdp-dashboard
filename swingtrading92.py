import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Smart Money Institutional Trading System")

# --- INITIALIZE SESSION STATE FOR TRADE HISTORY ---
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "active_trades" not in st.session_state:
    st.session_state.active_trades = {}

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
def calculate_sl_tp(df, idx, strategy_type, sl_type, tp_type, custom_sl, custom_tp, direction):
    row = df.iloc[idx]
    close = row['Close']
    atr = row['ATR_14'] if 'ATR_14' in df.columns and not pd.isna(row['ATR_14']) else close * 0.01
    
    # Base SL calculation
    if sl_type == "Custom SL" or sl_type == "Trailing SL":
        sl_val = close - custom_sl if direction == "BUY" else close + custom_sl
    elif sl_type == "ATR based SL":
        sl_val = close - (1.5 * atr) if direction == "BUY" else close + (1.5 * atr)
    elif sl_type == "Signal based SL" or sl_type == "Logical SL":
        if direction == "BUY":
            sl_val = row['Rolling_Low'] if not pd.isna(row['Rolling_Low']) else close * 0.99
        else:
            sl_val = row['Rolling_High'] if not pd.isna(row['Rolling_High']) else close * 1.01
    elif sl_type == "Support Resistance based SL":
        sl_val = row['Rolling_Low'] if direction == "BUY" else row['Rolling_High']
    else:
        sl_val = close * 0.99 if direction == "BUY" else close * 1.01
        
    # Base TP calculation
    if tp_type == "Custom Target" or tp_type == "Trailing Target":
        tp1 = close + custom_tp if direction == "BUY" else close - custom_tp
    elif tp_type == "ATR based Target":
        tp1 = close + (3 * atr) if direction == "BUY" else close - (3 * atr)
    elif tp_type == "Signal based Target" or tp_type == "Logical Target" or tp_type == "Support Resistance based Target":
        if direction == "BUY":
            tp1 = row['Rolling_High'] if not pd.isna(row['Rolling_High']) else close * 1.02
        else:
            tp1 = row['Rolling_Low'] if not pd.isna(row['Rolling_Low']) else close * 0.98
    else:
        tp1 = close * 1.02 if direction == "BUY" else close * 0.98
        
    return round(float(sl_val), 2), round(float(tp1), 2)

# --- BACKTEST PNL SIMULATOR ---
def simulate_backtest_pnl(processed_df, found_signals, sl_mode, tp_mode, custom_sl_input, custom_tp_input):
    records = []
    for sig_idx, (timestamp, sig_type, pattern, entry_p, conf) in enumerate(found_signals):
        try:
            idx_pos = processed_df.index.get_loc(timestamp)
            if isinstance(idx_pos, slice):
                idx_pos = idx_pos.start
        except KeyError:
            continue
            
        sl, tp = calculate_sl_tp(processed_df, idx_pos, strategy_choice, sl_mode, tp_mode, custom_sl_input, custom_tp_input, sig_type)
        
        # Look ahead to calculate simulated PnL outcome
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
                    # 70% Book near TP1 rule adjustment calculation
                    pnl = (tp - entry_p)
                    exit_price = tp
                    status = "Closed (Target)"
                    break
            elif sig_type == "SELL":
                if f_row['High'] <= sl: # Fixed index short mapping logic
                    pnl = entry_p - sl
                    exit_price = sl
                    status = "Closed (Stop Loss)"
                    break
                elif f_row['Low'] >= tp:
                    pnl = entry_p - tp
                    exit_price = tp
                    status = "Closed (Target)"
                    break
                    
        # If trade is still running by end of dataset
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
            "Confidence Score": f"{conf}%"
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
                signals.append((df.index[i], "SELL", "Bull Trap", row['Close'], score))
                
        elif is_bear_breakdown and large_range and weak_volume:
            if row['Close'] > prev_row['Rolling_Low']:
                score = 65
                if row['Low'] < prev_row['Low']: score += 15
                if row['Close'] > row['Open']: score += 10
                signals.append((df.index[i], "BUY", "Bear Trap", row['Close'], score))
                
    return df, signals

# --- ALTERNATIVE STRATEGY RUNNERS ---
def run_alternative_strategies(df, strategy_name):
    df = calculate_indicators(df)
    signals = []
    if len(df) < 20: return df, signals
    
    if strategy_name == "EMA Crossover Strategy":
        for i in range(1, len(df)):
            if df['EMA_9'].iloc[i-1] <= df['EMA_15'].iloc[i-1] and df['EMA_9'].iloc[i] > df['EMA_15'].iloc[i]:
                signals.append((df.index[i], "BUY", "EMA Gold Cross", df['Close'].iloc[i], 80))
            elif df['EMA_9'].iloc[i-1] >= df['EMA_15'].iloc[i-1] and df['EMA_9'].iloc[i] < df['EMA_15'].iloc[i]:
                signals.append((df.index[i], "SELL", "EMA Death Cross", df['Close'].iloc[i], 80))
                
    elif strategy_name == "RSI Overbought Oversold Strategy":
        for i in range(1, len(df)):
            if df['RSI_14'].iloc[i-1] >= 30 and df['RSI_14'].iloc[i] < 30:
                signals.append((df.index[i], "BUY", "RSI Oversold Pivot", df['Close'].iloc[i], 70))
            elif df['RSI_14'].iloc[i-1] <= 70 and df['RSI_14'].iloc[i] > 70:
                signals.append((df.index[i], "SELL", "RSI Overbought Pivot", df['Close'].iloc[i], 70))
                
    elif strategy_name == "VWAP Based Strategy":
        for i in range(1, len(df)):
            if df['Close'].iloc[i-1] <= df['VWAP'].iloc[i-1] and df['Close'].iloc[i] > df['VWAP'].iloc[i]:
                signals.append((df.index[i], "BUY", "VWAP Reclaim", df['Close'].iloc[i], 75))
            elif df['Close'].iloc[i-1] >= df['VWAP'].iloc[i-1] and df['Close'].iloc[i] < df['VWAP'].iloc[i]:
                signals.append((df.index[i], "SELL", "VWAP Breakdown", df['Close'].iloc[i], 75))
                
    return df, signals

# --- FETCH DATA WITH RATE LIMIT SAFETY & MULTI-INDEX FIX ---
def fetch_ticker_data(ticker, period, interval):
    time.sleep(1.0) # Rate limit security buffer rule
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return pd.DataFrame()
            
        # Hard Fix for Multi-Indexed data structure drops to stop Fragment crashes
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Clean duplicate column structures or formatting mismatches safely
        data = data.loc[:, ~data.columns.duplicated()].copy()
        
        # Ensure structural float mappings
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
        return data.dropna(subset=['Close'])
    except Exception as e:
        st.error(f"Error fetching data from yfinance API: {e}")
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
tab1, tab2, tab3 = st.tabs(["📊 Backtesting & Signal History", "⚡ Live Micro-Execution Framework", "📜 System Trade Ledger"])

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
                    # Run comprehensive Backtest Log with clear simulated performance data outputs
                    simulated_records = simulate_backtest_pnl(processed_df, found_signals, sl_mode, tp_mode, custom_sl_input, custom_tp_input)
                    df_rec = pd.DataFrame(simulated_records)
                    
                    # Calculate cumulative backtest performance summary matrix values
                    total_pnl = df_rec["Simulated PnL (Pts)"].sum()
                    win_trades = len(df_rec[df_rec["Simulated PnL (Pts)"] > 0])
                    win_rate = (win_trades / len(df_rec)) * 100
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Net Aggregated PnL", f"{total_pnl:.2f} Pts", delta=f"{total_pnl:.2f}")
                    c2.metric("Total Generated Trades", len(df_rec))
                    c3.metric("Win-Rate Ratio Profile", f"{win_rate:.1f}%")
                    
                    st.dataframe(df_rec, use_container_width=True)
                    
                    # Plotly chart canvas setup
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
    st.caption("Auto-refresh cycle uses optimized isolated fragments to calculate Real-time Running PnL metrics safely.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        is_live = st.toggle("Activate Live Tracking Feeds", value=False)
        st.metric("Monitoring Target Asset", ticker_symbol)
        st.metric("Assigned Logic Routine", strategy_choice)
        
        # Cumulative running summary inside panel
        live_pnl_placeholder = st.empty()
    
    # Isolated streaming component to mitigate full UI redraw overhead
    @st.fragment(run_every=1.0 if is_live else None)
    def live_streaming_fragment():
        if not is_live:
            st.info("Switch the toggle above to connect live streaming engines.")
            return
            
        # Data Pull
        live_df = fetch_ticker_data(ticker_symbol, "5d", interval_choice)
        if live_df.empty:
            st.error("No active market telemetry lines returned.")
            return
            
        if strategy_choice == "Institutional Trap Strategy":
            processed_df, found_signals = run_trap_strategy(live_df)
        else:
            processed_df, found_signals = run_alternative_strategies(live_df, strategy_choice)
            
        current_ltp = float(processed_df['Close'].iloc[-1])
        highest_p = float(processed_df['High'].max())
        lowest_p = float(processed_df['Low'].min())
        
        st.subheader("Telemetry Status Blocks")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Current Asset LTP", f"{current_ltp:.2f}")
        m_col2.metric("Frame Highest / Lowest", f"{highest_p:.2f} / {lowest_p:.2f}")
        
        # Determine tracking states safely without .get_loc dependencies
        current_signal = "NO TRADE"
        pattern_detected = "None"
        entry_price = 0.0
        calculated_sl = 0.0
        calculated_tp = 0.0
        confidence = 0
        
        if found_signals:
            last_sig = found_signals[-1]
            # Verify recency via relative index matrix offsets instead of string checks to stop crash triggers
            if len(processed_df) - processed_df.index.get_indexer([last_sig[0]])[0] <= 4:
                current_signal = last_sig[1]
                pattern_detected = last_sig[2]
                entry_price = float(last_sig[3])
                confidence = last_sig[4]
                
                # Safe position indexing parsing fallback
                idx_pos = int(processed_df.index.get_indexer([last_sig[0]])[0])
                calculated_sl, calculated_tp = calculate_sl_tp(
                    processed_df, idx_pos, strategy_choice, sl_mode, tp_mode, custom_sl_input, custom_tp_input, current_signal
                )
        
        # Dynamically evaluate open running active positions to update running Session Ledger states
        for trade in st.session_state.trade_history:
            if trade["Ticker"] == ticker_symbol and trade["System Status"] == "ACTIVE":
                # Real-Time Live Running PnL Equation Calculation Block
                if trade["Type"] == "BUY":
                    running_pnl = current_ltp - trade["Entry Fill"]
                    # Risk validation boundary breaches
                    if current_ltp <= trade["Hard Stop"]:
                        trade["System Status"] = "CLOSED (SL hit)"
                        trade["Live Running PnL (Pts)"] = trade["Hard Stop"] - trade["Entry Fill"]
                    elif current_ltp >= trade["Primary Target"]:
                        trade["System Status"] = "CLOSED (TP hit)"
                        trade["Live Running PnL (Pts)"] = trade["Primary Target"] - trade["Entry Fill"]
                    else:
                        trade["Live Running PnL (Pts)"] = round(running_pnl, 2)
                else: # Short trade mapping execution rules path
                    running_pnl = trade["Entry Fill"] - current_ltp
                    if current_ltp >= trade["Hard Stop"]:
                        trade["System Status"] = "CLOSED (SL hit)"
                        trade["Live Running PnL (Pts)"] = trade["Entry Fill"] - trade["Hard Stop"]
                    elif current_ltp <= trade["Primary Target"]:
                        trade["System Status"] = "CLOSED (TP hit)"
                        trade["Live Running PnL (Pts)"] = trade["Entry Fill"] - trade["Primary Target"]
                    else:
                        trade["Live Running PnL (Pts)"] = round(running_pnl, 2)

        # Update Sidebar/Main component display metric for Cumulative Profile Net Tracking
        if st.session_state.trade_history:
            net_live_sum = sum([t.get("Live Running PnL (Pts)", 0.0) for t in st.session_state.trade_history])
            live_pnl_placeholder.metric("Net Total Live Session PnL", f"{net_live_sum:.2f} Pts")
        else:
            live_pnl_placeholder.metric("Net Total Live Session PnL", "0.00 Pts")

        # Display Live Execution Metrics Matrix
        with col2:
            st.subheader("Dynamic Execution Matrix")
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.markdown(f"**Signal Type:** `{current_signal}`")
            s_col2.markdown(f"**Pattern Trigger:** `{pattern_detected}`")
            s_col3.markdown(f"**Target Entry:** `{entry_price:.2f}`")
            s_col4.markdown(f"**Confidence Matrix:** `{confidence}%`")
            
            s_col1.markdown(f"**Risk Floor (SL):** `{calculated_sl:.2f}`")
            s_col2.markdown(f"**Dynamic Target (TP1):** `{calculated_tp:.2f}`")
            
            # Active Position Logic Loop
            if current_signal in ["BUY", "SELL"] and entry_price > 0:
                trade_key = f"{ticker_symbol}_{last_sig[0].strftime('%H%M%S')}"
                if trade_key not in st.session_state.active_trades:
                    st.session_state.active_trades[trade_key] = True
                    new_trade_entry = {
                        "Timestamp": last_sig[0].strftime('%Y-%m-%d %H:%M:%S'),
                        "Ticker": ticker_symbol,
                        "Type": current_signal,
                        "Strategy Pattern": pattern_detected,
                        "Entry Fill": round(entry_price, 2),
                        "Hard Stop": round(calculated_sl, 2),
                        "Primary Target": round(calculated_tp, 2),
                        "Live Running PnL (Pts)": 0.0,
                        "System Status": "ACTIVE"
                    }
                    st.session_state.trade_history.append(new_trade_entry)
                    st.toast(f"New Order Filled: {current_signal} {ticker_symbol} @ {entry_price}", icon="🚀")
            
            # Render Live Candlestick Graph Map Engine Window
            fig_live = go.Figure()
            plot_df = processed_df.tail(40)
            fig_live.add_trace(go.Candlestick(
                x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'], name="Live Candles"
            ))
            if 'EMA_9' in plot_df.columns:
                fig_live.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_9'], line=dict(color='orange', width=1.5), name='9 EMA'))
            if 'EMA_15' in plot_df.columns:
                fig_live.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_15'], line=dict(color='blue', width=1.5), name='15 EMA'))
            
            fig_live.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_live, use_container_width=True, key="live_plotly_chart")
            
    live_streaming_fragment()

# ==================== TAB 3: SYSTEM TRADE LEDGER ====================
with tab3:
    st.header("Operational Signal & Trade Execution Ledger")
    st.caption("All executions captured dynamically inside running streaming frameworks are formatted below with instant calculated realization targets.")
    
    if len(st.session_state.trade_history) == 0:
        st.info("No active trades recorded in this session yet.")
    else:
        ledger_df = pd.DataFrame(st.session_state.trade_history)
        
        # Display formatted trade logs
        st.dataframe(ledger_df, use_container_width=True)
        
        # Download ledger functionality
        csv_data = ledger_df.to_csv(index=False).encode('utf-8')
        st.download_button("Export Order Ledger (.CSV)", csv_data, "institutional_trade_ledger.csv", "text/csv")
