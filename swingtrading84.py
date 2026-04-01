import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as plotly_go

# ==========================================
# PAGE CONFIGURATION & STATE INITIALIZATION
# ==========================================
st.set_page_config(page_title="Smart Investing", layout="wide")

# Initialize persistent history if not exists
if "trade_history" not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=[
        "Entry Time", "Exit Time", "Type", "Entry Price", "Exit Price", 
        "SL", "Target", "High", "Low", "PnL", "Result", "Reason"
    ])
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "current_position" not in st.session_state:
    st.session_state.current_position = None
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = 0

# ==========================================
# UTILITIES
# ==========================================
TICKER_MAP = {
    "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "BTC": "BTC-USD", "ETH": "ETH-USD", "GOLD": "GC=F", "SILVER": "SI=F", "CUSTOM": "CUSTOM"
}

TIMEFRAME_PERIODS = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["1d", "5d", "7d", "1mo"],
    "15m": ["1d", "5d", "7d", "1mo"],
    "1h": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y"]
}

def fetch_data(ticker, period, interval):
    yf_ticker = st.session_state.get('custom_ticker_input', 'RELIANCE.NS') if ticker == "CUSTOM" else TICKER_MAP[ticker]
    try:
        df = yf.download(yf_ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

def calculate_indicators(df, fast_ema, slow_ema):
    df['EMA_Fast'] = df['Close'].ewm(span=fast_ema, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_ema, adjust=False).mean()
    df['Buy_Signal'] = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
    df['Sell_Signal'] = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.selectbox("Select Ticker", list(TICKER_MAP.keys()))
if ticker == "CUSTOM":
    st.session_state.custom_ticker_input = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")

interval = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()))
period = st.sidebar.selectbox("Period", TIMEFRAME_PERIODS[interval])
qty = st.sidebar.number_input("Quantity", value=1, min_value=1)
strategy = st.sidebar.selectbox("Strategy", ["EMA Crossover", "Simple Buy", "Simple Sell"])
fast_ema_val = st.sidebar.number_input("Fast EMA", value=9)
slow_ema_val = st.sidebar.number_input("Slow EMA", value=15)

sl_val = st.sidebar.number_input("SL Points", value=10.0)
tgt_val = st.sidebar.number_input("Target Points", value=20.0)

st.sidebar.markdown("---")
use_dhan = st.sidebar.checkbox("Enable Dhan Broker", value=False)

# ==========================================
# MAIN UI
# ==========================================
st.title("📈 Smart Investing")
tab1, tab2, tab3 = st.tabs(["📊 Backtesting", "🔴 Live Trading", "📝 Trade History"])

# ------------------------------------------
# TAB 1: BACKTESTING
# ------------------------------------------
with tab1:
    if st.button("Run Backtest"):
        df_bt = fetch_data(ticker, period, interval)
        if df_bt is not None:
            df_bt = calculate_indicators(df_bt, fast_ema_val, slow_ema_val)
            trades, in_pos, pos_type = [], False, None
            entry_p, entry_t, sl_p, tgt_p = 0, None, 0, 0
            
            for idx, row in df_bt.iterrows():
                if in_pos:
                    exit_p = 0
                    if pos_type == 'Buy':
                        if row['Low'] <= sl_p: exit_p, reason = sl_p, "SL Hit"
                        elif row['High'] >= tgt_p: exit_p, reason = tgt_p, "Target Hit"
                    else:
                        if row['High'] >= sl_p: exit_p, reason = sl_p, "SL Hit"
                        elif row['Low'] <= tgt_p: exit_p, reason = tgt_p, "Target Hit"
                    
                    if exit_p != 0:
                        pnl_pts = (exit_p - entry_p) if pos_type == 'Buy' else (entry_p - exit_p)
                        trades.append({
                            "Entry Time": entry_t, "Exit Time": idx, "Type": pos_type, 
                            "Entry Price": entry_p, "Exit Price": exit_p, "SL": sl_p, "Target": tgt_p,
                            "High": row['High'], "Low": row['Low'], "Points": pnl_pts, "PnL": pnl_pts * qty, 
                            "Result": "Win" if pnl_pts > 0 else "Loss"
                        })
                        in_pos = False

                if not in_pos:
                    sig = None
                    if strategy == "Simple Buy": sig = 'Buy'
                    elif strategy == "Simple Sell": sig = 'Sell'
                    elif row['Buy_Signal']: sig = 'Buy'
                    elif row['Sell_Signal']: sig = 'Sell'
                    
                    if sig:
                        in_pos, pos_type, entry_p, entry_t = True, sig, row['Close'], idx
                        m = 1 if sig == 'Buy' else -1
                        sl_p = entry_p - (sl_val * m)
                        tgt_p = entry_p + (tgt_val * m)

            if trades:
                tdf = pd.DataFrame(trades)
                # Summary Calculations
                win_trades = tdf[tdf['Points'] > 0]
                loss_trades = tdf[tdf['Points'] <= 0]
                
                st.subheader("Backtesting Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Trades", len(tdf))
                c2.metric("Accuracy", f"{(len(win_trades)/len(tdf))*100:.2f}%")
                c3.metric("Total PnL (Pts)", f"{tdf['Points'].sum():.2f}")
                c4.metric("Total PnL (Amt)", f"{tdf['PnL'].sum():.2f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Profit Trades", len(win_trades))
                c6.metric("Loss Trades", len(loss_trades))
                c7.metric("Avg Profit (Pts)", f"{win_trades['Points'].mean():.2f}" if not win_trades.empty else "0")
                c8.metric("Avg Loss (Pts)", f"{loss_trades['Points'].mean():.2f}" if not loss_trades.empty else "0")

                st.markdown("---")
                st.subheader("Detailed Results")
                st.dataframe(tdf, use_container_width=True)

# ------------------------------------------
# TAB 2: LIVE TRADING
# ------------------------------------------
with tab2:
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Live", use_container_width=True): st.session_state.live_running = True
    if col2.button("🛑 Stop Live", use_container_width=True): st.session_state.live_running = False
    
    live_dashboard = st.empty()
    live_chart = st.empty()
    
    if st.session_state.live_running:
        while st.session_state.live_running:
            df_l = fetch_data(ticker, "1d", interval)
            if df_l is not None:
                df_l = calculate_indicators(df_l, fast_ema_val, slow_ema_val)
                latest = df_l.iloc[-1]
                cur_p = float(latest['Close'])
                
                with live_dashboard.container():
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("LTP", f"{cur_p:.2f}")
                    c2.metric(f"EMA {fast_ema_val}", f"{latest['EMA_Fast']:.2f}")
                    c3.metric(f"EMA {slow_ema_val}", f"{latest['EMA_Slow']:.2f}")
                    
                    # Entry Logic
                    if st.session_state.current_position is None:
                        trigger = False
                        if "Simple" in strategy:
                            trigger, sig = True, ('Buy' if "Buy" in strategy else 'Sell')
                        else:
                            # Candle-close logic for EMA
                            min_now = datetime.now().minute
                            int_num = int(''.join(filter(str.isdigit, interval))) if interval[0].isdigit() else 1
                            if min_now % int_num == 0:
                                if latest['Buy_Signal']: sig, trigger = 'Buy', True
                                elif latest['Sell_Signal']: sig, trigger = 'Sell', True
                        
                        if trigger:
                            m = 1 if sig == 'Buy' else -1
                            st.session_state.current_position = {
                                'type': sig, 'price': cur_p, 'time': datetime.now(), 
                                'sl': cur_p - (sl_val * m), 'tgt': cur_p + (tgt_val * m)
                            }
                            st.toast(f"Entered {sig} at {cur_p}")

                    # Position Tracking
                    if st.session_state.current_position:
                        pos = st.session_state.current_position
                        pnl = (cur_p - pos['price']) * qty if pos['type'] == 'Buy' else (pos['price'] - cur_p) * qty
                        color = "green" if pnl >= 0 else "red"
                        
                        st.markdown(f"### Live PnL: <span style='color:{color};'>{pnl:.2f}</span>", unsafe_allow_html=True)
                        st.write(f"**Trade Info:** {pos['type']} | Entry: {pos['price']} | SL: {pos['sl']:.2f} | TGT: {pos['tgt']:.2f}")
                        
                        # Exit Logic
                        exit_now = False
                        if pos['type'] == 'Buy':
                            if cur_p <= pos['sl']: exit_now, res, reason = True, "Loss", "SL Hit"
                            elif cur_p >= pos['tgt']: exit_now, res, reason = True, "Win", "Target Hit"
                        else:
                            if cur_p >= pos['sl']: exit_now, res, reason = True, "Loss", "SL Hit"
                            elif cur_p <= pos['tgt']: exit_now, res, reason = True, "Win", "Target Hit"
                        
                        if exit_now:
                            new_trade = {
                                "Entry Time": pos['time'], "Exit Time": datetime.now(), "Type": pos['type'],
                                "Entry Price": pos['price'], "Exit Price": cur_p, "SL": pos['sl'], "Target": pos['tgt'],
                                "High": latest['High'], "Low": latest['Low'], "PnL": pnl, "Result": res, "Reason": reason
                            }
                            # Save to persistent history
                            st.session_state.trade_history = pd.concat([st.session_state.trade_history, pd.DataFrame([new_trade])], ignore_index=True)
                            # CLEAR SESSION PROPERLY
                            st.session_state.current_position = None
                            st.toast(f"Trade Closed: {reason}")
                            st.rerun() # Forces UI refresh to clear PnL display immediately
                    else:
                        st.info("Scanning for next signal...")

                with live_chart.container():
                    fig = plotly_go.Figure()
                    df_plot = df_l.tail(30)
                    fig.add_trace(plotly_go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"))
                    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{time.time()}")
            
            time.sleep(1.5)

# ------------------------------------------
# TAB 3: TRADE HISTORY
# ------------------------------------------
with tab3:
    st.subheader("Trade Ledger (Live Session)")
    if not st.session_state.trade_history.empty:
        # Use newer .map for styling to avoid deprecation errors
        def style_pnl(val):
            return 'color: green' if val > 0 else 'color: red'
        
        st.dataframe(st.session_state.trade_history.style.applymap(style_pnl, subset=['PnL']), use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.trade_history = pd.DataFrame(columns=st.session_state.trade_history.columns)
            st.rerun()
    else:
        st.info("No trades saved. History will populate once a Live Trade is closed.")
