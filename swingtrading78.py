import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import itertools

st.set_page_config(layout="wide", page_title="Algo Trading Dashboard")

# --- GLOBAL SETTINGS & DICTIONARIES ---
TICKERS = {
    'Nifty 50': '^NSEI', 'Bank Nifty': '^NSEBANK', 'Sensex': '^BSESN',
    'BTC/USD': 'BTC-USD', 'ETH/USD': 'ETH-USD', 'USD/INR': 'INR=X',
    'Gold': 'GC=F', 'Silver': 'SI=F', 'EUR/USD': 'EURUSD=X', 'Custom': 'CUSTOM'
}

STRATEGIES = [
    'EMA Crossover', 'RSI Overbought/Oversold', 'Simple Buy', 'Simple Sell', 
    'Price Threshold', 'Bollinger Bands'
]

SL_TYPES = ['Custom Points', 'Trailing SL (Points)', 'Trailing Prev Candle Low/High', 'ATR Based']
TP_TYPES = ['Custom Points', 'Trailing Target (Never Hit)', 'Risk Reward Based']

# --- HELPER FUNCTIONS ---
def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Ensure standard OHLCV columns exist
        cols = [c.capitalize() for c in df.columns]
        df.columns = cols
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def apply_strategy(df, strategy, p1, p2):
    df = df.copy()
    df['Signal'] = 0 # 1 for Buy, -1 for Sell
    
    if strategy == 'EMA Crossover':
        df['EMA_Fast'] = df['Close'].ewm(span=p1, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=p2, adjust=False).mean()
        # Signal on Close
        buy_cond = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
        sell_cond = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))
        df.loc[buy_cond, 'Signal'] = 1
        df.loc[sell_cond, 'Signal'] = -1

    elif strategy == 'RSI Overbought/Oversold':
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # p1 = Oversold (Buy), p2 = Overbought (Sell)
        buy_cond = (df['RSI'] < p1) & (df['RSI'].shift(1) >= p1)
        sell_cond = (df['RSI'] > p2) & (df['RSI'].shift(1) <= p2)
        df.loc[buy_cond, 'Signal'] = 1
        df.loc[sell_cond, 'Signal'] = -1
        
    elif strategy == 'Simple Buy':
        df['Signal'] = 1 # Constant buy pressure (for testing)
        
    elif strategy == 'Bollinger Bands':
        df['SMA'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['SMA'] + (df['STD'] * 2)
        df['Lower'] = df['SMA'] - (df['STD'] * 2)
        df.loc[df['Close'] < df['Lower'], 'Signal'] = 1
        df.loc[df['Close'] > df['Upper'], 'Signal'] = -1
        
    # ATR Calculation for SL/TP
    df['TR'] = np.maximum((df['High'] - df['Low']), 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    return df

def run_backtest(df, sl_type, sl_val, tp_type, tp_val):
    trades = []
    in_position = False
    
    i = 0
    while i < len(df) - 1:
        if not in_position:
            signal = df['Signal'].iloc[i]
            if signal != 0:
                # ENTRY LOGIC: Entry at OPEN of bar i+1
                entry_idx = i + 1
                entry_price = df['Open'].iloc[entry_idx]
                pos_type = signal
                in_position = True
                entry_time = df.index[entry_idx]
                
                # Calculate Initial SL and TP
                if pos_type == 1: # LONG
                    sl = entry_price - sl_val if sl_type == 'Custom Points' else entry_price - df['ATR'].iloc[entry_idx]
                    tp = entry_price + tp_val if tp_type == 'Custom Points' else entry_price + (sl_val * tp_val) # Assuming tp_val is RR ratio
                else: # SHORT
                    sl = entry_price + sl_val if sl_type == 'Custom Points' else entry_price + df['ATR'].iloc[entry_idx]
                    tp = entry_price - tp_val if tp_type == 'Custom Points' else entry_price - (sl_val * tp_val)

                highest_price = entry_price
                lowest_price = entry_price
                
                # EXIT LOGIC: Checking starts from i+1 onward
                for j in range(entry_idx, len(df)):
                    high = df['High'].iloc[j]
                    low = df['Low'].iloc[j]
                    
                    highest_price = max(highest_price, high)
                    lowest_price = min(lowest_price, low)
                    
                    exit_reason = None
                    exit_price = 0
                    
                    # Check SL and TP (Conservative: Check SL first)
                    if pos_type == 1: # LONG
                        if low <= sl:
                            exit_reason = 'SL Hit'
                            exit_price = sl
                        elif high >= tp and tp_type != 'Trailing Target (Never Hit)':
                            exit_reason = 'Target Hit'
                            exit_price = tp
                    else: # SHORT
                        if high >= sl:
                            exit_reason = 'SL Hit'
                            exit_price = sl
                        elif low <= tp and tp_type != 'Trailing Target (Never Hit)':
                            exit_reason = 'Target Hit'
                            exit_price = tp
                            
                    # Update Trailing SL if applicable
                    if sl_type == 'Trailing SL (Points)' and not exit_reason:
                        if pos_type == 1 and high - sl_val > sl:
                            sl = high - sl_val
                        elif pos_type == -1 and low + sl_val < sl:
                            sl = low + sl_val
                            
                    if exit_reason:
                        pts_gained = (exit_price - entry_price) if pos_type == 1 else (entry_price - exit_price)
                        trades.append({
                            'Entry Time': entry_time, 'Exit Time': df.index[j],
                            'Type': 'LONG' if pos_type == 1 else 'SHORT',
                            'Entry Price': entry_price, 'Exit Price': exit_price,
                            'Highest': highest_price, 'Lowest': lowest_price,
                            'Reason': exit_reason, 'PnL Points': pts_gained
                        })
                        in_position = False
                        i = j # Move pointer to exit candle
                        break
                
                if in_position: # End of data reached without exit
                    i = len(df)
            else:
                i += 1
        else:
            i += 1

    return pd.DataFrame(trades)

# --- SIDEBAR UI ---
st.sidebar.header("Strategy Configuration")

ticker_key = st.sidebar.selectbox("Select Asset", list(TICKERS.keys()))
if ticker_key == 'Custom':
    symbol = st.sidebar.text_input("Enter Custom Ticker (e.g., AAPL)")
else:
    symbol = TICKERS[ticker_key]

period = st.sidebar.selectbox("Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'])
interval = st.sidebar.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '1d', '1wk'])

strategy = st.sidebar.selectbox("Strategy", STRATEGIES)

if strategy == 'EMA Crossover':
    param1 = st.sidebar.number_input("Fast EMA", value=9)
    param2 = st.sidebar.number_input("Slow EMA", value=15)
elif strategy == 'RSI Overbought/Oversold':
    param1 = st.sidebar.number_input("Oversold (Buy)", value=30)
    param2 = st.sidebar.number_input("Overbought (Sell)", value=70)
else:
    param1, param2 = 0, 0 # Placeholders

sl_type = st.sidebar.selectbox("Stop Loss Type", SL_TYPES)
sl_val = st.sidebar.number_input("SL Value (Points/ATR)", value=10.0)

tp_type = st.sidebar.selectbox("Target Type", TP_TYPES)
tp_val = st.sidebar.number_input("Target Value (Points/Ratio)", value=20.0)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Backtesting", "🔴 Live Trading", "⚙️ Optimization"])

df_raw = fetch_data(symbol, period, interval)

# --- TAB 1: BACKTESTING ---
with tab1:
    if df_raw is not None:
        st.subheader(f"Backtest Results: {symbol} | {interval}")
        df_strat = apply_strategy(df_raw, strategy, param1, param2)
        trades_df = run_backtest(df_strat, sl_type, sl_val, tp_type, tp_val)
        
        # Metrics
        if not trades_df.empty:
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['PnL Points'] > 0])
            accuracy = (winning_trades / total_trades) * 100
            total_pnl = trades_df['PnL Points'].sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total PnL (Points)", f"{total_pnl:.2f}")
            col2.metric("Total Trades", total_trades)
            col3.metric("Accuracy", f"{accuracy:.2f}%")
            
            st.dataframe(trades_df, use_container_width=True)
            
            # Plotly Chart
            fig = go.Figure(data=[go.Candlestick(x=df_strat.index,
                            open=df_strat['Open'], high=df_strat['High'],
                            low=df_strat['Low'], close=df_strat['Close'], name='OHLC')])
            
            if strategy == 'EMA Crossover':
                fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_Fast'], mode='lines', name='Fast EMA'))
                fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_Slow'], mode='lines', name='Slow EMA'))
                
            fig.update_layout(title='Candlestick Chart with Indicators', xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Original OHLC Data")
            st.dataframe(df_raw.tail(50), use_container_width=True)
        else:
            st.info("No trades executed with current parameters.")
    else:
        st.error("Failed to fetch data. Check ticker or timeframe compatibility.")

# --- TAB 2: LIVE TRADING ---
with tab2:
    st.subheader(f"Live Trading Engine: {symbol}")
    st.warning("⚠️ Live trading polls yfinance every 1.5 seconds. YFinance Indian indices data is often delayed by 15 mins. Use Forex/Crypto for real-time testing.")
    
    if 'live_trading' not in st.session_state:
        st.session_state.live_trading = False

    def toggle_live():
        st.session_state.live_trading = not st.session_state.live_trading

    st.button("Start/Stop Live Trading", on_click=toggle_live, type="primary" if st.session_state.live_trading else "secondary")
    
    # Placeholders for dynamic updates
    live_status = st.empty()
    param_display = st.empty()
    live_chart = st.empty()
    
    if st.session_state.live_trading:
        live_status.success("Live Trading Active... Polling API")
        
        # Display Current Params
        param_display.markdown(f"**Selected:** Strategy: `{strategy}` | SL: `{sl_type} ({sl_val})` | TP: `{tp_type} ({tp_val})`")
        
        while st.session_state.live_trading:
            # Fetch minimal data to save bandwidth
            live_df = fetch_data(symbol, '1d', interval)
            if live_df is not None:
                live_strat = apply_strategy(live_df, strategy, param1, param2)
                
                # Plot live EMA chart
                fig = go.Figure(data=[go.Candlestick(x=live_strat.index,
                            open=live_strat['Open'], high=live_strat['High'],
                            low=live_strat['Low'], close=live_strat['Close'])])
                if strategy == 'EMA Crossover':
                    fig.add_trace(go.Scatter(x=live_strat.index, y=live_strat['EMA_Fast'], mode='lines', name='Fast EMA'))
                    fig.add_trace(go.Scatter(x=live_strat.index, y=live_strat['EMA_Slow'], mode='lines', name='Slow EMA'))
                fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=30, b=0))
                live_chart.plotly_chart(fig, use_container_width=True, key=str(time.time())) # Unique key forces update
                
            time.sleep(1.5) # Graceful API Rate Limit handling

# --- TAB 3: OPTIMIZATION ---
with tab3:
    st.subheader("Strategy Parameter Optimization")
    target_accuracy = st.number_input("Desired Target Accuracy (%)", min_value=1.0, max_value=100.0, value=90.0)
    
    if st.button("Run Optimization"):
        if df_raw is not None and strategy == 'EMA Crossover':
            st.info("Running grid search for Fast EMA (5-15) and Slow EMA (15-30)...")
            best_acc = 0
            best_params = None
            best_trades = None
            
            progress = st.progress(0)
            total_iters = len(range(5, 16)) * len(range(15, 31))
            iters = 0
            
            for fast, slow in itertools.product(range(5, 16), range(15, 31)):
                if fast >= slow: continue
                opt_df = apply_strategy(df_raw, strategy, fast, slow)
                opt_trades = run_backtest(opt_df, sl_type, sl_val, tp_type, tp_val)
                
                if not opt_trades.empty:
                    acc = len(opt_trades[opt_trades['PnL Points'] > 0]) / len(opt_trades) * 100
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (fast, slow)
                        best_trades = opt_trades
                        
                    if acc >= target_accuracy:
                        st.success(f"🎯 Target hit! Fast EMA: {fast}, Slow EMA: {slow} | Accuracy: {acc:.2f}%")
                        break # Stop early if target met
                
                iters += 1
                progress.progress(min(iters / total_iters, 1.0))
                
            if best_acc < target_accuracy:
                st.warning(f"Could not reach {target_accuracy}%. Best optimized result:")
                st.write(f"**Fast EMA:** {best_params[0]} | **Slow EMA:** {best_params[1]} | **Accuracy:** {best_acc:.2f}%")
                
            if best_trades is not None:
                st.dataframe(best_trades, use_container_width=True)
        else:
            st.warning("Optimization currently structured for EMA Crossover in this demo.")
