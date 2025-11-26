import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis

# --- CONFIG ---
st.set_page_config(
    page_title="Algo Dashboard with Backtesting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL CONSTANTS ---
STANDARD_TICKERS = [
    "^NSEI", "^BANKNIFTY", "^BSESN", "BTC-USD", "ETH-USD", "GC=F", "SI=F", "INR=X"
]
TIME_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
PERIODS = ["1d", "5d", "1mo", "6mo", "1y", "5y", "10y"]
IST_TIMEZONE = 'Asia/Kolkata' 
ATR_WINDOW = 20 # Common for volatility (Keltner/ATR)
RSI_WINDOW = 14
KC_MULTIPLIER = 2.0 # Keltner Channel width

# --- SESSION STATE INITIALIZATION ---
for key in ['data_fetched', 'df1', 'df2', 'ticker1', 'ticker2', 'interval', 'period']:
    if key not in st.session_state:
        if key.startswith('df'):
            st.session_state[key] = pd.DataFrame()
        elif key == 'data_fetched':
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# --- CORE UTILITY FUNCTIONS ---

def calculate_ema(data_series, window):
    """Calculates Exponential Moving Average (EMA). Needed for KC mid-line."""
    return data_series.ewm(span=window, adjust=False).mean()

def calculate_rsi(close_prices, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0) 
    
    avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan) 
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, window=14):
    """Calculates Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False, min_periods=window).mean()
    return atr

def calculate_keltner_channels(df, window=20, multiplier=2.0):
    """Calculates Keltner Channels (KC)."""
    df['KC_Middle'] = calculate_ema(df['Close'], window)
    df['KC_ATR'] = calculate_atr(df, window)
    df['KC_Upper'] = df['KC_Middle'] + multiplier * df['KC_ATR']
    df['KC_Lower'] = df['KC_Middle'] - multiplier * df['KC_ATR']
    return df

def apply_leading_indicators(df):
    """Applies leading and relevant non-lagging indicators."""
    if df.empty:
        return df
    
    df['RSI'] = calculate_rsi(df['Close'], window=RSI_WINDOW)
    df['ATR'] = calculate_atr(df, window=ATR_WINDOW)
    df['Volatility_Pct'] = df['Close'].pct_change().rolling(window=20).std() * 100
    
    # New Leading Indicator: Keltner Channels
    df = calculate_keltner_channels(df, window=ATR_WINDOW, multiplier=KC_MULTIPLIER)
    
    return df

@st.cache_data(ttl=3600)
def fetch_and_process_data(ticker, interval, period, sleep_sec):
    """Fetches yfinance data with rate limiting, flattens MultiIndex, and converts to IST."""
    try:
        time.sleep(sleep_sec) 
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if df.empty:
            return pd.DataFrame()

        # Gracefully flatten MultiIndex DataFrame
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns] 
            
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST_TIMEZONE)
        else:
            df.index = df.index.tz_convert(IST_TIMEZONE)
            
        df = apply_leading_indicators(df)
        st.toast(f"✅ Data for {ticker} fetched and processed.")
        return df

    except Exception as e:
        st.error(f"An error occurred fetching data for {ticker}: {e}")
        return pd.DataFrame()
        
def extract_scalar(value):
    """Ensures the value is a scalar, handling potential Pandas Series or NumPy arrays."""
    if isinstance(value, pd.Series):
        if not value.empty and len(value) == 1:
            return value.item()
        return np.nan
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return np.nan
    return value

def get_fibonacci_levels(df):
    """Calculates high/low and returns simple Fibonacci retracement levels (Leading S/R)."""
    high = extract_scalar(df['High'].max())
    low = extract_scalar(df['Low'].min())
    
    if pd.isna(high) or pd.isna(low):
         return {lvl: np.nan for lvl in ['0.0%', '23.6%', '38.2%', '50.0%', '61.8%', '78.6%', '100%']}
         
    diff = high - low
    
    levels = {
        '0.0% (Resistance)': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.500 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100% (Support)': low
    }
    return levels

def detect_rsi_divergence(df, lookback=40):
    """Simple check for recent divergence: price Higher Highs with RSI Lower Highs (Bearish)"""
    # Logic simplified for integration; full implementation is complex.
    
    close = df['Close'].iloc[-lookback:]
    rsi = df['RSI'].iloc[-lookback:]
    
    if len(close) < 5 or close.isnull().any() or rsi.isnull().any():
        return "N/A (Insufficient Data)"
        
    # Check for Bearish Divergence (Price HH, RSI LH)
    price_highs = close[close == close.rolling(window=5, center=True).max()].dropna()
    if len(price_highs) >= 2 and price_highs.iloc[-1] > price_highs.iloc[-2]:
        if rsi[price_highs.index[-1]] < rsi[price_highs.index[-2]]:
            return f"⚠️ **BEARISH DIVERGENCE** (Price HH/RSI LH)"
    
    # Check for Bullish Divergence (Price LL, RSI HL)
    price_lows = close[close == close.rolling(window=5, center=True).min()].dropna()
    if len(price_lows) >= 2 and price_lows.iloc[-1] < price_lows.iloc[-2]:
        if rsi[price_lows.index[-1]] > rsi[price_lows.index[-2]]:
            return f"✅ **BULLISH DIVERGENCE** (Price LL/RSI HL)"

    return "No Recent Divergence Detected"

# --- BACKTESTING MODULE ---

def run_backtest(df, ticker_label, current_price):
    """
    Runs a backtest on the algorithm using Keltner Channel and VaR rules.
    
    Entry: Price closes outside KC AND Divergence is present.
    SL: 99% VaR price equivalent.
    TP: Price hits opposite KC_Middle line.
    """
    st.subheader(f"📊 Algorithm Backtest Results ({ticker_label})")
    
    df_test = df.copy().dropna()
    
    if len(df_test) < ATR_WINDOW + 2:
        st.warning("Insufficient data for backtesting after dropping NaNs (need > 22 periods).")
        return None

    # Calculate 99% VaR (for Stop Loss) over the entire backtest period
    returns = df_test['Close'].pct_change().dropna()
    if returns.empty:
        st.warning("Cannot calculate returns for VaR.")
        return None
        
    VaR_99_hist = extract_scalar(returns.quantile(0.01))
    
    # --- Backtest Core Logic Setup ---
    trades = []
    in_trade = False
    entry_price = 0
    trade_type = None # 'Long' or 'Short'
    
    # Iterate through data, starting after indicators have populated
    start_index = df_test['ATR'].first_valid_index()
    if start_index is None:
        st.warning("Indicators did not populate correctly.")
        return None
        
    start_loc = df_test.index.get_loc(start_index)
    
    # Simplified divergence check for backtesting: use RSI level crossing 50
    df_test['RSI_Signal'] = np.where(df_test['RSI'] > 50, 1, -1) 
    
    for i in range(start_loc, len(df_test)):
        current_close = df_test['Close'].iloc[i]
        
        # Calculate VaR Stop Loss for the current position
        if in_trade:
            # Use the calculated VaR percentage
            if trade_type == 'Long':
                sl_price = entry_price * (1 + VaR_99_hist) # VaR is negative, so adding VaR is reducing the entry price
            else: # Short
                sl_price = entry_price * (1 - VaR_99_hist) # Symmetrical level above entry

        # 1. Trade Exit Check (SL or TP)
        if in_trade:
            # SL Check (Price breaches the VaR stop loss)
            if (trade_type == 'Long' and current_close < sl_price) or \
               (trade_type == 'Short' and current_close > sl_price):
                exit_price = sl_price # Exit at SL price
                points = (exit_price - entry_price) if trade_type == 'Long' else (entry_price - exit_price)
                trades.append({'Entry': entry_price, 'Exit': exit_price, 'Type': trade_type, 
                               'Points': points, 'Result': 'Loss (SL)'})
                in_trade = False
                continue

            # TP Check (Price hits the opposite KC_Middle)
            tp_price = df_test['KC_Middle'].iloc[i]
            if (trade_type == 'Long' and current_close < tp_price) or \
               (trade_type == 'Short' and current_close > tp_price):
                exit_price = current_close # Exit at current close (simplified TP)
                points = (exit_price - entry_price) if trade_type == 'Long' else (entry_price - exit_price)
                trades.append({'Entry': entry_price, 'Exit': exit_price, 'Type': trade_type, 
                               'Points': points, 'Result': 'Profit (TP)' if points > 0 else 'Loss (TP)'})
                in_trade = False
                continue
        
        # 2. Trade Entry Check (Only enter if not currently in a trade)
        if not in_trade:
            upper = df_test['KC_Upper'].iloc[i]
            lower = df_test['KC_Lower'].iloc[i]
            rsi_signal = df_test['RSI_Signal'].iloc[i]

            # Long Entry: Close below KC_Lower AND RSI showing positive momentum (proxy for Bullish Divergence)
            if current_close < lower and rsi_signal > 0:
                in_trade = True
                entry_price = current_close
                trade_type = 'Long'

            # Short Entry: Close above KC_Upper AND RSI showing negative momentum (proxy for Bearish Divergence)
            elif current_close > upper and rsi_signal < 0:
                in_trade = True
                entry_price = current_close
                trade_type = 'Short'
    
    # --- Summarize Results ---
    if not trades:
        st.info("No trades were generated by the Keltner/VaR algorithm in this period.")
        return None

    trade_df = pd.DataFrame(trades)
    
    # Calculate Metrics
    total_trades = len(trade_df)
    profitable_trades = len(trade_df[trade_df['Points'] > 0])
    losing_trades = total_trades - profitable_trades
    accuracy = (profitable_trades / total_trades) * 100
    
    total_points = extract_scalar(trade_df['Points'].sum())
    total_percent = (total_points / entry_price) * 100 if total_trades > 0 else 0
    
    # Final Result Dictionary
    results = {
        'Total Trades': total_trades,
        'Profit Trades': profitable_trades,
        'Loss Trades': losing_trades,
        'Accuracy': f"{accuracy:.2f}%",
        'Total Points Gained/Lost': f"{total_points:,.2f}",
        'Total Return (%)': f"{total_percent:.2f}%"
    }

    # Display results in a clear table
    st.markdown("### 📈 Performance Metrics")
    st.table(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']))

    st.markdown("### 📜 Trade Log Sample")
    st.dataframe(trade_df.tail(10), use_container_width=True)
    
    return results

# --- Visualization and Display Functions (omitted for brevity, assume updated) ---
# ... (calculate_basic_metrics, perform_leading_analysis, perform_value_at_risk remain the same)

def generate_candlestick_chart(df, ticker_label):
    """Generates the interactive Candlestick Chart with KC, ATR and RSI."""
    if df.empty:
        st.warning(f"No data to chart for {ticker_label}.")
        return

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='Candles'))
    
    # --- Keltner Channels ---
    if 'KC_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['KC_Upper'], line=dict(color='red', width=1), name='KC Upper', opacity=0.8))
        fig.add_trace(go.Scatter(x=df.index, y=df['KC_Lower'], line=dict(color='green', width=1), name='KC Lower', opacity=0.8))
        fig.add_trace(go.Scatter(x=df.index, y=df['KC_Middle'], line=dict(color='orange', width=1, dash='dot'), name='KC Middle (EMA)', opacity=0.6))
    
    # Fibonacci Levels
    fibo = get_fibonacci_levels(df)
    if not pd.isna(fibo['50.0%']):
        fig.add_hline(y=fibo['50.0%'], line_dash="dash", line_color="purple", annotation_text="50% Fib", opacity=0.5)

    fig.update_layout(title=f'{ticker_label} Price Chart (KC/Fib Focus)', 
                      xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI Subplot (unchanged)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", name="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", name="Oversold")
    fig_rsi.update_layout(height=200, title='Relative Strength Index (RSI)', yaxis_range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # ATR Subplot (unchanged)
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df.index, y=df['ATR'], name=f'ATR ({ATR_WINDOW})', line=dict(color='orange')))
    fig_atr.update_layout(height=200, title='Average True Range (ATR)')
    st.plotly_chart(fig_atr, use_container_width=True)
    


# --- MAIN LAYOUT FUNCTION ---
def main_dashboard():
    st.title("🎛️ Algo Dashboard with Keltner Channel & Backtesting")
    st.markdown("---")
    
    # --- 1. Data Fetching & Management (Sidebar) ---
    with st.sidebar:
        st.header("⚙️ Data & Config")
        
        col_t1, col_c1 = st.columns(2)
        ticker1_symbol = col_t1.selectbox("Ticker 1 Symbol", STANDARD_TICKERS, index=0)
        custom_ticker1 = col_c1.text_input("Custom Ticker 1 (Override)", value="")
        ticker1 = custom_ticker1.upper() if custom_ticker1 else ticker1_symbol
        st.session_state.ticker1 = ticker1
        
        col_i, col_p = st.columns(2)
        interval = col_i.selectbox("Timeframe", TIME_INTERVALS, index=4)
        period = col_p.selectbox("Period", PERIODS, index=2)
        st.session_state.interval = interval
        st.session_state.period = period
        
        enable_ratio = st.checkbox("Enable Ratio Analysis", value=False)
        
        ticker2 = None
        if enable_ratio:
            st.subheader("Ticker 2 (Ratio Basis)")
            col_t2, col_c2 = st.columns(2)
            ticker2_symbol = col_t2.selectbox("Ticker 2 Symbol", STANDARD_TICKERS, index=1)
            custom_ticker2 = col_c2.text_input("Custom Ticker 2 (Override)", value="")
            ticker2 = custom_ticker2.upper() if custom_ticker2 else ticker2_symbol
            st.session_state.ticker2 = ticker2
        
        sleep_sec = st.slider("API Delay (seconds)", 0.5, 5.0, 2.5, 0.5)
        st.info(f"API Rate Limit Delay: {sleep_sec}s")
        
        if st.button("🚀 Fetch/Refresh Data"):
            st.session_state.data_fetched = False
            
            with st.spinner(f"Fetching data for {ticker1}...") :
                st.session_state.df1 = fetch_and_process_data(ticker1, interval, period, 0.1)
                
            if enable_ratio and ticker2:
                with st.spinner(f"Fetching data for {ticker2} (waiting {sleep_sec}s)...") :
                    st.session_state.df2 = fetch_and_process_data(ticker2, interval, period, sleep_sec)
            
            st.session_state.data_fetched = True
            st.rerun()
            
    # --- Main Content ---
    if st.session_state.data_fetched and not st.session_state.df1.empty:
        
        # --- 2. Basic Statistics Display ---
        st.header("📈 Current Market Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price1, points1 = calculate_basic_metrics(st.session_state.df1, ticker1, interval)
        
        if enable_ratio and not st.session_state.df2.empty:
            with col2:
                price2, points2 = calculate_basic_metrics(st.session_state.df2, ticker2, interval)
            
            with col3:
                ratio = price1 / price2 if price1 and price2 and not pd.isna(price1) and not pd.isna(price2) else np.nan
                st.metric(label="Current Ratio (T1/T2)", value=f"{ratio:,.4f}" if not np.isnan(ratio) else "N/A")
        
        st.markdown("---")
        st.subheader("Raw Data Sample")
        st.dataframe(
            st.session_state.df1[['Open', 'High', 'Low', 'Close', 'RSI', 'ATR', 'KC_Upper']].tail(5), 
            use_container_width=True,
            column_config={"__index__": st.column_config.DatetimeColumn("DateTime (IST)")}
        )
        st.markdown("---")


        # --- 3. Tabbed Layout for Advanced Analysis ---
        tab_charts, tab_leading_analysis, tab_var, tab_recommendation = st.tabs([
            "📊 Interactive Charts (KC/Fib/ATR)", 
            "⚡ Leading Analysis", 
            "📉 Value at Risk (VaR)", 
            "🎯 Final Recommendation"
        ])
        
        with tab_charts:
            generate_candlestick_chart(st.session_state.df1, ticker1)
            if enable_ratio and not st.session_state.df2.empty:
                st.markdown("---")
                generate_candlestick_chart(st.session_state.df2, ticker2)

        with tab_leading_analysis:
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                # Assuming perform_leading_analysis is available from the last valid response
                perform_leading_analysis(st.session_state.df1, ticker1)
            
            if enable_ratio and not st.session_state.df2.empty:
                with col_l2:
                    perform_leading_analysis(st.session_state.df2, ticker2)
                
                st.markdown("---")
                # Assuming perform_ratio_analysis is available from the last valid response
                perform_ratio_analysis(st.session_state.df1, st.session_state.df2)


        with tab_var:
            # Assuming perform_value_at_risk is available from the last valid response
            perform_value_at_risk(st.session_state.df1, ticker1)
            if enable_ratio and not st.session_state.df2.empty:
                st.markdown("---")
                perform_value_at_risk(st.session_state.df2, ticker2)

        with tab_recommendation:
            st.header("🎯 FINAL TRADING RECOMMENDATION")
            
            # --- BACKTESTING EXECUTION ---
            backtest_results = run_backtest(st.session_state.df1, ticker1, price1)
            st.markdown("---")
            
            # --- Synthesis based on Keltner Channel and Divergence ---
            st.subheader(f"Synthesis & Forward Signal for {ticker1}")
            
            current_price = extract_scalar(st.session_state.df1['Close'].iloc[-1])
            last_row = st.session_state.df1.iloc[-1]
            kc_upper = extract_scalar(last_row['KC_Upper'])
            kc_lower = extract_scalar(last_row['KC_Lower'])
            
            divergence_signal = detect_rsi_divergence(st.session_state.df1)
            
            signal = "**NEUTRAL / WATCH**"
            logic = "Price is within the Keltner Channels (consolidation) and no divergence is present."

            if not pd.isna(current_price):
                # Strong Buy Setup: Price outside KC AND Bullish Divergence
                if current_price < kc_lower and "BULLISH DIVERGENCE" in divergence_signal:
                    signal = "**STRONG BUY (Volatility Compression Reversal)**"
                    logic = "Price is below the lower Keltner Channel (extreme low volatility) while Bullish Divergence suggests momentum is reversing. High probability long entry."
                # Strong Sell Setup: Price outside KC AND Bearish Divergence
                elif current_price > kc_upper and "BEARISH DIVERGENCE" in divergence_signal:
                    signal = "**STRONG SELL (Volatility Compression Reversal)**"
                    logic = "Price is above the upper Keltner Channel (extreme high volatility) while Bearish Divergence suggests momentum is reversing. High probability short entry."
                # Moderate Continuation Buy: Price above KC mean (KC_Middle)
                elif current_price > extract_scalar(last_row['KC_Middle']) and "BULLISH" not in divergence_signal:
                    signal = "**MILD BUY (Trend Following)**"
                    logic = "Price is above the Keltner Channel EMA, indicating a slight upward bias, but lacks the momentum confirmation from Divergence."
                
            st.markdown(f"**Key Price/Indicator Confluence:**")
            st.markdown(f"* **RSI Divergence:** {divergence_signal}")
            st.markdown(f"* **Keltner Channel Upper/Lower:** `{kc_upper:.2f}` / `{kc_lower:.2f}`")
            
            st.markdown(f"## Final Recommendation: {signal}")
            st.info(f"**Logic:** {logic}")
            
            # Risk Management based on VaR (unchanged)
            returns = st.session_state.df1['Close'].pct_change().dropna()
            VaR_99_hist = extract_scalar(returns.quantile(0.01))
            
            st.markdown("""
            ### Risk Management (VaR Based)
            """)
            if not pd.isna(current_price) and not pd.isna(VaR_99_hist):
                VaR_99_price_long = current_price * (1 + VaR_99_hist)
                VaR_99_price_short = current_price * (1 - VaR_99_hist)

                if signal.startswith("**BUY"):
                    st.markdown(f"* **Recommended Stop Loss (99% VaR Price):** **`{VaR_99_price_long:.2f}`**")
                    st.markdown(f"* **Recommended Take Profit (KC Middle Target):** **`{extract_scalar(last_row['KC_Middle']):.2f}`** (Target the mean reversion)")
                elif signal.startswith("**SELL"):
                     st.markdown(f"* **Recommended Stop Loss (Symmetrical VaR Price):** **`{VaR_99_price_short:.2f}`**")
                     st.markdown(f"* **Recommended Take Profit (KC Middle Target):** **`{extract_scalar(last_row['KC_Middle']):.2f}`** (Target the mean reversion)")
                else:
                    st.markdown("* **Risk Management:** Decision point requires market confirmation.")
            else:
                 st.markdown("* **Risk Management:** VaR levels cannot be calculated.")


    elif st.session_state.data_fetched and st.session_state.df1.empty:
        st.error(f"No valid data to display for {st.session_state.ticker1}. Please check the ticker symbol, period, and interval.")

    else:
        st.info("Configure your tickers and click 'Fetch/Refresh Data' in the sidebar to begin professional analysis.")
        
# --- EXECUTE ---
if __name__ == "__main__":
    main_dashboard()
