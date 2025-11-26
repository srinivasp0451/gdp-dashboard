import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(
    page_title="Algo Dashboard with Backtesting & MTC",
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
ATR_WINDOW = 20 
RSI_WINDOW = 14
KC_MULTIPLIER = 2.0 

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
    """Calculates Exponential Moving Average (EMA)."""
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
    
    df = calculate_keltner_channels(df, window=ATR_WINDOW, multiplier=KC_MULTIPLIER)
    
    return df

@st.cache_data(ttl=3600)
def fetch_and_process_data(ticker, interval, period, sleep_sec):
    """
    Fetches yfinance data.
    Returns: (DataFrame, status) where status is True (success), False (no data), or an error string.
    """
    try:
        time.sleep(sleep_sec) 
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if df.empty:
            return pd.DataFrame(), False 

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns] 
            
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST_TIMEZONE)
        else:
            df.index = df.index.tz_convert(IST_TIMEZONE)
            
        df = apply_leading_indicators(df)
        
        return df, True

    except Exception as e:
        return pd.DataFrame(), f"An error occurred: {e}"
        
def extract_scalar(value):
    """Ensures the value is a scalar."""
    if isinstance(value, pd.Series):
        if not value.empty and len(value) == 1:
            return value.item()
        return np.nan
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return np.nan
    return value

def calculate_basic_metrics(df, label, interval):
    """Calculates current price, change, and points gained/lost."""
    if df.empty or len(df) < 2:
        return None, None
    
    current_price = extract_scalar(df['Close'].iloc[-1])
    prev_close = extract_scalar(df['Close'].iloc[-2])
    
    if pd.isna(current_price) or pd.isna(prev_close):
        return None, None

    points_change = current_price - prev_close
    percent_change = (points_change / prev_close) * 100
    
    delta_str = f"{points_change:+.2f} pts ({percent_change:+.2f}%)"
    
    st.metric(label=f"Current Price ({label} / {interval})", 
              value=f"{current_price:,.2f}", 
              delta=delta_str,
              delta_color="normal")
    
    return current_price, points_change

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
    """Simple check for recent divergence."""
    close = df['Close'].iloc[-lookback:]
    rsi = df['RSI'].iloc[-lookback:]
    
    if len(close) < 5 or close.isnull().any() or rsi.isnull().any():
        return "No Recent Divergence Detected"
        
    price_highs = close[close == close.rolling(window=5, center=True).max()].dropna()
    if len(price_highs) >= 2 and price_highs.iloc[-1] > price_highs.iloc[-2]:
        if rsi[price_highs.index[-1]] < rsi[price_highs.index[-2]]:
            return f"⚠️ **BEARISH DIVERGENCE** (Price HH/RSI LH)"
    
    price_lows = close[close == close.rolling(window=5, center=True).min()].dropna()
    if len(price_lows) >= 2 and price_lows.iloc[-1] < price_lows.iloc[-2]:
        if rsi[price_lows.index[-1]] > rsi[price_lows.index[-2]]:
            return f"✅ **BULLISH DIVERGENCE** (Price LL/RSI HL)"

    return "No Recent Divergence Detected"

def get_kc_divergence_signal(df):
    """Generates the primary signal based on KC/Divergence logic."""
    if df.empty:
        return "NEUTRAL", "Insufficient data."

    last_row = df.iloc[-1]
    current_price = extract_scalar(last_row['Close'])
    kc_upper = extract_scalar(last_row['KC_Upper'])
    kc_lower = extract_scalar(last_row['KC_Lower'])
    kc_middle = extract_scalar(last_row['KC_Middle'])
    
    divergence_signal = detect_rsi_divergence(df)

    # 1. Strong Reversal Signal (KC Extreme + Divergence)
    if current_price < kc_lower and "BULLISH DIVERGENCE" in divergence_signal:
        return "STRONG BUY", "Price below KC Lower with Bullish Divergence (High-Prob Reversal)."
    elif current_price > kc_upper and "BEARISH DIVERGENCE" in divergence_signal:
        return "STRONG SELL", "Price above KC Upper with Bearish Divergence (High-Prob Reversal)."
    
    # 2. Mild Trend Following Signal
    elif current_price > kc_middle and "DIVERGENCE" not in divergence_signal:
        return "MILD BUY", "Price above KC Middle (EMA), suggesting upward bias."
    elif current_price < kc_middle and "DIVERGENCE" not in divergence_signal:
        return "MILD SELL", "Price below KC Middle (EMA), suggesting downward bias."
    
    # 3. Neutral/Watch
    else:
        return "NEUTRAL", "Consolidation within Keltner Channels or ambiguous Divergence."


# --- BACKTESTING MODULE (CORRECTED) ---

def run_backtest(df, ticker_label, current_price):
    """
    Runs a backtest and formats results based on user request.
    """
    st.subheader(f"📊 Algorithm Backtest Results ({ticker_label})")
    
    df_test = df.copy().dropna(subset=['KC_Upper', 'KC_Lower', 'RSI'])
    
    if len(df_test) < ATR_WINDOW + 2:
        st.warning("Insufficient data for backtesting after dropping NaNs.")
        return None

    returns = df_test['Close'].pct_change().dropna()
    if returns.empty:
        st.warning("Cannot calculate returns for VaR.")
        return None
        
    VaR_99_hist = extract_scalar(returns.quantile(0.01))
    
    trades = []
    in_trade = False
    entry_price = 0
    trade_type = None 
    
    df_test['RSI_Signal'] = np.where(df_test['RSI'] > 50, 1, -1) 
    
    for i in range(len(df_test)):
        current_close = df_test['Close'].iloc[i]
        
        # 1. Trade Exit Check (SL or TP)
        if in_trade:
            # VaR Stop Loss Price Calculation
            sl_price = entry_price * (1 + VaR_99_hist) if trade_type == 'Long' else entry_price * (1 - VaR_99_hist)

            # SL Check 
            if (trade_type == 'Long' and current_close < sl_price) or \
               (trade_type == 'Short' and current_close > sl_price):
                exit_price = sl_price 
                points = (exit_price - entry_price) if trade_type == 'Long' else (entry_price - exit_price)
                trades.append({'Entry': entry_price, 'Exit': exit_price, 'Type': trade_type, 
                               'Points': points, 'Result': 'Loss (SL)'})
                in_trade = False
                continue

            # TP Check (Price hits the opposite KC_Middle line)
            tp_price = df_test['KC_Middle'].iloc[i]
            if (trade_type == 'Long' and current_close > tp_price) or \
               (trade_type == 'Short' and current_close < tp_price):
                exit_price = current_close 
                points = (exit_price - entry_price) if trade_type == 'Long' else (entry_price - exit_price)
                trades.append({'Entry': entry_price, 'Exit': exit_price, 'Type': trade_type, 
                               'Points': points, 'Result': 'Profit (TP)' if points > 0 else 'Loss (TP)'})
                in_trade = False
                continue
        
        # 2. Trade Entry Check
        if not in_trade:
            upper = df_test['KC_Upper'].iloc[i]
            lower = df_test['KC_Lower'].iloc[i]
            rsi_signal = df_test['RSI_Signal'].iloc[i]

            # Long Entry
            if current_close < lower and rsi_signal == 1:
                in_trade = True
                entry_price = current_close
                trade_type = 'Long'

            # Short Entry
            elif current_close > upper and rsi_signal == -1:
                in_trade = True
                entry_price = current_close
                trade_type = 'Short'
    
    # --- Summarize Results ---
    if not trades:
        st.info("No trades were generated by the Keltner/VaR algorithm in this period.")
        return None

    trade_df = pd.DataFrame(trades)
    
    # Calculate Metrics - ACCURACY, POINTS, PERCENTAGE, PROFIT/LOSS TRADES
    trade_df['Is_Profit'] = trade_df['Points'] > 0
    
    total_trades = len(trade_df)
    profitable_trades = trade_df['Is_Profit'].sum()
    losing_trades = total_trades - profitable_trades
    accuracy = (profitable_trades / total_trades) * 100
    
    total_points = extract_scalar(trade_df['Points'].sum())
    
    # Calculate Total Return Percentage
    trade_df['Return_Pct'] = trade_df.apply(
        lambda row: (row['Exit'] - row['Entry']) / row['Entry'] if row['Type'] == 'Long' else (row['Entry'] - row['Exit']) / row['Entry'], axis=1
    )
    total_percent = trade_df['Return_Pct'].sum() * 100
    
    # Final Result Dictionary
    results = {
        'Total Trades': total_trades,
        'Profit Trades': profitable_trades,
        'Loss Trades': losing_trades,
        'Accuracy': f"{accuracy:.2f}%",
        'Total Points Gained/Lost': f"{total_points:,.2f}",
        'Total Return (%)': f"{total_percent:.2f}%"
    }

    st.markdown("### 📈 Performance Metrics (Backtest)")
    st.table(pd.DataFrame(list(results.items()), columns=['Metric', 'Value']))
    
    # Returning the structure for explicit verification
    return results

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
        
        # --- Timeframe Selection ---
        col_i, col_p = st.columns(2)
        interval = col_i.selectbox("Timeframe (Short-Term / Entry)", TIME_INTERVALS, index=4, help="Select the timeframe for the chart and entry signal (e.g., 1h).")
        period = col_p.selectbox("Data Period", PERIODS, index=2)
        st.session_state.interval = interval
        st.session_state.period = period
        
        # New: Mid-Term Timeframe for Confluence
        st_interval_index = TIME_INTERVALS.index(interval)
        default_mt_index = min(st_interval_index + 1, len(TIME_INTERVALS) - 1)
        st.session_state.mt_interval = st.selectbox("Mid-Term Timeframe (Trend Filter)", TIME_INTERVALS, index=default_mt_index, help="Select a longer timeframe (e.g., 1d if Entry is 1h) for trend filtering.")
        
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
        
        if st.button("🚀 Fetch/Refresh Data"):
            st.session_state.data_fetched = False
            
            # Fetch Short-Term Data (for Chart/Entry/Backtest)
            with st.spinner(f"Fetching Short-Term data for {ticker1}..."):
                st.session_state.df1, status1 = fetch_and_process_data(ticker1, interval, period, 0.1)
            
            if status1 is True:
                st.toast(f"✅ ST data for {ticker1} fetched.")
            elif status1 is False:
                st.error(f"No ST data available for {ticker1}.")
            else: 
                st.error(f"Error fetching {ticker1} ST: {status1}")

            # Fetch Mid-Term Data (for MTC)
            with st.spinner(f"Fetching Mid-Term data for {ticker1}...") :
                st.session_state.df_mt, status_mt = fetch_and_process_data(ticker1, st.session_state.mt_interval, period, 0.1)
            
            if status_mt is True:
                st.toast(f"✅ MT data for {ticker1} fetched.")
            elif status_mt is False:
                st.warning(f"No MT data available for {ticker1}. Cannot perform MTC.")
                st.session_state.df_mt = pd.DataFrame()
            else: 
                st.error(f"Error fetching {ticker1} MT: {status_mt}")

            # Fetch Ticker 2 (if enabled)
            if enable_ratio and ticker2:
                with st.spinner(f"Fetching Ticker 2 data (waiting {sleep_sec}s)..."):
                    st.session_state.df2, status2 = fetch_and_process_data(ticker2, interval, period, sleep_sec)
                    
                if status2 is True:
                    st.toast(f"✅ Data for {ticker2} fetched.")
                elif status2 is False:
                    st.error(f"No data available for {ticker2}.")
                else: 
                    st.error(f"Error fetching {ticker2}: {status2}")
            
            st.session_state.data_fetched = True
            st.rerun()
            
    # --- Main Content ---
    if st.session_state.data_fetched and not st.session_state.df1.empty:
        
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
        
        tab_charts, tab_leading_analysis, tab_var, tab_recommendation = st.tabs([
            "📊 Interactive Charts", 
            "⚡ Leading Analysis", 
            "📉 Value at Risk (VaR)", 
            "🎯 Final Recommendation"
        ])
        
        with tab_charts:
            generate_candlestick_chart(st.session_state.df1, ticker1)
        # ... (other tabs remain the same or assume functionality) ...

        with tab_recommendation:
            st.header("🎯 FINAL TRADING RECOMMENDATION")
            
            # --- BACKTESTING EXECUTION ---
            backtest_results = run_backtest(st.session_state.df1, ticker1, price1)
            st.markdown("---")
            
            st.subheader(f"Synthesis & Multi-Timeframe Confluence (MTC)")
            
            # 1. Get Short-Term (ST) Signal (Entry Signal)
            st_signal, st_logic = get_kc_divergence_signal(st.session_state.df1)
            
            # 2. Get Mid-Term (MT) Signal (Trend Filter)
            if not st.session_state.df_mt.empty:
                mt_signal, _ = get_kc_divergence_signal(st.session_state.df_mt)
            else:
                mt_signal = "NEUTRAL"

            # 3. Determine Final Confluence and Signal
            
            final_signal = st_signal
            final_logic = st_logic
            confidence = "LOW"

            if st_signal.endswith("BUY") and mt_signal.endswith("BUY"):
                final_signal = "**HIGH CONFIDENCE BUY**"
                final_logic += f" MTC Confirmed: Mid-Term ({st.session_state.mt_interval}) signal is also BUY."
                confidence = "HIGH"
            elif st_signal.endswith("SELL") and mt_signal.endswith("SELL"):
                final_signal = "**HIGH CONFIDENCE SELL**"
                final_logic += f" MTC Confirmed: Mid-Term ({st.session_state.mt_interval}) signal is also SELL."
                confidence = "HIGH"
            elif st_signal == "NEUTRAL":
                final_signal = "**NEUTRAL / WATCH**"
                final_logic = st_logic
                confidence = "LOW"
            else:
                 # Conflicting signals
                 final_signal = f"**{st_signal} (DIVERGENT)**"
                 final_logic = f"Signal is conflicting: Short-Term is {st_signal} but Mid-Term ({st.session_state.mt_interval}) is {mt_signal}. **Avoid Trading.**"
                 confidence = "MEDIUM"


            st.markdown(f"### ⏱️ Confluence Check (ST: {st.session_state.interval} vs MT: {st.session_state.mt_interval})")
            st.table(pd.DataFrame({
                'Timeframe': [f"Short-Term ({st.session_state.interval})", f"Mid-Term ({st.session_state.mt_interval})", "FINAL"],
                'Signal': [st_signal, mt_signal, final_signal],
                'Confidence': ["Entry", "Trend Filter", confidence]
            }))
            
            st.markdown(f"## Final Recommendation: {final_signal}")
            st.info(f"**Logic:** {final_logic}")
            
            # Risk Management (VaR)
            returns = st.session_state.df1['Close'].pct_change().dropna()
            VaR_99_hist = extract_scalar(returns.quantile(0.01))
            current_price = extract_scalar(st.session_state.df1['Close'].iloc[-1])
            last_row = st.session_state.df1.iloc[-1]
            
            st.markdown("""
            ### Risk Management (VaR Based)
            """)
            if not pd.isna(current_price) and not pd.isna(VaR_99_hist) and "HIGH CONFIDENCE" in final_signal:
                VaR_99_price_long = current_price * (1 + VaR_99_hist)
                VaR_99_price_short = current_price * (1 - VaR_99_hist)

                if "BUY" in final_signal:
                    st.markdown(f"* **Recommended Stop Loss (99% VaR Price):** **`{VaR_99_price_long:.2f}`**")
                    st.markdown(f"* **Recommended Take Profit (KC Middle Target):** **`{extract_scalar(last_row['KC_Middle']):.2f}`** (Target the mean reversion)")
                elif "SELL" in final_signal:
                     st.markdown(f"* **Recommended Stop Loss (Symmetrical VaR Price):** **`{VaR_99_price_short:.2f}`**")
                     st.markdown(f"* **Recommended Take Profit (KC Middle Target):** **`{extract_scalar(last_row['KC_Middle']):.2f}`** (Target the mean reversion)")
            else:
                 st.markdown("* **Risk Management:** Decision point requires market confirmation or signals conflict. **No Confident Trade Setup.**")


    elif st.session_state.data_fetched and st.session_state.df1.empty:
        st.error(f"No valid data to display for {st.session_state.ticker1}. Please check the ticker symbol, period, and interval.")

    else:
        st.info("Configure your tickers and click 'Fetch/Refresh Data' in the sidebar to begin professional analysis.")
        
# --- EXECUTE ---
if __name__ == "__main__":
    main_dashboard()
