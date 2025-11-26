import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis

# --- CONFIG ---
st.set_page_config(
    page_title="Leading Indicator Algo Dashboard",
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
ATR_WINDOW = 14
RSI_WINDOW = 14

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

def calculate_rsi(close_prices, window=14):
    """Calculates Relative Strength Index (RSI). Considered leading for divergences."""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0) 
    
    avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan) 
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, window=14):
    """Calculates Average True Range (ATR) - Leading volatility measure."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # ATR is typically an SMA or EMA of the TR. We use EWMA (like Wilder's method).
    atr = tr.ewm(span=window, adjust=False, min_periods=window).mean()
    return atr

def apply_leading_indicators(df):
    """Applies leading and relevant non-lagging indicators."""
    if df.empty:
        return df
    
    df['RSI'] = calculate_rsi(df['Close'], window=RSI_WINDOW)
    df['ATR'] = calculate_atr(df, window=ATR_WINDOW)
    df['Volatility_Pct'] = df['Close'].pct_change().rolling(window=20).std() * 100
    
    return df

@st.cache_data(ttl=3600)
def fetch_and_process_data(ticker, interval, period, sleep_sec):
    """Fetches yfinance data with rate limiting, flattens MultiIndex, and converts to IST."""
    try:
        time.sleep(sleep_sec) 
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if df.empty:
            return pd.DataFrame()

        # --- FIX: Gracefully flatten MultiIndex DataFrame ---
        if isinstance(df.columns, pd.MultiIndex):
            # Select the main price data columns (e.g., ('Close', 'TICKER'))
            # We assume the columns are ['Open', 'High', 'Low', 'Close', 'Volume']
            df.columns = [col[0] for col in df.columns] 
            # If the ticker symbol is used as the first level index, this flattens it.
        # --- END FIX ---
            
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
    """
    Ensures the value is a scalar, handling potential Pandas Series or 
    NumPy arrays of length 1, preventing format string errors.
    """
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
    
    # --- FIX: Ensure max/min operations return scalar values to avoid Ambiguous Series Error ---
    high = extract_scalar(df['High'].max())
    low = extract_scalar(df['Low'].min())
    # --- END FIX ---
    
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

# --- ADVANCED ANALYSIS MODULES ---

def detect_rsi_divergence(df, lookback=20):
    """Simple check for recent divergence: price Higher Highs with RSI Lower Highs (Bearish)"""
    
    close = df['Close'].iloc[-lookback:]
    rsi = df['RSI'].iloc[-lookback:]
    
    if len(close) < 5 or close.isnull().any() or rsi.isnull().any():
        return "N/A (Insufficient Data)"
        
    # Check for Bearish Divergence (Price HH, RSI LH)
    # Simplified check: Find the last significant peak/trough
    
    # 1. Price Highs
    price_highs = close[close == close.rolling(window=5, center=True).max()].dropna()
    if len(price_highs) >= 2:
        latest_price = price_highs.iloc[-1]
        prev_price = price_highs.iloc[-2]
        
        if latest_price > prev_price:  # Price Higher High (HH)
            # Check corresponding RSI values
            latest_rsi = rsi[price_highs.index[-1]]
            prev_rsi = rsi[price_highs.index[-2]]
            
            if latest_rsi < prev_rsi: # RSI Lower High (LH)
                return f"⚠️ **BEARISH DIVERGENCE** (Price HH/RSI LH @ {latest_price:.2f})"
    
    # Check for Bullish Divergence (Price LL, RSI HL)
    price_lows = close[close == close.rolling(window=5, center=True).min()].dropna()
    if len(price_lows) >= 2:
        latest_price = price_lows.iloc[-1]
        prev_price = price_lows.iloc[-2]
        
        if latest_price < prev_price:  # Price Lower Low (LL)
            # Check corresponding RSI values
            latest_rsi = rsi[price_lows.index[-1]]
            prev_rsi = rsi[price_lows.index[-2]]
            
            if latest_rsi > prev_rsi: # RSI Higher Low (HL)
                return f"✅ **BULLISH DIVERGENCE** (Price LL/RSI HL @ {latest_price:.2f})"

    return "No Recent Divergence Detected"


def perform_leading_analysis(df, ticker_label):
    """Focuses on Volatility, Fibonacci and Momentum Divergence."""
    st.subheader(f"⚡ Leading Analysis: {ticker_label}")
    if df.empty:
        st.info("Data not available for Leading Analysis.")
        return

    last_row = df.iloc[-1]
    current_close = extract_scalar(last_row['Close'])
    
    # 1. Volatility (ATR)
    atr_value = extract_scalar(last_row['ATR'])
    volatility_pct = extract_scalar(last_row['Volatility_Pct'])
    
    st.markdown(f"""
    * **Current ATR ({ATR_WINDOW} periods):** `{atr_value:,.2f}` (Risk Proxy)
    * **Rolling Volatility (20 periods):** `{volatility_pct:.2f}%` (Market Energy)
    """)
    
    # 2. Fibonacci Support/Resistance
    fibo_levels = get_fibonacci_levels(df)
    st.markdown(f"**Key 50% Fibonacci Level:** `{fibo_levels['50.0%']:.2f}`")
    
    # Determine proximity to 50% level
    proximity = abs(current_close - fibo_levels['50.0%']) / atr_value if atr_value > 0 else np.inf
    
    if proximity <= 1.5 and atr_value > 0:
        st.warning(f"Price is within 1.5 ATR of the 50% Fibonacci level ({fibo_levels['50.0%']:.2f}). **Expect a Decision Point.**")
    elif current_close > fibo_levels['50.0%']:
        st.success("Price is trading above the 50% Fibonacci Retracement.")
    else:
        st.error("Price is trading below the 50% Fibonacci Retracement.")

    # 3. Momentum Divergence
    divergence = detect_rsi_divergence(df, lookback=40)
    st.markdown(f"**RSI Divergence Check:** {divergence}")
    
    st.info("💡 Summary: Trading decisions should focus on confluence between Divergence signals and price action around Fibonacci/ATR levels.")


def perform_ratio_analysis(df1, df2):
    """Performs Ratio Calculation and displays basic results."""
    st.subheader("⚖️ Ratio Analysis (Cointegration Check)")
    
    df_combined = pd.concat([df1['Close'], df2['Close']], axis=1).dropna()
    df_combined.columns = ['Close_1', 'Close_2']
    df_combined['Ratio'] = df_combined['Close_1'] / df_combined['Close_2']
    df_combined['Ratio_RSI'] = calculate_rsi(df_combined['Ratio'])
    
    if df_combined.empty:
        st.warning("No overlapping data found for ratio calculation.")
        return

    ratio_mean = extract_scalar(df_combined['Ratio'].mean())
    ratio_std = extract_scalar(df_combined['Ratio'].std())
    current_ratio = extract_scalar(df_combined['Ratio'].iloc[-1])
    
    st.markdown(f"""
    * **Mean Ratio:** `{ratio_mean:.4f}`
    * **Std Dev:** `{ratio_std:.4f}`
    * **Current Ratio:** `{current_ratio:.4f}`
    """)
    
    st.dataframe(
        df_combined[['Ratio', 'Ratio_RSI']].tail(10), 
        use_container_width=True, 
        column_config={"__index__": st.column_config.DatetimeColumn("DateTime (IST)")}
    )
    
    st.info("💡 Advanced: Ratio trading is a leading strategy. When the Ratio RSI is extreme (e.g., < 10 or > 90), the pair is statistically likely to mean-revert.")

def perform_value_at_risk(df, ticker_label):
    """Calculates Value at Risk (VaR) and Conditional VaR (CVaR)."""
    st.subheader(f"📉 Value at Risk (VaR) Analysis ({ticker_label})")
    
    if len(df) < 2:
        st.info("Insufficient data for VaR analysis.")
        return

    returns = df['Close'].pct_change().dropna()
    if returns.empty:
        st.info("Cannot calculate returns distribution.")
        return
        
    current_price = extract_scalar(df['Close'].iloc[-1])
    if pd.isna(current_price):
         st.info("Current price not available for VaR price calculation.")
         return

    mu = extract_scalar(returns.mean())
    sigma = extract_scalar(returns.std())
    
    # 1. Historical VaR (Non-parametric)
    VaR_95_hist = extract_scalar(returns.quantile(0.05)) # Worst 5% loss
    VaR_99_hist = extract_scalar(returns.quantile(0.01)) # Worst 1% loss

    # 2. Conditional VaR (Expected Shortfall) - Average of the worst losses
    CVaR_95 = extract_scalar(returns[returns <= VaR_95_hist].mean()) if not pd.isna(VaR_95_hist) else np.nan
    
    st.markdown(f"""
    **Current Price:** `{current_price:,.2f}`
    
    * **95% VaR (Historical):** `{VaR_95_hist*100:.3f}%` | Price Equivalent: `{current_price * (1 + VaR_95_hist):.2f}`
    * **99% VaR (Historical):** `{VaR_99_hist*100:.3f}%` | Price Equivalent: `{current_price * (1 + VaR_99_hist):.2f}`
    * **Conditional VaR (CVaR) 95%:** `{CVaR_95*100:.3f}%` (Expected loss if VaR is breached)
    """)
    
    st.warning("⚠️ VaR provides a **leading risk boundary**. The 99% VaR price equivalent is a strong candidate for a **Stop Loss** level, as it represents a 1% chance of being breached.")


def generate_candlestick_chart(df, ticker_label):
    """Generates the interactive Candlestick Chart with ATR and RSI."""
    if df.empty:
        st.warning(f"No data to chart for {ticker_label}.")
        return

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='Candles'))
    
    # Fibonacci Levels
    fibo = get_fibonacci_levels(df)
    
    # Check if fibo levels are valid before adding lines
    if not pd.isna(fibo['50.0%']):
        fig.add_hline(y=fibo['50.0%'], line_dash="dash", line_color="orange", annotation_text="50% Fib")
    if not pd.isna(fibo['100% (Support)']):
        fig.add_hline(y=fibo['100% (Support)'], line_dash="dash", line_color="green", annotation_text="100% Support")
    if not pd.isna(fibo['0.0% (Resistance)']):
        fig.add_hline(y=fibo['0.0% (Resistance)'], line_dash="dash", line_color="red", annotation_text="0% Resistance")


    # RSI Subplot
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", name="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", name="Oversold")
    fig_rsi.update_layout(height=200, title='Relative Strength Index (RSI)', yaxis_range=[0, 100])

    # ATR Subplot
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df.index, y=df['ATR'], name=f'ATR ({ATR_WINDOW})', line=dict(color='orange')))
    fig_atr.update_layout(height=200, title='Average True Range (ATR)')


    fig.update_layout(title=f'{ticker_label} Price Chart (Fibonacci/ATR Focus)', 
                      xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.plotly_chart(fig_atr, use_container_width=True)


# --- MAIN LAYOUT FUNCTION ---
def main_dashboard():
    st.title("🎛️ Leading Indicator Algo Trading Dashboard")
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
            st.session_state.df1[['Open', 'High', 'Low', 'Close', 'RSI', 'ATR']].tail(5), 
            use_container_width=True,
            column_config={"__index__": st.column_config.DatetimeColumn("DateTime (IST)")}
        )
        st.markdown("---")


        # --- 3. Tabbed Layout for Advanced Analysis ---
        tab_charts, tab_leading_analysis, tab_var, tab_recommendation = st.tabs([
            "📊 Interactive Charts (Fib/ATR)", 
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
                perform_leading_analysis(st.session_state.df1, ticker1)
            
            if enable_ratio and not st.session_state.df2.empty:
                with col_l2:
                    perform_leading_analysis(st.session_state.df2, ticker2)
                
                st.markdown("---")
                perform_ratio_analysis(st.session_state.df1, st.session_state.df2)
            else:
                # If ratio is disabled, ensure the column layout is handled gracefully
                pass


        with tab_var:
            perform_value_at_risk(st.session_state.df1, ticker1)
            if enable_ratio and not st.session_state.df2.empty:
                st.markdown("---")
                perform_value_at_risk(st.session_state.df2, ticker2)

        with tab_recommendation:
            st.header("🎯 FINAL TRADING RECOMMENDATION")
            
            # --- Synthesis based on Leading Indicators and VaR ---
            st.subheader(f"Synthesis for {ticker1}")
            
            current_price = extract_scalar(st.session_state.df1['Close'].iloc[-1])
            fibo_levels = get_fibonacci_levels(st.session_state.df1)
            fibo_50 = fibo_levels['50.0%']
            divergence_signal = detect_rsi_divergence(st.session_state.df1)
            
            signal = "**NEUTRAL / WATCH**"
            logic = "Price is consolidating near the 50% Fibonacci or no clear leading signal (Divergence) is present."

            if not pd.isna(current_price) and not pd.isna(fibo_50):
                if "BULLISH DIVERGENCE" in divergence_signal and current_price < fibo_50:
                    signal = "**STRONG BUY (Divergence at Support)**"
                    logic = "Bullish RSI Divergence detected, suggesting momentum is turning up while price is below the 50% Fibonacci level (potential support entry)."
                elif "BEARISH DIVERGENCE" in divergence_signal and current_price > fibo_50:
                    signal = "**STRONG SELL (Divergence at Resistance)**"
                    logic = "Bearish RSI Divergence detected, suggesting momentum is turning down while price is above the 50% Fibonacci level (potential resistance entry)."
                elif current_price > fibo_50:
                    signal = "**BUY (Above Key Fib)**"
                    logic = "No divergence, but price holding strongly above the 50% Fibonacci level. Look for a break of the 0% resistance."
                
            st.markdown(f"**Key Price/Indicator Confluence:**")
            st.markdown(f"* **RSI Divergence:** {divergence_signal}")
            st.markdown(f"* **50% Fibonacci Level:** `{fibo_50:.2f}`")
            
            st.markdown(f"## Final Recommendation: {signal}")
            st.info(f"**Logic:** {logic}")
            
            # Risk Management based on VaR
            returns = st.session_state.df1['Close'].pct_change().dropna()
            VaR_99_hist = extract_scalar(returns.quantile(0.01))
            
            st.markdown("""
            ### Risk Management (VaR Based)
            """)
            if not pd.isna(current_price) and not pd.isna(VaR_99_hist):
                VaR_99_price_long = current_price * (1 + VaR_99_hist)
                VaR_99_price_short = current_price * (1 - VaR_99_hist) # Symmetrical for short position

                if signal.startswith("**BUY"):
                    st.markdown(f"* **Recommended Stop Loss (99% VaR Price):** **`{VaR_99_price_long:.2f}`** (Set SL below this price to manage extreme risk.)")
                elif signal.startswith("**SELL"):
                     st.markdown(f"* **Recommended Stop Loss (Symmetrical VaR Price):** **`{VaR_99_price_short:.2f}`** (Symmetrical level for short position protection.)")
                else:
                    st.markdown("* **Risk Management:** Decision point requires market confirmation.")
            else:
                 st.markdown("* **Risk Management:** VaR levels cannot be calculated due to insufficient historical returns data.")


    elif st.session_state.data_fetched and st.session_state.df1.empty:
        st.error(f"No valid data to display for {st.session_state.ticker1}. Please check the ticker symbol, period, and interval.")

    else:
        st.info("Configure your tickers and click 'Fetch/Refresh Data' in the sidebar to begin professional analysis.")
        
# --- EXECUTE ---
if __name__ == "__main__":
    main_dashboard()
