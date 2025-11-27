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
for key in ['data_fetched', 'df1', 'df2', 'ticker1', 'ticker2', 'interval', 'period', 'mt_interval', 'df_mt']:
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

# --- ADVANCED ANALYSIS MODULES ---

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

# --- CHARTING MODULE ---

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
    
    # Keltner Channels
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
    
    
    # RSI Subplot
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", name="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", name="Oversold")
    fig_rsi.update_layout(height=200, title='Relative Strength Index (RSI)', yaxis_range=[0, 100])
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # ATR Subplot
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df.index, y=df['ATR'], name=f'ATR ({ATR_WINDOW})', line=dict(color='orange')))
    fig_atr.update_layout(height=200, title='Average True Range (ATR)')
    st.plotly_chart(fig_atr, use_container_width=True)

# --- BACKTESTING MODULE ---

def run_backtest(df, ticker_label, current_price):
    """
    Runs a backtest and formats results based on user request (Accuracy, Points, P/L Trades, % Return).
    
    ***FIXED: TP is now the opposite KC band (Upper/Lower) for better Risk/Reward.***
    """
    st.subheader(f"📊 Strategy Backtest Results ({ticker_label})")
    
    df_test = df.copy().dropna(subset=['KC_Upper', 'KC_Lower', 'RSI'])
    
    if len(df_test) < ATR_WINDOW + 2:
        st.warning("Insufficient data for backtesting after dropping NaNs.")
        return None

    returns = df_test['Close'].pct_change().dropna()
    if returns.empty:
        st.warning("Cannot calculate returns for VaR.")
        return None
        
    VaR_99_hist = extract_scalar(returns.quantile(0.01)) # Typically negative
    
    trades = []
    in_trade = False
    entry_price = 0
    trade_type = None 
    entry_date = None
    
    df_test['RSI_Signal'] = np.where(df_test['RSI'] > 50, 1, -1) 
    
    for i in range(len(df_test)):
        current_close = df_test['Close'].iloc[i]
        current_date = df_test.index[i]
        
        # Get the KC bands for both exit and potential entry
        upper = df_test['KC_Upper'].iloc[i]
        lower = df_test['KC_Lower'].iloc[i]
        
        # 1. Trade Exit Check (SL or TP)
        if in_trade:
            # VaR Stop Loss Price Calculation (Emergency Exit)
            # VaR_99_hist is usually a negative percentage (e.g., -0.01). 
            # For Long: Entry * (1 - 0.01) = lower price (SL)
            # For Short: Entry * (1 + 0.01) = higher price (SL)
            sl_price = entry_price * (1 + VaR_99_hist) if trade_type == 'Long' else entry_price * (1 - VaR_99_hist)

            # --- FIXED TP/SL LOGIC ---

            # SL Check (Price hits VaR Stop Loss)
            if (trade_type == 'Long' and current_close < sl_price) or \
               (trade_type == 'Short' and current_close > sl_price):
                exit_price = sl_price 
                points = (exit_price - entry_price) if trade_type == 'Long' else (entry_price - exit_price)
                trades.append({'Entry_Date': entry_date, 'Exit_Date': current_date, 'Entry': entry_price, 'Exit': exit_price, 'Type': trade_type, 
                               'Points': points, 'Result': 'Loss (SL)'})
                in_trade = False
                continue

            # TP Check (Price hits the opposite KC band)
            # Long TP: Current close hits KC_Upper
            if trade_type == 'Long' and current_close >= upper:
                exit_price = current_close 
                points = exit_price - entry_price
                trades.append({'Entry_Date': entry_date, 'Exit_Date': current_date, 'Entry': entry_price, 'Exit': exit_price, 'Type': trade_type, 
                               'Points': points, 'Result': 'Profit (KC_Upper)' if points > 0 else 'Loss (KC_Upper)'})
                in_trade = False
                continue
            
            # Short TP: Current close hits KC_Lower
            elif trade_type == 'Short' and current_close <= lower:
                exit_price = current_close 
                points = entry_price - exit_price
                trades.append({'Entry_Date': entry_date, 'Exit_Date': current_date, 'Entry': entry_price, 'Exit': exit_price, 'Type': trade_type, 
                               'Points': points, 'Result': 'Profit (KC_Lower)' if points > 0 else 'Loss (KC_Lower)'})
                in_trade = False
                continue
            
            # --- END FIXED TP/SL LOGIC ---
        
        # 2. Trade Entry Check
        if not in_trade:
            rsi_signal = df_test['RSI_Signal'].iloc[i]

            # Long Entry (Extreme low + Bullish RSI bias)
            if current_close < lower and rsi_signal == 1:
                in_trade = True
                entry_price = current_close
                trade_type = 'Long'
                entry_date = current_date

            # Short Entry (Extreme high + Bearish RSI bias)
            elif current_close > upper and rsi_signal == -1:
                in_trade = True
                entry_price = current_close
                trade_type = 'Short'
                entry_date = current_date
    
    # --- Summarize Results ---
    if not trades:
        st.info("No trades were generated by the Keltner/VaR algorithm in this period.")
        return None

    trade_df = pd.DataFrame(trades)
    
    # Calculate Metrics 
    trade_df['Is_Profit'] = trade_df['Points'] > 0
    
    total_trades = len(trade_df)
    profitable_trades = trade_df['Is_Profit'].sum()
    losing_trades = total_trades - profitable_trades
    accuracy = (profitable_trades / total_trades) * 100
    
    total_points = extract_scalar(trade_df['Points'].sum())
    
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

    st.markdown("### 📜 Trade Log Sample")
    # Displaying the trade log directly in the UI
    st.dataframe(
        trade_df[['Entry_Date', 'Exit_Date', 'Type', 'Entry', 'Exit', 'Points', 'Return_Pct', 'Result']].tail(10), 
        use_container_width=True,
        column_config={
            "Entry_Date": st.column_config.DatetimeColumn("Entry Date", format="YYYY-MM-DD HH:mm"),
            "Exit_Date": st.column_config.DatetimeColumn("Exit Date", format="YYYY-MM-DD HH:mm"),
            "Return_Pct": st.column_config.NumberColumn("Return (%)", format="%.2f%%")
        }
    )
    
    return results

# --- Main Layout Function ---

def perform_leading_analysis(df, ticker_label):
    """Focuses on Volatility, Fibonacci and Momentum Divergence."""
    st.subheader(f"⚡ Leading Analysis: {ticker_label}")
    if df.empty:
        st.info("Data not available for Leading Analysis.")
        return

    last_row = df.iloc[-1]
    current_close = extract_scalar(last_row['Close'])
    
    atr_value = extract_scalar(last_row['ATR'])
    volatility_pct = extract_scalar(last_row['Volatility_Pct'])
    
    st.markdown(f"""
    * **Current ATR ({ATR_WINDOW} periods):** `{atr_value:,.2f}` (Risk Proxy)
    * **Rolling Volatility (20 periods):** `{volatility_pct:.2f}%` (Market Energy)
    """)
    
    fibo_levels = get_fibonacci_levels(df)
    st.markdown(f"**Key 50% Fibonacci Level:** `{fibo_levels['50.0%']:.2f}`")
    
    proximity = abs(current_close - fibo_levels['50.0%']) / atr_value if atr_value > 0 else np.inf
    
    if proximity <= 1.5 and atr_value > 0:
        st.warning(f"Price is within 1.5 ATR of the 50% Fibonacci level ({fibo_levels['50.0%']:.2f}). **Expect a Decision Point.**")
    elif current_close > fibo_levels['50.0%']:
        st.success("Price is trading above the 50% Fibonacci Retracement.")
    else:
        st.error("Price is trading below the 50% Fibonacci Retracement.")

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

    VaR_95_hist = extract_scalar(returns.quantile(0.05))
    VaR_99_hist = extract_scalar(returns.quantile(0.01))

    CVaR_95 = extract_scalar(returns[returns <= VaR_95_hist].mean()) if not pd.isna(VaR_95_hist) else np.nan
    
    st.markdown(f"""
    **Current Price:** `{current_price:,.2f}`
    
    * **95% VaR (Historical):** `{VaR_95_hist*100:.3f}%` | Price Equivalent: `{current_price * (1 + VaR_95_hist):.2f}`
    * **99% VaR (Historical):** `{VaR_99_hist*100:.3f}%` | Price Equivalent (Long SL): `{current_price * (1 + VaR_99_hist):.2f}`
    * **Conditional VaR (CVaR) 95%:** `{CVaR_95*100:.3f}%` (Expected loss if VaR is breached)
    """)
    
    st.warning("⚠️ VaR provides a **leading risk boundary**. The 99% VaR price equivalent is a strong candidate for a **Stop Loss** level, as it represents a 1% chance of being breached.")


def main_dashboard():
    st.title("🎛️ Algo Dashboard with Keltner Channel & Backtesting")
    st.markdown("---")
    
    # --- 1. Data Fetching & Management (Sidebar) ---
    with st.sidebar:
        st.header("⚙️ Data & Config")
        
        col_t1, col_c1 = st.columns(2)
        ticker1_symbol = col_t1.selectbox("Ticker 1 Symbol", STANDARD_TICKERS, index=0)
        custom_ticker1 = col_c1.text_input("Custom Ticker 1 (Override)", value=st.session_state.get('ticker1', ''))
        
        ticker2_symbol = st.selectbox("Ticker 2 Symbol (Ratio Analysis)", STANDARD_TICKERS, index=1)
        
        interval = st.selectbox("Interval", TIME_INTERVALS, index=4) # Default 1h
        period = st.selectbox("Historical Period", PERIODS, index=4) # Default 1y
        
        # Update session state for future use
        st.session_state.ticker1 = custom_ticker1 if custom_ticker1 else ticker1_symbol
        st.session_state.ticker2 = ticker2_symbol
        st.session_state.interval = interval
        st.session_state.period = period
        
        if st.button("Fetch & Run Analysis"):
            st.session_state.data_fetched = False
            with st.spinner(f"Fetching data for {st.session_state.ticker1} and {st.session_state.ticker2}..."):
                # Fetch Ticker 1
                df1, status1 = fetch_and_process_data(st.session_state.ticker1, st.session_state.interval, st.session_state.period, 1)
                st.session_state.df1 = df1
                
                # Fetch Ticker 2
                df2, status2 = fetch_and_process_data(st.session_state.ticker2, st.session_state.interval, st.session_state.period, 0)
                st.session_state.df2 = df2
                
                if status1 is True and status2 is True:
                    st.session_state.data_fetched = True
                    st.success("Data fetched and processed successfully!")
                else:
                    st.error(f"Data fetch error T1: {status1 if status1 is not True else 'OK'}. T2: {status2 if status2 is not True else 'OK'}")
                    st.session_state.data_fetched = False
                    
        st.markdown("---")
        st.header("Multi-Timeframe (MTF)")
        st.session_state.mt_interval = st.selectbox("MTF Interval (e.g., Higher Timeframe)", ["1d", "1wk", "1mo"], index=0)
        
        if st.button("Fetch MTF Data"):
             with st.spinner(f"Fetching MTF data for {st.session_state.ticker1} at {st.session_state.mt_interval}..."):
                df_mt, status_mt = fetch_and_process_data(st.session_state.ticker1, st.session_state.mt_interval, st.session_state.period, 0)
                st.session_state.df_mt = df_mt
                if status_mt is not True:
                     st.error(f"MTF Data fetch error: {status_mt}")
                else:
                    st.success("MTF data fetched.")
                    
    # --- 2. Main Dashboard Content ---
    if not st.session_state.data_fetched and st.session_state.df1.empty:
        st.info("Please select tickers and click 'Fetch & Run Analysis' in the sidebar to begin.")
        return

    # A. Current Metrics & Signal
    col_metrics, col_signal = st.columns([1, 1])
    
    current_price, _ = calculate_basic_metrics(st.session_state.df1, st.session_state.ticker1, st.session_state.interval)
    signal, reason = get_kc_divergence_signal(st.session_state.df1)
    
    with col_signal:
        st.markdown("### 🎯 **Primary Trading Signal**")
        if "STRONG" in signal:
            st.markdown(f"## {signal}", unsafe_allow_html=True)
        else:
            st.markdown(f"### {signal}")
        st.info(reason)

    # B. Charting
    st.markdown("---")
    generate_candlestick_chart(st.session_state.df1, st.session_state.ticker1)
    
    # C. Advanced Analysis (Tabs)
    st.markdown("---")
    tab_backtest, tab_leading, tab_risk, tab_ratio, tab_mtf = st.tabs([
        "📊 Backtesting", 
        "⚡ Leading Indicators", 
        "📉 Risk (VaR)", 
        "⚖️ Ratio Analysis", 
        "🕰️ MTF Analysis"
    ])
    
    with tab_backtest:
        if current_price:
            run_backtest(st.session_state.df1, st.session_state.ticker1, current_price)
        else:
            st.warning("Cannot run backtest: Current price data is missing.")

    with tab_leading:
        perform_leading_analysis(st.session_state.df1, st.session_state.ticker1)
        
    with tab_risk:
        perform_value_at_risk(st.session_state.df1, st.session_state.ticker1)

    with tab_ratio:
        if st.session_state.df2.empty:
            st.info(f"Data for Ticker 2 ({st.session_state.ticker2}) is not available. Please ensure it was fetched successfully.")
        else:
            perform_ratio_analysis(st.session_state.df1, st.session_state.df2)

    with tab_mtf:
        if st.session_state.df_mt.empty:
            st.info("Please fetch MTF data in the sidebar first.")
        else:
            st.subheader(f"🕰️ MTF Check ({st.session_state.ticker1} on {st.session_state.mt_interval})")
            
            mtf_signal, mtf_reason = get_kc_divergence_signal(st.session_state.df_mt)
            
            # Display current MTF metric (if available)
            if not st.session_state.df_mt.empty:
                calculate_basic_metrics(st.session_state.df_mt, st.session_state.ticker1, st.session_state.mt_interval)
            
            st.markdown(f"**Higher Timeframe Signal:** {mtf_signal}")
            st.info(f"Reason: {mtf_reason}")
            
            if mtf_signal == signal:
                st.success("✅ **CONFLUENCE DETECTED!** The lower and higher timeframes agree on the primary signal.")
            else:
                st.warning("⚠️ **DIVERGENCE DETECTED!** The lower timeframe signal contradicts the higher timeframe signal. Trade with caution.")

if __name__ == "__main__":
    main_dashboard()
