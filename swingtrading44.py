import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# --- 1. CONFIGURATION AND CONSTANTS ---
st.set_page_config(layout="wide", page_title="Advanced Trading Analysis")

# YFinance supported intervals and periods
INTERVALS = [
    '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
]
# Mapping user-requested periods to yfinance values
PERIOD_MAP = {
    '1 Day': '1d', '5 Days': '5d', '7 Days': '7d', '1 Month': '1mo',
    '3 Months': '3mo', '6 Months': '6mo', '1 Year': '1y', '2 Years': '2y',
    '5 Years': '5y', '10 Years': '10y', '15 Years': '15y', '20 Years': '20y',
    '25 Years': '25y', '30 Years': '30y'
}

# Technical Indicator Parameters
RSI_PERIOD = 14
EMA_PERIOD = 50
DIVERGENCE_LOOKBACK = 20  # Lookback period in candles for divergence checks

# --- 2. DATA FETCHING AND PRE-PROCESSING ---

@st.cache_data(ttl=600)  # Cache data for 10 minutes to handle API rate limits
def fetch_data(ticker, period, interval):
    """
    Fetches data from yfinance and applies necessary cleaning and timezone conversion.
    Includes comprehensive debugging logs.
    """
    st.info(f"Fetching data for {ticker} over {period} using {interval} interval. This data is cached for 10 minutes.")
    
    try:
        # Use yf.download for more robust data access
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True, # Auto adjust true typically merges Adj Close into Close
            prepost=False,
            threads=True,
            proxy=None
        )
        
        if df.empty:
            st.error(f"No data found for the provided ticker: {ticker}. Please check the ticker symbol.")
            return None
        
        # --- DEBUG LOGGING 1: Raw Data Structure ---
        st.subheader("--- DEBUG 1: Raw Data from yfinance ---")
        st.info("Initial DataFrame columns (may be a MultiIndex):")
        st.code(df.columns.tolist())
        st.info("Initial DataFrame Head:")
        st.code(df.head().to_string())


        # 1. Flatten Multi-Index DataFrame (Required)
        df = df.reset_index()
        
        # 2. Standardize column names (FIX for KeyError: 'Close')
        # Robustly handle MultiIndex columns returned by yf.download when only one ticker is passed.
        new_columns = []
        for col in df.columns:
            # If the column is a tuple (MultiIndex structure)
            if isinstance(col, tuple):
                # We take the first element, which is the field name (e.g., 'Close', 'Open')
                name = str(col[0])
            else:
                # If it's already a simple string (like 'Date' or 'Datetime')
                name = str(col)

            # Apply standardization: uppercase and replace spaces with underscores
            new_columns.append(name.upper().replace(' ', '_'))
            
        df.columns = new_columns
        
        # Map standardized names back to required application names
        # IMPORTANT: 'ADJ_CLOSE' is mapped to 'Close' as a fallback/redundancy
        column_map = {
            'DATETIME': 'Date', 'DATE': 'Date', 
            'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 
            'CLOSE': 'Close', 'ADJ_CLOSE': 'Close', 'VOLUME': 'Volume'
        }
        
        # Only rename columns that actually exist in the DataFrame
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # --- DEBUG LOGGING 2: After Column Standardization ---
        st.subheader("--- DEBUG 2: After Standardization/Renaming ---")
        st.info("Current DataFrame columns:")
        st.code(df.columns.tolist())

        
        # Check if 'Close' column exists after mapping
        if 'Close' not in df.columns:
             st.error("Could not find the 'Close' price column in the fetched data. Data structure might be unusual.")
             return None

        # 3. Timezone Handling: Convert to IST (Asia/Kolkata) (Required)
        # Check if 'Date' column exists and is datetime
        if 'Date' in df.columns:
            # Attempt to ensure 'Date' is datetime
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception as e:
                st.warning(f"Failed to convert 'Date' column to datetime: {e}")
                
            if pd.api.types.is_datetime64_any_dtype(df['Date']):
                # Handle localization and conversion to IST
                if df['Date'].dt.tz is None:
                    # Assume raw timestamps from yfinance are UTC and localize/convert to IST
                    df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
                else: 
                    # Already timezone aware (convert to IST if it's not)
                    df['Date'] = df['Date'].dt.tz_convert('Asia/Kolkata')
        
        # 4. Ensure Volume is present
        if 'Volume' not in df.columns:
            df['Volume'] = 0 # Safety measure for indices that lack volume data

        st.success("Data fetch and standardization successful. Proceeding to analysis.")
        return df
    
    except Exception as e:
        st.error(f"FATAL ERROR during data fetch or processing: {e}")
        return None

# --- 3. MANUAL TECHNICAL INDICATOR CALCULATIONS ---

def calculate_indicators(df, rsi_period=RSI_PERIOD, ema_period=EMA_PERIOD):
    """Calculates EMA and RSI manually without external TA libraries."""
    
    # --- A. EMA (50) Calculation ---
    # Using pandas EWM (Exponential Weighted Moving) which is the correct mathematical implementation of EMA
    df[f'EMA_{ema_period}'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # --- B. RSI (14) Calculation ---
    
    # 1. Calculate price change
    delta = df['Close'].diff()
    
    # 2. Separate gains (positive changes) and losses (negative changes)
    gain = delta.apply(lambda x: x if x > 0 else 0)
    loss = delta.apply(lambda x: -x if x < 0 else 0)
    
    # 3. Calculate Exponential Moving Average of gains and losses
    # Initial smoothing is crucial for true RSI. Adjust=False mimics classic Wilder's smoothing.
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    
    # 4. Calculate Relative Strength (RS)
    # Check for division by zero (avg_loss = 0)
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    
    # 5. Calculate RSI
    df[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))
    
    return df

# --- 4. DIVERGENCE DETECTION (Simplified High-Level Check) ---

def detect_divergence(df, lookback=DIVERGENCE_LOOKBACK):
    """
    Detects RSI Divergence over the last 'lookback' period.
    NOTE: True divergence detection requires complex swing-point identification.
    This implementation uses a simplified check of the most recent price/RSI points.
    """
    
    # Check for the last 5 swing points in the lookback window
    recent_data = df.tail(lookback).copy()
    
    # Simplified swing points: look for 2 recent highs/lows
    
    # Find the previous swing point (2nd to last major extreme)
    swing_lows = recent_data[(recent_data['Close'] < recent_data['Close'].shift(1)) & (recent_data['Close'] < recent_data['Close'].shift(-1))].iloc[:-1] # Exclude latest possible point
    swing_highs = recent_data[(recent_data['Close'] > recent_data['Close'].shift(1)) & (recent_data['Close'] > recent_data['Close'].shift(-1))].iloc[:-1] # Exclude latest possible point
    
    # Bullish Divergence check (Lower Low in Price, Higher Low in RSI)
    if len(swing_lows) >= 2:
        recent_low = swing_lows['Close'].iloc[-1]
        previous_low = swing_lows['Close'].iloc[-2]
        
        recent_low_rsi = swing_lows[f'RSI_{RSI_PERIOD}'].iloc[-1]
        previous_low_rsi = swing_lows[f'RSI_{RSI_PERIOD}'].iloc[-2]
        
        if (recent_low < previous_low) and (recent_low_rsi > previous_low_rsi) and (recent_low_rsi < 40): # Added 40 threshold for better signal
            return "Bullish Divergence (Reversal Up)", previous_low, recent_low
            
    # Bearish Divergence check (Higher High in Price, Lower High in RSI)
    if len(swing_highs) >= 2:
        recent_high = swing_highs['Close'].iloc[-1]
        previous_high = swing_highs['Close'].iloc[-2]
        
        recent_high_rsi = swing_highs[f'RSI_{RSI_PERIOD}'].iloc[-1]
        previous_high_rsi = swing_highs[f'RSI_{RSI_PERIOD}'].iloc[-2]
        
        if (recent_high > previous_high) and (recent_high_rsi < previous_high_rsi) and (recent_high_rsi > 60): # Added 60 threshold for better signal
            return "Bearish Divergence (Reversal Down)", previous_high, recent_high
            
    return "No Divergence Detected", None, None

# --- 5. UI METRICS & DATA TABLE PREP ---

def prepare_display_df(df, ticker, interval):
    """Prepares the final DataFrame for display with calculated metrics and formatting."""
    
    display_df = df.copy()
    display_df = display_df.rename(columns={'Close': f'{ticker}_Close'})
    
    # 1. Points Gained/Lost from Previous Timeframe (Close vs Previous Close)
    display_df['Prev_Close'] = display_df[f'{ticker}_Close'].shift(1)
    display_df['Points_Gained_Lost'] = display_df[f'{ticker}_Close'] - display_df['Prev_Close']
    
    # 2. % of Points Gained/Lost from Previous Day Close (Requires 1d data or finding the previous day's close)
    # We use the previous candle's close for simplicity in intraday/non-daily views.
    display_df['%_Change_from_Prev'] = (display_df['Points_Gained_Lost'] / display_df['Prev_Close']) * 100
    
    # Drop first row which has NaN for the shift
    display_df = display_df.dropna(subset=['Prev_Close']).copy()
    
    # Select and rename columns for clarity
    display_df = display_df[[
        'Date', 'Open', 'High', 'Low', f'{ticker}_Close', 'Volume', 
        'Points_Gained_Lost', '%_Change_from_Prev', f'EMA_{EMA_PERIOD}', f'RSI_{RSI_PERIOD}'
    ]]
    display_df.columns = [
        'Timestamp (IST)', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Points Gained/Lost (Prev. Candle)', '% Change (Prev. Candle)', 
        f'EMA ({EMA_PERIOD})', f'RSI ({RSI_PERIOD})'
    ]
    
    # Apply conditional formatting for the Streamlit table
    def color_delta(val):
        """Colors the change columns green for positive, red for negative."""
        try:
            val = float(val)
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}'
        except:
            return ''

    # Define a style function for the DataFrame
    styler = display_df.style.map(color_delta, subset=['Points Gained/Lost (Prev. Candle)', '% Change (Prev. Candle)'])
    
    return display_df, styler

# --- 6. HEATMAP GENERATION ---

def plot_returns_heatmap(df, title):
    """Generates a heatmap of returns (Day vs Year or Month vs Year)."""
    
    # Calculate daily returns (as percentage change)
    df['Return'] = df['Close'].pct_change() * 100
    
    # For heatmaps, group by Year, Month, and Day
    df['Year'] = df['Timestamp (IST)'].dt.year
    df['Month'] = df['Timestamp (IST)'].dt.month
    df['Day'] = df['Timestamp (IST)'].dt.day
    df['DayOfWeek'] = df['Timestamp (IST)'].dt.day_name().str[:3] # Mon, Tue, etc.

    # Option 1: Month vs Year Heatmap (Average Monthly Return)
    monthly_returns = df.groupby(['Year', 'Month'])['Return'].mean().unstack(level='Year')
    monthly_returns.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_month = px.imshow(
        monthly_returns,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdYlGn',
        title=f"Heatmap of Average Monthly Returns (%) for {title}"
    )
    fig_month.update_xaxes(side="top")
    st.plotly_chart(fig_month, use_container_width=True)

    # Option 2: Day of Week vs Month Heatmap (Average Daily Return)
    daily_returns_monthly = df.groupby(['DayOfWeek', 'Month'])['Return'].mean().unstack(level='Month')
    # Reorder columns to calendar months
    daily_returns_monthly = daily_returns_monthly[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    daily_returns_monthly.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Reorder rows to standard week
    daily_returns_monthly = daily_returns_monthly.reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
    
    fig_day = px.imshow(
        daily_returns_monthly,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdYlGn',
        title=f"Heatmap of Average Daily Returns (%) by Day of Week for {title}"
    )
    fig_day.update_xaxes(side="top")
    st.plotly_chart(fig_day, use_container_width=True)


# --- 7. FINAL RECOMMENDATION ENGINE ---

def get_recommendation(df, divergence_info):
    """Generates a simple, human-readable recommendation based on indicators."""
    
    latest_data = df.iloc[-1]
    close = latest_data['Close']
    ema = latest_data[f'EMA_{EMA_PERIOD}']
    rsi = latest_data[f'RSI_{RSI_PERIOD}']
    
    signal = "HOLD"
    reason = "The market is currently in a neutral or consolidating phase, which means there is no strong evidence for a big move up or down. Wait for a clearer signal."
    entry = "N/A"
    sl = "N/A"
    t1 = "N/A"
    
    divergence, p_point, c_point = divergence_info
    
    # --- Check for Strong Divergence Signal ---
    if divergence.startswith("Bullish Divergence"):
        # Confirmation check: Price must be crossing above EMA
        if close > ema:
            signal = "BUY"
            entry = f"Enter a long position (Buy) at the current closing price of {close:.2f} or on a pullback."
            sl = f"Place Stop Loss (SL) below the recent swing low at {c_point:.2f}. This is the point where the bullish structure would be broken."
            t1 = f"Target 1 (T1) could be the last major high or a minimum 1:2 Risk-Reward ratio from the entry point. Start by aiming for the previous high at {p_point:.2f}."
            reason = (
                f"**STRONG BUY SIGNAL (RSI Divergence):** The price made a lower low ({c_point:.2f}) but the market momentum (RSI) made a higher low. "
                f"This means sellers are likely exhausted. The price is currently above the {EMA_PERIOD}-period EMA ({ema:.2f}), confirming the start of an uptrend. "
                "The current market structure suggests a potential strong reversal to the upside."
            )
    elif divergence.startswith("Bearish Divergence"):
        # Confirmation check: Price must be crossing below EMA
        if close < ema:
            signal = "SELL"
            entry = f"Enter a short position (Sell/Short) at the current closing price of {close:.2f} or on a rally."
            sl = f"Place Stop Loss (SL) above the recent swing high at {c_point:.2f}. This is the point where the bearish structure would be broken."
            t1 = f"Target 1 (T1) could be the last major low or a minimum 1:2 Risk-Reward ratio from the entry point. Start by aiming for the previous low at {p_point:.2f}."
            reason = (
                f"**STRONG SELL SIGNAL (RSI Divergence):** The price made a higher high ({c_point:.2f}) but the market momentum (RSI) made a lower high. "
                f"This means buyers are likely exhausted. The price is currently below the {EMA_PERIOD}-period EMA ({ema:.2f}), confirming the start of a downtrend. "
                "The current market structure suggests a potential strong reversal to the downside."
            )
    
    # --- Check for Simple EMA/RSI Signal (if no divergence) ---
    else:
        if close > ema and rsi < 70:
            signal = "BUY (Trend Following)"
            entry = f"Enter a long position (Buy) on a small dip towards the EMA ({ema:.2f})."
            sl = f"Place Stop Loss (SL) below the {EMA_PERIOD}-period EMA at {ema:.2f}. This helps protect profit if the trend reverses."
            t1 = "Target 1 (T1) should be set based on the previous swing high, aiming for a favorable Risk-Reward ratio (e.g., 1:1.5 or 1:2)."
            reason = (
                f"**MILD BUY SIGNAL:** The price ({close:.2f}) is trading clearly above the **{EMA_PERIOD}-period EMA** ({ema:.2f}), indicating a healthy short-term uptrend. "
                f"The **RSI is currently {rsi:.2f}** (not yet overbought), suggesting room for further upward movement. The market structure is bullish (higher highs and higher lows)."
            )
        elif close < ema and rsi > 30:
            signal = "SELL (Trend Following)"
            entry = f"Enter a short position (Sell) on a small rally towards the EMA ({ema:.2f})."
            sl = f"Place Stop Loss (SL) above the {EMA_PERIOD}-period EMA at {ema:.2f}. This helps protect against a reversal."
            t1 = "Target 1 (T1) should be set based on the previous swing low, aiming for a favorable Risk-Reward ratio."
            reason = (
                f"**MILD SELL SIGNAL:** The price ({close:.2f}) is trading clearly below the **{EMA_PERIOD}-period EMA** ({ema:.2f}), indicating a strong short-term downtrend. "
                f"The **RSI is currently {rsi:.2f}** (not yet oversold), suggesting room for further downward movement. The market structure is bearish (lower highs and lower lows)."
            )
        else: # Neutral/Overbought/Oversold but not confirming
            if rsi >= 70 and close > ema:
                reason = f"**CAUTION (Overbought):** Price is in a strong uptrend (above EMA), but RSI is high at {rsi:.2f}. Expect a minor pullback or consolidation soon. It's best to **HOLD** existing positions and wait for a clear entry."
            elif rsi <= 30 and close < ema:
                reason = f"**CAUTION (Oversold):** Price is in a strong downtrend (below EMA), but RSI is low at {rsi:.2f}. Expect a minor bounce or consolidation soon. It's best to **HOLD** existing positions and wait for a clear entry."
            
            signal = "HOLD"
    
    return signal, reason, entry, sl, t1


# --- 8. UI/STREAMLIT APP LAYOUT ---

def main():
    st.title("Financial Market Analysis Dashboard")
    st.markdown(
        """
        Welcome to the Market Analyzer. This tool is designed to provide simple, clear, and data-backed trading insights 
        without complex jargon. It uses common technical indicators to assess market momentum and potential reversals.
        
        **Important Note:** Bank Nifty F&O data is not directly available via the free `yfinance` source. 
        This tool uses major indices like **Nifty 50** (`^NSEI`) or **Bank Index** (e.g., `^NSEBANK` if supported) 
        and applies the same analysis logic. Use a stock ticker for best results.
        """
    )
    
    # --- Sidebar for Inputs ---
    st.sidebar.header("Data Selection")
    
    default_ticker = "^NSEI"
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., ^NSEI, RELIANCE.NS)", default_ticker).upper()
    
    # Filter periods based on interval (yfinance rule)
    available_intervals = INTERVALS
    interval = st.sidebar.selectbox("Select Timeframe (Interval)", available_intervals, index=available_intervals.index('1d'))
    
    # Filter out very long periods for intraday intervals
    available_periods = list(PERIOD_MAP.keys())
    if interval in ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h']:
        # yfinance limits intraday data period to 60 days
        available_periods = [p for p in available_periods if PERIOD_MAP[p] in ['1d', '5d', '7d', '1mo', '3mo']]
    
    # Set default period index based on interval type
    default_period_index = available_periods.index('6 Months') if '6 Months' in available_periods and '6 Months' in PERIOD_MAP else 0
    if interval in ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h']:
        default_period_index = available_periods.index('7 Days') if '7 Days' in available_periods and '7 Days' in PERIOD_MAP else 0


    period_key = st.sidebar.selectbox("Select Lookback Period", available_periods, index=default_period_index)
    period = PERIOD_MAP[period_key]

    if st.sidebar.button("Analyze Market"):
        if not ticker:
            st.sidebar.error("Please enter a valid ticker symbol.")
            return

        df = fetch_data(ticker, period, interval)
        
        # Guard clause to stop execution if data is missing
        if df is None or df.empty:
            return
        
        # --- CALCULATIONS ---
        df = calculate_indicators(df)
        
        # Prepare display data
        display_df, styled_df = prepare_display_df(df, ticker, interval)
        
        # Get latest values for recommendation
        divergence_info = detect_divergence(df)
        
        # --- DISPLAY: FINAL RECOMMENDATION ---
        st.header(f"📈 Current Market Recommendation for {ticker} ({interval})")
        
        signal, reason, entry, sl, t1 = get_recommendation(display_df, divergence_info)
        
        col_sig, col_rec = st.columns([1, 3])
        
        with col_sig:
            if "BUY" in signal:
                st.success(f"**Recommendation:** {signal}")
            elif "SELL" in signal:
                st.error(f"**Recommendation:** {signal}")
            else:
                st.warning(f"**Recommendation:** {signal}")
        
        with col_rec:
            st.markdown(f"**The Logic (Plain English):** {reason}")
        
        st.subheader("Actionable Trading Plan (Entry, SL, Target)")
        st.markdown(f"- **Entry Point:** {entry}")
        st.markdown(f"- **Stop Loss (SL):** {sl}")
        st.markdown(f"- **Target 1 (T1):** {t1}")

        
        # --- DISPLAY: INTERACTIVE CHART (Candlestick + EMA + RSI) ---
        st.header("Interactive Price Chart and Indicators")
        
        # Candlestick Chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=df['Date'],
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='Price (OHLC)'
            ),
            go.Scatter(
                x=df['Date'], y=df[f'EMA_{EMA_PERIOD}'], line=dict(color='orange', width=2),
                name=f'EMA ({EMA_PERIOD})'
            )
        ])
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            title=f'{ticker} Price Action with {EMA_PERIOD}-EMA',
            yaxis_title='Price',
        )

        # RSI Subplot
        fig_rsi = go.Figure(data=[
            go.Scatter(x=df['Date'], y=df[f'RSI_{RSI_PERIOD}'], line=dict(color='purple', width=1), name=f'RSI ({RSI_PERIOD})'),
            go.Layout(yaxis_range=[0, 100])
        ])
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(title=f'RSI ({RSI_PERIOD}) Momentum Indicator', yaxis_title='RSI Value')

        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig_rsi, use_container_width=True)

        
        # --- DISPLAY: HEATMAPS ---
        st.header("Historical Returns Heatmap (Structure Analysis)")
        st.markdown(
            """
            These heatmaps show the average percentage return for different time periods. 
            They help a common man understand which days or months have historically been strongest (Green) or weakest (Red).
            """
        )
        plot_returns_heatmap(display_df, ticker)

        # --- DISPLAY: DATA TABLE ---
        st.header("Raw OHLCV and Indicator Data")
        st.markdown(
            """
            This table shows the raw price data and calculated indicators. The 'Points Gained/Lost' 
            and '% Change' columns use colors to quickly show the momentum of the last candle/period 
            compared to the one before it.
            """
        )
        st.dataframe(styled_df, use_container_width=True, height=350)

        # --- EXPORT FUNCTIONALITY ---
        st.header("Data Export")
        
        # Remove the styler columns for clean export
        export_df = display_df.drop(columns=['Points Gained/Lost (Prev. Candle)', '% Change (Prev. Candle)'])
        
        # Ensure Timestamp (IST) is converted to a simple string format for CSV/Excel compatibility
        export_df['Timestamp (IST)'] = export_df['Timestamp (IST)'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        # Export to CSV
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Data to CSV",
            data=csv,
            file_name=f'{ticker}_{period}_{interval}_data.csv',
            mime='text/csv',
        )
        
        # Export to Excel (using io.BytesIO for in-memory file)
        excel_buffer = io.BytesIO()
        export_df.to_excel(excel_buffer, index=False, sheet_name='Trading Data')
        excel_buffer.seek(0)
        st.download_button(
            label="Export Data to Excel",
            data=excel_buffer,
            file_name=f'{ticker}_{period}_{interval}_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

if __name__ == "__main__":
    main()

