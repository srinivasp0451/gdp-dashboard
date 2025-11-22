import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
from io import BytesIO

# Page configuration
st.set_page_config(page_title="Algo Trading Dashboard", layout="wide")

# Helper Functions
def convert_to_ist(df):
    """Convert timezone-aware datetime to IST"""
    ist = pytz.timezone('Asia/Kolkata')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(ist)
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI manually"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    """Calculate EMA manually"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calculate SMA manually"""
    return data.rolling(window=period).mean()

def calculate_support_resistance(data, window=20):
    """Calculate support and resistance levels"""
    highs = data['High'].rolling(window=window).max()
    lows = data['Low'].rolling(window=window).min()
    return lows.iloc[-1], highs.iloc[-1]

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    return {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100.0%': low
    }

def fetch_data_with_delay(ticker, interval, period, delay=1):
    """Fetch data with delay to respect API limits"""
    time.sleep(delay)
    try:
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        if not data.empty:
            data = convert_to_ist(data)
        return data
    except Exception as e:
        st.error(f"Error fetching {ticker} for {interval}/{period}: {str(e)}")
        return pd.DataFrame()

def analyze_timeframe(data, timeframe_name):
    """Analyze a single timeframe"""
    if data.empty or len(data) < 200:
        return None
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # Trend
    trend = "Up" if close.iloc[-1] > close.iloc[0] else "Down"
    
    # Price metrics
    max_close = close.max()
    min_close = close.min()
    current_close = close.iloc[-1]
    
    # Fibonacci
    fib_levels = calculate_fibonacci_levels(max_close, min_close)
    fib_50 = fib_levels['50.0%']
    
    # Volatility (standard deviation)
    volatility = close.std()
    
    # Returns
    pct_change = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    points_change = close.iloc[-1] - close.iloc[0]
    
    # Support and Resistance
    support, resistance = calculate_support_resistance(data)
    
    # RSI
    rsi = calculate_rsi(close)
    current_rsi = rsi.iloc[-1] if not rsi.empty else np.nan
    
    if current_rsi < 30:
        rsi_status = "Oversold"
        rsi_color = "green"
    elif current_rsi > 70:
        rsi_status = "Overbought"
        rsi_color = "red"
    else:
        rsi_status = "Neutral"
        rsi_color = "yellow"
    
    # EMAs
    ema_9 = calculate_ema(close, 9).iloc[-1]
    ema_20 = calculate_ema(close, 20).iloc[-1]
    ema_21 = calculate_ema(close, 21).iloc[-1]
    ema_33 = calculate_ema(close, 33).iloc[-1]
    ema_50 = calculate_ema(close, 50).iloc[-1]
    ema_100 = calculate_ema(close, 100).iloc[-1]
    ema_150 = calculate_ema(close, 150).iloc[-1]
    ema_200 = calculate_ema(close, 200).iloc[-1]
    
    # SMAs
    sma_20 = calculate_sma(close, 20).iloc[-1]
    sma_50 = calculate_sma(close, 50).iloc[-1]
    sma_100 = calculate_sma(close, 100).iloc[-1]
    sma_150 = calculate_sma(close, 150).iloc[-1]
    sma_200 = calculate_sma(close, 200).iloc[-1]
    
    return {
        'Timeframe': timeframe_name,
        'Trend': trend,
        'Max': f"{max_close:.2f}",
        'Min': f"{min_close:.2f}",
        'Fib 50%': f"{fib_50:.2f}",
        'Volatility': f"{volatility:.2f}",
        '% Change': pct_change,
        'Points': points_change,
        'Support': f"{support:.2f}",
        'Resistance': f"{resistance:.2f}",
        'RSI': f"{current_rsi:.2f}",
        'RSI Status': rsi_status,
        'RSI Color': rsi_color,
        '9 EMA': f"{ema_9:.2f}",
        '20 EMA': f"{ema_20:.2f}",
        '21 EMA': f"{ema_21:.2f}",
        '33 EMA': f"{ema_33:.2f}",
        '50 EMA': f"{ema_50:.2f}",
        '100 EMA': f"{ema_100:.2f}",
        '150 EMA': f"{ema_150:.2f}",
        '200 EMA': f"{ema_200:.2f}",
        'vs 20 EMA': 'Above' if current_close > ema_20 else 'Below',
        'vs 50 EMA': 'Above' if current_close > ema_50 else 'Below',
        'vs 100 EMA': 'Above' if current_close > ema_100 else 'Below',
        'vs 150 EMA': 'Above' if current_close > ema_150 else 'Below',
        'vs 200 EMA': 'Above' if current_close > ema_200 else 'Below',
        '20 SMA': f"{sma_20:.2f}",
        '50 SMA': f"{sma_50:.2f}",
        '100 SMA': f"{sma_100:.2f}",
        '150 SMA': f"{sma_150:.2f}",
        '200 SMA': f"{sma_200:.2f}",
        'vs 20 SMA': 'Above' if current_close > sma_20 else 'Below',
        'vs 50 SMA': 'Above' if current_close > sma_50 else 'Below',
        'vs 100 SMA': 'Above' if current_close > sma_100 else 'Below',
        'vs 150 SMA': 'Above' if current_close > sma_150 else 'Below',
        'vs 200 SMA': 'Above' if current_close > sma_200 else 'Below',
    }

def style_dataframe(df):
    """Apply styling to dataframe"""
    def color_trend(val):
        color = 'green' if val == 'Up' else 'red'
        return f'background-color: {color}; color: white'
    
    def color_pct(val):
        try:
            v = float(val)
            color = 'green' if v > 0 else 'red' if v < 0 else 'gray'
            return f'background-color: {color}; color: white'
        except:
            return ''
    
    def color_position(val):
        color = 'green' if val == 'Above' else 'red'
        return f'background-color: {color}; color: white'
    
    styled = df.style
    if 'Trend' in df.columns:
        styled = styled.applymap(color_trend, subset=['Trend'])
    if '% Change' in df.columns:
        styled = styled.applymap(color_pct, subset=['% Change'])
    
    ema_sma_cols = [col for col in df.columns if col.startswith('vs ')]
    for col in ema_sma_cols:
        styled = styled.applymap(color_position, subset=[col])
    
    return styled

# Title
st.title("🚀 Advanced Algorithmic Trading Dashboard")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Ticker Selection
ticker_options = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "Custom": "Custom"
}

ticker1_name = st.sidebar.selectbox("Select Ticker 1", list(ticker_options.keys()))
if ticker1_name == "Custom":
    ticker1 = st.sidebar.text_input("Enter Ticker 1 Symbol", "AAPL")
else:
    ticker1 = ticker_options[ticker1_name]

ticker2_name = st.sidebar.selectbox("Select Ticker 2", list(ticker_options.keys()), index=1)
if ticker2_name == "Custom":
    ticker2 = st.sidebar.text_input("Enter Ticker 2 Symbol", "MSFT")
else:
    ticker2 = ticker_options[ticker2_name]

# Timeframe and Period
interval = st.sidebar.selectbox(
    "Select Interval",
    ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
)

period = st.sidebar.selectbox(
    "Select Period",
    ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'ticker1_data' not in st.session_state:
    st.session_state.ticker1_data = None
if 'ticker2_data' not in st.session_state:
    st.session_state.ticker2_data = None

# Fetch Data Button
if st.sidebar.button("🔄 Fetch Data & Analyze", type="primary"):
    with st.spinner("Fetching data... Please wait."):
        st.session_state.ticker1_data = fetch_data_with_delay(ticker1, interval, period, delay=1)
        st.session_state.ticker2_data = fetch_data_with_delay(ticker2, interval, period, delay=2)
        st.session_state.data_fetched = True
        st.success("Data fetched successfully!")

# Main Analysis
if st.session_state.data_fetched and st.session_state.ticker1_data is not None:
    data1 = st.session_state.ticker1_data
    data2 = st.session_state.ticker2_data
    
    if not data1.empty and not data2.empty:
        # Calculate Ratio
        ratio_data = data1['Close'] / data2['Close']
        
        # Section 1: Basic Statistics
        st.header("📊 Basic Statistics & Ratio Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{ticker1} Current Price", f"{data1['Close'].iloc[-1]:.2f}",
                     f"{((data1['Close'].iloc[-1] - data1['Close'].iloc[0])/data1['Close'].iloc[0]*100):.2f}%")
        with col2:
            st.metric(f"{ticker2} Current Price", f"{data2['Close'].iloc[-1]:.2f}",
                     f"{((data2['Close'].iloc[-1] - data2['Close'].iloc[0])/data2['Close'].iloc[0]*100):.2f}%")
        with col3:
            st.metric("Current Ratio", f"{ratio_data.iloc[-1]:.4f}",
                     f"{((ratio_data.iloc[-1] - ratio_data.iloc[0])/ratio_data.iloc[0]*100):.2f}%")
        
        # Ratio Analysis Table
        st.subheader("Ticker Comparison with RSI")
        ratio_df = pd.DataFrame({
            'DateTime (IST)': data1.index,
            'Ticker1 Price': data1['Close'].values,
            'Ticker2 Price': data2['Close'].values,
            'Ratio': ratio_data.values,
            'RSI Ticker1': calculate_rsi(data1['Close']).values,
            'RSI Ticker2': calculate_rsi(data2['Close']).values,
            'RSI Ratio': calculate_rsi(ratio_data).values
        })
        st.dataframe(ratio_df.tail(50), use_container_width=True)
        
        # Export Button
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()
        
        excel_data = convert_df_to_excel(ratio_df)
        st.download_button(
            label="📥 Download Ratio Data (Excel)",
            data=excel_data,
            file_name=f"{ticker1}_{ticker2}_ratio_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Multi-Timeframe Analysis
        st.header("📈 Multi-Timeframe Analysis")
        
        timeframes = [
            ("1m", "1d"), ("5m", "5d"), ("15m", "5d"), ("30m", "1mo"),
            ("1h", "1mo"), ("2h", "3mo"), ("4h", "6mo"), ("1d", "1y"),
            ("1wk", "5y"), ("1mo", "10y")
        ]
        
        analysis_results = []
        
        with st.spinner("Performing multi-timeframe analysis..."):
            progress_bar = st.progress(0)
            for idx, (tf_interval, tf_period) in enumerate(timeframes):
                tf_data = fetch_data_with_delay(ticker1, tf_interval, tf_period, delay=1.5)
                if not tf_data.empty:
                    result = analyze_timeframe(tf_data, f"{tf_interval}/{tf_period}")
                    if result:
                        analysis_results.append(result)
                progress_bar.progress((idx + 1) / len(timeframes))
        
        if analysis_results:
            mtf_df = pd.DataFrame(analysis_results)
            st.dataframe(style_dataframe(mtf_df), use_container_width=True, height=400)
            
            # Summary
            st.subheader("🔍 Multi-Timeframe Summary")
            up_trends = sum(1 for r in analysis_results if r['Trend'] == 'Up')
            down_trends = len(analysis_results) - up_trends
            avg_rsi = np.mean([float(r['RSI']) for r in analysis_results if r['RSI'] != 'nan'])
            
            st.markdown(f"""
            **Key Insights:**
            - **Bullish Timeframes:** {up_trends}/{len(analysis_results)}
            - **Bearish Timeframes:** {down_trends}/{len(analysis_results)}
            - **Average RSI:** {avg_rsi:.2f}
            - **Overall Bias:** {'**BULLISH** 🟢' if up_trends > down_trends else '**BEARISH** 🔴' if down_trends > up_trends else '**NEUTRAL** 🟡'}
            
            **Recommendation:** Based on multi-timeframe analysis, the market shows {'strong upward momentum' if up_trends > down_trends else 'downward pressure' if down_trends > up_trends else 'consolidation'}. 
            Current price: {data1['Close'].iloc[-1]:.2f} ({((data1['Close'].iloc[-1] - data1['Close'].iloc[0])/data1['Close'].iloc[0]*100):.2f}% change)
            """)
        
        # Volatility Binning Analysis
        st.header("📊 Volatility Bins Analysis")
        
        volatility = data1['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        vol_bins = pd.qcut(volatility.dropna(), q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        vol_analysis = pd.DataFrame({
            'DateTime (IST)': data1.index[20:],
            'Volatility Bin': vol_bins.values,
            'Returns (Points)': data1['Close'].diff().iloc[20:].values,
            'Returns (%)': data1['Close'].pct_change().iloc[20:].values * 100
        })
        
        st.dataframe(vol_analysis.tail(30), use_container_width=True)
        
        current_vol_bin = vol_bins.iloc[-1] if len(vol_bins) > 0 else 'Unknown'
        st.info(f"**Current Volatility Bin:** {current_vol_bin}")
        
        # Ratio Bins Analysis
        st.header("📊 Ratio Bins Analysis")
        
        ratio_bins = pd.qcut(ratio_data.dropna(), q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        ratio_analysis = pd.DataFrame({
            'DateTime (IST)': data1.index[:len(ratio_bins)],
            'Ratio Bin': ratio_bins.values,
            'Ticker1 Returns (Points)': data1['Close'][:len(ratio_bins)].diff().values,
            'Ticker1 Returns (%)': data1['Close'][:len(ratio_bins)].pct_change().values * 100,
            'Ticker2 Returns (Points)': data2['Close'][:len(ratio_bins)].diff().values,
            'Ticker2 Returns (%)': data2['Close'][:len(ratio_bins)].pct_change().values * 100
        })
        
        st.dataframe(ratio_analysis.tail(30), use_container_width=True)
        
        current_ratio_bin = ratio_bins.iloc[-1] if len(ratio_bins) > 0 else 'Unknown'
        st.info(f"**Current Ratio Bin:** {current_ratio_bin}")
        
        # Plotting
        st.header("📈 Interactive Charts")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(f'{ticker1} Price', f'{ticker2} Price', 'Ratio',
                           f'RSI {ticker1}', f'RSI {ticker2}', 'RSI Ratio'),
            vertical_spacing=0.1
        )
        
        # Price charts
        fig.add_trace(go.Scatter(x=data1.index, y=data1['Close'], name=ticker1, line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data2.index, y=data2['Close'], name=ticker2, line=dict(color='orange')), row=1, col=2)
        fig.add_trace(go.Scatter(x=data1.index, y=ratio_data, name='Ratio', line=dict(color='green')), row=2, col=1)
        
        # RSI charts
        rsi1 = calculate_rsi(data1['Close'])
        rsi2 = calculate_rsi(data2['Close'])
        rsi_ratio = calculate_rsi(ratio_data)
        
        fig.add_trace(go.Scatter(x=data1.index, y=rsi1, name=f'RSI {ticker1}', line=dict(color='purple')), row=2, col=2)
        fig.add_trace(go.Scatter(x=data2.index, y=rsi2, name=f'RSI {ticker2}', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data1.index, y=rsi_ratio, name='RSI Ratio', line=dict(color='brown')), row=3, col=2)
        
        # Add RSI levels
        for row in [2, 3]:
            for col in [1, 2]:
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=col)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=col)
        
        fig.update_layout(height=1000, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Chart Insights")
        st.markdown("""
        **Divergence Analysis:** Look for divergences between price and RSI. Bullish divergence (price makes lower low, RSI makes higher low) 
        suggests potential reversal to upside. Bearish divergence indicates potential downside.
        
        **Current Market Status:** Compare current price action with historical patterns to identify similar setups.
        """)
        
        # Returns Distribution
        st.header("📊 Returns Distribution Analysis")
        
        returns1_points = data1['Close'].diff()
        returns1_pct = data1['Close'].pct_change() * 100
        returns2_points = data2['Close'].diff()
        ratio_returns = ratio_data.diff()
        
        fig_hist = make_subplots(
            rows=2, cols=3,
            subplot_titles=(f'{ticker1} Returns (Points)', f'{ticker1} Returns (%)', 
                           f'{ticker2} Returns (Points)', f'{ticker2} Returns (%)',
                           'Ratio Returns (Points)', 'Ratio Returns (%)'),
            vertical_spacing=0.15
        )
        
        fig_hist.add_trace(go.Histogram(x=returns1_points.dropna(), name=f'{ticker1} Points', nbinsx=50), row=1, col=1)
        fig_hist.add_trace(go.Histogram(x=returns1_pct.dropna(), name=f'{ticker1} %', nbinsx=50), row=1, col=2)
        fig_hist.add_trace(go.Histogram(x=returns2_points.dropna(), name=f'{ticker2} Points', nbinsx=50), row=1, col=3)
        fig_hist.add_trace(go.Histogram(x=(data2['Close'].pct_change() * 100).dropna(), name=f'{ticker2} %', nbinsx=50), row=2, col=1)
        fig_hist.add_trace(go.Histogram(x=ratio_returns.dropna(), name='Ratio Points', nbinsx=50), row=2, col=2)
        fig_hist.add_trace(go.Histogram(x=(ratio_data.pct_change() * 100).dropna(), name='Ratio %', nbinsx=50), row=2, col=3)
        
        fig_hist.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Z-Score Analysis
        st.header("📊 Z-Score Analysis")
        
        z_returns1 = (returns1_points - returns1_points.mean()) / returns1_points.std()
        z_returns2 = (returns2_points - returns2_points.mean()) / returns2_points.std()
        z_ratio = (ratio_returns - ratio_returns.mean()) / ratio_returns.std()
        
        z_df = pd.DataFrame({
            'DateTime (IST)': data1.index,
            'Z-Score Ticker1': z_returns1.values,
            'Z-Score Ticker2': z_returns2.values,
            'Z-Score Ratio': z_ratio.values
        })
        
        st.dataframe(z_df.tail(30), use_container_width=True)
        
        current_z1 = z_returns1.iloc[-1]
        current_z2 = z_returns2.iloc[-1]
        current_z_ratio = z_ratio.iloc[-1]
        
        st.subheader("🔍 Z-Score Insights")
        st.markdown(f"""
        **Current Z-Scores:**
        - {ticker1}: {current_z1:.2f}
        - {ticker2}: {current_z2:.2f}
        - Ratio: {current_z_ratio:.2f}
        
        **Interpretation:**
        - Z-Score > 2: Extreme positive deviation (potential reversal down)
        - Z-Score < -2: Extreme negative deviation (potential reversal up)
        - -1 < Z-Score < 1: Normal range
        
        **Forecast:** {'Ticker1 may reverse downward' if current_z1 > 2 else 'Ticker1 may reverse upward' if current_z1 < -2 else 'Ticker1 in normal range'}
        """)
        
        # Final Summary & Trading Signal
        st.header("🎯 Final Trading Recommendation")
        
        # Calculate signals
        trend_signal = 1 if up_trends > down_trends else -1 if down_trends > up_trends else 0
        rsi_signal = -1 if avg_rsi > 70 else 1 if avg_rsi < 30 else 0
        z_signal = -1 if current_z1 > 2 else 1 if current_z1 < -2 else 0
        
        total_signal = trend_signal + rsi_signal + z_signal
        
        current_price = data1['Close'].iloc[-1]
        atr = (data1['High'] - data1['Low']).rolling(14).mean().iloc[-1]
        
        if total_signal >= 2:
            action = "🟢 STRONG BUY"
            entry = current_price
            target = current_price + (2 * atr)
            sl = current_price - atr
        elif total_signal >= 1:
            action = "🟢 BUY"
            entry = current_price
            target = current_price + (1.5 * atr)
            sl = current_price - (0.75 * atr)
        elif total_signal <= -2:
            action = "🔴 STRONG SELL"
            entry = current_price
            target = current_price - (2 * atr)
            sl = current_price + atr
        elif total_signal <= -1:
            action = "🔴 SELL"
            entry = current_price
            target = current_price - (1.5 * atr)
            sl = current_price + (0.75 * atr)
        else:
            action = "🟡 HOLD"
            entry = current_price
            target = current_price
            sl = current_price
        
        st.markdown(f"""
        ## {action}
        
        **Signal Strength:** {abs(total_signal)}/3
        
        **Trade Setup:**
        - **Entry Price:** {entry:.2f}
        - **Target Price:** {target:.2f} ({((target-entry)/entry*100):.2f}%)
        - **Stop Loss:** {sl:.2f} ({((sl-entry)/entry*100):.2f}%)
        - **Risk/Reward Ratio:** {abs((target-entry)/(sl-entry)):.2f}
        
        **Logic:**
        - Multi-timeframe trend: {trend_signal} ({up_trends} bullish / {down_trends} bearish timeframes)
        - RSI indicator: {rsi_signal} (Average RSI: {avg_rsi:.2f})
        - Z-Score deviation: {z_signal} (Current: {current_z1:.2f})
        
        **Risk Management:**
        - Position size should not exceed 2% of portfolio
        - Use trailing stop loss after 50% profit
        - Monitor volume and volatility for exit signals
        
        **Disclaimer:** This is an algorithmic analysis based on historical data. Always do your own research and consult a financial advisor.
        """)
        
    else:
        st.warning("Unable to fetch data for one or both tickers. Please try different symbols or timeframes.")

else:
    st.info("👈 Configure your parameters in the sidebar and click 'Fetch Data & Analyze' to begin.")
    
    st.markdown("""
    ### Features:
    - 📊 Multi-asset support (Indices, Crypto, Forex, Commodities)
    - ⏰ Multiple timeframes (1m to 1mo)
    - 📈 Advanced technical analysis (RSI, EMA, SMA, Fibonacci)
    - 🔄 Ratio analysis between two tickers
    - 📉 Volatility and return distribution
    - 🎯 Z-Score statistical analysis
    - 🤖 Algorithmic trading signals
    - 💾 Export data to Excel
    
    ### How to Use:
    1. Select two tickers from the sidebar
    2. Choose your preferred interval and period
    3. Click "Fetch Data & Analyze"
    4. Explore comprehensive analysis and trading signals
    """)
