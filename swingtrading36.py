import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Page config
st.set_page_config(page_title="Algo Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📈 Advanced Algo Trading Dashboard</div>', unsafe_allow_html=True)

# Predefined symbols mapping
SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "Custom": "CUSTOM"
}

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = None
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Ticker selection
ticker_choice = st.sidebar.selectbox("Select Asset", list(SYMBOLS.keys()))

if ticker_choice == "Custom":
    ticker_symbol = st.sidebar.text_input("Enter Yahoo Finance Ticker", value="AAPL").upper()
else:
    ticker_symbol = SYMBOLS[ticker_choice]

# Comparison ticker
st.sidebar.subheader("Ratio Analysis")
comparison_choice = st.sidebar.selectbox("Compare with", ["None"] + list(SYMBOLS.keys()), index=0)

if comparison_choice == "Custom":
    comparison_symbol = st.sidebar.text_input("Enter Comparison Ticker", value="SPY").upper()
elif comparison_choice == "None":
    comparison_symbol = None
else:
    comparison_symbol = SYMBOLS[comparison_choice]

# Timeframe and Period
col1, col2 = st.sidebar.columns(2)
with col1:
    timeframe = st.selectbox("Timeframe", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])
with col2:
    period = st.selectbox("Period", ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])

# RSI parameters
st.sidebar.subheader("📊 RSI Settings")
rsi_period = st.sidebar.slider("RSI Period", 5, 50, 14)
rsi_overbought = st.sidebar.slider("Overbought Level", 60, 90, 70)
rsi_oversold = st.sidebar.slider("Oversold Level", 10, 40, 30)

# Fetch button
fetch_button = st.sidebar.button("🔄 Fetch Data & Analyze", type="primary", use_container_width=True)

# Functions
def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_divergences(price, rsi, window=5):
    """Detect bullish and bearish divergences"""
    bullish_divs = []
    bearish_divs = []
    
    for i in range(window, len(price) - window):
        # Price local minima
        if price.iloc[i] == price.iloc[i-window:i+window+1].min():
            # Check if RSI is making higher lows
            for j in range(i-window*3, i):
                if price.iloc[j] == price.iloc[max(0, j-window):min(len(price), j+window+1)].min():
                    if price.iloc[i] < price.iloc[j] and rsi.iloc[i] > rsi.iloc[j]:
                        bullish_divs.append((price.index[j], price.index[i], price.iloc[j], price.iloc[i]))
                        break
        
        # Price local maxima
        if price.iloc[i] == price.iloc[i-window:i+window+1].max():
            # Check if RSI is making lower highs
            for j in range(i-window*3, i):
                if price.iloc[j] == price.iloc[max(0, j-window):min(len(price), j+window+1)].max():
                    if price.iloc[i] > price.iloc[j] and rsi.iloc[i] < rsi.iloc[j]:
                        bearish_divs.append((price.index[j], price.index[i], price.iloc[j], price.iloc[i]))
                        break
    
    return bullish_divs, bearish_divs

def fetch_data(symbol, period, interval):
    """Fetch data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_metrics(df):
    """Calculate performance metrics"""
    if df is None or len(df) < 2:
        return None
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    first_price = df['Close'].iloc[0]
    
    daily_change = current_price - prev_price
    daily_change_pct = (daily_change / prev_price) * 100
    total_change = current_price - first_price
    total_change_pct = (total_change / first_price) * 100
    
    high = df['High'].max()
    low = df['Low'].min()
    
    return {
        'current_price': current_price,
        'daily_change': daily_change,
        'daily_change_pct': daily_change_pct,
        'total_change': total_change,
        'total_change_pct': total_change_pct,
        'high': high,
        'low': low,
        'range': high - low
    }

# Fetch data when button is clicked
if fetch_button:
    with st.spinner('📥 Fetching data...'):
        data = fetch_data(ticker_symbol, period, timeframe)
        if data is not None:
            st.session_state.data = data
            st.session_state.ticker_symbol = ticker_symbol
            
            # Fetch comparison data if needed
            if comparison_symbol:
                comparison_data = fetch_data(comparison_symbol, period, timeframe)
                st.session_state.comparison_data = comparison_data
            else:
                st.session_state.comparison_data = None
            
            st.success('✅ Data fetched successfully!')

# Display analysis if data exists
if st.session_state.data is not None:
    data = st.session_state.data
    ticker_symbol = st.session_state.ticker_symbol
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    if metrics:
        # Display metrics
        st.subheader(f"📊 {ticker_symbol} - Summary Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            color = "green" if metrics['daily_change'] >= 0 else "red"
            st.metric("Current Price", f"₹{metrics['current_price']:.2f}", 
                     f"{metrics['daily_change']:.2f} ({metrics['daily_change_pct']:.2f}%)")
        
        with col2:
            color = "green" if metrics['total_change'] >= 0 else "red"
            st.metric("Period Change", f"{metrics['total_change']:.2f}", 
                     f"{metrics['total_change_pct']:.2f}%")
        
        with col3:
            st.metric("Period High", f"₹{metrics['high']:.2f}")
        
        with col4:
            st.metric("Period Low", f"₹{metrics['low']:.2f}")
        
        with col5:
            st.metric("Range", f"₹{metrics['range']:.2f}")
        
        # Data table
        st.subheader("📋 Price Data")
        display_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        display_df['Change'] = display_df['Close'].diff()
        display_df['Change %'] = display_df['Close'].pct_change() * 100
        
        # Color code the table
        def color_negative_red(val):
            if isinstance(val, (int, float)):
                color = 'green' if val >= 0 else 'red'
                return f'color: {color}'
            return ''
        
        st.dataframe(display_df.tail(20).style.applymap(color_negative_red, subset=['Change', 'Change %']), 
                    use_container_width=True, height=300)
        
        # Calculate RSI
        data['RSI'] = calculate_rsi(data, rsi_period)
        
        # Calculate ratio if comparison data exists
        if st.session_state.comparison_data is not None:
            comparison_data = st.session_state.comparison_data
            # Align dates
            aligned_data = pd.concat([data['Close'], comparison_data['Close']], axis=1, join='inner')
            aligned_data.columns = ['Primary', 'Comparison']
            aligned_data['Ratio'] = aligned_data['Primary'] / aligned_data['Comparison']
            aligned_data['Ratio_RSI'] = calculate_rsi(aligned_data[['Ratio']].rename(columns={'Ratio': 'Close'}), rsi_period)
        else:
            aligned_data = None
        
        # Find divergences
        bullish_divs, bearish_divs = find_divergences(data['Close'], data['RSI'])
        
        # Create plots
        st.subheader("📈 Technical Analysis Charts")
        
        # Determine number of subplots
        num_plots = 3 if aligned_data is None else 4
        subplot_titles = ['Price Chart', 'RSI', 'Price RSI Divergences']
        if aligned_data is not None:
            subplot_titles = ['Price Chart', f'Ratio: {ticker_symbol}/{comparison_symbol}', 
                            'Price RSI', 'Ratio RSI']
        
        fig = make_subplots(
            rows=num_plots, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=[0.4] + [0.2] * (num_plots - 1)
        )
        
        # 1. Price Chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add divergence lines on price chart
        for div in bullish_divs:
            fig.add_trace(go.Scatter(
                x=[div[0], div[1]],
                y=[div[2], div[3]],
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                showlegend=False,
                hovertemplate='Bullish Divergence<extra></extra>'
            ), row=1, col=1)
        
        for div in bearish_divs:
            fig.add_trace(go.Scatter(
                x=[div[0], div[1]],
                y=[div[2], div[3]],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False,
                hovertemplate='Bearish Divergence<extra></extra>'
            ), row=1, col=1)
        
        current_row = 2
        
        # 2. Ratio Chart (if comparison exists)
        if aligned_data is not None:
            fig.add_trace(go.Scatter(
                x=aligned_data.index,
                y=aligned_data['Ratio'],
                mode='lines',
                name='Ratio',
                line=dict(color='purple', width=2)
            ), row=current_row, col=1)
            
            # Find divergences on ratio
            ratio_bullish, ratio_bearish = find_divergences(aligned_data['Ratio'], aligned_data['Ratio_RSI'].dropna())
            
            for div in ratio_bullish:
                fig.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[div[2], div[3]],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=False
                ), row=current_row, col=1)
            
            for div in ratio_bearish:
                fig.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[div[2], div[3]],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ), row=current_row, col=1)
            
            current_row += 1
        
        # 3. Price RSI
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue', width=2)
        ), row=current_row, col=1)
        
        fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", 
                     annotation_text="Overbought", row=current_row, col=1)
        fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", 
                     annotation_text="Oversold", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        
        # Add RSI divergence lines
        rsi_data_for_div = data['RSI'].dropna()
        rsi_bullish, rsi_bearish = find_divergences(rsi_data_for_div, rsi_data_for_div)
        
        for div in bullish_divs:
            if div[0] in data.index and div[1] in data.index:
                fig.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[data.loc[div[0], 'RSI'], data.loc[div[1], 'RSI']],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=False
                ), row=current_row, col=1)
        
        for div in bearish_divs:
            if div[0] in data.index and div[1] in data.index:
                fig.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[data.loc[div[0], 'RSI'], data.loc[div[1], 'RSI']],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ), row=current_row, col=1)
        
        current_row += 1
        
        # 4. Ratio RSI (if comparison exists)
        if aligned_data is not None:
            fig.add_trace(go.Scatter(
                x=aligned_data.index,
                y=aligned_data['Ratio_RSI'],
                mode='lines',
                name='Ratio RSI',
                line=dict(color='orange', width=2)
            ), row=current_row, col=1)
            
            fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", 
                         row=current_row, col=1)
            fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", 
                         row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        
        # Update layout
        fig.update_layout(
            height=300 * num_plots,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Divergence Summary
        st.subheader("🔍 Divergence Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bullish Divergences", len(bullish_divs), delta="Potential Upside")
        
        with col2:
            st.metric("Bearish Divergences", len(bearish_divs), delta="Potential Downside", delta_color="inverse")
        
        # Download options
        st.subheader("💾 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv = data.to_csv()
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{ticker_symbol}_{period}_{timeframe}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Price Data')
                if aligned_data is not None:
                    aligned_data.to_excel(writer, sheet_name='Ratio Data')
            
            st.download_button(
                label="📥 Download as Excel",
                data=buffer.getvalue(),
                file_name=f"{ticker_symbol}_{period}_{timeframe}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

else:
    # Instructions
    st.info("👈 Configure your settings in the sidebar and click '🔄 Fetch Data & Analyze' to start")
    
    st.markdown("""
    ### 🚀 Features:
    - **Multiple Assets**: NIFTY 50, Bank NIFTY, SENSEX, Crypto, Forex, Commodities, Stocks
    - **Flexible Timeframes**: 1m to 1mo intervals
    - **Multiple Periods**: 1 day to max available history
    - **Technical Analysis**: RSI, Divergences (Bullish/Bearish)
    - **Ratio Analysis**: Compare two assets and analyze relative strength
    - **Visual Divergence Detection**: Automatic trendlines on price and RSI charts
    - **Performance Metrics**: Daily and period returns with color-coded changes
    - **Data Export**: Download as CSV or Excel
    
    ### 📌 How to Use:
    1. Select an asset from the dropdown or enter a custom ticker
    2. (Optional) Select a comparison asset for ratio analysis
    3. Choose your preferred timeframe and period
    4. Adjust RSI settings if needed
    5. Click "Fetch Data & Analyze" button
    6. Analyze the charts and download data as needed
    
    ### ⚠️ Note:
    - Data is fetched only when you click the button to respect Yahoo Finance API limits
    - UI remains intact after fetching data
    - All divergences are automatically marked with trendlines
    """)

# Footer
st.markdown("---")
st.markdown("**Developed for Advanced Algo Trading Analysis** | Data Source: Yahoo Finance")
