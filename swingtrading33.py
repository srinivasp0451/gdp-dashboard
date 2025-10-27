import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Algo Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# Session state initialization
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = None

# Title
st.title("🚀 Advanced Algo Trading Dashboard")

# Sidebar inputs
st.sidebar.header("⚙️ Configuration")

# Predefined tickers mapping
ticker_map = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "Custom Ticker": "CUSTOM"
}

selected_ticker = st.sidebar.selectbox("Select Asset", list(ticker_map.keys()))

if ticker_map[selected_ticker] == "CUSTOM":
    ticker_symbol = st.sidebar.text_input("Enter Custom Ticker (e.g., AAPL, RELIANCE.NS)", "AAPL")
else:
    ticker_symbol = ticker_map[selected_ticker]
    st.sidebar.info(f"Ticker: {ticker_symbol}")

# Timeframe selection
timeframes = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
selected_timeframe = st.sidebar.selectbox("Select Timeframe", timeframes, index=8)

# Period selection
periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
selected_period = st.sidebar.selectbox("Select Period", periods, index=5)

# Ratio chart options
st.sidebar.header("📊 Ratio Analysis")
enable_ratio = st.sidebar.checkbox("Enable Ratio Chart")
if enable_ratio:
    ratio_ticker = st.sidebar.text_input("Compare with Ticker", "^NSEI")
    ratio_bins = st.sidebar.slider("Number of Ratio Bins", 5, 20, 10)

# Fibonacci levels
enable_fib = st.sidebar.checkbox("Show Fibonacci Levels")

# Fetch button
fetch_button = st.sidebar.button("🔄 Fetch Data & Analyze", type="primary", use_container_width=True)

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes to avoid rate limits
def fetch_data(ticker, period, interval):
    """Fetch data from yfinance with caching"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Handle empty data
        if data is None or len(data) == 0:
            return None
        
        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            st.error(f"Missing required columns. Available: {list(data.columns)}")
            return None
        
        # Reset index to ensure datetime index
        data.index = pd.to_datetime(data.index)
        
        # Remove any duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Fill NaN with neutral value

def detect_divergence(price, rsi, window=20):
    """Detect RSI divergence"""
    divergences = []
    
    # Ensure we have enough data
    if len(price) < window + 5:
        return divergences
    
    for i in range(window, len(price)):
        price_slice = price.iloc[i-window:i]
        rsi_slice = rsi.iloc[i-window:i]
        
        # Skip if any NaN values
        if price_slice.isna().any() or rsi_slice.isna().any():
            continue
        
        current_price = price.iloc[i]
        current_rsi = rsi.iloc[i]
        
        # Bullish divergence: price lower low, RSI higher low
        if current_price <= price_slice.min() and current_rsi >= rsi_slice.min():
            if current_price < price_slice.min() or current_rsi > rsi_slice.min():
                divergences.append(('Bullish', i, current_price, current_rsi))
        
        # Bearish divergence: price higher high, RSI lower high
        if current_price >= price_slice.max() and current_rsi <= rsi_slice.max():
            if current_price > price_slice.max() or current_rsi < rsi_slice.max():
                divergences.append(('Bearish', i, current_price, current_rsi))
    
    return divergences

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    levels = {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.500 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100.0%': low
    }
    return levels

def create_candlestick_chart(df, ticker, fib_levels=None, divergences=None):
    """Create interactive candlestick chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker} Price Chart', 'Volume', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Fibonacci levels
    if fib_levels:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
        for idx, (level, price) in enumerate(fib_levels.items()):
            fig.add_hline(y=price, line_dash="dash", line_color=colors[idx % len(colors)],
                         annotation_text=level, row=1, col=1)
    
    # Divergence markers
    if divergences:
        for div_type, idx, price, rsi in divergences:
            color = 'green' if div_type == 'Bullish' else 'red'
            fig.add_trace(go.Scatter(
                x=[df.index[idx]],
                y=[price],
                mode='markers',
                marker=dict(size=12, color=color, symbol='triangle-up' if div_type == 'Bullish' else 'triangle-down'),
                name=f'{div_type} Divergence',
                showlegend=True
            ), row=1, col=1)
    
    # Volume
    colors_vol = ['#ef5350' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#26a69a' 
                  for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol),
                  row=2, col=1)
    
    # RSI
    rsi = calculate_rsi(df)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')),
                  row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought",
                  row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold",
                  row=3, col=1)
    
    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig, rsi

def create_returns_heatmap(df):
    """Create returns heatmap by day of week and month"""
    df_returns = df.copy()
    df_returns['Returns'] = df_returns['Close'].pct_change() * 100
    df_returns['DayOfWeek'] = df_returns.index.day_name()
    df_returns['Month'] = df_returns.index.month_name()
    
    # Remove NaN values
    df_returns = df_returns.dropna(subset=['Returns'])
    
    pivot_table = df_returns.pivot_table(
        values='Returns',
        index='DayOfWeek',
        columns='Month',
        aggfunc='mean'
    )
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    available_days = [d for d in day_order if d in pivot_table.index]
    if available_days:
        pivot_table = pivot_table.reindex(available_days)
    
    fig = px.imshow(
        pivot_table,
        labels=dict(x="Month", y="Day of Week", color="Avg Return %"),
        color_continuous_scale='RdYlGn',
        aspect="auto",
        title="Returns Heatmap: Day vs Month"
    )
    fig.update_layout(height=400)
    
    return fig, pivot_table

def analyze_price_movements(df):
    """Analyze daily price movements"""
    df_analysis = df.copy()
    df_analysis['Daily_Change'] = df_analysis['Close'].diff()
    df_analysis['Daily_Return_%'] = df_analysis['Close'].pct_change() * 100
    df_analysis['Direction'] = df_analysis['Daily_Change'].apply(lambda x: '📈 Up' if x > 0 else '📉 Down')
    
    return df_analysis[['Open', 'High', 'Low', 'Close', 'Daily_Change', 'Daily_Return_%', 'Direction']].dropna()

def create_ratio_chart(df1, df2, ticker1, ticker2, bins):
    """Create ratio analysis chart"""
    ratio = df1['Close'] / df2['Close']
    ratio_df = pd.DataFrame({
        'Ratio': ratio,
        'Price1': df1['Close'],
        'Price2': df2['Close']
    }).dropna()
    
    # Binning analysis
    ratio_df['Ratio_Bin'] = pd.cut(ratio_df['Ratio'], bins=bins)
    ratio_df['Future_Return'] = ratio_df['Price1'].pct_change(5).shift(-5) * 100
    
    bin_analysis = ratio_df.groupby('Ratio_Bin').agg({
        'Future_Return': ['mean', 'count'],
        'Ratio': 'mean'
    }).round(2)
    bin_analysis.columns = ['Avg_Future_Return_%', 'Count', 'Avg_Ratio']
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df['Ratio'], 
                            mode='lines', name=f'{ticker1}/{ticker2}'))
    fig.update_layout(
        title=f'Ratio Chart: {ticker1} vs {ticker2}',
        xaxis_title='Date',
        yaxis_title='Ratio',
        height=400,
        template='plotly_dark'
    )
    
    return fig, bin_analysis, ratio_df

def generate_insights(df, rsi, divergences, analysis_type="price"):
    """Generate human-readable insights"""
    insights = []
    
    if analysis_type == "price":
        latest_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
        change_pct = ((latest_close - prev_close) / prev_close) * 100
        
        trend = "upward" if change_pct > 0 else "downward"
        insights.append(f"Current price shows {trend} momentum with {abs(change_pct):.2f}% change.")
        
    if analysis_type == "rsi":
        latest_rsi = rsi.iloc[-1]
        if latest_rsi > 70:
            insights.append(f"RSI at {latest_rsi:.1f} indicates overbought conditions. Potential pullback expected.")
        elif latest_rsi < 30:
            insights.append(f"RSI at {latest_rsi:.1f} signals oversold territory. Bounce likely.")
        else:
            insights.append(f"RSI at {latest_rsi:.1f} shows neutral momentum. No extreme conditions.")
    
    if divergences and len(divergences) > 0:
        latest_div = divergences[-1]
        insights.append(f"{latest_div[0]} divergence detected, suggesting potential reversal.")
    
    return " ".join(insights)[:200]

# Main execution
if fetch_button:
    with st.spinner("Fetching data..."):
        df = fetch_data(ticker_symbol, selected_period, selected_timeframe)
        
        if df is not None and not df.empty:
            st.session_state.df = df
            st.session_state.ticker_symbol = ticker_symbol
            st.session_state.data_fetched = True
            st.success(f"✅ Data fetched successfully! {len(df)} records loaded.")
        else:
            st.error("❌ Failed to fetch data. Check ticker symbol and try again.")
            st.session_state.data_fetched = False

# Display results if data is fetched
if st.session_state.data_fetched and st.session_state.df is not None:
    df = st.session_state.df
    ticker = st.session_state.ticker_symbol
    
    # Tabs for organized display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Analysis", 
        "📈 RSI & Divergence", 
        "🔥 Returns Heatmap", 
        "⚖️ Ratio Analysis", 
        "📋 Raw Data"
    ])
    
    # Tab 1: Price Analysis
    with tab1:
        st.subheader("Price Movement Analysis")
        
        # Calculate Fibonacci if enabled
        fib_levels = None
        if enable_fib:
            high = df['High'].max()
            low = df['Low'].min()
            fib_levels = calculate_fibonacci_levels(high, low)
        
        # Calculate RSI and divergences
        rsi = calculate_rsi(df)
        divergences = detect_divergence(df['Close'], rsi)
        
        # Create candlestick chart
        fig_candle, rsi_series = create_candlestick_chart(df, ticker, fib_levels, divergences)
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Insights
        st.info(f"💡 **Insight**: {generate_insights(df, rsi, divergences, 'price')}")
        
        # Price movements table
        st.subheader("Daily Price Movements")
        movements_df = analyze_price_movements(df)
        
        # Color styling
        def color_negative_red(val):
            if isinstance(val, (int, float)):
                color = '#26a69a' if val > 0 else '#ef5350'
                return f'color: {color}'
            return ''
        
        styled_df = movements_df.tail(30).style.applymap(
            color_negative_red, 
            subset=['Daily_Change', 'Daily_Return_%']
        )
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        with col2:
            up_days = (movements_df['Daily_Change'] > 0).sum()
            st.metric("Up Days", up_days, f"{(up_days/len(movements_df)*100):.1f}%")
        with col3:
            down_days = (movements_df['Daily_Change'] < 0).sum()
            st.metric("Down Days", down_days, f"{(down_days/len(movements_df)*100):.1f}%")
        with col4:
            avg_return = movements_df['Daily_Return_%'].mean()
            st.metric("Avg Daily Return", f"{avg_return:.2f}%")
    
    # Tab 2: RSI & Divergence
    with tab2:
        st.subheader("RSI Analysis & Divergence Detection")
        
        # RSI distribution
        fig_rsi_dist = go.Figure()
        fig_rsi_dist.add_trace(go.Histogram(x=rsi, nbinsx=50, name='RSI Distribution'))
        fig_rsi_dist.update_layout(
            title="RSI Distribution",
            xaxis_title="RSI Value",
            yaxis_title="Frequency",
            height=300,
            template='plotly_dark'
        )
        st.plotly_chart(fig_rsi_dist, use_container_width=True)
        
        # Divergence summary
        if divergences:
            st.success(f"🔍 Found {len(divergences)} divergence signals")
            div_df = pd.DataFrame(divergences, columns=['Type', 'Index', 'Price', 'RSI'])
            div_df['Date'] = df.index[div_df['Index'].values].values
            st.dataframe(div_df[['Date', 'Type', 'Price', 'RSI']], use_container_width=True)
        else:
            st.warning("No significant divergences detected in the selected period.")
        
        st.info(f"💡 **Insight**: {generate_insights(df, rsi, divergences, 'rsi')}")
    
    # Tab 3: Returns Heatmap
    with tab3:
        st.subheader("Returns Heatmap Analysis")
        
        fig_heatmap, pivot_data = create_returns_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Insights from heatmap
        best_day = pivot_data.mean(axis=1).idxmax()
        worst_day = pivot_data.mean(axis=1).idxmin()
        best_month = pivot_data.mean(axis=0).idxmax()
        
        st.info(f"💡 **Insight**: Best performing day is {best_day}. Strongest month historically is {best_month}. "
                f"Consider timing entries on {worst_day} for better risk-reward.")
        
        st.dataframe(pivot_data.style.background_gradient(cmap='RdYlGn', axis=None), 
                     use_container_width=True)
    
    # Tab 4: Ratio Analysis
    with tab4:
        if enable_ratio:
            st.subheader(f"Ratio Analysis: {ticker} vs {ratio_ticker}")
            
            with st.spinner("Fetching comparison data..."):
                df_ratio = fetch_data(ratio_ticker, selected_period, selected_timeframe)
                
                if df_ratio is not None and not df_ratio.empty:
                    # Align dataframes
                    common_idx = df.index.intersection(df_ratio.index)
                    df_aligned = df.loc[common_idx]
                    df_ratio_aligned = df_ratio.loc[common_idx]
                    
                    fig_ratio, bin_analysis, ratio_df = create_ratio_chart(
                        df_aligned, df_ratio_aligned, ticker, ratio_ticker, ratio_bins
                    )
                    
                    st.plotly_chart(fig_ratio, use_container_width=True)
                    
                    st.subheader("Ratio Bin Analysis")
                    st.dataframe(
                        bin_analysis.style.background_gradient(
                            subset=['Avg_Future_Return_%'], 
                            cmap='RdYlGn'
                        ),
                        use_container_width=True
                    )
                    
                    # Insights
                    best_bin = bin_analysis['Avg_Future_Return_%'].idxmax()
                    worst_bin = bin_analysis['Avg_Future_Return_%'].idxmin()
                    st.info(f"💡 **Insight**: Historically, ratio range {best_bin} shows strongest future returns. "
                            f"Range {worst_bin} indicates potential weakness. Current ratio: {ratio_df['Ratio'].iloc[-1]:.4f}")
                else:
                    st.error("Failed to fetch comparison ticker data.")
        else:
            st.info("Enable 'Ratio Analysis' in the sidebar to view ratio charts.")
    
    # Tab 5: Raw Data
    with tab5:
        st.subheader("Raw OHLCV Data")
        st.dataframe(df.tail(500), use_container_width=True, height=600)
        
        # Download button
        csv = df.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"{ticker}_{selected_period}_{selected_timeframe}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Configure your parameters in the sidebar and click 'Fetch Data & Analyze' to begin.")
    
    # Display example usage
    st.markdown("""
    ### 🎯 Features:
    - **Multi-Asset Support**: NIFTY, Bank NIFTY, SENSEX, Crypto, Forex, Commodities
    - **Multiple Timeframes**: From 1-minute to yearly data
    - **Advanced Analysis**: RSI divergence, Fibonacci levels, ratio charts
    - **Visual Insights**: Candlestick charts, heatmaps, volume analysis
    - **Smart Caching**: Prevents API rate limit issues
    - **Interactive UI**: All components remain visible after data fetch
    
    ### 📊 Analysis Capabilities:
    1. **Price Analysis**: Daily movements, support/resistance, Fibonacci levels
    2. **RSI & Divergence**: Detect bullish/bearish signals
    3. **Returns Heatmap**: Identify best trading days and months
    4. **Ratio Analysis**: Compare assets and find historical patterns
    5. **Raw Data Export**: Download complete dataset
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tip**: Data is cached for 5 minutes to avoid rate limits.")
st.sidebar.markdown("**⚠️ Disclaimer**: For educational purposes only. Not financial advice.")
