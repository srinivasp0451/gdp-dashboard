import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import time
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
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'df_ratio' not in st.session_state:
    st.session_state.df_ratio = None

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

# Display last fetch time
if st.session_state.last_fetch_time:
    time_diff = (datetime.now() - st.session_state.last_fetch_time).seconds
    st.sidebar.warning(f"⏱️ Last fetch: {time_diff}s ago\n\nWait 60s between fetches to avoid rate limits!")

# Fetch button
fetch_button = st.sidebar.button("🔄 Fetch Data & Analyze", type="primary", use_container_width=True)

# Helper functions
def fetch_data_with_retry(ticker, period, interval, max_retries=3, delay=2):
    """Fetch data from yfinance with retry logic and rate limit handling"""
    for attempt in range(max_retries):
        try:
            # Add delay between attempts
            if attempt > 0:
                time.sleep(delay * attempt)
            
            data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
            
            # Handle empty data
            if data is None or len(data) == 0:
                if attempt < max_retries - 1:
                    continue
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
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying... Error: {str(e)}")
                continue
            else:
                st.error(f"Error fetching data after {max_retries} attempts: {str(e)}")
                return None
    
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
    """Detect RSI divergence and return points for drawing lines"""
    divergences = []
    divergence_lines = []
    
    # Ensure we have enough data
    if len(price) < window + 5:
        return divergences, divergence_lines
    
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
                prev_low_idx = i - window + price_slice.argmin()
                divergences.append(('Bullish', i, current_price, current_rsi))
                divergence_lines.append({
                    'type': 'Bullish',
                    'x0': price.index[prev_low_idx],
                    'x1': price.index[i],
                    'y0': price.iloc[prev_low_idx],
                    'y1': current_price,
                    'rsi_y0': rsi.iloc[prev_low_idx],
                    'rsi_y1': current_rsi
                })
        
        # Bearish divergence: price higher high, RSI lower high
        if current_price >= price_slice.max() and current_rsi <= rsi_slice.max():
            if current_price > price_slice.max() or current_rsi < rsi_slice.max():
                prev_high_idx = i - window + price_slice.argmax()
                divergences.append(('Bearish', i, current_price, current_rsi))
                divergence_lines.append({
                    'type': 'Bearish',
                    'x0': price.index[prev_high_idx],
                    'x1': price.index[i],
                    'y0': price.iloc[prev_high_idx],
                    'y1': current_price,
                    'rsi_y0': rsi.iloc[prev_high_idx],
                    'rsi_y1': current_rsi
                })
    
    return divergences, divergence_lines

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

def create_candlestick_chart(df, ticker, fib_levels=None, divergence_lines=None):
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
                         annotation_text=level, row=1, col=1, line_width=1)
    
    # Divergence lines on price chart
    if divergence_lines:
        for div_line in divergence_lines:
            color = 'lime' if div_line['type'] == 'Bullish' else 'red'
            fig.add_trace(go.Scatter(
                x=[div_line['x0'], div_line['x1']],
                y=[div_line['y0'], div_line['y1']],
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=8),
                name=f"{div_line['type']} Div",
                showlegend=False
            ), row=1, col=1)
    
    # Volume
    colors_vol = ['#ef5350' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#26a69a' 
                  for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol,
                         showlegend=False),
                  row=2, col=1)
    
    # RSI
    rsi = calculate_rsi(df)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple', width=2)),
                  row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought",
                  row=3, col=1, line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold",
                  row=3, col=1, line_width=1)
    
    # Add divergence lines on RSI chart
    if divergence_lines:
        for div_line in divergence_lines:
            color = 'lime' if div_line['type'] == 'Bullish' else 'red'
            fig.add_trace(go.Scatter(
                x=[div_line['x0'], div_line['x1']],
                y=[div_line['rsi_y0'], div_line['rsi_y1']],
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=8),
                name=f"{div_line['type']} Div RSI",
                showlegend=False
            ), row=3, col=1)
    
    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig, rsi

def create_multiple_heatmaps(df):
    """Create multiple return heatmaps: Day vs Month, Week vs Month, Month vs Year, Quarter vs Year"""
    df_returns = df.copy()
    df_returns['Returns'] = df_returns['Close'].pct_change() * 100
    df_returns = df_returns.dropna(subset=['Returns'])
    
    # Extract time components
    df_returns['DayOfWeek'] = df_returns.index.day_name()
    df_returns['DayOfMonth'] = df_returns.index.day
    df_returns['Week'] = df_returns.index.isocalendar().week
    df_returns['Month'] = df_returns.index.month_name()
    df_returns['MonthNum'] = df_returns.index.month
    df_returns['Quarter'] = df_returns.index.quarter
    df_returns['Year'] = df_returns.index.year
    
    heatmaps = {}
    
    # 1. Day of Week vs Month
    pivot1 = df_returns.pivot_table(values='Returns', index='DayOfWeek', columns='Month', aggfunc='mean')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    available_days = [d for d in day_order if d in pivot1.index]
    if available_days:
        pivot1 = pivot1.reindex(available_days)
    heatmaps['Day vs Month'] = pivot1
    
    # 2. Week vs Month
    pivot2 = df_returns.pivot_table(values='Returns', index='Week', columns='Month', aggfunc='mean')
    heatmaps['Week vs Month'] = pivot2
    
    # 3. Month vs Year
    pivot3 = df_returns.pivot_table(values='Returns', index='Month', columns='Year', aggfunc='mean')
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    available_months = [m for m in month_order if m in pivot3.index]
    if available_months:
        pivot3 = pivot3.reindex(available_months)
    heatmaps['Month vs Year'] = pivot3
    
    # 4. Quarter vs Year
    pivot4 = df_returns.pivot_table(values='Returns', index='Quarter', columns='Year', aggfunc='mean')
    heatmaps['Quarter vs Year'] = pivot4
    
    # Create figures
    figures = {}
    for name, pivot in heatmaps.items():
        if not pivot.empty:
            fig = px.imshow(
                pivot,
                labels=dict(x=name.split(' vs ')[1], y=name.split(' vs ')[0], color="Avg Return %"),
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                aspect="auto",
                title=f"Returns Heatmap: {name}",
                text_auto='.2f'  # Show values on heatmap
            )
            fig.update_layout(
                height=500,
                font=dict(size=14, color='white'),
                xaxis=dict(tickfont=dict(size=12)),
                yaxis=dict(tickfont=dict(size=12))
            )
            fig.update_traces(
                textfont=dict(size=10),
                hovertemplate='<b>%{y}</b><br>%{x}<br>Return: %{z:.2f}%<extra></extra>'
            )
            figures[name] = (fig, pivot)
    
    return figures

def analyze_price_movements(df):
    """Analyze daily price movements"""
    df_analysis = df.copy()
    df_analysis['Daily_Change'] = df_analysis['Close'].diff()
    df_analysis['Daily_Return_%'] = df_analysis['Close'].pct_change() * 100
    df_analysis['Direction'] = df_analysis['Daily_Change'].apply(lambda x: '📈 Up' if x > 0 else '📉 Down' if x < 0 else '➡️ Flat')
    df_analysis['Range'] = df_analysis['High'] - df_analysis['Low']
    df_analysis['Range_%'] = (df_analysis['Range'] / df_analysis['Low']) * 100
    
    return df_analysis[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Change', 'Daily_Return_%', 'Range', 'Range_%', 'Direction']].dropna()

def create_ratio_chart(df1, df2, ticker1, ticker2, bins):
    """Create ratio analysis chart"""
    # Align dataframes by index
    common_idx = df1.index.intersection(df2.index)
    df1_aligned = df1.loc[common_idx]
    df2_aligned = df2.loc[common_idx]
    
    ratio = df1_aligned['Close'] / df2_aligned['Close']
    ratio_df = pd.DataFrame({
        'Ratio': ratio,
        'Price1': df1_aligned['Close'],
        'Price2': df2_aligned['Close']
    }).dropna()
    
    if len(ratio_df) == 0:
        return None, None, None
    
    # Binning analysis
    ratio_df['Ratio_Bin'] = pd.cut(ratio_df['Ratio'], bins=bins)
    ratio_df['Future_Return'] = ratio_df['Price1'].pct_change(5).shift(-5) * 100
    
    bin_analysis = ratio_df.groupby('Ratio_Bin', observed=True).agg({
        'Future_Return': ['mean', 'count'],
        'Ratio': 'mean'
    }).round(2)
    bin_analysis.columns = ['Avg_Future_Return_%', 'Count', 'Avg_Ratio']
    
    # Plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Ratio Chart: {ticker1} / {ticker2}', 'Price Comparison'),
                        row_heights=[0.6, 0.4])
    
    # Ratio line
    fig.add_trace(go.Scatter(x=ratio_df.index, y=ratio_df['Ratio'], 
                            mode='lines', name=f'{ticker1}/{ticker2}',
                            line=dict(color='cyan', width=2)), row=1, col=1)
    
    # Mean line
    mean_ratio = ratio_df['Ratio'].mean()
    fig.add_hline(y=mean_ratio, line_dash="dash", line_color="yellow",
                 annotation_text=f"Mean: {mean_ratio:.4f}", row=1, col=1)
    
    # Price comparison
    fig.add_trace(go.Scatter(x=df1_aligned.index, y=df1_aligned['Close'],
                            mode='lines', name=ticker1, line=dict(color='#26a69a')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df2_aligned.index, y=df2_aligned['Close'],
                            mode='lines', name=ticker2, line=dict(color='#ef5350')), row=2, col=1)
    
    fig.update_layout(
        height=700,
        template='plotly_dark',
        hovermode='x unified'
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
        strength = "strong" if abs(change_pct) > 2 else "moderate" if abs(change_pct) > 0.5 else "weak"
        insights.append(f"Current price shows {strength} {trend} momentum with {abs(change_pct):.2f}% change.")
        
    if analysis_type == "rsi":
        latest_rsi = rsi.iloc[-1]
        if latest_rsi > 70:
            insights.append(f"RSI at {latest_rsi:.1f} indicates overbought conditions. Potential pullback expected.")
        elif latest_rsi < 30:
            insights.append(f"RSI at {latest_rsi:.1f} signals oversold territory. Bounce likely.")
        else:
            insights.append(f"RSI at {latest_rsi:.1f} shows neutral momentum.")
    
    if divergences and len(divergences) > 0:
        latest_div = divergences[-1]
        insights.append(f"{latest_div[0]} divergence detected, suggesting reversal.")
    
    return " ".join(insights)[:250]

# Main execution
if fetch_button:
    # Check rate limit (60 seconds between fetches)
    if st.session_state.last_fetch_time:
        time_since_last = (datetime.now() - st.session_state.last_fetch_time).seconds
        if time_since_last < 60:
            st.error(f"⚠️ Please wait {60 - time_since_last} seconds before fetching again to avoid API rate limits!")
            st.stop()
    
    with st.spinner("Fetching primary data..."):
        df = fetch_data_with_retry(ticker_symbol, selected_period, selected_timeframe)
        
        if df is not None and len(df) > 0:
            st.session_state.df = df
            st.session_state.ticker_symbol = ticker_symbol
            st.session_state.data_fetched = True
            st.session_state.last_fetch_time = datetime.now()
            st.success(f"✅ Data fetched successfully! {len(df)} records loaded.")
            
            # Fetch ratio data if enabled
            if enable_ratio:
                with st.spinner(f"Fetching comparison data for {ratio_ticker}..."):
                    time.sleep(1)  # Small delay between requests
                    df_ratio = fetch_data_with_retry(ratio_ticker, selected_period, selected_timeframe)
                    if df_ratio is not None and len(df_ratio) > 0:
                        st.session_state.df_ratio = df_ratio
                        st.success(f"✅ Comparison data fetched: {len(df_ratio)} records")
                    else:
                        st.error("❌ Failed to fetch comparison ticker data")
                        st.session_state.df_ratio = None
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
        divergences, divergence_lines = detect_divergence(df['Close'], rsi)
        
        # Create candlestick chart
        fig_candle, rsi_series = create_candlestick_chart(df, ticker, fib_levels, divergence_lines)
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Insights
        st.info(f"💡 **Insight**: {generate_insights(df, rsi, divergences, 'price')}")
        
        # Price movements table
        st.subheader("Complete Price Movements Data")
        movements_df = analyze_price_movements(df)
        
        # Color styling function
        def highlight_returns(row):
            if row['Daily_Return_%'] > 0:
                return ['background-color: rgba(38, 166, 154, 0.3)'] * len(row)
            elif row['Daily_Return_%'] < 0:
                return ['background-color: rgba(239, 83, 80, 0.3)'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = movements_df.style.apply(highlight_returns, axis=1).format({
            'Open': '{:.2f}',
            'High': '{:.2f}',
            'Low': '{:.2f}',
            'Close': '{:.2f}',
            'Volume': '{:,.0f}',
            'Daily_Change': '{:.2f}',
            'Daily_Return_%': '{:.2f}',
            'Range': '{:.2f}',
            'Range_%': '{:.2f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Summary statistics
        col1, col2, col3, col4, col5 = st.columns(5)
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
        with col5:
            max_range = movements_df['Range_%'].max()
            st.metric("Max Range %", f"{max_range:.2f}%")
    
    # Tab 2: RSI & Divergence
    with tab2:
        st.subheader("RSI Analysis & Divergence Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI distribution
            fig_rsi_dist = go.Figure()
            fig_rsi_dist.add_trace(go.Histogram(x=rsi, nbinsx=50, name='RSI Distribution',
                                                marker_color='purple'))
            fig_rsi_dist.update_layout(
                title="RSI Distribution",
                xaxis_title="RSI Value",
                yaxis_title="Frequency",
                height=400,
                template='plotly_dark'
            )
            st.plotly_chart(fig_rsi_dist, use_container_width=True)
        
        with col2:
            # RSI statistics
            rsi_stats = pd.DataFrame({
                'Metric': ['Current RSI', 'Average RSI', 'Max RSI', 'Min RSI', 
                          'Overbought (>70)', 'Oversold (<30)'],
                'Value': [
                    f"{rsi.iloc[-1]:.2f}",
                    f"{rsi.mean():.2f}",
                    f"{rsi.max():.2f}",
                    f"{rsi.min():.2f}",
                    f"{(rsi > 70).sum()} days",
                    f"{(rsi < 30).sum()} days"
                ]
            })
            st.dataframe(rsi_stats, use_container_width=True, height=400, hide_index=True)
        
        # Divergence summary
        st.subheader("Divergence Signals")
        if divergences:
            st.success(f"🔍 Found {len(divergences)} divergence signals")
            div_df = pd.DataFrame(divergences, columns=['Type', 'Index', 'Price', 'RSI'])
            div_df['Date'] = df.index[div_df['Index'].values].values
            div_df = div_df[['Date', 'Type', 'Price', 'RSI']].sort_values('Date', ascending=False)
            
            # Color code divergences
            def color_divergence(row):
                if row['Type'] == 'Bullish':
                    return ['background-color: rgba(0, 255, 0, 0.2)'] * len(row)
                else:
                    return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
            
            styled_div = div_df.style.apply(color_divergence, axis=1)
            st.dataframe(styled_div, use_container_width=True, height=400)
        else:
            st.warning("No significant divergences detected in the selected period.")
        
        st.info(f"💡 **Insight**: {generate_insights(df, rsi, divergences, 'rsi')}")
    
    # Tab 3: Returns Heatmap
    with tab3:
        st.subheader("Multi-Dimensional Returns Heatmap Analysis")
        
        heatmap_figures = create_multiple_heatmaps(df)
        
        if heatmap_figures:
            # Display all heatmaps
            for name, (fig, pivot_data) in heatmap_figures.items():
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                with st.expander(f"📊 View {name} Data Table"):
                    st.dataframe(
                        pivot_data.style.background_gradient(cmap='RdYlGn', axis=None).format('{:.2f}'),
                        use_container_width=True
                    )
                
                # Generate insights for each heatmap
                if not pivot_data.empty:
                    try:
                        best_row = pivot_data.mean(axis=1).idxmax()
                        worst_row = pivot_data.mean(axis=1).idxmin()
                        best_col = pivot_data.mean(axis=0).idxmax()
                        worst_col = pivot_data.mean(axis=0).idxmin()
                        
                        st.info(f"💡 **{name} Insight**: Best performing {name.split(' vs ')[0].lower()}: **{best_row}** "
                               f"(avg: {pivot_data.mean(axis=1)[best_row]:.2f}%). "
                               f"Strongest {name.split(' vs ')[1].lower()}: **{best_col}**. "
                               f"Avoid {worst_row} during {worst_col} for better risk-reward.")
                    except:
                        pass
                
                st.markdown("---")
        else:
            st.warning("Not enough data to generate heatmaps.")
    
    # Tab 4: Ratio Analysis
    with tab4:
        if enable_ratio:
            st.subheader(f"Ratio Analysis: {ticker} vs {ratio_ticker}")
            
            if st.session_state.df_ratio is not None and len(st.session_state.df_ratio) > 0:
                df_ratio_data = st.session_state.df_ratio
                
                fig_ratio, bin_analysis, ratio_df = create_ratio_chart(
                    df, df_ratio_data, ticker, ratio_ticker, ratio_bins
                )
                
                if fig_ratio is not None:
                    st.plotly_chart(fig_ratio, use_container_width=True)
                    
                    # Ratio statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Ratio", f"{ratio_df['Ratio'].iloc[-1]:.4f}")
                    with col2:
                        st.metric("Mean Ratio", f"{ratio_df['Ratio'].mean():.4f}")
                    with col3:
                        st.metric("Max Ratio", f"{ratio_df['Ratio'].max():.4f}")
                    with col4:
                        st.metric("Min Ratio", f"{ratio_df['Ratio'].min():.4f}")
                    
                    st.subheader("📊 Ratio Bin Analysis - Historical Performance")
                    st.markdown("*This table shows average future returns based on historical ratio ranges*")
                    
                    if bin_analysis is not None and not bin_analysis.empty:
                        # Enhanced styling for bin analysis
                        styled_bins = bin_analysis.style.background_gradient(
                            subset=['Avg_Future_Return_%'], 
                            cmap='RdYlGn'
                        ).format({
                            'Avg_Future_Return_%': '{:.2f}%',
                            'Count': '{:.0f}',
                            'Avg_Ratio': '{:.4f}'
                        })
                        st.dataframe(styled_bins, use_container_width=True)
                        
                        # Insights
                        best_bin = bin_analysis['Avg_Future_Return_%'].idxmax()
                        worst_bin = bin_analysis['Avg_Future_Return_%'].idxmin()
                        best_return = bin_analysis.loc[best_bin, 'Avg_Future_Return_%']
                        worst_return = bin_analysis.loc[worst_bin, 'Avg_Future_Return_%']
                        current_ratio = ratio_df['Ratio'].iloc[-1]
                        
                        st.info(f"💡 **Ratio Insight**: Historically, ratio range **{best_bin}** shows strongest future returns "
                               f"(avg: **{best_return:.2f}%**). Range **{worst_bin}** indicates potential weakness "
                               f"(avg: **{worst_return:.2f}%**). Current ratio: **{current_ratio:.4f}**. "
                               f"Use this to identify optimal entry/exit zones.")
                    
                    # Complete ratio data
                    st.subheader("Complete Ratio Time Series Data")
                    ratio_display = ratio_df.copy()
                    ratio_display['Date'] = ratio_display.index
                    ratio_display = ratio_display[['Date', 'Ratio', 'Price1', 'Price2', 'Ratio_Bin', 'Future_Return']]
                    ratio_display = ratio_display.sort_index(ascending=False)
                    
                    st.dataframe(
                        ratio_display.style.format({
                            'Ratio': '{:.4f}',
                            'Price1': '{:.2f}',
                            'Price2': '{:.2f}',
                            'Future_Return': '{:.2f}%'
                        }),
                        use_container_width=True,
                        height=600
                    )
                else:
                    st.error("Unable to create ratio chart. Not enough overlapping data.")
            else:
                st.error("Comparison data not available. Please fetch data again with ratio analysis enabled.")
        else:
            st.info("ℹ️ Enable 'Ratio Analysis' in the sidebar and fetch data to view ratio charts and bin analysis.")
            st.markdown("""
            ### 📊 What is Ratio Analysis?
            
            Ratio analysis helps you understand the **relative strength** between two assets:
            - Compare indices (e.g., NIFTY vs BANK NIFTY)
            - Compare stocks vs indices (e.g., RELIANCE vs NIFTY)
            - Compare assets vs currencies (e.g., NIFTY vs USD/INR)
            
            **Bin Analysis** shows you:
            - Which ratio ranges historically led to rises or falls
            - Optimal entry/exit zones based on historical patterns
            - Risk-reward scenarios for different ratio levels
            """)
    
    # Tab 5: Raw Data
    with tab5:
        st.subheader("Complete OHLCV Raw Data")
        
        # Display complete data (all rows)
        display_df = df.copy()
        display_df['Date'] = display_df.index
        display_df = display_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        display_df = display_df.sort_index(ascending=False)
        
        st.dataframe(
            display_df.style.format({
                'Open': '{:.2f}',
                'High': '{:.2f}',
                'Low': '{:.2f}',
                'Close': '{:.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True,
            height=600
        )
        
        # Data statistics
        st.subheader("📈 Data Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            stats_df = pd.DataFrame({
                'Metric': ['Total Records', 'Date Range', 'Highest Price', 'Lowest Price', 
                          'Avg Volume', 'Total Volume'],
                'Value': [
                    len(df),
                    f"{df.index.min().date()} to {df.index.max().date()}",
                    f"${df['High'].max():.2f}",
                    f"${df['Low'].min():.2f}",
                    f"{df['Volume'].mean():,.0f}",
                    f"{df['Volume'].sum():,.0f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            price_stats = df['Close'].describe()
            st.dataframe(
                price_stats.to_frame('Close Price').style.format('{:.2f}'),
                use_container_width=True
            )
        
        # Download button
        csv = df.to_csv()
        st.download_button(
            label="📥 Download Complete Data as CSV",
            data=csv,
            file_name=f"{ticker}_{selected_period}_{selected_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👆 Configure your parameters in the sidebar and click **'Fetch Data & Analyze'** to begin.")
    
    # Display example usage
    st.markdown("""
    ### 🎯 Key Features:
    
    #### 📊 **Multi-Asset Support**
    - Indices: NIFTY 50, Bank NIFTY, SENSEX
    - Crypto: BTC, ETH
    - Commodities: Gold, Silver
    - Forex: USD/INR, EUR/USD
    - Custom tickers: Any yfinance symbol
    
    #### ⏱️ **Flexible Timeframes**
    - Intraday: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
    - Daily/Weekly: 1d, 5d, 1wk
    - Monthly+: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y
    
    #### 📈 **Advanced Analysis**
    1. **Price Analysis Tab**: 
       - Professional candlestick charts with volume
       - Complete price movements table (all data visible)
       - Fibonacci retracement levels
       - Daily/Range statistics
    
    2. **RSI & Divergence Tab**:
       - RSI indicator with overbought/oversold zones
       - Automatic bullish/bearish divergence detection
       - **Divergence lines** drawn on both price and RSI charts
       - RSI distribution and statistics
    
    3. **Returns Heatmap Tab**:
       - **4 Different Heatmaps**:
         - Day of Week vs Month
         - Week vs Month
         - Month vs Year
         - Quarter vs Year
       - Color-coded returns with visible percentages
       - Identify best/worst trading periods
    
    4. **Ratio Analysis Tab**:
       - Compare any two assets
       - Historical bin analysis for pattern recognition
       - Entry/exit zone identification
       - Complete time series data
    
    5. **Raw Data Tab**:
       - Complete scrollable dataset (all records)
       - Comprehensive statistics
       - CSV download with timestamp
    
    #### 🛡️ **Protection Features**
    - **60-second rate limit** between fetches
    - Smart retry logic with exponential backoff
    - Session state management
    - Error handling for API failures
    - Visual countdown timer
    
    #### 🎨 **Enhanced UI**
    - All components persist after data fetch
    - Color-coded tables (green=profit, red=loss)
    - Dark theme with professional visuals
    - Readable fonts and proper spacing
    - Interactive plotly charts
    - Complete data visibility (no truncation)
    
    ### 🚀 Getting Started:
    1. Select your asset or enter custom ticker
    2. Choose timeframe and period
    3. Enable optional features (Ratio, Fibonacci)
    4. Click **"Fetch Data & Analyze"**
    5. Explore all 5 tabs for comprehensive analysis
    
    ### ⚠️ Important Notes:
    - Wait **60 seconds** between fetches to avoid rate limits
    - For ratio analysis, both tickers must have overlapping dates
    - Intraday data (1m-60m) limited to last 7-60 days by yfinance
    - All analysis is for **educational purposes only**
    
    ### 💡 Pro Tips:
    - Use **1d timeframe with 1y-5y period** for best divergence detection
    - Enable **Fibonacci levels** to identify support/resistance
    - Check **heatmaps** to optimize entry timing
    - Use **ratio analysis** for pairs trading strategies
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Tips")
st.sidebar.info("""
**Rate Limit Protection:**
- 60s minimum between fetches
- Automatic retry on failures
- Timer shows last fetch time

**Best Practices:**
- Use daily data for divergence
- Enable Fibonacci for S/R levels
- Check all 4 heatmap types
- Compare related assets in ratio
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ Disclaimer**: Educational tool only. Not financial advice. Trade at your own risk.")
