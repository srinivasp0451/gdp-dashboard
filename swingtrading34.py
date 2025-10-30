import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(page_title="Algo Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'plotly_dark'

# Predefined tickers with descriptions
PREDEFINED_TICKERS = {
    "Indian Indices": {
        "^NSEI": "Nifty 50",
        "^NSEBANK": "Bank Nifty",
        "^BSESN": "Sensex",
        "^NSEMDCP50": "Nifty Midcap 50"
    },
    "Cryptocurrencies": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "SOL-USD": "Solana",
        "BNB-USD": "Binance Coin",
        "XRP-USD": "Ripple"
    },
    "Commodities (USD)": {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "HG=F": "Copper",
        "CL=F": "Crude Oil"
    },
    "Forex": {
        "INR=X": "USD/INR",
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "JPYUSD=X": "JPY/USD",
        "AUDUSD=X": "AUD/USD"
    },
    "Indian Stocks": {
        "INFY.NS": "Infosys",
        "TCS.NS": "TCS",
        "RELIANCE.NS": "Reliance",
        "HDFCBANK.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "SBIN.NS": "SBI",
        "WIPRO.NS": "Wipro",
        "ITC.NS": "ITC",
        "AXISBANK.NS": "Axis Bank",
        "BHARTIARTL.NS": "Bharti Airtel"
    }
}

# Functions
def flatten_multiindex_dataframe(df):
    """Flatten multi-index dataframe from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        # For multi-ticker downloads, take the first ticker
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to make date a column, then set it back
    df = df.reset_index()
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    elif 'Datetime' in df.columns:
        df.set_index('Datetime', inplace=True)
    
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def find_divergences(prices, rsi, window=5):
    """Find RSI divergences"""
    divergences = []
    
    if len(prices) < window * 2:
        return divergences
    
    for i in range(window, len(prices) - window):
        try:
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_min = prices.iloc[i-window:i+window].min()
            if prices.iloc[i] == price_min and prices.iloc[i] < prices.iloc[i-window]:
                if rsi.iloc[i] > rsi.iloc[i-window]:
                    divergences.append(('Bullish', prices.index[i], prices.iloc[i]))
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            price_max = prices.iloc[i-window:i+window].max()
            if prices.iloc[i] == price_max and prices.iloc[i] > prices.iloc[i-window]:
                if rsi.iloc[i] < rsi.iloc[i-window]:
                    divergences.append(('Bearish', prices.index[i], prices.iloc[i]))
        except:
            continue
    
    return divergences

def calculate_fibonacci_levels(df):
    """Calculate Fibonacci retracement levels"""
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price
    
    levels = {
        '0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '78.6%': max_price - 0.786 * diff,
        '100%': min_price
    }
    return levels

def create_candlestick_chart(df, ticker_name, show_line=False, theme='plotly_dark'):
    """Create interactive candlestick chart"""
    fig = go.Figure()
    
    if show_line:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00d4ff', width=2)
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker_name,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff3366'
        ))
    
    fig.update_layout(
        title=f'{ticker_name} Price Chart',
        yaxis_title='Price',
        xaxis_title='Date',
        template=theme,
        height=600,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_rsi_chart(df, rsi, divergences, theme='plotly_dark'):
    """Create RSI chart with divergences"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=rsi,
        mode='lines',
        name='RSI',
        line=dict(color='#ffa500', width=2)
    ))
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
    
    # Add divergence markers
    for div_type, date, price in divergences:
        color = 'green' if div_type == 'Bullish' else 'red'
        try:
            rsi_value = rsi.loc[date]
            fig.add_trace(go.Scatter(
                x=[date],
                y=[rsi_value],
                mode='markers',
                name=f'{div_type} Divergence',
                marker=dict(size=15, color=color, symbol='star'),
                showlegend=True
            ))
        except:
            continue
    
    fig.update_layout(
        title='RSI Indicator with Divergences',
        yaxis_title='RSI Value',
        xaxis_title='Date',
        template=theme,
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_fibonacci_chart(df, fib_levels, ticker_name, theme='plotly_dark'):
    """Create price chart with Fibonacci levels"""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker_name,
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff3366'
    ))
    
    colors = ['#ff0000', '#ff8800', '#ffff00', '#00ff00', '#00ffff', '#0088ff', '#0000ff']
    for (level, price), color in zip(fib_levels.items(), colors):
        fig.add_hline(
            y=price,
            line_dash="dash",
            line_color=color,
            annotation_text=f"Fib {level}: {price:.2f}",
            annotation_position="right"
        )
    
    fig.update_layout(
        title=f'{ticker_name} with Fibonacci Retracement Levels',
        yaxis_title='Price',
        xaxis_title='Date',
        template=theme,
        height=600,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_returns_heatmap(df, heatmap_type='monthly_yearly', theme='plotly_dark'):
    """Create returns heatmap based on type"""
    df_copy = df.copy()
    df_copy['Returns'] = df_copy['Close'].pct_change() * 100
    
    if heatmap_type == 'monthly_yearly':
        df_copy['Year'] = df_copy.index.year
        df_copy['Month'] = df_copy.index.month
        pivot_table = df_copy.pivot_table(values='Returns', index='Month', columns='Year', aggfunc='sum')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.index = [month_names[i-1] if i <= 12 else str(i) for i in pivot_table.index]
        title = 'Monthly Returns by Year (%)'
        x_label, y_label = 'Year', 'Month'
        
    elif heatmap_type == 'daily_monthly':
        df_copy['Month'] = df_copy.index.to_period('M').astype(str)
        df_copy['Day'] = df_copy.index.day
        pivot_table = df_copy.pivot_table(values='Returns', index='Day', columns='Month', aggfunc='sum')
        title = 'Daily Returns by Month (%)'
        x_label, y_label = 'Month', 'Day'
        
    elif heatmap_type == 'quarterly_yearly':
        df_copy['Year'] = df_copy.index.year
        df_copy['Quarter'] = df_copy.index.quarter
        pivot_table = df_copy.pivot_table(values='Returns', index='Quarter', columns='Year', aggfunc='sum')
        pivot_table.index = [f'Q{i}' for i in pivot_table.index]
        title = 'Quarterly Returns by Year (%)'
        x_label, y_label = 'Year', 'Quarter'
        
    elif heatmap_type == 'weekly_yearly':
        df_copy['Year'] = df_copy.index.year
        df_copy['Week'] = df_copy.index.isocalendar().week
        pivot_table = df_copy.pivot_table(values='Returns', index='Week', columns='Year', aggfunc='sum')
        title = 'Weekly Returns by Year (%)'
        x_label, y_label = 'Year', 'Week'
    
    # Calculate proper aspect ratio for square cells
    n_rows, n_cols = pivot_table.shape
    cell_size = 40
    height = max(400, n_rows * cell_size)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn',
        text=np.round(pivot_table.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Returns %"),
        xgap=3,
        ygap=3
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=theme,
        height=height,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig, pivot_table

def analyze_ratio(df1, df2, ticker1_name, ticker2_name, bins=20, theme='plotly_dark'):
    """Analyze ratio between two tickers"""
    # Align dataframes
    combined = pd.DataFrame({
        'Price1': df1['Close'],
        'Price2': df2['Close']
    }).dropna()
    
    combined['Ratio'] = combined['Price1'] / combined['Price2']
    combined['Change'] = combined['Ratio'].diff()
    combined['Direction'] = combined['Change'].apply(lambda x: 'Up' if x > 0 else 'Down' if x < 0 else 'Neutral')
    
    # Create bins
    combined['Ratio_Bin'] = pd.cut(combined['Ratio'], bins=bins)
    
    # Analyze each bin
    bin_analysis = combined.groupby('Ratio_Bin', observed=True).agg({
        'Direction': lambda x: (x == 'Up').sum() / len(x) * 100 if len(x) > 0 else 0,
        'Ratio': ['count', 'mean'],
        'Price1': 'mean',
        'Price2': 'mean'
    })
    
    bin_analysis.columns = ['Up_Percentage', 'Count', 'Avg_Ratio', 'Avg_Price1', 'Avg_Price2']
    bin_analysis['Down_Percentage'] = 100 - bin_analysis['Up_Percentage']
    bin_analysis['Bin_Range'] = bin_analysis.index.astype(str)
    bin_analysis = bin_analysis.reset_index(drop=True)
    
    # Create ratio chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined.index,
        y=combined['Ratio'],
        mode='lines',
        name='Ratio',
        line=dict(color='#00d4ff', width=2),
        hovertemplate='Date: %{x}<br>Ratio: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{ticker1_name} / {ticker2_name} Ratio Chart',
        yaxis_title='Ratio',
        xaxis_title='Date',
        template=theme,
        height=500,
        hovermode='x unified'
    )
    
    # Comparison table
    comparison_df = combined[['Price1', 'Price2', 'Ratio']].copy()
    comparison_df.columns = [f'{ticker1_name} Close', f'{ticker2_name} Close', 'Ratio']
    comparison_df[f'{ticker1_name} Change %'] = comparison_df[f'{ticker1_name} Close'].pct_change() * 100
    comparison_df[f'{ticker2_name} Change %'] = comparison_df[f'{ticker2_name} Close'].pct_change() * 100
    comparison_df['Ratio Change %'] = comparison_df['Ratio'].pct_change() * 100
    
    return fig, bin_analysis, comparison_df

def generate_insights(df, rsi, divergences, fib_levels):
    """Generate human-readable insights"""
    insights = []
    
    # Price trend
    if len(df) >= 5:
        recent_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
        trend = "uptrend" if recent_change > 0 else "downtrend"
        insights.append(f"Price is in {trend} with {abs(recent_change):.2f}% change over last 5 periods.")
    
    # RSI analysis
    current_rsi = rsi.iloc[-1]
    if current_rsi > 70:
        insights.append(f"RSI at {current_rsi:.1f} indicates overbought conditions - potential reversal.")
    elif current_rsi < 30:
        insights.append(f"RSI at {current_rsi:.1f} shows oversold conditions - potential bounce expected.")
    else:
        insights.append(f"RSI at {current_rsi:.1f} is neutral - no extreme conditions.")
    
    # Divergences
    if divergences:
        last_div = divergences[-1]
        insights.append(f"Latest {last_div[0]} divergence detected - {last_div[0].lower()} signal confirmed.")
    
    return " ".join(insights)

def calculate_daily_stats(df):
    """Calculate daily statistics"""
    stats_df = df.copy()
    stats_df['Daily_Change'] = stats_df['Close'].diff()
    stats_df['Daily_Return_%'] = stats_df['Close'].pct_change() * 100
    stats_df['Points_Moved'] = stats_df['High'] - stats_df['Low']
    stats_df['Range_%'] = (stats_df['Points_Moved'] / stats_df['Open']) * 100
    
    return stats_df[['Open', 'High', 'Low', 'Close', 'Volume', 
                     'Daily_Change', 'Daily_Return_%', 'Points_Moved', 'Range_%']].dropna()

# Sidebar
st.sidebar.title("🚀 Algo Trading Dashboard")
st.sidebar.markdown("---")

# Theme selector
st.sidebar.subheader("🎨 Theme")
theme_choice = st.sidebar.radio("Select Theme", ["Dark", "Light"], horizontal=True)
st.session_state.theme = 'plotly_dark' if theme_choice == 'Dark' else 'plotly_white'

# Apply custom CSS based on theme
if theme_choice == 'Dark':
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Ticker selection
st.sidebar.subheader("📊 Select Instrument")
category = st.sidebar.selectbox("Category", list(PREDEFINED_TICKERS.keys()) + ["Custom Ticker"])

if category == "Custom Ticker":
    ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
else:
    ticker_options = PREDEFINED_TICKERS[category]
    selected = st.sidebar.selectbox("Ticker", list(ticker_options.keys()), 
                                    format_func=lambda x: f"{ticker_options[x]} ({x})")
    ticker = selected

# Timeframe and Period
st.sidebar.subheader("⏰ Time Settings")
interval = st.sidebar.selectbox(
    "Interval",
    ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "5d", "1wk", "1mo", "3mo"]
)

period = st.sidebar.selectbox(
    "Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
)

# Chart options
st.sidebar.subheader("📈 Chart Options")
show_line_chart = st.sidebar.checkbox("Show Line Chart", value=False)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)

# Heatmap options
st.sidebar.subheader("🔥 Heatmap Type")
heatmap_type = st.sidebar.selectbox(
    "Select Heatmap",
    ["monthly_yearly", "daily_monthly", "quarterly_yearly", "weekly_yearly"],
    format_func=lambda x: {
        'monthly_yearly': 'Month vs Year',
        'daily_monthly': 'Day vs Month',
        'quarterly_yearly': 'Quarter vs Year',
        'weekly_yearly': 'Week vs Year'
    }[x]
)

# Ratio Analysis
st.sidebar.subheader("📊 Ratio Analysis")
enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis")
if enable_ratio:
    ratio_category = st.sidebar.selectbox("Compare with", list(PREDEFINED_TICKERS.keys()) + ["Custom"])
    if ratio_category == "Custom":
        ratio_ticker = st.sidebar.text_input("Ratio Ticker", "^NSEI")
    else:
        ratio_options = PREDEFINED_TICKERS[ratio_category]
        ratio_selected = st.sidebar.selectbox("Ratio Ticker", list(ratio_options.keys()),
                                              format_func=lambda x: f"{ratio_options[x]} ({x})")
        ratio_ticker = ratio_selected
    ratio_bins = st.sidebar.slider("Number of Bins", 10, 50, 20)

# Fetch button
fetch_button = st.sidebar.button("🔄 Fetch Data & Analyze", type="primary", use_container_width=True)

# Main content
st.title(f"📈 Advanced Algo Trading Analysis")
st.markdown("---")

if fetch_button:
    with st.spinner("Fetching data from yfinance..."):
        try:
            # Fetch main ticker data
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                st.error(f"No data found for ticker: {ticker}")
            else:
                # Flatten multi-index dataframe
                data = flatten_multiindex_dataframe(data)
                
                st.session_state.df = data
                st.session_state.ticker_symbol = ticker
                st.session_state.data_fetched = True
                
                # Fetch ratio ticker if enabled
                if enable_ratio:
                    ratio_data = yf.download(ratio_ticker, period=period, interval=interval, progress=False)
                    ratio_data = flatten_multiindex_dataframe(ratio_data)
                    st.session_state.ratio_df = ratio_data
                    st.session_state.ratio_ticker = ratio_ticker
                
                st.success(f"✅ Data fetched successfully for {ticker}!")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Display analysis if data is fetched
if st.session_state.data_fetched and st.session_state.df is not None:
    df = st.session_state.df
    ticker_name = st.session_state.ticker_symbol
    theme = st.session_state.theme
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
    divergences = find_divergences(df['Close'], df['RSI'])
    fib_levels = calculate_fibonacci_levels(df)
    daily_stats = calculate_daily_stats(df)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        change_val = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) >= 2 else 0
        st.metric("Current Price", f"{df['Close'].iloc[-1]:.2f}", f"{change_val:.2f}%")
    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            daily_stats.to_excel(writer, sheet_name='Data')
        st.download_button(
            label="📥 Download Excel",
            data=buffer.getvalue(),
            file_name=f'{ticker_name}_data.xlsx',
            mime='application/vnd.ms-excel'
        )
    
    # Ratio Analysis
    if enable_ratio and 'ratio_df' in st.session_state:
        st.markdown("---")
        st.subheader(f"⚖️ Ratio Analysis: {ticker_name} / {st.session_state.ratio_ticker}")
        
        ratio_fig, bin_analysis, comparison_df = analyze_ratio(
            df, 
            st.session_state.ratio_df, 
            ticker_name, 
            st.session_state.ratio_ticker,
            ratio_bins,
            theme
        )
        
        st.plotly_chart(ratio_fig, use_container_width=True)
        
        # Ratio insights
        current_ratio = comparison_df['Ratio'].iloc[-1]
        ratio_mean = comparison_df['Ratio'].mean()
        ratio_std = comparison_df['Ratio'].std()
        ratio_insight = f"Current ratio: {current_ratio:.4f} vs historical mean: {ratio_mean:.4f} (±{ratio_std:.4f}). "
        
        if current_ratio > ratio_mean + ratio_std:
            ratio_insight += f"{ticker_name} is relatively expensive compared to {st.session_state.ratio_ticker}. Potential mean reversion downward."
        elif current_ratio < ratio_mean - ratio_std:
            ratio_insight += f"{ticker_name} is relatively cheap compared to {st.session_state.ratio_ticker}. Potential mean reversion upward."
        else:
            ratio_insight += "Ratio is near historical average. No extreme divergence detected."
        st.info(f"**Insight:** {ratio_insight}")
        
        st.subheader("📊 Ratio Bin Analysis")
        st.markdown("**This table shows historical behavior in different ratio ranges:**")
        
        # Format bin analysis for better readability
        bin_display = bin_analysis[['Bin_Range', 'Count', 'Avg_Ratio', 'Up_Percentage', 'Down_Percentage', 'Avg_Price1', 'Avg_Price2']].copy()
        bin_display.columns = ['Ratio Range', 'Occurrences', 'Avg Ratio', 'Up %', 'Down %', f'Avg {ticker_name}', f'Avg {st.session_state.ratio_ticker}']
        
        st.dataframe(
            bin_display.style.background_gradient(cmap='RdYlGn', subset=['Up %']).format({
                'Avg Ratio': '{:.4f}',
                'Up %': '{:.1f}%',
                'Down %': '{:.1f}%',
                f'Avg {ticker_name}': '{:.2f}',
                f'Avg {st.session_state.ratio_ticker}': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        st.subheader("📋 Detailed Price Comparison")
        st.markdown(f"**Comparison of {ticker_name} vs {st.session_state.ratio_ticker} with ratio calculations:**")
        
        # Display last 50 rows of comparison
        comparison_display = comparison_df.tail(50).style.format({
            f'{ticker_name} Close': '{:.2f}',
            f'{st.session_state.ratio_ticker} Close': '{:.2f}',
            'Ratio': '{:.4f}',
            f'{ticker_name} Change %': '{:.2f}%',
            f'{st.session_state.ratio_ticker} Change %': '{:.2f}%',
            'Ratio Change %': '{:.2f}%'
        }).applymap(
            color_negative_red,
            subset=[f'{ticker_name} Change %', f'{st.session_state.ratio_ticker} Change %', 'Ratio Change %']
        )
        
        st.dataframe(comparison_display, use_container_width=True, height=400)
        
        # Ratio comparison export
        col1, col2 = st.columns(2)
        with col1:
            ratio_csv = comparison_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Download Ratio CSV",
                data=ratio_csv,
                file_name=f'{ticker_name}_{st.session_state.ratio_ticker}_ratio.csv',
                mime='text/csv',
            )
        
        with col2:
            ratio_buffer = io.BytesIO()
            with pd.ExcelWriter(ratio_buffer, engine='xlsxwriter') as writer:
                comparison_df.to_excel(writer, sheet_name='Ratio Analysis')
                bin_analysis.to_excel(writer, sheet_name='Bin Analysis')
            st.download_button(
                label="📥 Download Ratio Excel",
                data=ratio_buffer.getvalue(),
                file_name=f'{ticker_name}_{st.session_state.ratio_ticker}_ratio.xlsx',
                mime='application/vnd.ms-excel'
            )
        
        # Ratio prediction based on bins
        st.markdown("**Historical Pattern Analysis:**")
        
        # Find which bin current ratio falls into
        current_bin = None
        for idx, row in bin_display.iterrows():
            try:
                range_str = row['Ratio Range']
                # Parse the range string
                if '(' in range_str and ')' in range_str:
                    range_clean = range_str.strip('()[]')
                    parts = range_clean.split(',')
                    if len(parts) == 2:
                        low = float(parts[0].strip())
                        high = float(parts[1].strip())
                        if low <= current_ratio <= high:
                            current_bin = row
                            break
            except:
                continue
        
        if current_bin is not None:
            up_pct = current_bin['Up %']
            down_pct = current_bin['Down %']
            occurrences = current_bin['Occurrences']
            
            if up_pct > 60:
                prediction_text = f"📈 **Bullish Signal**: Historically, when ratio was in this range, price moved up {up_pct:.1f}% of the time ({int(occurrences)} occurrences)."
                st.success(prediction_text)
            elif down_pct > 60:
                prediction_text = f"📉 **Bearish Signal**: Historically, when ratio was in this range, price moved down {down_pct:.1f}% of the time ({int(occurrences)} occurrences)."
                st.error(prediction_text)
            else:
                prediction_text = f"⚖️ **Neutral Signal**: Historical data shows mixed signals - Up: {up_pct:.1f}%, Down: {down_pct:.1f}% ({int(occurrences)} occurrences)."
                st.warning(prediction_text)
    
    # Overall Market View
    st.markdown("---")
    st.subheader("🎯 Overall Market View")
    
    # Calculate confluence
    signals = []
    
    # RSI signal
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi > 70:
        signals.append(("RSI", "Bearish", "Overbought", f"RSI: {current_rsi:.1f}"))
    elif current_rsi < 30:
        signals.append(("RSI", "Bullish", "Oversold", f"RSI: {current_rsi:.1f}"))
    else:
        signals.append(("RSI", "Neutral", "Mid-range", f"RSI: {current_rsi:.1f}"))
    
    # Divergence signal
    if divergences:
        last_div = divergences[-1][0]
        signals.append(("Divergence", last_div, f"{last_div} divergence", f"Latest: {last_div}"))
    else:
        signals.append(("Divergence", "Neutral", "No divergence", "None detected"))
    
    # Price vs Fibonacci
    current_price = df['Close'].iloc[-1]
    if current_price > fib_levels['50%']:
        fib_distance = ((current_price - fib_levels['50%']) / fib_levels['50%'] * 100)
        signals.append(("Fibonacci", "Bullish", "Above 50% level", f"+{fib_distance:.2f}% from 50%"))
    else:
        fib_distance = ((fib_levels['50%'] - current_price) / fib_levels['50%'] * 100)
        signals.append(("Fibonacci", "Bearish", "Below 50% level", f"-{fib_distance:.2f}% from 50%"))
    
    # Trend
    if len(df) >= 5:
        recent_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100
        if recent_change > 2:
            signals.append(("Short-term Trend", "Bullish", "Uptrend", f"+{recent_change:.2f}%"))
        elif recent_change < -2:
            signals.append(("Short-term Trend", "Bearish", "Downtrend", f"{recent_change:.2f}%"))
        else:
            signals.append(("Short-term Trend", "Neutral", "Sideways", f"{recent_change:.2f}%"))
    
    # Volume analysis
    avg_volume = df['Volume'].tail(20).mean()
    current_volume = df['Volume'].iloc[-1]
    volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1
    
    if volume_ratio > 1.5:
        signals.append(("Volume", "Strong", "Above average", f"{volume_ratio:.2f}x avg"))
    elif volume_ratio < 0.5:
        signals.append(("Volume", "Weak", "Below average", f"{volume_ratio:.2f}x avg"))
    else:
        signals.append(("Volume", "Normal", "Average range", f"{volume_ratio:.2f}x avg"))
    
    # Volatility (using ATR-like calculation)
    df['TR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], 
                                                                  abs(x['High'] - x['Close']), 
                                                                  abs(x['Low'] - x['Close'])), axis=1)
    current_volatility = df['TR'].tail(14).mean()
    avg_volatility = df['TR'].mean()
    
    if current_volatility > avg_volatility * 1.5:
        signals.append(("Volatility", "High", "Elevated", f"{(current_volatility/avg_volatility):.2f}x avg"))
    elif current_volatility < avg_volatility * 0.5:
        signals.append(("Volatility", "Low", "Compressed", f"{(current_volatility/avg_volatility):.2f}x avg"))
    else:
        signals.append(("Volatility", "Normal", "Average", f"{(current_volatility/avg_volatility):.2f}x avg"))
    
    # Display signals
    signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Status', 'Details'])
    
    # Color code the signals
    def color_signals(val):
        if val == 'Bullish':
            return 'background-color: #00ff8844; color: #00ff88'
        elif val == 'Bearish':
            return 'background-color: #ff336644; color: #ff3366'
        elif val == 'Neutral':
            return 'background-color: #ffaa0044; color: #ffaa00'
        elif val == 'Strong':
            return 'background-color: #00ff8844; color: #00ff88'
        elif val == 'Weak':
            return 'background-color: #ff336644; color: #ff3366'
        return ''
    
    styled_signals = signal_df.style.applymap(color_signals, subset=['Signal'])
    st.dataframe(styled_signals, use_container_width=True, hide_index=True)
    
    # Final verdict
    bullish_signals = sum(1 for s in signals if s[1] == "Bullish" or s[1] == "Strong")
    bearish_signals = sum(1 for s in signals if s[1] == "Bearish" or s[1] == "Weak")
    neutral_signals = sum(1 for s in signals if s[1] == "Neutral" or s[1] == "Normal")
    
    st.markdown("### 🎲 Final Market Prediction")
    
    total_signals = len(signals)
    bullish_pct = (bullish_signals / total_signals) * 100
    bearish_pct = (bearish_signals / total_signals) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bullish Signals", f"{bullish_signals}/{total_signals}", f"{bullish_pct:.0f}%")
    with col2:
        st.metric("Bearish Signals", f"{bearish_signals}/{total_signals}", f"{bearish_pct:.0f}%")
    with col3:
        st.metric("Neutral Signals", f"{neutral_signals}/{total_signals}", f"{(neutral_signals/total_signals)*100:.0f}%")
    
    if bullish_signals > bearish_signals + 1:
        st.success(f"✅ **STRONG BULLISH CONFLUENCE** - {bullish_signals} bullish signals suggest upward momentum. Consider long positions with proper risk management.")
    elif bullish_signals > bearish_signals:
        st.success(f"✅ **BULLISH BIAS** - {bullish_signals} bullish vs {bearish_signals} bearish signals. Cautiously bullish stance recommended.")
    elif bearish_signals > bullish_signals + 1:
        st.error(f"⚠️ **STRONG BEARISH CONFLUENCE** - {bearish_signals} bearish signals indicate downward pressure. Consider short positions or protective stops.")
    elif bearish_signals > bullish_signals:
        st.error(f"⚠️ **BEARISH BIAS** - {bearish_signals} bearish vs {bullish_signals} bullish signals. Cautiously bearish stance recommended.")
    else:
        st.warning(f"⚖️ **NEUTRAL/MIXED SIGNALS** - Equal signals ({bullish_signals} bullish, {bearish_signals} bearish). Wait for clearer direction or trade range-bound.")
    
    # Historical performance prediction
    st.markdown("### 📊 Historical Pattern-Based Prediction")
    
    # Calculate win rate
    positive_days = (daily_stats['Daily_Return_%'] > 0).sum()
    total_days = len(daily_stats)
    win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0
    
    avg_gain = daily_stats[daily_stats['Daily_Return_%'] > 0]['Daily_Return_%'].mean()
    avg_loss = daily_stats[daily_stats['Daily_Return_%'] < 0]['Daily_Return_%'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{positive_days}/{total_days} days")
    with col2:
        st.metric("Avg Gain", f"{avg_gain:.2f}%", "On up days")
    with col3:
        st.metric("Avg Loss", f"{avg_loss:.2f}%", "On down days")
    
    # Risk-Reward ratio
    if not pd.isna(avg_loss) and avg_loss != 0:
        risk_reward = abs(avg_gain / avg_loss)
        st.info(f"**Risk-Reward Ratio:** {risk_reward:.2f}:1 - Historical average gain is {risk_reward:.2f}x the average loss.")
    
    # Expected value
    if not pd.isna(avg_gain) and not pd.isna(avg_loss):
        expected_value = (win_rate/100 * avg_gain) + ((100-win_rate)/100 * avg_loss)
        if expected_value > 0:
            st.success(f"**Positive Expected Value:** +{expected_value:.2f}% per period. Historically profitable pattern.")
        else:
            st.warning(f"**Negative Expected Value:** {expected_value:.2f}% per period. Exercise caution.")
    
else:
    st.info("👈 Please select your parameters and click 'Fetch Data & Analyze' to begin.")
    
    # Display sample tickers
    st.subheader("Available Instruments")
    for category, tickers in PREDEFINED_TICKERS.items():
        with st.expander(f"📁 {category}"):
            ticker_list = [f"**{name}** (`{symbol}`)" for symbol, name in tickers.items()]
            st.markdown(" • " + "\n • ".join(ticker_list))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ This tool is for educational purposes only. Not financial advice. Always do your own research.</p>
    <p>Data provided by Yahoo Finance. Past performance does not guarantee future results.</p>
</div>
""", unsafe_allow_html=True)
st.metric("High", f"{df['High'].max():.2f}")
with col3:
    st.metric("Low", f"{df['Low'].min():.2f}")
with col4:
    st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
with col5:
    st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    
st.markdown("---")
    
# Candlestick Chart
st.subheader("📊 Price Action")
candle_fig = create_candlestick_chart(df, ticker_name, show_line_chart, theme)
st.plotly_chart(candle_fig, use_container_width=True)
    
price_insight = generate_insights(df, df['RSI'], divergences, fib_levels)
st.info(f"**Insight:** {price_insight}")
    
# Price comparison table
price_comparison = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20).copy()
st.dataframe(price_comparison.style.format({
    'Open': '{:.2f}',
    'High': '{:.2f}',
    'Low': '{:.2f}',
    'Close': '{:.2f}',
    'Volume': '{:,.0f}'
}), use_container_width=True, height=300)
    
# RSI Chart
st.subheader("📈 RSI Indicator & Divergences")
rsi_fig = create_rsi_chart(df, df['RSI'], divergences, theme)
st.plotly_chart(rsi_fig, use_container_width=True)
    
if divergences:
    div_text = f"Found {len(divergences)} divergence(s). "
    bullish_count = sum(1 for d in divergences if d[0] == 'Bullish')
    bearish_count = len(divergences) - bullish_count
    prediction = "Bullish trend expected" if bullish_count > bearish_count else "Bearish trend expected"
    st.success(f"**Prediction:** {div_text}{prediction} based on divergence analysis.")
else:
    st.warning("**Prediction:** No significant divergences detected. Monitor for confirmation signals.")
    
# Divergences table
if divergences:
    div_df = pd.DataFrame(divergences, columns=['Type', 'Date', 'Price'])
    st.dataframe(div_df, use_container_width=True)
    
# Fibonacci Levels
st.subheader("🎯 Fibonacci Retracement Levels")
fib_fig = create_fibonacci_chart(df, fib_levels, ticker_name, theme)
st.plotly_chart(fib_fig, use_container_width=True)
    
fib_insight = f"Key support at {fib_levels['61.8%']:.2f} and resistance at {fib_levels['23.6%']:.2f}. "
current_price = df['Close'].iloc[-1]
if current_price < fib_levels['50%']:
    fib_insight += "Price below 50% level suggests bearish momentum."
else:
    fib_insight += "Price above 50% level indicates bullish strength."
st.info(f"**Insight:** {fib_insight}")
    
# Fibonacci levels table
fib_df = pd.DataFrame(list(fib_levels.items()), columns=['Level', 'Price'])
fib_df['Distance from Current'] = ((fib_df['Price'] - current_price) / current_price * 100)
st.dataframe(fib_df.style.format({
    'Price': '{:.2f}',
    'Distance from Current': '{:.2f}%'
}), use_container_width=True)
    
    # Returns Heatmap
    st.subheader("🔥 Returns Heatmap")
    heatmap_fig, pivot_data = create_returns_heatmap(df, heatmap_type, theme)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Heatmap statistics
    positive_cells = (pivot_data.values > 0).sum()
    negative_cells = (pivot_data.values < 0).sum()
    avg_return = pivot_data.values.flatten()
    avg_return = avg_return[~np.isnan(avg_return)].mean()
    st.info(f"**Insight:** {positive_cells} positive periods vs {negative_cells} negative periods. "
            f"Average return: {avg_return:.1f}%.")
    
    # Heatmap data table
    st.dataframe(pivot_data.style.background_gradient(cmap='RdYlGn').format('{:.1f}'),
                use_container_width=True, height=300)
    
    # Daily Statistics Table
    st.subheader("📋 Daily Statistics & Performance")
    
    # Color coding for returns
    def color_negative_red(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            color = '#00ff88' if val > 0 else '#ff3366' if val < 0 else 'white'
            return f'color: {color}'
        return ''
    
    styled_stats = daily_stats.tail(50).style.applymap(
        color_negative_red, 
        subset=['Daily_Change', 'Daily_Return_%']
    ).format({
        'Open': '{:.2f}',
        'High': '{:.2f}',
        'Low': '{:.2f}',
        'Close': '{:.2f}',
        'Volume': '{:,.0f}',
        'Daily_Change': '{:.2f}',
        'Daily_Return_%': '{:.2f}%',
        'Points_Moved': '{:.2f}',
        'Range_%': '{:.2f}%'
    })
    
    st.dataframe(styled_stats, use_container_width=True, height=400)
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        csv = daily_stats.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f'{ticker_name}_data.csv',
            mime='text/csv',
        )
    
    with col2:
