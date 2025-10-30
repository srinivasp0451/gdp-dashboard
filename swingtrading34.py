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

# Custom CSS for better visuals
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = None

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
def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_divergences(prices, rsi, window=5):
    """Find RSI divergences"""
    divergences = []
    
    for i in range(window, len(prices) - window):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (prices.iloc[i] < prices.iloc[i-window] and 
            rsi.iloc[i] > rsi.iloc[i-window] and 
            prices.iloc[i] == prices.iloc[i-window:i+window].min()):
            divergences.append(('Bullish', prices.index[i], prices.iloc[i]))
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (prices.iloc[i] > prices.iloc[i-window] and 
            rsi.iloc[i] < rsi.iloc[i-window] and 
            prices.iloc[i] == prices.iloc[i-window:i+window].max()):
            divergences.append(('Bearish', prices.index[i], prices.iloc[i]))
    
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

def create_candlestick_chart(df, ticker_name, show_line=False):
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
        template='plotly_dark',
        height=600,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_rsi_chart(df, rsi, divergences):
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
        fig.add_trace(go.Scatter(
            x=[date],
            y=[rsi.loc[date]],
            mode='markers',
            name=f'{div_type} Divergence',
            marker=dict(size=15, color=color, symbol='star'),
            showlegend=True
        ))
    
    fig.update_layout(
        title='RSI Indicator with Divergences',
        yaxis_title='RSI Value',
        xaxis_title='Date',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_fibonacci_chart(df, fib_levels, ticker_name):
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
        template='plotly_dark',
        height=600,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_returns_heatmap(df):
    """Create monthly returns heatmap"""
    df_copy = df.copy()
    df_copy['Returns'] = df_copy['Close'].pct_change() * 100
    df_copy['Year'] = df_copy.index.year
    df_copy['Month'] = df_copy.index.month
    
    pivot_table = df_copy.pivot_table(
        values='Returns',
        index='Month',
        columns='Year',
        aggfunc='sum'
    )
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table.index = [month_names[i-1] for i in pivot_table.index]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlGn',
        text=np.round(pivot_table.values, 2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Returns %")
    ))
    
    fig.update_layout(
        title='Monthly Returns Heatmap (%)',
        xaxis_title='Year',
        yaxis_title='Month',
        template='plotly_dark',
        height=500
    )
    
    return fig

def analyze_ratio(df1, df2, ticker1_name, ticker2_name, bins=20):
    """Analyze ratio between two tickers"""
    ratio_df = pd.DataFrame()
    ratio_df['Ratio'] = df1['Close'] / df2['Close']
    ratio_df['Change'] = ratio_df['Ratio'].diff()
    ratio_df['Direction'] = np.where(ratio_df['Change'] > 0, 'Up', 'Down')
    
    # Create bins
    ratio_df['Ratio_Bin'] = pd.cut(ratio_df['Ratio'], bins=bins)
    
    # Analyze each bin
    bin_analysis = ratio_df.groupby('Ratio_Bin').agg({
        'Direction': lambda x: (x == 'Up').sum() / len(x) * 100,
        'Ratio': 'count'
    }).rename(columns={'Direction': 'Up_Percentage', 'Ratio': 'Count'})
    
    bin_analysis['Down_Percentage'] = 100 - bin_analysis['Up_Percentage']
    
    # Create ratio chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratio_df.index,
        y=ratio_df['Ratio'],
        mode='lines',
        name='Ratio',
        line=dict(color='#00d4ff', width=2)
    ))
    
    fig.update_layout(
        title=f'{ticker1_name} / {ticker2_name} Ratio Chart',
        yaxis_title='Ratio',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    return fig, bin_analysis, ratio_df

def generate_insights(df, rsi, divergences, fib_levels):
    """Generate human-readable insights"""
    insights = []
    
    # Price trend
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
    ratio_bins = st.sidebar.slider("Ratio Bins", 10, 50, 20)

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
                st.session_state.df = data
                st.session_state.ticker_symbol = ticker
                st.session_state.data_fetched = True
                
                # Fetch ratio ticker if enabled
                if enable_ratio:
                    ratio_data = yf.download(ratio_ticker, period=period, interval=interval, progress=False)
                    st.session_state.ratio_df = ratio_data
                    st.session_state.ratio_ticker = ratio_ticker
                
                st.success(f"✅ Data fetched successfully for {ticker}!")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Display analysis if data is fetched
if st.session_state.data_fetched and st.session_state.df is not None:
    df = st.session_state.df
    ticker_name = st.session_state.ticker_symbol
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df, period=rsi_period)
    divergences = find_divergences(df['Close'], df['RSI'])
    fib_levels = calculate_fibonacci_levels(df)
    daily_stats = calculate_daily_stats(df)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Current Price", f"{df['Close'].iloc[-1]:.2f}",
                 f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}%")
    with col2:
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
    candle_fig = create_candlestick_chart(df, ticker_name, show_line_chart)
    st.plotly_chart(candle_fig, use_container_width=True)
    
    price_insight = generate_insights(df, df['RSI'], divergences, fib_levels)
    st.info(f"**Insight:** {price_insight}")
    
    # RSI Chart
    st.subheader("📈 RSI Indicator & Divergences")
    rsi_fig = create_rsi_chart(df, df['RSI'], divergences)
    st.plotly_chart(rsi_fig, use_container_width=True)
    
    if divergences:
        div_text = f"Found {len(divergences)} divergence(s). "
        bullish_count = sum(1 for d in divergences if d[0] == 'Bullish')
        bearish_count = len(divergences) - bullish_count
        prediction = "Bullish trend expected" if bullish_count > bearish_count else "Bearish trend expected"
        st.success(f"**Prediction:** {div_text}{prediction} based on divergence analysis.")
    else:
        st.warning("**Prediction:** No significant divergences detected. Monitor for confirmation signals.")
    
    # Fibonacci Levels
    st.subheader("🎯 Fibonacci Retracement Levels")
    fib_fig = create_fibonacci_chart(df, fib_levels, ticker_name)
    st.plotly_chart(fib_fig, use_container_width=True)
    
    fib_insight = f"Key support at {fib_levels['61.8%']:.2f} and resistance at {fib_levels['23.6%']:.2f}. "
    current_price = df['Close'].iloc[-1]
    if current_price < fib_levels['50%']:
        fib_insight += "Price below 50% level suggests bearish momentum."
    else:
        fib_insight += "Price above 50% level indicates bullish strength."
    st.info(f"**Insight:** {fib_insight}")
    
    # Returns Heatmap
    st.subheader("🔥 Monthly Returns Heatmap")
    heatmap_fig = create_returns_heatmap(df)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Calculate average positive and negative months
    monthly_returns = df['Close'].resample('M').last().pct_change() * 100
    positive_months = (monthly_returns > 0).sum()
    negative_months = (monthly_returns < 0).sum()
    st.info(f"**Insight:** Historically, {positive_months} positive months vs {negative_months} negative months. "
            f"Average monthly return: {monthly_returns.mean():.2f}%.")
    
    # Daily Statistics Table
    st.subheader("📋 Daily Statistics & Performance")
    
    # Color coding for returns
    def color_negative_red(val):
        if isinstance(val, (int, float)):
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
        
        ratio_fig, bin_analysis, ratio_df = analyze_ratio(
            df, 
            st.session_state.ratio_df, 
            ticker_name, 
            st.session_state.ratio_ticker,
            ratio_bins
        )
        
        st.plotly_chart(ratio_fig, use_container_width=True)
        
        st.subheader("📊 Ratio Bin Analysis")
        st.dataframe(bin_analysis.style.background_gradient(cmap='RdYlGn', subset=['Up_Percentage']),
                    use_container_width=True)
        
        # Ratio insights
        current_ratio = ratio_df['Ratio'].iloc[-1]
        ratio_mean = ratio_df['Ratio'].mean()
        ratio_insight = f"Current ratio: {current_ratio:.4f} vs historical mean: {ratio_mean:.4f}. "
        if current_ratio > ratio_mean * 1.1:
            ratio_insight += f"{ticker_name} is relatively expensive compared to {st.session_state.ratio_ticker}."
        elif current_ratio < ratio_mean * 0.9:
            ratio_insight += f"{ticker_name} is relatively cheap compared to {st.session_state.ratio_ticker}."
        else:
            ratio_insight += "Ratio is near historical average."
        st.info(f"**Insight:** {ratio_insight}")
    
    # Overall Market View
    st.markdown("---")
    st.subheader("🎯 Overall Market View")
    
    # Calculate confluence
    signals = []
    
    # RSI signal
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi > 70:
        signals.append(("RSI", "Bearish", "Overbought"))
    elif current_rsi < 30:
        signals.append(("RSI", "Bullish", "Oversold"))
    else:
        signals.append(("RSI", "Neutral", "Mid-range"))
    
    # Divergence signal
    if divergences:
        last_div = divergences[-1][0]
        signals.append(("Divergence", last_div, f"{last_div} divergence detected"))
    
    # Price vs Fibonacci
    if current_price > fib_levels['50%']:
        signals.append(("Fibonacci", "Bullish", "Above 50% level"))
    else:
        signals.append(("Fibonacci", "Bearish", "Below 50% level"))
    
    # Trend
    if recent_change > 0:
        signals.append(("Trend", "Bullish", f"+{recent_change:.2f}%"))
    else:
        signals.append(("Trend", "Bearish", f"{recent_change:.2f}%"))
    
    # Display signals
    signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Details'])
    st.dataframe(signal_df, use_container_width=True, hide_index=True)
    
    # Final verdict
    bullish_signals = sum(1 for s in signals if s[1] == "Bullish")
    bearish_signals = sum(1 for s in signals if s[1] == "Bearish")
    
    if bullish_signals > bearish_signals:
        st.success(f"✅ **BULLISH CONFLUENCE** ({bullish_signals} bullish vs {bearish_signals} bearish signals)")
    elif bearish_signals > bullish_signals:
        st.error(f"⚠️ **BEARISH CONFLUENCE** ({bearish_signals} bearish vs {bullish_signals} bullish signals)")
    else:
        st.warning(f"⚖️ **NEUTRAL** (Mixed signals - {bullish_signals} bullish, {bearish_signals} bearish)")
    
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
</div>
""", unsafe_allow_html=True)
