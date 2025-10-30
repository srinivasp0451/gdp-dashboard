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
    """Flatten multi-index dataframe from yfinance and remove timezone"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    if 'Date' in df.columns:
        if df['Date'].dtype == 'datetime64[ns, UTC]' or hasattr(df['Date'].dtype, 'tz'):
            df['Date'] = df['Date'].dt.tz_localize(None)
        df.set_index('Date', inplace=True)
    elif 'Datetime' in df.columns:
        if df['Datetime'].dtype == 'datetime64[ns, UTC]' or hasattr(df['Datetime'].dtype, 'tz'):
            df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        df.set_index('Datetime', inplace=True)
    
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
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
            price_min = prices.iloc[i-window:i+window].min()
            if prices.iloc[i] == price_min and prices.iloc[i] < prices.iloc[i-window]:
                if rsi.iloc[i] > rsi.iloc[i-window]:
                    divergences.append(('Bullish', prices.index[i], prices.iloc[i]))
            
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

def create_synchronized_charts(df, rsi, ticker_name, ratio_df=None, ratio_ticker_name=None, show_line=False, theme='plotly_dark'):
    """Create synchronized charts - Price, RSI, and optionally Ratio"""
    rows = 3 if ratio_df is not None else 2
    row_heights = [0.5, 0.25, 0.25] if ratio_df is not None else [0.7, 0.3]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(f'{ticker_name} Price Chart', 'RSI Indicator', 'Ratio Chart' if ratio_df is not None else '')
    )
    
    # Price chart
    if show_line:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines', name='Close Price',
            line=dict(color='#00d4ff', width=2)
        ), row=1, col=1)
    else:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name=ticker_name,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff3366'
        ), row=1, col=1)
    
    # RSI chart
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        mode='lines', name='RSI',
        line=dict(color='#ffa500', width=2)
    ), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Ratio chart if provided
    if ratio_df is not None:
        ratio = df['Close'] / ratio_df['Close']
        fig.add_trace(go.Scatter(
            x=ratio.index, y=ratio,
            mode='lines', name=f'{ticker_name}/{ratio_ticker_name}',
            line=dict(color='#00ffff', width=2)
        ), row=3, col=1)
        
        fig.update_yaxes(title_text="Ratio", row=3, col=1)
    
    fig.update_xaxes(title_text="Date", row=rows, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    fig.update_layout(
        template=theme,
        height=900,
        hovermode='x unified',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def validate_predictions(df, rsi, divergences, fib_levels, lookback=10):
    """Validate prediction accuracy based on historical signals"""
    validation_results = {
        'rsi_overbought': {'correct': 0, 'total': 0},
        'rsi_oversold': {'correct': 0, 'total': 0},
        'bullish_div': {'correct': 0, 'total': 0},
        'bearish_div': {'correct': 0, 'total': 0},
        'fib_support': {'correct': 0, 'total': 0},
        'fib_resistance': {'correct': 0, 'total': 0}
    }
    
    # Validate RSI signals
    for i in range(len(df) - lookback):
        if i < lookback:
            continue
            
        current_rsi = rsi.iloc[i]
        future_price = df['Close'].iloc[i + lookback]
        current_price = df['Close'].iloc[i]
        
        # RSI Overbought
        if current_rsi > 70:
            validation_results['rsi_overbought']['total'] += 1
            if future_price < current_price:
                validation_results['rsi_overbought']['correct'] += 1
        
        # RSI Oversold
        if current_rsi < 30:
            validation_results['rsi_oversold']['total'] += 1
            if future_price > current_price:
                validation_results['rsi_oversold']['correct'] += 1
    
    # Validate divergences
    for div_type, date, price in divergences:
        try:
            idx = df.index.get_loc(date)
            if idx + lookback < len(df):
                future_price = df['Close'].iloc[idx + lookback]
                
                if div_type == 'Bullish':
                    validation_results['bullish_div']['total'] += 1
                    if future_price > price:
                        validation_results['bullish_div']['correct'] += 1
                else:
                    validation_results['bearish_div']['total'] += 1
                    if future_price < price:
                        validation_results['bearish_div']['correct'] += 1
        except:
            continue
    
    # Validate Fibonacci levels
    current_price = df['Close'].iloc[-1]
    fib_50 = fib_levels['50%']
    
    for i in range(len(df) - lookback):
        if i < lookback:
            continue
        
        price_at_i = df['Close'].iloc[i]
        future_price = df['Close'].iloc[i + lookback]
        
        # Below Fib 50% (support test)
        if price_at_i < fib_50 * 1.02 and price_at_i > fib_50 * 0.98:
            validation_results['fib_support']['total'] += 1
            if future_price > price_at_i:
                validation_results['fib_support']['correct'] += 1
        
        # Above Fib 50% (resistance test)
        fib_23 = fib_levels['23.6%']
        if price_at_i < fib_23 * 1.02 and price_at_i > fib_23 * 0.98:
            validation_results['fib_resistance']['total'] += 1
            if future_price < price_at_i:
                validation_results['fib_resistance']['correct'] += 1
    
    return validation_results

def analyze_ratio_ranges(comparison_df, bins=20):
    """Analyze ratio ranges and categorize returns as High/Medium/Low"""
    ratio_data = comparison_df.copy()
    ratio_data['Future_Return'] = ratio_data['Ratio'].shift(-5).pct_change(5) * 100
    ratio_data = ratio_data.dropna()
    
    if len(ratio_data) == 0:
        return None
    
    # Create bins
    ratio_data['Ratio_Bin'] = pd.cut(ratio_data['Ratio'], bins=bins)
    
    # Analyze returns in each bin
    bin_stats = ratio_data.groupby('Ratio_Bin', observed=True).agg({
        'Future_Return': ['mean', 'std', 'count']
    }).round(2)
    
    bin_stats.columns = ['Avg_Return', 'Std_Dev', 'Count']
    bin_stats = bin_stats.reset_index()
    bin_stats['Ratio_Range'] = bin_stats['Ratio_Bin'].astype(str)
    
    # Categorize into High/Medium/Low
    returns = bin_stats['Avg_Return'].values
    if len(returns) > 0:
        high_threshold = np.percentile(returns[~np.isnan(returns)], 75)
        low_threshold = np.percentile(returns[~np.isnan(returns)], 25)
        
        def categorize(val):
            if pd.isna(val):
                return 'Unknown'
            if val >= high_threshold:
                return 'High Return'
            elif val <= low_threshold:
                return 'Low Return'
            else:
                return 'Medium Return'
        
        bin_stats['Category'] = bin_stats['Avg_Return'].apply(categorize)
    else:
        bin_stats['Category'] = 'Unknown'
    
    return bin_stats

def create_summary_insights(df, rsi, divergences, fib_levels, validation_results, ratio_stats=None):
    """Create comprehensive summary of all insights"""
    summary = {
        'price_analysis': {},
        'rsi_analysis': {},
        'divergence_analysis': {},
        'fibonacci_analysis': {},
        'prediction_accuracy': {},
        'ratio_analysis': {}
    }
    
    # Price Analysis
    current_price = df['Close'].iloc[-1]
    price_range = df['Close'].max() - df['Close'].min()
    price_position = (current_price - df['Close'].min()) / price_range * 100
    
    if price_position > 75:
        price_level = "High"
    elif price_position > 25:
        price_level = "Medium"
    else:
        price_level = "Low"
    
    summary['price_analysis'] = {
        'current': current_price,
        'range': price_range,
        'position': price_position,
        'level': price_level,
        'interpretation': f"Price is in the {price_level.lower()} range ({price_position:.1f}% of historical range)"
    }
    
    # RSI Analysis
    current_rsi = rsi.iloc[-1]
    if current_rsi > 70:
        rsi_level = "High (Overbought)"
    elif current_rsi > 50:
        rsi_level = "Medium-High"
    elif current_rsi > 30:
        rsi_level = "Medium"
    else:
        rsi_level = "Low (Oversold)"
    
    summary['rsi_analysis'] = {
        'current': current_rsi,
        'level': rsi_level,
        'interpretation': f"RSI is {rsi_level} suggesting " + 
                         ("potential reversal down" if current_rsi > 70 else
                          "potential bounce up" if current_rsi < 30 else
                          "neutral momentum")
    }
    
    # Divergence Analysis
    bullish_divs = sum(1 for d in divergences if d[0] == 'Bullish')
    bearish_divs = sum(1 for d in divergences if d[0] == 'Bearish')
    
    summary['divergence_analysis'] = {
        'bullish_count': bullish_divs,
        'bearish_count': bearish_divs,
        'interpretation': f"Found {bullish_divs} bullish and {bearish_divs} bearish divergences. " +
                         ("Bullish bias" if bullish_divs > bearish_divs else
                          "Bearish bias" if bearish_divs > bullish_divs else
                          "Neutral")
    }
    
    # Fibonacci Analysis
    fib_50 = fib_levels['50%']
    if current_price > fib_levels['23.6%']:
        fib_level = "High (Above 23.6%)"
    elif current_price > fib_50:
        fib_level = "Medium-High (Above 50%)"
    elif current_price > fib_levels['61.8%']:
        fib_level = "Medium-Low (Above 61.8%)"
    else:
        fib_level = "Low (Below 61.8%)"
    
    summary['fibonacci_analysis'] = {
        'level': fib_level,
        'support': fib_levels['61.8%'],
        'resistance': fib_levels['23.6%'],
        'interpretation': f"Price is {fib_level}, key support at {fib_levels['61.8%']:.2f}"
    }
    
    # Prediction Accuracy
    for key, val in validation_results.items():
        if val['total'] > 0:
            accuracy = (val['correct'] / val['total']) * 100
            summary['prediction_accuracy'][key] = {
                'accuracy': accuracy,
                'correct': val['correct'],
                'total': val['total']
            }
    
    # Ratio Analysis
    if ratio_stats is not None:
        high_return_bins = ratio_stats[ratio_stats['Category'] == 'High Return']
        medium_return_bins = ratio_stats[ratio_stats['Category'] == 'Medium Return']
        low_return_bins = ratio_stats[ratio_stats['Category'] == 'Low Return']
        
        summary['ratio_analysis'] = {
            'high_return_ranges': high_return_bins['Ratio_Range'].tolist(),
            'medium_return_ranges': medium_return_bins['Ratio_Range'].tolist(),
            'low_return_ranges': low_return_bins['Ratio_Range'].tolist(),
            'high_avg': high_return_bins['Avg_Return'].mean() if len(high_return_bins) > 0 else 0,
            'medium_avg': medium_return_bins['Avg_Return'].mean() if len(medium_return_bins) > 0 else 0,
            'low_avg': low_return_bins['Avg_Return'].mean() if len(low_return_bins) > 0 else 0
        }
    
    return summary

def prepare_df_for_excel(df):
    """Prepare dataframe for Excel export by removing timezone from index"""
    df_copy = df.copy()
    if hasattr(df_copy.index, 'tz') and df_copy.index.tz is not None:
        df_copy.index = df_copy.index.tz_localize(None)
    return df_copy

def calculate_daily_stats(df):
    """Calculate daily statistics"""
    stats_df = df.copy()
    stats_df['Daily_Change'] = stats_df['Close'].diff()
    stats_df['Daily_Return_%'] = stats_df['Close'].pct_change() * 100
    stats_df['Points_Moved'] = stats_df['High'] - stats_df['Low']
    stats_df['Range_%'] = (stats_df['Points_Moved'] / stats_df['Open']) * 100
    
    return stats_df[['Open', 'High', 'Low', 'Close', 'Volume', 
                     'Daily_Change', 'Daily_Return_%', 'Points_Moved', 'Range_%']].dropna()

def analyze_ratio(df1, df2, ticker1_name, ticker2_name, bins=20):
    """Analyze ratio between two tickers"""
    combined = pd.DataFrame({
        'Price1': df1['Close'],
        'Price2': df2['Close']
    }).dropna()
    
    combined['Ratio'] = combined['Price1'] / combined['Price2']
    combined['Change'] = combined['Ratio'].diff()
    combined['Direction'] = combined['Change'].apply(lambda x: 'Up' if x > 0 else 'Down' if x < 0 else 'Neutral')
    
    comparison_df = combined[['Price1', 'Price2', 'Ratio']].copy()
    comparison_df.columns = [f'{ticker1_name} Close', f'{ticker2_name} Close', 'Ratio']
    comparison_df[f'{ticker1_name} Change %'] = comparison_df[f'{ticker1_name} Close'].pct_change() * 100
    comparison_df[f'{ticker2_name} Change %'] = comparison_df[f'{ticker2_name} Close'].pct_change() * 100
    comparison_df['Ratio Change %'] = comparison_df['Ratio'].pct_change() * 100
    
    return comparison_df

# Sidebar
st.sidebar.title("🚀 Algo Trading Dashboard")
st.sidebar.markdown("---")

# Theme selector
st.sidebar.subheader("🎨 Theme")
theme_choice = st.sidebar.radio("Select Theme", ["Dark", "Light"], horizontal=True)
st.session_state.theme = 'plotly_dark' if theme_choice == 'Dark' else 'plotly_white'

if theme_choice == 'Dark':
    st.markdown('<style>.stApp {background-color: #0e1117;}</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>.stApp {background-color: #ffffff;}</style>', unsafe_allow_html=True)

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
interval = st.sidebar.selectbox("Interval", ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "5d", "1wk", "1mo", "3mo"])
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])

# Chart options
st.sidebar.subheader("📈 Chart Options")
show_line_chart = st.sidebar.checkbox("Show Line Chart", value=False)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
lookback_period = st.sidebar.slider("Prediction Validation Lookback", 5, 30, 10)

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
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                st.error(f"No data found for ticker: {ticker}")
            else:
                data = flatten_multiindex_dataframe(data)
                st.session_state.df = data
                st.session_state.ticker_symbol = ticker
                st.session_state.data_fetched = True
                
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
    
    # Validate predictions
    validation_results = validate_predictions(df, df['RSI'], divergences, fib_levels, lookback_period)
    
    # Ratio analysis if enabled
    ratio_stats = None
    comparison_df = None
    if enable_ratio and 'ratio_df' in st.session_state:
        comparison_df = analyze_ratio(df, st.session_state.ratio_df, ticker_name, st.session_state.ratio_ticker, ratio_bins)
        ratio_stats = analyze_ratio_ranges(comparison_df, ratio_bins)
    
    # Generate comprehensive summary
    summary = create_summary_insights(df, df['RSI'], divergences, fib_levels, validation_results, ratio_stats)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        change_val = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) >= 2 else 0
        st.metric("Current Price", f"{df['Close'].iloc[-1]:.2f}", f"{change_val:.2f}%")
    with col2:
        st.metric("High", f"{df['High'].max():.2f}")
    with col3:
        st.metric("Low", f"{df['Low'].min():.2f}")
    with col4:
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    with col5:
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    
    st.markdown("---")
    
    # Synchronized Charts
    st.subheader("📊 Synchronized Charts (Price + RSI + Ratio)")
    sync_fig = create_synchronized_charts(
        df, df['RSI'], ticker_name,
        st.session_state.ratio_df if enable_ratio and 'ratio_df' in st.session_state else None,
        st.session_state.ratio_ticker if enable_ratio and 'ratio_df' in st.session_state else None,
        show_line_chart, theme
    )
    st.plotly_chart(sync_fig, use_container_width=True)
    
    st.info(f"**💡 Tip:** Hover over any chart to see synchronized values across all charts!")
    
    st.markdown("---")
    
    # COMPREHENSIVE SUMMARY
    st.header("📊 COMPREHENSIVE INSIGHTS SUMMARY")
    
    # Price Analysis Summary
    st.subheader("💰 Price Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Price Level", summary['price_analysis']['level'])
    with col2:
        st.metric("Position in Range", f"{summary['price_analysis']['position']:.1f}%")
    with col3:
        st.metric("Range", f"{summary['price_analysis']['range']:.2f}")
    st.info(summary['price_analysis']['interpretation'])
    
    # RSI Analysis Summary
    st.subheader("📈 RSI Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RSI Level", summary['rsi_analysis']['level'])
    with col2:
        st.metric("RSI Value", f"{summary['rsi_analysis']['current']:.1f}")
    st.info(summary['rsi_analysis']['interpretation'])
    
    # Divergence Summary
    st.subheader("🔄 Divergence Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bullish Divergences", summary['divergence_analysis']['bullish_count'])
    with col2:
        st.metric("Bearish Divergences", summary['divergence_analysis']['bearish_count'])
    st.info(summary['divergence_analysis']['interpretation'])
    
    # Fibonacci Summary
    st.subheader("🎯 Fibonacci Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Level", summary['fibonacci_analysis']['level'])
    with col2:
        st.metric("Support", f"{summary['fibonacci_analysis']['support']:.2f}")
    with col3:
        st.metric("Resistance", f"{summary['fibonacci_analysis']['resistance']:.2f}")
    st.info(summary['fibonacci_analysis']['interpretation'])
    
    # PREDICTION ACCURACY
    st.markdown("---")
    st.header("🎯 PREDICTION ACCURACY VALIDATION")
    st.markdown(f"**Analysis Period:** Last {lookback_period} periods for each signal")
    
    accuracy_data = []
    for key, val in summary['prediction_accuracy'].items():
        signal_name = key.replace('_', ' ').title()
        accuracy_data.append({
            'Signal Type': signal_name,
            'Accuracy %': f"{val['accuracy']:.1f}%",
            'Correct Predictions': val['correct'],
            'Total Signals': val['total'],
            'Status': '✅ High' if val['accuracy'] >= 70 else '⚠️ Medium' if val['accuracy'] >= 50 else '❌ Low'
        })
    
    if accuracy_data:
        accuracy_df = pd.DataFrame(accuracy_data)
        st.dataframe(accuracy_df, use_container_width=True, hide_index=True)
        
        # Overall accuracy
        total_correct = sum(v['correct'] for v in summary['prediction_accuracy'].values())
        total_signals = sum(v['total'] for v in summary['prediction_accuracy'].values())
        overall_accuracy = (total_correct / total_signals * 100) if total_signals > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
        with col2:
            st.metric("Total Correct", total_correct)
        with col3:
            st.metric("Total Signals", total_signals)
        
        if overall_accuracy >= 70:
            st.success(f"✅ **HIGH CONFIDENCE**: Historical predictions have {overall_accuracy:.1f}% accuracy. Signals are reliable!")
        elif overall_accuracy >= 50:
            st.warning(f"⚠️ **MEDIUM CONFIDENCE**: Historical predictions have {overall_accuracy:.1f}% accuracy. Use with caution.")
        else:
            st.error(f"❌ **LOW CONFIDENCE**: Historical predictions have {overall_accuracy:.1f}% accuracy. Signals may be unreliable.")
    else:
        st.info("Not enough historical data to validate predictions.")
    
    # RATIO ANALYSIS SUMMARY
    if ratio_stats is not None and enable_ratio:
        st.markdown("---")
        st.header("⚖️ RATIO ANALYSIS SUMMARY")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🟢 HIGH RETURN RANGES")
            st.metric("Avg Return", f"{summary['ratio_analysis']['high_avg']:.2f}%")
            if summary['ratio_analysis']['high_return_ranges']:
                for rng in summary['ratio_analysis']['high_return_ranges'][:3]:
                    st.text(f"📍 {rng}")
            else:
                st.text("No high return ranges")
        
        with col2:
            st.markdown("### 🟡 MEDIUM RETURN RANGES")
            st.metric("Avg Return", f"{summary['ratio_analysis']['medium_avg']:.2f}%")
            if summary['ratio_analysis']['medium_return_ranges']:
                for rng in summary['ratio_analysis']['medium_return_ranges'][:3]:
                    st.text(f"📍 {rng}")
            else:
                st.text("No medium return ranges")
        
        with col3:
            st.markdown("### 🔴 LOW RETURN RANGES")
            st.metric("Avg Return", f"{summary['ratio_analysis']['low_avg']:.2f}%")
            if summary['ratio_analysis']['low_return_ranges']:
                for rng in summary['ratio_analysis']['low_return_ranges'][:3]:
                    st.text(f"📍 {rng}")
            else:
                st.text("No low return ranges")
        
        # Current ratio position
        current_ratio = comparison_df['Ratio'].iloc[-1]
        st.markdown("### 📊 Current Ratio Position")
        
        # Find which category current ratio falls into
        current_category = "Unknown"
        if ratio_stats is not None:
            for _, row in ratio_stats.iterrows():
                try:
                    range_str = row['Ratio_Range']
                    if '(' in range_str and ')' in range_str:
                        range_clean = range_str.strip('()[]')
                        parts = range_clean.split(',')
                        if len(parts) == 2:
                            low = float(parts[0].strip())
                            high = float(parts[1].strip())
                            if low <= current_ratio <= high:
                                current_category = row['Category']
                                break
                except:
                    continue
        
        if current_category == "High Return":
            st.success(f"✅ Current ratio {current_ratio:.4f} is in HIGH RETURN range. Historically favorable for gains!")
        elif current_category == "Medium Return":
            st.info(f"⚠️ Current ratio {current_ratio:.4f} is in MEDIUM RETURN range. Moderate opportunity.")
        elif current_category == "Low Return":
            st.warning(f"❌ Current ratio {current_ratio:.4f} is in LOW RETURN range. Historically unfavorable.")
        else:
            st.info(f"Current ratio: {current_ratio:.4f}")
        
        # Detailed ratio table
        st.subheader("📋 Detailed Ratio Range Analysis")
        if ratio_stats is not None:
            ratio_display = ratio_stats[['Ratio_Range', 'Avg_Return', 'Std_Dev', 'Count', 'Category']].copy()
            
            def color_category(val):
                if val == 'High Return':
                    return 'background-color: #00ff8844; color: #00ff88'
                elif val == 'Low Return':
                    return 'background-color: #ff336644; color: #ff3366'
                elif val == 'Medium Return':
                    return 'background-color: #ffaa0044; color: #ffaa00'
                return ''
            
            styled_ratio = ratio_display.style.applymap(color_category, subset=['Category']).format({
                'Avg_Return': '{:.2f}%',
                'Std_Dev': '{:.2f}',
                'Count': '{:.0f}'
            })
            
            st.dataframe(styled_ratio, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Fibonacci Levels Table
    st.subheader("🎯 Fibonacci Retracement Levels")
    fib_df = pd.DataFrame(list(fib_levels.items()), columns=['Level', 'Price'])
    current_price = df['Close'].iloc[-1]
    fib_df['Distance from Current'] = ((fib_df['Price'] - current_price) / current_price * 100)
    fib_df['Status'] = fib_df['Distance from Current'].apply(
        lambda x: 'Support ⬇️' if x < -1 else 'Resistance ⬆️' if x > 1 else 'Current Level 🎯'
    )
    st.dataframe(fib_df.style.format({
        'Price': '{:.2f}',
        'Distance from Current': '{:.2f}%'
    }), use_container_width=True)
    
    # Divergences Table
    if divergences:
        st.subheader("🔄 Detected Divergences")
        div_df = pd.DataFrame(divergences, columns=['Type', 'Date', 'Price'])
        
        # Add validation for each divergence if possible
        div_df['Validated'] = '⏳ Pending'
        for idx, row in div_df.iterrows():
            try:
                div_date = row['Date']
                div_idx = df.index.get_loc(div_date)
                if div_idx + lookback_period < len(df):
                    future_price = df['Close'].iloc[div_idx + lookback_period]
                    current_div_price = row['Price']
                    
                    if row['Type'] == 'Bullish':
                        div_df.at[idx, 'Validated'] = '✅ Success' if future_price > current_div_price else '❌ Failed'
                    else:
                        div_df.at[idx, 'Validated'] = '✅ Success' if future_price < current_div_price else '❌ Failed'
            except:
                continue
        
        st.dataframe(div_df, use_container_width=True, hide_index=True)
    
    # Daily Statistics Table
    st.subheader("📋 Daily Statistics & Performance")
    
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
    st.subheader("📥 Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = daily_stats.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="📥 Download Price Data CSV",
            data=csv,
            file_name=f'{ticker_name}_data.csv',
            mime='text/csv',
        )
    
    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            daily_stats_export = prepare_df_for_excel(daily_stats)
            daily_stats_export.to_excel(writer, sheet_name='Price Data')
            
            # Add summary sheet
            summary_df = pd.DataFrame([
                ['Price Level', summary['price_analysis']['level']],
                ['RSI Level', summary['rsi_analysis']['level']],
                ['Overall Accuracy', f"{overall_accuracy:.1f}%"],
                ['Total Divergences', len(divergences)]
            ], columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
        st.download_button(
            label="📥 Download Full Report Excel",
            data=buffer.getvalue(),
            file_name=f'{ticker_name}_report.xlsx',
            mime='application/vnd.ms-excel'
        )
    
    with col3:
        if comparison_df is not None:
            ratio_csv = comparison_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Download Ratio CSV",
                data=ratio_csv,
                file_name=f'{ticker_name}_ratio.csv',
                mime='text/csv',
            )
    
    # Historical Performance
    st.markdown("---")
    st.header("📊 Historical Pattern-Based Prediction")
    
    positive_days = (daily_stats['Daily_Return_%'] > 0).sum()
    total_days = len(daily_stats)
    win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0
    
    avg_gain = daily_stats[daily_stats['Daily_Return_%'] > 0]['Daily_Return_%'].mean()
    avg_loss = daily_stats[daily_stats['Daily_Return_%'] < 0]['Daily_Return_%'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{positive_days}/{total_days} days")
    with col2:
        st.metric("Avg Gain", f"{avg_gain:.2f}%" if not pd.isna(avg_gain) else "N/A", "On up days")
    with col3:
        st.metric("Avg Loss", f"{avg_loss:.2f}%" if not pd.isna(avg_loss) else "N/A", "On down days")
    with col4:
        if not pd.isna(avg_loss) and avg_loss != 0:
            risk_reward = abs(avg_gain / avg_loss)
            st.metric("Risk-Reward", f"{risk_reward:.2f}:1")
        else:
            st.metric("Risk-Reward", "N/A")
    
    # Expected value
    if not pd.isna(avg_gain) and not pd.isna(avg_loss):
        expected_value = (win_rate/100 * avg_gain) + ((100-win_rate)/100 * avg_loss)
        if expected_value > 0:
            st.success(f"**Positive Expected Value:** +{expected_value:.2f}% per period. Historically profitable pattern.")
        else:
            st.warning(f"**Negative Expected Value:** {expected_value:.2f}% per period. Exercise caution.")
    
    # Final Trading Recommendation
    st.markdown("---")
    st.header("🎯 FINAL TRADING RECOMMENDATION")
    
    # Calculate overall score
    score = 0
    max_score = 0
    
    # RSI score
    max_score += 1
    if df['RSI'].iloc[-1] < 30:
        score += 1
    elif df['RSI'].iloc[-1] > 70:
        score -= 1
    
    # Divergence score
    max_score += 1
    if summary['divergence_analysis']['bullish_count'] > summary['divergence_analysis']['bearish_count']:
        score += 1
    elif summary['divergence_analysis']['bearish_count'] > summary['divergence_analysis']['bullish_count']:
        score -= 1
    
    # Fibonacci score
    max_score += 1
    if current_price > fib_levels['50%']:
        score += 1
    else:
        score -= 1
    
    # Accuracy score
    max_score += 1
    if overall_accuracy >= 70:
        score += 1
    elif overall_accuracy < 50:
        score -= 0.5
    
    # Win rate score
    max_score += 1
    if win_rate > 55:
        score += 1
    elif win_rate < 45:
        score -= 1
    
    recommendation_pct = (score / max_score) * 100 if max_score > 0 else 50
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Recommendation Score", f"{recommendation_pct:.0f}/100")
    with col2:
        st.metric("Confidence Level", 
                 "High" if overall_accuracy >= 70 else "Medium" if overall_accuracy >= 50 else "Low")
    
    if recommendation_pct >= 70:
        st.success(f"""
        ✅ **STRONG BUY SIGNAL**
        - Multiple bullish indicators aligned
        - High historical accuracy ({overall_accuracy:.1f}%)
        - Positive risk-reward ratio
        - Consider entering long positions with proper risk management
        """)
    elif recommendation_pct >= 40:
        st.info(f"""
        ⚖️ **NEUTRAL/HOLD SIGNAL**
        - Mixed signals from different indicators
        - Moderate accuracy ({overall_accuracy:.1f}%)
        - Wait for stronger confirmation
        - Consider range-bound trading strategies
        """)
    else:
        st.error(f"""
        ⚠️ **CAUTION/SELL SIGNAL**
        - Multiple bearish indicators present
        - Accuracy: {overall_accuracy:.1f}%
        - Consider protective stops or short positions
        - Risk management is critical
        """)
    
    st.markdown("---")
    st.info("💡 **Disclaimer**: This analysis is for educational purposes only. Always conduct your own research and consider consulting with financial advisors before making investment decisions.")

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
