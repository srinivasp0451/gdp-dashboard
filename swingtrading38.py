import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Advanced Trading Pattern Analyzer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #ff7f0e; margin-top: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .positive {color: #00ff00; font-weight: bold;}
    .negative {color: #ff0000; font-weight: bold;}
    .neutral {color: #ffaa00; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = None

# Ticker mappings
TICKER_MAP = {
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

def convert_to_ist(df):
    """Convert dataframe index to IST timezone"""
    try:
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
    except:
        pass
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_pattern_matches(df, window_size=10, top_n=20):
    """Find similar patterns in historical data using sliding window"""
    if len(df) < window_size * 2:
        return []
    
    # Normalize close prices
    closes = df['Close'].values
    
    # Current pattern (last window_size points)
    current_pattern = closes[-window_size:]
    current_normalized = (current_pattern - current_pattern.min()) / (current_pattern.max() - current_pattern.min() + 1e-10)
    
    # Current volatility
    current_volatility = df['Close'].tail(window_size).pct_change().std()
    
    matches = []
    
    # Search through historical data
    for i in range(window_size, len(df) - window_size - 10):
        historical_pattern = closes[i-window_size:i]
        hist_normalized = (historical_pattern - historical_pattern.min()) / (historical_pattern.max() - historical_pattern.min() + 1e-10)
        
        # Calculate similarity (Euclidean distance)
        distance = euclidean(current_normalized, hist_normalized)
        
        # Calculate correlation
        correlation, _ = pearsonr(current_normalized, hist_normalized)
        
        # Historical volatility
        hist_volatility = df['Close'].iloc[i-window_size:i].pct_change().std()
        
        # Future movement (what happened next)
        future_returns = []
        for j in [1, 3, 5, 10]:
            if i + j < len(df):
                ret = (closes[i+j] - closes[i]) / closes[i] * 100
                future_returns.append(ret)
        
        if future_returns:
            matches.append({
                'index': i,
                'date': df.index[i],
                'distance': distance,
                'correlation': correlation,
                'volatility': hist_volatility,
                'current_volatility': current_volatility,
                'volatility_diff': abs(current_volatility - hist_volatility),
                'future_1d': future_returns[0] if len(future_returns) > 0 else 0,
                'future_3d': future_returns[1] if len(future_returns) > 1 else 0,
                'future_5d': future_returns[2] if len(future_returns) > 2 else 0,
                'future_10d': future_returns[3] if len(future_returns) > 3 else 0,
                'price_at_match': closes[i]
            })
    
    # Sort by combination of low distance and similar volatility
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df['score'] = (1 / (1 + matches_df['distance'])) * (1 / (1 + matches_df['volatility_diff'])) * matches_df['correlation']
        matches_df = matches_df.sort_values('score', ascending=False).head(top_n)
    
    return matches_df

def create_forecast_summary(matches_df, current_price):
    """Create forecast summary based on pattern matches"""
    if matches_df.empty:
        return "Insufficient data for pattern matching."
    
    summary = []
    summary.append("### 📊 PATTERN MATCHING FORECAST\n")
    
    # Average expected returns
    avg_1d = matches_df['future_1d'].mean()
    avg_3d = matches_df['future_3d'].mean()
    avg_5d = matches_df['future_5d'].mean()
    avg_10d = matches_df['future_10d'].mean()
    
    # Calculate confidence
    consistency_1d = (matches_df['future_1d'] > 0).sum() / len(matches_df) * 100
    
    summary.append(f"**Found {len(matches_df)} similar historical patterns**\n")
    summary.append(f"**Current Price:** ₹{current_price:.2f}\n")
    summary.append(f"\n**Expected Price Movements:**")
    summary.append(f"- Next 1 period: {avg_1d:+.2f}% (Target: ₹{current_price * (1 + avg_1d/100):.2f})")
    summary.append(f"- Next 3 periods: {avg_3d:+.2f}% (Target: ₹{current_price * (1 + avg_3d/100):.2f})")
    summary.append(f"- Next 5 periods: {avg_5d:+.2f}% (Target: ₹{current_price * (1 + avg_5d/100):.2f})")
    summary.append(f"- Next 10 periods: {avg_10d:+.2f}% (Target: ₹{current_price * (1 + avg_10d/100):.2f})")
    
    summary.append(f"\n**Bullish Probability:** {consistency_1d:.1f}%")
    
    # Recommendation
    if avg_1d > 2 and consistency_1d > 60:
        recommendation = "🟢 **STRONG BUY** - High probability of upward movement"
    elif avg_1d > 0.5 and consistency_1d > 50:
        recommendation = "🟢 **BUY** - Moderate bullish signals"
    elif avg_1d < -2 and consistency_1d < 40:
        recommendation = "🔴 **STRONG SELL** - High probability of downward movement"
    elif avg_1d < -0.5 and consistency_1d < 50:
        recommendation = "🔴 **SELL** - Moderate bearish signals"
    else:
        recommendation = "🟡 **HOLD** - No clear directional bias"
    
    summary.append(f"\n**Recommendation:** {recommendation}\n")
    
    return "\n".join(summary)

def analyze_returns_heatmap(df, timeframe):
    """Analyze returns by day, month, year"""
    df = df.copy()
    df['Returns'] = df['Close'].pct_change() * 100
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    analysis = []
    analysis.append("### 📈 RETURNS & VOLATILITY ANALYSIS\n")
    
    # Overall statistics
    total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    avg_daily_return = df['Returns'].mean()
    volatility = df['Returns'].std()
    max_gain = df['Returns'].max()
    max_loss = df['Returns'].min()
    
    analysis.append(f"**Overall Performance:**")
    analysis.append(f"- Total Return: {total_return:+.2f}%")
    analysis.append(f"- Average Daily Return: {avg_daily_return:+.4f}%")
    analysis.append(f"- Volatility: {volatility:.2f}%")
    analysis.append(f"- Best Day: {max_gain:+.2f}%")
    analysis.append(f"- Worst Day: {max_loss:+.2f}%")
    
    # Monthly analysis
    if len(df) > 30:
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        monthly_returns = df.groupby('Month')['Returns'].agg(['mean', 'std', 'count'])
        
        best_month = monthly_returns['mean'].idxmax()
        worst_month = monthly_returns['mean'].idxmin()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        analysis.append(f"\n**Monthly Patterns:**")
        analysis.append(f"- Best Month: **{month_names[best_month-1]}** (Avg: {monthly_returns.loc[best_month, 'mean']:+.2f}%)")
        analysis.append(f"- Worst Month: **{month_names[worst_month-1]}** (Avg: {monthly_returns.loc[worst_month, 'mean']:+.2f}%)")
    
    # Winning streaks
    df['Win'] = df['Returns'] > 0
    df['Streak'] = (df['Win'] != df['Win'].shift()).cumsum()
    streaks = df.groupby('Streak')['Returns'].agg(['sum', 'count'])
    max_win_streak = streaks[streaks['sum'] > 0]['count'].max() if not streaks[streaks['sum'] > 0].empty else 0
    max_loss_streak = streaks[streaks['sum'] < 0]['count'].max() if not streaks[streaks['sum'] < 0].empty else 0
    
    analysis.append(f"\n**Streak Analysis:**")
    analysis.append(f"- Maximum Winning Streak: {int(max_win_streak)} periods")
    analysis.append(f"- Maximum Losing Streak: {int(max_loss_streak)} periods")
    
    return "\n".join(analysis), df

def create_ratio_analysis(df1, df2, ticker1, ticker2):
    """Analyze ratio between two tickers"""
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Align dataframes
    common_index = df1.index.intersection(df2.index)
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    
    ratio = df1['Close'] / df2['Close']
    ratio_df = pd.DataFrame({'Ratio': ratio, 'Ticker1': df1['Close'], 'Ticker2': df2['Close']})
    ratio_df['Returns'] = ratio_df['Ticker1'].pct_change() * 100
    
    # Create ratio bins
    ratio_df['Ratio_Bin'] = pd.qcut(ratio_df['Ratio'], q=10, labels=False, duplicates='drop')
    
    # Analyze returns by ratio bin
    bin_analysis = ratio_df.groupby('Ratio_Bin').agg({
        'Returns': ['mean', 'std', 'count'],
        'Ratio': ['min', 'max', 'mean']
    }).round(4)
    
    current_ratio = ratio_df['Ratio'].iloc[-1]
    current_bin = ratio_df['Ratio_Bin'].iloc[-1]
    
    summary = []
    summary.append(f"### 🔄 RATIO ANALYSIS: {ticker1} / {ticker2}\n")
    summary.append(f"**Current Ratio:** {current_ratio:.4f}")
    summary.append(f"**Current Bin:** {current_bin}\n")
    
    if not bin_analysis.empty and current_bin in bin_analysis.index:
        expected_return = bin_analysis.loc[current_bin, ('Returns', 'mean')]
        ratio_range = f"{bin_analysis.loc[current_bin, ('Ratio', 'min')]:.4f} - {bin_analysis.loc[current_bin, ('Ratio', 'max')]:.4f}"
        
        summary.append(f"**Historical Performance in Current Bin:**")
        summary.append(f"- Ratio Range: {ratio_range}")
        summary.append(f"- Average Returns: {expected_return:+.2f}%")
        summary.append(f"- Volatility: {bin_analysis.loc[current_bin, ('Returns', 'std')]:.2f}%")
        summary.append(f"- Sample Size: {int(bin_analysis.loc[current_bin, ('Returns', 'count')])} periods")
        
        if expected_return > 2:
            summary.append(f"\n**Recommendation:** 🟢 **BULLISH** - Historical data suggests strong positive returns at this ratio level")
        elif expected_return < -2:
            summary.append(f"\n**Recommendation:** 🔴 **BEARISH** - Historical data suggests negative returns at this ratio level")
        else:
            summary.append(f"\n**Recommendation:** 🟡 **NEUTRAL** - No strong historical bias at this ratio level")
    
    return "\n".join(summary), ratio_df, bin_analysis

def create_rsi_analysis(df):
    """Analyze RSI patterns and create bins"""
    df = df.copy()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Future_Returns'] = df['Close'].pct_change(5).shift(-5) * 100
    
    # Create RSI bins
    rsi_bins = [0, 30, 40, 50, 60, 70, 100]
    rsi_labels = ['Oversold (<30)', 'Weak (30-40)', 'Neutral-Weak (40-50)', 
                  'Neutral-Strong (50-60)', 'Strong (60-70)', 'Overbought (>70)']
    
    df['RSI_Bin'] = pd.cut(df['RSI'], bins=rsi_bins, labels=rsi_labels, include_lowest=True)
    
    # Analyze returns by RSI bin
    rsi_analysis = df.groupby('RSI_Bin', observed=True)['Future_Returns'].agg(['mean', 'std', 'count']).round(2)
    
    current_rsi = df['RSI'].iloc[-1]
    current_bin = df['RSI_Bin'].iloc[-1]
    
    summary = []
    summary.append(f"### 📉 RSI ANALYSIS & FORECAST\n")
    summary.append(f"**Current RSI:** {current_rsi:.2f}")
    summary.append(f"**Current Status:** {current_bin}\n")
    
    if not rsi_analysis.empty and pd.notna(current_bin):
        expected_return = rsi_analysis.loc[current_bin, 'mean']
        
        summary.append(f"**Historical Performance in Current RSI Zone:**")
        summary.append(f"- Expected 5-period Return: {expected_return:+.2f}%")
        summary.append(f"- Volatility: {rsi_analysis.loc[current_bin, 'std']:.2f}%")
        summary.append(f"- Sample Size: {int(rsi_analysis.loc[current_bin, 'count'])} occurrences\n")
        
        # RSI-based recommendation
        if current_rsi < 30:
            summary.append("**Signal:** 🟢 **OVERSOLD - Potential Buy** - RSI indicates oversold conditions, possible reversal")
        elif current_rsi > 70:
            summary.append("**Signal:** 🔴 **OVERBOUGHT - Potential Sell** - RSI indicates overbought conditions, possible correction")
        elif 40 < current_rsi < 60:
            summary.append("**Signal:** 🟡 **NEUTRAL** - RSI in neutral zone, no strong signals")
        else:
            summary.append("**Signal:** 🟡 **TRANSITIONAL** - RSI in transitional zone, monitor closely")
    
    return "\n".join(summary), rsi_analysis

def plot_comprehensive_charts(df, ticker_name, matches_df=None):
    """Create comprehensive trading charts"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker_name} - Candlestick Chart', 'Volume', 'RSI (14)', 'Returns %'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff0000'
    ), row=1, col=1)
    
    # Add pattern matches as markers
    if matches_df is not None and not matches_df.empty:
        for _, match in matches_df.head(5).iterrows():
            match_date = match['date']
            if match_date in df.index:
                fig.add_annotation(
                    x=match_date,
                    y=df.loc[match_date, 'High'],
                    text=f"Similar<br>{match['future_1d']:+.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#00ffff",
                    bgcolor="#1f77b4",
                    font=dict(color="white", size=10),
                    row=1, col=1
                )
    
    # Volume
    colors = ['#00ff00' if row['Close'] >= row['Open'] else '#ff0000' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    # RSI
    rsi = calculate_rsi(df['Close'])
    fig.add_trace(go.Scatter(
        x=df.index,
        y=rsi,
        name='RSI',
        line=dict(color='#9467bd', width=2)
    ), row=3, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # Returns
    returns = df['Close'].pct_change() * 100
    fig.add_trace(go.Scatter(
        x=df.index,
        y=returns,
        name='Returns %',
        fill='tozeroy',
        line=dict(color='#ff7f0e', width=1)
    ), row=4, col=1)
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        height=1200,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig

def plot_ratio_chart(ratio_df, ticker1, ticker2):
    """Plot ratio analysis chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker1} Price', f'{ticker2} Price', f'Ratio: {ticker1}/{ticker2}'),
        row_heights=[0.33, 0.33, 0.34]
    )
    
    fig.add_trace(go.Scatter(
        x=ratio_df.index,
        y=ratio_df['Ticker1'],
        name=ticker1,
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ratio_df.index,
        y=ratio_df['Ticker2'],
        name=ticker2,
        line=dict(color='#ff7f0e', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=ratio_df.index,
        y=ratio_df['Ratio'],
        name='Ratio',
        line=dict(color='#2ca02c', width=2),
        fill='tozeroy'
    ), row=3, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig

# Main App
st.markdown('<p class="main-header">📊 Advanced Trading Pattern Analyzer</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ticker selection
    ticker_choice = st.selectbox("Select Asset", list(TICKER_MAP.keys()))
    
    if ticker_choice == "Custom":
        custom_ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA)")
        ticker_symbol = custom_ticker.upper()
    else:
        ticker_symbol = TICKER_MAP[ticker_choice]
    
    # Timeframe and period
    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Timeframe", ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'], index=7)
    with col2:
        period = st.selectbox("Period", ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y'], index=6)
    
    # Fetch button
    fetch_button = st.button("🔄 Fetch Data & Analyze", type="primary", use_container_width=True)
    
    st.divider()
    
    # Ratio Analysis
    st.subheader("🔄 Ratio Analysis")
    enable_ratio = st.checkbox("Enable Ratio Analysis")
    
    if enable_ratio:
        ticker2_choice = st.selectbox("Select Second Asset", list(TICKER_MAP.keys()), key='ticker2')
        if ticker2_choice == "Custom":
            custom_ticker2 = st.text_input("Enter Second Ticker", key='custom2')
            ticker2_symbol = custom_ticker2.upper()
        else:
            ticker2_symbol = TICKER_MAP[ticker2_choice]
    
    st.divider()
    st.info("💡 This tool uses advanced pattern matching to forecast market movements based on historical similarities.")

# Main content
if fetch_button:
    if not ticker_symbol:
        st.error("Please enter a valid ticker symbol!")
    else:
        with st.spinner(f"📥 Fetching data for {ticker_symbol}..."):
            try:
                # Fetch data
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    st.error("No data returned. Please check the ticker symbol and try again.")
                else:
                    # Convert to IST
                    df = convert_to_ist(df)
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.ticker_symbol = ticker_symbol
                    st.session_state.data_fetched = True
                    
                    st.success(f"✅ Data fetched successfully! {len(df)} data points retrieved.")
            
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.session_state.data_fetched = False

# Display analysis if data is fetched
if st.session_state.data_fetched and st.session_state.df is not None:
    df = st.session_state.df
    ticker_symbol = st.session_state.ticker_symbol
    
    # Current price info
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
    with col2:
        st.metric("High", f"₹{df['High'].iloc[-1]:.2f}")
    with col3:
        st.metric("Low", f"₹{df['Low'].iloc[-1]:.2f}")
    with col4:
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Pattern Analysis", "📊 Returns & Stats", "🔄 Ratio Analysis", "📉 RSI Analysis", "💾 Data Export"])
    
    with tab1:
        st.markdown('<p class="sub-header">Pattern Matching & Forecast</p>', unsafe_allow_html=True)
        
        with st.spinner("🔍 Searching for similar patterns..."):
            matches_df = find_pattern_matches(df, window_size=10, top_n=20)
            
            if not matches_df.empty:
                # Forecast summary
                forecast = create_forecast_summary(matches_df, current_price)
                st.markdown(forecast)
                
                st.divider()
                
                # Display top matches
                st.subheader("🎯 Top Similar Patterns Found")
                display_matches = matches_df[['date', 'distance', 'correlation', 'future_1d', 'future_3d', 'future_5d', 'future_10d', 'price_at_match']].head(10)
                display_matches.columns = ['Date', 'Distance', 'Correlation', '1-Period', '3-Period', '5-Period', '10-Period', 'Price']
                
                st.dataframe(
                    display_matches.style.format({
                        'Distance': '{:.4f}',
                        'Correlation': '{:.4f}',
                        '1-Period': '{:+.2f}%',
                        '3-Period': '{:+.2f}%',
                        '5-Period': '{:+.2f}%',
                        '10-Period': '{:+.2f}%',
                        'Price': '₹{:.2f}'
                    }).background_gradient(subset=['Correlation'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Plot charts
                st.subheader("📉 Interactive Charts with Pattern Matches")
                fig = plot_comprehensive_charts(df, ticker_symbol, matches_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart explanation
                with st.expander("📖 Chart Explanation"):
                    st.markdown("""
                    **Candlestick Chart:** Shows price movements (open, high, low, close). Green candles = price up, Red candles = price down.
                    Blue annotations mark historically similar patterns and their outcomes.
                    
                    **Volume:** Trading volume intensity. Higher bars indicate stronger market participation.
                    
                    **RSI (Relative Strength Index):** Momentum indicator (0-100). Above 70 = overbought, Below 30 = oversold.
                    
                    **Returns %:** Period-over-period percentage change in price.
                    """)
            else:
                st.warning("Not enough data to find similar patterns. Try a longer period.")
    
    with tab2:
        st.markdown('<p class="sub-header">Returns & Volatility Analysis</p>', unsafe_allow_html=True)
        
        analysis_text, df_with_returns = analyze_returns_heatmap(df, interval)
        st.markdown(analysis_text)
        
        # Monthly returns heatmap
        if len(df) > 60:
            st.subheader("📅 Monthly Returns Heatmap")
            
            df_pivot = df_with_returns.copy()
            df_pivot['Month'] = df_pivot.index.month
            df_pivot['Year'] = df_pivot.index.year
            
            pivot_table = df_pivot.groupby(['Year', 'Month'])['Returns'].sum().reset_index()
            pivot_table = pivot_table.pivot(index='Year', columns='Month', values='Returns')
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='RdYlGn',
                zmid=0,
                text=pivot_table.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                colorbar=dict(title="Returns %")
            ))
            
            fig_heatmap.update_layout(
                title="Monthly Returns Heatmap",
                xaxis_title="Month",
                yaxis_title="Year",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Distribution of returns
        st.subheader("📊 Returns Distribution")
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_with_returns['Returns'].dropna(),
            nbinsx=50,
            name='Returns',
            marker_color='#1f77b4'
        ))
        
        fig_dist.update_layout(
            title="Distribution of Returns",
            xaxis_title="Returns %",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab3:
        st.markdown('<p class="sub-header">Ratio Analysis</p>', unsafe_allow_html=True)
        
        if enable_ratio and ticker2_symbol:
            with st.spinner(f"📥 Fetching data for {ticker2_symbol}..."):
                try:
                    ticker2 = yf.Ticker(ticker2_symbol)
                    df2 = ticker2.history(period=period, interval=interval)
                    df2 = convert_to_ist(df2)
                    
                    if not df2.empty:
                        ratio_summary, ratio_df, bin_analysis = create_ratio_analysis(df, df2, ticker_symbol, ticker2_symbol)
                        
                        st.markdown(ratio_summary)
                        
                        st.divider()
                        
                        # Display bin analysis
                        st.subheader("📊 Ratio Bins Performance")
                        
                        bin_display = bin_analysis.copy()
                        bin_display.columns = ['_'.join(col).strip() for col in bin_display.columns.values]
                        bin_display = bin_display.reset_index()
                        
                        st.dataframe(
                            bin_display.style.format({
                                'Returns_mean': '{:+.2f}%',
                                'Returns_std': '{:.2f}%',
                                'Returns_count': '{:.0f}',
                                'Ratio_min': '{:.4f}',
                                'Ratio_max': '{:.4f}',
                                'Ratio_mean': '{:.4f}'
                            }).background_gradient(subset=['Returns_mean'], cmap='RdYlGn'),
                            use_container_width=True
                        )
                        
                        # Plot ratio chart
                        st.subheader("📈 Ratio Charts")
                        fig_ratio = plot_ratio_chart(ratio_df, ticker_symbol, ticker2_symbol)
                        st.plotly_chart(fig_ratio, use_container_width=True)
                        
                    else:
                        st.error(f"No data found for {ticker2_symbol}")
                
                except Exception as e:
                    st.error(f"Error in ratio analysis: {str(e)}")
        else:
            st.info("👈 Enable ratio analysis in the sidebar and select a second ticker to compare.")
    
    with tab4:
        st.markdown('<p class="sub-header">RSI Analysis & Bins</p>', unsafe_allow_html=True)
        
        rsi_summary, rsi_analysis = create_rsi_analysis(df)
        
        st.markdown(rsi_summary)
        
        st.divider()
        
        # Display RSI bin analysis
        st.subheader("📊 RSI Bins Historical Performance")
        
        rsi_display = rsi_analysis.reset_index()
        rsi_display.columns = ['RSI Zone', 'Avg 5-Period Return %', 'Volatility %', 'Count']
        
        st.dataframe(
            rsi_display.style.format({
                'Avg 5-Period Return %': '{:+.2f}',
                'Volatility %': '{:.2f}',
                'Count': '{:.0f}'
            }).background_gradient(subset=['Avg 5-Period Return %'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # RSI history chart
        st.subheader("📉 RSI History")
        
        df_rsi = df.copy()
        df_rsi['RSI'] = calculate_rsi(df_rsi['Close'])
        
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df_rsi.index,
            y=df_rsi['RSI'],
            name='RSI',
            line=dict(color='#9467bd', width=2),
            fill='tozeroy'
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
        
        fig_rsi.update_layout(
            title="RSI (14) Timeline",
            xaxis_title="Date",
            yaxis_title="RSI",
            height=400,
            yaxis_range=[0, 100]
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # RSI divergence detection
        with st.expander("🔍 Understanding RSI Signals"):
            st.markdown("""
            **RSI (Relative Strength Index)** measures momentum on a scale of 0-100:
            
            - **RSI > 70:** Overbought zone - potential sell signal or correction expected
            - **RSI < 30:** Oversold zone - potential buy signal or reversal expected
            - **RSI 40-60:** Neutral zone - no strong signals
            - **RSI Divergence:** When price makes new highs/lows but RSI doesn't - signals potential reversal
            
            **Trading Strategy:**
            - Buy when RSI crosses above 30 from oversold
            - Sell when RSI crosses below 70 from overbought
            - Use with other indicators for confirmation
            """)
    
    with tab5:
        st.markdown('<p class="sub-header">Data Export & Statistics</p>', unsafe_allow_html=True)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Summary Statistics")
            
            stats_data = {
                'Metric': ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility'],
                'Latest': [
                    f"₹{df['Open'].iloc[-1]:.2f}",
                    f"₹{df['High'].iloc[-1]:.2f}",
                    f"₹{df['Low'].iloc[-1]:.2f}",
                    f"₹{df['Close'].iloc[-1]:.2f}",
                    f"{df['Volume'].iloc[-1]:,.0f}",
                    f"{df['Close'].pct_change().std() * 100:.2f}%"
                ],
                'Mean': [
                    f"₹{df['Open'].mean():.2f}",
                    f"₹{df['High'].mean():.2f}",
                    f"₹{df['Low'].mean():.2f}",
                    f"₹{df['Close'].mean():.2f}",
                    f"{df['Volume'].mean():,.0f}",
                    "-"
                ],
                'Max': [
                    f"₹{df['Open'].max():.2f}",
                    f"₹{df['High'].max():.2f}",
                    f"₹{df['Low'].max():.2f}",
                    f"₹{df['Close'].max():.2f}",
                    f"{df['Volume'].max():,.0f}",
                    "-"
                ],
                'Min': [
                    f"₹{df['Open'].min():.2f}",
                    f"₹{df['High'].min():.2f}",
                    f"₹{df['Low'].min():.2f}",
                    f"₹{df['Close'].min():.2f}",
                    f"{df['Volume'].min():,.0f}",
                    "-"
                ]
            }
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📈 Performance Metrics")
            
            total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            max_price = df['High'].max()
            min_price = df['Low'].min()
            avg_volume = df['Volume'].mean()
            
            perf_data = {
                'Metric': ['Total Return', 'Price Range', 'Avg Daily Volume', 'Data Points', 'Start Date', 'End Date'],
                'Value': [
                    f"{total_return:+.2f}%",
                    f"₹{min_price:.2f} - ₹{max_price:.2f}",
                    f"{avg_volume:,.0f}",
                    f"{len(df):,}",
                    df.index[0].strftime('%Y-%m-%d %H:%M'),
                    df.index[-1].strftime('%Y-%m-%d %H:%M')
                ]
            }
            
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Data preview
        st.subheader("📋 Raw Data Preview")
        
        display_df = df.copy()
        display_df['Returns %'] = display_df['Close'].pct_change() * 100
        display_df['RSI'] = calculate_rsi(display_df['Close'])
        
        st.dataframe(
            display_df.tail(50).style.format({
                'Open': '₹{:.2f}',
                'High': '₹{:.2f}',
                'Low': '₹{:.2f}',
                'Close': '₹{:.2f}',
                'Volume': '{:,.0f}',
                'Returns %': '{:+.2f}',
                'RSI': '{:.2f}'
            }),
            use_container_width=True
        )
        
        st.divider()
        
        # Export options
        st.subheader("💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = display_df.to_csv(index=True)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{ticker_symbol}_{interval}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel export
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                display_df.to_excel(writer, sheet_name='OHLCV Data', index=True)
                
                # Add summary sheet
                summary_df = pd.DataFrame({
                    'Metric': ['Ticker', 'Interval', 'Period', 'Total Return %', 'Volatility %', 'Max Price', 'Min Price'],
                    'Value': [
                        ticker_symbol,
                        interval,
                        period,
                        f"{total_return:.2f}",
                        f"{df['Close'].pct_change().std() * 100:.2f}",
                        f"{max_price:.2f}",
                        f"{min_price:.2f}"
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"{ticker_symbol}_{interval}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Final comprehensive recommendation
    st.divider()
    st.markdown('<p class="sub-header">🎯 Final Recommendation</p>', unsafe_allow_html=True)
    
    # Calculate all signals
    rsi_current = calculate_rsi(df['Close']).iloc[-1]
    price_ma20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Close'].mean()
    price_current = df['Close'].iloc[-1]
    
    # Pattern signal
    if st.session_state.get('data_fetched'):
        matches_df = find_pattern_matches(df, window_size=10, top_n=20)
        if not matches_df.empty:
            pattern_signal = matches_df['future_1d'].mean()
        else:
            pattern_signal = 0
    else:
        pattern_signal = 0
    
    # RSI signal
    if rsi_current < 30:
        rsi_signal = 1  # Buy
    elif rsi_current > 70:
        rsi_signal = -1  # Sell
    else:
        rsi_signal = 0  # Neutral
    
    # Trend signal
    trend_signal = 1 if price_current > price_ma20 else -1
    
    # Combined signal
    signals = []
    if pattern_signal > 1:
        signals.append("✅ Pattern analysis suggests upward movement")
    elif pattern_signal < -1:
        signals.append("⚠️ Pattern analysis suggests downward movement")
    
    if rsi_signal == 1:
        signals.append("✅ RSI indicates oversold - potential buy opportunity")
    elif rsi_signal == -1:
        signals.append("⚠️ RSI indicates overbought - potential sell opportunity")
    
    if trend_signal == 1:
        signals.append("✅ Price above 20-period moving average - uptrend")
    else:
        signals.append("⚠️ Price below 20-period moving average - downtrend")
    
    # Final recommendation
    combined_score = pattern_signal + (rsi_signal * 2) + trend_signal
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Key Signals:**")
        for signal in signals:
            st.markdown(f"- {signal}")
    
    with col2:
        if combined_score > 2:
            st.success("### 🟢 STRONG BUY")
            st.markdown("**Action:** Consider entering long position")
        elif combined_score > 0:
            st.info("### 🟢 BUY")
            st.markdown("**Action:** Favorable for long position")
        elif combined_score < -2:
            st.error("### 🔴 STRONG SELL")
            st.markdown("**Action:** Consider exiting or short position")
        elif combined_score < 0:
            st.warning("### 🔴 SELL")
            st.markdown("**Action:** Caution advised")
        else:
            st.warning("### 🟡 HOLD")
            st.markdown("**Action:** Wait for clearer signals")
    
    st.info("⚠️ **Disclaimer:** This is an algorithmic analysis tool for educational purposes. Always do your own research and consider multiple factors before making investment decisions. Past performance does not guarantee future results.")

else:
    # Welcome screen
    st.info("👆 Configure your analysis in the sidebar and click 'Fetch Data & Analyze' to begin.")
    
    st.markdown("""
    ## 🎯 Features
    
    ### 📊 Pattern Matching Algorithm
    - **Similarity Detection:** Uses Euclidean distance and correlation to find historical patterns similar to current market conditions
    - **Volatility Matching:** Considers volatility alongside price patterns for more accurate predictions
    - **Multi-Period Forecast:** Predicts movements for 1, 3, 5, and 10 periods ahead
    - **Visual Pattern Markers:** Highlights similar patterns on charts with their historical outcomes
    
    ### 📈 Comprehensive Analysis
    - **RSI Analysis:** Momentum-based signals with historical performance by RSI zones
    - **Returns Heatmaps:** Monthly and yearly returns visualization
    - **Ratio Analysis:** Compare two assets and find optimal ratio zones
    - **Volatility Tracking:** Identify high and low volatility periods
    
    ### 🎯 Actionable Recommendations
    - **Buy/Sell/Hold Signals:** Clear recommendations based on multiple indicators
    - **Confidence Scores:** Probability-based forecasts from historical patterns
    - **Risk Assessment:** Volatility and drawdown analysis
    
    ### 💾 Data Management
    - **Export to CSV/Excel:** Download complete OHLCV data with calculations
    - **IST Timezone:** All data automatically converted to Indian Standard Time
    - **Rate Limit Protection:** Smart caching to avoid API limits
    
    ## 🚀 How It Works
    
    1. **Fetch Data:** Select asset, timeframe, and period
    2. **Pattern Search:** Algorithm scans entire history for similar patterns
    3. **Forecast Generation:** Predicts future movements based on historical outcomes
    4. **Multi-Factor Analysis:** Combines patterns, RSI, trends, and volatility
    5. **Clear Recommendation:** Provides actionable buy/sell/hold signals
    
    ## 📚 Supported Assets
    - Indian Indices (Nifty 50, Bank Nifty, Sensex)
    - Cryptocurrencies (BTC, ETH)
    - Commodities (Gold, Silver)
    - Forex (USD/INR, EUR/USD)
    - Custom stocks via ticker symbols
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>🤖 Advanced Trading Pattern Analyzer | Built with Streamlit & yfinance</p>
    <p>⚡ Real-time pattern matching • 📊 Historical backtesting • 🎯 Actionable insights</p>
</div>
""", unsafe_allow_html=True)
