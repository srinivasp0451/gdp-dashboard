import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import calendar

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
    .bullish-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .bearish-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .neutral-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
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
    
    if len(price) < window * 4:
        return bullish_divs, bearish_divs
    
    try:
        for i in range(window, len(price) - window):
            # Price local minima
            if price.iloc[i] == price.iloc[i-window:i+window+1].min():
                # Check if RSI is making higher lows
                for j in range(max(window, i-window*3), i):
                    if j < len(price) and price.iloc[j] == price.iloc[max(0, j-window):min(len(price), j+window+1)].min():
                        if j < len(rsi) and i < len(rsi) and price.iloc[i] < price.iloc[j] and rsi.iloc[i] > rsi.iloc[j]:
                            bullish_divs.append((price.index[j], price.index[i], price.iloc[j], price.iloc[i]))
                            break
            
            # Price local maxima
            if price.iloc[i] == price.iloc[i-window:i+window+1].max():
                # Check if RSI is making lower highs
                for j in range(max(window, i-window*3), i):
                    if j < len(price) and price.iloc[j] == price.iloc[max(0, j-window):min(len(price), j+window+1)].max():
                        if j < len(rsi) and i < len(rsi) and price.iloc[i] > price.iloc[j] and rsi.iloc[i] < rsi.iloc[j]:
                            bearish_divs.append((price.index[j], price.index[i], price.iloc[j], price.iloc[i]))
                            break
    except Exception as e:
        st.warning(f"Divergence detection warning: {str(e)}")
    
    return bullish_divs, bearish_divs

def fetch_data(symbol, period, interval):
    """Fetch data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None
        # Remove timezone info to avoid Excel export issues
        df.index = df.index.tz_localize(None)
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

def create_returns_heatmap(data, timeframe):
    """Create returns heatmap based on timeframe"""
    df = data.copy()
    df['Returns'] = df['Close'].pct_change() * 100
    
    heatmaps = []
    
    try:
        # Daily returns by Month vs Day
        if timeframe in ['1d', '5d', '1wk', '1mo']:
            df['Year'] = df.index.year
            df['Month'] = df.index.month
            df['Day'] = df.index.day
            
            for year in df['Year'].unique():
                year_data = df[df['Year'] == year]
                pivot = year_data.pivot_table(values='Returns', index='Day', columns='Month', aggfunc='sum')
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[calendar.month_abbr[i] for i in pivot.columns],
                    y=pivot.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=np.round(pivot.values, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Returns %")
                ))
                
                fig.update_layout(
                    title=f'Daily Returns Heatmap - {year} (Day vs Month)',
                    xaxis_title='Month',
                    yaxis_title='Day of Month',
                    height=500
                )
                heatmaps.append(('day_month', year, fig))
        
        # Weekly returns by Month
        df['Week'] = df.index.isocalendar().week
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        
        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            pivot = year_data.pivot_table(values='Returns', index='Week', columns='Month', aggfunc='sum')
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=[calendar.month_abbr[i] for i in pivot.columns] if len(pivot.columns) > 0 else [],
                y=pivot.index,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(pivot.values, 2),
                texttemplate='%{text}%',
                textfont={"size": 10},
                colorbar=dict(title="Returns %")
            ))
            
            fig.update_layout(
                title=f'Weekly Returns Heatmap - {year} (Week vs Month)',
                xaxis_title='Month',
                yaxis_title='Week of Year',
                height=500
            )
            heatmaps.append(('week_month', year, fig))
        
        # Monthly returns by Year
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        
        pivot = df.pivot_table(values='Returns', index='Month', columns='Year', aggfunc='sum')
        
        if not pivot.empty:
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=[calendar.month_abbr[i] for i in pivot.index],
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(pivot.values, 2),
                texttemplate='%{text}%',
                textfont={"size": 12},
                colorbar=dict(title="Returns %")
            ))
            
            fig.update_layout(
                title='Monthly Returns Heatmap (Month vs Year)',
                xaxis_title='Year',
                yaxis_title='Month',
                height=500
            )
            heatmaps.append(('month_year', 'all', fig))
            
    except Exception as e:
        st.warning(f"Could not generate all heatmaps: {str(e)}")
    
    return heatmaps

def generate_market_prediction(metrics, bullish_divs, bearish_divs, rsi_current, rsi_overbought, rsi_oversold):
    """Generate market prediction based on analysis"""
    signals = []
    bullish_score = 0
    bearish_score = 0
    
    # RSI Analysis
    if rsi_current > rsi_overbought:
        signals.append(f"🔴 RSI at {rsi_current:.1f} is OVERBOUGHT (>{rsi_overbought}) - Bearish signal")
        bearish_score += 2
    elif rsi_current < rsi_oversold:
        signals.append(f"🟢 RSI at {rsi_current:.1f} is OVERSOLD (<{rsi_oversold}) - Bullish signal")
        bullish_score += 2
    elif rsi_current > 50:
        signals.append(f"🟡 RSI at {rsi_current:.1f} shows bullish momentum")
        bullish_score += 1
    else:
        signals.append(f"🟡 RSI at {rsi_current:.1f} shows bearish momentum")
        bearish_score += 1
    
    # Divergence Analysis
    if len(bullish_divs) > len(bearish_divs):
        signals.append(f"🟢 {len(bullish_divs)} Bullish divergence(s) detected - Potential reversal UP")
        bullish_score += len(bullish_divs) * 2
    elif len(bearish_divs) > len(bullish_divs):
        signals.append(f"🔴 {len(bearish_divs)} Bearish divergence(s) detected - Potential reversal DOWN")
        bearish_score += len(bearish_divs) * 2
    elif len(bullish_divs) == len(bearish_divs) and len(bullish_divs) > 0:
        signals.append(f"🟡 Equal divergences ({len(bullish_divs)} each) - Mixed signals")
    
    # Price momentum
    if metrics['daily_change_pct'] > 2:
        signals.append(f"🟢 Strong daily gain of {metrics['daily_change_pct']:.2f}% - Bullish momentum")
        bullish_score += 1
    elif metrics['daily_change_pct'] < -2:
        signals.append(f"🔴 Strong daily loss of {metrics['daily_change_pct']:.2f}% - Bearish momentum")
        bearish_score += 1
    
    # Period performance
    if metrics['total_change_pct'] > 5:
        signals.append(f"🟢 Period gain of {metrics['total_change_pct']:.2f}% - Strong uptrend")
        bullish_score += 1
    elif metrics['total_change_pct'] < -5:
        signals.append(f"🔴 Period loss of {metrics['total_change_pct']:.2f}% - Strong downtrend")
        bearish_score += 1
    
    # Final prediction
    if bullish_score > bearish_score + 2:
        prediction = "📈 BULLISH"
        confidence = min(90, 50 + (bullish_score - bearish_score) * 10)
        color_class = "bullish-box"
    elif bearish_score > bullish_score + 2:
        prediction = "📉 BEARISH"
        confidence = min(90, 50 + (bearish_score - bullish_score) * 10)
        color_class = "bearish-box"
    else:
        prediction = "➡️ NEUTRAL/SIDEWAYS"
        confidence = 50
        color_class = "neutral-box"
    
    return prediction, confidence, signals, color_class

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
        # Calculate RSI
        data['RSI'] = calculate_rsi(data, rsi_period)
        rsi_current = data['RSI'].iloc[-1]
        
        # Find divergences
        bullish_divs, bearish_divs = find_divergences(data['Close'], data['RSI'])
        
        # Generate market prediction
        prediction, confidence, signals, color_class = generate_market_prediction(
            metrics, bullish_divs, bearish_divs, rsi_current, rsi_overbought, rsi_oversold
        )
        
        # Display prediction box
        st.markdown(f"""
        <div class="{color_class}">
            <h2>🎯 MARKET PREDICTION: {prediction}</h2>
            <h3>Confidence Level: {confidence}%</h3>
            <h4>Analysis Signals:</h4>
            <ul>
                {''.join([f'<li>{signal}</li>' for signal in signals])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display metrics
        st.subheader(f"📊 {ticker_symbol} - Summary Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"₹{metrics['current_price']:.2f}", 
                     f"{metrics['daily_change']:.2f} ({metrics['daily_change_pct']:.2f}%)")
        
        with col2:
            st.metric("Period Change", f"{metrics['total_change']:.2f}", 
                     f"{metrics['total_change_pct']:.2f}%")
        
        with col3:
            st.metric("Period High", f"₹{metrics['high']:.2f}")
        
        with col4:
            st.metric("Period Low", f"₹{metrics['low']:.2f}")
        
        with col5:
            st.metric("Range", f"₹{metrics['range']:.2f}")
        
        # Metrics prediction insight
        if metrics['current_price'] > (metrics['high'] + metrics['low']) / 2:
            st.info(f"💡 **Price Position:** Trading in upper half of range - Shows strength. If breaks {metrics['high']:.2f}, expect continuation UP.")
        else:
            st.warning(f"💡 **Price Position:** Trading in lower half of range - Shows weakness. If breaks {metrics['low']:.2f}, expect continuation DOWN.")
        
        # Data table
        st.subheader("📋 Recent Price Data (Last 50 bars)")
        display_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        display_df['Change'] = display_df['Close'].diff()
        display_df['Change %'] = display_df['Close'].pct_change() * 100
        
        # Color code the table
        def color_negative_red(val):
            if isinstance(val, (int, float)):
                color = 'green' if val >= 0 else 'red'
                return f'color: {color}'
            return ''
        
        st.dataframe(display_df.tail(50).style.applymap(color_negative_red, subset=['Change', 'Change %']), 
                    use_container_width=True, height=400)
        
        # Calculate ratio if comparison data exists
        comparison_data = None
        aligned_data = None
        if st.session_state.comparison_data is not None:
            comparison_data = st.session_state.comparison_data
            comparison_data['RSI'] = calculate_rsi(comparison_data, rsi_period)
            # Align dates
            aligned_data = pd.concat([data['Close'], comparison_data['Close']], axis=1, join='inner')
            aligned_data.columns = ['Primary', 'Comparison']
            aligned_data['Ratio'] = aligned_data['Primary'] / aligned_data['Comparison']
            aligned_data['Ratio_RSI'] = calculate_rsi(aligned_data[['Ratio']].rename(columns={'Ratio': 'Close'}), rsi_period)
            
            # Get comparison metrics
            comp_metrics = calculate_metrics(comparison_data)
            comp_rsi_current = comparison_data['RSI'].iloc[-1]
            comp_bullish_divs, comp_bearish_divs = find_divergences(comparison_data['Close'], comparison_data['RSI'])
            
            # Ratio analysis
            ratio_current = aligned_data['Ratio'].iloc[-1]
            ratio_change = ((ratio_current - aligned_data['Ratio'].iloc[0]) / aligned_data['Ratio'].iloc[0]) * 100
            ratio_rsi_current = aligned_data['Ratio_RSI'].iloc[-1]
            ratio_bullish_divs, ratio_bearish_divs = find_divergences(aligned_data['Ratio'], aligned_data['Ratio_RSI'].dropna())
        
        # Create charts
        st.subheader("📈 Technical Analysis Charts")
        
        # Chart 1: Ticker 1 Price Chart
        st.markdown(f"### 1️⃣ {ticker_symbol} - Price Chart")
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Add divergence lines
        for div in bullish_divs:
            fig1.add_trace(go.Scatter(
                x=[div[0], div[1]],
                y=[div[2], div[3]],
                mode='lines',
                line=dict(color='green', width=3, dash='dash'),
                showlegend=False,
                hovertemplate='Bullish Divergence<extra></extra>'
            ))
        
        for div in bearish_divs:
            fig1.add_trace(go.Scatter(
                x=[div[0], div[1]],
                y=[div[2], div[3]],
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                showlegend=False,
                hovertemplate='Bearish Divergence<extra></extra>'
            ))
        
        fig1.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Prediction for Chart 1
        if len(bullish_divs) > 0:
            st.success(f"✅ **{ticker_symbol} PREDICTION:** {len(bullish_divs)} bullish divergence(s) detected. Price likely to REVERSE UP from current levels.")
        if len(bearish_divs) > 0:
            st.error(f"⚠️ **{ticker_symbol} PREDICTION:** {len(bearish_divs)} bearish divergence(s) detected. Price likely to REVERSE DOWN from current levels.")
        if len(bullish_divs) == 0 and len(bearish_divs) == 0:
            st.info(f"ℹ️ **{ticker_symbol} PREDICTION:** No divergences detected. Follow the current trend direction.")
        
        # Chart 2: Ratio Chart (if comparison exists)
        if aligned_data is not None:
            st.markdown(f"### 2️⃣ Ratio Chart: {ticker_symbol}/{comparison_symbol}")
            
            fig2 = go.Figure()
            
            # Create candlestick-like representation for ratio
            fig2.add_trace(go.Scatter(
                x=aligned_data.index,
                y=aligned_data['Ratio'],
                mode='lines',
                name='Ratio',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.1)'
            ))
            
            # Add divergences
            for div in ratio_bullish_divs:
                fig2.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[div[2], div[3]],
                    mode='lines',
                    line=dict(color='green', width=3, dash='dash'),
                    showlegend=False
                ))
            
            for div in ratio_bearish_divs:
                fig2.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[div[2], div[3]],
                    mode='lines',
                    line=dict(color='red', width=3, dash='dash'),
                    showlegend=False
                ))
            
            fig2.update_layout(
                height=500,
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )
            
            fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Prediction for Ratio
            if ratio_change > 0:
                st.success(f"✅ **RATIO PREDICTION:** {ticker_symbol} is OUTPERFORMING {comparison_symbol} by {ratio_change:.2f}%. Continue favoring {ticker_symbol}.")
            else:
                st.error(f"⚠️ **RATIO PREDICTION:** {ticker_symbol} is UNDERPERFORMING {comparison_symbol} by {abs(ratio_change):.2f}%. Consider switching to {comparison_symbol}.")
            
            if len(ratio_bullish_divs) > 0:
                st.success(f"✅ **RATIO DIVERGENCE:** {len(ratio_bullish_divs)} bullish divergence(s). {ticker_symbol} likely to start OUTPERFORMING.")
            if len(ratio_bearish_divs) > 0:
                st.error(f"⚠️ **RATIO DIVERGENCE:** {len(ratio_bearish_divs)} bearish divergence(s). {ticker_symbol} likely to start UNDERPERFORMING.")
        
        # Chart 3: RSI Chart
        st.markdown(f"### 3️⃣ {ticker_symbol} - RSI Indicator")
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(100, 149, 237, 0.2)'
        ))
        
        fig3.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", 
                     annotation_text="Overbought")
        fig3.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", 
                     annotation_text="Oversold")
        fig3.add_hline(y=50, line_dash="dot", line_color="gray")
        
        # Add RSI divergence lines
        for div in bullish_divs:
            if div[0] in data.index and div[1] in data.index:
                fig3.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[data.loc[div[0], 'RSI'], data.loc[div[1], 'RSI']],
                    mode='lines',
                    line=dict(color='green', width=3, dash='dash'),
                    showlegend=False
                ))
        
        for div in bearish_divs:
            if div[0] in data.index and div[1] in data.index:
                fig3.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[data.loc[div[0], 'RSI'], data.loc[div[1], 'RSI']],
                    mode='lines',
                    line=dict(color='red', width=3, dash='dash'),
                    showlegend=False
                ))
        
        fig3.update_layout(
            height=400,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            yaxis_title='RSI',
            yaxis_range=[0, 100]
        )
        
        fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # RSI Prediction
        if rsi_current > rsi_overbought:
            st.error(f"⚠️ **RSI PREDICTION:** RSI at {rsi_current:.1f} is OVERBOUGHT. High probability of price reversal DOWN. Consider taking profits or SHORT positions.")
        elif rsi_current < rsi_oversold:
            st.success(f"✅ **RSI PREDICTION:** RSI at {rsi_current:.1f} is OVERSOLD. High probability of price reversal UP. Consider BUYING or LONG positions.")
        elif rsi_current > 50:
            st.info(f"📊 **RSI PREDICTION:** RSI at {rsi_current:.1f} shows bullish momentum. Continue holding LONG positions until RSI reaches overbought zone.")
        else:
            st.warning(f"📊 **RSI PREDICTION:** RSI at {rsi_current:.1f} shows bearish momentum. Avoid LONG positions, consider SHORT if breaks key support.")
        
        # Chart 4: Ratio RSI (if comparison exists)
        if aligned_data is not None:
            st.markdown(f"### 4️⃣ Ratio RSI: {ticker_symbol}/{comparison_symbol}")
            
            fig4 = go.Figure()
            
            fig4.add_trace(go.Scatter(
                x=aligned_data.index,
                y=aligned_data['Ratio_RSI'],
                mode='lines',
                name='Ratio RSI',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.2)'
            ))
            
            fig4.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", 
                         annotation_text="Overbought")
            fig4.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", 
                         annotation_text="Oversold")
            fig4.add_hline(y=50, line_dash="dot", line_color="gray")
            
            fig4.update_layout(
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                yaxis_title='Ratio RSI',
                yaxis_range=[0, 100]
            )
            
            fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            st.plotly_chart(fig4, use_container_width=True)
            
            # Ratio RSI Prediction
            if ratio_rsi_current > rsi_overbought:
                st.error(f"⚠️ **RATIO RSI PREDICTION:** Ratio RSI at {ratio_rsi_current:.1f} is OVERBOUGHT. {ticker_symbol} has outperformed too much. Expect mean reversion - {comparison_symbol} likely to catch up. Consider SWITCHING to {comparison_symbol}.")
            elif ratio_rsi_current < rsi_oversold:
                st.success(f"✅ **RATIO RSI PREDICTION:** Ratio RSI at {ratio_rsi_current:.1f} is OVERSOLD. {ticker_symbol} has underperformed too much. Expect mean reversion - {ticker_symbol} likely to outperform. BUY {ticker_symbol} over {comparison_symbol}.")
            elif ratio_rsi_current > 50:
                st.info(f"📊 **RATIO RSI PREDICTION:** Ratio RSI at {ratio_rsi_current:.1f} shows {ticker_symbol} is in strong relative uptrend. Continue favoring {ticker_symbol}.")
            else:
                st.warning(f"📊 **RATIO RSI PREDICTION:** Ratio RSI at {ratio_rsi_current:.1f} shows {comparison_symbol} is in strong relative uptrend. Continue favoring {comparison_symbol}.")
        
        # Chart 5: Comparison Ticker Price (if exists)
        if comparison_data is not None:
            st.markdown(f"### 5️⃣ {comparison_symbol} - Price Chart")
            
            fig5 = go.Figure()
            
            fig5.add_trace(go.Candlestick(
                x=comparison_data.index,
                open=comparison_data['Open'],
                high=comparison_data['High'],
                low=comparison_data['Low'],
                close=comparison_data['Close'],
                name='Price'
            ))
            
            # Add divergence lines
            for div in comp_bullish_divs:
                fig5.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[div[2], div[3]],
                    mode='lines',
                    line=dict(color='green', width=3, dash='dash'),
                    showlegend=False,
                    hovertemplate='Bullish Divergence<extra></extra>'
                ))
            
            for div in comp_bearish_divs:
                fig5.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[div[2], div[3]],
                    mode='lines',
                    line=dict(color='red', width=3, dash='dash'),
                    showlegend=False,
                    hovertemplate='Bearish Divergence<extra></extra>'
                ))
            
            fig5.update_layout(
                height=500,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )
            
            fig5.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig5.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            st.plotly_chart(fig5, use_container_width=True)
            
            # Prediction for Comparison Chart
            if len(comp_bullish_divs) > 0:
                st.success(f"✅ **{comparison_symbol} PREDICTION:** {len(comp_bullish_divs)} bullish divergence(s) detected. Price likely to REVERSE UP from current levels.")
            if len(comp_bearish_divs) > 0:
                st.error(f"⚠️ **{comparison_symbol} PREDICTION:** {len(comp_bearish_divs)} bearish divergence(s) detected. Price likely to REVERSE DOWN from current levels.")
            if len(comp_bullish_divs) == 0 and len(comp_bearish_divs) == 0:
                st.info(f"ℹ️ **{comparison_symbol} PREDICTION:** No divergences detected. Follow the current trend direction.")
        
        # Chart 6: Comparison RSI (if exists)
        if comparison_data is not None:
            st.markdown(f"### 6️⃣ {comparison_symbol} - RSI Indicator")
            
            fig6 = go.Figure()
            
            fig6.add_trace(go.Scatter(
                x=comparison_data.index,
                y=comparison_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='orange', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.2)'
            ))
            
            fig6.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", 
                         annotation_text="Overbought")
            fig6.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", 
                         annotation_text="Oversold")
            fig6.add_hline(y=50, line_dash="dot", line_color="gray")
            
            # Add RSI divergence lines
            for div in comp_bullish_divs:
                if div[0] in comparison_data.index and div[1] in comparison_data.index:
                    fig6.add_trace(go.Scatter(
                        x=[div[0], div[1]],
                        y=[comparison_data.loc[div[0], 'RSI'], comparison_data.loc[div[1], 'RSI']],
                        mode='lines',
                        line=dict(color='green', width=3, dash='dash'),
                        showlegend=False
                    ))
            
            for div in comp_bearish_divs:
                if div[0] in comparison_data.index and div[1] in comparison_data.index:
                    fig6.add_trace(go.Scatter(
                        x=[div[0], div[1]],
                        y=[comparison_data.loc[div[0], 'RSI'], comparison_data.loc[div[1], 'RSI']],
                        mode='lines',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
            
            fig6.update_layout(
                height=400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                yaxis_title='RSI',
                yaxis_range=[0, 100]
            )
            
            fig6.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig6.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            st.plotly_chart(fig6, use_container_width=True)
            
            # Comparison RSI Prediction
            if comp_rsi_current > rsi_overbought:
                st.error(f"⚠️ **{comparison_symbol} RSI PREDICTION:** RSI at {comp_rsi_current:.1f} is OVERBOUGHT. High probability of price reversal DOWN. Consider taking profits or SHORT positions.")
            elif comp_rsi_current < rsi_oversold:
                st.success(f"✅ **{comparison_symbol} RSI PREDICTION:** RSI at {comp_rsi_current:.1f} is OVERSOLD. High probability of price reversal UP. Consider BUYING or LONG positions.")
            elif comp_rsi_current > 50:
                st.info(f"📊 **{comparison_symbol} RSI PREDICTION:** RSI at {comp_rsi_current:.1f} shows bullish momentum. Continue holding LONG positions.")
            else:
                st.warning(f"📊 **{comparison_symbol} RSI PREDICTION:** RSI at {comp_rsi_current:.1f} shows bearish momentum. Avoid LONG positions.")
        
        # Divergence Summary
        st.subheader("🔍 Divergence Analysis Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bullish Divergences", len(bullish_divs), help="Price making lower lows while RSI makes higher lows")
            if len(bullish_divs) > 0:
                st.success(f"**PREDICTION:** {len(bullish_divs)} bullish signal(s) suggest UPWARD reversal imminent. Look for entry on next pullback.")
        
        with col2:
            st.metric("Bearish Divergences", len(bearish_divs), delta_color="inverse", help="Price making higher highs while RSI makes lower highs")
            if len(bearish_divs) > 0:
                st.error(f"**PREDICTION:** {len(bearish_divs)} bearish signal(s) suggest DOWNWARD reversal imminent. Consider exit or SHORT entry.")
        
        # Returns Heatmaps
        st.subheader("🔥 Returns Heatmap Analysis")
        st.markdown("**PREDICTION INSIGHT:** Heatmaps reveal seasonal patterns. Green zones = historically profitable periods. Red zones = historically loss periods. Use this to time entries/exits.")
        
        with st.spinner("Generating heatmaps..."):
            heatmaps = create_returns_heatmap(data, timeframe)
            
            for hm_type, year, fig in heatmaps:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add prediction based on heatmap type
                if hm_type == 'day_month':
                    st.info(f"**PREDICTION ({year}):** Look for recurring green days/months for HIGH probability trades. Avoid red zones unless strong reversal signals present.")
                elif hm_type == 'week_month':
                    st.info(f"**PREDICTION ({year}):** Weekly patterns show best times to enter. Green weeks = favorable for LONG positions. Red weeks = favorable for SHORT positions.")
                elif hm_type == 'month_year':
                    st.info("**PREDICTION:** Monthly patterns across years reveal seasonality. Consistently green months = HIGH confidence LONG entries. Consistently red months = HIGH confidence SHORT entries.")
        
        # Final Trading Recommendation
        st.subheader("🎯 FINAL TRADING RECOMMENDATION")
        
        if prediction == "📈 BULLISH":
            st.markdown(f"""
            <div class="bullish-box">
                <h3>🟢 ACTION: BUY / GO LONG on {ticker_symbol}</h3>
                <h4>Confidence: {confidence}%</h4>
                <h4>Strategy:</h4>
                <ul>
                    <li><strong>Entry:</strong> Current price ₹{metrics['current_price']:.2f} or on minor pullback</li>
                    <li><strong>Stop Loss:</strong> Below ₹{metrics['low']:.2f} (period low)</li>
                    <li><strong>Target 1:</strong> ₹{metrics['current_price'] * 1.03:.2f} (+3%)</li>
                    <li><strong>Target 2:</strong> ₹{metrics['current_price'] * 1.05:.2f} (+5%)</li>
                    <li><strong>Target 3:</strong> Above ₹{metrics['high']:.2f} (period high breakout)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif prediction == "📉 BEARISH":
            st.markdown(f"""
            <div class="bearish-box">
                <h3>🔴 ACTION: SELL / GO SHORT on {ticker_symbol}</h3>
                <h4>Confidence: {confidence}%</h4>
                <h4>Strategy:</h4>
                <ul>
                    <li><strong>Entry:</strong> Current price ₹{metrics['current_price']:.2f} or on minor bounce</li>
                    <li><strong>Stop Loss:</strong> Above ₹{metrics['high']:.2f} (period high)</li>
                    <li><strong>Target 1:</strong> ₹{metrics['current_price'] * 0.97:.2f} (-3%)</li>
                    <li><strong>Target 2:</strong> ₹{metrics['current_price'] * 0.95:.2f} (-5%)</li>
                    <li><strong>Target 3:</strong> Below ₹{metrics['low']:.2f} (period low breakdown)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="neutral-box">
                <h3>🟡 ACTION: STAY NEUTRAL / WAIT on {ticker_symbol}</h3>
                <h4>Confidence: {confidence}%</h4>
                <h4>Strategy:</h4>
                <ul>
                    <li><strong>No Clear Direction:</strong> Mixed signals detected</li>
                    <li><strong>Wait for:</strong> Break above ₹{metrics['high']:.2f} (GO LONG) or below ₹{metrics['low']:.2f} (GO SHORT)</li>
                    <li><strong>Range:</strong> Price consolidating between ₹{metrics['low']:.2f} - ₹{metrics['high']:.2f}</li>
                    <li><strong>Alternative:</strong> Trade the range - BUY near ₹{metrics['low']:.2f}, SELL near ₹{metrics['high']:.2f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Download options
        st.subheader("💾 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download - only OHLCV
            export_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            export_df.index.name = 'Date'
            csv = export_df.to_csv()
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{ticker_symbol}_{period}_{timeframe}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel download - only OHLCV
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                export_df.index.name = 'Date'
                export_df.to_excel(writer, sheet_name=f'{ticker_symbol}')
                
                if comparison_data is not None:
                    comp_export_df = comparison_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    comp_export_df.index.name = 'Date'
                    comp_export_df.to_excel(writer, sheet_name=f'{comparison_symbol}')
            
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
    - **Market Prediction Engine**: Get clear BUY/SELL/HOLD signals with confidence levels
    - **Multiple Assets**: NIFTY 50, Bank NIFTY, SENSEX, Crypto, Forex, Commodities, Stocks
    - **6 Technical Charts**: Price, Ratio, RSI with divergences marked
    - **Actionable Insights**: Entry/Exit/Stop Loss recommendations for every analysis
    - **Returns Heatmaps**: Identify seasonal patterns for better timing
    - **Export Data**: Download OHLCV data in CSV/Excel format
    
    ### 📌 How Predictions Work:
    1. **RSI Analysis**: Overbought (>70) = SELL signal, Oversold (<30) = BUY signal
    2. **Divergences**: Bullish divergences = Price reversal UP, Bearish = Price reversal DOWN
    3. **Momentum**: Recent price trends indicate short-term direction
    4. **Ratio Analysis**: Shows relative strength between two assets
    5. **Confidence Score**: Higher confidence = stronger signals across multiple indicators
    
    ### 🎯 Trading Recommendations Include:
    - Clear BUY/SELL/HOLD action
    - Specific entry prices
    - Stop loss levels
    - Multiple profit targets
    - Risk management guidelines
    """)

# Footer
st.markdown("---")
st.markdown("**⚠️ DISCLAIMER:** This is for educational purposes only. Always do your own research and consult financial advisors before trading.")
