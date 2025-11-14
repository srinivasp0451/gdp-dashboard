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
        st.markdown("**Key Insight:** Last 50 data points showing OHLCV with change metrics. Green indicates gains, red indicates losses.")
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
        
        # Calculate RSI
        data['RSI'] = calculate_rsi(data, rsi_period)
        
        # Calculate ratio if comparison data exists
        if st.session_state.comparison_data is not None:
            comparison_data = st.session_state.comparison_data
            comparison_data['RSI'] = calculate_rsi(comparison_data, rsi_period)
            # Align dates
            aligned_data = pd.concat([data['Close'], comparison_data['Close']], axis=1, join='inner')
            aligned_data.columns = ['Primary', 'Comparison']
            aligned_data['Ratio'] = aligned_data['Primary'] / aligned_data['Comparison']
            aligned_data['Ratio_RSI'] = calculate_rsi(aligned_data[['Ratio']].rename(columns={'Ratio': 'Close'}), rsi_period)
        else:
            comparison_data = None
            aligned_data = None
        
        # Find divergences
        bullish_divs, bearish_divs = find_divergences(data['Close'], data['RSI'])
        
        # Create plots
        st.subheader("📈 Technical Analysis Charts")
        st.markdown("""
        **Key Insights:** 
        - 🟢 **Green dashed lines** indicate bullish divergences (price makes lower lows while RSI makes higher lows - potential reversal up)
        - 🔴 **Red dashed lines** indicate bearish divergences (price makes higher highs while RSI makes lower highs - potential reversal down)
        - Divergences are powerful signals when combined with support/resistance levels
        """)
        
        # Chart 1: Price Chart with Divergences
        st.markdown("### 1️⃣ Price Chart with Divergence Analysis")
        st.markdown("**Insight:** Candlestick chart showing price action with bullish and bearish divergences marked. Watch for divergences near key support/resistance levels.")
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Add divergence lines on price chart
        for div in bullish_divs:
            fig1.add_trace(go.Scatter(
                x=[div[0], div[1]],
                y=[div[2], div[3]],
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                showlegend=False,
                hovertemplate='Bullish Divergence<extra></extra>'
            ))
        
        for div in bearish_divs:
            fig1.add_trace(go.Scatter(
                x=[div[0], div[1]],
                y=[div[2], div[3]],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
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
        
        # Chart 2: Comparison Price Chart with RSI (if comparison exists)
        if comparison_data is not None:
            st.markdown(f"### 2️⃣ Comparison: {ticker_symbol} vs {comparison_symbol}")
            st.markdown(f"**Insight:** Direct price comparison between {ticker_symbol} and {comparison_symbol}. Helps identify which asset is performing better.")
            
            comp_bullish, comp_bearish = find_divergences(comparison_data['Close'], comparison_data['RSI'])
            
            fig2 = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[f'{ticker_symbol} vs {comparison_symbol}', f'RSI Comparison'],
                row_heights=[0.6, 0.4]
            )
            
            # Primary ticker
            fig2.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=ticker_symbol,
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            # Comparison ticker
            fig2.add_trace(go.Scatter(
                x=comparison_data.index,
                y=comparison_data['Close'],
                mode='lines',
                name=comparison_symbol,
                line=dict(color='orange', width=2)
            ), row=1, col=1)
            
            # RSI for both
            fig2.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name=f'{ticker_symbol} RSI',
                line=dict(color='blue', width=2)
            ), row=2, col=1)
            
            fig2.add_trace(go.Scatter(
                x=comparison_data.index,
                y=comparison_data['RSI'],
                mode='lines',
                name=f'{comparison_symbol} RSI',
                line=dict(color='orange', width=2)
            ), row=2, col=1)
            
            fig2.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
            fig2.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
            fig2.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
            
            fig2.update_layout(
                height=700,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 3: Ratio Chart with RSI (if comparison exists)
        if aligned_data is not None:
            st.markdown(f"### 3️⃣ Ratio Analysis: {ticker_symbol}/{comparison_symbol}")
            st.markdown(f"**Insight:** Ratio chart shows relative strength. Rising ratio = {ticker_symbol} outperforming. Falling ratio = {comparison_symbol} outperforming. Divergences on ratio can signal momentum shifts.")
            
            # Find divergences on ratio
            ratio_bullish, ratio_bearish = find_divergences(aligned_data['Ratio'], aligned_data['Ratio_RSI'].dropna())
            
            fig3 = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=['Ratio Chart', 'Ratio RSI'],
                row_heights=[0.6, 0.4]
            )
            
            # Ratio chart
            fig3.add_trace(go.Scatter(
                x=aligned_data.index,
                y=aligned_data['Ratio'],
                mode='lines',
                name='Ratio',
                line=dict(color='purple', width=2)
            ), row=1, col=1)
            
            # Add divergences on ratio
            for div in ratio_bullish:
                fig3.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[div[2], div[3]],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=False
                ), row=1, col=1)
            
            for div in ratio_bearish:
                fig3.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[div[2], div[3]],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ), row=1, col=1)
            
            # Ratio RSI
            fig3.add_trace(go.Scatter(
                x=aligned_data.index,
                y=aligned_data['Ratio_RSI'],
                mode='lines',
                name='Ratio RSI',
                line=dict(color='purple', width=2)
            ), row=2, col=1)
            
            fig3.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
            fig3.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
            fig3.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
            
            fig3.update_layout(
                height=700,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Chart 4: Price RSI with Divergences
        st.markdown("### 4️⃣ RSI Analysis with Divergences")
        st.markdown(f"""
        **Insight:** RSI measures momentum. 
        - **Above {rsi_overbought}:** Overbought (potential reversal down)
        - **Below {rsi_oversold}:** Oversold (potential reversal up)
        - **Divergences:** Most reliable when RSI is in extreme zones
        """)
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(100, 149, 237, 0.2)'
        ))
        
        fig4.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", 
                     annotation_text="Overbought")
        fig4.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", 
                     annotation_text="Oversold")
        fig4.add_hline(y=50, line_dash="dot", line_color="gray")
        
        # Add RSI divergence lines
        for div in bullish_divs:
            if div[0] in data.index and div[1] in data.index:
                fig4.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[data.loc[div[0], 'RSI'], data.loc[div[1], 'RSI']],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=False
                ))
        
        for div in bearish_divs:
            if div[0] in data.index and div[1] in data.index:
                fig4.add_trace(go.Scatter(
                    x=[div[0], div[1]],
                    y=[data.loc[div[0], 'RSI'], data.loc[div[1], 'RSI']],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=False
                ))
        
        fig4.update_layout(
            height=400,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            yaxis_title='RSI',
            yaxis_range=[0, 100]
        )
        
        fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Divergence Summary
        st.subheader("🔍 Divergence Analysis Summary")
        st.markdown("**Insight:** Divergences are early warning signals. Multiple divergences in the same direction strengthen the signal. Always confirm with other indicators and price action.")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bullish Divergences", len(bullish_divs), delta="Potential Upside", help="Price making lower lows while RSI makes higher lows")
        
        with col2:
            st.metric("Bearish Divergences", len(bearish_divs), delta="Potential Downside", delta_color="inverse", help="Price making higher highs while RSI makes lower highs")
        
        # Returns Heatmaps
        st.subheader("🔥 Returns Heatmap Analysis")
        st.markdown("**Insight:** Heatmaps reveal seasonal patterns and performance trends. Green = positive returns, Red = negative returns. Look for recurring patterns in specific months or weeks.")
        
        with st.spinner("Generating heatmaps..."):
            heatmaps = create_returns_heatmap(data, timeframe)
            
            for hm_type, year, fig in heatmaps:
                st.plotly_chart(fig, use_container_width=True)
        
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
