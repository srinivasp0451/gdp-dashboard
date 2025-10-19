import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(page_title="Ratio Chart Analyzer", layout="wide")

# Custom CSS for better visibility
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for caching
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# RSI Calculation
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Flatten multi-index DataFrame
def flatten_dataframe(df):
    """Flatten multi-index columns from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
    # If columns still have ticker names, remove them
    df.columns = [col.split('_')[-1] if '_' in str(col) else col for col in df.columns]
    return df

# Fetch data with caching mechanism
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data_cached(ticker, period, interval):
    cache_key = f"{ticker}_{period}_{interval}"
    current_time = time.time()
    
    # Check if data exists in session state and is recent (within 5 minutes)
    if cache_key in st.session_state.cached_data:
        if cache_key in st.session_state.last_fetch_time:
            time_diff = current_time - st.session_state.last_fetch_time[cache_key]
            if time_diff < 300:  # 5 minutes
                return st.session_state.cached_data[cache_key]
    
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if not data.empty:
            # Flatten multi-index columns
            data = flatten_dataframe(data)
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                st.error(f"Missing required columns for {ticker}")
                return None
            st.session_state.cached_data[cache_key] = data
            st.session_state.last_fetch_time[cache_key] = current_time
            return data
        return None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Analyze divergence
def analyze_divergence(price_data, rsi_data):
    divergences = []
    
    # Ensure we have pandas Series
    if isinstance(price_data, np.ndarray):
        price_data = pd.Series(price_data)
    if isinstance(rsi_data, np.ndarray):
        rsi_data = pd.Series(rsi_data)
    
    # Remove NaN values
    price_data = price_data.dropna()
    rsi_data = rsi_data.dropna()
    
    # Price trend (last 20 periods)
    if len(price_data) >= 20 and len(rsi_data) >= 20:
        price_trend = "rising" if price_data.iloc[-1] > price_data.iloc[-20] else "falling"
        rsi_trend = "rising" if rsi_data.iloc[-1] > rsi_data.iloc[-20] else "falling"
        
        if price_trend == "rising" and rsi_trend == "falling":
            divergences.append("bearish divergence detected (price up, RSI down)")
        elif price_trend == "falling" and rsi_trend == "rising":
            divergences.append("bullish divergence detected (price down, RSI up)")
    
    # RSI levels
    if len(rsi_data) > 0:
        current_rsi = rsi_data.iloc[-1]
        if current_rsi > 70:
            divergences.append(f"overbought (RSI: {current_rsi:.1f})")
        elif current_rsi < 30:
            divergences.append(f"oversold (RSI: {current_rsi:.1f})")
    
    return divergences

# Generate summary
def generate_summary(ticker1, ticker2, ratio_data, rsi_ratio, price1_data, rsi1, price2_data, rsi2):
    summary_parts = []
    
    # Convert to pandas Series if needed
    if isinstance(ratio_data, np.ndarray):
        ratio_data = pd.Series(ratio_data)
    if isinstance(price1_data, np.ndarray):
        price1_data = pd.Series(price1_data)
    if isinstance(price2_data, np.ndarray):
        price2_data = pd.Series(price2_data)
    if isinstance(rsi_ratio, np.ndarray):
        rsi_ratio = pd.Series(rsi_ratio)
    if isinstance(rsi1, np.ndarray):
        rsi1 = pd.Series(rsi1)
    if isinstance(rsi2, np.ndarray):
        rsi2 = pd.Series(rsi2)
    
    # Remove NaN values
    ratio_data = ratio_data.dropna()
    price1_data = price1_data.dropna()
    price2_data = price2_data.dropna()
    rsi_ratio = rsi_ratio.dropna()
    rsi1 = rsi1.dropna()
    rsi2 = rsi2.dropna()
    
    # Ratio trend
    if len(ratio_data) >= 20:
        ratio_change = ((ratio_data.iloc[-1] - ratio_data.iloc[-20]) / ratio_data.iloc[-20] * 100)
        if ratio_change > 5:
            summary_parts.append(f"{ticker1} is outperforming {ticker2} by {ratio_change:.1f}%")
        elif ratio_change < -5:
            summary_parts.append(f"{ticker2} is outperforming {ticker1} by {abs(ratio_change):.1f}%")
        else:
            summary_parts.append(f"Ratio relatively stable")
    
    # Divergence analysis
    div1 = analyze_divergence(price1_data, rsi1)
    div2 = analyze_divergence(price2_data, rsi2)
    div_ratio = analyze_divergence(ratio_data, rsi_ratio)
    
    if div1:
        summary_parts.append(f"{ticker1}: {', '.join(div1)}")
    if div2:
        summary_parts.append(f"{ticker2}: {', '.join(div2)}")
    if div_ratio:
        summary_parts.append(f"Ratio: {', '.join(div_ratio)}")
    
    summary = ". ".join(summary_parts[:3]) + "."  # Limit to keep under 50 words
    return summary

# Plot candlestick with RSI
def plot_candlestick_with_rsi(data, ticker, title):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{title}', 'RSI'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker
        ),
        row=1, col=1
    )
    
    # Calculate and plot RSI
    rsi = calculate_rsi(data['Close'])
    fig.add_trace(
        go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    
    return fig, rsi

# Plot ratio chart with RSI
def plot_ratio_with_rsi(data1, data2, ticker1, ticker2):
    ratio = data1['Close'] / data2['Close']
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker1}/{ticker2} Ratio', 'RSI'),
        row_heights=[0.7, 0.3]
    )
    
    # Ratio line chart
    fig.add_trace(
        go.Scatter(x=ratio.index, y=ratio, name='Ratio', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Calculate and plot RSI of ratio
    rsi_ratio = calculate_rsi(ratio)
    fig.add_trace(
        go.Scatter(x=ratio.index, y=rsi_ratio, name='RSI', line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    
    return fig, ratio, rsi_ratio

# Create heatmap
def create_return_heatmap(data, ticker, freq='M'):
    returns = data['Close'].pct_change() * 100
    
    if freq == 'M':  # Month vs Day
        pivot_data = returns.groupby([returns.index.month, returns.index.day]).mean().unstack(fill_value=0)
        x_label = "Day of Month"
        y_label = "Month"
        y_ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    else:  # Year vs Month
        pivot_data = returns.groupby([returns.index.year, returns.index.month]).mean().unstack(fill_value=0)
        x_label = "Month"
        y_label = "Year"
        y_ticktext = [str(y) for y in sorted(returns.index.year.unique())]
    
    # Limit heatmap size for better visibility
    if pivot_data.shape[0] > 20:
        pivot_data = pivot_data.iloc[-20:]
    if pivot_data.shape[1] > 31:
        pivot_data = pivot_data.iloc[:, :31]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn',
        text=np.round(pivot_data.values, 2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Returns %")
    ))
    
    fig.update_layout(
        title=f'{ticker} Returns Heatmap ({freq})',
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500
    )
    
    if freq == 'M':
        fig.update_yaxes(ticktext=y_ticktext, tickvals=list(range(1, 13)))
    
    return fig

# Main app
st.title("📊 Advanced Ratio Chart Analyzer with RSI")
st.markdown("**Analyze stock ratios, detect divergences, and visualize returns with heatmaps**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ticker inputs
    ticker1 = st.text_input("Ticker 1 (Numerator)", value="^NSEI", help="e.g., ^NSEI (Nifty 50), RELIANCE.NS")
    ticker2 = st.text_input("Ticker 2 (Denominator)", value="^BSESN", help="e.g., ^BSESN (Sensex), TCS.NS")
    
    # Timeframe selection
    interval_options = {
        "1 Minute": "1m", "3 Minutes": "3m", "5 Minutes": "5m", 
        "10 Minutes": "10m", "15 Minutes": "15m", "30 Minutes": "30m",
        "1 Hour": "1h", "2 Hours": "2h", "4 Hours": "4h",
        "1 Day": "1d", "5 Days": "5d", "1 Week": "1wk", "1 Month": "1mo"
    }
    interval = st.selectbox("Interval", list(interval_options.keys()), index=8)
    
    # Period selection
    period_options = {
        "1 Day": "1d", "5 Days": "5d", "7 Days": "7d", 
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "1 Year": "1y", "2 Years": "2y", "3 Years": "3y",
        "5 Years": "5y", "6 Years": "6y", "10 Years": "10y",
        "15 Years": "15y", "20 Years": "20y", "25 Years": "25y", "30 Years": "30y"
    }
    
    # Smart period selection based on interval
    if interval_options[interval] in ['1m', '3m', '5m', '10m', '15m', '30m']:
        default_period = "1 Day" if interval_options[interval] in ['1m', '3m', '5m'] else "5 Days"
        available_periods = list(period_options.keys())[:7]  # Limit to reasonable periods
    else:
        default_period = "1 Year"
        available_periods = list(period_options.keys())
    
    period = st.selectbox("Period", available_periods, 
                         index=available_periods.index(default_period) if default_period in available_periods else 0)
    
    # Heatmap frequency
    heatmap_freq = st.radio("Heatmap Type", ["Month vs Day", "Year vs Month"])
    
    # RSI Period
    rsi_period = st.slider("RSI Period", min_value=5, max_value=50, value=14)
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze", type="primary")

# Main content area
if analyze_button:
    st.session_state.analysis_done = True

if st.session_state.analysis_done:
    with st.spinner("Fetching data and generating analysis..."):
        try:
            # Fetch data
            data1 = fetch_data_cached(ticker1, period_options[period], interval_options[interval])
            time.sleep(0.5)  # Rate limit protection
            data2 = fetch_data_cached(ticker2, period_options[period], interval_options[interval])
            
            if data1 is None or data2 is None or data1.empty or data2.empty:
                st.error("❌ Unable to fetch data. Please check ticker symbols and try again.")
                st.stop()
            
            # Align data on common dates
            common_index = data1.index.intersection(data2.index)
            data1 = data1.loc[common_index]
            data2 = data2.loc[common_index]
            
            if len(data1) < 20:
                st.warning("⚠️ Limited data available. Results may not be comprehensive.")
            
            # Individual charts
            st.subheader(f"📈 Individual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {ticker1}")
                fig1, rsi1 = plot_candlestick_with_rsi(data1, ticker1, f"{ticker1} Price")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown(f"### {ticker2}")
                fig2, rsi2 = plot_candlestick_with_rsi(data2, ticker2, f"{ticker2} Price")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Ratio chart
            st.subheader(f"📊 Ratio Analysis: {ticker1}/{ticker2}")
            fig_ratio, ratio_data, rsi_ratio = plot_ratio_with_rsi(data1, data2, ticker1, ticker2)
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Summary
            summary = generate_summary(
                ticker1, ticker2, ratio_data, rsi_ratio,
                data1['Close'], rsi1, data2['Close'], rsi2
            )
            
            st.info(f"**💡 Summary:** {summary}")
            
            # Heatmaps
            st.subheader("🔥 Returns Heatmap")
            
            col3, col4 = st.columns(2)
            
            freq_code = 'M' if heatmap_freq == "Month vs Day" else 'Y'
            
            with col3:
                st.markdown(f"### {ticker1} Returns")
                fig_heat1 = create_return_heatmap(data1, ticker1, freq_code)
                st.plotly_chart(fig_heat1, use_container_width=True)
            
            with col4:
                st.markdown(f"### {ticker2} Returns")
                fig_heat2 = create_return_heatmap(data2, ticker2, freq_code)
                st.plotly_chart(fig_heat2, use_container_width=True)
            
            # Statistics
            with st.expander("📊 Detailed Statistics"):
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.metric(f"{ticker1} Current", f"{data1['Close'].iloc[-1]:.2f}")
                    st.metric(f"{ticker1} RSI", f"{rsi1.iloc[-1]:.2f}")
                
                with col_stat2:
                    st.metric(f"{ticker2} Current", f"{data2['Close'].iloc[-1]:.2f}")
                    st.metric(f"{ticker2} RSI", f"{rsi2.iloc[-1]:.2f}")
                
                with col_stat3:
                    st.metric("Ratio", f"{ratio_data.iloc[-1]:.4f}")
                    st.metric("Ratio RSI", f"{rsi_ratio.iloc[-1]:.2f}")
            
            st.success("✅ Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
else:
    st.info("👈 Configure your analysis parameters in the sidebar and click **Analyze** to start.")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Indian Market Ticker Examples:
        - **Indices:** ^NSEI (Nifty 50), ^BSESN (Sensex), ^NSEBANK (Bank Nifty)
        - **Stocks:** RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS
        - **MCX:** GOLD.BO, SILVER.BO (add .BO for commodities)
        
        ### Features:
        1. **Dynamic Ticker Selection:** Enter any valid ticker symbol
        2. **Multiple Timeframes:** From 1-minute to monthly intervals
        3. **Ratio Analysis:** Compare relative performance of two instruments
        4. **RSI Indicators:** Identify overbought/oversold conditions
        5. **Divergence Detection:** Automatically detect price-RSI divergences
        6. **Returns Heatmap:** Visualize performance patterns over time
        7. **Smart Caching:** Prevents API rate limit issues
        
        ### Tips:
        - For Indian stocks, add `.NS` (NSE) or `.BO` (BSE) suffix
        - Start with longer periods (1 Year) for better analysis
        - Check the summary for quick insights and divergence alerts
        """)

# Footer
st.markdown("---")
st.caption("⚡ Powered by yfinance | Data cached for 5 minutes to prevent API rate limits")
