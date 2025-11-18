import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Algo Trading Analysis", layout="wide", initial_sidebar_state="expanded")

# Utility functions
def convert_to_ist(df):
    """Convert datetime to IST and handle timezone awareness"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(ist)
        else:
            df.index = df.index.tz_convert(ist)
        df.index = df.index.tz_localize(None)
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

def calculate_volatility(data, window=14):
    """Calculate rolling volatility"""
    returns = data.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

def find_pattern_similarity(data, window=20, top_n=5):
    """Find similar patterns in historical data"""
    if len(data) < window * 2:
        return []
    
    current_pattern = data[-window:].values
    current_pattern_norm = (current_pattern - current_pattern.mean()) / (current_pattern.std() + 1e-8)
    
    similarities = []
    for i in range(window, len(data) - window):
        historical_pattern = data[i-window:i].values
        historical_pattern_norm = (historical_pattern - historical_pattern.mean()) / (historical_pattern.std() + 1e-8)
        
        correlation = np.corrcoef(current_pattern_norm, historical_pattern_norm)[0, 1]
        if not np.isnan(correlation):
            similarities.append({
                'index': i,
                'date': data.index[i],
                'correlation': correlation,
                'future_return': (data.iloc[i+window] - data.iloc[i]) / data.iloc[i] * 100
            })
    
    similarities.sort(key=lambda x: x['correlation'], reverse=True)
    return similarities[:top_n]

# Streamlit UI
st.title("🚀 Advanced Algorithmic Trading Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ticker selection
    ticker1_options = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Sensex": "^BSESN",
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "USD/INR": "USDINR=X",
        "Custom": "CUSTOM"
    }
    
    ticker1_name = st.selectbox("Select Ticker 1", list(ticker1_options.keys()))
    if ticker1_name == "Custom":
        ticker1 = st.text_input("Enter Ticker 1 Symbol", "AAPL")
    else:
        ticker1 = ticker1_options[ticker1_name]
    
    ticker2_name = st.selectbox("Select Ticker 2", list(ticker1_options.keys()), index=1)
    if ticker2_name == "Custom":
        ticker2 = st.text_input("Enter Ticker 2 Symbol", "MSFT")
    else:
        ticker2 = ticker1_options[ticker2_name]
    
    # Timeframe and period
    timeframe = st.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])
    period = st.selectbox("Period", ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "6y", "10y", "15y", "20y", "25y", "30y"])
    
    # Fetch button
    fetch_button = st.button("📊 Fetch Data & Analyze", type="primary", use_container_width=True)

# Session state to store data
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False

# Main analysis
if fetch_button:
    with st.spinner("Fetching data from Yahoo Finance..."):
        try:
            # Fetch data
            data1 = yf.download(ticker1, period=period, interval=timeframe, progress=False)
            data2 = yf.download(ticker2, period=period, interval=timeframe, progress=False)
            
            if data1.empty or data2.empty:
                st.error("❌ Failed to fetch data. Please check ticker symbols and try again.")
                st.stop()
            
            # Convert to IST
            data1 = convert_to_ist(data1)
            data2 = convert_to_ist(data2)
            
            # Store in session state
            st.session_state.data1 = data1
            st.session_state.data2 = data2
            st.session_state.ticker1 = ticker1
            st.session_state.ticker2 = ticker2
            st.session_state.ticker1_name = ticker1_name if ticker1_name != "Custom" else ticker1
            st.session_state.ticker2_name = ticker2_name if ticker2_name != "Custom" else ticker2
            st.session_state.data_fetched = True
            
            st.success("✅ Data fetched successfully!")
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.stop()

# Display analysis if data is fetched
if st.session_state.data_fetched:
    data1 = st.session_state.data1
    data2 = st.session_state.data2
    ticker1 = st.session_state.ticker1
    ticker2 = st.session_state.ticker2
    ticker1_name = st.session_state.ticker1_name
    ticker2_name = st.session_state.ticker2_name
    
    # Calculate ratio
    ratio = data1['Close'] / data2['Close']
    ratio_df = pd.DataFrame(ratio, columns=['Ratio'])
    
    # Calculate RSI
    rsi1 = calculate_rsi(data1['Close'])
    rsi2 = calculate_rsi(data2['Close'])
    rsi_ratio = calculate_rsi(ratio)
    
    # Calculate volatility
    vol1 = calculate_volatility(data1['Close'])
    vol2 = calculate_volatility(data2['Close'])
    vol_ratio = calculate_volatility(ratio)
    
    # ============= SECTION 1: BASIC TICKER ANALYSIS =============
    st.header("📈 1. Basic Ticker Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{ticker1_name} Analysis")
        latest_price1 = data1['Close'].iloc[-1]
        prev_price1 = data1['Close'].iloc[-2] if len(data1) > 1 else latest_price1
        change1 = latest_price1 - prev_price1
        pct_change1 = (change1 / prev_price1) * 100
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Latest Price", f"{float(latest_price1):.2f}")
        metric_col2.metric("Change", f"{float(change1):.2f}", f"{float(pct_change1):.2f}%")
        metric_col3.metric("RSI", f"{rsi1.iloc[-1]:.2f}")
        
        # Basic stats table
        stats1 = pd.DataFrame({
            'DateTime': data1.index[-10:],
            'Close': data1['Close'].iloc[-10:].values,
            'Change': data1['Close'].iloc[-10:].diff().values,
            '% Change': (data1['Close'].iloc[-10:].pct_change() * 100).values
        })
        st.dataframe(stats1.style.background_gradient(subset=['Change', '% Change'], cmap='RdYlGn'), use_container_width=True)
    
    with col2:
        st.subheader(f"{ticker2_name} Analysis")
        latest_price2 = data2['Close'].iloc[-1]
        prev_price2 = data2['Close'].iloc[-2] if len(data2) > 1 else latest_price2
        change2 = latest_price2 - prev_price2
        pct_change2 = (change2 / prev_price2) * 100
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Latest Price", f"{float(latest_price2):.2f}")
        metric_col2.metric("Change", f"{float(change2):.2f}", f"{float(pct_change2):.2f}%")
        metric_col3.metric("RSI", f"{rsi2.iloc[-1]:.2f}")
        
        # Basic stats table
        stats2 = pd.DataFrame({
            'DateTime': data2.index[-10:],
            'Close': data2['Close'].iloc[-10:].values,
            'Change': data2['Close'].iloc[-10:].diff().values,
            '% Change': (data2['Close'].iloc[-10:].pct_change() * 100).values
        })
        st.dataframe(stats2.style.background_gradient(subset=['Change', '% Change'], cmap='RdYlGn'), use_container_width=True)
    
    # ============= SECTION 2: RATIO ANALYSIS =============
    st.header("🔀 2. Ratio Analysis")
    
    ratio_analysis = pd.DataFrame({
        'DateTime': data1.index,
        f'{ticker1_name} Price': data1['Close'].values,
        f'{ticker2_name} Price': data2['Close'].values,
        'Ratio': ratio.values,
        f'RSI {ticker1_name}': rsi1.values,
        f'RSI {ticker2_name}': rsi2.values,
        'RSI Ratio': rsi_ratio.values
    })
    
    st.dataframe(ratio_analysis.tail(20).style.background_gradient(subset=['Ratio', 'RSI Ratio'], cmap='viridis'), use_container_width=True)
    
    # Ratio insights
    st.subheader("📊 Ratio Insights")
    current_ratio = ratio.iloc[-1]
    mean_ratio = ratio.mean()
    std_ratio = ratio.std()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Ratio", f"{float(current_ratio):.4f}")
    col2.metric("Mean Ratio", f"{float(mean_ratio):.4f}")
    col3.metric("Std Dev", f"{std_ratio:.4f}")
    
    # Find historical rallies in ratio
    ratio_changes = ratio.pct_change()
    rally_threshold = ratio_changes.quantile(0.9)
    rallies = ratio_changes[ratio_changes > rally_threshold]
    
    if len(rallies) > 0:
        st.write("### 🚀 Historical Ratio Rallies Detected:")
        rally_data = []
        for idx in rallies.index[-5:]:
            idx_loc = ratio.index.get_loc(idx)
            if idx_loc < len(ratio) - 1:
                future_change = (ratio.iloc[idx_loc+1] - ratio.iloc[idx_loc]) / ratio.iloc[idx_loc] * 100
                rally_data.append({
                    'DateTime': idx,
                    'Ratio Value': ratio.loc[idx],
                    'Change %': rallies.loc[idx] * 100,
                    'Next Period Change %': future_change
                })
        
        rally_df = pd.DataFrame(rally_data)
        st.dataframe(rally_df.style.background_gradient(subset=['Change %'], cmap='Greens'), use_container_width=True)
        
        # Pattern comparison
        st.write(f"""
        **📌 Key Insight:** Historical data shows {len(rallies)} significant ratio movements (>{rally_threshold*100:.2f}%). 
        The current ratio of {float(current_ratio):.4f} is {'above' if current_ratio > mean_ratio else 'below'} the mean ({mean_ratio:.4f}), 
        suggesting potential {'reversion to mean' if abs(current_ratio - mean_ratio) > std_ratio else 'stability'}.
        """)
    
    # ============= SECTION 3: RATIO BINNING ANALYSIS =============
    st.header("📊 3. Ratio Binning & Rally Analysis")
    
    # Create ratio bins
    ratio_bins = pd.qcut(ratio.dropna(), q=10, duplicates='drop')
    ratio_binned = pd.DataFrame({
        'DateTime': ratio.index,
        'Ratio': ratio.values,
        'Bin': ratio_bins,
        'Price1': data1['Close'].values,
        'Price2': data2['Close'].values
    })
    
    # Calculate rally points in each bin
    ratio_binned['Forward_Return'] = ratio_binned['Ratio'].pct_change().shift(-1) * 100
    bin_analysis = ratio_binned.groupby('Bin').agg({
        'Forward_Return': ['mean', 'std', 'count'],
        'Ratio': ['min', 'max', 'mean']
    }).round(4)
    
    st.dataframe(bin_analysis.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    
    st.write("""
    **📌 Bin Analysis Insights:** This table shows the average forward returns for each ratio bin. 
    Bins with consistently positive forward returns indicate strong buying opportunities, 
    while negative returns suggest caution or shorting opportunities.
    """)
    
    # ============= SECTION 4: CANDLESTICK & RSI PLOTS =============
    st.header("📉 4. Technical Charts")
    
    # Create subplots
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker1_name} Candlesticks', f'{ticker2_name} Candlesticks', 
                       'Ratio Candlesticks', f'{ticker1_name} RSI', f'{ticker2_name} RSI', 'Ratio RSI'),
        row_heights=[0.2, 0.2, 0.2, 0.13, 0.13, 0.14]
    )
    
    # Ticker 1 candlesticks
    fig.add_trace(go.Candlestick(
        x=data1.index,
        open=data1['Open'],
        high=data1['High'],
        low=data1['Low'],
        close=data1['Close'],
        name=ticker1_name
    ), row=1, col=1)
    
    # Ticker 2 candlesticks
    fig.add_trace(go.Candlestick(
        x=data2.index,
        open=data2['Open'],
        high=data2['High'],
        low=data2['Low'],
        close=data2['Close'],
        name=ticker2_name
    ), row=2, col=1)
    
    # Ratio as line chart (since we don't have OHLC for ratio)
    fig.add_trace(go.Scatter(
        x=data1.index,
        y=ratio,
        mode='lines',
        name='Ratio',
        line=dict(color='purple', width=2)
    ), row=3, col=1)
    
    # RSI plots
    fig.add_trace(go.Scatter(x=data1.index, y=rsi1, name=f'RSI {ticker1_name}', line=dict(color='blue')), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    
    fig.add_trace(go.Scatter(x=data2.index, y=rsi2, name=f'RSI {ticker2_name}', line=dict(color='orange')), row=5, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=5, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=5, col=1)
    
    fig.add_trace(go.Scatter(x=data1.index, y=rsi_ratio, name='RSI Ratio', line=dict(color='purple')), row=6, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=6, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=6, col=1)
    
    fig.update_layout(height=1800, showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # ============= SECTION 5: DATA EXPORT =============
    st.header("💾 5. Export Data")
    
    export_data = pd.DataFrame({
        'DateTime_IST': data1.index,
        f'{ticker1_name}_Open': data1['Open'].values,
        f'{ticker1_name}_High': data1['High'].values,
        f'{ticker1_name}_Low': data1['Low'].values,
        f'{ticker1_name}_Close': data1['Close'].values,
        f'{ticker1_name}_Volume': data1['Volume'].values,
        f'{ticker2_name}_Open': data2['Open'].values,
        f'{ticker2_name}_High': data2['High'].values,
        f'{ticker2_name}_Low': data2['Low'].values,
        f'{ticker2_name}_Close': data2['Close'].values,
        f'{ticker2_name}_Volume': data2['Volume'].values,
        'Ratio': ratio.values
    })
    
    col1, col2 = st.columns(2)
    with col1:
        csv = export_data.to_csv(index=False)
        st.download_button("📥 Download as CSV", csv, "trading_data.csv", "text/csv")
    with col2:
        # For Excel, we'll use CSV as fallback
        st.download_button("📥 Download as Excel (CSV)", csv, "trading_data.xlsx", "text/csv")
    
    # ============= SECTION 6: VOLATILITY ANALYSIS =============
    st.header("📊 6. Volatility & Points Analysis")
    
    points_data = pd.DataFrame({
        'DateTime_IST': data1.index,
        f'{ticker1_name} Price': data1['Close'].values,
        f'{ticker2_name} Price': data2['Close'].values,
        'Ratio': ratio.values,
        f'Vol {ticker1_name}': vol1.values,
        f'Vol {ticker2_name}': vol2.values,
        'Vol Ratio': vol_ratio.values,
        f'Points {ticker1_name}': data1['Close'].diff().values,
        f'Points {ticker2_name}': data2['Close'].diff().values,
        'Points Ratio': ratio.diff().values,
        f'% Change {ticker1_name}': (data1['Close'].pct_change() * 100).values,
        f'% Change {ticker2_name}': (data2['Close'].pct_change() * 100).values,
        '% Change Ratio': (ratio.pct_change() * 100).values
    })
    
    st.dataframe(points_data.tail(20).style.background_gradient(subset=[f'Points {ticker1_name}', f'Points {ticker2_name}'], cmap='RdYlGn'), 
                use_container_width=True)
    
    st.write(f"""
    **📌 Volatility Insights:** 
    - Current {ticker1_name} volatility: {vol1.iloc[-1]:.4f} (annualized)
    - Current {ticker2_name} volatility: {vol2.iloc[-1]:.4f} (annualized)
    - Current Ratio volatility: {vol_ratio.iloc[-1]:.4f} (annualized)
    - Higher volatility indicates higher risk but also potential for larger moves.
    """)
    
    # ============= SECTION 7: VOLATILITY BINNING =============
    st.header("📊 7. Volatility Binning Analysis")
    
    vol_binned = points_data.dropna().copy()
    vol_binned[f'Vol_Bin_{ticker1_name}'] = pd.qcut(vol_binned[f'Vol {ticker1_name}'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    
    vol_analysis = vol_binned.groupby(f'Vol_Bin_{ticker1_name}').agg({
        f'Points {ticker1_name}': ['mean', 'std', 'sum'],
        f'Points {ticker2_name}': ['mean', 'std', 'sum'],
        'Points Ratio': ['mean', 'std', 'sum']
    }).round(4)
    
    st.dataframe(vol_analysis.style.background_gradient(cmap='coolwarm'), use_container_width=True)
    
    st.write("""
    **📌 Volatility Binning Insights:** This shows how price movements correlate with volatility levels.
    High volatility periods typically see larger point movements (both up and down), while low volatility
    indicates consolidation phases. Use this to adjust position sizing and risk management.
    """)
    
    # ============= SECTION 8: RETURNS HEATMAPS =============
    st.header("🔥 8. Returns Heatmaps")
    
    # Day vs Week returns
    returns1_daily = data1['Close'].pct_change() * 100
    returns2_daily = data2['Close'].pct_change() * 100
    ratio_returns_daily = ratio.pct_change() * 100
    
    # Create heatmap data
    if len(data1) > 7:
        # Reshape for heatmap
        st.subheader(f"{ticker1_name} Returns Heatmap")
        returns1_matrix = returns1_daily.tail(50).values.reshape(-1, 5) if len(returns1_daily) >= 50 else returns1_daily.values.reshape(-1, 5)[:10]
        fig1 = go.Figure(data=go.Heatmap(z=returns1_matrix, colorscale='RdYlGn', zmid=0))
        fig1.update_layout(title=f'{ticker1_name} Returns Pattern', height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader(f"{ticker2_name} Returns Heatmap")
        returns2_matrix = returns2_daily.tail(50).values.reshape(-1, 5) if len(returns2_daily) >= 50 else returns2_daily.values.reshape(-1, 5)[:10]
        fig2 = go.Figure(data=go.Heatmap(z=returns2_matrix, colorscale='RdYlGn', zmid=0))
        fig2.update_layout(title=f'{ticker2_name} Returns Pattern', height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Ratio Returns Heatmap")
        ratio_returns_matrix = ratio_returns_daily.tail(50).values.reshape(-1, 5) if len(ratio_returns_daily) >= 50 else ratio_returns_daily.values.reshape(-1, 5)[:10]
        fig3 = go.Figure(data=go.Heatmap(z=ratio_returns_matrix, colorscale='RdYlGn', zmid=0))
        fig3.update_layout(title='Ratio Returns Pattern', height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.write("""
        **📌 Returns Heatmap Insights:** Green zones indicate positive returns, red indicates losses.
        Look for clusters of green for potential buy zones and clusters of red for selling opportunities.
        Current market positioning relative to these patterns can guide entry/exit decisions.
        """)
    
    # ============= SECTION 9: PATTERN SIMILARITY =============
    st.header("🔍 9. Pattern Similarity Detection")
    
    similarities1 = find_pattern_similarity(data1['Close'], window=min(20, len(data1)//4))
    similarities2 = find_pattern_similarity(data2['Close'], window=min(20, len(data2)//4))
    similarities_ratio = find_pattern_similarity(ratio, window=min(20, len(ratio)//4))
    
    if similarities1:
        st.subheader(f"{ticker1_name} Similar Patterns")
        sim_df1 = pd.DataFrame(similarities1)
        st.dataframe(sim_df1.style.background_gradient(subset=['correlation', 'future_return'], cmap='RdYlGn'), use_container_width=True)
        
        avg_future_return = sim_df1['future_return'].mean()
        st.write(f"""
        **📌 {ticker1_name} Pattern Analysis:** Found {len(similarities1)} similar historical patterns.
        Average future return after similar patterns: {avg_future_return:.2f}%.
        Historical dates: {', '.join([d.strftime('%Y-%m-%d %H:%M') for d in sim_df1['date'].head(3)])}
        """)
    
    if similarities_ratio:
        st.subheader("Ratio Similar Patterns")
        sim_df_ratio = pd.DataFrame(similarities_ratio)
        st.dataframe(sim_df_ratio.style.background_gradient(subset=['correlation', 'future_return'], cmap='RdYlGn'), use_container_width=True)
        
        avg_future_return_ratio = sim_df_ratio['future_return'].mean()
        st.write(f"""
        **📌 Ratio Pattern Analysis:** Based on {len(similarities_ratio)} similar patterns,
        the ratio showed an average movement of {avg_future_return_ratio:.2f}% following similar setups.
        Most similar period: {sim_df_ratio.iloc[0]['date'].strftime('%Y-%m-%d %H:%M')} with {sim_df_ratio.iloc[0]['correlation']:.2%} correlation.
        """)
    
    # ============= SECTION 10: VOLATILITY HEATMAPS =============
    st.header("🌡️ 10. Volatility Heatmaps")
    
    if len(vol1.dropna()) > 25:
        vol1_matrix = vol1.dropna().tail(50).values.reshape(-1, 5) if len(vol1.dropna()) >= 50 else vol1.dropna().values.reshape(-1, 5)[:5]
        fig_vol1 = go.Figure(data=go.Heatmap(z=vol1_matrix, colorscale='Reds'))
        fig_vol1.update_layout(title=f'{ticker1_name} Volatility Heatmap', height=400)
        st.plotly_chart(fig_vol1, use_container_width=True)
        
        vol2_matrix = vol2.dropna().tail(50).values.reshape(-1, 5) if len(vol2.dropna()) >= 50 else vol2.dropna().values.reshape(-1, 5)[:5]
        fig_vol2 = go.Figure(data=go.Heatmap(z=vol2_matrix, colorscale='Reds'))
        fig_vol2.update_layout(title=f'{ticker2_name} Volatility Heatmap', height=400)
        st.plotly_chart(fig_vol2, use_container_width=True)
        
        st.write("""
        **📌 Volatility Heatmap Insights:** Dark red zones indicate high volatility periods - these are
        periods of uncertainty and potential large moves. Lighter zones show consolidation.
        Trade smaller sizes during high volatility and look for breakouts during consolidation.
        """)
    
    # ============= SECTION 11: PRICE ACTION SIMILARITY & FORECAST =============
    st.header("🎯 11. Price Action Forecast")
    
    # Advanced forecasting based on similar patterns
    if similarities1 and similarities_ratio:
        st.subheader("Forecast Based on Historical Similarity")
        
        # Get current prices and patterns
        current_price1 = data1['Close'].iloc[-1]
        current_price2 = data2['Close'].iloc[-1]
        current_ratio_val = ratio.iloc[-1]
        
        # Calculate average outcomes from similar patterns
        forecast_change1 = sim_df1['future_return'].mean()
        forecast_price1 = current_price1 * (1 + forecast_change1/100)
        forecast_points1 = forecast_price1 - current_price1
        
        forecast_change_ratio = sim_df_ratio['future_return'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{ticker1_name} Forecast", f"{forecast_price1:.2f}", f"{forecast_points1:.2f} pts ({forecast_change1:.2f}%)")
        col2.metric("Current Ratio", f"{current_ratio_val:.4f}")
        col3.metric("Forecast Ratio Move", f"{forecast_change_ratio:.2f}%")
        
        st.write(f"""
        **📌 Forecast Analysis (IST: {data1.index[-1].strftime('%Y-%m-%d %H:%M:%S')}):**
        
        Based on {len(similarities1)} similar historical patterns, the market shows:
        - {ticker1_name} expected to move {forecast_points1:.2f} points ({forecast_change1:.2f}%)
        - Ratio expected to move {forecast_change_ratio:.2f}%
        - Confidence: {sim_df1['correlation'].mean():.2%} average correlation
        
        **Historical Reference:**
        Most similar pattern occurred on {sim_df1.iloc[0]['date'].strftime('%Y-%m-%d %H:%M')} IST,
        which was followed by a {sim_df1.iloc[0]['future_return']:.2f}% move.
        """)
    
    # ============= SECTION 12: REVERSAL FORECASTING =============
    st.header("🔄 12. Reversal Detection & Forecasting")
    
    # Detect historical reversals
    returns1 = data1['Close'].pct_change()
    cumulative_returns = (1 + returns1).cumprod()
    
    # Find peaks and troughs
    from scipy.signal import argrelextrema
    peaks_idx = argrelextrema(data1['Close'].values, np.greater, order=5)[0]
    troughs_idx = argrelextrema(data1['Close'].values, np.less, order=5)[0]
    
    if len(peaks_idx) > 0 and len(troughs_idx) > 0:
        # Analyze recent rallies before reversals
        reversal_data = []
        for peak_idx in peaks_idx[-5:]:
            if peak_idx > 20 and peak_idx < len(data1) - 5:
                # Find the trough before this peak
                prior_troughs = troughs_idx[troughs_idx < peak_idx]
                if len(prior_troughs) > 0:
                    trough_idx = prior_troughs[-1]
                    rally_points = data1['Close'].iloc[peak_idx] - data1['Close'].iloc[trough_idx]
                    rally_pct = (rally_points / data1['Close'].iloc[trough_idx]) * 100
                    
                    # Check what happened after the peak
                    reversal_points = data1['Close'].iloc[peak_idx+5] - data1['Close'].iloc[peak_idx]
                    reversal_pct = (reversal_points / data1['Close'].iloc[peak_idx]) * 100
                    
                    reversal_data.append({
                        'Peak_DateTime_IST': data1.index[peak_idx],
                        'Peak_Price': data1['Close'].iloc[peak_idx],
                        'Rally_Points': rally_points,
                        'Rally_%': rally_pct,
                        'Reversal_Points': reversal_points,
                        'Reversal_%': reversal_pct,
                        'RSI_at_Peak': rsi1.iloc[peak_idx]
                    })
        
        if reversal_data:
            reversal_df = pd.DataFrame(reversal_data)
            st.dataframe(reversal_df.style.background_gradient(subset=['Rally_%', 'Reversal_%'], cmap='RdYlGn'), 
                        use_container_width=True)
            
            # Current market position analysis
            current_price = data1['Close'].iloc[-1]
            recent_low_idx = data1['Close'].iloc[-20:].idxmin()
            recent_low = data1['Close'].loc[recent_low_idx]
            current_rally_points = current_price - recent_low
            current_rally_pct = (current_rally_points / recent_low) * 100
            
            avg_rally_before_reversal = reversal_df['Rally_%'].mean()
            avg_reversal = reversal_df['Reversal_%'].mean()
            
            st.write(f"""
            **📌 Reversal Analysis (Current Time IST: {data1.index[-1].strftime('%Y-%m-%d %H:%M:%S')}):**
            
            **Historical Pattern:**
            - Average rally before reversal: {avg_rally_before_reversal:.2f}% ({reversal_df['Rally_Points'].mean():.2f} points)
            - Average reversal magnitude: {avg_reversal:.2f}% ({reversal_df['Reversal_Points'].mean():.2f} points)
            - Typical RSI at reversal: {reversal_df['RSI_at_Peak'].mean():.2f}
            
            **Current Market Setup:**
            - Current price: {current_price:.2f}
            - Rally from recent low ({recent_low_idx.strftime('%Y-%m-%d %H:%M')} IST): {current_rally_points:.2f} points ({current_rally_pct:.2f}%)
            - Current RSI: {rsi1.iloc[-1]:.2f}
            - Current {ticker1_name}/{ticker2_name} Ratio: {current_ratio_val:.4f}
            
            **⚠️ Reversal Warning:**
            """)
            
            if current_rally_pct >= avg_rally_before_reversal * 0.8:
                st.warning(f"""
                🚨 Market has rallied {current_rally_pct:.2f}%, approaching historical reversal threshold 
                of {avg_rally_before_reversal:.2f}%. Consider taking profits or tightening stops.
                Expected reversal could be {avg_reversal:.2f}% ({abs(avg_reversal * current_price / 100):.2f} points).
                """)
            else:
                st.success(f"""
                ✅ Current rally of {current_rally_pct:.2f}% is still below typical reversal point.
                Historical data suggests room for {avg_rally_before_reversal - current_rally_pct:.2f}% more upside
                before reversal risk increases significantly.
                """)
    
    # Ratio reversal analysis
    st.subheader("Ratio Reversal Patterns")
    ratio_peaks_idx = argrelextrema(ratio.values, np.greater, order=5)[0]
    ratio_troughs_idx = argrelextrema(ratio.values, np.less, order=5)[0]
    
    if len(ratio_peaks_idx) > 0:
        ratio_reversal_data = []
        for peak_idx in ratio_peaks_idx[-5:]:
            if peak_idx > 10 and peak_idx < len(ratio) - 5:
                prior_troughs = ratio_troughs_idx[ratio_troughs_idx < peak_idx]
                if len(prior_troughs) > 0:
                    trough_idx = prior_troughs[-1]
                    ratio_rally = ratio.iloc[peak_idx] - ratio.iloc[trough_idx]
                    ratio_rally_pct = (ratio_rally / ratio.iloc[trough_idx]) * 100
                    
                    ratio_reversal_data.append({
                        'DateTime_IST': ratio.index[peak_idx],
                        'Ratio_Peak': ratio.iloc[peak_idx],
                        'Rally_%': ratio_rally_pct,
                        f'{ticker1_name}_Price': data1['Close'].iloc[peak_idx],
                        f'{ticker2_name}_Price': data2['Close'].iloc[peak_idx]
                    })
        
        if ratio_reversal_data:
            ratio_rev_df = pd.DataFrame(ratio_reversal_data)
            st.dataframe(ratio_rev_df.style.background_gradient(subset=['Rally_%'], cmap='coolwarm'), 
                        use_container_width=True)
    
    # ============= SECTION 13: ULTIMATE RECOMMENDATION =============
    st.header("🎯 13. Ultimate Trading Recommendation")
    
    # Aggregate all analysis
    st.subheader("📊 Comprehensive Analysis Summary")
    
    # Calculate recommendation score
    recommendation_score = 0
    reasons = []
    
    # 1. RSI Analysis
    if rsi1.iloc[-1] < 30:
        recommendation_score += 2
        reasons.append(f"✅ {ticker1_name} RSI ({rsi1.iloc[-1]:.2f}) is oversold - Strong BUY signal")
    elif rsi1.iloc[-1] > 70:
        recommendation_score -= 2
        reasons.append(f"⚠️ {ticker1_name} RSI ({rsi1.iloc[-1]:.2f}) is overbought - SELL signal")
    else:
        recommendation_score += 0
        reasons.append(f"➖ {ticker1_name} RSI ({rsi1.iloc[-1]:.2f}) is neutral")
    
    # 2. Ratio Analysis
    ratio_zscore = (current_ratio_val - mean_ratio) / std_ratio
    if ratio_zscore < -1:
        recommendation_score += 1.5
        reasons.append(f"✅ Ratio ({current_ratio_val:.4f}) is {abs(ratio_zscore):.2f} std below mean - {ticker1_name} undervalued vs {ticker2_name}")
    elif ratio_zscore > 1:
        recommendation_score -= 1.5
        reasons.append(f"⚠️ Ratio ({current_ratio_val:.4f}) is {ratio_zscore:.2f} std above mean - {ticker1_name} overvalued vs {ticker2_name}")
    
    # 3. Pattern Similarity
    if similarities1:
        avg_future_ret = sim_df1['future_return'].mean()
        if avg_future_ret > 2:
            recommendation_score += 2
            reasons.append(f"✅ Historical patterns suggest {avg_future_ret:.2f}% upside (avg {sim_df1['future_return'].mean():.2f}%)")
        elif avg_future_ret < -2:
            recommendation_score -= 2
            reasons.append(f"⚠️ Historical patterns suggest {avg_future_ret:.2f}% downside")
    
    # 4. Volatility
    current_vol = vol1.iloc[-1]
    avg_vol = vol1.mean()
    if current_vol > avg_vol * 1.5:
        recommendation_score -= 1
        reasons.append(f"⚠️ High volatility ({current_vol:.4f}) - Exercise caution, reduce position size")
    elif current_vol < avg_vol * 0.7:
        recommendation_score += 0.5
        reasons.append(f"✅ Low volatility ({current_vol:.4f}) - Favorable for entry")
    
    # 5. Reversal Risk
    if 'reversal_df' in locals() and len(reversal_df) > 0:
        if current_rally_pct >= avg_rally_before_reversal * 0.8:
            recommendation_score -= 2
            reasons.append(f"⚠️ HIGH REVERSAL RISK: Rally at {current_rally_pct:.2f}% vs historical avg {avg_rally_before_reversal:.2f}%")
        elif current_rally_pct < avg_rally_before_reversal * 0.5:
            recommendation_score += 1
            reasons.append(f"✅ Low reversal risk: Rally at {current_rally_pct:.2f}% with room to {avg_rally_before_reversal:.2f}%")
    
    # 6. Momentum
    recent_change = data1['Close'].pct_change(5).iloc[-1] * 100
    if recent_change > 3:
        recommendation_score += 1
        reasons.append(f"✅ Strong positive momentum: +{recent_change:.2f}% over last 5 periods")
    elif recent_change < -3:
        recommendation_score -= 1
        reasons.append(f"⚠️ Negative momentum: {recent_change:.2f}% over last 5 periods")
    
    # Final Recommendation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if recommendation_score >= 4:
            st.success("### 🚀 STRONG BUY")
            action = "STRONG BUY"
            color = "green"
        elif recommendation_score >= 2:
            st.success("### ✅ BUY")
            action = "BUY"
            color = "lightgreen"
        elif recommendation_score >= 0:
            st.info("### ➖ HOLD/NEUTRAL")
            action = "HOLD"
            color = "gray"
        elif recommendation_score >= -2:
            st.warning("### ⚠️ SELL")
            action = "SELL"
            color = "orange"
        else:
            st.error("### 🔴 STRONG SELL")
            action = "STRONG SELL"
            color = "red"
        
        st.metric("Recommendation Score", f"{recommendation_score:.1f}/10")
    
    st.markdown("---")
    
    # Detailed recommendation
    st.subheader("📋 Detailed Recommendation Breakdown")
    
    for reason in reasons:
        st.write(reason)
    
    st.markdown("---")
    
    # Quantitative targets
    st.subheader("🎯 Price Targets & Risk Management")
    
    if 'forecast_price1' in locals():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=f"Target Price ({ticker1_name})",
                value=f"{forecast_price1:.2f}",
                delta=f"{forecast_points1:.2f} pts ({forecast_change1:.2f}%)"
            )
        
        with col2:
            # Calculate stop loss (2 standard deviations)
            stop_loss = current_price - (2 * data1['Close'].std())
            st.metric(
                label="Suggested Stop Loss",
                value=f"{stop_loss:.2f}",
                delta=f"-{current_price - stop_loss:.2f} pts ({-((current_price - stop_loss)/current_price)*100:.2f}%)"
            )
        
        with col3:
            # Risk-reward ratio
            risk = current_price - stop_loss
            reward = forecast_price1 - current_price
            rr_ratio = reward / risk if risk > 0 else 0
            st.metric(
                label="Risk-Reward Ratio",
                value=f"{rr_ratio:.2f}:1"
            )
    
    st.markdown("---")
    
    # Summary box
    st.subheader("📌 Executive Summary")
    summary_text = f"""
    **Analysis Timestamp:** {data1.index[-1].strftime('%Y-%m-%d %H:%M:%S')} IST
    
    **Current Market Status:**
    - {ticker1_name}: {current_price:.2f} ({pct_change1:+.2f}%)
    - {ticker2_name}: {current_price2:.2f} ({pct_change2:+.2f}%)
    - Ratio: {current_ratio_val:.4f}
    
    **Recommendation:** {action}
    **Confidence Score:** {recommendation_score:.1f}/10
    
    **Key Points:**
    """
    
    for i, reason in enumerate(reasons[:5], 1):
        summary_text += f"\n{i}. {reason}"
    
    if 'forecast_price1' in locals():
        summary_text += f"""
        
        **Expected Move:**
        - {ticker1_name}: {forecast_points1:+.2f} points ({forecast_change1:+.2f}%)
        - Target: {forecast_price1:.2f}
        - Stop Loss: {stop_loss:.2f}
        - Risk-Reward: {rr_ratio:.2f}:1
        """
    
    summary_text += f"""
    
    **⚠️ Disclaimer:** This analysis is based on historical patterns and statistical models. 
    Past performance does not guarantee future results. Always do your own research and consider 
    your risk tolerance before making trading decisions.
    """
    
    st.info(summary_text)
    
    # Additional insights
    st.markdown("---")
    st.subheader("💡 Additional Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Volume Analysis:**")
        avg_volume1 = data1['Volume'].mean()
        current_volume1 = data1['Volume'].iloc[-1]
        volume_ratio = current_volume1 / avg_volume1
        
        if volume_ratio > 1.5:
            st.write(f"🔊 High volume: {current_volume1:,.0f} ({volume_ratio:.1f}x avg) - Strong conviction")
        elif volume_ratio < 0.5:
            st.write(f"🔉 Low volume: {current_volume1:,.0f} ({volume_ratio:.1f}x avg) - Weak conviction")
        else:
            st.write(f"🔈 Normal volume: {current_volume1:,.0f} ({volume_ratio:.1f}x avg)")
    
    with col2:
        st.write("**Market Correlation:**")
        correlation = data1['Close'].corr(data2['Close'])
        st.write(f"Correlation between {ticker1_name} and {ticker2_name}: {correlation:.2f}")
        
        if abs(correlation) > 0.8:
            st.write("📊 Strong correlation - Markets moving together")
        elif abs(correlation) < 0.3:
            st.write("📊 Weak correlation - Independent movements")
    
    st.markdown("---")
    st.caption(f"Analysis completed at {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')} IST")
    st.caption("⚠️ This tool is for educational purposes only. Not financial advice.")

else:
    st.info("👆 Configure your analysis parameters in the sidebar and click 'Fetch Data & Analyze' to begin.")
    
    # Display example/help section
    with st.expander("📖 How to Use This Dashboard"):
        st.write("""
        ### Features:
        
        1. **Multi-Asset Support**: Analyze Nifty50, BankNifty, Sensex, Crypto, Commodities, Forex, and custom stocks
        2. **Multiple Timeframes**: From 1-minute to monthly data
        3. **Ratio Analysis**: Compare two assets and find trading opportunities
        4. **Pattern Detection**: AI-powered similarity matching with historical data
        5. **Reversal Forecasting**: Identify potential market reversals before they happen
        6. **RSI & Volatility**: Complete technical analysis with actionable insights
        7. **Heatmaps**: Visual representation of returns and volatility patterns
        8. **Data Export**: Download analysis results in CSV format
        9. **Ultimate Recommendation**: AI-powered trading signals with risk management
        
        ### Quick Start:
        
        1. Select two tickers from the sidebar (e.g., Nifty 50 vs Bank Nifty)
        2. Choose your timeframe and period
        3. Click "Fetch Data & Analyze"
        4. Review the comprehensive analysis and recommendations
        
        ### Tips:
        
        - Use longer periods (1y+) for reliable pattern detection
        - Compare correlated assets for ratio trading opportunities
        - Pay attention to reversal warnings when RSI is extreme
        - Consider risk-reward ratios before entering trades
        - Use stop losses as suggested by the analysis
        
        **⚠️ Important**: This tool respects yfinance API rate limits by fetching data only when you click the button.
        """)
    
    with st.expander("🎓 Understanding the Analysis"):
        st.write("""
        ### Key Metrics Explained:
        
        **RSI (Relative Strength Index)**
        - Below 30: Oversold (potential buy)
        - Above 70: Overbought (potential sell)
        - 30-70: Neutral zone
        
        **Ratio Analysis**
        - Compares relative strength of two assets
        - High ratio: Asset 1 outperforming Asset 2
        - Low ratio: Asset 2 outperforming Asset 1
        - Mean reversion opportunities when ratio is extreme
        
        **Volatility**
        - High volatility: Large price swings, higher risk
        - Low volatility: Consolidation, potential breakout
        - Annualized for comparison across timeframes
        
        **Pattern Similarity**
        - Uses correlation to find similar historical periods
        - Shows what happened after similar patterns
        - Higher correlation = more reliable forecast
        
        **Reversal Detection**
        - Identifies historical rally exhaustion points
        - Warns when current rally approaches historical limits
        - Helps with profit-taking decisions
        
        **Recommendation Score**
        - Combines all metrics into single score (-10 to +10)
        - Positive scores suggest buying
        - Negative scores suggest selling/caution
        - Based on multiple confirmation signals
        """)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Data is cached to respect API limits. Use the Fetch button to refresh.")
st.sidebar.caption("Powered by yfinance & Streamlit")
