import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import argrelextrema
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
        if df.index.tz is     
    # Detailed volatility-based rally analysis
    st.subheader("🎯 Volatility-Based Rally Analysis")
    
    # Find periods of high volatility and subsequent rallies
    vol_ratio_high = vol_ratio[vol_ratio > vol_ratio.quantile(0.75)].dropna()
    
    if len(vol_ratio_high) > 0:
        vol_rally_analysis = []
        for idx in vol_ratio_high.index[-5:]:
            idx_loc = ratio.index.get_loc(idx)
            if idx_loc < len(ratio) - 10:
                # Look at next 10 periods after high volatility
                future_idx = min(idx_loc + 10, len(ratio) - 1)
                
                vol_at_time = float(vol_ratio.loc[idx])
                ratio_at_time = float(ratio.iloc[idx_loc])
                ratio_future = float(ratio.iloc[future_idx])
                ratio_change_pts = ratio_future - ratio_at_time
                ratio_change_pct = (ratio_change_pts / ratio_at_time) * 100
                
                price1_at_time = float(data1['Close'].iloc[idx_loc])
                price2_at_time = float(data2['Close'].iloc[idx_loc])
                price1_future = float(data1['Close'].iloc[future_idx])
                price2_future = float(data2['Close'].iloc[future_idx])
                
                price1_change_pts = price1_future - price1_at_time
                price1_change_pct = (price1_change_pts / price1_at_time) * 100
                price2_change_pts = price2_future - price2_at_time
                price2_change_pct = (price2_change_pts / price2_at_time) * 100
                
                vol_rally_analysis.append({
                    'DateTime_IST': str(idx),
                    'Volatility_%': vol_at_time,
                    f'{ticker1_name}_Price': price1_at_time,
                    f'{ticker2_name}_Price': price2_at_time,
                    'Ratio': ratio_at_time,
                    f'{ticker1_name}_Rally_Pts': price1_change_pts,
                    f'{ticker1_name}_Rally_%': price1_change_pct,
                    f'{ticker2_name}_Rally_Pts': price2_change_pts,
                    f'{ticker2_name}_Rally_%': price2_change_pct,
                    'Ratio_Change_Pts': ratio_change_pts,
                    'Ratio_Change_%': ratio_change_pct
                })
        
        if vol_rally_analysis:
            vol_rally_df = pd.DataFrame(vol_rally_analysis)
            st.dataframe(vol_rally_df.style.background_gradient(subset=['Ratio_Change_%', f'{ticker1_name}_Rally_%'], cmap='RdYlGn'), 
                        use_container_width=True)
            
            # Detailed explanation
            avg_ratio_rally = vol_rally_df['Ratio_Change_%'].mean()
            avg_t1_rally = vol_rally_df[f'{ticker1_name}_Rally_%'].mean()
            
            st.write(f"""
            **📌 Historical High Volatility Rally Pattern:**
            
            **Key Historical Events:**
            
            Most recent high volatility period: **{vol_rally_df.iloc[-1]['DateTime_IST']}**
            - Volatility was at: **{vol_rally_df.iloc[-1]['Volatility_%']:.2f}%**
            - {ticker1_name} was at: **{vol_rally_df.iloc[-1][f'{ticker1_name}_Price']:.2f}**
            - {ticker2_name} was at: **{vol_rally_df.iloc[-1][f'{ticker2_name}_Price']:.2f}**
            - Ratio was at: **{vol_rally_df.iloc[-1]['Ratio']:.4f}**
            
            **What Happened Next (10 periods later):**
            - {ticker1_name} moved: **{vol_rally_df.iloc[-1][f'{ticker1_name}_Rally_Pts']:.2f} points** (**{vol_rally_df.iloc[-1][f'{ticker1_name}_Rally_%']:.2f}%**)
            - {ticker2_name} moved: **{vol_rally_df.iloc[-1][f'{ticker2_name}_Rally_Pts']:.2f} points** (**{vol_rally_df.iloc[-1][f'{ticker2_name}_Rally_%']:.2f}%**)
            - Ratio moved: **{vol_rally_df.iloc[-1]['Ratio_Change_Pts']:.4f} points** (**{vol_rally_df.iloc[-1]['Ratio_Change_%']:.2f}%**)
            
            **Average Pattern Across All High Volatility Events:**
            - Average {ticker1_name} move: **{avg_t1_rally:.2f}%**
            - Average Ratio move: **{avg_ratio_rally:.2f}%**
            
            **Current Market Status vs Historical Pattern:**
            - Current volatility: **{vol_ratio_current:.2f}%** vs Historical high vol avg: **{vol_ratio_high.mean():.2f}%**
            - Current {ticker1_name}: **{latest_price1:.2f}**
            - Current Ratio: **{current_ratio:.4f}**
            
            **Expected Scenario:**
            {f'🚀 If pattern repeats, expect {ticker1_name} to move approximately **{abs(latest_price1 * avg_t1_rally / 100):.2f} points** (**{avg_t1_rally:.2f}%**) in next 10 periods' if vol_ratio_current > vol_ratio.quantile(0.7) else '📊 Current volatility is moderate. Major moves less likely unless volatility increases'}
            
            **Trading Implication:**
            {f'⚠️ HIGH VOLATILITY - Expect large swings. Reduce position size or use wider stops' if vol_ratio_current > vol_ratio.quantile(0.75) else '✅ NORMAL VOLATILITY - Standard position sizing appropriate'}
            """)
    
    # ============= SECTION 7: VOLATILITY BINNING =============
    st.header("📊 7. Volatility Binning Analysis")
    
    try:
        vol_binned = points_data.dropna().copy()
        if len(vol_binned) >= 5 and vol_binned[f'Vol% {ticker1_name}'].nunique() >= 5:
            vol_binned[f'Vol_Bin'] = pd.qcut(vol_binned[f'Vol% {ticker1_name}'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            
            vol_analysis = vol_binned.groupby(f'Vol_Bin').agg({
                f'Points {ticker1_name}': ['mean', 'std', 'sum', 'count'],
                f'Points {ticker2_name}': ['mean', 'std', 'sum'],
                'Points Ratio': ['mean', 'std', 'sum'],
                f'% Change {ticker1_name}': ['mean'],
                f'% Change {ticker2_name}': ['mean'],
                '% Change Ratio': ['mean']
            }).round(4)
            
            st.dataframe(vol_analysis.style.background_gradient(cmap='coolwarm'), use_container_width=True)
            
            st.write("""
            **📌 Volatility Binning Insights:**
            
            This table categorizes market periods by volatility levels and shows typical price movements in each category.
            
            **Key Observations:**
            - **Very High Volatility**: Large absolute point movements (both up and down)
            - **Very Low Volatility**: Small movements, consolidation phase
            - **Medium Volatility**: Normal trading conditions
            
            **How to Use:**
            1. Check current volatility level
            2. Review typical movements in that volatility bin
            3. Adjust position sizing: smaller in high vol, larger in low vol
            4. Set stop losses based on typical movements in current volatility regime
            """)
    except Exception as e:
        st.warning(f"Unable to create volatility bins: {str(e)}")
    
    # ============= SECTION 8: RETURNS HEATMAPS =============
    st.header("🔥 8. Returns Heatmaps")
    
    returns1_daily = data1['Close'].pct_change() * 100
    returns2_daily = data2['Close'].pct_change() * 100
    ratio_returns_daily = ratio.pct_change() * 100
    
    if len(data1) >= 50:
        try:
            # Create properly labeled heatmaps
            st.subheader(f"{ticker1_name} Returns Heatmap")
            returns1_recent = returns1_daily.tail(50).values
            n_rows = len(returns1_recent) // 10
            if n_rows > 0:
                returns1_matrix = returns1_recent[:n_rows*10].reshape(n_rows, 10)
                fig1 = go.Figure(data=go.Heatmap(
                    z=returns1_matrix, 
                    colorscale='RdYlGn', 
                    zmid=0,
                    text=np.round(returns1_matrix, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Return %")
                ))
                fig1.update_layout(
                    title=f'{ticker1_name} Returns Pattern (% Change)',
                    xaxis_title='Period',
                    yaxis_title='Week',
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader(f"{ticker2_name} Returns Heatmap")
            returns2_recent = returns2_daily.tail(50).values
            if n_rows > 0:
                returns2_matrix = returns2_recent[:n_rows*10].reshape(n_rows, 10)
                fig2 = go.Figure(data=go.Heatmap(
                    z=returns2_matrix, 
                    colorscale='RdYlGn', 
                    zmid=0,
                    text=np.round(returns2_matrix, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Return %")
                ))
                fig2.update_layout(
                    title=f'{ticker2_name} Returns Pattern (% Change)',
                    xaxis_title='Period',
                    yaxis_title='Week',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("Ratio Returns Heatmap")
            ratio_returns_recent = ratio_returns_daily.tail(50).values
            if n_rows > 0:
                ratio_returns_matrix = ratio_returns_recent[:n_rows*10].reshape(n_rows, 10)
                fig3 = go.Figure(data=go.Heatmap(
                    z=ratio_returns_matrix, 
                    colorscale='RdYlGn', 
                    zmid=0,
                    text=np.round(ratio_returns_matrix, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Return %")
                ))
                fig3.update_layout(
                    title='Ratio Returns Pattern (% Change)',
                    xaxis_title='Period',
                    yaxis_title='Week',
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # Calculate insights
            positive_returns1 = (returns1_daily > 0).sum()
            negative_returns1 = (returns1_daily < 0).sum()
            avg_positive1 = returns1_daily[returns1_daily > 0].mean()
            avg_negative1 = returns1_daily[returns1_daily < 0].mean()
            
            st.write(f"""
            **📌 Returns Heatmap Insights:**
            
            **{ticker1_name} Pattern Analysis:**
            - Positive return periods: **{positive_returns1}** (Avg: **{avg_positive1:.2f}%**)
            - Negative return periods: **{negative_returns1}** (Avg: **{avg_negative1:.2f}%**)
            - Win rate: **{(positive_returns1 / (positive_returns1 + negative_returns1) * 100):.1f}%**
            
            **Visual Pattern Recognition:**
            - 🟢 **Green clusters**: Periods of sustained gains - optimal for trend following
            - 🔴 **Red clusters**: Drawdown periods - time for defensive positioning
            - 🟡 **Mixed patterns**: Choppy market - range-bound strategies
            
            **Current Market Context:**
            - Latest return: **{returns1_daily.iloc[-1]:.2f}%**
            - Last 5 periods avg: **{returns1_daily.tail(5).mean():.2f}%**
            
            **Trading Opportunity:**
            {f'✅ Recent green pattern suggests momentum - Consider LONG positions' if returns1_daily.tail(5).mean() > 1 else '⚠️ Recent red/mixed pattern suggests caution - Wait for confirmation' if returns1_daily.tail(5).mean() < -1 else '➖ Neutral pattern - No clear directional bias'}
            """)
        except Exception as e:
            st.info(f"Not enough data points for heatmap visualization: {str(e)}")
    
    # ============= SECTION 9: PATTERN SIMILARITY =============
    st.header("🔍 9. Pattern Similarity Detection & Forecast")
    
    similarities1 = find_pattern_similarity(data1['Close'], window=min(20, len(data1)//4))
    similarities2 = find_pattern_similarity(data2['Close'], window=min(20, len(data2)//4))
    similarities_ratio = find_pattern_similarity(ratio, window=min(20, len(ratio)//4))
    
    if similarities1:
        st.subheader(f"🎯 {ticker1_name} Similar Patterns & Forecast")
        sim_data1 = []
        for sim in similarities1:
            sim_data1.append({
                'DateTime_IST': str(sim['date']),
                'Price_Then': sim['price_at_pattern'],
                'Correlation': sim['correlation'],
                'Future_Price': sim['future_price'],
                'Future_Points': sim['future_points'],
                'Future_%': sim['future_return_pct']
            })
        
        sim_df1 = pd.DataFrame(sim_data1)
        st.dataframe(sim_df1.style.background_gradient(subset=['Correlation', 'Future_%'], cmap='RdYlGn'), use_container_width=True)
        
        avg_future_pts = sim_df1['Future_Points'].mean()
        avg_future_pct = sim_df1['Future_%'].mean()
        forecast_price1 = latest_price1 + avg_future_pts
        
        st.write(f"""
        **📌 {ticker1_name} Pattern-Based Forecast:**
        
        **Historical Similar Patterns Found: {len(similarities1)}**
        
        **Most Similar Pattern (Highest Correlation):**
        - Occurred on: **{sim_df1.iloc[0]['DateTime_IST']}** IST
        - Price then: **{sim_df1.iloc[0]['Price_Then']:.2f}**
        - Correlation: **{sim_df1.iloc[0]['Correlation']*100:.1f}%**
        - What happened next: Moved **{sim_df1.iloc[0]['Future_Points']:.2f} points** (**{sim_df1.iloc[0]['Future_%']:.2f}%**)
        - Price reached: **{sim_df1.iloc[0]['Future_Price']:.2f}**
        
        **Average Across All Similar Patterns:**
        - Average move: **{avg_future_pts:.2f} points** (**{avg_future_pct:.2f}%**)
        - Historical dates: {', '.join([str(sim_df1.iloc[i]['DateTime_IST']) for i in range(min(3, len(sim_df1)))])}
        
        **FORECAST for Current Market:**
        - Current {ticker1_name} price: **{latest_price1:.2f}**
        - Expected target: **{forecast_price1:.2f}**
        - Expected move: **{avg_future_pts:+.2f} points** (**{avg_future_pct:+.2f}%**)
        - Confidence: **{sim_df1['Correlation'].mean()*100:.1f}%** (based on pattern correlation)
        
        **Interpretation:**
        {f'🚀 Strong BULLISH forecast - Historical similar patterns showed **{avg_future_pct:.2f}%** gains averaging **{avg_future_pts:.2f} points**' if avg_future_pct > 2 else f'🔴 BEARISH forecast - Historical patterns suggest **{avg_future_pct:.2f}%** decline (**{avg_future_pts:.2f} points**)' if avg_future_pct < -2 else f'➖ NEUTRAL forecast - Expect sideways movement around current levels'}
        """)
    
    if similarities_ratio:
        st.subheader("🎯 Ratio Similar Patterns & Forecast")
        sim_data_ratio = []
        for sim in similarities_ratio:
            sim_data_ratio.append({
                'DateTime_IST': str(sim['date']),
                'Ratio_Then': sim['price_at_pattern'],
                'Correlation': sim['correlation'],
                'Future_Ratio': sim['future_price'],
                'Future_Points': sim['future_points'],
                'Future_%': sim['future_return_pct']
            })
        
        sim_df_ratio = pd.DataFrame(sim_data_ratio)
        st.dataframe(sim_df_ratio.style.background_gradient(subset=['Correlation', 'Future_%'], cmap='RdYlGn'), use_container_width=True)
        
        avg_ratio_pts = sim_df_ratio['Future_Points'].mean()
        avg_ratio_pct = sim_df_ratio['Future_%'].mean()
        forecast_ratio = current_ratio + avg_ratio_pts
        
        st.write(f"""
        **📌 Ratio Pattern-Based Forecast:**
        
        **Historical Similar Ratio Patterns Found: {len(similarities_ratio)}**
        
        **Most Similar Pattern:**
        - Occurred on: **{sim_df_ratio.iloc[0]['DateTime_IST']}** IST
        - Ratio then: **{sim_df_ratio.iloc[0]['Ratio_Then']:.4f}**
        - Correlation: **{sim_df_ratio.iloc[0]['Correlation']*100:.1f}%**
        - What happened: Moved **{sim_df_ratio.iloc[0]['Future_Points']:.4f} points** (**{sim_df_ratio.iloc[0]['Future_%']:.2f}%**)
        
        **Average Pattern Outcome:**
        - Average move: **{avg_ratio_pts:.4f} points** (**{avg_ratio_pct:.2f}%**)
        
        **FORECAST for Current Ratio:**
        - Current ratio: **{current_ratio:.4f}**
        - Expected ratio: **{forecast_ratio:.4f}**
        - Expected move: **{avg_ratio_pts:+.4f} points** (**{avg_ratio_pct:+.2f}%**)
        
        **What This Means:**
        {f'✅ {ticker1_name} expected to OUTPERFORM {ticker2_name} by **{avg_ratio_pct:.2f}%**' if avg_ratio_pct > 1 else f'⚠️ {ticker2_name} expected to OUTPERFORM {ticker1_name} by **{abs(avg_ratio_pct):.2f}%**' if avg_ratio_pct < -1 else f'➖ Both assets expected to move in tandem'}
        
        **Trading Strategy:**
        {f'📈 BUY {ticker1_name}, SELL/SHORT {ticker2_name} (pairs trade)' if avg_ratio_pct > 2 else f'📉 SELL/SHORT {ticker1_name}, BUY {ticker2_name} (pairs trade)' if avg_ratio_pct < -2 else '➖ No clear pairs trading opportunity'}
        """)
    
    # ============= SECTION 10: VOLATILITY HEATMAPS =============
    st.header("🌡️ 10. Volatility Heatmaps")
    
    if len(vol1.dropna()) >= 25:
        try:
            vol1_recent = vol1.dropna().tail(50).values
            n_rows_vol = len(vol1_recent) // 10
            if n_rows_vol > 0:
                vol1_matrix = vol1_recent[:n_rows_vol*10].reshape(n_rows_vol, 10)
                fig_vol1 = go.Figure(data=go.Heatmap(
                    z=vol1_matrix, 
                    colorscale='Reds',
                    text=np.round(vol1_matrix, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Vol %")
                ))
                fig_vol1.update_layout(
                    title=f'{ticker1_name} Volatility Heatmap (%)',
                    xaxis_title='Period',
                    yaxis_title='Week',
                    height=400
                )
                st.plotly_chart(fig_vol1, use_container_width=True)
            
            vol2_recent = vol2.dropna().tail(50).values
            if n_rows_vol > 0 and len(vol2_recent) >= n_rows_vol * 10:
                vol2_matrix = vol2_recent[:n_rows_vol*10].reshape(n_rows_vol, 10)
                fig_vol2 = go.Figure(data=go.Heatmap(
                    z=vol2_matrix, 
                    colorscale='Reds',
                    text=np.round(vol2_matrix, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Vol %")
                ))
                fig_vol2.update_layout(
                    title=f'{ticker2_name} Volatility Heatmap (%)',
                    xaxis_title='Period',
                    yaxis_title='Week',
                    height=400
                )
                st.plotly_chart(fig_vol2, use_container_width=True)
            
            st.write(f"""
            **📌 Volatility Heatmap Insights:**
            
            **Volatility Pattern Analysis:**
            - 🔴 **Dark Red zones**: High volatility (**>{vol1.quantile(0.75):.2f}%**) - Periods of uncertainty, large price swings
            - 🟠 **Orange zones**: Moderate volatility - Normal market conditions  
            - ⚪ **Light zones**: Low volatility (**<{vol1.quantile(0.25):.2f}%**) - Consolidation, potential breakout setup
            
            **Current Volatility Status:**
            - {ticker1_name}: **{vol1_current:.2f}%** - {:
            df.index = df.index.tz_localize('UTC').tz_convert(ist)
        else:
            df.index = df.index.tz_convert(ist)
        df.index = df.index.tz_localize(None)
    except:
        pass
    return df

def align_data(data1, data2):
    """Align two dataframes by common index"""
    common_index = data1.index.intersection(data2.index)
    return data1.loc[common_index], data2.loc[common_index]

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
    volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100  # As percentage
    return volatility

def find_pattern_similarity(data, window=20, top_n=5):
    """Find similar patterns based on normalized percentage changes"""
    if len(data) < window * 2:
        return []
    
    # Use percentage changes for scale-invariant comparison
    pct_changes = data.pct_change().fillna(0)
    current_pattern = pct_changes.iloc[-window:].values
    current_pattern_norm = (current_pattern - current_pattern.mean()) / (current_pattern.std() + 1e-8)
    
    similarities = []
    for i in range(window, len(data) - window):
        historical_pattern = pct_changes.iloc[i-window:i].values
        historical_pattern_norm = (historical_pattern - historical_pattern.mean()) / (historical_pattern.std() + 1e-8)
        
        correlation = np.corrcoef(current_pattern_norm, historical_pattern_norm)[0, 1]
        if not np.isnan(correlation) and correlation > 0.7:  # Only strong correlations
            future_points = float(data.iloc[i+window] - data.iloc[i])
            future_pct = (future_points / float(data.iloc[i])) * 100
            
            similarities.append({
                'index': i,
                'date': data.index[i],
                'correlation': correlation,
                'price_at_pattern': float(data.iloc[i]),
                'future_price': float(data.iloc[i+window]) if i+window < len(data) else float(data.iloc[i]),
                'future_points': future_points,
                'future_return_pct': future_pct
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
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk"])
    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
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
            
            # Handle multi-index columns if present
            if isinstance(data1.columns, pd.MultiIndex):
                data1.columns = data1.columns.get_level_values(0)
            if isinstance(data2.columns, pd.MultiIndex):
                data2.columns = data2.columns.get_level_values(0)
            
            # Convert to IST
            data1 = convert_to_ist(data1)
            data2 = convert_to_ist(data2)
            
            # Align data by common timestamps
            data1, data2 = align_data(data1, data2)
            
            # Ensure data is sorted
            data1 = data1.sort_index()
            data2 = data2.sort_index()
            
            # Check if Volume exists, if not add zeros
            if 'Volume' not in data1.columns:
                data1['Volume'] = 0
            if 'Volume' not in data2.columns:
                data2['Volume'] = 0
            
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
    data1 = st.session_state.data1.copy()
    data2 = st.session_state.data2.copy()
    ticker1 = st.session_state.ticker1
    ticker2 = st.session_state.ticker2
    ticker1_name = st.session_state.ticker1_name
    ticker2_name = st.session_state.ticker2_name
    
    # Calculate ratio
    ratio = data1['Close'] / data2['Close']
    
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
        latest_price1 = float(data1['Close'].iloc[-1])
        prev_price1 = float(data1['Close'].iloc[-2]) if len(data1) > 1 else latest_price1
        change1 = latest_price1 - prev_price1
        pct_change1 = (change1 / prev_price1) * 100
        rsi1_current = float(rsi1.iloc[-1]) if not pd.isna(rsi1.iloc[-1]) else 50.0
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Latest Price", f"{latest_price1:.2f}")
        metric_col2.metric("Change", f"{change1:.2f}", f"{pct_change1:.2f}%")
        metric_col3.metric("RSI", f"{rsi1_current:.2f}")
        
        # Basic stats table
        stats1_data = {
            'DateTime': [str(dt) for dt in data1.index[-10:]],
            'Close': [float(x) for x in data1['Close'].iloc[-10:].values],
            'Change': [float(x) if not pd.isna(x) else 0.0 for x in data1['Close'].iloc[-10:].diff().values],
            '% Change': [float(x) if not pd.isna(x) else 0.0 for x in (data1['Close'].iloc[-10:].pct_change() * 100).values]
        }
        stats1 = pd.DataFrame(stats1_data)
        st.dataframe(stats1.style.background_gradient(subset=['Change', '% Change'], cmap='RdYlGn'), use_container_width=True)
    
    with col2:
        st.subheader(f"{ticker2_name} Analysis")
        latest_price2 = float(data2['Close'].iloc[-1])
        prev_price2 = float(data2['Close'].iloc[-2]) if len(data2) > 1 else latest_price2
        change2 = latest_price2 - prev_price2
        pct_change2 = (change2 / prev_price2) * 100
        rsi2_current = float(rsi2.iloc[-1]) if not pd.isna(rsi2.iloc[-1]) else 50.0
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Latest Price", f"{latest_price2:.2f}")
        metric_col2.metric("Change", f"{change2:.2f}", f"{pct_change2:.2f}%")
        metric_col3.metric("RSI", f"{rsi2_current:.2f}")
        
        # Basic stats table
        stats2_data = {
            'DateTime': [str(dt) for dt in data2.index[-10:]],
            'Close': [float(x) for x in data2['Close'].iloc[-10:].values],
            'Change': [float(x) if not pd.isna(x) else 0.0 for x in data2['Close'].iloc[-10:].diff().values],
            '% Change': [float(x) if not pd.isna(x) else 0.0 for x in (data2['Close'].iloc[-10:].pct_change() * 100).values]
        }
        stats2 = pd.DataFrame(stats2_data)
        st.dataframe(stats2.style.background_gradient(subset=['Change', '% Change'], cmap='RdYlGn'), use_container_width=True)
    
    # ============= SECTION 2: RATIO ANALYSIS =============
    st.header("🔀 2. Ratio Analysis")
    
    # Ensure all arrays have same length
    min_len = min(len(data1), len(data2), len(ratio), len(rsi1), len(rsi2), len(rsi_ratio))
    
    ratio_analysis = pd.DataFrame({
        'DateTime': [str(dt) for dt in data1.index[-min_len:]],
        f'{ticker1_name} Price': [float(x) for x in data1['Close'].iloc[-min_len:].values],
        f'{ticker2_name} Price': [float(x) for x in data2['Close'].iloc[-min_len:].values],
        'Ratio': [float(x) for x in ratio.iloc[-min_len:].values],
        f'RSI {ticker1_name}': [float(x) if not pd.isna(x) else 50.0 for x in rsi1.iloc[-min_len:].values],
        f'RSI {ticker2_name}': [float(x) if not pd.isna(x) else 50.0 for x in rsi2.iloc[-min_len:].values],
        'RSI Ratio': [float(x) if not pd.isna(x) else 50.0 for x in rsi_ratio.iloc[-min_len:].values]
    })
    
    st.dataframe(ratio_analysis.tail(20).style.background_gradient(subset=['Ratio', 'RSI Ratio'], cmap='viridis'), use_container_width=True)
    
    # Ratio insights
    st.subheader("📊 Ratio Insights")
    current_ratio = float(ratio.iloc[-1])
    mean_ratio = float(ratio.mean())
    std_ratio = float(ratio.std())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Ratio", f"{current_ratio:.4f}")
    col2.metric("Mean Ratio", f"{mean_ratio:.4f}")
    col3.metric("Std Dev", f"{std_ratio:.4f}")
    
    # Find historical rallies in ratio
    ratio_changes = ratio.pct_change()
    rally_threshold = float(ratio_changes.quantile(0.9))
    rallies = ratio_changes[ratio_changes > rally_threshold]
    
    if len(rallies) > 0:
        st.write("### 🚀 Historical Ratio Rallies Detected:")
        rally_data = []
        for idx in rallies.index[-5:]:
            idx_loc = ratio.index.get_loc(idx)
            if idx_loc < len(ratio) - 1:
                ratio_before = float(ratio.iloc[idx_loc])
                ratio_after = float(ratio.iloc[idx_loc+1])
                future_change_pts = ratio_after - ratio_before
                future_change_pct = (future_change_pts / ratio_before) * 100
                
                rally_data.append({
                    'DateTime_IST': str(idx),
                    'Ratio Value': ratio_before,
                    'Change %': float(rallies.loc[idx]) * 100,
                    'Next Period Pts': future_change_pts,
                    'Next Period %': future_change_pct,
                    f'{ticker1_name} Price': float(data1['Close'].loc[idx]),
                    f'{ticker2_name} Price': float(data2['Close'].loc[idx])
                })
        
        if rally_data:
            rally_df = pd.DataFrame(rally_data)
            st.dataframe(rally_df.style.background_gradient(subset=['Change %', 'Next Period %'], cmap='Greens'), use_container_width=True)
            
            # Detailed explanation
            st.write(f"""
            ** Key Insight - Historical Ratio Rally Analysis:**
            
            Historical data shows **{len(rallies)} significant ratio movements** (>{rally_threshold*100:.2f}%).
            
            **Current Market Status:**
            - Current Ratio: **{current_ratio:.4f}**
            - Mean Ratio: **{mean_ratio:.4f}**
            - The ratio is currently **{abs((current_ratio - mean_ratio) / mean_ratio * 100):.2f}%** {'above' if current_ratio > mean_ratio else 'below'} the historical mean
            - Current {ticker1_name} price: **{latest_price1:.2f}** ({change1:+.2f} pts, {pct_change1:+.2f}%)
            - Current {ticker2_name} price: **{latest_price2:.2f}** ({change2:+.2f} pts, {pct_change2:+.2f}%)
            
            **Historical Rally Pattern:**
            The most recent strong rally occurred on **{rally_df.iloc[-1]['DateTime_IST']}** when:
            - Ratio was at **{rally_df.iloc[-1]['Ratio Value']:.4f}**
            - {ticker1_name} was at **{rally_df.iloc[-1][f'{ticker1_name} Price']:.2f}**
            - {ticker2_name} was at **{rally_df.iloc[-1][f'{ticker2_name} Price']:.2f}**
            - Rally magnitude: **{rally_df.iloc[-1]['Change %']:.2f}%**
            - After rally, moved **{rally_df.iloc[-1]['Next Period Pts']:.4f} points** (**{rally_df.iloc[-1]['Next Period %']:.2f}%**)
            
            **Market Implication:**
            {'⚠️ CAUTION: Ratio approaching historical high - potential mean reversion expected' if current_ratio > mean_ratio + std_ratio else '✅ OPPORTUNITY: Ratio below mean - potential upside in ' + ticker1_name + ' relative to ' + ticker2_name}
            """)
    
    # ============= SECTION 3: RATIO BINNING ANALYSIS =============
    st.header("📊 3. Ratio Binning & Rally Analysis")
    
    try:
        ratio_clean = ratio.dropna()
        if len(ratio_clean) >= 10:
            ratio_bins = pd.qcut(ratio_clean, q=10, duplicates='drop', labels=False)
            ratio_binned_df = pd.DataFrame({
                'Ratio': ratio_clean.values,
                'Bin': ratio_bins
            }, index=ratio_clean.index)
            
            # Calculate forward returns
            ratio_binned_df['Forward_Return_%'] = ratio_clean.pct_change().shift(-1) * 100
            ratio_binned_df['Forward_Points'] = ratio_clean.diff().shift(-1)
            
            bin_analysis = ratio_binned_df.groupby('Bin').agg({
                'Forward_Return_%': ['mean', 'std', 'count'],
                'Forward_Points': ['mean', 'sum'],
                'Ratio': ['min', 'max', 'mean']
            }).round(4)
            
            st.dataframe(bin_analysis.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
            
            # Find current bin
            current_bin = pd.qcut([current_ratio], q=10, labels=False, duplicates='drop')[0] if current_ratio >= ratio_clean.min() and current_ratio <= ratio_clean.max() else None
            
            st.write(f"""
            **📌 Bin Analysis Insights:**
            
            This table divides the ratio into 10 equal bins and shows average forward movement in each bin.
            
            **Current Position:**
            - Current ratio **{current_ratio:.4f}** falls in Bin **{current_bin if current_bin is not None else 'N/A'}**
            - Historical forward return in this bin: **{float(bin_analysis.loc[current_bin, ('Forward_Return_%', 'mean')]):.2f}%** if current_bin is not None else 'N/A'
            - Historical forward points in this bin: **{float(bin_analysis.loc[current_bin, ('Forward_Points', 'mean')]):.4f}** if current_bin is not None else 'N/A'
            
            **Key Observations:**
            - **Best performing bins** (highest forward returns) indicate optimal buying zones
            - **Worst performing bins** suggest profit-taking or shorting opportunities
            - Bins with positive mean returns and low std suggest consistent profitable patterns
            
            **Actionable Insight:**
            {'✅ Current bin shows positive historical returns - Consider LONG position' if current_bin is not None and float(bin_analysis.loc[current_bin, ('Forward_Return_%', 'mean')]) > 0 else '⚠️ Current bin shows negative historical returns - Consider CAUTION or SHORT position'}
            """)
    except Exception as e:
        st.warning(f"Unable to create bins: {str(e)}")
    
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
    
    # Ratio as line chart
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
        'DateTime_IST': [str(dt) for dt in data1.index],
        f'{ticker1_name}_Open': [float(x) for x in data1['Open'].values],
        f'{ticker1_name}_High': [float(x) for x in data1['High'].values],
        f'{ticker1_name}_Low': [float(x) for x in data1['Low'].values],
        f'{ticker1_name}_Close': [float(x) for x in data1['Close'].values],
        f'{ticker1_name}_Volume': [float(x) for x in data1['Volume'].values],
        f'{ticker2_name}_Open': [float(x) for x in data2['Open'].values],
        f'{ticker2_name}_High': [float(x) for x in data2['High'].values],
        f'{ticker2_name}_Low': [float(x) for x in data2['Low'].values],
        f'{ticker2_name}_Close': [float(x) for x in data2['Close'].values],
        f'{ticker2_name}_Volume': [float(x) for x in data2['Volume'].values],
        'Ratio': [float(x) for x in ratio.values]
    })
    
    col1, col2 = st.columns(2)
    with col1:
        csv = export_data.to_csv(index=False)
        st.download_button("📥 Download as CSV", csv, "trading_data.csv", "text/csv")
    with col2:
        st.download_button("📥 Download as Excel (CSV)", csv, "trading_data.xlsx", "text/csv")
    
    # ============= SECTION 6: VOLATILITY ANALYSIS =============
    st.header("📊 6. Volatility & Points Analysis")
    
    # Ensure consistent lengths
    points_len = min(len(data1), len(data2), len(ratio), len(vol1), len(vol2), len(vol_ratio))
    
    points_data = pd.DataFrame({
        'DateTime_IST': [str(dt) for dt in data1.index[-points_len:]],
        f'{ticker1_name} Price': [float(x) for x in data1['Close'].iloc[-points_len:].values],
        f'{ticker2_name} Price': [float(x) for x in data2['Close'].iloc[-points_len:].values],
        'Ratio': [float(x) for x in ratio.iloc[-points_len:].values],
        f'Vol% {ticker1_name}': [float(x) if not pd.isna(x) else 0.0 for x in vol1.iloc[-points_len:].values],
        f'Vol% {ticker2_name}': [float(x) if not pd.isna(x) else 0.0 for x in vol2.iloc[-points_len:].values],
        'Vol% Ratio': [float(x) if not pd.isna(x) else 0.0 for x in vol_ratio.iloc[-points_len:].values],
        f'Points {ticker1_name}': [float(x) if not pd.isna(x) else 0.0 for x in data1['Close'].iloc[-points_len:].diff().values],
        f'Points {ticker2_name}': [float(x) if not pd.isna(x) else 0.0 for x in data2['Close'].iloc[-points_len:].diff().values],
        'Points Ratio': [float(x) if not pd.isna(x) else 0.0 for x in ratio.iloc[-points_len:].diff().values],
        f'% Change {ticker1_name}': [float(x) if not pd.isna(x) else 0.0 for x in (data1['Close'].iloc[-points_len:].pct_change() * 100).values],
        f'% Change {ticker2_name}': [float(x) if not pd.isna(x) else 0.0 for x in (data2['Close'].iloc[-points_len:].pct_change() * 100).values],
        '% Change Ratio': [float(x) if not pd.isna(x) else 0.0 for x in (ratio.iloc[-points_len:].pct_change() * 100).values]
    })
    
    st.dataframe(points_data.tail(20).style.background_gradient(subset=[f'Points {ticker1_name}', f'Points {ticker2_name}', 'Points Ratio'], cmap='RdYlGn'), 
                use_container_width=True)
    
    vol1_current = float(vol1.iloc[-1]) if not pd.isna(vol1.iloc[-1]) else 0.0
    vol2_current = float(vol2.iloc[-1]) if not pd.isna(vol2.iloc[-1]) else 0.0
    vol_ratio_current = float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else 0.0
    
    st.write(f"""
    **📌 Volatility & Movement Insights:** 
    
    **Current Volatility (Annualized %):**
    - {ticker1_name}: **{vol1_current:.2f}%** (annualized)
    - {ticker2_name}: **{vol2_current:.2f}%** (annualized)
    - Ratio: **{vol_ratio_current:.2f}%** (annualized)
    
    **Recent Movement (Latest Period):**
    - {ticker1_name}: **{points_data.iloc[-1][f'Points {ticker1_name}']:.2f} points** (**{points_data.iloc[-1][f'% Change {ticker1_name}']:.2f}%**)
    - {ticker2_name}: **{points_data.iloc[-1][f'Points {ticker2_name}']:.2f} points** (**{points_data.iloc[-1][f'% Change {ticker2_name}']:.2f}%**)
    - Ratio: **{points_data.iloc[-1]['Points Ratio']:.4f} points** (**{points_data.iloc[-1]['% Change Ratio']:.2f}%**)
    
    **Interpretation:**
    Higher volatility indicates higher risk but also potential for larger moves. Use volatility to size positions appropriately.
    """)
