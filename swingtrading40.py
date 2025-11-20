import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Advanced Trading Analyzer", layout="wide")

# Title
st.title("🚀 Advanced Trading Pattern & Forecasting Analyzer")
st.markdown("*AI-Powered Pattern Recognition for Precise Market Forecasting*")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Ticker mapping
TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BITCOIN": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "USD/INR": "INR=X",
    "Custom": ""
}

# Ticker 1 selection
ticker1_type = st.sidebar.selectbox("Select Ticker 1", list(TICKER_MAP.keys()))
if ticker1_type == "Custom":
    ticker1 = st.sidebar.text_input("Enter Ticker 1 Symbol", "AAPL")
else:
    ticker1 = TICKER_MAP[ticker1_type]

# Ticker 2 selection
ticker2_type = st.sidebar.selectbox("Select Ticker 2", list(TICKER_MAP.keys()), index=1)
if ticker2_type == "Custom":
    ticker2 = st.sidebar.text_input("Enter Ticker 2 Symbol", "MSFT")
else:
    ticker2 = TICKER_MAP[ticker2_type]

# Timeframe and period
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d"]
)

period = st.sidebar.selectbox(
    "Period",
    ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "25y", "30y"]
)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None

# Helper functions
def safe_get_column(df, column_name):
    """Safely extract column values handling MultiIndex"""
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # For MultiIndex, get the first level
            col_data = df.xs(column_name, axis=1, level=0)
            if isinstance(col_data, pd.DataFrame):
                return col_data.iloc[:, 0].values
            return col_data.values
        else:
            return df[column_name].values
    except Exception as e:
        st.error(f"Error extracting {column_name}: {str(e)}")
        return np.array([])

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    if len(data) < period:
        return pd.Series([np.nan] * len(data))
    delta = pd.Series(data).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volatility(data, window=20):
    """Calculate rolling volatility"""
    if len(data) < window:
        return np.array([np.nan] * len(data))
    returns = pd.Series(data).pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility.values

def fit_distribution(data, name="Returns"):
    """Fit normal distribution to data and return statistics"""
    clean_data = data[~np.isnan(data)]
    if len(clean_data) < 2:
        return None
    
    mu, std = stats.norm.fit(clean_data)
    
    # Current value percentile
    current_val = clean_data[-1] if len(clean_data) > 0 else 0
    percentile = stats.percentileofscore(clean_data, current_val)
    
    # Probability ranges
    prob_positive = 1 - stats.norm.cdf(0, mu, std)
    prob_extreme_positive = 1 - stats.norm.cdf(mu + 2*std, mu, std)
    prob_extreme_negative = stats.norm.cdf(mu - 2*std, mu, std)
    
    return {
        'name': name,
        'mean': mu,
        'std': std,
        'current_value': current_val,
        'current_percentile': percentile,
        'prob_positive': prob_positive * 100,
        'prob_extreme_positive': prob_extreme_positive * 100,
        'prob_extreme_negative': prob_extreme_negative * 100,
        'data': clean_data
    }

# Fetch data button
if st.sidebar.button("📊 Fetch Data & Analyze", type="primary"):
    with st.spinner("Fetching data from Yahoo Finance..."):
        try:
            # Fetch data
            data1 = yf.download(ticker1, period=period, interval=timeframe, progress=False)
            data2 = yf.download(ticker2, period=period, interval=timeframe, progress=False)
            
            if data1.empty or data2.empty:
                st.error("❌ Failed to fetch data. Please check ticker symbols and try again.")
            else:
                # Convert to IST
                ist = pytz.timezone('Asia/Kolkata')
                
                # Handle timezone conversion properly
                if data1.index.tz is None:
                    data1.index = data1.index.tz_localize('UTC').tz_convert(ist)
                else:
                    data1.index = data1.index.tz_convert(ist)
                
                if data2.index.tz is None:
                    data2.index = data2.index.tz_localize('UTC').tz_convert(ist)
                else:
                    data2.index = data2.index.tz_convert(ist)
                
                st.session_state.df1 = data1
                st.session_state.df2 = data2
                st.session_state.data_fetched = True
                st.success("✅ Data fetched successfully!")
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")

# Main analysis
if st.session_state.data_fetched:
    df1 = st.session_state.df1.copy()
    df2 = st.session_state.df2.copy()
    
    # Extract values safely
    close1 = safe_get_column(df1, 'Close')
    close2 = safe_get_column(df2, 'Close')
    open1 = safe_get_column(df1, 'Open')
    open2 = safe_get_column(df2, 'Open')
    high1 = safe_get_column(df1, 'High')
    high2 = safe_get_column(df2, 'High')
    low1 = safe_get_column(df1, 'Low')
    low2 = safe_get_column(df2, 'Low')
    volume1 = safe_get_column(df1, 'Volume')
    volume2 = safe_get_column(df2, 'Volume')
    
    # Calculate returns
    returns1 = pd.Series(close1).pct_change().values * 100
    returns2 = pd.Series(close2).pct_change().values * 100
    
    # Calculate RSI and Volatility
    rsi1 = calculate_rsi(close1)
    rsi2 = calculate_rsi(close2)
    volatility1 = calculate_volatility(close1)
    volatility2 = calculate_volatility(close2)
    
    # Align dataframes
    common_index = df1.index.intersection(df2.index)
    
    # Find indices for alignment
    idx1 = [i for i, idx in enumerate(df1.index) if idx in common_index]
    idx2 = [i for i, idx in enumerate(df2.index) if idx in common_index]
    
    # Create aligned arrays
    close1_aligned = close1[idx1]
    close2_aligned = close2[idx2]
    open1_aligned = open1[idx1]
    open2_aligned = open2[idx2]
    high1_aligned = high1[idx1]
    high2_aligned = high2[idx2]
    low1_aligned = low1[idx1]
    low2_aligned = low2[idx2]
    volume1_aligned = volume1[idx1]
    rsi1_aligned = rsi1[idx1]
    rsi2_aligned = rsi2[idx2]
    volatility1_aligned = volatility1[idx1]
    volatility2_aligned = volatility2[idx2]
    returns1_aligned = returns1[idx1]
    returns2_aligned = returns2[idx2]
    
    # Calculate ratio
    ratio = close1_aligned / (close2_aligned + 1e-10)  # Avoid division by zero
    ratio_rsi = calculate_rsi(ratio)
    ratio_volatility = calculate_volatility(ratio)
    ratio_returns = pd.Series(ratio).pct_change().values * 100
    
    # Create combined analysis dataframe
    analysis_df = pd.DataFrame({
        'DateTime': common_index,
        'Ticker1_Price': close1_aligned,
        'Ticker2_Price': close2_aligned,
        'Ratio': ratio,
        'RSI1': rsi1_aligned,
        'RSI2': rsi2_aligned,
        'Volatility1': volatility1_aligned,
        'Volatility2': volatility2_aligned,
        'Returns1': returns1_aligned,
        'Returns2': returns2_aligned,
        'Ratio_Returns': ratio_returns,
    })
    
    analysis_df['Ratio_RSI'] = ratio_rsi
    analysis_df['Ratio_Volatility'] = ratio_volatility
    analysis_df['Prev_Ticker1'] = analysis_df['Ticker1_Price'].shift(1)
    analysis_df['Prev_Ticker2'] = analysis_df['Ticker2_Price'].shift(1)
    analysis_df['Change_Ticker1'] = analysis_df['Ticker1_Price'] - analysis_df['Prev_Ticker1']
    analysis_df['Change_Ticker2'] = analysis_df['Ticker2_Price'] - analysis_df['Prev_Ticker2']
    analysis_df['Change_Pct1'] = (analysis_df['Change_Ticker1'] / (analysis_df['Prev_Ticker1'] + 1e-10)) * 100
    analysis_df['Change_Pct2'] = (analysis_df['Change_Ticker2'] / (analysis_df['Prev_Ticker2'] + 1e-10)) * 100
    
    # Drop NaN rows
    analysis_df = analysis_df.dropna(subset=['Returns1', 'Returns2'])
    
    # Display current status
    st.header("📈 Current Market Status")
    col1, col2, col3, col4 = st.columns(4)
    
    if len(analysis_df) > 1:
        latest = analysis_df.iloc[-1]
        prev = analysis_df.iloc[-2]
        
        with col1:
            change1 = latest['Ticker1_Price'] - prev['Ticker1_Price']
            pct1 = (change1 / prev['Ticker1_Price']) * 100
            st.metric(f"{ticker1_type}", 
                     f"₹{latest['Ticker1_Price']:.2f}", 
                     f"{change1:+.2f} ({pct1:+.2f}%)")
        
        with col2:
            change2 = latest['Ticker2_Price'] - prev['Ticker2_Price']
            pct2 = (change2 / prev['Ticker2_Price']) * 100
            st.metric(f"{ticker2_type}", 
                     f"₹{latest['Ticker2_Price']:.2f}", 
                     f"{change2:+.2f} ({pct2:+.2f}%)")
        
        with col3:
            st.metric("Ratio", f"{latest['Ratio']:.4f}")
        
        with col4:
            st.metric("RSI (Ratio)", f"{latest['Ratio_RSI']:.2f}")
    
    # Detailed Analysis Table
    st.header("📊 Detailed Analysis Table")
    
    display_df = analysis_df.copy()
    display_df['DateTime'] = display_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
    
    # Color styling function
    def color_negative_red(val):
        if isinstance(val, (int, float)) and not np.isnan(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}'
        return ''
    
    styled_df = display_df.tail(20).style.applymap(
        color_negative_red, 
        subset=['Change_Ticker1', 'Change_Ticker2', 'Returns1', 'Returns2', 'Change_Pct1', 'Change_Pct2']
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Bell Curve Distribution Analysis
    st.header("🔔 Bell Curve Distribution Analysis")
    
    st.markdown("""
    **Understanding Normal Distribution:**
    - Bell curves show the probability distribution of returns
    - Mean (μ): Average return - center of the bell
    - Std Dev (σ): Volatility - width of the bell
    - Current position tells us if we're in normal or extreme territory
    - ~68% of returns fall within ±1σ, ~95% within ±2σ
    """)
    
    # Fit distributions
    dist_ticker1 = fit_distribution(analysis_df['Returns1'].values, f"{ticker1_type} Returns")
    dist_ticker2 = fit_distribution(analysis_df['Returns2'].values, f"{ticker2_type} Returns")
    dist_ratio = fit_distribution(analysis_df['Ratio_Returns'].values, "Ratio Returns")
    
    # Create distribution plots
    fig_dist = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f'{ticker1_type} Returns Distribution', 
                       f'{ticker2_type} Returns Distribution', 
                       'Ratio Returns Distribution')
    )
    
    for idx, dist in enumerate([dist_ticker1, dist_ticker2, dist_ratio], 1):
        if dist:
            # Create histogram
            hist_data = dist['data']
            
            # Fit normal distribution curve
            x_range = np.linspace(hist_data.min(), hist_data.max(), 100)
            pdf = stats.norm.pdf(x_range, dist['mean'], dist['std'])
            
            # Add histogram
            fig_dist.add_trace(
                go.Histogram(x=hist_data, name='Actual', nbinsx=30, 
                           histnorm='probability density', opacity=0.7),
                row=1, col=idx
            )
            
            # Add fitted curve
            fig_dist.add_trace(
                go.Scatter(x=x_range, y=pdf, name='Normal Fit', 
                          line=dict(color='red', width=3)),
                row=1, col=idx
            )
            
            # Add current value line
            fig_dist.add_vline(
                x=dist['current_value'], 
                line_dash="dash", 
                line_color="green",
                row=1, col=idx
            )
    
    fig_dist.update_layout(height=400, showlegend=True, title_text="Returns Distribution Analysis")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Distribution statistics table
    st.subheader("📊 Distribution Statistics & Forecasting")
    
    dist_stats = []
    for dist in [dist_ticker1, dist_ticker2, dist_ratio]:
        if dist:
            dist_stats.append({
                'Asset': dist['name'],
                'Mean (μ)': f"{dist['mean']:.4f}%",
                'Std Dev (σ)': f"{dist['std']:.4f}%",
                'Current Value': f"{dist['current_value']:.4f}%",
                'Percentile': f"{dist['current_percentile']:.1f}%",
                'Prob(+ve Move)': f"{dist['prob_positive']:.1f}%",
                'Prob(Extreme+)': f"{dist['prob_extreme_positive']:.1f}%",
                'Prob(Extreme-)': f"{dist['prob_extreme_negative']:.1f}%"
            })
    
    dist_df = pd.DataFrame(dist_stats)
    st.dataframe(dist_df, use_container_width=True)
    
    # Distribution insights
    st.markdown("### 🎯 Distribution-Based Forecast")
    
    for dist in [dist_ticker1, dist_ticker2, dist_ratio]:
        if dist:
            with st.expander(f"📈 {dist['name']} Analysis"):
                current_z = (dist['current_value'] - dist['mean']) / dist['std']
                
                st.markdown(f"""
                **Current Position:**
                - Current Return: **{dist['current_value']:.3f}%**
                - Mean Return: **{dist['mean']:.3f}%**
                - Volatility (σ): **{dist['std']:.3f}%**
                - Z-Score: **{current_z:.2f}σ** (Current value is {abs(current_z):.2f} standard deviations {'above' if current_z > 0 else 'below'} mean)
                - Percentile: **{dist['current_percentile']:.1f}%** (Better than {dist['current_percentile']:.1f}% of historical returns)
                
                **Probabilistic Forecast:**
                - Probability of positive move: **{dist['prob_positive']:.1f}%**
                - Probability of extreme positive (+2σ): **{dist['prob_extreme_positive']:.1f}%**
                - Probability of extreme negative (-2σ): **{dist['prob_extreme_negative']:.1f}%**
                
                **Expected Range (68% confidence):**
                - Next move likely between: **{dist['mean'] - dist['std']:.3f}%** to **{dist['mean'] + dist['std']:.3f}%**
                
                **Expected Range (95% confidence):**
                - Next move likely between: **{dist['mean'] - 2*dist['std']:.3f}%** to **{dist['mean'] + 2*dist['std']:.3f}%**
                
                **Interpretation:**
                {f"⚠️ **EXTREME OVERBOUGHT** - Current at {current_z:.1f}σ! Strong reversal expected. {100-dist['current_percentile']:.1f}% probability of mean reversion." if current_z > 2 else
                 f"⚠️ **EXTREME OVERSOLD** - Current at {current_z:.1f}σ! Strong bounce expected. {dist['current_percentile']:.1f}% probability of mean reversion." if current_z < -2 else
                 f"🟢 **NORMAL RANGE** - Price action within expected volatility. Follow trend." if abs(current_z) < 1 else
                 f"🟡 **MODERATE DEVIATION** - Slight deviation from mean. Watch for continuation or reversal."}
                
                **Trading Signal:**
                {f"🔴 **SELL/SHORT** - Extremely overbought. Expect {dist['std']:.2f}% correction toward mean." if current_z > 2 else
                 f"🟢 **BUY** - Extremely oversold. Expect {dist['std']:.2f}% bounce toward mean." if current_z < -2 else
                 f"⚪ **HOLD** - Within normal distribution. No extreme signal." if abs(current_z) < 1 else
                 f"🟡 **CAUTION** - Monitor for breakout or reversal."}
                """)
    
    # Pattern Recognition - Ratio Bins Analysis
    st.header("🔍 Pattern Recognition: Ratio Bins Analysis")
    
    try:
        analysis_df['Ratio_Bin'] = pd.qcut(analysis_df['Ratio'], q=10, labels=[f'Bin{i}' for i in range(1, 11)], duplicates='drop')
        
        ratio_bin_analysis = analysis_df.groupby('Ratio_Bin').agg({
            'Change_Pct1': ['mean', 'std', 'min', 'max', 'count'],
            'Change_Pct2': ['mean', 'std', 'min', 'max'],
            'Returns1': ['mean', 'sum'],
            'Returns2': ['mean', 'sum']
        }).round(2)
        
        st.subheader("📌 Ratio Bin Statistics")
        st.dataframe(ratio_bin_analysis, use_container_width=True)
        
        # Current bin analysis
        current_bin = analysis_df.iloc[-1]['Ratio_Bin']
        current_bin_data = analysis_df[analysis_df['Ratio_Bin'] == current_bin]
        
        st.markdown(f"### 🎯 Current Market Analysis (Ratio Bin: {current_bin})")
        
        avg_return1 = current_bin_data['Change_Pct1'].mean()
        avg_return2 = current_bin_data['Change_Pct2'].mean()
        max_rally1 = current_bin_data['Change_Pct1'].max()
        min_rally1 = current_bin_data['Change_Pct1'].min()
        
        st.markdown(f"""
        **Key Insights for {current_bin}:**
        - **Average Return (Ticker1):** {avg_return1:.2f}%
        - **Average Return (Ticker2):** {avg_return2:.2f}%
        - **Maximum Rally:** {max_rally1:.2f}%
        - **Maximum Decline:** {min_rally1:.2f}%
        - **Historical Occurrences:** {len(current_bin_data)} times
        
        **Interpretation:** 
        Historically, when ratio is in {current_bin}, Ticker1 has moved {avg_return1:.2f}% on average. 
        {'This suggests BULLISH potential.' if avg_return1 > 0 else 'This suggests BEARISH potential.'}
        """)
    except Exception as e:
        st.warning(f"Ratio bin analysis skipped: {str(e)}")
    
    # Volatility Bins Analysis
    st.header("📊 Volatility-Based Pattern Recognition")
    
    try:
        analysis_df['Vol1_Bin'] = pd.qcut(analysis_df['Volatility1'].dropna(), q=5, labels=['VL', 'L', 'M', 'H', 'VH'], duplicates='drop')
        
        vol_bin_analysis = analysis_df.groupby('Vol1_Bin').agg({
            'Change_Pct1': ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
        
        st.dataframe(vol_bin_analysis, use_container_width=True)
        
        current_vol_bin = analysis_df.iloc[-1]['Vol1_Bin']
        current_vol_data = analysis_df[analysis_df['Vol1_Bin'] == current_vol_bin]
        
        avg_vol_return = current_vol_data['Change_Pct1'].mean()
        
        st.markdown(f"""
        ### 🌡️ Current Volatility Regime: {current_vol_bin}
        
        **Key Insights:**
        - **Average Move in this Volatility:** {avg_vol_return:.2f}%
        - **Typical Range:** {current_vol_data['Change_Pct1'].min():.2f}% to {current_vol_data['Change_Pct1'].max():.2f}%
        
        **Market Condition:** 
        {f'High volatility regime - Expect larger price swings ({abs(avg_vol_return):.2f}% moves typical)' if current_vol_bin in ['H', 'VH'] else 'Low volatility regime - Expect smaller, rangebound moves'}
        """)
    except Exception as e:
        st.warning(f"Volatility bin analysis skipped: {str(e)}")
    
    # Mean Reversion Analysis
    st.header("🔄 Mean Reversion Pattern Detection")
    
    try:
        analysis_df['Price_Zscore'] = stats.zscore(analysis_df['Ticker1_Price'].dropna())
        analysis_df['Next_Return'] = analysis_df['Returns1'].shift(-1)
        
        extreme_moves = analysis_df[abs(analysis_df['Price_Zscore']) > 2].copy()
        
        if len(extreme_moves) > 0:
            reversion_rate = (extreme_moves['Next_Return'] * -np.sign(extreme_moves['Price_Zscore']) > 0).mean() * 100
            
            current_zscore = analysis_df.iloc[-1]['Price_Zscore']
            
            st.markdown(f"""
            ### 📉 Mean Reversion Statistics
            
            - **Current Z-Score:** {current_zscore:.2f}
            - **Reversion Success Rate:** {reversion_rate:.1f}% (after extreme moves)
            - **Current Status:** {'⚠️ EXTREME - High reversion probability' if abs(current_zscore) > 2 else '✅ Normal range'}
            
            **Interpretation:**
            {f'Price is {abs(current_zscore):.1f} standard deviations from mean. Strong reversion signal!' if abs(current_zscore) > 2 else 'Price is within normal range. No strong reversion signal.'}
            """)
    except Exception as e:
        st.warning(f"Mean reversion analysis skipped: {str(e)}")
    
    # Similarity Pattern Search
    st.header("🧬 Similarity Pattern Matching (Greedy Algorithm)")
    
    try:
        # Use last N periods to find similar patterns
        lookback = min(20, len(analysis_df) // 4)
        current_pattern = analysis_df['Returns1'].tail(lookback).values
        
        similarities = []
        for i in range(lookback, len(analysis_df) - lookback):
            past_pattern = analysis_df['Returns1'].iloc[i-lookback:i].values
            if len(past_pattern) == len(current_pattern):
                correlation = np.corrcoef(current_pattern, past_pattern)[0, 1]
                if not np.isnan(correlation):
                    future_return = analysis_df['Returns1'].iloc[i:i+5].sum()
                    similarities.append({
                        'Date': analysis_df.iloc[i]['DateTime'],
                        'Correlation': correlation,
                        'Future_5Period_Return': future_return
                    })
        
        if similarities:
            similarity_df = pd.DataFrame(similarities).sort_values('Correlation', ascending=False).head(10)
            
            st.markdown("### 🎯 Top 10 Most Similar Historical Patterns")
            st.dataframe(similarity_df, use_container_width=True)
            
            avg_future_return = similarity_df['Future_5Period_Return'].mean()
            
            st.markdown(f"""
            **Pattern-Based Forecast:**
            - **Average Return (Next 5 periods):** {avg_future_return:.2f}%
            - **Direction:** {'🟢 BULLISH' if avg_future_return > 0 else '🔴 BEARISH'}
            - **Confidence:** {similarity_df['Correlation'].mean():.2%}
            """)
    except Exception as e:
        st.warning(f"Pattern matching skipped: {str(e)}")
    
    # Interactive Charts
    st.header("📊 Interactive Price & Indicator Charts")
    
    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=('Ticker 1 Price', 'Ticker 2 Price', 'Ratio', 'RSI Indicators', 'Volume'),
        row_heights=[0.25, 0.25, 0.2, 0.2, 0.1],
        vertical_spacing=0.05
    )
    
    # Ticker 1 candlestick
    fig.add_trace(go.Candlestick(
        x=common_index,
        open=open1_aligned,
        high=high1_aligned,
        low=low1_aligned,
        close=close1_aligned,
        name=ticker1
    ), row=1, col=1)
    
    # Ticker 2 candlestick
    fig.add_trace(go.Candlestick(
        x=common_index,
        open=open2_aligned,
        high=high2_aligned,
        low=low2_aligned,
        close=close2_aligned,
        name=ticker2
    ), row=2, col=1)
    
    # Ratio
    fig.add_trace(go.Scatter(
        x=analysis_df['DateTime'],
        y=analysis_df['Ratio'],
        name='Ratio',
        line=dict(color='purple', width=2)
    ), row=3, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=analysis_df['DateTime'], y=analysis_df['RSI1'], name='RSI1', line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=analysis_df['DateTime'], y=analysis_df['RSI2'], name='RSI2', line=dict(color='orange')), row=4, col=1)
    fig.add_trace(go.Scatter(x=analysis_df['DateTime'], y=analysis_df['Ratio_RSI'], name='Ratio RSI', line=dict(color='purple')), row=4, col=1)
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    
    # Volume
    if not np.all(volume1_aligned == 0):
        fig.add_trace(go.Bar(x=common_index, y=volume1_aligned, name='Volume', marker_color='lightblue'), row=5, col=1)
    
    fig.update_layout(height=1400, showlegend=True, title_text="Complete Market Analysis")
    fig.update_xaxes(rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Chart Analysis:**
    - **Price Action:** Candlestick patterns show market structure and trends
    - **Ratio:** Relative strength between two assets - rising ratio = Ticker1 outperforming
    - **RSI:** >70 = Overbought (potential reversal), <30 = Oversold (potential bounce)
    - **Volume:** Confirms price moves - high volume = strong conviction
    """)
    
    # Heatmaps
    st.header("🔥 Returns & Volatility Heatmaps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            df1_returns = pd.Series(returns1, index=df1.index).dropna()
            if len(df1_returns) > 0 and hasattr(df1_returns.index, 'hour'):
                df1_returns_pivot = df1_returns.groupby([df1_returns.index.hour, df1_returns.index.dayofweek]).mean().unstack()
                
                if not df1_returns_pivot.empty:
                    fig_heat1 = px.imshow(
                        df1_returns_pivot,
                        labels=dict(x="Day of Week", y="Hour", color="Return %"),
                        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        color_continuous_scale='RdYlGn',
                        title=f'{ticker1_type} Returns Heatmap',
                        text_auto='.2f'
                    )
                    fig_heat1.update_traces(textfont_color='white')
                    st.plotly_chart(fig_heat1, use_container_width=True)
        except Exception as e:
            st.info(f"Heatmap for {ticker1_type}: Not enough data")
    
    with col2:
        try:
            df2_returns = pd.Series(returns2, index=df2.index).dropna()
            if len(df2_returns) > 0 and hasattr(df2_returns.index, 'hour'):
                df2_returns_pivot = df2_returns.groupby([df2_returns.index.hour, df2_returns.index.dayofweek]).mean().unstack()
                
                if not df2_returns_pivot.empty:
                    fig_heat2 = px.imshow(
                        df2_returns_pivot,
                        labels=dict(x="Day of Week", y="Hour", color="Return %"),
                        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        color_continuous_scale='RdYlGn',
                        title=f'{ticker2_type} Returns Heatmap',
                        text_auto='.2f'
                    )
                    fig_heat2.update_traces(textfont_color='white')
                    st.plotly_chart(fig_heat2, use_container_width=True)
        except Exception as e:
            st.info(f"Heatmap for {ticker2_type}: Not enough data")
    
    # Final Recommendation
    st.header("🎯 AI-Powered Final Recommendation")
    
    # Scoring system
    score = 0
    reasons = []
    
    latest = analysis_df.iloc[-1]
    
    # RSI analysis
    if latest['RSI1'] < 30:
        score += 2
        reasons.append("✅ RSI oversold (<30) - bounce potential")
    elif latest['RSI1'] > 70:
        score -= 2
        reasons.append("⚠️ RSI overbought (>70) - correction risk")
    
    # Distribution-based analysis
    if dist_ticker1:
        current_z = (dist_ticker1['current_value'] - dist_ticker1['mean']) / dist_ticker1['std']
        if current_z < -2:
            score += 2
            reasons.append(f"✅ Extreme oversold (Z={current_z:.2f}) - mean reversion expected")
        elif current_z > 2:
            score -= 2
            reasons.append(f"⚠️ Extreme overbought (Z={current_z:.2f}) - correction expected")
        
        # Probability-based scoring
        if dist_ticker1['prob_positive'] > 60:
            score += 1
            reasons.append(f"✅ High probability of positive move ({dist_ticker1['prob_positive']:.1f}%)")
        elif dist_ticker1['prob_positive'] < 40:
            score -= 1
            reasons.append(f"⚠️ Low probability of positive move ({dist_ticker1['prob_positive']:.1f}%)")
    
    # Mean reversion
    try:
        current_zscore = analysis_df.iloc[-1]['Price_Zscore']
        if abs(current_zscore) > 2:
            score += 1 if current_zscore < 0 else -1
            reasons.append(f"✅ Mean reversion signal (Z={current_zscore:.2f})")
    except:
        pass
    
    # Pattern similarity
    try:
        if similarities and len(similarity_df) > 0:
            avg_future_return = similarity_df['Future_5Period_Return'].mean()
            if avg_future_return > 0.5:
                score += 2
                reasons.append(f"✅ Similar patterns suggest +{avg_future_return:.2f}% move")
            elif avg_future_return < -0.5:
                score -= 2
                reasons.append(f"⚠️ Similar patterns suggest {avg_future_return:.2f}% decline")
    except:
        pass
    
    # Ratio analysis
    try:
        if 'avg_return1' in locals() and avg_return1 > 1:
            score += 1
            reasons.append(f"✅ Current ratio bin historically bullish (+{avg_return1:.2f}%)")
        elif 'avg_return1' in locals() and avg_return1 < -1:
            score -= 1
            reasons.append(f"⚠️ Current ratio bin historically bearish ({avg_return1:.2f}%)")
    except:
        pass
    
    # Volatility regime
    try:
        if 'current_vol_bin' in locals() and current_vol_bin in ['VH', 'H']:
            reasons.append("🌡️ High volatility - wider stops recommended")
    except:
        pass
    
    # Final signal
    if score >= 3:
        signal = "🟢 STRONG BUY"
        color = "green"
        recommendation = "Strong bullish setup with multiple positive factors. Consider BUYING on dips with tight stops."
    elif score >= 1:
        signal = "🟡 BUY"
        color = "lightgreen"
        recommendation = "Bullish bias but wait for confirmation. Consider small positions with defined risk."
    elif score <= -3:
        signal = "🔴 STRONG SELL"
        color = "red"
        recommendation = "Strong bearish setup with multiple negative factors. Consider SELLING or shorting with proper risk management."
    elif score <= -1:
        signal = "🟠 SELL"
        color = "orange"
        recommendation = "Bearish bias. Reduce exposure or wait for better entry points."
    else:
        signal = "⚪ HOLD"
        color = "gray"
        recommendation = "Mixed signals. HOLD current positions and wait for clearer direction."
    
    st.markdown(f"### <span style='color:{color}; font-size:32px;'>{signal}</span>", unsafe_allow_html=True)
    
    st.markdown("**Analysis Summary:**")
    for reason in reasons:
        st.markdown(f"- {reason}")
    
    # Calculate targets
    if dist_ticker1:
        expected_return = dist_ticker1['mean']
        stop_loss_pct = 2 * dist_ticker1['std']
        target_pct = abs(expected_return) + dist_ticker1['std']
    else:
        expected_return = 0
        stop_loss_pct = 2
        target_pct = 3
    
    current_price = latest['Ticker1_Price']
    stop_loss = current_price * (1 - stop_loss_pct/100) if score > 0 else current_price * (1 + stop_loss_pct/100)
    target = current_price * (1 + target_pct/100) if score > 0 else current_price * (1 - target_pct/100)
    
    st.markdown(f"""
    **Confidence Score:** {abs(score)}/10
    
    **Professional Recommendation:**
    {recommendation}
    
    **Risk Management:**
    - **Current Price:** ₹{current_price:.2f}
    - **Stop Loss:** ₹{stop_loss:.2f} ({stop_loss_pct:.2f}% from entry)
    - **Take Profit Target:** ₹{target:.2f} ({target_pct:.2f}% from entry)
    - **Risk/Reward Ratio:** 1:{target_pct/stop_loss_pct:.2f}
    
    **Distribution Insights:**
    {f"Expected move based on normal distribution: {expected_return:.2f}% ± {dist_ticker1['std']:.2f}%" if dist_ticker1 else "N/A"}
    
    **Position Sizing:**
    - Risk per trade: 1-2% of capital
    - {f"If risking ₹10,000, position size = ₹{10000/(stop_loss_pct/100):.0f}" if stop_loss_pct > 0 else "Calculate based on stop loss"}
    """)
    
    # Additional Bell Curve Insights
    st.markdown("### 🔔 Key Distribution Insights")
    
    if dist_ticker1:
        st.markdown(f"""
        **What the Bell Curve Tells Us:**
        1. **Mean ({dist_ticker1['mean']:.3f}%)**: This is the "center of gravity" - price tends to gravitate here
        2. **Current Position ({dist_ticker1['current_percentile']:.1f} percentile)**: You're in the {'top' if dist_ticker1['current_percentile'] > 50 else 'bottom'} half of the distribution
        3. **Probability Edge**: {dist_ticker1['prob_positive']:.1f}% chance of positive move based on historical distribution
        4. **Extreme Events**: {dist_ticker1['prob_extreme_positive']:.1f}% chance of +2σ move, {dist_ticker1['prob_extreme_negative']:.1f}% chance of -2σ move
        
        **Trading Strategy Based on Distribution:**
        {f"- **Mean Reversion Play**: Price is {abs(current_z):.1f}σ from mean - expect {dist_ticker1['std']:.2f}% move toward {dist_ticker1['mean']:.3f}%" if abs(current_z) > 1.5 else
         f"- **Trend Following**: Price near mean - wait for breakout above {dist_ticker1['mean'] + dist_ticker1['std']:.3f}% or below {dist_ticker1['mean'] - dist_ticker1['std']:.3f}%"}
        """)
    
    # Summary Table
    st.markdown("### 📊 Complete Analysis Summary")
    
    summary_data = {
        'Metric': ['Current Price', 'RSI', 'Z-Score', 'Volatility Regime', 'Signal', 'Confidence'],
        'Ticker 1': [
            f"₹{latest['Ticker1_Price']:.2f}",
            f"{latest['RSI1']:.2f}",
            f"{current_zscore:.2f}" if 'current_zscore' in locals() else "N/A",
            current_vol_bin if 'current_vol_bin' in locals() else "N/A",
            signal,
            f"{abs(score)}/10"
        ]
    }
    
    if dist_ticker1:
        summary_data['Distribution Mean'] = [
            f"{dist_ticker1['mean']:.3f}%",
            "-",
            "-",
            "-",
            "-",
            f"{dist_ticker1['prob_positive']:.1f}%"
        ]
    
    summary_table = pd.DataFrame(summary_data)
    st.dataframe(summary_table, use_container_width=True)
    
    # Download CSV
    st.header("📥 Download Data")
    
    csv_df = pd.DataFrame({
        'DateTime': [dt.strftime('%Y-%m-%d %H:%M:%S IST') for dt in common_index],
        'Open': open1_aligned,
        'High': high1_aligned,
        'Low': low1_aligned,
        'Close': close1_aligned,
        'Volume': volume1_aligned
    })
    
    csv = csv_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Ticker 1 OHLCV CSV",
        data=csv,
        file_name=f"{ticker1}_{timeframe}_{period}.csv",
        mime="text/csv",
    )
    
    # Download analysis CSV
    analysis_csv = analysis_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Complete Analysis CSV",
        data=analysis_csv,
        file_name=f"analysis_{ticker1}_{ticker2}_{timeframe}_{period}.csv",
        mime="text/csv",
    )
    
else:
    st.info("👆 Configure your analysis parameters in the sidebar and click 'Fetch Data & Analyze' to begin!")
    
    st.markdown("""
    ### 🎯 Features:
    
    1. **Multi-Asset Analysis** - Analyze any two assets and their ratio
    2. **Bell Curve Distribution** - Normal distribution analysis for probabilistic forecasting
    3. **Pattern Recognition** - Greedy algorithm finds similar historical patterns
    4. **Mean Reversion Detection** - Identifies extreme price deviations
    5. **Ratio Bin Analysis** - Discovers which price ratios correlate with moves
    6. **Volatility Regime Analysis** - Different strategies for different volatility levels
    7. **RSI Multi-Timeframe** - Overbought/oversold conditions
    8. **Similarity Scoring** - Finds historical patterns matching current market
    9. **AI Recommendation** - Combines all signals for BUY/SELL/HOLD decision
    10. **Interactive Charts** - Candlesticks, RSI, Volume, Ratio analysis
    11. **Heatmaps** - Visualize best times/days for returns
    
    ### 🔔 Bell Curve Analysis (NEW!):
    
    The app now includes **Normal Distribution Analysis** which:
    - Fits historical returns to a bell curve (normal distribution)
    - Calculates mean (μ) and standard deviation (σ)
    - Shows current position relative to historical distribution
    - Provides probabilistic forecasts:
      - Probability of positive move
      - Probability of extreme moves (±2σ)
      - Expected range with 68% and 95% confidence
    - Identifies overbought/oversold based on z-scores
    - Gives mean reversion signals when price is >2σ from mean
    
    **How it works:**
    - ~68% of returns fall within ±1σ of mean
    - ~95% of returns fall within ±2σ of mean
    - If current return is >2σ: EXTREME and likely to revert
    - The bell curve predicts future moves based on probability
    
    ### 📊 How Pattern Recognition Works:
    
    The algorithm uses multiple techniques to find patterns:
    - **Change-based correlation** instead of absolute prices
    - **Binning analysis** for ratio and volatility regimes  
    - **Mean reversion** detection using z-scores
    - **Pattern similarity** matching using returns correlation
    - **Distribution analysis** for probabilistic edge
    - **Multi-factor scoring** combining all signals
    
    All predictions are based on historical pattern recognition with confidence scores!
    """)
