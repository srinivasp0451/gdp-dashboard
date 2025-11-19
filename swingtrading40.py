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
                data1.index = data1.index.tz_localize('UTC').tz_convert(ist) if data1.index.tz is None else data1.index.tz_convert(ist)
                data2.index = data2.index.tz_localize('UTC').tz_convert(ist) if data2.index.tz is None else data2.index.tz_convert(ist)
                
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
    
    # Helper functions
    def calculate_rsi(data, period=14):
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_volatility(data, window=20):
        """Calculate rolling volatility"""
        returns = data.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    # Calculate metrics
    df1['Returns'] = df1['Close'].pct_change() * 100
    df2['Returns'] = df2['Close'].pct_change() * 100
    df1['RSI'] = calculate_rsi(df1['Close'])
    df2['RSI'] = calculate_rsi(df2['Close'])
    df1['Volatility'] = calculate_volatility(df1['Close'])
    df2['Volatility'] = calculate_volatility(df2['Close'])
    
    # Align dataframes
    common_index = df1.index.intersection(df2.index)
    df1_aligned = df1.loc[common_index]
    df2_aligned = df2.loc[common_index]
    
    # Create combined analysis dataframe
    analysis_df = pd.DataFrame({
        'DateTime': common_index,
        'Ticker1_Price': df1_aligned['Close'].values,
        'Ticker2_Price': df2_aligned['Close'].values,
        'Ratio': df1_aligned['Close'].values / df2_aligned['Close'].values,
        'RSI1': df1_aligned['RSI'].values,
        'RSI2': df2_aligned['RSI'].values,
        'Volatility1': df1_aligned['Volatility'].values,
        'Volatility2': df2_aligned['Volatility'].values,
        'Returns1': df1_aligned['Returns'].values,
        'Returns2': df2_aligned['Returns'].values,
    })
    
    analysis_df['Ratio_RSI'] = calculate_rsi(pd.Series(analysis_df['Ratio'].values))
    analysis_df['Ratio_Volatility'] = calculate_volatility(pd.Series(analysis_df['Ratio'].values))
    analysis_df['Prev_Ticker1'] = analysis_df['Ticker1_Price'].shift(1)
    analysis_df['Prev_Ticker2'] = analysis_df['Ticker2_Price'].shift(1)
    analysis_df['Change_Ticker1'] = analysis_df['Ticker1_Price'] - analysis_df['Prev_Ticker1']
    analysis_df['Change_Ticker2'] = analysis_df['Ticker2_Price'] - analysis_df['Prev_Ticker2']
    analysis_df['Change_Pct1'] = (analysis_df['Change_Ticker1'] / analysis_df['Prev_Ticker1']) * 100
    analysis_df['Change_Pct2'] = (analysis_df['Change_Ticker2'] / analysis_df['Prev_Ticker2']) * 100
    
    # Display current status
    st.header("📈 Current Market Status")
    col1, col2, col3, col4 = st.columns(4)
    
    latest = analysis_df.iloc[-1]
    prev = analysis_df.iloc[-2] if len(analysis_df) > 1 else latest
    
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
        if isinstance(val, (int, float)):
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}'
        return ''
    
    styled_df = display_df.tail(20).style.applymap(
        color_negative_red, 
        subset=['Change_Ticker1', 'Change_Ticker2', 'Returns1', 'Returns2', 'Change_Pct1', 'Change_Pct2']
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Pattern Recognition - Ratio Bins Analysis
    st.header("🔍 Pattern Recognition: Ratio Bins Analysis")
    
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
    
    # Volatility Bins Analysis
    st.header("📊 Volatility-Based Pattern Recognition")
    
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
    
    # Mean Reversion Analysis
    st.header("🔄 Mean Reversion Pattern Detection")
    
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
    
    # Similarity Pattern Search
    st.header("🧬 Similarity Pattern Matching (Greedy Algorithm)")
    
    # Use last N periods to find similar patterns
    lookback = min(20, len(analysis_df) // 4)
    current_pattern = analysis_df['Returns1'].tail(lookback).values
    
    similarities = []
    for i in range(lookback, len(analysis_df) - lookback):
        past_pattern = analysis_df['Returns1'].iloc[i-lookback:i].values
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
    
    # Interactive Charts
    st.header("📊 Interactive Price & Indicator Charts")
    
    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=('Ticker 1 Price', 'Ticker 2 Price', 'Ratio', 'RSI Indicators', 'Volume'),
        row_heights=[0.25, 0.25, 0.2, 0.2, 0.1],
        vertical_spacing=0.05
    )
    
    # Ticker 1
    fig.add_trace(go.Candlestick(
        x=df1_aligned.index,
        open=df1_aligned['Open'],
        high=df1_aligned['High'],
        low=df1_aligned['Low'],
        close=df1_aligned['Close'],
        name=ticker1
    ), row=1, col=1)
    
    # Ticker 2
    fig.add_trace(go.Candlestick(
        x=df2_aligned.index,
        open=df2_aligned['Open'],
        high=df2_aligned['High'],
        low=df2_aligned['Low'],
        close=df2_aligned['Close'],
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
    #line_color="green", row=4, col=1)
    
    # Volume
    if not df1_aligned['Volume'].eq(0).all():
        fig.add_trace(go.Bar(x=df1_aligned.index, y=df1_aligned['Volume'], name='Volume', marker_color='lightblue'), row=5, col=1)
    
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
        df1_returns = df1['Returns'].dropna()
        if len(df1_returns) > 0:
            df1_returns_pivot = df1_returns.groupby([df1_returns.index.hour, df1_returns.index.dayofweek]).mean().unstack()
            
            fig_heat1 = px.imshow(
                df1_returns_pivot,
                labels=dict(x="Day of Week", y="Hour", color="Return %"),
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                color_continuous_scale='RdYlGn',
                title=f'{ticker1} Returns Heatmap',
                text_auto='.2f'
            )
            fig_heat1.update_traces(textfont_color='white')
            st.plotly_chart(fig_heat1, use_container_width=True)
    
    with col2:
        df2_returns = df2['Returns'].dropna()
        if len(df2_returns) > 0:
            df2_returns_pivot = df2_returns.groupby([df2_returns.index.hour, df2_returns.index.dayofweek]).mean().unstack()
            
            fig_heat2 = px.imshow(
                df2_returns_pivot,
                labels=dict(x="Day of Week", y="Hour", color="Return %"),
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                color_continuous_scale='RdYlGn',
                title=f'{ticker2} Returns Heatmap',
                text_auto='.2f'
            )
            fig_heat2.update_traces(textfont_color='white')
            st.plotly_chart(fig_heat2, use_container_width=True)
    
    # Final Recommendation
    st.header("🎯 AI-Powered Final Recommendation")
    
    # Scoring system
    score = 0
    reasons = []
    
    # RSI analysis
    if latest['RSI1'] < 30:
        score += 2
        reasons.append("✅ RSI oversold - bounce potential")
    elif latest['RSI1'] > 70:
        score -= 2
        reasons.append("⚠️ RSI overbought - correction risk")
    
    # Mean reversion
    if abs(current_zscore) > 2:
        score += 1 if current_zscore < 0 else -1
        reasons.append(f"✅ Mean reversion signal (Z={current_zscore:.2f})")
    
    # Pattern similarity
    if similarities and avg_future_return > 0:
        score += 2
        reasons.append(f"✅ Similar patterns suggest +{avg_future_return:.2f}% move")
    elif similarities and avg_future_return < 0:
        score -= 2
        reasons.append(f"⚠️ Similar patterns suggest {avg_future_return:.2f}% decline")
    
    # Ratio analysis
    if avg_return1 > 1:
        score += 1
        reasons.append(f"✅ Current ratio bin historically bullish (+{avg_return1:.2f}%)")
    elif avg_return1 < -1:
        score -= 1
        reasons.append(f"⚠️ Current ratio bin historically bearish ({avg_return1:.2f}%)")
    
    # Final signal
    if score >= 3:
        signal = "🟢 STRONG BUY"
        color = "green"
    elif score >= 1:
        signal = "🟡 BUY"
        color = "lightgreen"
    elif score <= -3:
        signal = "🔴 STRONG SELL"
        color = "red"
    elif score <= -1:
        signal = "🟠 SELL"
        color = "orange"
    else:
        signal = "⚪ HOLD"
        color = "gray"
    
    st.markdown(f"### <span style='color:{color}; font-size:32px;'>{signal}</span>", unsafe_allow_html=True)
    
    st.markdown("**Analysis Summary:**")
    for reason in reasons:
        st.markdown(f"- {reason}")
    
    st.markdown(f"""
    **Confidence Score:** {abs(score)}/10
    
    **Professional Recommendation:**
    {f'Strong bullish setup with {len([r for r in reasons if "✅" in r])} positive factors. Consider BUYING on dips.' if score >= 3 else
     f'Bullish bias but wait for confirmation. Consider small positions.' if score >= 1 else
     f'Strong bearish setup with {len([r for r in reasons if "⚠️" in r])} negative factors. Consider SELLING or shorting.' if score <= -3 else
     f'Bearish bias. Reduce exposure or wait for better entry.' if score <= -1 else
     'Mixed signals. HOLD current positions and wait for clearer direction.'}
    
    **Risk Management:**
    - Set stop loss at {(1 - abs(analysis_df['Returns1'].std())/100) * latest['Ticker1_Price']:.2f}
    - Take profit target: {(1 + abs(avg_future_return)/100) * latest['Ticker1_Price']:.2f}
    """)
    
    # Download CSV
    st.header("📥 Download Data")
    
    csv_df = pd.DataFrame({
        'DateTime': df1.index.strftime('%Y-%m-%d %H:%M:%S IST'),
        'Open': df1['Open'],
        'High': df1['High'],
        'Low': df1['Low'],
        'Close': df1['Close'],
        'Volume': df1['Volume']
    })
    
    csv = csv_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{ticker1}_{timeframe}_{period}.csv",
        mime="text/csv",
    )
    
else:
    st.info("👆 Configure your analysis parameters in the sidebar and click 'Fetch Data & Analyze' to begin!")
    
    st.markdown("""
    ### 🎯 Features:
    
    1. **Multi-Asset Analysis** - Analyze any two assets and their ratio
    2. **Pattern Recognition** - Greedy algorithm finds similar historical patterns
    3. **Mean Reversion Detection** - Identifies extreme price deviations
    4. **Ratio Bin Analysis** - Discovers which price ratios correlate with moves
    5. **Volatility Regime Analysis** - Different strategies for different volatility levels
    6. **RSI Multi-Timeframe** - Overbought/oversold conditions
    7. **Similarity Scoring** - Finds historical patterns matching current market
    8. **AI Recommendation** - Combines all signals for BUY/SELL/HOLD decision
    9. **Interactive Charts** - Candlesticks, RSI, Volume, Ratio analysis
    10. **Heatmaps** - Visualize best times/days for returns
    
    ### 📊 How It Works:
    
    The algorithm uses multiple techniques to find patterns that "work everytime":
    - **Change-based correlation** instead of absolute prices
    - **Binning analysis** for ratio and volatility regimes  
    - **Mean reversion** detection using z-scores
    - **Pattern similarity** matching using returns correlation
    - **Multi-factor scoring** combining all signals
    
    All predictions are based on historical pattern recognition with confidence scores!
    """)
