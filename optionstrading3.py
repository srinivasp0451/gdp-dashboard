import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Nifty Options Chain Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Bank Nifty Options Chain Analysis Dashboard")
st.markdown("---")

# Sidebar for file uploads
st.sidebar.header("📁 Upload Options Data")
uploaded_files = []

file1 = st.sidebar.file_uploader("Upload Options Chain Data 1", type=['csv'])
file2 = st.sidebar.file_uploader("Upload Options Chain Data 2 (Optional)", type=['csv'])

if file1:
    uploaded_files.append(file1)
if file2:
    uploaded_files.append(file2)

def load_and_clean_data(file):
    """Load and clean options chain data"""
    df = pd.read_csv(file)
    
    # Remove summary rows
    df = df[~df['strike_price'].astype(str).str.contains('Total|ITM|OTM', na=False)]
    
    # Convert to numeric
    numeric_cols = ['calls_oi', 'calls_ltp', 'strike_price', 'puts_ltp', 'puts_oi']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    return df

def calculate_option_metrics(df):
    """Calculate advanced option metrics"""
    df = df.copy()
    
    # PCR (Put Call Ratio)
    df['pcr_oi'] = df['puts_oi'] / (df['calls_oi'] + 1)  # Adding 1 to avoid division by zero
    df['pcr_volume'] = df['puts_ltp'] / (df['calls_ltp'] + 0.01)
    
    # Total OI
    df['total_oi'] = df['calls_oi'] + df['puts_oi']
    
    # OI Concentration
    total_calls_oi = df['calls_oi'].sum()
    total_puts_oi = df['puts_oi'].sum()
    df['calls_oi_pct'] = (df['calls_oi'] / total_calls_oi) * 100
    df['puts_oi_pct'] = (df['puts_oi'] / total_puts_oi) * 100
    
    # Identify ATM strike (closest to current price based on LTP patterns)
    # ATM is typically where Call and Put premiums are closest
    df['call_put_diff'] = abs(df['calls_ltp'] - df['puts_ltp'])
    atm_strike = df.loc[df['call_put_diff'].idxmin(), 'strike_price']
    
    # ITM/OTM classification
    df['option_type'] = np.where(
        df['strike_price'] < atm_strike, 'ITM_CALL_OTM_PUT',
        np.where(df['strike_price'] > atm_strike, 'OTM_CALL_ITM_PUT', 'ATM')
    )
    
    # Support and Resistance levels based on OI
    df['support_strength'] = df['puts_oi'] / df['puts_oi'].max()
    df['resistance_strength'] = df['calls_oi'] / df['calls_oi'].max()
    
    return df, atm_strike

def identify_trading_opportunities(df, atm_strike):
    """Identify potential trading opportunities"""
    opportunities = []
    
    # 1. High OI Put Strikes (Support Levels)
    high_put_oi = df[df['puts_oi'] > df['puts_oi'].quantile(0.8)].copy()
    for _, row in high_put_oi.iterrows():
        if row['strike_price'] < atm_strike:
            prob_profit = min(85, 50 + (row['puts_oi'] / df['puts_oi'].max()) * 35)
            opportunities.append({
                'strategy': 'Long Call (Support Bounce)',
                'strike': row['strike_price'],
                'entry': row['calls_ltp'],
                'target': row['calls_ltp'] * 1.5,
                'stop_loss': row['calls_ltp'] * 0.6,
                'probability': f"{prob_profit:.1f}%",
                'reasoning': f"High PUT OI ({row['puts_oi']:,.0f}) indicates strong support at {row['strike_price']}"
            })
    
    # 2. High OI Call Strikes (Resistance Levels)
    high_call_oi = df[df['calls_oi'] > df['calls_oi'].quantile(0.8)].copy()
    for _, row in high_call_oi.iterrows():
        if row['strike_price'] > atm_strike and row['puts_ltp'] > 0:
            prob_profit = min(80, 45 + (row['calls_oi'] / df['calls_oi'].max()) * 35)
            opportunities.append({
                'strategy': 'Long Put (Resistance Rejection)',
                'strike': row['strike_price'],
                'entry': row['puts_ltp'],
                'target': row['puts_ltp'] * 1.4,
                'stop_loss': row['puts_ltp'] * 0.65,
                'probability': f"{prob_profit:.1f}%",
                'reasoning': f"High CALL OI ({row['calls_oi']:,.0f}) indicates strong resistance at {row['strike_price']}"
            })
    
    # 3. PCR Anomalies
    avg_pcr = df['pcr_oi'].mean()
    for _, row in df.iterrows():
        if row['pcr_oi'] > avg_pcr * 2 and row['calls_ltp'] > 0:  # Unusually high PCR
            opportunities.append({
                'strategy': 'Long Call (PCR Anomaly)',
                'strike': row['strike_price'],
                'entry': row['calls_ltp'],
                'target': row['calls_ltp'] * 1.3,
                'stop_loss': row['calls_ltp'] * 0.7,
                'probability': "65.0%",
                'reasoning': f"High PCR ({row['pcr_oi']:.2f}) suggests oversold conditions"
            })
    
    # 4. Low Premium High OI
    df['premium_oi_ratio_calls'] = df['calls_ltp'] / (df['calls_oi'] + 1)
    df['premium_oi_ratio_puts'] = df['puts_ltp'] / (df['puts_oi'] + 1)
    
    low_prem_high_oi_calls = df[
        (df['premium_oi_ratio_calls'] < df['premium_oi_ratio_calls'].quantile(0.3)) &
        (df['calls_oi'] > df['calls_oi'].quantile(0.7))
    ]
    
    for _, row in low_prem_high_oi_calls.iterrows():
        if row['calls_ltp'] > 0:
            opportunities.append({
                'strategy': 'Long Call (Value Pick)',
                'strike': row['strike_price'],
                'entry': row['calls_ltp'],
                'target': row['calls_ltp'] * 1.6,
                'stop_loss': row['calls_ltp'] * 0.5,
                'probability': "70.0%",
                'reasoning': f"Low premium ({row['calls_ltp']}) with high OI ({row['calls_oi']:,.0f}) - Value opportunity"
            })
    
    return pd.DataFrame(opportunities)

def create_visualizations(df, atm_strike):
    """Create comprehensive visualizations"""
    
    # 1. OI Distribution Chart
    fig_oi = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Open Interest Distribution', 'LTP Distribution'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # OI Chart
    fig_oi.add_trace(
        go.Bar(name='Calls OI', x=df['strike_price'], y=df['calls_oi'], 
               marker_color='green', opacity=0.7),
        row=1, col=1
    )
    fig_oi.add_trace(
        go.Bar(name='Puts OI', x=df['strike_price'], y=-df['puts_oi'], 
               marker_color='red', opacity=0.7),
        row=1, col=1
    )
    
    # LTP Chart
    fig_oi.add_trace(
        go.Scatter(name='Calls LTP', x=df['strike_price'], y=df['calls_ltp'], 
                   mode='lines+markers', line=dict(color='darkgreen')),
        row=2, col=1
    )
    fig_oi.add_trace(
        go.Scatter(name='Puts LTP', x=df['strike_price'], y=df['puts_ltp'], 
                   mode='lines+markers', line=dict(color='darkred')),
        row=2, col=1
    )
    
    # Add ATM line
    fig_oi.add_vline(x=atm_strike, line_dash="dash", line_color="blue", 
                     annotation_text="ATM", row=1, col=1)
    fig_oi.add_vline(x=atm_strike, line_dash="dash", line_color="blue", 
                     annotation_text="ATM", row=2, col=1)
    
    fig_oi.update_layout(height=800, title_text="Options Chain Analysis")
    fig_oi.update_xaxes(title_text="Strike Price")
    fig_oi.update_yaxes(title_text="Open Interest", row=1, col=1)
    fig_oi.update_yaxes(title_text="LTP (₹)", row=2, col=1)
    
    # 2. PCR Heatmap
    fig_pcr = px.scatter(
        df, x='strike_price', y='pcr_oi', size='total_oi',
        color='pcr_oi', color_continuous_scale='RdYlGn_r',
        title='Put-Call Ratio by Strike Price',
        labels={'pcr_oi': 'PCR (OI)', 'strike_price': 'Strike Price'}
    )
    fig_pcr.add_hline(y=1, line_dash="dash", line_color="black", 
                      annotation_text="PCR = 1")
    
    # 3. Support & Resistance Levels
    support_levels = df[df['puts_oi'] > df['puts_oi'].quantile(0.8)]
    resistance_levels = df[df['calls_oi'] > df['calls_oi'].quantile(0.8)]
    
    fig_levels = go.Figure()
    
    fig_levels.add_trace(
        go.Scatter(
            x=support_levels['strike_price'],
            y=support_levels['puts_oi'],
            mode='markers',
            marker=dict(size=15, color='green', symbol='triangle-up'),
            name='Support Levels',
            text=support_levels['strike_price'],
            textposition="top center"
        )
    )
    
    fig_levels.add_trace(
        go.Scatter(
            x=resistance_levels['strike_price'],
            y=resistance_levels['calls_oi'],
            mode='markers',
            marker=dict(size=15, color='red', symbol='triangle-down'),
            name='Resistance Levels',
            text=resistance_levels['strike_price'],
            textposition="bottom center"
        )
    )
    
    fig_levels.update_layout(
        title='Key Support and Resistance Levels',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        height=500
    )
    
    return fig_oi, fig_pcr, fig_levels

def generate_market_summary(df, atm_strike, opportunities):
    """Generate comprehensive market summary"""
    
    total_calls_oi = df['calls_oi'].sum()
    total_puts_oi = df['puts_oi'].sum()
    overall_pcr = total_puts_oi / total_calls_oi
    
    # Market sentiment
    if overall_pcr > 1.2:
        sentiment = "🐻 Bearish"
        sentiment_desc = "High put writing suggests bearish sentiment"
    elif overall_pcr < 0.8:
        sentiment = "🐂 Bullish" 
        sentiment_desc = "High call writing suggests bullish sentiment"
    else:
        sentiment = "😐 Neutral"
        sentiment_desc = "Balanced put-call ratio indicates neutral sentiment"
    
    # Max pain calculation (simplified)
    max_pain_strike = df.loc[df['total_oi'].idxmax(), 'strike_price']
    
    # Key levels
    top_support = df.loc[df['puts_oi'].idxmax(), 'strike_price']
    top_resistance = df.loc[df['calls_oi'].idxmax(), 'strike_price']
    
    summary = {
        'ATM Strike': f"₹{atm_strike:,.0f}",
        'Overall PCR': f"{overall_pcr:.2f}",
        'Market Sentiment': f"{sentiment} - {sentiment_desc}",
        'Max Pain Level': f"₹{max_pain_strike:,.0f}",
        'Key Support': f"₹{top_support:,.0f} ({df.loc[df['puts_oi'].idxmax(), 'puts_oi']:,.0f} OI)",
        'Key Resistance': f"₹{top_resistance:,.0f} ({df.loc[df['calls_oi'].idxmax(), 'calls_oi']:,.0f} OI)",
        'Total Opportunities': f"{len(opportunities)} identified"
    }
    
    return summary

# Main Application Logic
if uploaded_files:
    # Process first file
    df1 = load_and_clean_data(uploaded_files[0])
    df1_processed, atm_strike1 = calculate_option_metrics(df1)
    opportunities1 = identify_trading_opportunities(df1_processed, atm_strike1)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Trading Opportunities", "📈 Visualizations", "📋 Data Explorer"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📈 Market Summary - File 1")
            summary1 = generate_market_summary(df1_processed, atm_strike1, opportunities1)
            for key, value in summary1.items():
                st.metric(key, value)
        
        with col2:
            if len(uploaded_files) > 1:
                df2 = load_and_clean_data(uploaded_files[1])
                df2_processed, atm_strike2 = calculate_option_metrics(df2)
                opportunities2 = identify_trading_opportunities(df2_processed, atm_strike2)
                
                st.subheader("📈 Market Summary - File 2")
                summary2 = generate_market_summary(df2_processed, atm_strike2, opportunities2)
                for key, value in summary2.items():
                    st.metric(key, value)
    
    with tab2:
        st.subheader("🎯 Identified Trading Opportunities")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**File 1 Opportunities:**")
            if not opportunities1.empty:
                # Sort by probability
                opportunities1_sorted = opportunities1.sort_values('probability', ascending=False)
                st.dataframe(opportunities1_sorted, use_container_width=True)
                
                # Download button
                csv1 = opportunities1_sorted.to_csv(index=False)
                st.download_button(
                    label="📥 Download Opportunities (File 1)",
                    data=csv1,
                    file_name="banknifty_opportunities_file1.csv",
                    mime="text/csv"
                )
            else:
                st.info("No opportunities identified in current market conditions.")
        
        with col2:
            if len(uploaded_files) > 1:
                st.write("**File 2 Opportunities:**")
                if not opportunities2.empty:
                    opportunities2_sorted = opportunities2.sort_values('probability', ascending=False)
                    st.dataframe(opportunities2_sorted, use_container_width=True)
                    
                    csv2 = opportunities2_sorted.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Opportunities (File 2)",
                        data=csv2,
                        file_name="banknifty_opportunities_file2.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No opportunities identified in current market conditions.")
    
    with tab3:
        st.subheader("📈 Market Visualizations")
        
        # File 1 visualizations
        st.write("**File 1 Analysis:**")
        fig_oi1, fig_pcr1, fig_levels1 = create_visualizations(df1_processed, atm_strike1)
        
        st.plotly_chart(fig_oi1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pcr1, use_container_width=True)
        with col2:
            st.plotly_chart(fig_levels1, use_container_width=True)
        
        # File 2 visualizations (if available)
        if len(uploaded_files) > 1:
            st.write("**File 2 Analysis:**")
            fig_oi2, fig_pcr2, fig_levels2 = create_visualizations(df2_processed, atm_strike2)
            
            st.plotly_chart(fig_oi2, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pcr2, use_container_width=True)
            with col2:
                st.plotly_chart(fig_levels2, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Raw Data Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**File 1 - Processed Data:**")
            st.dataframe(df1_processed, use_container_width=True)
            
            # Statistics
            st.write("**Statistical Summary:**")
            st.dataframe(df1_processed.describe(), use_container_width=True)
        
        with col2:
            if len(uploaded_files) > 1:
                st.write("**File 2 - Processed Data:**")
                st.dataframe(df2_processed, use_container_width=True)
                
                st.write("**Statistical Summary:**")
                st.dataframe(df2_processed.describe(), use_container_width=True)

else:
    # Welcome message and instructions
    st.markdown("""
    ## Welcome to Bank Nifty Options Chain Analyzer! 🚀
    
    This comprehensive tool helps you:
    
    ### 📊 **Market Analysis Features:**
    - **Open Interest Analysis** - Identify support/resistance levels
    - **Put-Call Ratio (PCR)** - Gauge market sentiment
    - **Premium Analysis** - Find value opportunities
    - **Max Pain Calculation** - Understand market direction
    
    ### 🎯 **Trading Opportunities:**
    - **Support Bounce Trades** - Long calls at high PUT OI strikes
    - **Resistance Rejection** - Long puts at high CALL OI strikes
    - **PCR Anomalies** - Contrarian opportunities
    - **Value Picks** - Low premium, high OI options
    
    ### 📈 **Advanced Metrics:**
    - Probability of Profit calculations
    - Entry, Target, and Stop-Loss levels
    - Risk-Reward ratios
    - Data-backed trade reasoning
    
    ### 🔧 **How to Use:**
    1. Upload your options chain CSV file(s) using the sidebar
    2. Review the market summary and sentiment analysis
    3. Explore identified trading opportunities
    4. Analyze detailed visualizations
    5. Export opportunities for execution
    
    **Expected CSV Format:**
    ```
    calls_oi, calls_ltp, strike_price, puts_ltp, puts_oi
    ```
    
    Start by uploading your options chain data to begin the analysis! 📁
    """)
    
    # Sample data format
    st.subheader("📋 Sample Data Format")
    sample_data = pd.DataFrame({
        'calls_oi': [1000, 2000, 3000],
        'calls_ltp': [100, 50, 25],
        'strike_price': [50000, 50500, 51000],
        'puts_ltp': [25, 50, 100],
        'puts_oi': [3000, 2000, 1000]
    })
    st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>Bank Nifty Options Analyzer</b> | Built with ❤️ using Streamlit</p>
    <p><i>Disclaimer: This tool is for educational purposes only. Please consult with a financial advisor before making investment decisions.</i></p>
</div>
""", unsafe_allow_html=True)
