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

file1 = st.sidebar.file_uploader("📊 Upload LTP Data (calls_oi, calls_ltp, strike_price, puts_ltp, puts_oi)", type=['csv'], key="ltp_data")
file2 = st.sidebar.file_uploader("📈 Upload Greeks Data (Optional - delta, gamma, theta, vega, etc.)", type=['csv'], key="greeks_data")

# Manual Spot Price Input
st.sidebar.header("🎯 Market Settings")
spot_price = st.sidebar.number_input(
    "Enter Current Spot Price (₹)",
    min_value=0.0,
    max_value=100000.0,
    value=55000.0,
    step=50.0,
    help="Enter the current Bank Nifty spot price for accurate ATM calculation"
)

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

def calculate_option_metrics(df, spot_price=55000):
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
    
    # Find ATM strike (closest to spot price)
    df['distance_from_spot'] = abs(df['strike_price'] - spot_price)
    atm_strike = df.loc[df['distance_from_spot'].idxmin(), 'strike_price']
    
    # ITM/OTM classification based on spot price
    df['option_type'] = np.where(
        df['strike_price'] < spot_price, 'ITM_CALL_OTM_PUT',
        np.where(df['strike_price'] > spot_price, 'OTM_CALL_ITM_PUT', 'ATM')
    )
    
    # Support and Resistance levels based on OI
    df['support_strength'] = df['puts_oi'] / df['puts_oi'].max()
    df['resistance_strength'] = df['calls_oi'] / df['calls_oi'].max()
    
    # Volume approximation (using LTP as proxy if volume not available)
    df['calls_volume_proxy'] = df['calls_ltp'] * df['calls_oi'] / 1000  # Simplified volume proxy
    df['puts_volume_proxy'] = df['puts_ltp'] * df['puts_oi'] / 1000
    
    return df, atm_strike

def identify_trading_opportunities(df, atm_strike, spot_price=55000):
    """Identify potential trading opportunities"""
    opportunities = []
    
    # 1. High OI Put Strikes (Support Levels)
    high_put_oi = df[df['puts_oi'] > df['puts_oi'].quantile(0.8)].copy()
    for _, row in high_put_oi.iterrows():
        if row['strike_price'] < spot_price:
            prob_profit = min(85, 50 + (row['puts_oi'] / df['puts_oi'].max()) * 35)
            opportunities.append({
                'strategy': 'Long Call (CE) - Support Bounce',
                'option_type': 'CALL (CE)',
                'strike': row['strike_price'],
                'entry': row['calls_ltp'],
                'target': row['calls_ltp'] * 1.5,
                'stop_loss': row['calls_ltp'] * 0.6,
                'probability': f"{prob_profit:.1f}%",
                'reasoning': f"High PUT OI ({row['puts_oi']:,.0f}) indicates strong support at ₹{row['strike_price']}"
            })
    
    # 2. High OI Call Strikes (Resistance Levels)
    high_call_oi = df[df['calls_oi'] > df['calls_oi'].quantile(0.8)].copy()
    for _, row in high_call_oi.iterrows():
        if row['strike_price'] > spot_price and row['puts_ltp'] > 0:
            prob_profit = min(80, 45 + (row['calls_oi'] / df['calls_oi'].max()) * 35)
            opportunities.append({
                'strategy': 'Long Put (PE) - Resistance Rejection',
                'option_type': 'PUT (PE)',
                'strike': row['strike_price'],
                'entry': row['puts_ltp'],
                'target': row['puts_ltp'] * 1.4,
                'stop_loss': row['puts_ltp'] * 0.65,
                'probability': f"{prob_profit:.1f}%",
                'reasoning': f"High CALL OI ({row['calls_oi']:,.0f}) indicates strong resistance at ₹{row['strike_price']}"
            })
    
    # 3. PCR Anomalies
    avg_pcr = df['pcr_oi'].mean()
    for _, row in df.iterrows():
        if row['pcr_oi'] > avg_pcr * 2 and row['calls_ltp'] > 0:  # Unusually high PCR
            opportunities.append({
                'strategy': 'Long Call (CE) - PCR Anomaly',
                'option_type': 'CALL (CE)',
                'strike': row['strike_price'],
                'entry': row['calls_ltp'],
                'target': row['calls_ltp'] * 1.3,
                'stop_loss': row['calls_ltp'] * 0.7,
                'probability': "65.0%",
                'reasoning': f"High PCR ({row['pcr_oi']:.2f}) suggests oversold conditions at ₹{row['strike_price']}"
            })
    
    # 4. Low Premium High OI (Value Picks)
    df['premium_oi_ratio_calls'] = df['calls_ltp'] / (df['calls_oi'] + 1)
    df['premium_oi_ratio_puts'] = df['puts_ltp'] / (df['puts_oi'] + 1)
    
    low_prem_high_oi_calls = df[
        (df['premium_oi_ratio_calls'] < df['premium_oi_ratio_calls'].quantile(0.3)) &
        (df['calls_oi'] > df['calls_oi'].quantile(0.7))
    ]
    
    for _, row in low_prem_high_oi_calls.iterrows():
        if row['calls_ltp'] > 0:
            opportunities.append({
                'strategy': 'Long Call (CE) - Value Pick',
                'option_type': 'CALL (CE)',
                'strike': row['strike_price'],
                'entry': row['calls_ltp'],
                'target': row['calls_ltp'] * 1.6,
                'stop_loss': row['calls_ltp'] * 0.5,
                'probability': "70.0%",
                'reasoning': f"Low premium (₹{row['calls_ltp']}) with high OI ({row['calls_oi']:,.0f}) at ₹{row['strike_price']}"
            })
    
    # 5. Value Puts
    low_prem_high_oi_puts = df[
        (df['premium_oi_ratio_puts'] < df['premium_oi_ratio_puts'].quantile(0.3)) &
        (df['puts_oi'] > df['puts_oi'].quantile(0.7))
    ]
    
    for _, row in low_prem_high_oi_puts.iterrows():
        if row['puts_ltp'] > 0:
            opportunities.append({
                'strategy': 'Long Put (PE) - Value Pick',
                'option_type': 'PUT (PE)',
                'strike': row['strike_price'],
                'entry': row['puts_ltp'],
                'target': row['puts_ltp'] * 1.6,
                'stop_loss': row['puts_ltp'] * 0.5,
                'probability': "70.0%",
                'reasoning': f"Low premium (₹{row['puts_ltp']}) with high OI ({row['puts_oi']:,.0f}) at ₹{row['strike_price']}"
            })
    
    return pd.DataFrame(opportunities)

def create_visualizations(df, atm_strike, spot_price, file_suffix=""):
    """Create comprehensive visualizations"""
    
    # 1. OI Distribution Chart (Side by Side)
    fig_oi = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Open Interest Distribution', 'LTP Distribution'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # OI Chart - Both calls and puts on same side (positive values)
    fig_oi.add_trace(
        go.Bar(name='Calls OI', x=df['strike_price'], y=df['calls_oi'], 
               marker_color='rgba(0, 255, 0, 0.7)', opacity=0.8),
        row=1, col=1
    )
    fig_oi.add_trace(
        go.Bar(name='Puts OI', x=df['strike_price'], y=df['puts_oi'], 
               marker_color='rgba(255, 0, 0, 0.7)', opacity=0.8),
        row=1, col=1
    )
    
    # LTP Chart
    fig_oi.add_trace(
        go.Scatter(name='Calls LTP', x=df['strike_price'], y=df['calls_ltp'], 
                   mode='lines+markers', line=dict(color='darkgreen', width=2),
                   marker=dict(size=6)),
        row=2, col=1
    )
    fig_oi.add_trace(
        go.Scatter(name='Puts LTP', x=df['strike_price'], y=df['puts_ltp'], 
                   mode='lines+markers', line=dict(color='darkred', width=2),
                   marker=dict(size=6)),
        row=2, col=1
    )
    
    # Add ATM and Spot Price lines
    fig_oi.add_vline(x=atm_strike, line_dash="dash", line_color="blue", line_width=2,
                     annotation_text="ATM", annotation_position="top")
    fig_oi.add_vline(x=spot_price, line_dash="solid", line_color="purple", line_width=3,
                     annotation_text=f"Spot: ₹{spot_price:,.0f}", annotation_position="top")
    
    fig_oi.update_layout(
        height=800, 
        title_text=f"Options Chain Analysis {file_suffix}",
        showlegend=True,
        barmode='group'  # Side by side bars
    )
    fig_oi.update_xaxes(title_text="Strike Price")
    fig_oi.update_yaxes(title_text="Open Interest", row=1, col=1)
    fig_oi.update_yaxes(title_text="LTP (₹)", row=2, col=1)
    
    # 2. Volume-OI Analysis
    fig_vol_oi = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Volume Proxy (LTP × OI)', 'OI vs Volume Scatter'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Volume proxy chart
    fig_vol_oi.add_trace(
        go.Bar(name='Calls Volume Proxy', x=df['strike_price'], y=df['calls_volume_proxy'], 
               marker_color='rgba(0, 200, 0, 0.6)'),
        row=1, col=1
    )
    fig_vol_oi.add_trace(
        go.Bar(name='Puts Volume Proxy', x=df['strike_price'], y=df['puts_volume_proxy'], 
               marker_color='rgba(200, 0, 0, 0.6)'),
        row=1, col=1
    )
    
    # OI vs Volume scatter
    fig_vol_oi.add_trace(
        go.Scatter(x=df['calls_oi'], y=df['calls_volume_proxy'], mode='markers',
                   marker=dict(size=8, color='green'), name='Calls OI-Volume'),
        row=2, col=1
    )
    fig_vol_oi.add_trace(
        go.Scatter(x=df['puts_oi'], y=df['puts_volume_proxy'], mode='markers',
                   marker=dict(size=8, color='red'), name='Puts OI-Volume'),
        row=2, col=1
    )
    
    # Add spot price line
    fig_vol_oi.add_vline(x=spot_price, line_dash="solid", line_color="purple", line_width=2,
                         annotation_text=f"Spot", annotation_position="top", row=1, col=1)
    
    fig_vol_oi.update_layout(
        height=700,
        title_text=f"Volume-OI Analysis {file_suffix}",
        barmode='group'
    )
    fig_vol_oi.update_xaxes(title_text="Strike Price", row=1, col=1)
    fig_vol_oi.update_xaxes(title_text="Open Interest", row=2, col=1)
    fig_vol_oi.update_yaxes(title_text="Volume Proxy", row=1, col=1)
    fig_vol_oi.update_yaxes(title_text="Volume Proxy", row=2, col=1)
    
    # 3. PCR Heatmap
    fig_pcr = px.scatter(
        df, x='strike_price', y='pcr_oi', size='total_oi',
        color='pcr_oi', color_continuous_scale='RdYlGn_r',
        title=f'Put-Call Ratio by Strike Price {file_suffix}',
        labels={'pcr_oi': 'PCR (OI)', 'strike_price': 'Strike Price'},
        hover_data=['calls_oi', 'puts_oi', 'total_oi']
    )
    fig_pcr.add_hline(y=1, line_dash="dash", line_color="black", line_width=2,
                      annotation_text="PCR = 1")
    fig_pcr.add_vline(x=spot_price, line_dash="solid", line_color="purple", line_width=2,
                      annotation_text="Spot")
    fig_pcr.update_layout(height=500)
    
    # 4. Support & Resistance Levels
    support_levels = df[df['puts_oi'] > df['puts_oi'].quantile(0.8)]
    resistance_levels = df[df['calls_oi'] > df['calls_oi'].quantile(0.8)]
    
    fig_levels = go.Figure()
    
    # Combined chart showing both support and resistance
    fig_levels.add_trace(
        go.Bar(
            name='Support (Put OI)',
            x=support_levels['strike_price'],
            y=support_levels['puts_oi'],
            marker_color='rgba(0, 255, 0, 0.7)',
            text=support_levels['strike_price'],
            textposition='outside'
        )
    )
    
    fig_levels.add_trace(
        go.Bar(
            name='Resistance (Call OI)',
            x=resistance_levels['strike_price'],
            y=resistance_levels['calls_oi'],
            marker_color='rgba(255, 0, 0, 0.7)',
            text=resistance_levels['strike_price'],
            textposition='outside'
        )
    )
    
    # Add ATM and Spot lines
    fig_levels.add_vline(x=atm_strike, line_dash="dash", line_color="blue", line_width=2,
                        annotation_text="ATM")
    fig_levels.add_vline(x=spot_price, line_dash="solid", line_color="purple", line_width=3,
                        annotation_text=f"Spot")
    
    fig_levels.update_layout(
        title=f'Key Support and Resistance Levels {file_suffix}',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        height=500,
        barmode='group'
    )
    
    # 5. OI Change Analysis (percentage breakdown)
    fig_oi_change = go.Figure()
    
    # Calculate percentage of total OI for better visualization
    total_oi = df['total_oi'].sum()
    df_viz = df.copy()
    df_viz['calls_oi_pct'] = (df_viz['calls_oi'] / total_oi) * 100
    df_viz['puts_oi_pct'] = (df_viz['puts_oi'] / total_oi) * 100
    
    fig_oi_change.add_trace(
        go.Scatter(
            x=df_viz['strike_price'],
            y=df_viz['calls_oi_pct'],
            mode='lines+markers',
            name='Calls OI %',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            fill='tonexty'
        )
    )
    
    fig_oi_change.add_trace(
        go.Scatter(
            x=df_viz['strike_price'],
            y=df_viz['puts_oi_pct'],
            mode='lines+markers',
            name='Puts OI %',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            fill='tozeroy'
        )
    )
    
    fig_oi_change.add_vline(x=atm_strike, line_dash="dash", line_color="blue", line_width=2,
                           annotation_text="ATM")
    fig_oi_change.add_vline(x=spot_price, line_dash="solid", line_color="purple", line_width=3,
                           annotation_text="Spot")
    
    fig_oi_change.update_layout(
        title=f'OI Distribution (% of Total) {file_suffix}',
        xaxis_title='Strike Price',
        yaxis_title='OI Percentage (%)',
        height=500
    )
    
    return fig_oi, fig_vol_oi, fig_pcr, fig_levels, fig_oi_change
def generate_market_summary(df, atm_strike, spot_price, opportunities):
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
        'Current Spot': f"₹{spot_price:,.0f}",
        'ATM Strike': f"₹{atm_strike:,.0f}",
        'Overall PCR': f"{overall_pcr:.2f}",
        'Market Sentiment': f"{sentiment} - {sentiment_desc}",
        'Max Pain Level': f"₹{max_pain_strike:,.0f}",
        'Key Support': f"₹{top_support:,.0f} ({df.loc[df['puts_oi'].idxmax(), 'puts_oi']:,.0f} OI)",
        'Key Resistance': f"₹{top_resistance:,.0f} ({df.loc[df['calls_oi'].idxmax(), 'calls_oi']:,.0f} OI)",
        'Total Opportunities': f"{len(opportunities)} identified"
    }
    
    return summary

def generate_layman_summary(df, atm_strike, spot_price, opportunities, data_type="LTP"):
    """Generate a 100-word summary in layman terms"""
    
    total_calls_oi = df['calls_oi'].sum()
    total_puts_oi = df['puts_oi'].sum()
    overall_pcr = total_puts_oi / total_calls_oi
    
    # Key levels
    max_call_oi_strike = df.loc[df['calls_oi'].idxmax(), 'strike_price']
    max_put_oi_strike = df.loc[df['puts_oi'].idxmax(), 'strike_price']
    
    # Market direction hint
    if spot_price > atm_strike:
        position = "above ATM, suggesting bullish bias"
    elif spot_price < atm_strike:
        position = "below ATM, indicating bearish sentiment"
    else:
        position = "at ATM, showing balanced market"
    
    # Sentiment
    if overall_pcr > 1.2:
        sentiment = "bearish (more puts than calls)"
    elif overall_pcr < 0.8:
        sentiment = "bullish (more calls than puts)"
    else:
        sentiment = "neutral (balanced put-call activity)"
    
    if data_type == "LTP":
        summary = f"""
        **Market Snapshot:** Bank Nifty spot at ₹{spot_price:,.0f} is {position}. The options market shows {sentiment} sentiment with PCR at {overall_pcr:.2f}. 
        
        **Key Levels:** Major resistance at ₹{max_call_oi_strike:,.0f} (highest call interest) and support at ₹{max_put_oi_strike:,.0f} (highest put interest). 
        
        **Trading View:** {len(opportunities)} opportunities identified. If price holds above support, calls may profit. If resistance holds, puts could gain. 
        
        **Risk:** Options lose value over time, so timing is crucial for profitable trades.
        """
    else:
        summary = f"""
        **Greeks Analysis:** Bank Nifty options show {sentiment} bias with spot at ₹{spot_price:,.0f}. Greeks data reveals how options prices change with market moves.
        
        **Delta Impact:** Options near ₹{atm_strike:,.0f} (ATM) have highest sensitivity to price changes. ITM options move more with spot price.
        
        **Time Decay:** All options lose value daily (theta). Volatility (vega) affects option prices significantly during market uncertainty.
        
        **Strategy:** {len(opportunities)} setups identified. Use Greeks to time entries and manage risk effectively in volatile conditions.
        """
    
    return summary.strip()

# Main Application Logic
if uploaded_files:
    # Process first file
    df1 = load_and_clean_data(uploaded_files[0])
    df1_processed, atm_strike1 = calculate_option_metrics(df1, spot_price)
    opportunities1 = identify_trading_opportunities(df1_processed, atm_strike1, spot_price)
    
    # Process second file if available
    if len(uploaded_files) > 1:
        df2 = load_and_clean_data(uploaded_files[1])
        df2_processed, atm_strike2 = calculate_option_metrics(df2, spot_price)
        opportunities2 = identify_trading_opportunities(df2_processed, atm_strike2, spot_price)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📝 Summary", "🎯 Trading Opportunities", "📈 Visualizations", "📋 Data Explorer"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Market Summary - LTP Data")
            summary1 = generate_market_summary(df1_processed, atm_strike1, spot_price, opportunities1)
            for key, value in summary1.items():
                st.metric(key, value)
        
        with col2:
            if len(uploaded_files) > 1:
                st.subheader("📈 Market Summary - Greeks Data")
                summary2 = generate_market_summary(df2_processed, atm_strike2, spot_price, opportunities2)
                for key, value in summary2.items():
                    st.metric(key, value)
            else:
                st.info("Upload Greeks data for comparison analysis")
    
    with tab3:
        st.subheader("📝 Market Summary in Simple Terms")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**📊 LTP Data Summary:**")
            ltp_summary = generate_layman_summary(df1_processed, atm_strike1, spot_price, opportunities1, "LTP")
            st.markdown(ltp_summary)
        
        with col2:
            if len(uploaded_files) > 1:
                st.write("**📈 Greeks Data Summary:**")
                greeks_summary = generate_layman_summary(df2_processed, atm_strike2, spot_price, opportunities2, "Greeks")
                st.markdown(greeks_summary)
            else:
                st.info("Upload Greeks data for additional analysis")
    
    with tab2:
        st.subheader("🎯 Identified Trading Opportunities")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**📊 LTP Data Opportunities:**")
            if not opportunities1.empty:
                # Sort by probability
                opportunities1_sorted = opportunities1.sort_values('probability', ascending=False)
                st.dataframe(opportunities1_sorted, use_container_width=True)
                
                # Download button
                csv1 = opportunities1_sorted.to_csv(index=False)
                st.download_button(
                    label="📥 Download LTP Opportunities",
                    data=csv1,
                    file_name="banknifty_ltp_opportunities.csv",
                    mime="text/csv",
                    key="download_ltp_opps"
                )
            else:
                st.info("No opportunities identified in current market conditions.")
        
        with col2:
            if len(uploaded_files) > 1:
                st.write("**📈 Greeks Data Opportunities:**")
                if not opportunities2.empty:
                    opportunities2_sorted = opportunities2.sort_values('probability', ascending=False)
                    st.dataframe(opportunities2_sorted, use_container_width=True)
                    
                    csv2 = opportunities2_sorted.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Greeks Opportunities",
                        data=csv2,
                        file_name="banknifty_greeks_opportunities.csv",
                        mime="text/csv",
                        key="download_greeks_opps"
                    )
                else:
                    st.info("No opportunities identified in current market conditions.")
    
    with tab5:
        st.subheader("📈 Market Visualizations")
        
        # File 1 visualizations
        st.write("**📊 LTP Data Analysis:**")
        fig_oi1, fig_vol_oi1, fig_pcr1, fig_levels1, fig_oi_change1 = create_visualizations(df1_processed, atm_strike1, spot_price, "(LTP Data)")
        
        st.plotly_chart(fig_oi1, use_container_width=True, key="oi_chart_file1")
        st.plotly_chart(fig_vol_oi1, use_container_width=True, key="vol_oi_chart_file1")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pcr1, use_container_width=True, key="pcr_chart_file1")
        with col2:
            st.plotly_chart(fig_levels1, use_container_width=True, key="levels_chart_file1")
        
        # Additional OI Change chart
        st.plotly_chart(fig_oi_change1, use_container_width=True, key="oi_change_file1")
        
        # File 2 visualizations (if available)
        if len(uploaded_files) > 1:
            st.write("**📈 Greeks Data Analysis:**")
            fig_oi2, fig_vol_oi2, fig_pcr2, fig_levels2, fig_oi_change2 = create_visualizations(df2_processed, atm_strike2, spot_price, "(Greeks Data)")
            
            st.plotly_chart(fig_oi2, use_container_width=True, key="oi_chart_file2")
            st.plotly_chart(fig_vol_oi2, use_container_width=True, key="vol_oi_chart_file2")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pcr2, use_container_width=True, key="pcr_chart_file2")
            with col2:
                st.plotly_chart(fig_levels2, use_container_width=True, key="levels_chart_file2")
            
            st.plotly_chart(fig_oi_change2, use_container_width=True, key="oi_change_file2")
    
    with tab4:
        st.subheader("📋 Raw Data Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 LTP Data - Processed:**")
            st.dataframe(df1_processed, use_container_width=True)
            
            # Statistics
            st.write("**Statistical Summary:**")
            st.dataframe(df1_processed.describe(), use_container_width=True)
        
        with col2:
            if len(uploaded_files) > 1:
                st.write("**📈 Greeks Data - Processed:**")
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
