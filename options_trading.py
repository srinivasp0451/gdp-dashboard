import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NIFTY Options Trading Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3498db, #2980b9);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    
    .opportunity-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #27ae60;
    }
    
    .call-option {
        border-left-color: #27ae60 !important;
    }
    
    .put-option {
        border-left-color: #e74c3c !important;
    }
    
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    
    .volume-high { background-color: #27ae60; }
    .volume-medium { background-color: #f39c12; }
    .volume-low { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

def load_and_process_data(uploaded_file):
    """Load and process uploaded CSV data"""
    try:
        # Read CSV with proper handling of whitespace in headers
        df = pd.read_csv(uploaded_file)
        
        # Clean column names - strip whitespace
        df.columns = df.columns.str.strip()
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        
        # Convert numeric columns
        numeric_cols = ['Strike Price', 'Open', 'High', 'Low', 'Close', 'LTP', 'Settle Price', 
                       'No. of contracts', 'Open Int', 'Change in OI', 'Underlying Value']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate additional metrics
        df['Price Change'] = df['Close'] - df['Open']
        df['Price Change %'] = (df['Price Change'] / df['Open']) * 100
        df['Volume Score'] = pd.qcut(df['No. of contracts'].fillna(0), q=3, labels=['Low', 'Medium', 'High'])
        df['OI Score'] = pd.qcut(df['Change in OI'].fillna(0), q=3, labels=['Low', 'Medium', 'High'])
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def calculate_trading_opportunities(ce_data, pe_data):
    """Calculate trading opportunities based on options data"""
    opportunities = []
    
    if ce_data is not None:
        # Call options analysis
        ce_opportunities = analyze_call_options(ce_data)
        opportunities.extend(ce_opportunities)
    
    if pe_data is not None:
        # Put options analysis
        pe_opportunities = analyze_put_options(pe_data)
        opportunities.extend(pe_opportunities)
    
    return opportunities

def analyze_call_options(ce_data):
    """Analyze call options for trading opportunities"""
    opportunities = []
    
    # Filter for high volume and OI options
    latest_date = ce_data['Date'].max()
    latest_data = ce_data[ce_data['Date'] == latest_date].copy()
    
    # Sort by volume and OI
    latest_data = latest_data.sort_values(['No. of contracts', 'Change in OI'], ascending=False)
    
    # Get top opportunities
    top_calls = latest_data.head(10)
    
    for _, row in top_calls.iterrows():
        if pd.notna(row['Close']) and row['Close'] > 0:
            strike = row['Strike Price']
            current_price = row['Close']
            
            # Calculate entry, target, and stop loss
            entry_price = current_price * 1.02  # 2% above close
            target_price = current_price * 1.4   # 40% target
            stop_loss = current_price * 0.85     # 15% stop loss
            
            # Risk metrics
            max_loss = entry_price - stop_loss
            max_gain = target_price - entry_price
            risk_reward = max_gain / max_loss if max_loss > 0 else 0
            
            # Volume classification
            volume_class = classify_volume(row['No. of contracts'], latest_data['No. of contracts'])
            oi_change_class = classify_oi_change(row['Change in OI'])
            
            opportunity = {
                'Type': 'CALL',
                'Strike': strike,
                'Entry': entry_price,
                'Target': target_price,
                'Stop_Loss': stop_loss,
                'Current_Price': current_price,
                'Risk_Reward': risk_reward,
                'Max_Loss': max_loss,
                'Max_Gain': max_gain,
                'Volume': row['No. of contracts'],
                'Volume_Class': volume_class,
                'OI_Change': row['Change in OI'],
                'OI_Class': oi_change_class,
                'Rationale': generate_rationale('CALL', row, volume_class, oi_change_class)
            }
            opportunities.append(opportunity)
    
    return opportunities

def analyze_put_options(pe_data):
    """Analyze put options for trading opportunities"""
    opportunities = []
    
    # Filter for high volume and OI options
    latest_date = pe_data['Date'].max()
    latest_data = pe_data[pe_data['Date'] == latest_date].copy()
    
    # Sort by volume and OI
    latest_data = latest_data.sort_values(['No. of contracts', 'Change in OI'], ascending=False)
    
    # Get top opportunities
    top_puts = latest_data.head(10)
    
    for _, row in top_puts.iterrows():
        if pd.notna(row['Close']) and row['Close'] > 0:
            strike = row['Strike Price']
            current_price = row['Close']
            
            # Calculate entry, target, and stop loss
            entry_price = current_price * 1.02  # 2% above close
            target_price = current_price * 1.5   # 50% target
            stop_loss = current_price * 0.8      # 20% stop loss
            
            # Risk metrics
            max_loss = entry_price - stop_loss
            max_gain = target_price - entry_price
            risk_reward = max_gain / max_loss if max_loss > 0 else 0
            
            # Volume classification
            volume_class = classify_volume(row['No. of contracts'], latest_data['No. of contracts'])
            oi_change_class = classify_oi_change(row['Change in OI'])
            
            opportunity = {
                'Type': 'PUT',
                'Strike': strike,
                'Entry': entry_price,
                'Target': target_price,
                'Stop_Loss': stop_loss,
                'Current_Price': current_price,
                'Risk_Reward': risk_reward,
                'Max_Loss': max_loss,
                'Max_Gain': max_gain,
                'Volume': row['No. of contracts'],
                'Volume_Class': volume_class,
                'OI_Change': row['Change in OI'],
                'OI_Class': oi_change_class,
                'Rationale': generate_rationale('PUT', row, volume_class, oi_change_class)
            }
            opportunities.append(opportunity)
    
    return opportunities

def classify_volume(volume, all_volumes):
    """Classify volume as High/Medium/Low"""
    if pd.isna(volume):
        return 'Low'
    
    q75 = all_volumes.quantile(0.75)
    q25 = all_volumes.quantile(0.25)
    
    if volume >= q75:
        return 'High'
    elif volume >= q25:
        return 'Medium'
    else:
        return 'Low'

def classify_oi_change(oi_change):
    """Classify OI change"""
    if pd.isna(oi_change):
        return 'Neutral'
    
    if oi_change > 500000:  # 5 lakh
        return 'Very High'
    elif oi_change > 100000:  # 1 lakh
        return 'High'
    elif oi_change > 0:
        return 'Positive'
    else:
        return 'Negative'

def generate_rationale(option_type, row, volume_class, oi_change_class):
    """Generate trading rationale"""
    rationale = f"{option_type} option with {volume_class.lower()} volume"
    
    if oi_change_class in ['Very High', 'High']:
        rationale += f" and {oi_change_class.lower()} OI buildup"
    
    if option_type == 'CALL':
        rationale += ". Good for bullish recovery plays"
    else:
        rationale += ". Suitable for bearish continuation or hedging"
    
    return rationale

def create_price_chart(data, option_type):
    """Create price movement chart"""
    if data is None or data.empty:
        return None
    
    # Group by date and strike price for multi-day data
    chart_data = data.groupby(['Date', 'Strike Price']).agg({
        'Close': 'last',
        'No. of contracts': 'sum',
        'Change in OI': 'sum'
    }).reset_index()
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{option_type} Option Prices', 'Volume & OI Change'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    for strike in chart_data['Strike Price'].unique()[:10]:  # Top 10 strikes
        strike_data = chart_data[chart_data['Strike Price'] == strike]
        fig.add_trace(
            go.Scatter(
                x=strike_data['Date'],
                y=strike_data['Close'],
                name=f'{strike} {option_type}',
                mode='lines+markers'
            ),
            row=1, col=1
        )
    
    # Volume bar chart
    volume_data = chart_data.groupby('Date')['No. of contracts'].sum().reset_index()
    fig.add_trace(
        go.Bar(
            x=volume_data['Date'],
            y=volume_data['No. of contracts'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title=f'{option_type} Options Analysis',
        showlegend=True
    )
    
    return fig

def create_oi_analysis_chart(ce_data, pe_data):
    """Create OI analysis chart"""
    fig = go.Figure()
    
    if ce_data is not None:
        ce_latest = ce_data[ce_data['Date'] == ce_data['Date'].max()]
        fig.add_trace(go.Bar(
            x=ce_latest['Strike Price'],
            y=ce_latest['Open Int'],
            name='CE Open Interest',
            marker_color='green',
            opacity=0.7
        ))
    
    if pe_data is not None:
        pe_latest = pe_data[pe_data['Date'] == pe_data['Date'].max()]
        fig.add_trace(go.Bar(
            x=pe_latest['Strike Price'],
            y=-pe_latest['Open Int'],  # Negative for put side
            name='PE Open Interest',
            marker_color='red',
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Open Interest Distribution',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        height=500
    )
    
    return fig

# Main Streamlit App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 NIFTY Options Trading Analyzer</h1>
        <p>Advanced Options Analysis with Entry, Target & Stop Loss Recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Data Files")
        
        ce_file = st.file_uploader(
            "Upload CE (Call) Options Data",
            type=['csv'],
            help="Upload CSV file containing call options data"
        )
        
        pe_file = st.file_uploader(
            "Upload PE (Put) Options Data", 
            type=['csv'],
            help="Upload CSV file containing put options data"
        )
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        risk_appetite = st.selectbox(
            "Risk Appetite",
            ["Conservative", "Moderate", "Aggressive"],
            index=1
        )
        
        min_volume = st.number_input(
            "Minimum Volume Threshold",
            min_value=0,
            value=10000,
            help="Filter options with minimum volume"
        )
        
        show_charts = st.checkbox("Show Charts", value=True)
    
    # Process uploaded files
    ce_data = None
    pe_data = None
    
    if ce_file is not None:
        ce_data = load_and_process_data(ce_file)
        if ce_data is not None:
            st.success(f"✅ CE data loaded: {len(ce_data)} records")
    
    if pe_file is not None:
        pe_data = load_and_process_data(pe_file)
        if pe_data is not None:
            st.success(f"✅ PE data loaded: {len(pe_data)} records")
    
    # Main analysis
    if ce_data is not None or pe_data is not None:
        
        # Market Summary
        st.header("📊 Market Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate market metrics
        if ce_data is not None:
            latest_underlying = ce_data['Underlying Value'].iloc[-1]
            prev_underlying = ce_data['Underlying Value'].iloc[0] if len(ce_data) > 1 else latest_underlying
        else:
            latest_underlying = pe_data['Underlying Value'].iloc[-1]
            prev_underlying = pe_data['Underlying Value'].iloc[0] if len(pe_data) > 1 else latest_underlying
        
        daily_change = latest_underlying - prev_underlying
        daily_change_pct = (daily_change / prev_underlying) * 100 if prev_underlying != 0 else 0
        
        with col1:
            st.metric("NIFTY Level", f"{latest_underlying:.2f}")
        
        with col2:
            st.metric("Daily Change", f"{daily_change:.2f}", f"{daily_change_pct:.2f}%")
        
        with col3:
            total_ce_volume = ce_data['No. of contracts'].sum() if ce_data is not None else 0
            total_pe_volume = pe_data['No. of contracts'].sum() if pe_data is not None else 0
            st.metric("Total Volume", f"{(total_ce_volume + total_pe_volume):,.0f}")
        
        with col4:
            # PCR calculation
            if ce_data is not None and pe_data is not None:
                ce_oi = ce_data['Open Int'].sum()
                pe_oi = pe_data['Open Int'].sum()
                pcr = pe_oi / ce_oi if ce_oi != 0 else 0
                st.metric("Put-Call Ratio", f"{pcr:.2f}")
        
        # Trading Opportunities
        st.header("🎯 Trading Opportunities")
        
        opportunities = calculate_trading_opportunities(ce_data, pe_data)
        
        if opportunities:
            # Filter by volume threshold
            opportunities = [opp for opp in opportunities if opp['Volume'] >= min_volume]
            
            # Sort by risk-reward ratio
            opportunities = sorted(opportunities, key=lambda x: x.get('Risk_Reward', 0), reverse=True)
            
            # Display opportunities
            tab1, tab2, tab3 = st.tabs(["🚀 Best Opportunities", "📈 Call Options", "📉 Put Options"])
            
            with tab1:
                st.subheader("Top Trading Opportunities")
                
                for i, opp in enumerate(opportunities[:6]):  # Top 6 opportunities
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                        
                        with col1:
                            option_class = "call-option" if opp['Type'] == 'CALL' else "put-option"
                            st.markdown(f"""
                            <div class="opportunity-card {option_class}">
                                <h4>NIFTY {opp['Strike']:.0f} {opp['Type']}</h4>
                                <p><strong>Volume:</strong> {opp['Volume']:,.0f} ({opp['Volume_Class']})</p>
                                <p><strong>OI Change:</strong> {opp['OI_Change']:,.0f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Entry", f"₹{opp['Entry']:.1f}")
                            st.metric("Current", f"₹{opp['Current_Price']:.1f}")
                        
                        with col3:
                            st.metric("Target", f"₹{opp['Target']:.1f}")
                            st.metric("Stop Loss", f"₹{opp['Stop_Loss']:.1f}")
                        
                        with col4:
                            risk_color = "risk-low" if opp['Risk_Reward'] > 2 else "risk-medium" if opp['Risk_Reward'] > 1.5 else "risk-high"
                            st.markdown(f"<p class='{risk_color}'>Risk:Reward = 1:{opp['Risk_Reward']:.1f}</p>", unsafe_allow_html=True)
                            st.write(f"**Max Loss:** ₹{opp['Max_Loss']:.1f}")
                            st.write(f"**Max Gain:** ₹{opp['Max_Gain']:.1f}")
                            
                        st.markdown("---")
            
            with tab2:
                call_opps = [opp for opp in opportunities if opp['Type'] == 'CALL']
                if call_opps:
                    for opp in call_opps[:8]:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.write(f"**{opp['Strike']:.0f} CE**")
                        col2.write(f"₹{opp['Entry']:.1f}")
                        col3.write(f"₹{opp['Target']:.1f}")
                        col4.write(f"₹{opp['Stop_Loss']:.1f}")
                        col5.write(f"1:{opp['Risk_Reward']:.1f}")
                else:
                    st.info("No call options data available")
            
            with tab3:
                put_opps = [opp for opp in opportunities if opp['Type'] == 'PUT']
                if put_opps:
                    for opp in put_opps[:8]:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.write(f"**{opp['Strike']:.0f} PE**")
                        col2.write(f"₹{opp['Entry']:.1f}")
                        col3.write(f"₹{opp['Target']:.1f}")
                        col4.write(f"₹{opp['Stop_Loss']:.1f}")
                        col5.write(f"1:{opp['Risk_Reward']:.1f}")
                else:
                    st.info("No put options data available")
        
        # Charts
        if show_charts:
            st.header("📈 Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if ce_data is not None:
                    ce_chart = create_price_chart(ce_data, "CALL")
                    if ce_chart:
                        st.plotly_chart(ce_chart, use_container_width=True)
            
            with col2:
                if pe_data is not None:
                    pe_chart = create_price_chart(pe_data, "PUT")
                    if pe_chart:
                        st.plotly_chart(pe_chart, use_container_width=True)
            
            # OI Analysis
            if ce_data is not None or pe_data is not None:
                oi_chart = create_oi_analysis_chart(ce_data, pe_data)
                st.plotly_chart(oi_chart, use_container_width=True)
        
        # Risk Management Section
        st.header("⚠️ Risk Management Guidelines")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **Position Sizing**
            - Risk only 1-2% of capital per trade
            - Diversify across multiple strikes
            - Don't put all capital in one expiry
            """)
        
        with col2:
            st.warning("""
            **Time Decay (Theta)**
            - Monitor theta acceleration
            - Close positions 1-2 days before expiry
            - Avoid holding on expiry day
            """)
        
        with col3:
            st.error("""
            **Stop Loss Rules**
            - Always set stop loss before entry
            - Stick to predefined levels
            - Don't average down on losses
            """)
    
    else:
        # Welcome message when no data is uploaded
        st.info("""
        👆 **Please upload your options data files using the sidebar**
        
        **Expected CSV format:**
        - Symbol, Date, Expiry, Option type, Strike Price, Open, High, Low, Close, LTP, Settle Price
        - No. of contracts, Turnover, Premium Turnover, Open Int, Change in OI, Underlying Value
        
        **Features:**
        - 🎯 Automated entry, target, and stop-loss calculations
        - 📊 Risk-reward analysis
        - 📈 Volume and OI-based opportunity identification
        - 🔍 Interactive charts and visualizations
        - ⚙️ Customizable risk parameters
        """)
        
        # Sample data format
        st.subheader("📋 Sample Data Format")
        sample_df = pd.DataFrame({
            'Symbol': ['NIFTY', 'NIFTY'],
            'Date': ['07-Aug-2025', '07-Aug-2025'],
            'Expiry': ['14-Aug-2025', '14-Aug-2025'],
            'Option type': ['CE', 'PE'],
            'Strike Price': [24500, 24500],
            'Open': [170, 150],
            'High': [246, 222],
            'Low': [88, 65],
            'Close': [211, 181],
            'LTP': [238, 189],
            'No. of contracts': [657973, 594365],
            'Open Int': [2561700, 4626675],
            'Change in OI': [1871400, 3001950],
            'Underlying Value': [24596.15, 24596.15]
        })
        st.dataframe(sample_df)

if __name__ == "__main__":
    main()
