import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
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
    
    .freshness-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .fresh-data { background-color: #d4edda; color: #155724; }
    .stale-data { background-color: #fff3cd; color: #856404; }
    .old-data { background-color: #f8d7da; color: #721c24; }
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

def get_market_session_info():
    """Get current market session information"""
    now = datetime.now()
    current_time = now.time()
    
    # Market sessions
    pre_market = (current_time >= datetime.strptime("09:00", "%H:%M").time() and 
                 current_time < datetime.strptime("09:15", "%H:%M").time())
    
    morning_session = (current_time >= datetime.strptime("09:15", "%H:%M").time() and 
                      current_time < datetime.strptime("11:30", "%H:%M").time())
    
    midday_session = (current_time >= datetime.strptime("11:30", "%H:%M").time() and 
                     current_time < datetime.strptime("14:30", "%H:%M").time())
    
    afternoon_session = (current_time >= datetime.strptime("14:30", "%H:%M").time() and 
                        current_time < datetime.strptime("15:30", "%H:%M").time())
    
    post_market = current_time >= datetime.strptime("15:30", "%H:%M").time()
    
    if pre_market:
        return "Pre-Market", "🌅", "Wait for market opening"
    elif morning_session:
        return "Morning Session", "🚀", "Best time for fresh analysis"
    elif midday_session:
        return "Midday Session", "☀️", "Monitor for significant moves"
    elif afternoon_session:
        return "Afternoon Session", "🔥", "High volatility - frequent updates needed"
    else:
        return "Post-Market", "🌙", "Review positions, plan tomorrow"

def calculate_data_freshness(upload_time):
    """Calculate how fresh the data is"""
    if upload_time is None:
        return "Unknown", "⚪", 0
    
    now = datetime.now()
    time_diff = now - upload_time
    hours_old = time_diff.total_seconds() / 3600
    
    if hours_old < 1:
        return "Very Fresh", "🟢", hours_old
    elif hours_old < 2:
        return "Fresh", "🟡", hours_old
    elif hours_old < 4:
        return "Moderately Stale", "🟠", hours_old
    else:
        return "Stale", "🔴", hours_old

def calculate_trading_opportunities(ce_data, pe_data, risk_appetite="Moderate"):
    """Calculate trading opportunities based on options data"""
    opportunities = []
    
    # Risk multipliers based on appetite
    risk_multipliers = {
        "Conservative": {"target": 1.3, "sl": 0.9},
        "Moderate": {"target": 1.5, "sl": 0.85},
        "Aggressive": {"target": 2.0, "sl": 0.8}
    }
    
    multiplier = risk_multipliers.get(risk_appetite, risk_multipliers["Moderate"])
    
    if ce_data is not None:
        # Call options analysis
        ce_opportunities = analyze_call_options(ce_data, multiplier)
        opportunities.extend(ce_opportunities)
    
    if pe_data is not None:
        # Put options analysis
        pe_opportunities = analyze_put_options(pe_data, multiplier)
        opportunities.extend(pe_opportunities)
    
    return opportunities

def analyze_call_options(ce_data, multiplier):
    """Analyze call options for trading opportunities"""
    opportunities = []
    
    # Filter for high volume and OI options
    latest_date = ce_data['Date'].max()
    latest_data = ce_data[ce_data['Date'] == latest_date].copy()
    
    # Sort by volume and OI
    latest_data = latest_data.sort_values(['No. of contracts', 'Change in OI'], ascending=False)
    
    # Get top opportunities
    top_calls = latest_data.head(15)
    
    for _, row in top_calls.iterrows():
        if pd.notna(row['Close']) and row['Close'] > 0:
            strike = row['Strike Price']
            current_price = row['Close']
            
            # Calculate entry, target, and stop loss based on risk appetite
            entry_price = current_price * 1.02  # 2% above close
            target_price = current_price * multiplier["target"]
            stop_loss = current_price * multiplier["sl"]
            
            # Risk metrics
            max_loss = entry_price - stop_loss
            max_gain = target_price - entry_price
            risk_reward = max_gain / max_loss if max_loss > 0 else 0
            
            # Volume classification
            volume_class = classify_volume(row['No. of contracts'], latest_data['No. of contracts'])
            oi_change_class = classify_oi_change(row['Change in OI'])
            
            # Calculate probability score
            prob_score = calculate_probability_score(row, volume_class, oi_change_class, "CALL")
            
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
                'Probability_Score': prob_score,
                'Rationale': generate_rationale('CALL', row, volume_class, oi_change_class)
            }
            opportunities.append(opportunity)
    
    return opportunities

def analyze_put_options(pe_data, multiplier):
    """Analyze put options for trading opportunities"""
    opportunities = []
    
    # Filter for high volume and OI options
    latest_date = pe_data['Date'].max()
    latest_data = pe_data[pe_data['Date'] == latest_date].copy()
    
    # Sort by volume and OI
    latest_data = latest_data.sort_values(['No. of contracts', 'Change in OI'], ascending=False)
    
    # Get top opportunities
    top_puts = latest_data.head(15)
    
    for _, row in top_puts.iterrows():
        if pd.notna(row['Close']) and row['Close'] > 0:
            strike = row['Strike Price']
            current_price = row['Close']
            
            # Calculate entry, target, and stop loss based on risk appetite
            entry_price = current_price * 1.02  # 2% above close
            target_price = current_price * (multiplier["target"] + 0.2)  # Slightly higher for puts
            stop_loss = current_price * (multiplier["sl"] - 0.05)  # Slightly tighter for puts
            
            # Risk metrics
            max_loss = entry_price - stop_loss
            max_gain = target_price - entry_price
            risk_reward = max_gain / max_loss if max_loss > 0 else 0
            
            # Volume classification
            volume_class = classify_volume(row['No. of contracts'], latest_data['No. of contracts'])
            oi_change_class = classify_oi_change(row['Change in OI'])
            
            # Calculate probability score
            prob_score = calculate_probability_score(row, volume_class, oi_change_class, "PUT")
            
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
                'Probability_Score': prob_score,
                'Rationale': generate_rationale('PUT', row, volume_class, oi_change_class)
            }
            opportunities.append(opportunity)
    
    return opportunities

def calculate_probability_score(row, volume_class, oi_change_class, option_type):
    """Calculate probability score for the opportunity"""
    score = 0
    
    # Volume score
    if volume_class == 'High':
        score += 30
    elif volume_class == 'Medium':
        score += 20
    else:
        score += 10
    
    # OI change score
    if oi_change_class in ['Very High', 'High']:
        score += 25
    elif oi_change_class == 'Positive':
        score += 15
    else:
        score += 5
    
    # Price action score
    if pd.notna(row['Price Change %']):
        if abs(row['Price Change %']) > 5:  # Strong price movement
            score += 20
        elif abs(row['Price Change %']) > 2:
            score += 15
        else:
            score += 10
    
    # Normalize to 100
    return min(score, 100)

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
    
    if oi_change > 1000000:  # 10 lakh
        return 'Very High'
    elif oi_change > 500000:  # 5 lakh
        return 'High'
    elif oi_change > 100000:  # 1 lakh
        return 'Positive'
    elif oi_change > 0:
        return 'Low Positive'
    else:
        return 'Negative'

def generate_rationale(option_type, row, volume_class, oi_change_class):
    """Generate trading rationale"""
    rationale = f"{option_type} option with {volume_class.lower()} volume"
    
    if oi_change_class in ['Very High', 'High']:
        rationale += f" and {oi_change_class.lower()} OI buildup"
    
    # Add price action context
    if pd.notna(row['Price Change %']):
        if row['Price Change %'] > 5:
            rationale += ". Strong bullish momentum"
        elif row['Price Change %'] < -5:
            rationale += ". Strong bearish momentum"
    
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
    for strike in sorted(chart_data['Strike Price'].unique())[:8]:  # Top 8 strikes
        strike_data = chart_data[chart_data['Strike Price'] == strike]
        fig.add_trace(
            go.Scatter(
                x=strike_data['Date'],
                y=strike_data['Close'],
                name=f'{strike} {option_type}',
                mode='lines+markers',
                line=dict(width=2)
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
            marker_color='lightblue',
            showlegend=False
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
        ce_sorted = ce_latest.nlargest(20, 'Open Int')  # Top 20 strikes
        fig.add_trace(go.Bar(
            x=ce_sorted['Strike Price'],
            y=ce_sorted['Open Int'],
            name='CE Open Interest',
            marker_color='green',
            opacity=0.7
        ))
    
    if pe_data is not None:
        pe_latest = pe_data[pe_data['Date'] == pe_data['Date'].max()]
        pe_sorted = pe_latest.nlargest(20, 'Open Int')  # Top 20 strikes
        fig.add_trace(go.Bar(
            x=pe_sorted['Strike Price'],
            y=-pe_sorted['Open Int'],  # Negative for put side
            name='PE Open Interest',
            marker_color='red',
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Open Interest Distribution (Top 20 Strikes Each)',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        height=500,
        hovermode='x unified'
    )
    
    return fig

# Main Streamlit App
def main():
    # Initialize session state
    if 'upload_time' not in st.session_state:
        st.session_state.upload_time = None
    if 'last_nifty_level' not in st.session_state:
        st.session_state.last_nifty_level = None
    
    # Header with real-time info
    session_info, session_emoji, session_advice = get_market_session_info()
    
    st.markdown(f"""
    <div class="main-header">
        <h1>📈 NIFTY Options Trading Analyzer</h1>
        <p>Advanced Options Analysis with Smart Refresh Alerts</p>
        <p>{session_emoji} {session_info} - {session_advice}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads with enhanced features
    with st.sidebar:
        st.header("📁 Upload Data Files")
        
        # Data freshness indicator
        current_time = datetime.now().strftime("%H:%M:%S")
        st.info(f"🕐 Current Time: {current_time}")
        
        # Upload time tracking and freshness
        if st.session_state.upload_time:
            freshness, freshness_emoji, hours_old = calculate_data_freshness(st.session_state.upload_time)
            st.markdown(f"""
            <div class="freshness-indicator {'fresh-data' if hours_old < 2 else 'stale-data' if hours_old < 4 else 'old-data'}">
                {freshness_emoji} Data Status: {freshness}<br>
                Uploaded: {hours_old:.1f} hours ago
            </div>
            """, unsafe_allow_html=True)
        
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
        
        # Smart refresh recommendations
        if st.session_state.upload_time:
            hours_old = (datetime.now() - st.session_state.upload_time).total_seconds() / 3600
            
            if hours_old > 4:
                st.error("🚨 Data is very stale! Please refresh immediately.")
                if st.button("🔄 I'll Refresh Now"):
                    st.session_state.upload_time = None
                    st.experimental_rerun()
            elif hours_old > 2:
                st.warning("⚠️ Consider refreshing data for better accuracy.")
            
        # Data validity guidelines
        with st.expander("📋 Data Refresh Guidelines"):
            st.write("""
            **🕘 Best Upload Times:**
            - **9:45-10:00 AM**: Post opening volatility
            - **12:30-1:00 PM**: Mid-day review  
            - **2:45-3:00 PM**: Pre-closing analysis
            
            **🔄 Mandatory Refresh When:**
            - NIFTY moves >1% from upload level
            - Major news/events occur
            - Volatility spikes significantly
            - Data becomes >4 hours old
            
            **📊 Optional Refresh:**
            - Every 2-3 hours in normal conditions
            - Before taking new positions
            - After lunch session (2:00 PM)
            """)
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        risk_appetite = st.selectbox(
            "Risk Appetite",
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="Adjusts target and stop-loss levels"
        )
        
        min_volume = st.number_input(
            "Minimum Volume Threshold",
            min_value=0,
            value=10000,
            help="Filter options with minimum volume"
        )
        
        min_probability = st.slider(
            "Minimum Probability Score",
            min_value=0,
            max_value=100,
            value=60,
            help="Filter opportunities by probability score"
        )
        
        show_charts = st.checkbox("Show Charts", value=True)
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
    
    # Process uploaded files
    ce_data = None
    pe_data = None
    
    if ce_file is not None:
        ce_data = load_and_process_data(ce_file)
        if ce_data is not None:
            st.session_state.upload_time = datetime.now()
            st.success(f"✅ CE data loaded: {len(ce_data)} records")
            
            # Data freshness check
            data_date = ce_data['Date'].max()
            if data_date.date() != datetime.now().date():
                st.warning(f"⚠️ Data is from {data_date.strftime('%Y-%m-%d')} - not today's data!")
    
    if pe_file is not None:
        pe_data = load_and_process_data(pe_file)
        if pe_data is not None:
            if 'upload_time' not in st.session_state or st.session_state.upload_time is None:
                st.session_state.upload_time = datetime.now()
            st.success(f"✅ PE data loaded: {len(pe_data)} records")
            
            # Data freshness check
            data_date = pe_data['Date'].max()
            if data_date.date() != datetime.now().date():
                st.warning(f"⚠️ Data is from {data_date.strftime('%Y-%m-%d')} - not today's data!")
    
    # Main analysis
    if ce_data is not None or pe_data is not None:
        
        # Market Summary with change detection
        st.header("📊 Market Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate market metrics
        if ce_data is not None:
            latest_underlying = ce_data['Underlying Value'].iloc[-1]
            prev_underlying = ce_data['Underlying Value'].iloc[0] if len(ce_data) > 1 else latest_underlying
        else:
            latest_underlying = pe_data['Underlying Value'].iloc[-1]
            prev_underlying = pe_data['Underlying Value'].iloc[0] if len(pe_data) > 1 else latest_underlying
        
        # Check for significant moves
        if st.session_state.last_nifty_level:
            level_change = abs(latest_underlying - st.session_state.last_nifty_level)
            level_change_pct = (level_change / st.session_state.last_nifty_level) * 100
            
            if level_change_pct > 1:
                st.error(f"🚨 NIFTY moved {level_change_pct:.1f}% since last analysis! Consider refreshing data.")
        
        st.session_state.last_nifty_level = latest_underlying
        
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
        
        # Trading Opportunities with enhanced filtering
        st.header("🎯 Trading Opportunities")
        
        opportunities = calculate_trading_opportunities(ce_data, pe_data, risk_appetite)
        
        if opportunities:
            # Enhanced filtering
            opportunities = [
                opp for opp in opportunities 
                if opp['Volume'] >= min_volume and opp['Probability_Score'] >= min_probability
            ]
            
            # Sort by probability score and risk-reward
            opportunities = sorted(
                opportunities, 
                key=lambda x: (x.get('Probability_Score', 0), x.get('Risk_Reward', 0)), 
                reverse=True
            )
            
            if opportunities:
                # Display opportunities in enhanced tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🏆 Best Opportunities", 
                    "📈 Call Options", 
                    "📉 Put Options",
                    "📋 All Opportunities"
                ])
                
                with tab1:
                    st.subheader("🎯 Top Probability Opportunities")
                    
                    for i, opp in enumerate(opportunities[:6]):  # Top 6 opportunities
                        with st.container():
                            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
                            
                            with col1:
                                option_class = "call-option" if opp['Type'] == 'CALL' else "put-option"
                                st.markdown(f"""
                                <div class="opportunity-card {option_class}">
                                    <h4>NIFTY {opp['Strike']:.0f} {opp['Type']} 
                                    <span style="float: right; color: {'#27ae60' if opp['Probability_Score'] >= 80 else '#f39c12' if opp['Probability_Score'] >= 60 else '#e74c3c'}">{opp['Probability_Score']:.0f}%</span></h4>
                                    <p><strong>Volume:</strong> {opp['Volume']:,.0f} ({opp['Volume_Class']})</p>
                                    <p><strong>OI Change:</strong> {opp['OI_Change']:,.0f}</p>
                                    <p><small>{opp['Rationale']}</small></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.metric("Entry", f"₹{opp['Entry']:.1f}")
                            
                            with col3:
                                st.metric("Target", f"₹{opp['Target']:.1f}")
                            
                            with col4:
                                st.metric("Stop Loss", f"₹{opp['Stop_Loss']:.1f}")
                            
                            with col5:
                                risk_color = "risk-low" if opp['Risk_Reward'] > 2 else "risk-medium" if opp['Risk_Reward'] > 1.5 else "risk-high"
                                st.markdown(f"<p class='{risk_color}'>Risk:Reward = 1:{opp['Risk_Reward']:.1f}</p>", unsafe_allow_html=True)
                                st.write(f"**Max Loss:** ₹{opp['Max_Loss']:.1f}")
                                st.write(f"**Max Gain:** ₹{opp['Max_Gain']:.1f}")
