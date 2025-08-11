import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nifty Options Analysis", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OptionsAnalyzer:
    def __init__(self):
        self.current_time = datetime.now().time()
        self.market_hours = (time(9, 15), time(15, 30))
        
    def load_and_clean_data(self, uploaded_file):
        """Load and clean options chain data with error handling"""
        try:
            # Read CSV with flexible parsing
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            
            # Handle different CSV formats
            if df.shape[1] < 20:  # If columns are merged
                df = pd.read_csv(uploaded_file, sep=',', header=0)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Extract relevant columns (handle different formats)
            required_cols = ['STRIKE', 'CALLS', 'PUTS']
            
            # Try to identify strike column
            strike_col = None
            for col in df.columns:
                if 'STRIKE' in col.upper() or any(x in str(col).upper() for x in ['STRIKE', 'PRICE']):
                    strike_col = col
                    break
            
            if strike_col is None:
                # Try to find numeric column that looks like strikes
                for col in df.columns:
                    try:
                        values = pd.to_numeric(df[col], errors='coerce')
                        if values.notna().sum() > 0 and values.min() > 20000 and values.max() < 30000:
                            strike_col = col
                            break
                    except:
                        continue
            
            # Parse the data based on structure
            calls_data, puts_data = self.parse_options_data(df, strike_col)
            
            return calls_data, puts_data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None
    
    def parse_options_data(self, df, strike_col):
        """Parse options data from different CSV formats"""
        calls_data = []
        puts_data = []
        
        try:
            # Method 1: If data is in separate CALLS/PUTS sections
            if 'CALLS' in df.columns and 'PUTS' in df.columns:
                # Process row by row
                for idx, row in df.iterrows():
                    try:
                        strike = pd.to_numeric(row[strike_col] if strike_col else row.iloc[11], errors='coerce')
                        if pd.isna(strike):
                            continue
                            
                        # Extract call data (left side)
                        call_oi = pd.to_numeric(str(row.iloc[0]).replace(',', ''), errors='coerce') or 0
                        call_chng_oi = pd.to_numeric(str(row.iloc[1]).replace(',', ''), errors='coerce') or 0
                        call_volume = pd.to_numeric(str(row.iloc[2]).replace(',', ''), errors='coerce') or 0
                        call_iv = pd.to_numeric(row.iloc[3], errors='coerce') or 0
                        call_ltp = pd.to_numeric(str(row.iloc[4]).replace(',', ''), errors='coerce') or 0
                        call_chng = pd.to_numeric(row.iloc[5], errors='coerce') or 0
                        
                        # Extract put data (right side)
                        put_oi = pd.to_numeric(str(row.iloc[-2]).replace(',', ''), errors='coerce') or 0
                        put_chng_oi = pd.to_numeric(str(row.iloc[-3]).replace(',', ''), errors='coerce') or 0
                        put_volume = pd.to_numeric(str(row.iloc[-4]).replace(',', ''), errors='coerce') or 0
                        put_iv = pd.to_numeric(row.iloc[-5], errors='coerce') or 0
                        put_ltp = pd.to_numeric(str(row.iloc[-6]).replace(',', ''), errors='coerce') or 0
                        put_chng = pd.to_numeric(row.iloc[-7], errors='coerce') or 0
                        
                        calls_data.append({
                            'strike': strike, 'oi': call_oi, 'chng_oi': call_chng_oi,
                            'volume': call_volume, 'iv': call_iv, 'ltp': call_ltp, 'chng': call_chng
                        })
                        
                        puts_data.append({
                            'strike': strike, 'oi': put_oi, 'chng_oi': put_chng_oi,
                            'volume': put_volume, 'iv': put_iv, 'ltp': put_ltp, 'chng': put_chng
                        })
                        
                    except Exception as e:
                        continue
            
            # Convert to DataFrames
            calls_df = pd.DataFrame(calls_data).dropna(subset=['strike'])
            puts_df = pd.DataFrame(puts_data).dropna(subset=['strike'])
            
            # Remove rows with zero LTP (likely empty)
            calls_df = calls_df[calls_df['ltp'] > 0]
            puts_df = puts_df[puts_df['ltp'] > 0]
            
            return calls_df, puts_df
            
        except Exception as e:
            st.error(f"Error parsing options data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_probabilities(self, df, current_spot, option_type='call'):
        """Calculate probability of profit using Black-Scholes approximation"""
        probabilities = []
        
        for _, row in df.iterrows():
            try:
                strike = row['strike']
                iv = row['iv'] / 100 if row['iv'] > 1 else row['iv']
                
                # Simple probability calculation based on moneyness and IV
                if option_type == 'call':
                    moneyness = current_spot / strike
                    prob = max(0, min(100, 100 * (1 - np.exp(-2 * max(0, moneyness - 1) / (iv + 0.01)))))
                else:
                    moneyness = strike / current_spot
                    prob = max(0, min(100, 100 * (1 - np.exp(-2 * max(0, moneyness - 1) / (iv + 0.01)))))
                
                probabilities.append(prob)
            except:
                probabilities.append(0)
        
        return probabilities
    
    def get_entry_exit_signals(self, calls_df, puts_df, current_spot):
        """Generate precise entry, target, and stop-loss recommendations"""
        recommendations = []
        
        # Market time check
        is_market_open = self.market_hours[0] <= self.current_time <= self.market_hours[1]
        market_phase = self.get_market_phase()
        
        # ATM and nearby strikes analysis
        atm_strike = round(current_spot / 50) * 50
        
        # Analyze calls
        for _, row in calls_df.iterrows():
            if abs(row['strike'] - current_spot) <= 300:  # Within 300 points
                signal = self.analyze_option_signal(row, current_spot, 'CALL', market_phase)
                if signal['recommendation'] != 'AVOID':
                    recommendations.append(signal)
        
        # Analyze puts
        for _, row in puts_df.iterrows():
            if abs(row['strike'] - current_spot) <= 300:  # Within 300 points
                signal = self.analyze_option_signal(row, current_spot, 'PUT', market_phase)
                if signal['recommendation'] != 'AVOID':
                    recommendations.append(signal)
        
        # Sort by score (higher is better)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def analyze_option_signal(self, row, current_spot, option_type, market_phase):
        """Detailed analysis for individual option"""
        strike = row['strike']
        ltp = row['ltp']
        volume = row['volume']
        oi = row['oi']
        chng_oi = row.get('chng_oi', 0)
        iv = row['iv']
        
        # Calculate moneyness
        if option_type == 'CALL':
            moneyness = (current_spot - strike) / current_spot * 100
        else:
            moneyness = (strike - current_spot) / current_spot * 100
        
        # Scoring factors
        volume_score = min(100, (volume / 100000) * 20)  # Volume in lakhs
        oi_score = min(100, (oi / 50000) * 20)
        iv_score = max(0, 100 - abs(iv - 15) * 5)  # Optimal IV around 15%
        
        # OI change analysis
        oi_signal = 'NEUTRAL'
        if chng_oi > 10000:
            oi_signal = 'BULLISH' if option_type == 'CALL' else 'BEARISH'
        elif chng_oi < -10000:
            oi_signal = 'BEARISH' if option_type == 'CALL' else 'BULLISH'
        
        # Entry conditions
        entry_conditions = []
        
        # Volume condition
        if volume > 50000:
            entry_conditions.append("High Volume")
            
        # OI condition
        if oi > 25000:
            entry_conditions.append("Good OI")
            
        # IV condition
        if 10 <= iv <= 20:
            entry_conditions.append("Optimal IV")
            
        # Time-based conditions
        if market_phase == 'OPENING':
            if volume > 100000:
                entry_conditions.append("Opening Momentum")
        elif market_phase == 'CLOSING':
            if abs(moneyness) < 2:  # Near ATM
                entry_conditions.append("Closing Play")
        
        # Calculate targets and stop loss
        if option_type == 'CALL':
            if moneyness > 2:  # ITM
                target_1 = ltp * 1.3
                target_2 = ltp * 1.6
                stop_loss = ltp * 0.75
            elif abs(moneyness) <= 2:  # ATM
                target_1 = ltp * 1.5
                target_2 = ltp * 2.0
                stop_loss = ltp * 0.7
            else:  # OTM
                target_1 = ltp * 1.8
                target_2 = ltp * 3.0
                stop_loss = ltp * 0.6
        else:  # PUT
            if moneyness > 2:  # ITM
                target_1 = ltp * 1.3
                target_2 = ltp * 1.6
                stop_loss = ltp * 0.75
            elif abs(moneyness) <= 2:  # ATM
                target_1 = ltp * 1.5
                target_2 = ltp * 2.0
                stop_loss = ltp * 0.7
            else:  # OTM
                target_1 = ltp * 1.8
                target_2 = ltp * 3.0
                stop_loss = ltp * 0.6
        
        # Overall score
        total_score = (volume_score + oi_score + iv_score) / 3
        
        # Recommendation logic
        if len(entry_conditions) >= 2 and total_score > 50:
            if total_score > 80:
                recommendation = 'STRONG BUY'
            elif total_score > 65:
                recommendation = 'BUY'
            else:
                recommendation = 'WEAK BUY'
        else:
            recommendation = 'AVOID'
        
        # Calculate probability
        if option_type == 'CALL':
            probability = max(5, min(95, 50 + moneyness * 2 - (abs(moneyness) * 0.5)))
        else:
            probability = max(5, min(95, 50 - moneyness * 2 - (abs(moneyness) * 0.5)))
        
        return {
            'option_type': option_type,
            'strike': strike,
            'ltp': ltp,
            'volume': volume,
            'oi': oi,
            'iv': iv,
            'moneyness': moneyness,
            'recommendation': recommendation,
            'probability': round(probability, 1),
            'target_1': round(target_1, 2),
            'target_2': round(target_2, 2),
            'stop_loss': round(stop_loss, 2),
            'entry_conditions': entry_conditions,
            'oi_signal': oi_signal,
            'score': total_score,
            'risk_level': 'LOW' if abs(moneyness) < 1 else 'MEDIUM' if abs(moneyness) < 3 else 'HIGH'
        }
    
    def get_market_phase(self):
        """Determine current market phase"""
        current = self.current_time
        if time(9, 15) <= current <= time(10, 0):
            return 'OPENING'
        elif time(14, 30) <= current <= time(15, 30):
            return 'CLOSING'
        else:
            return 'MID_SESSION'
    
    def create_visualizations(self, calls_df, puts_df, current_spot):
        """Create comprehensive visualization plots"""
        
        # 1. CE PE LTP Plot
        fig_ltp = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Call Options (CE) - LTP', 'Put Options (PE) - LTP'),
            vertical_spacing=0.08
        )
        
        fig_ltp.add_trace(
            go.Scatter(x=calls_df['strike'], y=calls_df['ltp'], 
                      mode='lines+markers', name='CE LTP', 
                      line=dict(color='green', width=3)), row=1, col=1
        )
        
        fig_ltp.add_trace(
            go.Scatter(x=puts_df['strike'], y=puts_df['ltp'], 
                      mode='lines+markers', name='PE LTP', 
                      line=dict(color='red', width=3)), row=2, col=1
        )
        
        # Add current spot line
        fig_ltp.add_vline(x=current_spot, line_dash="dash", 
                         line_color="blue", annotation_text="Current Spot")
        
        fig_ltp.update_layout(height=600, title="Options Premium Analysis")
        
        # 2. Open Interest Analysis
        fig_oi = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Call OI', 'Put OI', 'Call OI Change', 'Put OI Change'),
            vertical_spacing=0.1, horizontal_spacing=0.1
        )
        
        fig_oi.add_trace(
            go.Bar(x=calls_df['strike'], y=calls_df['oi'], 
                  name='Call OI', marker_color='lightgreen'), row=1, col=1
        )
        
        fig_oi.add_trace(
            go.Bar(x=puts_df['strike'], y=puts_df['oi'], 
                  name='Put OI', marker_color='lightcoral'), row=1, col=2
        )
        
        fig_oi.add_trace(
            go.Bar(x=calls_df['strike'], y=calls_df['chng_oi'], 
                  name='Call OI Change', 
                  marker_color=['green' if x > 0 else 'red' for x in calls_df['chng_oi']]), 
            row=2, col=1
        )
        
        fig_oi.add_trace(
            go.Bar(x=puts_df['strike'], y=puts_df['chng_oi'], 
                  name='Put OI Change',
                  marker_color=['green' if x > 0 else 'red' for x in puts_df['chng_oi']]), 
            row=2, col=2
        )
        
        fig_oi.update_layout(height=700, title="Open Interest Analysis")
        
        # 3. Volume Analysis
        fig_vol = go.Figure()
        
        fig_vol.add_trace(
            go.Bar(x=calls_df['strike'], y=calls_df['volume'], 
                  name='Call Volume', marker_color='rgba(0,255,0,0.7)')
        )
        
        fig_vol.add_trace(
            go.Bar(x=puts_df['strike'], y=puts_df['volume'], 
                  name='Put Volume', marker_color='rgba(255,0,0,0.7)')
        )
        
        fig_vol.add_vline(x=current_spot, line_dash="dash", 
                         line_color="blue", annotation_text="Current Spot")
        
        fig_vol.update_layout(
            title="Volume Analysis", 
            xaxis_title="Strike Price", 
            yaxis_title="Volume",
            height=400
        )
        
        # 4. PCR Analysis
        total_call_oi = calls_df['oi'].sum()
        total_put_oi = puts_df['oi'].sum()
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        total_call_vol = calls_df['volume'].sum()
        total_put_vol = puts_df['volume'].sum()
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        return fig_ltp, fig_oi, fig_vol, pcr_oi, pcr_vol

# Main Streamlit App
def main():
    st.title("🚀 Nifty Options Analysis & Trading Signals")
    st.markdown("---")
    
    # Initialize analyzer
    analyzer = OptionsAnalyzer()
    
    # Sidebar
    st.sidebar.header("📊 Data Input")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Options Chain CSV", 
        type=['csv'],
        help="Upload the latest Nifty options chain data"
    )
    
    # Manual current spot input
    current_spot = st.sidebar.number_input(
        "Current Nifty Spot Price", 
        min_value=20000, max_value=30000, 
        value=24526, step=1,
        help="Enter current Nifty spot price"
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading and analyzing data..."):
            calls_df, puts_df = analyzer.load_and_clean_data(uploaded_file)
        
        if calls_df is not None and puts_df is not None and not calls_df.empty:
            
            # Market Overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Nifty", f"{current_spot:,.0f}")
            
            with col2:
                market_phase = analyzer.get_market_phase()
                st.metric("Market Phase", market_phase)
            
            with col3:
                total_call_vol = calls_df['volume'].sum()
                st.metric("Total Call Volume", f"{total_call_vol:,.0f}")
            
            with col4:
                total_put_vol = puts_df['volume'].sum()
                st.metric("Total Put Volume", f"{total_put_vol:,.0f}")
            
            st.markdown("---")
            
            # Generate recommendations
            with st.spinner("Generating trading signals..."):
                recommendations = analyzer.get_entry_exit_signals(calls_df, puts_df, current_spot)
            
            # Display recommendations
            st.header("🎯 Trading Recommendations")
            
            if recommendations:
                for i, rec in enumerate(recommendations[:5]):  # Top 5
                    
                    # Color coding based on recommendation
                    if rec['recommendation'] == 'STRONG BUY':
                        color = "green"
                        emoji = "🟢"
                    elif rec['recommendation'] == 'BUY':
                        color = "blue"
                        emoji = "🔵"
                    elif rec['recommendation'] == 'WEAK BUY':
                        color = "orange"
                        emoji = "🟡"
                    else:
                        color = "gray"
                        emoji = "⚪"
                    
                    with st.expander(f"{emoji} {rec['option_type']} {rec['strike']} @ ₹{rec['ltp']} - {rec['recommendation']}", expanded=(i==0)):
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**📈 Entry Details**")
                            st.write(f"**Entry Price:** ₹{rec['ltp']}")
                            st.write(f"**Probability:** {rec['probability']}%")
                            st.write(f"**Risk Level:** {rec['risk_level']}")
                            st.write(f"**Moneyness:** {rec['moneyness']:.1f}%")
                        
                        with col2:
                            st.markdown("**🎯 Targets & Stop Loss**")
                            st.write(f"**Target 1:** ₹{rec['target_1']}")
                            st.write(f"**Target 2:** ₹{rec['target_2']}")
                            st.write(f"**Stop Loss:** ₹{rec['stop_loss']}")
                            
                            profit_1 = (rec['target_1'] - rec['ltp']) / rec['ltp'] * 100
                            st.write(f"**Profit Potential:** {profit_1:.0f}%")
                        
                        with col3:
                            st.markdown("**📊 Market Data**")
                            st.write(f"**Volume:** {rec['volume']:,.0f}")
                            st.write(f"**Open Interest:** {rec['oi']:,.0f}")
                            st.write(f"**IV:** {rec['iv']:.1f}%")
                            st.write(f"**OI Signal:** {rec['oi_signal']}")
                        
                        # Entry conditions
                        if rec['entry_conditions']:
                            st.markdown("**✅ Entry Conditions Met:**")
                            for condition in rec['entry_conditions']:
                                st.write(f"• {condition}")
                        
                        # Strategy
                        st.markdown("**📋 Strategy:**")
                        if rec['option_type'] == 'CALL':
                            strategy_text = f"Buy {rec['option_type']} if Nifty shows upward momentum. "
                        else:
                            strategy_text = f"Buy {rec['option_type']} if Nifty shows downward pressure. "
                        
                        strategy_text += f"Target profit of {profit_1:.0f}% with strict stop loss at {rec['stop_loss']}."
                        st.write(strategy_text)
            
            st.markdown("---")
            
            # Visualizations
            st.header("📊 Market Analysis Charts")
            
            with st.spinner("Creating visualizations..."):
                fig_ltp, fig_oi, fig_vol, pcr_oi, pcr_vol = analyzer.create_visualizations(calls_df, puts_df, current_spot)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PCR (OI)", f"{pcr_oi:.2f}", 
                         help="Put-Call Ratio based on Open Interest")
            with col2:
                st.metric("PCR (Volume)", f"{pcr_vol:.2f}", 
                         help="Put-Call Ratio based on Volume")
            with col3:
                sentiment = "Bullish" if pcr_oi < 1.0 else "Bearish"
                st.metric("Market Sentiment", sentiment)
            
            # Display charts
            st.plotly_chart(fig_ltp, use_container_width=True)
            st.plotly_chart(fig_oi, use_container_width=True)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Data tables
            st.header("📋 Raw Data")
            
            tab1, tab2 = st.tabs(["Call Options", "Put Options"])
            
            with tab1:
                st.dataframe(calls_df.style.format({
                    'ltp': '₹{:.2f}',
                    'volume': '{:,.0f}',
                    'oi': '{:,.0f}',
                    'iv': '{:.2f}%'
                }), use_container_width=True)
            
            with tab2:
                st.dataframe(puts_df.style.format({
                    'ltp': '₹{:.2f}',
                    'volume': '{:,.0f}',
                    'oi': '{:,.0f}',
                    'iv': '{:.2f}%'
                }), use_container_width=True)
        
        else:
            st.error("Could not parse the uploaded file. Please check the format.")
    
    else:
        st.info("👆 Please upload an options chain CSV file to begin analysis.")
        
        # Sample data format info
        st.markdown("### Expected CSV Format:")
        st.markdown("""
        The CSV should contain options chain data with columns for:
        - Strike prices
        - Call and Put LTP (Last Traded Price)
        - Volume data
        - Open Interest (OI)
        - Implied Volatility (IV)
        - Change in OI
        """)
    
    # Risk disclaimer
    st.markdown("---")
    st.markdown("### ⚠️ Risk Disclaimer")
    st.markdown("""
    - **High Risk:** Options trading involves substantial risk and may not be suitable for all investors
    - **No Guarantee:** Past performance does not guarantee future results
    - **Use Stop Loss:** Always use proper risk management and position sizing
    - **Market Volatility:** Options prices can change rapidly due to various factors
    - **Consult Advisor:** Consider consulting with a financial advisor before making investment decisions
    """)

if __name__ == "__main__":
    main()
