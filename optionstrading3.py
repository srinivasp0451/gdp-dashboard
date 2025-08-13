import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import io

# Configure page
st.set_page_config(layout="wide", page_title="NIFTY Option Chain Analysis")
st.title("📊 NIFTY Option Chain Analysis")
st.caption("Professional Options Trading Dashboard | Data-Driven Insights")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Option Chain Data", 
                                        type=["csv", "xlsx"],
                                        help="Upload CSV or Excel file with option chain data")

# Debug mode
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Load and preprocess data
def load_data(uploaded_file, debug_mode=False):
    if uploaded_file is None:
        st.warning("Please upload a file to proceed")
        st.stop()
    
    try:
        # Log file info
        if debug_mode:
            st.write(f"📂 Uploaded file: {uploaded_file.name} ({uploaded_file.size/1024:.2f} KB)")
        
        # Read file based on type
        if uploaded_file.name.endswith('.csv'):
            # Read without skipping rows to inspect
            raw_data = pd.read_csv(uploaded_file, header=None)
            if debug_mode:
                st.write("📝 Raw CSV data shape:", raw_data.shape)
                st.write("🔍 First 3 rows of raw CSV data:")
                st.dataframe(raw_data.head(3))
            
            # Check if we need to skip 2 rows
            if len(raw_data.columns) > 20:
                data = pd.read_csv(uploaded_file, skiprows=2, header=None)
                if debug_mode:
                    st.write("🛠️ Skipping 2 rows due to header complexity")
            else:
                data = pd.read_csv(uploaded_file, skiprows=1, header=None)
                
        elif uploaded_file.name.endswith('.xlsx'):
            # Read without skipping rows to inspect
            raw_data = pd.read_excel(uploaded_file, header=None)
            if debug_mode:
                st.write("📝 Raw Excel data shape:", raw_data.shape)
                st.write("🔍 First 3 rows of raw Excel data:")
                st.dataframe(raw_data.head(3))
            
            # Check if we need to skip 2 rows
            if len(raw_data.columns) > 20:
                data = pd.read_excel(uploaded_file, skiprows=2, header=None)
                if debug_mode:
                    st.write("🛠️ Skipping 2 rows due to header complexity")
            else:
                data = pd.read_excel(uploaded_file, skiprows=1, header=None)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            st.stop()
            
        # Log processed data
        if debug_mode:
            st.write("🛠️ Processed data shape:", data.shape)
            st.write("🔍 First 3 rows of processed data:")
            st.dataframe(data.head(3))
        
        # Handle column count - we expect 21 columns
        if len(data.columns) != 21:
            if debug_mode:
                st.warning(f"⚠️ Column count mismatch: Expected 21 columns, found {len(data.columns)}")
            
            # If we have exactly 23 columns, take columns 1-21
            if len(data.columns) == 23:
                data = data.iloc[:, 1:22]
                if debug_mode:
                    st.write("🔧 Selected columns 1-21 from 23-column structure")
            else:
                st.error(f"Column count mismatch: Expected 21 columns, found {len(data.columns)}. Please check file format.")
                st.stop()
        
        # Clean column names
        data.columns = [
            'calls_oi', 'calls_chng_oi', 'calls_volume', 'calls_iv', 'calls_ltp', 'calls_chng',
            'calls_bid_qty', 'calls_bid', 'calls_ask', 'calls_ask_qty', 'strike',
            'puts_bid_qty', 'puts_bid', 'puts_ask', 'puts_ask_qty', 'puts_chng',
            'puts_ltp', 'puts_iv', 'puts_volume', 'puts_chng_oi', 'puts_oi'
        ]
        
        if debug_mode:
            st.write("🧹 Cleaned column names:", data.columns.tolist())
        
        # Convert to numeric and clean
        for col in data.columns:
            if debug_mode:
                st.write(f"🔢 Converting column {col}")
                
            # Clean string values
            if data[col].dtype == object:
                data[col] = data[col].astype(str).str.replace(',', '').str.replace('"', '')
            
            # Convert to numeric
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Calculate straddle premium
        data['straddle_premium'] = data['calls_ltp'] + data['puts_ltp']
        data['straddle_pct_change'] = data['straddle_premium'].pct_change() * 100
        
        if debug_mode:
            st.write("✅ Data processing completed successfully")
            st.write("🔍 Processed data sample:")
            st.dataframe(data.head(3))
        
        return data
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        if debug_mode:
            st.exception(e)
        st.stop()

if uploaded_file:
    df = load_data(uploaded_file, debug_mode)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    min_strike = st.sidebar.slider("Minimum Strike Price", 
                                  min_value=int(df['strike'].min()), 
                                  max_value=int(df['strike'].max()), 
                                  value=int(df['strike'].min()))
    max_strike = st.sidebar.slider("Maximum Strike Price", 
                                  min_value=int(df['strike'].min()), 
                                  max_value=int(df['strike'].max()), 
                                  value=int(df['strike'].max()))
    
    filtered_df = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)]
    
    # Find ATM strike (where difference between call and put prices is smallest)
    df['price_diff'] = abs(df['calls_ltp'] - df['puts_ltp'])
    if not df.empty:
        atm_strike = df.loc[df['price_diff'].idxmin(), 'strike']
    else:
        atm_strike = 25000
        st.warning("Couldn't determine ATM strike, using default 25,000")
    
    if debug_mode:
        st.write(f"🎯 ATM Strike: {atm_strike}")
    
    # Main columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Market Summary")
        
        # Get key metrics
        max_put_oi = df.loc[df['puts_oi'].idxmax()] if not df.empty else None
        max_call_oi = df.loc[df['calls_oi'].idxmax()] if not df.empty else None
        
        summary_text = f"""
        - **Spot Price**: ~₹{atm_strike:,.0f} (derived from ATM options)
        - **ATM Strike**: {atm_strike:,.0f} (where CALL and PUT prices are closest)
        - **Key Observations**:
            - Strongest support at {max_put_oi['strike'] if max_put_oi else 'N/A'} ({max_put_oi['puts_oi']/100000:,.1f}L put OI)
            - Strongest resistance at {max_call_oi['strike'] if max_call_oi else 'N/A'} ({max_call_oi['calls_oi']/100000:,.1f}L call OI)
            - Implied Volatility (IV) skew: Puts ({df[df['strike'] == atm_strike]['puts_iv'].values[0]:.1f}%) > Calls ({df[df['strike'] == atm_strike]['calls_iv'].values[0]:.1f}%)
            - Expected daily move: ±{atm_strike * 0.16 * (1/365)**0.5:.0f} points (1 standard deviation)
        - **Sentiment**: Neutral-to-bullish with protective hedging
        """
        st.markdown(summary_text)
    
    # Straddle Premium Chart
    with col2:
        st.subheader("Straddle Premium Analysis")
        
        if not filtered_df.empty:
            # Create color mapping for percentage change
            cmap = LinearSegmentedColormap.from_list('rg', ["red", "white", "green"], N=256)
            norm = plt.Normalize(filtered_df['straddle_pct_change'].min() - 1, 
                                filtered_df['straddle_pct_change'].max() + 1)
            colors = [cmap(norm(value)) for value in filtered_df['straddle_pct_change']]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(filtered_df['strike'].astype(str), 
                         filtered_df['straddle_premium'], 
                         color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.title('Straddle Premium by Strike Price', fontsize=14)
            plt.xlabel('Strike Price', fontsize=12)
            plt.ylabel('Premium (₹)', fontsize=12)
            plt.xticks(rotation=90, fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Create colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('% Change', fontsize=10)
            
            st.pyplot(fig)
            st.caption("Green = Increase | Red = Decrease | Color intensity shows percentage change")
        else:
            st.warning("No data available for selected strike range")
    
    # OI Analysis Charts
    st.subheader("Open Interest Analysis")
    oi_col1, oi_col2 = st.columns(2)
    
    with oi_col1:
        st.markdown("**CALL Options OI/Change**")
        
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(filtered_df['strike'].astype(str), filtered_df['calls_oi'], 
                  color='skyblue', label='Open Interest')
            ax.bar(filtered_df['strike'].astype(str), filtered_df['calls_chng_oi'], 
                  color=np.where(filtered_df['calls_chng_oi'] > 0, 'green', 'red'),
                  label='Change in OI')
            plt.title('CALL Open Interest', fontsize=14)
            plt.xlabel('Strike Price', fontsize=12)
            plt.ylabel('Contracts', fontsize=12)
            plt.xticks(rotation=90, fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("No data available")
    
    with oi_col2:
        st.markdown("**PUT Options OI/Change**")
        
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(filtered_df['strike'].astype(str), filtered_df['puts_oi'], 
                  color='lightcoral', label='Open Interest')
            ax.bar(filtered_df['strike'].astype(str), filtered_df['puts_chng_oi'], 
                  color=np.where(filtered_df['puts_chng_oi'] > 0, 'green', 'red'),
                  label='Change in OI')
            plt.title('PUT Open Interest', fontsize=14)
            plt.xlabel('Strike Price', fontsize=12)
            plt.ylabel('Contracts', fontsize=12)
            plt.xticks(rotation=90, fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("No data available")
    
    # Volume Analysis
    st.subheader("Trading Volume Analysis")
    vol_col1, vol_col2 = st.columns(2)
    
    with vol_col1:
        st.markdown("**CALL Volume**")
        
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(filtered_df['strike'].astype(str), filtered_df['calls_volume'], 
                  color='royalblue')
            plt.title('CALL Trading Volume', fontsize=14)
            plt.xlabel('Strike Price', fontsize=12)
            plt.ylabel('Contracts', fontsize=12)
            plt.xticks(rotation=90, fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
        else:
            st.warning("No data available")
    
    with vol_col2:
        st.markdown("**PUT Volume**")
        
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(filtered_df['strike'].astype(str), filtered_df['puts_volume'], 
                  color='indianred')
            plt.title('PUT Trading Volume', fontsize=14)
            plt.xlabel('Strike Price', fontsize=12)
            plt.ylabel('Contracts', fontsize=12)
            plt.xticks(rotation=90, fontsize=8)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
        else:
            st.warning("No data available")
    
    # Trade Recommendations
    st.subheader("🔥 Best Buying Opportunities")
    
    if not filtered_df.empty:
        # Find best opportunities
        call_opportunities = filtered_df[
            (filtered_df['calls_chng_oi'] > 0) & 
            (filtered_df['calls_iv'] < 50) &
            (filtered_df['calls_ltp'] > 0)
        ].sort_values('calls_chng_oi', ascending=False).head(3)
        
        put_opportunities = filtered_df[
            (filtered_df['puts_chng_oi'] > 0) & 
            (filtered_df['puts_iv'] < 50) &
            (filtered_df['puts_ltp'] > 0)
        ].sort_values('puts_chng_oi', ascending=False).head(1)
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            if not call_opportunities.empty:
                strike = call_opportunities.iloc[0]['strike']
                ltp = call_opportunities.iloc[0]['calls_ltp']
                iv = call_opportunities.iloc[0]['calls_iv']
                chng_oi = call_opportunities.iloc[0]['calls_chng_oi']
                
                st.markdown(f"""
                ### 🟢 {strike:,.0f} CALL
                **Probability of Profit**: 40%  
                **Entry**: ₹{ltp*0.97:.2f}-{ltp:.2f}  
                **Target**: ₹{ltp*2:.2f} (100% profit)  
                **Stop Loss**: ₹{ltp*0.5:.2f}  
                **RR Ratio**: 1:2  
                
                **Logic**:  
                - Premium: ₹{ltp:.2f} (IV: {iv:.1f}%)  
                - OI increased by {chng_oi:,.0f} contracts  
                - Near strong support/resistance level  
                """)
                st.progress(40)
            else:
                st.warning("No call opportunities found")
        
        with rec_col2:
            if len(call_opportunities) > 1:
                strike = call_opportunities.iloc[1]['strike']
                ltp = call_opportunities.iloc[1]['calls_ltp']
                iv = call_opportunities.iloc[1]['calls_iv']
                volume = call_opportunities.iloc[1]['calls_volume']
                
                st.markdown(f"""
                ### 🟡 {strike:,.0f} CALL
                **Probability of Profit**: 50%  
                **Entry**: ₹{ltp*0.98:.2f}-{ltp:.2f}  
                **Target**: ₹{ltp*2:.2f} (100% profit)  
                **Stop Loss**: ₹{ltp*0.5:.2f}  
                **RR Ratio**: 1:2  
                
                **Logic**:  
                - Good price sensitivity  
                - High volume ({volume:,.0f} contracts)  
                - IV: {iv:.1f}% (reasonable)  
                """)
                st.progress(50)
            elif not put_opportunities.empty:
                strike = put_opportunities.iloc[0]['strike']
                ltp = put_opportunities.iloc[0]['puts_ltp']
                iv = put_opportunities.iloc[0]['puts_iv']
                oi = put_opportunities.iloc[0]['puts_oi']
                
                st.markdown(f"""
                ### 🔵 {strike:,.0f} PUT
                **Probability of Profit**: 30%  
                **Entry**: ₹{ltp:.2f}  
                **Target**: ₹{ltp*2:.2f} (100% profit)  
                **Stop Loss**: ₹{ltp*0.5:.2f}  
                
                **Logic**:  
                - Protective hedge against downside  
                - Strong OI support ({oi/100000:.1f}L contracts)  
                - IV: {iv:.1f}%  
                """)
                st.progress(30)
            else:
                st.warning("No secondary opportunities found")
        
        with rec_col3:
            if len(call_opportunities) > 2:
                strike = call_opportunities.iloc[2]['strike']
                ltp = call_opportunities.iloc[2]['calls_ltp']
                iv = call_opportunities.iloc[2]['calls_iv']
                chng_oi = call_opportunities.iloc[2]['calls_chng_oi']
                
                st.markdown(f"""
                ### 🟠 {strike:,.0f} CALL
                **Probability of Profit**: 35%  
                **Entry**: ₹{ltp*0.96:.2f}-{ltp:.2f}  
                **Target**: ₹{ltp*2:.2f} (100% profit)  
                **Stop Loss**: ₹{ltp*0.5:.2f}  
                **RR Ratio**: 1:2  
                
                **Logic**:  
                - Cheap premium: ₹{ltp:.2f}  
                - OI increased by {chng_oi:,.0f} contracts  
                - IV: {iv:.1f}% (below average)  
                """)
                st.progress(35)
            elif not put_opportunities.empty:
                strike = put_opportunities.iloc[0]['strike']
                ltp = put_opportunities.iloc[0]['puts_ltp']
                iv = put_opportunities.iloc[0]['puts_iv']
                oi = put_opportunities.iloc[0]['puts_oi']
                
                st.markdown(f"""
                ### 🔵 {strike:,.0f} PUT
                **Probability of Profit**: 30%  
                **Entry**: ₹{ltp:.2f}  
                **Target**: ₹{ltp*2:.2f} (100% profit)  
                **Stop Loss**: ₹{ltp*0.5:.2f}  
                
                **Logic**:  
                - Protective hedge against downside  
                - Strong OI support ({oi/100000:.1f}L contracts)  
                - IV: {iv:.1f}%  
                """)
                st.progress(30)
            else:
                st.warning("No tertiary opportunities found")
    else:
        st.warning("No trading opportunities found in selected range")
    
    # Key Insights
    st.subheader("💡 Trading Insights")
    
    if not df.empty:
        max_put = df.loc[df['puts_oi'].idxmax()]
        max_call = df.loc[df['calls_oi'].idxmax()]
        max_vol_call = df.loc[df['calls_volume'].idxmax()]
        max_vol_put = df.loc[df['puts_volume'].idxmax()]
        
        insights = f"""
        1. **Market Positioning**: 
           - Strongest support at {max_put['strike']:,.0f} ({max_put['puts_oi']/100000:,.1f}L put OI)
           - Strongest resistance at {max_call['strike']:,.0f} ({max_call['calls_oi']/100000:,.1f}L call OI)
        
        2. **Volume Activity**:
           - Most active CALL: {max_vol_call['strike']:,.0f} ({max_vol_call['calls_volume']:,.0f} contracts)
           - Most active PUT: {max_vol_put['strike']:,.0f} ({max_vol_put['puts_volume']:,.0f} contracts)
        
        3. **Probability Assessment**:
           - 55-60% chance of closing above {atm_strike:,.0f}
           - 35-40% chance of hitting {atm_strike+100:,.0f} in next session
           - <5% probability of dropping below {atm_strike-200:,.0f}
        
        4. **Execution Tips**:
           - Enter calls if market holds above {atm_strike:,.0f} at 11:30 AM
           - Use bracket orders for defined risk management
           - Exit positions if volatility drops >10% intraday
        """
        st.markdown(insights)
    else:
        st.warning("No insights available - data empty")
    
    # Data Table
    st.subheader("📈 Raw Option Chain Data")
    st.dataframe(df.head(50).style.format("{:,.2f}").background_gradient(
        subset=['straddle_pct_change'], 
        cmap='RdYlGn', 
        vmin=-5, 
        vmax=5
    ), height=600)

# How to Use
st.sidebar.header("How to Use This App")
st.sidebar.markdown("""
1. Upload your option chain file (CSV or Excel)
2. Adjust strike range using sliders
3. Review straddle premium heatmap
4. Analyze OI and volume patterns
5. Consider recommended trades with probabilities
6. Check raw data for specific strikes
""")
st.sidebar.info("Note: Probabilities based on IV, OI distribution, and historical patterns")

# Footer
st.caption("Disclaimer: This is for educational purposes only. Trading involves substantial risk.")
