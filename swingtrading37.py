import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from collections import Counter

# Page configuration
st.set_page_config(page_title="Advanced Trading Dashboard", layout="wide", initial_sidebar_collapsed=False)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stButton>button {width: 100%; background-color: #0066cc; color: white;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚀 Advanced Trading Analysis Dashboard")
st.markdown("---")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Predefined tickers
    ticker_options = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Sensex": "^BSESN",
        "Bitcoin USD": "BTC-USD",
        "Ethereum USD": "ETH-USD",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "USD/INR": "INR=X",
        "EUR/USD": "EURUSD=X",
        "Custom Ticker": "CUSTOM"
    }
    
    selected_option = st.selectbox("Select Asset", list(ticker_options.keys()))
    
    if selected_option == "Custom Ticker":
        ticker = st.text_input("Enter Ticker Symbol", "RELIANCE.NS")
    else:
        ticker = ticker_options[selected_option]
        st.info(f"Selected: {ticker}")
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    )
    
    # Period selection
    period = st.selectbox(
        "Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
    )
    
    st.markdown("---")
    
    # Fetch button
    fetch_button = st.button("📊 Fetch Data & Analyze", type="primary")

# Main content
if fetch_button:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # Fetch data from yfinance
            data = yf.download(ticker, period=period, interval=timeframe, progress=False)
            
            if data.empty:
                st.error("No data found. Please check the ticker symbol and try again.")
            else:
                st.session_state.data = data
                st.session_state.analysis_done = True
                st.success(f"✅ Data fetched successfully! Total records: {len(data)}")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.session_state.analysis_done = False

# Display analysis if data is available
if st.session_state.analysis_done and st.session_state.data is not None:
    data = st.session_state.data.copy()
    
    # Data preparation
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data = data.reset_index()
    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
    
    # Calculate metrics
    if 'close' in data.columns:
        latest_close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2] if len(data) > 1 else latest_close
        
        points_change = latest_close - prev_close
        pct_change = ((latest_close - prev_close) / prev_close) * 100
        
        highest = data['high'].max()
        lowest = data['low'].min()
        avg_volume = data['volume'].mean() if 'volume' in data.columns else 0
        
        # Display metrics in columns
        st.markdown("## 📈 Market Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Latest Close", f"{latest_close:.2f}", 
                     f"{points_change:+.2f} ({pct_change:+.2f}%)")
        with col2:
            st.metric("Highest", f"{highest:.2f}")
        with col3:
            st.metric("Lowest", f"{lowest:.2f}")
        with col4:
            st.metric("Range", f"{highest - lowest:.2f}")
        with col5:
            st.metric("Avg Volume", f"{avg_volume:,.0f}" if avg_volume > 0 else "N/A")
        
        st.markdown("---")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "📋 Data Table", "🔍 Pattern Analysis", "📥 Download"])
        
        with tab1:
            st.markdown("### Interactive Price Chart")
            
            # Create candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data['date'] if 'date' in data.columns else data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume
            if 'volume' in data.columns:
                colors = ['red' if row['close'] < row['open'] else 'green' 
                         for _, row in data.iterrows()]
                fig.add_trace(
                    go.Bar(
                        x=data['date'] if 'date' in data.columns else data.index,
                        y=data['volume'],
                        name='Volume',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=700,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Detailed Data Table")
            
            # Calculate additional metrics
            display_data = data.copy()
            if 'close' in display_data.columns:
                display_data['daily_change'] = display_data['close'].diff()
                display_data['daily_change_pct'] = display_data['close'].pct_change() * 100
                display_data['points_from_high'] = display_data['high'] - display_data['close']
                display_data['points_from_low'] = display_data['close'] - display_data['low']
            
            # Style the dataframe
            def color_negative_red(val):
                if isinstance(val, (int, float)):
                    color = 'red' if val < 0 else 'green' if val > 0 else 'black'
                    return f'color: {color}'
                return ''
            
            styled_df = display_data.tail(100).style.applymap(
                color_negative_red, 
                subset=['daily_change', 'daily_change_pct']
            ).format({
                'open': '{:.2f}',
                'high': '{:.2f}',
                'low': '{:.2f}',
                'close': '{:.2f}',
                'daily_change': '{:+.2f}',
                'daily_change_pct': '{:+.2f}%',
                'points_from_high': '{:.2f}',
                'points_from_low': '{:.2f}',
                'volume': '{:,.0f}'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=500)
        
        with tab3:
            st.markdown("### 🔮 Pattern Analysis & Forecast")
            
            # Prepare data for pattern analysis
            analysis_data = data.copy()
            
            if 'date' in analysis_data.columns:
                analysis_data['date'] = pd.to_datetime(analysis_data['date'])
                analysis_data['day_of_week'] = analysis_data['date'].dt.day_name()
                analysis_data['day_of_month'] = analysis_data['date'].dt.day
                analysis_data['month'] = analysis_data['date'].dt.month_name()
                analysis_data['hour'] = analysis_data['date'].dt.hour
                analysis_data['year'] = analysis_data['date'].dt.year
            
            if 'close' in analysis_data.columns:
                analysis_data['price_change'] = analysis_data['close'].diff()
                analysis_data['price_change_pct'] = analysis_data['close'].pct_change() * 100
                analysis_data['is_up'] = analysis_data['price_change'] > 0
            
            # Pattern Recognition
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📅 Day of Week Pattern")
                if 'day_of_week' in analysis_data.columns:
                    dow_stats = analysis_data.groupby('day_of_week').agg({
                        'price_change': ['mean', 'std', 'count'],
                        'is_up': 'sum'
                    }).round(2)
                    
                    dow_stats.columns = ['Avg Change', 'Std Dev', 'Count', 'Up Days']
                    dow_stats['Win Rate %'] = (dow_stats['Up Days'] / dow_stats['Count'] * 100).round(2)
                    dow_stats = dow_stats.sort_values('Avg Change', ascending=False)
                    
                    st.dataframe(dow_stats, use_container_width=True)
                    
                    # Best day
                    best_day = dow_stats['Avg Change'].idxmax()
                    best_change = dow_stats.loc[best_day, 'Avg Change']
                    st.success(f"🏆 Best Day: **{best_day}** (Avg: +{best_change:.2f} points)")
            
            with col2:
                st.markdown("#### 📆 Month Pattern")
                if 'month' in analysis_data.columns:
                    month_stats = analysis_data.groupby('month').agg({
                        'price_change': ['mean', 'std', 'count'],
                        'is_up': 'sum'
                    }).round(2)
                    
                    month_stats.columns = ['Avg Change', 'Std Dev', 'Count', 'Up Days']
                    month_stats['Win Rate %'] = (month_stats['Up Days'] / month_stats['Count'] * 100).round(2)
                    month_stats = month_stats.sort_values('Avg Change', ascending=False)
                    
                    st.dataframe(month_stats, use_container_width=True)
                    
                    # Best month
                    best_month = month_stats['Avg Change'].idxmax()
                    best_month_change = month_stats.loc[best_month, 'Avg Change']
                    st.success(f"🏆 Best Month: **{best_month}** (Avg: +{best_month_change:.2f} points)")
            
            # Time-based patterns (for intraday)
            if 'hour' in analysis_data.columns and timeframe in ['1m', '2m', '5m', '15m', '30m', '60m', '1h']:
                st.markdown("#### ⏰ Hour of Day Pattern")
                hour_stats = analysis_data.groupby('hour').agg({
                    'price_change': ['mean', 'std', 'count'],
                    'is_up': 'sum'
                }).round(2)
                
                hour_stats.columns = ['Avg Change', 'Std Dev', 'Count', 'Up Periods']
                hour_stats['Win Rate %'] = (hour_stats['Up Periods'] / hour_stats['Count'] * 100).round(2)
                
                st.dataframe(hour_stats, use_container_width=True)
            
            # Similar Pattern Matching
            st.markdown("---")
            st.markdown("#### 🎯 Similar Pattern Detection")
            
            # Get recent pattern (last 5 periods)
            recent_pattern = analysis_data['price_change'].tail(5).values
            
            # Find similar patterns in history
            similar_patterns = []
            window_size = 5
            
            for i in range(len(analysis_data) - window_size - 1):
                historical_pattern = analysis_data['price_change'].iloc[i:i+window_size].values
                
                # Calculate similarity (using correlation)
                if not np.isnan(historical_pattern).any() and not np.isnan(recent_pattern).any():
                    correlation = np.corrcoef(recent_pattern, historical_pattern)[0, 1]
                    
                    if correlation > 0.7:  # High correlation threshold
                        next_move = analysis_data['price_change'].iloc[i+window_size]
                        similar_patterns.append({
                            'date': analysis_data['date'].iloc[i] if 'date' in analysis_data.columns else i,
                            'correlation': correlation,
                            'next_move': next_move
                        })
            
            if similar_patterns:
                similar_df = pd.DataFrame(similar_patterns).sort_values('correlation', ascending=False).head(10)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(similar_df.style.format({
                        'correlation': '{:.2%}',
                        'next_move': '{:+.2f}'
                    }), use_container_width=True)
                
                with col2:
                    avg_next_move = similar_df['next_move'].mean()
                    median_next_move = similar_df['next_move'].median()
                    up_count = (similar_df['next_move'] > 0).sum()
                    down_count = (similar_df['next_move'] < 0).sum()
                    
                    st.metric("Predicted Next Move (Avg)", f"{avg_next_move:+.2f} points")
                    st.metric("Predicted Next Move (Median)", f"{median_next_move:+.2f} points")
                    st.metric("Historical Outcome", f"{up_count} Up | {down_count} Down")
            else:
                st.info("No highly similar patterns found in historical data.")
            
            # Key Insights Summary
            st.markdown("---")
            st.markdown("### 🎓 Key Insights & Forecast")
            
            insights = []
            
            # Overall trend
            recent_avg_change = analysis_data['price_change'].tail(20).mean()
            if recent_avg_change > 0:
                insights.append(f"📈 **Bullish Trend**: Recent 20-period average shows +{recent_avg_change:.2f} points upward movement")
            else:
                insights.append(f"📉 **Bearish Trend**: Recent 20-period average shows {recent_avg_change:.2f} points downward movement")
            
            # Day of week insight
            if 'day_of_week' in analysis_data.columns:
                current_day = datetime.now().strftime('%A')
                if current_day in dow_stats.index:
                    day_avg = dow_stats.loc[current_day, 'Avg Change']
                    insights.append(f"📅 **{current_day} Pattern**: Historically averages {day_avg:+.2f} points")
            
            # Month insight
            if 'month' in analysis_data.columns:
                current_month = datetime.now().strftime('%B')
                if current_month in month_stats.index:
                    month_avg = month_stats.loc[current_month, 'Avg Change']
                    insights.append(f"📆 **{current_month} Pattern**: Historically averages {month_avg:+.2f} points")
            
            # Volatility
            recent_volatility = analysis_data['price_change'].tail(20).std()
            insights.append(f"📊 **Volatility**: Recent standard deviation is {recent_volatility:.2f} points")
            
            # Similar pattern forecast
            if similar_patterns:
                insights.append(f"🎯 **Pattern Match**: {len(similar_patterns)} similar patterns found, suggesting {avg_next_move:+.2f} points movement")
            
            # Display insights
            for insight in insights:
                st.markdown(insight)
            
            # Final forecast
            st.markdown("---")
            forecast_points = 0
            confidence_factors = []
            
            if recent_avg_change != 0:
                forecast_points += recent_avg_change * 0.3
                confidence_factors.append("Recent trend")
            
            if similar_patterns:
                forecast_points += avg_next_move * 0.4
                confidence_factors.append("Pattern matching")
            
            if 'day_of_week' in analysis_data.columns:
                current_day = datetime.now().strftime('%A')
                if current_day in dow_stats.index:
                    forecast_points += dow_stats.loc[current_day, 'Avg Change'] * 0.3
                    confidence_factors.append("Day of week")
            
            forecast_direction = "UPWARD 📈" if forecast_points > 0 else "DOWNWARD 📉" if forecast_points < 0 else "NEUTRAL ➡️"
            confidence = "High" if len(confidence_factors) >= 3 else "Medium" if len(confidence_factors) == 2 else "Low"
            
            st.markdown(f"""
            ### 🎯 FORECAST SUMMARY
            
            **Predicted Movement**: {forecast_direction}
            
            **Expected Points**: {forecast_points:+.2f}
            
            **Confidence Level**: {confidence}
            
            **Based on**: {', '.join(confidence_factors)}
            
            ---
            
            ⚠️ **Disclaimer**: This forecast is based on historical patterns and statistical analysis. 
            Markets are unpredictable and past performance doesn't guarantee future results. 
            Always do your own research and risk management.
            """)
        
        with tab4:
            st.markdown("### 📥 Download Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_buffer = io.StringIO()
                data.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"{ticker}_{timeframe}_{period}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name='Data')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"{ticker}_{timeframe}_{period}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            st.info(f"Total records: {len(data)}")

else:
    # Welcome screen
    st.info("👈 Configure your analysis settings in the sidebar and click '📊 Fetch Data & Analyze' to begin.")
    
    st.markdown("""
    ### 🎯 Features:
    
    1. **Multi-Asset Support**: Nifty, Bank Nifty, Sensex, Crypto, Forex, Commodities
    2. **Flexible Timeframes**: From 1-minute to monthly data
    3. **Pattern Recognition**: Discover recurring market patterns
    4. **Smart Forecasting**: AI-powered predictions based on historical data
    5. **Interactive Charts**: Beautiful candlestick charts with volume
    6. **Data Export**: Download analysis as CSV or Excel
    7. **Real-time Metrics**: Live updates on price movements
    
    ### 📊 Pattern Analysis Includes:
    - Day of week patterns
    - Monthly seasonality
    - Hourly patterns (for intraday)
    - Similar pattern matching
    - Volatility analysis
    - Trend forecasting
    
    ---
    
    ⚠️ **Note**: This tool respects yfinance API rate limits. Data is cached until you click the fetch button again.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📈 Advanced Trading Analysis Dashboard | Built with Streamlit & yfinance</p>
        <p style='font-size: 0.8em;'>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
""", unsafe_allow_html=True)
