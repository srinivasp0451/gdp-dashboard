import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io
from collections import Counter
import pytz

# Page configuration
st.set_page_config(page_title="Advanced Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

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
    
    # Convert timezone-aware datetime to IST and remove timezone
    if 'date' in data.columns:
        try:
            # Convert to IST timezone
            ist = pytz.timezone('Asia/Kolkata')
            if data['date'].dt.tz is not None:
                data['date'] = data['date'].dt.tz_convert(ist)
            else:
                data['date'] = data['date'].dt.tz_localize('UTC').dt.tz_convert(ist)
            # Remove timezone info for Excel compatibility
            data['date'] = data['date'].dt.tz_localize(None)
        except:
            # If already timezone-naive, just ensure it's datetime
            data['date'] = pd.to_datetime(data['date'])
    elif 'datetime' in data.columns:
        try:
            ist = pytz.timezone('Asia/Kolkata')
            if data['datetime'].dt.tz is not None:
                data['datetime'] = data['datetime'].dt.tz_convert(ist)
            else:
                data['datetime'] = data['datetime'].dt.tz_localize('UTC').dt.tz_convert(ist)
            data['datetime'] = data['datetime'].dt.tz_localize(None)
            data.rename(columns={'datetime': 'date'}, inplace=True)
        except:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.rename(columns={'datetime': 'date'}, inplace=True)
    
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Chart", "📋 Data Table", "🔍 Advanced Pattern Analysis", "🔥 Heatmaps & Volatility", "📥 Download"])
        
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
            st.markdown("### 🔮 Advanced Pattern Analysis & Forecast")
            
            # Prepare data for pattern analysis
            analysis_data = data.copy()
            
            if 'date' in analysis_data.columns:
                analysis_data['date'] = pd.to_datetime(analysis_data['date'])
                analysis_data['day_of_week'] = analysis_data['date'].dt.day_name()
                analysis_data['day_of_month'] = analysis_data['date'].dt.day
                analysis_data['month'] = analysis_data['date'].dt.month_name()
                analysis_data['hour'] = analysis_data['date'].dt.hour
                analysis_data['minute'] = analysis_data['date'].dt.minute
                analysis_data['time_str'] = analysis_data['date'].dt.strftime('%H:%M')
                analysis_data['year'] = analysis_data['date'].dt.year
                analysis_data['week_of_year'] = analysis_data['date'].dt.isocalendar().week
            
            if 'close' in analysis_data.columns:
                analysis_data['price_change'] = analysis_data['close'].diff()
                analysis_data['price_change_pct'] = analysis_data['close'].pct_change() * 100
                analysis_data['is_up'] = analysis_data['price_change'] > 0
                analysis_data['candle_body'] = analysis_data['close'] - analysis_data['open']
                analysis_data['candle_type'] = analysis_data['candle_body'].apply(lambda x: 'Green' if x > 0 else 'Red' if x < 0 else 'Doji')
                analysis_data['upper_wick'] = analysis_data['high'] - analysis_data[['open', 'close']].max(axis=1)
                analysis_data['lower_wick'] = analysis_data[['open', 'close']].min(axis=1) - analysis_data['low']
                analysis_data['candle_range'] = analysis_data['high'] - analysis_data['low']
                analysis_data['volatility'] = analysis_data['price_change'].rolling(window=20).std()
            
            # ADVANCED PATTERN DETECTION
            st.markdown("## 🎯 Multi-Candle Pattern Detection")
            
            # 1. Rally Detection (Consecutive Green/Red Candles)
            st.markdown("### 📈 Rally & Reversal Patterns")
            
            def detect_rallies(df, min_candles=3):
                rallies = []
                current_type = None
                current_start = 0
                current_points = 0
                
                for i, row in df.iterrows():
                    candle = row['candle_type']
                    
                    if candle == current_type and candle != 'Doji':
                        current_points += abs(row['candle_body'])
                    else:
                        if current_type and i - current_start >= min_candles:
                            rallies.append({
                                'start_date': df.loc[current_start, 'date'],
                                'end_date': df.loc[i-1, 'date'],
                                'type': f"{current_type} Rally",
                                'candles': i - current_start,
                                'points': current_points,
                                'next_move': df.loc[i, 'price_change'] if i < len(df) else np.nan
                            })
                        current_type = candle if candle != 'Doji' else None
                        current_start = i
                        current_points = abs(row['candle_body']) if candle != 'Doji' else 0
                
                return pd.DataFrame(rallies)
            
            rally_df = detect_rallies(analysis_data, min_candles=3)
            
            if not rally_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🚀 Bullish Rallies (3+ Green Candles)")
                    bullish_rallies = rally_df[rally_df['type'] == 'Green Rally'].tail(10)
                    if not bullish_rallies.empty:
                        st.dataframe(bullish_rallies.style.format({
                            'points': '{:.2f}',
                            'next_move': '{:+.2f}'
                        }), use_container_width=True)
                        
                        avg_bullish_next = bullish_rallies['next_move'].mean()
                        st.info(f"📊 After bullish rallies: Average next move = **{avg_bullish_next:+.2f} points**")
                    else:
                        st.warning("No bullish rallies found")
                
                with col2:
                    st.markdown("#### 🔻 Bearish Rallies (3+ Red Candles)")
                    bearish_rallies = rally_df[rally_df['type'] == 'Red Rally'].tail(10)
                    if not bearish_rallies.empty:
                        st.dataframe(bearish_rallies.style.format({
                            'points': '{:.2f}',
                            'next_move': '{:+.2f}'
                        }), use_container_width=True)
                        
                        avg_bearish_next = bearish_rallies['next_move'].mean()
                        st.info(f"📊 After bearish rallies: Average next move = **{avg_bearish_next:+.2f} points**")
                    else:
                        st.warning("No bearish rallies found")
            
            # 2. Liquidity Sweep Detection
            st.markdown("---")
            st.markdown("### 💧 Liquidity Sweep Patterns")
            
            def detect_liquidity_sweeps(df, lookback=20, sweep_threshold=0.5):
                sweeps = []
                
                for i in range(lookback, len(df)):
                    current = df.iloc[i]
                    previous = df.iloc[i-lookback:i]
                    
                    # Check if current high swept previous highs significantly
                    if current['high'] > previous['high'].max() * (1 + sweep_threshold/100):
                        next_moves = df.iloc[i+1:i+6]['price_change'].values if i+6 < len(df) else []
                        sweeps.append({
                            'date': current['date'],
                            'type': 'High Sweep',
                            'sweep_points': current['high'] - previous['high'].max(),
                            'next_1': next_moves[0] if len(next_moves) > 0 else np.nan,
                            'next_3_total': sum(next_moves[:3]) if len(next_moves) >= 3 else np.nan,
                            'next_5_total': sum(next_moves[:5]) if len(next_moves) >= 5 else np.nan
                        })
                    
                    # Check if current low swept previous lows significantly
                    if current['low'] < previous['low'].min() * (1 - sweep_threshold/100):
                        next_moves = df.iloc[i+1:i+6]['price_change'].values if i+6 < len(df) else []
                        sweeps.append({
                            'date': current['date'],
                            'type': 'Low Sweep',
                            'sweep_points': previous['low'].min() - current['low'],
                            'next_1': next_moves[0] if len(next_moves) > 0 else np.nan,
                            'next_3_total': sum(next_moves[:3]) if len(next_moves) >= 3 else np.nan,
                            'next_5_total': sum(next_moves[:5]) if len(next_moves) >= 5 else np.nan
                        })
                
                return pd.DataFrame(sweeps)
            
            sweep_df = detect_liquidity_sweeps(analysis_data)
            
            if not sweep_df.empty:
                st.dataframe(sweep_df.tail(15).style.format({
                    'sweep_points': '{:.2f}',
                    'next_1': '{:+.2f}',
                    'next_3_total': '{:+.2f}',
                    'next_5_total': '{:+.2f}'
                }), use_container_width=True)
                
                high_sweep_avg = sweep_df[sweep_df['type'] == 'High Sweep']['next_3_total'].mean()
                low_sweep_avg = sweep_df[sweep_df['type'] == 'Low Sweep']['next_3_total'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("After High Sweep (3 candles)", f"{high_sweep_avg:+.2f} pts" if not np.isnan(high_sweep_avg) else "N/A")
                with col2:
                    st.metric("After Low Sweep (3 candles)", f"{low_sweep_avg:+.2f} pts" if not np.isnan(low_sweep_avg) else "N/A")
            else:
                st.info("No significant liquidity sweeps detected")
            
            # 3. Opening Range Patterns (9:15 - 9:30 for Indian markets)
            if 'hour' in analysis_data.columns and 'minute' in analysis_data.columns:
                st.markdown("---")
                st.markdown("### 🔔 Opening Range Patterns (First 15 minutes)")
                
                opening_data = analysis_data[
                    ((analysis_data['hour'] == 9) & (analysis_data['minute'] >= 15)) |
                    ((analysis_data['hour'] == 9) & (analysis_data['minute'] < 30))
                ].copy()
                
                if not opening_data.empty:
                    opening_data['date_only'] = opening_data['date'].dt.date
                    opening_patterns = opening_data.groupby('date_only').agg({
                        'price_change': 'sum',
                        'candle_type': lambda x: 'Bullish' if (x == 'Green').sum() > (x == 'Red').sum() else 'Bearish',
                        'candle_range': 'sum'
                    }).reset_index()
                    opening_patterns.columns = ['Date', 'Opening_Move', 'Direction', 'Volatility']
                    
                    # Get rest of day movement
                    rest_of_day = []
                    for date in opening_patterns['Date']:
                        day_data = analysis_data[analysis_data['date'].dt.date == date]
                        opening_end = day_data[
                            (day_data['hour'] == 9) & (day_data['minute'] >= 30)
                        ].index.min() if not day_data.empty else None
                        
                        if opening_end is not None and opening_end < len(day_data) - 1:
                            rest_move = day_data.loc[opening_end:, 'price_change'].sum()
                            rest_of_day.append(rest_move)
                        else:
                            rest_of_day.append(np.nan)
                    
                    opening_patterns['Rest_of_Day'] = rest_of_day
                    
                    st.dataframe(opening_patterns.tail(20).style.format({
                        'Opening_Move': '{:+.2f}',
                        'Volatility': '{:.2f}',
                        'Rest_of_Day': '{:+.2f}'
                    }), use_container_width=True)
                    
                    # Statistics
                    bullish_open = opening_patterns[opening_patterns['Direction'] == 'Bullish']
                    bearish_open = opening_patterns[opening_patterns['Direction'] == 'Bearish']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if not bullish_open.empty:
                            avg_after_bullish = bullish_open['Rest_of_Day'].mean()
                            st.success(f"📈 After Bullish Opening: Rest of day averages **{avg_after_bullish:+.2f} points**")
                    
                    with col2:
                        if not bearish_open.empty:
                            avg_after_bearish = bearish_open['Rest_of_Day'].mean()
                            st.error(f"📉 After Bearish Opening: Rest of day averages **{avg_after_bearish:+.2f} points**")
            
            # 4. High Volatility Reversal Patterns
            st.markdown("---")
            st.markdown("### ⚡ High Volatility Reversal Patterns")
            
            high_vol_threshold = analysis_data['volatility'].quantile(0.80)
            high_vol_periods = analysis_data[analysis_data['volatility'] > high_vol_threshold].copy()
            
            if not high_vol_periods.empty:
                high_vol_periods['next_5_move'] = high_vol_periods.index.map(
                    lambda i: analysis_data.loc[i+1:i+5, 'price_change'].sum() 
                    if i+5 < len(analysis_data) else np.nan
                )
                
                st.dataframe(high_vol_periods[['date', 'volatility', 'price_change', 'next_5_move']].tail(15).style.format({
                    'volatility': '{:.2f}',
                    'price_change': '{:+.2f}',
                    'next_5_move': '{:+.2f}'
                }), use_container_width=True)
                
                avg_reversal = high_vol_periods['next_5_move'].mean()
                st.info(f"📊 After high volatility: Next 5 periods average **{avg_reversal:+.2f} points** movement")
            
            # 5. Similar Multi-Candle Pattern Matching
            st.markdown("---")
            st.markdown("### 🎯 Similar Multi-Candle Pattern Matching")
            
            pattern_length = st.slider("Pattern Length (number of candles)", 3, 10, 5)
            
            # Get recent pattern
            recent_pattern = analysis_data[['price_change', 'candle_body', 'candle_range']].tail(pattern_length).values
            
            # Find similar patterns in history
            similar_patterns = []
            
            for i in range(len(analysis_data) - pattern_length - 5):
                historical_pattern = analysis_data[['price_change', 'candle_body', 'candle_range']].iloc[i:i+pattern_length].values
                
                # Calculate similarity using multiple metrics
                if not np.isnan(historical_pattern).any() and not np.isnan(recent_pattern).any():
                    # Normalize patterns
                    hist_norm = (historical_pattern - historical_pattern.mean()) / (historical_pattern.std() + 1e-8)
                    recent_norm = (recent_pattern - recent_pattern.mean()) / (recent_pattern.std() + 1e-8)
                    
                    # Calculate correlation for each feature
                    correlations = []
                    for col in range(3):
                        corr = np.corrcoef(recent_norm[:, col], hist_norm[:, col])[0, 1]
                        correlations.append(corr)
                    
                    avg_correlation = np.mean(correlations)
                    
                    if avg_correlation > 0.65:  # Similarity threshold
                        # Get next moves
                        next_1 = analysis_data.iloc[i+pattern_length]['price_change']
                        next_3 = analysis_data.iloc[i+pattern_length:i+pattern_length+3]['price_change'].sum()
                        next_5 = analysis_data.iloc[i+pattern_length:i+pattern_length+5]['price_change'].sum()
                        
                        similar_patterns.append({
                            'date': analysis_data.iloc[i]['date'],
                            'similarity': avg_correlation * 100,
                            'next_1_candle': next_1,
                            'next_3_candles': next_3,
                            'next_5_candles': next_5,
                            'pattern_summary': f"{len([x for x in historical_pattern[:, 1] if x > 0])}G-{len([x for x in historical_pattern[:, 1] if x < 0])}R"
                        })
            
            if similar_patterns:
                similar_df = pd.DataFrame(similar_patterns).sort_values('similarity', ascending=False).head(20)
                
                st.dataframe(similar_df.style.format({
                    'similarity': '{:.1f}%',
                    'next_1_candle': '{:+.2f}',
                    'next_3_candles': '{:+.2f}',
                    'next_5_candles': '{:+.2f}'
                }).background_gradient(subset=['similarity'], cmap='Greens'), use_container_width=True)
                
                # Predictions based on similar patterns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    pred_1 = similar_df['next_1_candle'].mean()
                    confidence_1 = similar_df['similarity'].mean()
                    st.metric("Next 1 Candle Prediction", f"{pred_1:+.2f} pts", 
                             delta=f"Confidence: {confidence_1:.1f}%")
                
                with col2:
                    pred_3 = similar_df['next_3_candles'].mean()
                    st.metric("Next 3 Candles Prediction", f"{pred_3:+.2f} pts",
                             delta=f"Based on {len(similar_df)} patterns")
                
                with col3:
                    pred_5 = similar_df['next_5_candles'].mean()
                    win_rate = (similar_df['next_5_candles'] > 0).sum() / len(similar_df) * 100
                    st.metric("Next 5 Candles Prediction", f"{pred_5:+.2f} pts",
                             delta=f"Win Rate: {win_rate:.1f}%")
                
                # Explain the pattern
                st.markdown("#### 📖 Pattern Interpretation")
                recent_summary = f"{len([x for x in recent_pattern[:, 1] if x > 0])} Green, {len([x for x in recent_pattern[:, 1] if x < 0])} Red candles"
                st.info(f"""
                **Current Pattern**: {recent_summary} over last {pattern_length} candles
                
                **Found**: {len(similar_df)} similar historical patterns with avg {similar_df['similarity'].mean():.1f}% similarity
                
                **Historical Outcome**: When this pattern occurred before:
                - Next candle moved **{pred_1:+.2f} points** on average
                - Next 3 candles moved **{pred_3:+.2f} points** on average  
                - Next 5 candles moved **{pred_5:+.2f} points** on average
                - Win rate (positive movement in 5 candles): **{win_rate:.1f}%**
                """)
            else:
                st.warning("No highly similar patterns found. Try adjusting the pattern length.")
            
            # Pattern Recognition Stats
            st.markdown("---")
            st.markdown("### 📊 Day & Time Pattern Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'day_of_week' in analysis_data.columns:
                    st.markdown("#### 📅 Day of Week Performance")
                    dow_stats = analysis_data.groupby('day_of_week').agg({
                        'price_change': ['mean', 'std', 'count'],
                        'is_up': 'sum'
                    }).round(2)
                    
                    dow_stats.columns = ['Avg Change', 'Std Dev', 'Count', 'Up Days']
                    dow_stats['Win Rate %'] = (dow_stats['Up Days'] / dow_stats['Count'] * 100).round(2)
                    dow_stats = dow_stats.sort_values('Avg Change', ascending=False)
                    
                    st.dataframe(dow_stats, use_container_width=True)
            
            with col2:
                if 'week_of_year' in analysis_data.columns:
                    st.markdown("#### 📆 Week of Month Performance")
                    analysis_data['week_of_month'] = (analysis_data['day_of_month'] - 1) // 7 + 1
                    wom_stats = analysis_data.groupby('week_of_month').agg({
                        'price_change': ['mean', 'count'],
                        'is_up': 'sum'
                    }).round(2)
                    wom_stats.columns = ['Avg Change', 'Count', 'Up Days']
                    wom_stats['Win Rate %'] = (wom_stats['Up Days'] / wom_stats['Count'] * 100).round(2)
                    
                    st.dataframe(wom_stats, use_container_width=True)
            
            # Final Advanced Forecast
            st.markdown("---")
            st.markdown("### 🎯 COMPREHENSIVE FORECAST")
            
            forecast_signals = []
            
            # Signal 1: Similar patterns
            if similar_patterns:
                forecast_signals.append({
                    'signal': 'Similar Pattern Match',
                    'prediction': pred_5,
                    'confidence': similar_df['similarity'].mean(),
                    'weight': 0.35
                })
            
            # Signal 2: Recent rallies
            if not rally_df.empty:
                recent_rally = rally_df.tail(1)
                if not recent_rally.empty:
                    rally_signal = recent_rally['next_move'].values[0]
                    if not np.isnan(rally_signal):
                        forecast_signals.append({
                            'signal': 'Recent Rally Pattern',
                            'prediction': rally_signal,
                            'confidence': 70,
                            'weight': 0.20
                        })
            
            # Signal 3: Liquidity sweeps
            if not sweep_df.empty:
                recent_sweep = sweep_df.tail(1)
                if not recent_sweep.empty and not np.isnan(recent_sweep['next_3_total'].values[0]):
                    forecast_signals.append({
                        'signal': 'Liquidity Sweep',
                        'prediction': recent_sweep['next_3_total'].values[0],
                        'confidence': 65,
                        'weight': 0.15
                    })
            
            # Signal 4: Day of week
            if 'day_of_week' in analysis_data.columns:
                current_day = datetime.now().strftime('%A')
                if current_day in dow_stats.index:
                    forecast_signals.append({
                        'signal': f'{current_day} Historical',
                        'prediction': dow_stats.loc[current_day, 'Avg Change'],
                        'confidence': dow_stats.loc[current_day, 'Win Rate %'],
                        'weight': 0.15
                    })
            
            # Signal 5: Volatility trend
            if not high_vol_periods.empty and not np.isnan(avg_reversal):
                forecast_signals.append({
                    'signal': 'Volatility Pattern',
                    'prediction': avg_reversal,
                    'confidence': 60,
                    'weight': 0.15
                })
            
            if forecast_signals:
                # Calculate weighted forecast
                total_weight = sum([s['weight'] for s in forecast_signals])
                weighted_prediction = sum([s['prediction'] * s['weight'] for s in forecast_signals]) / total_weight
                avg_confidence = np.mean([s['confidence'] for s in forecast_signals])
                
                # Display signals
                signal_df = pd.DataFrame(forecast_signals)
                st.dataframe(signal_df.style.format({
                    'prediction': '{:+.2f}',
                    'confidence': '{:.1f}%',
                    'weight': '{:.0%}'
                }), use_container_width=True)
                
                # Final verdict
                direction = "📈 BULLISH" if weighted_prediction > 0 else "📉 BEARISH" if weighted_prediction < 0 else "➡️ NEUTRAL"
                confidence_level = "🟢 HIGH" if avg_confidence > 70 else "🟡 MEDIUM" if avg_confidence > 50 else "🔴 LOW"
                
                st.markdown(f"""
                ### 🎯 FINAL FORECAST
                
                **Direction**: {direction}
                
                **Expected Move**: **{weighted_prediction:+.2f} points** (next 3-5 candles)
                
                **Confidence**: {confidence_level} ({avg_confidence:.1f}%)
                
                **Based on**: {len(forecast_signals)} independent signals
                
                ---
                
                ⚠️ **Risk Disclaimer**: This is algorithmic analysis based on historical patterns. 
                Past performance does not guarantee future results. Always use proper risk management.
                """)
            else:
                st.warning("Insufficient data for comprehensive forecast. Need more historical patterns.")['next_move'].mean()
                median_next_move = similar_df['next_move'].median()
                up_count = (similar_df['next_move'] > 0).sum()
                down_count = (similar_df['next_move'] < 0).sum()
                    
                st.metric("Predicted Next Move (Avg)", f"{avg_next_move:+.2f} points")
                st.metric("Predicted Next Move (Median)", f"{median_next_move:+.2f} points")
                st.metric("Historical Outcome", f"{up_count} Up | {down_count} Down")
            #else:
            #   st.info("No highly similar patterns found in historical data.")
            
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
            st.markdown("### 🔥 Advanced Heatmaps & Volatility Analysis")
            
            # Prepare heatmap data
            heatmap_data = data.copy()
            
            if 'date' in heatmap_data.columns:
                heatmap_data['day_of_week'] = heatmap_data['date'].dt.day_name()
                heatmap_data['month'] = heatmap_data['date'].dt.month_name()
                heatmap_data['hour'] = heatmap_data['date'].dt.hour
                heatmap_data['year'] = heatmap_data['date'].dt.year
                heatmap_data['day_of_month'] = heatmap_data['date'].dt.day
            
            if 'close' in heatmap_data.columns:
                heatmap_data['returns'] = heatmap_data['close'].pct_change() * 100
                heatmap_data['volatility'] = heatmap_data['returns'].rolling(window=20).std()
                heatmap_data['price_range'] = heatmap_data['high'] - heatmap_data['low']
                heatmap_data['body_size'] = abs(heatmap_data['close'] - heatmap_data['open'])
            
            # 1. Day of Week vs Hour Heatmap (for intraday data)
            if 'hour' in heatmap_data.columns and heatmap_data['hour'].nunique() > 1:
                st.markdown("#### ⏰ Day of Week vs Hour - Average Returns Heatmap")
                
                pivot_returns = heatmap_data.pivot_table(
                    values='returns',
                    index='day_of_week',
                    columns='hour',
                    aggfunc='mean'
                )
                
                # Order days correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                pivot_returns = pivot_returns.reindex([d for d in day_order if d in pivot_returns.index])
                
                fig_hour_day = go.Figure(data=go.Heatmap(
                    z=pivot_returns.values,
                    x=pivot_returns.columns,
                    y=pivot_returns.index,
                    colorscale='RdYlGn',
                    text=pivot_returns.values,
                    texttemplate='%{text:.2f}%',
                    textfont={"size": 10},
                    colorbar=dict(title="Returns %")
                ))
                
                fig_hour_day.update_layout(
                    title="Average Returns by Day and Hour",
                    xaxis_title="Hour of Day",
                    yaxis_title="Day of Week",
                    height=400
                )
                
                st.plotly_chart(fig_hour_day, use_container_width=True)
                
                # Find best time slots
                best_slot = pivot_returns.stack().idxmax()
                best_return = pivot_returns.stack().max()
                st.success(f"🏆 Best Trading Time: **{best_slot[0]} at {best_slot[1]:02d}:00** (Avg: +{best_return:.2f}%)")
                
                # Volatility by hour
                st.markdown("#### 📊 Hour vs Volatility Heatmap")
                pivot_vol_hour = heatmap_data.pivot_table(
                    values='volatility',
                    index='day_of_week',
                    columns='hour',
                    aggfunc='mean'
                )
                pivot_vol_hour = pivot_vol_hour.reindex([d for d in day_order if d in pivot_vol_hour.index])
                
                fig_vol_hour = go.Figure(data=go.Heatmap(
                    z=pivot_vol_hour.values,
                    x=pivot_vol_hour.columns,
                    y=pivot_vol_hour.index,
                    colorscale='Reds',
                    text=pivot_vol_hour.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Volatility")
                ))
                
                fig_vol_hour.update_layout(
                    title="Volatility by Day and Hour",
                    xaxis_title="Hour of Day",
                    yaxis_title="Day of Week",
                    height=400
                )
                
                st.plotly_chart(fig_vol_hour, use_container_width=True)
                
                most_volatile = pivot_vol_hour.stack().idxmax()
                most_vol_value = pivot_vol_hour.stack().max()
                st.warning(f"⚡ Most Volatile: **{most_volatile[0]} at {most_volatile[1]:02d}:00** (Volatility: {most_vol_value:.2f})")
            
            # 2. Month vs Year Heatmap
            if 'year' in heatmap_data.columns and 'month' in heatmap_data.columns:
                st.markdown("---")
                st.markdown("#### 📆 Month vs Year - Returns Heatmap")
                
                pivot_month_year = heatmap_data.pivot_table(
                    values='returns',
                    index='month',
                    columns='year',
                    aggfunc='sum'
                )
                
                # Order months
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                pivot_month_year = pivot_month_year.reindex([m for m in month_order if m in pivot_month_year.index])
                
                fig_month_year = go.Figure(data=go.Heatmap(
                    z=pivot_month_year.values,
                    x=pivot_month_year.columns,
                    y=pivot_month_year.index,
                    colorscale='RdYlGn',
                    text=pivot_month_year.values,
                    texttemplate='%{text:.1f}%',
                    textfont={"size": 10},
                    colorbar=dict(title="Total Returns %")
                ))
                
                fig_month_year.update_layout(
                    title="Total Returns by Month and Year",
                    xaxis_title="Year",
                    yaxis_title="Month",
                    height=500
                )
                
                st.plotly_chart(fig_month_year, use_container_width=True)
                
                best_month_year = pivot_month_year.stack().idxmax()
                best_month_return = pivot_month_year.stack().max()
                worst_month_year = pivot_month_year.stack().idxmin()
                worst_month_return = pivot_month_year.stack().min()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🏆 Best: **{best_month_year[0]} {best_month_year[1]}** (+{best_month_return:.2f}%)")
                with col2:
                    st.error(f"📉 Worst: **{worst_month_year[0]} {worst_month_year[1]}** ({worst_month_return:.2f}%)")
            
            # 3. Day of Week vs Month Heatmap
            st.markdown("---")
            st.markdown("#### 📅 Day of Week vs Month - Returns Heatmap")
            
            pivot_day_month = heatmap_data.pivot_table(
                values='returns',
                index='day_of_week',
                columns='month',
                aggfunc='mean'
            )
            
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_day_month = pivot_day_month.reindex([d for d in day_order if d in pivot_day_month.index])
            
            fig_day_month = go.Figure(data=go.Heatmap(
                z=pivot_day_month.values,
                x=pivot_day_month.columns,
                y=pivot_day_month.index,
                colorscale='RdYlGn',
                text=pivot_day_month.values,
                texttemplate='%{text:.2f}%',
                textfont={"size": 10},
                colorbar=dict(title="Avg Returns %")
            ))
            
            fig_day_month.update_layout(
                title="Average Returns by Day of Week and Month",
                xaxis_title="Month",
                yaxis_title="Day of Week",
                height=400
            )
            
            st.plotly_chart(fig_day_month, use_container_width=True)
            
            # 4. Volatility by Day and Month
            st.markdown("#### ⚡ Volatility: Day vs Month Heatmap")
            
            pivot_vol_day_month = heatmap_data.pivot_table(
                values='volatility',
                index='day_of_week',
                columns='month',
                aggfunc='mean'
            )
            pivot_vol_day_month = pivot_vol_day_month.reindex([d for d in day_order if d in pivot_vol_day_month.index])
            
            fig_vol_day_month = go.Figure(data=go.Heatmap(
                z=pivot_vol_day_month.values,
                x=pivot_vol_day_month.columns,
                y=pivot_vol_day_month.index,
                colorscale='Oranges',
                text=pivot_vol_day_month.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Volatility")
            ))
            
            fig_vol_day_month.update_layout(
                title="Volatility by Day of Week and Month",
                xaxis_title="Month",
                yaxis_title="Day of Week",
                height=400
            )
            
            st.plotly_chart(fig_vol_day_month, use_container_width=True)
            
            # 5. Median Returns Heatmap
            st.markdown("---")
            st.markdown("#### 📊 Median Returns: Day vs Month")
            
            pivot_median = heatmap_data.pivot_table(
                values='returns',
                index='day_of_week',
                columns='month',
                aggfunc='median'
            )
            pivot_median = pivot_median.reindex([d for d in day_order if d in pivot_median.index])
            
            fig_median = go.Figure(data=go.Heatmap(
                z=pivot_median.values,
                x=pivot_median.columns,
                y=pivot_median.index,
                colorscale='PiYG',
                text=pivot_median.values,
                texttemplate='%{text:.2f}%',
                textfont={"size": 10},
                colorbar=dict(title="Median Returns %")
            ))
            
            fig_median.update_layout(
                title="Median Returns by Day and Month",
                xaxis_title="Month",
                yaxis_title="Day of Week",
                height=400
            )
            
            st.plotly_chart(fig_median, use_container_width=True)
            
            # 6. Variance Heatmap
            st.markdown("#### 📈 Variance (Risk): Day vs Month")
            
            pivot_variance = heatmap_data.pivot_table(
                values='returns',
                index='day_of_week',
                columns='month',
                aggfunc='var'
            )
            pivot_variance = pivot_variance.reindex([d for d in day_order if d in pivot_variance.index])
            
            fig_variance = go.Figure(data=go.Heatmap(
                z=pivot_variance.values,
                x=pivot_variance.columns,
                y=pivot_variance.index,
                colorscale='Reds',
                text=pivot_variance.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Variance")
            ))
            
            fig_variance.update_layout(
                title="Returns Variance (Risk) by Day and Month",
                xaxis_title="Month",
                yaxis_title="Day of Week",
                height=400
            )
            
            st.plotly_chart(fig_variance, use_container_width=True)
            
            # Summary insights from heatmaps
            st.markdown("---")
            st.markdown("### 📋 Heatmap Insights Summary")
            
            insights = []
            
            # Best day-month combo
            if not pivot_day_month.empty:
                best_combo = pivot_day_month.stack().idxmax()
                best_value = pivot_day_month.stack().max()
                insights.append(f"🏆 **Best Day-Month Combo**: {best_combo[0]} in {best_combo[1]} (Avg: +{best_value:.2f}%)")
            
            # Most volatile combo
            if not pivot_vol_day_month.empty:
                volatile_combo = pivot_vol_day_month.stack().idxmax()
                volatile_value = pivot_vol_day_month.stack().max()
                insights.append(f"⚡ **Most Volatile**: {volatile_combo[0]} in {volatile_combo[1]} (Vol: {volatile_value:.2f})")
            
            # Safest combo (lowest variance)
            if not pivot_variance.empty:
                safest_combo = pivot_variance.stack().idxmin()
                safest_value = pivot_variance.stack().min()
                insights.append(f"🛡️ **Lowest Risk**: {safest_combo[0]} in {safest_combo[1]} (Var: {safest_value:.2f})")
            
            # Current day/month prediction
            current_day = datetime.now().strftime('%A')
            current_month = datetime.now().strftime('%B')
            
            if not pivot_day_month.empty and current_day in pivot_day_month.index and current_month in pivot_day_month.columns:
                current_expected = pivot_day_month.loc[current_day, current_month]
                current_vol = pivot_vol_day_month.loc[current_day, current_month] if not pivot_vol_day_month.empty else 0
                
                direction = "📈 UP" if current_expected > 0 else "📉 DOWN"
                insights.append(f"📅 **Today's Pattern** ({current_day}, {current_month}): Expected {direction} by {abs(current_expected):.2f}% (Volatility: {current_vol:.2f})")
            
            for insight in insights:
                st.markdown(insight)
        
        with tab5:
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
                # Excel download - ensure datetime is timezone-naive
                excel_buffer = io.BytesIO()
                excel_data_export = data.copy()
                
                # Remove timezone from all datetime columns
                for col in excel_data_export.columns:
                    if pd.api.types.is_datetime64_any_dtype(excel_data_export[col]):
                        if hasattr(excel_data_export[col].dtype, 'tz') and excel_data_export[col].dt.tz is not None:
                            excel_data_export[col] = excel_data_export[col].dt.tz_localize(None)
                
                try:
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        excel_data_export.to_excel(writer, index=False, sheet_name='Data')
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📊 Download as Excel",
                        data=excel_data,
                        file_name=f"{ticker}_{timeframe}_{period}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Excel export error: {str(e)}")
                    st.info("You can still download as CSV above.")
            
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
