import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import pytz
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Professional Algo Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    .progress-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 10px 0;
    }
    .progress-fill {
        background-color: #4CAF50;
        height: 100%;
        transition: width 0.5s;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data persistence
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'ticker1_data' not in st.session_state:
    st.session_state.ticker1_data = {}
if 'ticker2_data' not in st.session_state:
    st.session_state.ticker2_data = {}
if 'ratio_data' not in st.session_state:
    st.session_state.ratio_data = {}
if 'current_selections' not in st.session_state:
    st.session_state.current_selections = {}

# Constants
TIMEZONE_MAPPING = {
    'NIFTY 50': 'Asia/Kolkata',
    'BANKNIFTY': 'Asia/Kolkata',
    'SENSEX': 'Asia/Kolkata',
    'USDINR': 'Asia/Kolkata',
    'GOLD': 'Asia/Kolkata',
    'SILVER': 'Asia/Kolkata',
    'MCX': 'Asia/Kolkata',
    'BTC': 'UTC',
    'ETH': 'UTC',
    'EURUSD': 'America/New_York',
    'GBPUSD': 'America/New_York'
}

ALLOWED_PERIODS = {
    '1m': ['1d', '5d'],
    '5m': ['1d', '1mo'],
    '15m': ['1mo'],
    '30m': ['1mo'],
    '1h': ['1mo'],
    '4h': ['1mo'],
    '1d': ['1mo', '1y', '2y', '5y'],
    '1wk': ['1mo', '1y', '5y', '10y', '20y'],
    '1mo': ['1y', '2y', '5y', '10y', '20y', '30y']
}

TICKER_MAPPING = {
    'NIFTY 50': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'USDINR': 'INR=X',
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'MCX': 'MCX.NS',
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X'
}

class AlgoTradingPlatform:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def get_yfinance_ticker(self, ticker_name):
        """Get yfinance ticker symbol"""
        return TICKER_MAPPING.get(ticker_name, ticker_name)
    
    def fetch_data_with_progress(self, ticker, interval, period, progress_bar, status_text):
        """Fetch data with progress tracking and rate limiting"""
        try:
            status_text.text(f"Fetching {ticker} ({interval}/{period})...")
            progress_bar.progress(0.3)
            
            # Add randomized delay to avoid rate limits
            time.sleep(np.random.uniform(1.0, 1.5))
            
            yf_ticker = self.get_yfinance_ticker(ticker)
            data = yf.download(
                yf_ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            progress_bar.progress(0.7)
            
            if data.empty:
                return None
            
            # Flatten multi-index DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = np.nan
            
            # Handle timezone
            if data.index.tz is None:
                # Determine source timezone
                source_tz = TIMEZONE_MAPPING.get(ticker, 'UTC')
                data.index = data.index.tz_localize(source_tz)
            
            # Convert to IST
            data.index = data.index.tz_convert(self.ist)
            
            # Sort index
            data = data.sort_index()
            
            progress_bar.progress(1.0)
            status_text.text(f"✓ {ticker} data fetched successfully")
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching {ticker}: {str(e)}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_divergence(self, price_series, indicator_series, lookback=20):
        """Calculate divergence between price and indicator"""
        divergences = []
        
        for i in range(lookback, len(price_series) - lookback):
            # Check for bearish divergence (price higher high, indicator lower high)
            if (price_series.iloc[i] > price_series.iloc[i-lookback:i].max() and
                indicator_series.iloc[i] < indicator_series.iloc[i-lookback:i].max()):
                divergences.append((price_series.index[i], 'bearish'))
            
            # Check for bullish divergence (price lower low, indicator higher low)
            if (price_series.iloc[i] < price_series.iloc[i-lookback:i].min() and
                indicator_series.iloc[i] > indicator_series.iloc[i-lookback:i].min()):
                divergences.append((price_series.index[i], 'bullish'))
        
        return divergences
    
    def calculate_future_movement(self, data, n_candles=20):
        """Calculate future price movement for next n candles"""
        movements = []
        
        for i in range(len(data) - n_candles):
            current_close = data['Close'].iloc[i]
            future_data = []
            
            for j in range(1, n_candles + 1):
                future_close = data['Close'].iloc[i + j]
                points_change = future_close - current_close
                percent_change = (points_change / current_close) * 100
                
                future_data.append({
                    'points': round(points_change, 2),
                    'percent': round(percent_change, 2)
                })
            
            movements.append(future_data)
        
        return movements
    
    def analyze_ratio(self, ratio_series, bins=10):
        """Analyze ratio distribution and significant levels"""
        # Remove NaN values
        clean_ratio = ratio_series.dropna()
        
        if len(clean_ratio) < bins:
            bins = len(clean_ratio)
        
        # Calculate histogram
        hist, bin_edges = np.histogram(clean_ratio, bins=bins)
        
        # Find significant levels (peaks in histogram)
        significant_levels = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                significant_levels.append(bin_edges[i])
        
        # Calculate statistics
        current_ratio = clean_ratio.iloc[-1]
        stats_dict = {
            'current': current_ratio,
            'mean': clean_ratio.mean(),
            'std': clean_ratio.std(),
            'min': clean_ratio.min(),
            'max': clean_ratio.max(),
            'q25': clean_ratio.quantile(0.25),
            'q75': clean_ratio.quantile(0.75),
            'significant_levels': significant_levels,
            'percentile': stats.percentileofscore(clean_ratio, current_ratio)
        }
        
        return stats_dict
    
    def generate_forecast(self, analysis_results, confidence_threshold=70):
        """Generate forecast based on multiple timeframe analysis"""
        bullish_count = 0
        bearish_count = 0
        sideways_count = 0
        
        for result in analysis_results:
            if 'forecast' in result:
                if result['forecast'] == 'bullish':
                    bullish_count += 1
                elif result['forecast'] == 'bearish':
                    bearish_count += 1
                else:
                    sideways_count += 1
        
        total = bullish_count + bearish_count + sideways_count
        if total == 0:
            return "Insufficient data", 0
        
        if bullish_count / total > 0.5:
            confidence = (bullish_count / total) * 100
            return "bullish", confidence
        elif bearish_count / total > 0.5:
            confidence = (bearish_count / total) * 100
            return "bearish", confidence
        else:
            confidence = (sideways_count / total) * 100
            return "sideways", confidence
    
    def calculate_entry_levels(self, current_price, forecast, volatility):
        """Calculate entry, stop loss, and target levels"""
        if forecast == 'bullish':
            entry = current_price * 0.995  # 0.5% below current
            sl = entry * 0.98  # 2% below entry
            target1 = entry * 1.02  # 2% above entry
            target2 = entry * 1.04  # 4% above entry
        elif forecast == 'bearish':
            entry = current_price * 1.005  # 0.5% above current
            sl = entry * 1.02  # 2% above entry
            target1 = entry * 0.98  # 2% below entry
            target2 = entry * 0.96  # 4% below entry
        else:
            entry = current_price
            sl = current_price * 0.99
            target1 = current_price * 1.01
            target2 = current_price * 1.02
        
        return {
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2),
            'risk_reward': round((target1 - entry) / (entry - sl), 2)
        }

# Initialize platform
platform = AlgoTradingPlatform()

# UI Header
st.markdown("<h1 class='main-header'>📈 Professional Algo Trading Platform</h1>", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    # Ticker selection
    ticker_options = ['NIFTY 50', 'BANKNIFTY', 'SENSEX', 'BTC', 'ETH', 
                      'USDINR', 'GOLD', 'SILVER', 'MCX', 'EURUSD', 'GBPUSD', 'Custom']
    
    col1, col2 = st.columns(2)
    with col1:
        ticker1 = st.selectbox(
            "Ticker 1",
            options=ticker_options,
            index=0,
            key="ticker1_select"
        )
        if ticker1 == 'Custom':
            ticker1 = st.text_input("Enter custom ticker 1:", value="^NSEI")
    
    with col2:
        ticker2 = st.selectbox(
            "Ticker 2",
            options=ticker_options,
            index=5,  # USDINR as default
            key="ticker2_select"
        )
        if ticker2 == 'Custom':
            ticker2 = st.text_input("Enter custom ticker 2:", value="INR=X")
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Timeframe",
        options=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo'],
        index=6  # Default to 1d
    )
    
    # Period selection based on timeframe
    available_periods = ALLOWED_PERIODS.get(timeframe, ['1mo'])
    period = st.selectbox(
        "Period",
        options=available_periods,
        index=0
    )
    
    # Other parameters
    col3, col4 = st.columns(2)
    with col3:
        bins = st.number_input("Number of Bins", min_value=5, max_value=50, value=10)
    
    with col4:
        n_candles = st.number_input("Next N Candles", min_value=1, max_value=30, value=15)
    
    # Fetch button
    fetch_button = st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True)
    
    # Progress section
    if fetch_button:
        progress_bar = st.progress(0)
        status_text = st.empty()

# Main content area
if fetch_button:
    # Clear previous data
    st.session_state.ticker1_data = {}
    st.session_state.ticker2_data = {}
    st.session_state.ratio_data = {}
    
    # Fetch data for both tickers
    ticker1_data = platform.fetch_data_with_progress(ticker1, timeframe, period, progress_bar, status_text)
    time.sleep(1)  # Rate limiting
    
    ticker2_data = platform.fetch_data_with_progress(ticker2, timeframe, period, progress_bar, status_text)
    
    if ticker1_data is not None and ticker2_data is not None:
        # Align timestamps
        common_index = ticker1_data.index.intersection(ticker2_data.index)
        ticker1_data = ticker1_data.loc[common_index]
        ticker2_data = ticker2_data.loc[common_index]
        
        # Calculate ratio
        ratio_series = ticker1_data['Close'] / ticker2_data['Close']
        
        # Store in session state
        st.session_state.ticker1_data = ticker1_data
        st.session_state.ticker2_data = ticker2_data
        st.session_state.ratio_data = ratio_series
        st.session_state.data_fetched = True
        st.session_state.current_selections = {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'timeframe': timeframe,
            'period': period,
            'bins': bins,
            'n_candles': n_candles
        }
        
        st.success("✅ Data fetched and processed successfully!")
    else:
        st.error("Failed to fetch data. Please check ticker symbols and try again.")

# Create tabs
if st.session_state.data_fetched:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ratio Charts", "🔄 RSI Divergence", "🧪 Backtesting", "📈 Statistics"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Ratio Analysis</h2>", unsafe_allow_html=True)
        
        # Get data from session state
        ticker1_data = st.session_state.ticker1_data
        ticker2_data = st.session_state.ticker2_data
        ratio_series = st.session_state.ratio_data
        selections = st.session_state.current_selections
        
        # Calculate future movements
        future_movements = platform.calculate_future_movement(ticker1_data, selections['n_candles'])
        
        # Create table with future movements
        st.markdown("### Future Price Movements")
        
        # Prepare table data
        table_data = []
        for i in range(min(20, len(future_movements))):
            row = {
                'Date': ticker1_data.index[i].strftime('%Y-%m-%d %H:%M'),
                'Close': round(ticker1_data['Close'].iloc[i], 2)
            }
            
            for j in range(selections['n_candles']):
                if j < len(future_movements[i]):
                    movement = future_movements[i][j]
                    row[f'Candle {j+1} (pts)'] = movement['points']
                    row[f'Candle {j+1} (%)'] = f"{movement['percent']}%"
            
            table_data.append(row)
        
        # Display table
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True)
        
        # Ratio analysis
        st.markdown("### Ratio Analysis Summary")
        
        ratio_stats = platform.analyze_ratio(ratio_series, selections['bins'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Ratio", f"{ratio_stats['current']:.4f}")
            st.metric("Mean Ratio", f"{ratio_stats['mean']:.4f}")
        
        with col2:
            st.metric("Min Ratio", f"{ratio_stats['min']:.4f}")
            st.metric("Max Ratio", f"{ratio_stats['max']:.4f}")
        
        with col3:
            st.metric("Std Deviation", f"{ratio_stats['std']:.4f}")
            st.metric("Current Percentile", f"{ratio_stats['percentile']:.1f}%")
        
        # Forecast and trading levels
        st.markdown("### Trading Forecast")
        
        # Determine forecast based on ratio position
        current_ratio = ratio_stats['current']
        if current_ratio < ratio_stats['q25']:
            forecast = "bullish"
            confidence = 75
            explanation = "Ratio is in lower quartile, suggesting potential upward movement for Ticker 1 relative to Ticker 2."
        elif current_ratio > ratio_stats['q75']:
            forecast = "bearish"
            confidence = 70
            explanation = "Ratio is in upper quartile, suggesting potential downward correction for Ticker 1 relative to Ticker 2."
        else:
            forecast = "sideways"
            confidence = 60
            explanation = "Ratio is in middle range, suggesting consolidation phase."
        
        # Calculate entry levels
        current_price = ticker1_data['Close'].iloc[-1]
        volatility = ticker1_data['Close'].pct_change().std() * 100
        entry_levels = platform.calculate_entry_levels(current_price, forecast, volatility)
        
        # Display forecast
        forecast_col1, forecast_col2 = st.columns(2)
        with forecast_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>📈 Forecast: {forecast.upper()}</h4>
                <p>Confidence: {confidence}%</p>
                <p>{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with forecast_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>🎯 Trading Levels</h4>
                <p>Entry: {entry_levels['entry']}</p>
                <p>Stop Loss: {entry_levels['sl']}</p>
                <p>Target 1: {entry_levels['target1']}</p>
                <p>Target 2: {entry_levels['target2']}</p>
                <p>Risk/Reward: {entry_levels['risk_reward']}:1</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create charts
        st.markdown("### Charts")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Ticker 1 Price', 'Ticker 2 Price', 'Ratio Chart'),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.4, 0.2]
        )
        
        # Ticker 1 price
        fig.add_trace(
            go.Candlestick(
                x=ticker1_data.index,
                open=ticker1_data['Open'],
                high=ticker1_data['High'],
                low=ticker1_data['Low'],
                close=ticker1_data['Close'],
                name=ticker1
            ),
            row=1, col=1
        )
        
        # Ticker 2 price
        fig.add_trace(
            go.Candlestick(
                x=ticker2_data.index,
                open=ticker2_data['Open'],
                high=ticker2_data['High'],
                low=ticker2_data['Low'],
                close=ticker2_data['Close'],
                name=ticker2
            ),
            row=2, col=1
        )
        
        # Ratio chart
        fig.add_trace(
            go.Scatter(
                x=ratio_series.index,
                y=ratio_series.values,
                mode='lines',
                name='Ratio',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # Add significant levels to ratio chart
        for level in ratio_stats['significant_levels']:
            fig.add_hline(
                y=level,
                line=dict(color='gray', dash='dash'),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume plots if available
        if 'Volume' in ticker1_data.columns and 'Volume' in ticker2_data.columns:
            fig_volume = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'{ticker1} Volume', f'{ticker2} Volume'),
                vertical_spacing=0.1
            )
            
            fig_volume.add_trace(
                go.Bar(
                    x=ticker1_data.index,
                    y=ticker1_data['Volume'],
                    name=f'{ticker1} Volume'
                ),
                row=1, col=1
            )
            
            fig_volume.add_trace(
                go.Bar(
                    x=ticker2_data.index,
                    y=ticker2_data['Volume'],
                    name=f'{ticker2} Volume'
                ),
                row=2, col=1
            )
            
            fig_volume.update_layout(height=400)
            st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab2:
        st.markdown("<h2 class='sub-header'>RSI Divergence Analysis</h2>", unsafe_allow_html=True)
        
        # Calculate RSI for both tickers
        rsi_ticker1 = platform.calculate_rsi(ticker1_data['Close'])
        rsi_ticker2 = platform.calculate_rsi(ticker2_data['Close'])
        
        # Find divergences
        divergences_t1 = platform.calculate_divergence(ticker1_data['Close'], rsi_ticker1)
        divergences_t2 = platform.calculate_divergence(ticker2_data['Close'], rsi_ticker2)
        
        # Display divergence points
        st.markdown("### Divergence Points")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{ticker1} Divergences:**")
            for dt, div_type in divergences_t1[-5:]:  # Show last 5
                st.write(f"- {dt.strftime('%Y-%m-%d %H:%M')}: {div_type}")
        
        with col2:
            st.markdown(f"**{ticker2} Divergences:**")
            for dt, div_type in divergences_t2[-5:]:
                st.write(f"- {dt.strftime('%Y-%m-%d %H:%M')}: {div_type}")
        
        # Create RSI divergence chart
        fig_rsi = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{ticker1} Price & RSI', f'{ticker2} Price & RSI'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )
        
        # Ticker 1 price and RSI
        fig_rsi.add_trace(
            go.Scatter(
                x=ticker1_data.index,
                y=ticker1_data['Close'],
                mode='lines',
                name=f'{ticker1} Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig_rsi.add_trace(
            go.Scatter(
                x=rsi_ticker1.index,
                y=rsi_ticker1.values,
                mode='lines',
                name=f'{ticker1} RSI',
                line=dict(color='orange', width=1),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Ticker 2 price and RSI
        fig_rsi.add_trace(
            go.Scatter(
                x=ticker2_data.index,
                y=ticker2_data['Close'],
                mode='lines',
                name=f'{ticker2} Price',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig_rsi.add_trace(
            go.Scatter(
                x=rsi_ticker2.index,
                y=rsi_ticker2.values,
                mode='lines',
                name=f'{ticker2} RSI',
                line=dict(color='red', width=1),
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        # Update layout with secondary y-axis for RSI
        fig_rsi.update_layout(
            height=600,
            showlegend=True,
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            yaxis4=dict(
                title="RSI",
                overlaying="y3",
                side="right",
                range=[0, 100]
            )
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Backtesting Results</h2>", unsafe_allow_html=True)
        
        # Simple backtesting simulation
        st.info("Backtesting engine coming soon...")
        
    with tab4:
        st.markdown("<h2 class='sub-header'>Statistical Analysis</h2>", unsafe_allow_html=True)
        
        # Prepare statistics data
        returns = ticker1_data['Close'].pct_change().dropna() * 100
        returns_abs = abs(returns)
        
        # Create statistics table
        stats_data = {
            'Metric': ['Mean Return', 'Std Deviation', 'Min Return', 'Max Return', 
                      'Sharpe Ratio', 'Win Rate', 'Avg Win', 'Avg Loss'],
            'Value': [
                f"{returns.mean():.2f}%",
                f"{returns.std():.2f}%",
                f"{returns.min():.2f}%",
                f"{returns.max():.2f}%",
                f"{returns.mean()/returns.std():.2f}" if returns.std() > 0 else "N/A",
                f"{(returns > 0).sum()/len(returns)*100:.1f}%",
                f"{returns[returns > 0].mean():.2f}%",
                f"{returns[returns < 0].mean():.2f}%"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Day of week analysis
        ticker1_data['Day'] = ticker1_data.index.day_name()
        ticker1_data['Return'] = returns
        ticker1_data['Abs_Return'] = returns_abs
        
        day_stats = ticker1_data.groupby('Day').agg({
            'Return': ['mean', 'std', 'count'],
            'Abs_Return': 'mean'
        }).round(2)
        
        st.markdown("### Day of Week Analysis")
        st.dataframe(day_stats, use_container_width=True)

else:
    # Welcome screen when no data fetched
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>Welcome to Professional Algo Trading Platform</h2>
        <p style='font-size: 1.2rem; color: #666; margin-top: 20px;'>
            Select your instruments, timeframe, and parameters in the sidebar,<br>
            then click <strong>"Fetch & Analyze"</strong> to begin.
        </p>
        <div style='margin-top: 40px;'>
            <h4>📋 Quick Start Guide:</h4>
            <ol style='text-align: left; display: inline-block;'>
                <li>Select Ticker 1 & 2 from dropdown</li>
                <li>Choose Timeframe and Period</li>
                <li>Set number of bins and future candles to analyze</li>
                <li>Click "Fetch & Analyze" to process</li>
                <li>Explore results across 4 different tabs</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
