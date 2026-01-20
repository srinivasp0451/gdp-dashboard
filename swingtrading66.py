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

# Custom CSS
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
    .forecast-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
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
    .bin-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .bin-table th, .bin-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .bin-table th {
        background-color: #1E88E5;
        color: white;
    }
    .bin-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .bin-table tr:hover {
        background-color: #f5f5f5;
    }
    .current-bin {
        background-color: #ffeb3b !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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
if 'all_analysis' not in st.session_state:
    st.session_state.all_analysis = {}

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
    '1wk': ['1mo', '1y',  '5y', '10y', '20y'],
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
        if ticker_name in TICKER_MAPPING:
            return TICKER_MAPPING[ticker_name]
        else:
            # For custom tickers
            return ticker_name
    
    def fetch_data_with_progress(self, ticker, interval, period, progress_bar, status_text, progress_value):
        """Fetch data with progress tracking"""
        try:
            status_text.text(f"Fetching {ticker} ({interval}/{period})...")
            progress_bar.progress(progress_value + 0.1)
            
            # Add randomized delay
            time.sleep(np.random.uniform(1.0, 1.5))
            
            yf_ticker = self.get_yfinance_ticker(ticker)
            data = yf.download(
                yf_ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                return None
            
            # Flatten multi-index DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]
                # Rename common columns
                rename_dict = {}
                for col in data.columns:
                    if 'Close' in col:
                        rename_dict[col] = 'Close'
                    elif 'Open' in col:
                        rename_dict[col] = 'Open'
                    elif 'High' in col:
                        rename_dict[col] = 'High'
                    elif 'Low' in col:
                        rename_dict[col] = 'Low'
                    elif 'Volume' in col:
                        rename_dict[col] = 'Volume'
                data = data.rename(columns=rename_dict)
            
            # Ensure required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = np.nan
            if 'Volume' not in data.columns:
                data['Volume'] = np.nan
            
            # Handle timezone
            if data.index.tz is None:
                source_tz = TIMEZONE_MAPPING.get(ticker, 'UTC')
                data.index = data.index.tz_localize(source_tz)
            
            # Convert to IST
            data.index = data.index.tz_convert(self.ist)
            
            # Sort index
            data = data.sort_index()
            
            progress_bar.progress(progress_value + 0.3)
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
    
    def create_bins_with_ranges(self, ratio_series, num_bins):
        """Create bins with ranges for ratio analysis"""
        if len(ratio_series) < 2:
            return [], []
        
        # Remove NaN values
        clean_ratio = ratio_series.dropna()
        
        if len(clean_ratio) < 2:
            return [], []
        
        # Create bins
        min_val = clean_ratio.min()
        max_val = clean_ratio.max()
        
        # Create bin edges
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        
        # Create bin ranges
        bin_ranges = []
        bin_counts = []
        
        for i in range(num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            bin_ranges.append(f"{lower:.6f} - {upper:.6f}")
            
            # Count values in this bin
            if i == num_bins - 1:  # Last bin includes upper edge
                count = ((clean_ratio >= lower) & (clean_ratio <= upper)).sum()
            else:
                count = ((clean_ratio >= lower) & (clean_ratio < upper)).sum()
            bin_counts.append(count)
        
        return bin_ranges, bin_counts, bin_edges
    
    def analyze_ratio_bins(self, ratio_series, num_bins):
        """Analyze ratio distribution in bins"""
        bin_ranges, bin_counts, bin_edges = self.create_bins_with_ranges(ratio_series, num_bins)
        
        if not bin_ranges:
            return None
        
        # Calculate statistics
        clean_ratio = ratio_series.dropna()
        current_ratio = clean_ratio.iloc[-1]
        
        # Find which bin contains current ratio
        current_bin_idx = -1
        for i, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if i == len(bin_edges) - 2:  # Last bin
                if lower <= current_ratio <= upper:
                    current_bin_idx = i
                    break
            else:
                if lower <= current_ratio < upper:
                    current_bin_idx = i
                    break
        
        # Calculate bin percentages
        total_count = sum(bin_counts)
        bin_percentages = [(count / total_count * 100) for count in bin_counts]
        
        # Find significant bins (bins with highest counts)
        significant_bins = []
        if bin_counts:
            max_count = max(bin_counts)
            threshold = max_count * 0.7  # 70% of max count
            for i, count in enumerate(bin_counts):
                if count >= threshold:
                    significant_bins.append({
                        'range': bin_ranges[i],
                        'count': count,
                        'percentage': bin_percentages[i]
                    })
        
        # Calculate historical impact
        historical_impact = []
        if len(bin_edges) > 1:
            midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            for i, midpoint in enumerate(midpoints):
                if bin_counts[i] > 0:
                    # Simple impact score based on position
                    position_score = i / len(midpoints)  # Normalized position
                    if position_score < 0.33:
                        impact = "Bullish rally zone"
                    elif position_score > 0.66:
                        impact = "Bearish decline zone"
                    else:
                        impact = "Neutral consolidation"
                    historical_impact.append({
                        'range': bin_ranges[i],
                        'impact': impact,
                        'midpoint': midpoint
                    })
        
        return {
            'bin_ranges': bin_ranges,
            'bin_counts': bin_counts,
            'bin_percentages': bin_percentages,
            'bin_edges': bin_edges,
            'current_ratio': current_ratio,
            'current_bin_idx': current_bin_idx,
            'current_bin_range': bin_ranges[current_bin_idx] if current_bin_idx >= 0 else "N/A",
            'significant_bins': significant_bins,
            'historical_impact': historical_impact,
            'min_value': clean_ratio.min(),
            'max_value': clean_ratio.max(),
            'mean_value': clean_ratio.mean(),
            'std_value': clean_ratio.std(),
            'percentile': stats.percentileofscore(clean_ratio, current_ratio)
        }
    
    def generate_forecast(self, analysis_results):
        """Generate forecast based on multiple analysis"""
        if not analysis_results:
            return "neutral", 0, []
        
        forecasts = []
        confidences = []
        
        for result in analysis_results:
            if 'forecast' in result:
                forecasts.append(result['forecast'])
                confidences.append(result.get('confidence', 50))
        
        if not forecasts:
            return "neutral", 0, []
        
        # Count occurrences
        forecast_counts = {}
        for forecast in forecasts:
            forecast_counts[forecast] = forecast_counts.get(forecast, 0) + 1
        
        # Determine majority forecast
        total = len(forecasts)
        majority_forecast = max(forecast_counts, key=forecast_counts.get)
        confidence = (forecast_counts[majority_forecast] / total) * 100
        
        # Calculate average confidence for majority forecast
        majority_confidences = [confidences[i] for i in range(len(forecasts)) if forecasts[i] == majority_forecast]
        avg_confidence = np.mean(majority_confidences) if majority_confidences else confidence
        
        return majority_forecast, avg_confidence, forecast_counts
    
    def calculate_trading_levels(self, current_price, forecast, volatility, ratio_position):
        """Calculate trading levels based on forecast"""
        if forecast == 'bullish':
            if ratio_position < 0.3:  # Very bullish
                entry = current_price * 0.99
                sl = entry * 0.97
                targets = [
                    entry * 1.03,
                    entry * 1.06,
                    entry * 1.10
                ]
                risk_reward = 3.0
            else:
                entry = current_price * 0.995
                sl = entry * 0.98
                targets = [
                    entry * 1.02,
                    entry * 1.04,
                    entry * 1.07
                ]
                risk_reward = 2.0
        elif forecast == 'bearish':
            if ratio_position > 0.7:  # Very bearish
                entry = current_price * 1.01
                sl = entry * 1.03
                targets = [
                    entry * 0.97,
                    entry * 0.94,
                    entry * 0.90
                ]
                risk_reward = 3.0
            else:
                entry = current_price * 1.005
                sl = entry * 1.02
                targets = [
                    entry * 0.98,
                    entry * 0.96,
                    entry * 0.93
                ]
                risk_reward = 2.0
        else:  # sideways
            entry = current_price
            sl = current_price * 0.985
            targets = [
                current_price * 1.015,
                current_price * 1.03,
                current_price * 1.05
            ]
            risk_reward = 1.5
        
        return {
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'targets': [round(t, 2) for t in targets],
            'risk_reward': round(risk_reward, 2),
            'points_risk': round(abs(entry - sl), 2),
            'points_reward1': round(abs(targets[0] - entry), 2),
            'points_reward2': round(abs(targets[1] - entry), 2),
            'points_reward3': round(abs(targets[2] - entry), 2)
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
            index=5,
            key="ticker2_select"
        )
        if ticker2 == 'Custom':
            ticker2 = st.text_input("Enter custom ticker 2:", value="INR=X")
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Timeframe",
        options=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo'],
        index=6
    )
    
    # Period selection
    available_periods = ALLOWED_PERIODS.get(timeframe, ['1mo'])
    period = st.selectbox(
        "Period",
        options=available_periods,
        index=0
    )
    
    # Other parameters
    col3, col4 = st.columns(2)
    with col3:
        bins = st.number_input("Number of Bins", min_value=3, max_value=20, value=10)
    
    with col4:
        n_candles = st.number_input("Next N Candles", min_value=1, max_value=30, value=15)
    
    # Fetch button
    fetch_button = st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True)
    
    # Progress section
    if fetch_button:
        progress_bar = st.progress(0)
        status_text = st.empty()

# Main content
if fetch_button:
    # Clear previous data
    st.session_state.ticker1_data = {}
    st.session_state.ticker2_data = {}
    st.session_state.ratio_data = {}
    st.session_state.all_analysis = {}
    
    # Fetch data
    with st.spinner("Fetching data..."):
        # Initialize progress
        progress = 0
        status_text.text("Starting data fetch...")
        progress_bar.progress(progress)
        
        # Fetch ticker1 data
        ticker1_data = platform.fetch_data_with_progress(ticker1, timeframe, period, progress_bar, status_text, progress)
        progress = 0.4
        time.sleep(1)
        
        # Fetch ticker2 data
        ticker2_data = platform.fetch_data_with_progress(ticker2, timeframe, period, progress_bar, status_text, progress)
        progress = 0.8
        
        if ticker1_data is not None and ticker2_data is not None:
            # Align timestamps
            common_index = ticker1_data.index.intersection(ticker2_data.index)
            if len(common_index) == 0:
                st.error("No common timestamps found between the two tickers. Please check timezones.")
            else:
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
                
                progress_bar.progress(1.0)
                status_text.text("✅ Data fetched and processed successfully!")
                st.success("Data ready for analysis!")
        else:
            st.error("Failed to fetch data. Please check ticker symbols and try again.")

# Create tabs if data is fetched
if st.session_state.data_fetched:
    # Get data
    ticker1_data = st.session_state.ticker1_data
    ticker2_data = st.session_state.ticker2_data
    ratio_series = st.session_state.ratio_data
    selections = st.session_state.current_selections
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ratio Charts", "🔄 RSI Divergence", "🧪 Backtesting", "📈 Statistics"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Ratio Analysis</h2>", unsafe_allow_html=True)
        
        # Calculate future movements
        with st.spinner("Calculating future movements..."):
            future_movements = platform.calculate_future_movement(ticker1_data, selections['n_candles'])
        
        # Display future movements table
        st.markdown("### 📈 Future Price Movements")
        
        table_data = []
        display_rows = min(20, len(future_movements))
        
        for i in range(display_rows):
            row = {
                'From DateTime': ticker1_data.index[i].strftime('%Y-%m-%d %H:%M IST'),
                'Close Price': round(ticker1_data['Close'].iloc[i], 2),
                'Ratio': round(ratio_series.iloc[i], 6)
            }
            
            # Add future candle movements
            for j in range(min(selections['n_candles'], 10)):  # Show first 10 candles
                if j < len(future_movements[i]):
                    movement = future_movements[i][j]
                    row[f'Candle {j+1}'] = f"{movement['points']:+} pts ({movement['percent']:+.2f}%)"
            
            table_data.append(row)
        
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, height=400)
        
        # Ratio Bins Analysis
        st.markdown("### 📊 Ratio Bins Analysis")
        
        bin_analysis = platform.analyze_ratio_bins(ratio_series, selections['bins'])
        
        if bin_analysis:
            # Display bins table
            st.markdown("#### Bins Distribution")
            
            bins_html = """
            <table class='bin-table'>
                <tr>
                    <th>Bin Range</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Historical Impact</th>
                </tr>
            """
            
            for i in range(len(bin_analysis['bin_ranges'])):
                bin_range = bin_analysis['bin_ranges'][i]
                count = bin_analysis['bin_counts'][i]
                percentage = f"{bin_analysis['bin_percentages'][i]:.1f}%"
                
                # Find historical impact for this bin
                impact = "Neutral"
                for hist_impact in bin_analysis['historical_impact']:
                    if hist_impact['range'] == bin_range:
                        impact = hist_impact['impact']
                        break
                
                # Highlight current bin
                row_class = "class='current-bin'" if i == bin_analysis['current_bin_idx'] else ""
                
                bins_html += f"""
                <tr {row_class}>
                    <td>{bin_range}</td>
                    <td>{count}</td>
                    <td>{percentage}</td>
                    <td>{impact}</td>
                </tr>
                """
            
            bins_html += "</table>"
            st.markdown(bins_html, unsafe_allow_html=True)
            
            # Current bin info
            st.markdown(f"""
            <div class='metric-card'>
                <h4>📍 Current Ratio Position</h4>
                <p><strong>Current Ratio:</strong> {bin_analysis['current_ratio']:.6f}</p>
                <p><strong>Current Bin:</strong> {bin_analysis['current_bin_range']}</p>
                <p><strong>Percentile:</strong> {bin_analysis['percentile']:.1f}%</p>
                <p><strong>Distance from Min:</strong> {((bin_analysis['current_ratio'] - bin_analysis['min_value']) / (bin_analysis['max_value'] - bin_analysis['min_value']) * 100):.1f}%</p>
                <p><strong>Distance from Max:</strong> {((bin_analysis['max_value'] - bin_analysis['current_ratio']) / (bin_analysis['max_value'] - bin_analysis['min_value']) * 100):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Historical significant levels
            st.markdown("#### 📜 Historical Significant Levels")
            
            if bin_analysis['significant_bins']:
                for sig_bin in bin_analysis['significant_bins'][:5]:  # Show top 5
                    st.write(f"**Range:** {sig_bin['range']} - **Frequency:** {sig_bin['count']} candles ({sig_bin['percentage']:.1f}%)")
            
            # Generate forecast
            current_price = ticker1_data['Close'].iloc[-1]
            current_ratio = bin_analysis['current_ratio']
            ratio_position = (current_ratio - bin_analysis['min_value']) / (bin_analysis['max_value'] - bin_analysis['min_value'])
            
            if ratio_position < 0.25:
                forecast = "bullish"
                confidence = 80 - (ratio_position * 100)
                explanation = "Ratio is in lower historical range (bullish zone). Historically, this level has led to upward movements."
            elif ratio_position > 0.75:
                forecast = "bearish"
                confidence = 80 - ((1 - ratio_position) * 100)
                explanation = "Ratio is in upper historical range (bearish zone). Historically, this level has led to downward corrections."
            else:
                forecast = "sideways"
                confidence = 60
                explanation = "Ratio is in middle range. Expect consolidation with possible breakout based on momentum."
            
            # Calculate trading levels
            volatility = ticker1_data['Close'].pct_change().std() * 100
            trading_levels = platform.calculate_trading_levels(current_price, forecast, volatility, ratio_position)
            
            # Display forecast and trading levels
            col1, col2 = st.columns(2)
            
            with col1:
                forecast_color = "green" if forecast == "bullish" else "red" if forecast == "bearish" else "orange"
                st.markdown(f"""
                <div class='forecast-card'>
                    <h3>🎯 Forecast: <span style='color:{forecast_color}'>{forecast.upper()}</span></h3>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p><strong>Explanation:</strong> {explanation}</p>
                    <p><strong>Current Ratio Position:</strong> {ratio_position:.1%} from minimum</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>💰 Trading Plan</h4>
                    <p><strong>Entry:</strong> {trading_levels['entry']}</p>
                    <p><strong>Stop Loss:</strong> {trading_levels['sl']} (Risk: {trading_levels['points_risk']} pts)</p>
                    <p><strong>Target 1:</strong> {trading_levels['targets'][0]} (Reward: {trading_levels['points_reward1']} pts)</p>
                    <p><strong>Target 2:</strong> {trading_levels['targets'][1]} (Reward: {trading_levels['points_reward2']} pts)</p>
                    <p><strong>Target 3:</strong> {trading_levels['targets'][2]} (Reward: {trading_levels['points_reward3']} pts)</p>
                    <p><strong>Risk/Reward:</strong> 1:{trading_levels['risk_reward']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Create charts
            st.markdown("### 📉 Charts")
            
            # Price and Ratio Chart
            fig1 = make_subplots(
                rows=3, cols=1,
                subplot_titles=(f'{ticker1} Price', f'{ticker2} Price', f'{ticker1}/{ticker2} Ratio'),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.4, 0.2]
            )
            
            # Ticker 1
            fig1.add_trace(
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
            
            # Ticker 2
            fig1.add_trace(
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
            
            # Ratio
            fig1.add_trace(
                go.Scatter(
                    x=ratio_series.index,
                    y=ratio_series.values,
                    mode='lines',
                    name='Ratio',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # Add bin ranges as horizontal lines
            for edge in bin_analysis['bin_edges']:
                fig1.add_hline(
                    y=edge,
                    line=dict(color='gray', dash='dot', width=1),
                    row=3, col=1
                )
            
            # Highlight current bin
            if bin_analysis['current_bin_idx'] >= 0:
                lower = bin_analysis['bin_edges'][bin_analysis['current_bin_idx']]
                upper = bin_analysis['bin_edges'][bin_analysis['current_bin_idx'] + 1]
                fig1.add_hrect(
                    y0=lower, y1=upper,
                    fillcolor="yellow", opacity=0.2,
                    line_width=0, row=3, col=1
                )
            
            fig1.update_layout(
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                title_text=f"Price and Ratio Analysis - {timeframe} timeframe"
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Volume charts if available
            if 'Volume' in ticker1_data.columns and ticker1_data['Volume'].notna().any():
                fig2 = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(f'{ticker1} Volume', f'{ticker2} Volume')
                )
                
                fig2.add_trace(
                    go.Bar(
                        x=ticker1_data.index,
                        y=ticker1_data['Volume'],
                        name=f'{ticker1} Volume'
                    ),
                    row=1, col=1
                )
                
                if 'Volume' in ticker2_data.columns and ticker2_data['Volume'].notna().any():
                    fig2.add_trace(
                        go.Bar(
                            x=ticker2_data.index,
                            y=ticker2_data['Volume'],
                            name=f'{ticker2} Volume'
                        ),
                        row=2, col=1
                    )
                
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        else:
            st.warning("Insufficient data for bin analysis.")
    
    with tab2:
        st.markdown("<h2 class='sub-header'>RSI Divergence Analysis</h2>", unsafe_allow_html=True)
        
        # Calculate RSI
        rsi_period = 14
        rsi_ticker1 = platform.calculate_rsi(ticker1_data['Close'], rsi_period)
        rsi_ticker2 = platform.calculate_rsi(ticker2_data['Close'], rsi_period)
        
        # Find divergences
        divergences_t1 = platform.calculate_divergence(ticker1_data['Close'], rsi_ticker1)
        divergences_t2 = platform.calculate_divergence(ticker2_data['Close'], rsi_ticker2)
        
        # Display divergence points
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {ticker1} Divergences")
            if divergences_t1:
                df_div1 = pd.DataFrame(divergences_t1, columns=['DateTime', 'Type'])
                df_div1['DateTime'] = df_div1['DateTime'].dt.strftime('%Y-%m-%d %H:%M IST')
                st.dataframe(df_div1.tail(10), use_container_width=True)
            else:
                st.info("No divergences found")
        
        with col2:
            st.markdown(f"#### {ticker2} Divergences")
            if divergences_t2:
                df_div2 = pd.DataFrame(divergences_t2, columns=['DateTime', 'Type'])
                df_div2['DateTime'] = df_div2['DateTime'].dt.strftime('%Y-%m-%d %H:%M IST')
                st.dataframe(df_div2.tail(10), use_container_width=True)
            else:
                st.info("No divergences found")
        
        # RSI Chart
        fig_rsi = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{ticker1} Price',
                f'{ticker1} RSI',
                f'{ticker2} Price',
                f'{ticker2} RSI'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Ticker 1 Price
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
        
        # Ticker 1 RSI
        fig_rsi.add_trace(
            go.Scatter(
                x=rsi_ticker1.index,
                y=rsi_ticker1.values,
                mode='lines',
                name=f'{ticker1} RSI',
                line=dict(color='orange', width=2)
            ),
            row=1, col=2
        )
        
        # Add RSI levels
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
        
        # Ticker 2 Price
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
        
        # Ticker 2 RSI
        fig_rsi.add_trace(
            go.Scatter(
                x=rsi_ticker2.index,
                y=rsi_ticker2.values,
                mode='lines',
                name=f'{ticker2} RSI',
                line=dict(color='red', width=2)
            ),
            row=2, col=2
        )
        
        # Add RSI levels
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=2)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=2)
        
        fig_rsi.update_layout(
            height=800,
            showlegend=True,
            title_text="RSI Divergence Analysis"
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Forecast based on RSI
        current_rsi1 = rsi_ticker1.iloc[-1]
        current_rsi2 = rsi_ticker2.iloc[-1]
        
        if current_rsi1 < 30 and current_rsi2 < 30:
            rsi_forecast = "bullish"
            rsi_confidence = 75
            rsi_explanation = "Both tickers in oversold territory (RSI < 30), suggesting potential upward reversal."
        elif current_rsi1 > 70 and current_rsi2 > 70:
            rsi_forecast = "bearish"
            rsi_confidence = 75
            rsi_explanation = "Both tickers in overbought territory (RSI > 70), suggesting potential downward correction."
        elif abs(current_rsi1 - 50) < 10 and abs(current_rsi2 - 50) < 10:
            rsi_forecast = "sideways"
            rsi_confidence = 60
            rsi_explanation = "Both RSIs near 50, indicating neutral momentum and possible consolidation."
        else:
            rsi_forecast = "mixed"
            rsi_confidence = 50
            rsi_explanation = "Mixed RSI signals between tickers, requiring confirmation from other indicators."
        
        st.markdown(f"""
        <div class='metric-card'>
            <h4>📊 RSI Analysis</h4>
            <p><strong>{ticker1} RSI:</strong> {current_rsi1:.1f} ({'Oversold' if current_rsi1 < 30 else 'Overbought' if current_rsi1 > 70 else 'Neutral'})</p>
            <p><strong>{ticker2} RSI:</strong> {current_rsi2:.1f} ({'Oversold' if current_rsi2 < 30 else 'Overbought' if current_rsi2 > 70 else 'Neutral'})</p>
            <p><strong>Forecast:</strong> {rsi_forecast.upper()} (Confidence: {rsi_confidence}%)</p>
            <p><strong>Explanation:</strong> {rsi_explanation}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Backtesting Results</h2>", unsafe_allow_html=True)
        
        # Simple backtesting simulation
        st.info("""
        **Backtesting Module Under Development**
        
        This module will implement:
        - Historical strategy testing
        - Entry/Exit signal generation
        - P&L calculation with commissions
        - Risk management metrics
        - Performance statistics
        """)
        
        # Placeholder for future implementation
        st.write("Coming soon: Complete backtesting engine with walk-forward optimization")
    
    with tab4:
        st.markdown("<h2 class='sub-header'>Statistical Analysis</h2>", unsafe_allow_html=True)
        
        # Prepare data for statistics
        returns = ticker1_data['Close'].pct_change().dropna() * 100
        returns_abs = abs(returns)
        
        # Add day of week
        ticker1_data['Day'] = ticker1_data.index.day_name()
        ticker1_data['Hour'] = ticker1_data.index.hour
        ticker1_data['Return'] = returns
        ticker1_data['Abs_Return'] = returns_abs
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Return", f"{returns.mean():.2f}%")
            st.metric("Standard Deviation", f"{returns.std():.2f}%")
            st.metric("Sharpe Ratio", f"{returns.mean()/returns.std():.2f}" if returns.std() > 0 else "N/A")
        
        with col2:
            st.metric("Min Return", f"{returns.min():.2f}%")
            st.metric("Max Return", f"{returns.max():.2f}%")
            st.metric("Win Rate", f"{(returns > 0).sum()/len(returns)*100:.1f}%")
        
        with col3:
            st.metric("Avg Win", f"{returns[returns > 0].mean():.2f}%")
            st.metric("Avg Loss", f"{returns[returns < 0].mean():.2f}%")
            st.metric("Profit Factor", f"{-returns[returns > 0].sum()/returns[returns < 0].sum():.2f}" if returns[returns < 0].sum() != 0 else "N/A")
        
        # Day of week analysis
        st.markdown("#### 📅 Day of Week Analysis")
        
        day_stats = ticker1_data.groupby('Day').agg({
            'Return': ['mean', 'std', 'count'],
            'Abs_Return': 'mean'
        }).round(2)
        
        day_stats.columns = ['Avg Return %', 'Std Dev %', 'Count', 'Avg Abs Return %']
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_stats = day_stats.reindex([day for day in day_order if day in day_stats.index])
        
        st.dataframe(day_stats, use_container_width=True)
        
        # Hourly analysis
        st.markdown("#### ⏰ Hourly Analysis")
        
        if timeframe in ['1m', '5m', '15m', '30m', '1h']:
            hour_stats = ticker1_data.groupby('Hour').agg({
                'Return': ['mean', 'std', 'count'],
                'Abs_Return': 'mean'
            }).round(2)
            
            hour_stats.columns = ['Avg Return %', 'Std Dev %', 'Count', 'Avg Abs Return %']
            st.dataframe(hour_stats, use_container_width=True)
        
        # Create return distribution chart
        st.markdown("#### 📊 Return Distribution")
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color='#1E88E5'
        ))
        
        fig_dist.update_layout(
            title="Return Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>Welcome to Professional Algo Trading Platform</h2>
        <p style='font-size: 1.2rem; color: #666; margin-top: 20px;'>
            Select your instruments, timeframe, and parameters in the sidebar,<br>
            then click <strong>"Fetch & Analyze"</strong> to begin.
        </p>
        <div style='margin-top: 40px; background: #f8f9fa; padding: 20px; border-radius: 10px;'>
            <h4>📋 Default Settings:</h4>
            <ul style='text-align: left; display: inline-block;'>
                <li><strong>Ticker 1:</strong> NIFTY 50 (Indian Index)</li>
                <li><strong>Ticker 2:</strong> USD/INR (Forex)</li>
                <li><strong>Timeframe:</strong> 1d (Daily)</li>
                <li><strong>Period:</strong> 1mo (1 Month)</li>
                <li><strong>Bins:</strong> 10 (for ratio analysis)</li>
                <li><strong>Next N Candles:</strong> 15 (for future movement analysis)</li>
            </ul>
        </div>
        <div style='margin-top: 30px;'>
            <h4>🎯 Features Available:</h4>
            <div style='display: flex; justify-content: center; gap: 20px; margin-top: 20px;'>
                <div style='padding: 15px; background: #e3f2fd; border-radius: 8px;'>
                    <h5>📊 Ratio Charts</h5>
                    <p>Bin-based ratio analysis</p>
                </div>
                <div style='padding: 15px; background: #f3e5f5; border-radius: 8px;'>
                    <h5>🔄 RSI Divergence</h5>
                    <p>Momentum analysis</p>
                </div>
                <div style='padding: 15px; background: #e8f5e8; border-radius: 8px;'>
                    <h5>📈 Statistics</h5>
                    <p>Performance metrics</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
