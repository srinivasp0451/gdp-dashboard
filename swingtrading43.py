import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
from scipy import stats

# Page configuration
st.set_page_config(page_title="Pairs Trading Strategy", layout="wide", initial_sidebar_state="expanded")

# IST Timezone
IST = pytz.timezone('Asia/Kolkata')

# Asset mappings
ASSET_MAPPINGS = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC-USD': 'BTC-USD',
    'ETH-USD': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'JPY=X',
}

TIMEFRAME_MAPPING = {
    '1m': '1m', '3m': '3m', '5m': '5m', '10m': '10m', 
    '15m': '15m', '30m': '30m', '1h': '1h', '2h': '2h', 
    '4h': '4h', '1d': '1d'
}

PERIOD_MAPPING = {
    '1d': '1d', '5d': '5d', '7d': '7d', '1mo': '1mo',
    '3mo': '3mo', '6mo': '6mo', '1y': '1y', '2y': '2y',
    '3y': '3y', '5y': '5y', '6y': '6y', '10y': '10y',
    '15y': '15y', '20y': '20y', '25y': '25y', '30y': '30y'
}

def convert_to_ist(df):
    """Convert DataFrame index to IST timezone"""
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
    except:
        pass
    return df

def fetch_data(ticker, period, interval, delay=2):
    """Fetch data with rate limiting"""
    try:
        time.sleep(delay)
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        data = convert_to_ist(data)
        return data
    except Exception as e:
        st.error(f"Error fetching {ticker}: {str(e)}")
        return None

def calculate_spread(asset1_prices, asset2_prices):
    """Calculate spread and hedge ratio using linear regression"""
    # Ensure we're working with Series
    if isinstance(asset1_prices, pd.DataFrame):
        asset1_prices = asset1_prices.squeeze()
    if isinstance(asset2_prices, pd.DataFrame):
        asset2_prices = asset2_prices.squeeze()
    
    # Remove any NaN values
    valid_idx = ~(np.isnan(asset1_prices) | np.isnan(asset2_prices))
    x = asset2_prices[valid_idx].values.reshape(-1, 1)
    y = asset1_prices[valid_idx].values
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(asset2_prices[valid_idx], asset1_prices[valid_idx])
    
    # Calculate spread
    spread = asset1_prices - (slope * asset2_prices + intercept)
    
    return spread, slope, intercept, r_value**2

def calculate_zscore(spread, window=20):
    """Calculate rolling z-score"""
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    # Ensure zscore is a Series, not DataFrame
    if isinstance(spread, pd.DataFrame):
        spread = spread.squeeze()
    if isinstance(mean, pd.DataFrame):
        mean = mean.squeeze()
    if isinstance(std, pd.DataFrame):
        std = std.squeeze()
    zscore = (spread - mean) / std
    return zscore, mean, std

def generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """Generate trading signals based on z-score"""
    signals = pd.DataFrame(index=zscore.index)
    signals['zscore'] = zscore
    signals['position'] = 0
    
    # Entry signals
    signals.loc[zscore > entry_threshold, 'position'] = -1  # Short spread
    signals.loc[zscore < -entry_threshold, 'position'] = 1   # Long spread
    
    # Exit signals
    signals.loc[(zscore < exit_threshold) & (zscore > -exit_threshold), 'position'] = 0
    
    # Generate trade signals
    signals['signal'] = signals['position'].diff()
    
    return signals

def calculate_trade_metrics(asset1_price, asset2_price, hedge_ratio, current_zscore, entry_zscore, position_type):
    """Calculate entry, target, and stop loss levels"""
    metrics = {}
    
    if position_type == 'LONG_SPREAD':
        # Long Asset1, Short Asset2
        metrics['action_asset1'] = 'BUY'
        metrics['action_asset2'] = 'SELL'
        metrics['entry_asset1'] = asset1_price
        metrics['entry_asset2'] = asset2_price
        metrics['target_zscore'] = 0  # Mean reversion target
        metrics['stop_zscore'] = entry_zscore - 0.5  # Further divergence
        
    elif position_type == 'SHORT_SPREAD':
        # Short Asset1, Long Asset2
        metrics['action_asset1'] = 'SELL'
        metrics['action_asset2'] = 'BUY'
        metrics['entry_asset1'] = asset1_price
        metrics['entry_asset2'] = asset2_price
        metrics['target_zscore'] = 0  # Mean reversion target
        metrics['stop_zscore'] = entry_zscore + 0.5  # Further divergence
    
    metrics['hedge_ratio'] = hedge_ratio
    metrics['current_zscore'] = current_zscore
    metrics['entry_zscore'] = entry_zscore
    
    return metrics

# Streamlit UI
st.title("📊 Pairs Trading Strategy - Z-Score Mean Reversion")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset 1 Selection
    st.subheader("Asset 1")
    asset1_type = st.radio("Select Asset 1 Type:", ["Preset Assets", "Custom Ticker"], key='asset1_type')
    
    if asset1_type == "Preset Assets":
        asset1_name = st.selectbox("Choose Asset 1:", list(ASSET_MAPPINGS.keys()), key='asset1_preset')
        asset1_ticker = ASSET_MAPPINGS[asset1_name]
    else:
        asset1_ticker = st.text_input("Enter Asset 1 Ticker:", value="AAPL", key='asset1_custom')
        asset1_name = asset1_ticker
    
    st.markdown("---")
    
    # Asset 2 Selection
    st.subheader("Asset 2")
    asset2_type = st.radio("Select Asset 2 Type:", ["Preset Assets", "Custom Ticker"], key='asset2_type')
    
    if asset2_type == "Preset Assets":
        asset2_name = st.selectbox("Choose Asset 2:", list(ASSET_MAPPINGS.keys()), key='asset2_preset', index=1)
        asset2_ticker = ASSET_MAPPINGS[asset2_name]
    else:
        asset2_ticker = st.text_input("Enter Asset 2 Ticker:", value="MSFT", key='asset2_custom')
        asset2_name = asset2_ticker
    
    st.markdown("---")
    
    # Timeframe settings
    st.subheader("📅 Timeframe Settings")
    period = st.selectbox("Period:", list(PERIOD_MAPPING.keys()), index=6)
    interval = st.selectbox("Interval:", list(TIMEFRAME_MAPPING.keys()), index=9)
    
    st.markdown("---")
    
    # Strategy parameters
    st.subheader("🎯 Strategy Parameters")
    zscore_window = st.slider("Z-Score Window:", 10, 100, 20, 5)
    entry_threshold = st.slider("Entry Threshold (Z-Score):", 1.0, 3.0, 2.0, 0.1)
    exit_threshold = st.slider("Exit Threshold (Z-Score):", 0.1, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    
    # API settings
    st.subheader("🔧 API Settings")
    api_delay = st.slider("API Delay (seconds):", 1.0, 5.0, 2.0, 0.5)
    
    st.markdown("---")
    
    # Fetch data button
    fetch_button = st.button("🚀 Fetch Data & Analyze", type="primary", use_container_width=True)

# Main content area
if fetch_button:
    with st.spinner("Fetching data..."):
        # Fetch data for both assets
        data1 = fetch_data(asset1_ticker, PERIOD_MAPPING[period], TIMEFRAME_MAPPING[interval], api_delay)
        data2 = fetch_data(asset2_ticker, PERIOD_MAPPING[period], TIMEFRAME_MAPPING[interval], api_delay)
        
        if data1 is not None and data2 is not None and not data1.empty and not data2.empty:
            st.success("✅ Data fetched successfully!")
            
            # Align data by index
            common_index = data1.index.intersection(data2.index)
            data1 = data1.loc[common_index]
            data2 = data2.loc[common_index]
            
            # Use Close prices
            if isinstance(data1['Close'], pd.DataFrame):
                asset1_prices = data1['Close'].squeeze()
            else:
                asset1_prices = data1['Close']
            
            if isinstance(data2['Close'], pd.DataFrame):
                asset2_prices = data2['Close'].squeeze()
            else:
                asset2_prices = data2['Close']
            
            # Calculate spread and hedge ratio
            spread, hedge_ratio, intercept, r_squared = calculate_spread(asset1_prices, asset2_prices)
            
            # Calculate z-score
            zscore, spread_mean, spread_std = calculate_zscore(spread, window=zscore_window)
            
            # Generate signals
            signals = generate_signals(zscore, entry_threshold, exit_threshold)
            
            # Current values
            current_asset1 = asset1_prices.iloc[-1]
            current_asset2 = asset2_prices.iloc[-1]
            current_spread = spread.iloc[-1]
            current_zscore = zscore.iloc[-1]
            
            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(f"{asset1_name}", f"${current_asset1:.2f}")
            
            with col2:
                st.metric(f"{asset2_name}", f"${current_asset2:.2f}")
            
            with col3:
                st.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
            
            with col4:
                st.metric("R² (Correlation)", f"{r_squared:.4f}")
            
            with col5:
                zscore_color = "inverse" if abs(current_zscore) > entry_threshold else "off"
                st.metric("Current Z-Score", f"{current_zscore:.2f}", delta_color=zscore_color)
            
            st.markdown("---")
            
            # Trading Signal Summary
            st.header("📈 Trading Signal Summary")
            
            if abs(current_zscore) > entry_threshold:
                if current_zscore > entry_threshold:
                    position_type = 'SHORT_SPREAD'
                    signal_type = "🔴 SHORT SPREAD"
                    signal_color = "red"
                    explanation = f"""
                    **Spread is overextended (Z-Score: {current_zscore:.2f})**
                    
                    The spread between {asset1_name} and {asset2_name} is currently {current_zscore:.2f} standard deviations 
                    above its mean. This suggests {asset1_name} is relatively expensive compared to {asset2_name}.
                    
                    **Strategy:** Short the spread by selling {asset1_name} and buying {asset2_name}, expecting mean reversion.
                    """
                else:
                    position_type = 'LONG_SPREAD'
                    signal_type = "🟢 LONG SPREAD"
                    signal_color = "green"
                    explanation = f"""
                    **Spread is underextended (Z-Score: {current_zscore:.2f})**
                    
                    The spread between {asset1_name} and {asset2_name} is currently {abs(current_zscore):.2f} standard deviations 
                    below its mean. This suggests {asset1_name} is relatively cheap compared to {asset2_name}.
                    
                    **Strategy:** Long the spread by buying {asset1_name} and selling {asset2_name}, expecting mean reversion.
                    """
                
                # Calculate trade metrics
                trade_metrics = calculate_trade_metrics(
                    current_asset1, current_asset2, hedge_ratio, 
                    current_zscore, current_zscore, position_type
                )
                
                st.markdown(f"### {signal_type}")
                st.info(explanation)
                
                # Trade Details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📍 Entry Targets")
                    st.markdown(f"""
                    **{asset1_name}:**
                    - Action: **{trade_metrics['action_asset1']}**
                    - Entry Price: **${trade_metrics['entry_asset1']:.2f}**
                    - Quantity Ratio: **1.00 unit**
                    
                    **{asset2_name}:**
                    - Action: **{trade_metrics['action_asset2']}**
                    - Entry Price: **${trade_metrics['entry_asset2']:.2f}**
                    - Quantity Ratio: **{abs(hedge_ratio):.4f} units**
                    """)
                
                with col2:
                    st.markdown("### 🎯 Exit Targets & Stop Loss")
                    st.markdown(f"""
                    **Target (Mean Reversion):**
                    - Target Z-Score: **{trade_metrics['target_zscore']:.2f}**
                    - Exit when spread returns to mean
                    
                    **Stop Loss:**
                    - Stop Z-Score: **{trade_metrics['stop_zscore']:.2f}**
                    - Exit if spread diverges further
                    
                    **Current Status:**
                    - Entry Z-Score: **{trade_metrics['entry_zscore']:.2f}**
                    - Current Z-Score: **{trade_metrics['current_zscore']:.2f}**
                    """)
                
                st.markdown("---")
                
                # Risk Management Notes
                st.markdown("### ⚠️ Risk Management Notes")
                st.warning(f"""
                **Position Sizing:** 
                - For every 1 unit of {asset1_name}, trade {abs(hedge_ratio):.4f} units of {asset2_name}
                - This maintains dollar neutrality based on historical correlation
                
                **Exit Strategy:**
                - Take profit when Z-Score reaches {exit_threshold:.2f} (mean reversion)
                - Cut losses if Z-Score moves to {trade_metrics['stop_zscore']:.2f} (further divergence)
                
                **Important:** Monitor correlation (R²: {r_squared:.4f}). Correlations can break down, especially during market stress.
                """)
                
            else:
                st.info(f"""
                ### ⏸️ No Trading Signal
                
                **Current Z-Score: {current_zscore:.2f}**
                
                The spread is within {exit_threshold} standard deviations of its mean, indicating no significant 
                trading opportunity at this time. Wait for Z-Score to exceed ±{entry_threshold:.2f} for entry signals.
                
                **Spread Status:** Within normal range - Monitor for opportunities
                """)
            
            st.markdown("---")
            
            # Visualizations
            st.header("📊 Analysis Charts")
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    f'{asset1_name} vs {asset2_name} Prices',
                    'Spread Over Time',
                    'Z-Score with Entry/Exit Levels',
                    'Trading Signals'
                ),
                vertical_spacing=0.08,
                row_heights=[0.25, 0.25, 0.25, 0.25]
            )
            
            # Plot 1: Price comparison
            fig.add_trace(
                go.Scatter(x=data1.index, y=asset1_prices, name=asset1_name, line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data2.index, y=asset2_prices, name=asset2_name, line=dict(color='orange'), yaxis='y2'),
                row=1, col=1
            )
            
            # Plot 2: Spread
            fig.add_trace(
                go.Scatter(x=spread.index, y=spread, name='Spread', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=spread_mean.index, y=spread_mean, name='Mean', line=dict(color='gray', dash='dash')),
                row=2, col=1
            )
            
            # Plot 3: Z-Score
            fig.add_trace(
                go.Scatter(x=zscore.index, y=zscore, name='Z-Score', line=dict(color='green')),
                row=3, col=1
            )
            fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=exit_threshold, line_dash="dash", line_color="yellow", row=3, col=1)
            fig.add_hline(y=-exit_threshold, line_dash="dash", line_color="yellow", row=3, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)
            
            # Plot 4: Signals
            fig.add_trace(
                go.Scatter(x=signals.index, y=signals['position'], name='Position', 
                          line=dict(color='black'), mode='lines'),
                row=4, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=1400,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Date", row=4, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Spread", row=2, col=1)
            fig.update_yaxes(title_text="Z-Score", row=3, col=1)
            fig.update_yaxes(title_text="Position", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical Summary
            st.header("📈 Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Spread Statistics")
                st.dataframe({
                    'Metric': ['Mean', 'Std Dev', 'Current Value', 'Min', 'Max'],
                    'Value': [
                        f"{spread.mean():.4f}",
                        f"{spread.std():.4f}",
                        f"{current_spread:.4f}",
                        f"{spread.min():.4f}",
                        f"{spread.max():.4f}"
                    ]
                })
            
            with col2:
                st.markdown("### Z-Score Statistics")
                st.dataframe({
                    'Metric': ['Mean', 'Std Dev', 'Current Value', 'Min', 'Max'],
                    'Value': [
                        f"{zscore.mean():.4f}",
                        f"{zscore.std():.4f}",
                        f"{current_zscore:.4f}",
                        f"{zscore.min():.4f}",
                        f"{zscore.max():.4f}"
                    ]
                })
            
        else:
            st.error("❌ Failed to fetch data. Please check ticker symbols and try again.")

else:
    # Welcome message
    st.info("""
    ### 👋 Welcome to Pairs Trading Strategy Application
    
    This application implements a **Z-Score Mean Reversion** strategy for pairs trading.
    
    **How it works:**
    1. Select two correlated assets from the sidebar
    2. Configure timeframe and strategy parameters
    3. Click "Fetch Data & Analyze" to see trading signals
    
    **Strategy Logic:**
    - Calculates the spread between two assets using linear regression
    - Monitors z-score of the spread (standard deviations from mean)
    - Generates signals when spread is overextended
    - Profits from mean reversion when spread normalizes
    
    **Configure your parameters in the sidebar and click the button to begin!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Pairs Trading Strategy - Z-Score Mean Reversion | Built with Streamlit</p>
    <p style='font-size: 0.8em;'>⚠️ This is for educational purposes only. Always do your own research before trading.</p>
</div>
""", unsafe_allow_html=True)
