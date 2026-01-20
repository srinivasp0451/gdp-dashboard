import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Professional AlgoTrading Platform", layout="wide", initial_sidebar_state="expanded")

# ==================== CONSTANTS ====================
TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "USDINR": "USDINR=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "MCX": "MCX.NS",
    "CUSTOM": ""
}

TIMEFRAME_PERIODS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "1mo"],
    "15m": ["1mo"],
    "30m": ["1mo"],
    "1h": ["1mo"],
    "4h": ["1mo"],
    "1d": ["1mo", "1y", "2y", "5y"],
    "1wk": ["1mo", "1y", "5y", "10y", "20y"],
    "1mo": ["1y", "2y", "5y", "10y", "20y", "30y"]
}

IST = pytz.timezone('Asia/Kolkata')

# ==================== HELPER FUNCTIONS ====================

def validate_timeframe_period(timeframe: str, period: str) -> bool:
    """Validate timeframe and period combination"""
    return period in TIMEFRAME_PERIODS.get(timeframe, [])

def fetch_data_with_retry(ticker: str, interval: str, period: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch data from yfinance with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            # Random delay between 1.0 and 1.5 seconds
            time.sleep(np.random.uniform(1.0, 1.5))
            
            data = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
            
            if data.empty:
                return None
            
            # Flatten multi-index if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
                # Keep only OHLCV columns
                ohlcv_cols = [col for col in data.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume'])]
                data = data[ohlcv_cols]
                # Rename to standard format
                data.columns = [col.split('_')[0] if '_' in col else col for col in data.columns]
            
            # Handle timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert(IST)
            else:
                data.index = data.index.tz_convert(IST)
            
            # Ensure standard OHLCV columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            for col in required_cols:
                if col not in data.columns:
                    st.warning(f"Missing {col} column for {ticker}")
                    return None
            
            return data
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch {ticker} after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ratio(data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
    """Calculate ratio between two datasets"""
    # Align dataframes by index
    common_index = data1.index.intersection(data2.index)
    if len(common_index) == 0:
        return pd.DataFrame()
    
    aligned_data1 = data1.loc[common_index]
    aligned_data2 = data2.loc[common_index]
    
    ratio_df = pd.DataFrame(index=common_index)
    ratio_df['Ratio'] = aligned_data1['Close'] / aligned_data2['Close']
    ratio_df['Ticker1_Close'] = aligned_data1['Close']
    ratio_df['Ticker2_Close'] = aligned_data2['Close']
    
    # Add volume if available
    if 'Volume' in aligned_data1.columns:
        ratio_df['Ticker1_Volume'] = aligned_data1['Volume']
    if 'Volume' in aligned_data2.columns:
        ratio_df['Ticker2_Volume'] = aligned_data2['Volume']
    
    return ratio_df

def calculate_future_movement(data: pd.DataFrame, n_candles: int = 15) -> pd.DataFrame:
    """Calculate future price movement for next N candles"""
    results = []
    
    for i in range(len(data)):
        row_data = {'DateTime': data.index[i], 'Current_Close': data['Close'].iloc[i]}
        
        for n in range(1, n_candles + 1):
            if i + n < len(data):
                future_close = data['Close'].iloc[i + n]
                future_datetime = data.index[i + n]
                points_change = future_close - data['Close'].iloc[i]
                pct_change = (points_change / data['Close'].iloc[i]) * 100
                
                row_data[f'Next_{n}_DateTime'] = future_datetime
                row_data[f'Next_{n}_Points'] = points_change
                row_data[f'Next_{n}_Pct'] = pct_change
            else:
                row_data[f'Next_{n}_DateTime'] = None
                row_data[f'Next_{n}_Points'] = np.nan
                row_data[f'Next_{n}_Pct'] = np.nan
        
        results.append(row_data)
    
    return pd.DataFrame(results)

def calculate_rsi_divergence(data1: pd.DataFrame, data2: pd.DataFrame, rsi_period: int = 14) -> Dict:
    """Calculate RSI divergence between two tickers"""
    common_index = data1.index.intersection(data2.index)
    if len(common_index) == 0:
        return {}
    
    aligned_data1 = data1.loc[common_index].copy()
    aligned_data2 = data2.loc[common_index].copy()
    
    # Calculate RSI for both
    aligned_data1['RSI'] = calculate_rsi(aligned_data1['Close'], rsi_period)
    aligned_data2['RSI'] = calculate_rsi(aligned_data2['Close'], rsi_period)
    
    # Calculate divergence
    divergence_df = pd.DataFrame(index=common_index)
    divergence_df['Ticker1_RSI'] = aligned_data1['RSI']
    divergence_df['Ticker2_RSI'] = aligned_data2['RSI']
    divergence_df['RSI_Diff'] = aligned_data1['RSI'] - aligned_data2['RSI']
    divergence_df['Ticker1_Close'] = aligned_data1['Close']
    
    return {
        'data': divergence_df,
        'ticker1_rsi': aligned_data1['RSI'],
        'ticker2_rsi': aligned_data2['RSI']
    }

def get_majority_forecast(movements: List[float]) -> Tuple[str, float]:
    """Calculate majority forecast based on movement analysis"""
    if not movements:
        return "SIDEWAYS", 0.0
    
    bullish = sum(1 for m in movements if m > 0.5)
    bearish = sum(1 for m in movements if m < -0.5)
    sideways = len(movements) - bullish - bearish
    
    total = len(movements)
    if bullish > bearish and bullish > sideways:
        return "BULLISH", (bullish / total) * 100
    elif bearish > bullish and bearish > sideways:
        return "BEARISH", (bearish / total) * 100
    else:
        return "SIDEWAYS", (sideways / total) * 100

def create_ratio_chart(ratio_df: pd.DataFrame, ticker1_name: str, ticker2_name: str, timeframe: str):
    """Create ratio chart with price and volume"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker1_name}/{ticker2_name} Ratio', 'RSI', 'Volume'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Ratio line
    fig.add_trace(go.Scatter(
        x=ratio_df.index, y=ratio_df['Ratio'],
        mode='lines', name='Ratio',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    # RSI
    if len(ratio_df) > 14:
        rsi = calculate_rsi(ratio_df['Ratio'])
        fig.add_trace(go.Scatter(
            x=ratio_df.index, y=rsi,
            mode='lines', name='RSI',
            line=dict(color='purple', width=1.5)
        ), row=2, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
    
    # Volume
    has_volume = False
    if 'Ticker1_Volume' in ratio_df.columns:
        fig.add_trace(go.Bar(
            x=ratio_df.index, y=ratio_df['Ticker1_Volume'],
            name=f'{ticker1_name} Volume',
            marker_color='lightblue', opacity=0.7
        ), row=3, col=1)
        has_volume = True
    
    fig.update_layout(
        title=f'{ticker1_name}/{ticker2_name} Analysis - {timeframe}',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

# ==================== STREAMLIT UI ====================

st.title("🚀 Professional AlgoTrading Platform")
st.markdown("---")

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ticker 1 selection
    ticker1_type = st.selectbox("Ticker 1", list(TICKER_MAP.keys()), index=0, key='ticker1_type')
    if ticker1_type == "CUSTOM":
        ticker1_custom = st.text_input("Enter Ticker 1 Symbol", "^NSEI", key='ticker1_custom')
        ticker1 = ticker1_custom
    else:
        ticker1 = TICKER_MAP[ticker1_type]
    
    # Ticker 2 selection
    ticker2_type = st.selectbox("Ticker 2", list(TICKER_MAP.keys()), index=5, key='ticker2_type')
    if ticker2_type == "CUSTOM":
        ticker2_custom = st.text_input("Enter Ticker 2 Symbol", "USDINR=X", key='ticker2_custom')
        ticker2 = ticker2_custom
    else:
        ticker2 = TICKER_MAP[ticker2_type]
    
    # Timeframe selection
    timeframe = st.selectbox("Timeframe", list(TIMEFRAME_PERIODS.keys()), index=6, key='timeframe')
    
    # Period selection (filtered based on timeframe)
    available_periods = TIMEFRAME_PERIODS[timeframe]
    period = st.selectbox("Period", available_periods, key='period')
    
    # Number of bins
    n_bins = st.number_input("Number of Bins", min_value=5, max_value=100, value=20, step=5, key='n_bins')
    
    # Next N candles
    n_candles = st.number_input("Next N Candles", min_value=1, max_value=50, value=15, step=1, key='n_candles')
    
    st.markdown("---")
    
    # Fetch button
    if st.button("🔄 Fetch & Analyze", type="primary", use_container_width=True):
        st.session_state.data_fetched = False
        st.session_state.analysis_results = {}
        
        # Validate combination
        if not validate_timeframe_period(timeframe, period):
            st.error(f"Invalid combination: {timeframe} with {period}")
        else:
            with st.spinner("Fetching and analyzing data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Fetch data for all timeframes
                all_results = {}
                all_timeframes = list(TIMEFRAME_PERIODS.keys())
                total_steps = len(all_timeframes) * 2 + 2
                current_step = 0
                
                for tf in all_timeframes:
                    for prd in TIMEFRAME_PERIODS[tf]:
                        current_step += 1
                        progress = current_step / total_steps
                        progress_bar.progress(progress)
                        status_text.text(f"Fetching {ticker1_type} - {tf}/{prd}... ({int(progress*100)}%)")
                        
                        data1 = fetch_data_with_retry(ticker1, tf, prd)
                        
                        current_step += 1
                        progress = current_step / total_steps
                        progress_bar.progress(progress)
                        status_text.text(f"Fetching {ticker2_type} - {tf}/{prd}... ({int(progress*100)}%)")
                        
                        data2 = fetch_data_with_retry(ticker2, tf, prd)
                        
                        if data1 is not None and data2 is not None:
                            key = f"{tf}_{prd}"
                            
                            # Calculate ratio
                            ratio_df = calculate_ratio(data1, data2)
                            
                            # Calculate future movements
                            movement_df = calculate_future_movement(data1, n_candles)
                            
                            # Calculate RSI divergence
                            rsi_div = calculate_rsi_divergence(data1, data2)
                            
                            all_results[key] = {
                                'data1': data1,
                                'data2': data2,
                                'ratio': ratio_df,
                                'movement': movement_df,
                                'rsi_divergence': rsi_div
                            }
                
                progress_bar.progress(1.0)
                status_text.text("Analysis complete! ✅")
                
                st.session_state.analysis_results = all_results
                st.session_state.data_fetched = True
                time.sleep(1)
                st.rerun()

# Main content area
if st.session_state.data_fetched and st.session_state.analysis_results:
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ratio Charts", "📈 RSI Divergence", "🎯 Backtesting", "📉 Statistics"])
    
    # Get current selection data
    current_key = f"{timeframe}_{period}"
    current_data = st.session_state.analysis_results.get(current_key)
    
    if current_data:
        
        # TAB 1: Ratio Charts
        with tab1:
            st.header(f"Ratio Analysis: {ticker1_type} / {ticker2_type}")
            
            ratio_df = current_data['ratio']
            movement_df = current_data['movement']
            
            if not ratio_df.empty:
                # Display last N bins
                st.subheader(f"Last {n_bins} Data Points")
                display_df = movement_df.tail(n_bins).copy()
                display_df['DateTime'] = display_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Analysis summary
                with st.expander("📋 Analysis Summary & Forecast", expanded=True):
                    current_ratio = ratio_df['Ratio'].iloc[-1]
                    max_ratio = ratio_df['Ratio'].max()
                    min_ratio = ratio_df['Ratio'].min()
                    mean_ratio = ratio_df['Ratio'].mean()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Ratio", f"{current_ratio:.4f}")
                    col2.metric("Max Ratio", f"{max_ratio:.4f}")
                    col3.metric("Min Ratio", f"{min_ratio:.4f}")
                    col4.metric("Mean Ratio", f"{mean_ratio:.4f}")
                    
                    # Calculate movements for forecast
                    recent_movements = movement_df.tail(20)['Next_1_Pct'].dropna().tolist()
                    forecast, confidence = get_majority_forecast(recent_movements)
                    
                    st.markdown(f"### 🎯 Forecast: **{forecast}** (Confidence: {confidence:.1f}%)")
                    
                    # Generate trading recommendation
                    if forecast == "BULLISH":
                        entry = current_ratio
                        sl = entry * 0.98
                        target1 = entry * 1.02
                        target2 = entry * 1.04
                        st.success(f"📈 **Entry**: {entry:.4f} | **SL**: {sl:.4f} | **T1**: {target1:.4f} | **T2**: {target2:.4f}")
                    elif forecast == "BEARISH":
                        entry = current_ratio
                        sl = entry * 1.02
                        target1 = entry * 0.98
                        target2 = entry * 0.96
                        st.error(f"📉 **Entry**: {entry:.4f} | **SL**: {sl:.4f} | **T1**: {target1:.4f} | **T2**: {target2:.4f}")
                    else:
                        st.info("↔️ Market showing sideways movement. Wait for clear signal.")
                    
                    st.markdown(f"""
                    **Highlights**: Current ratio at {current_ratio:.4f}, {((current_ratio - mean_ratio)/mean_ratio*100):.1f}% from mean. 
                    Historical range: {min_ratio:.4f} to {max_ratio:.4f}. Recent momentum suggests {forecast.lower()} bias 
                    based on {len(recent_movements)} recent candles with {confidence:.0f}% confidence.
                    """)
                
                # Plot charts for all timeframes
                st.subheader("📊 Multi-Timeframe Analysis")
                for key, data in st.session_state.analysis_results.items():
                    tf, prd = key.split('_')
                    with st.expander(f"{tf} / {prd}", expanded=(key == current_key)):
                        fig = create_ratio_chart(data['ratio'], ticker1_type, ticker2_type, f"{tf}/{prd}")
                        st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: RSI Divergence
        with tab2:
            st.header(f"RSI Divergence: {ticker1_type} vs {ticker2_type}")
            
            rsi_div = current_data['rsi_divergence']
            
            if rsi_div and 'data' in rsi_div:
                div_df = rsi_div['data']
                
                # Calculate movements with RSI
                st.subheader("Price Movement with RSI Analysis")
                movement_with_rsi = movement_df.tail(n_bins).copy()
                movement_with_rsi['DateTime'] = movement_with_rsi['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
                st.dataframe(movement_with_rsi, use_container_width=True, height=400)
                
                # RSI Divergence analysis
                with st.expander("📋 RSI Divergence Analysis & Forecast", expanded=True):
                    current_rsi1 = div_df['Ticker1_RSI'].iloc[-1]
                    current_rsi2 = div_df['Ticker2_RSI'].iloc[-1]
                    rsi_diff = current_rsi1 - current_rsi2
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"{ticker1_type} RSI", f"{current_rsi1:.2f}")
                    col2.metric(f"{ticker2_type} RSI", f"{current_rsi2:.2f}")
                    col3.metric("RSI Divergence", f"{rsi_diff:.2f}")
                    
                    # Forecast based on RSI
                    rsi_movements = []
                    if current_rsi1 < 30:
                        rsi_movements.extend([1, 1, 1])  # Oversold - bullish
                    elif current_rsi1 > 70:
                        rsi_movements.extend([-1, -1, -1])  # Overbought - bearish
                    
                    forecast, confidence = get_majority_forecast(rsi_movements + movement_df.tail(15)['Next_1_Pct'].dropna().tolist())
                    
                    st.markdown(f"### 🎯 RSI Forecast: **{forecast}** (Confidence: {confidence:.1f}%)")
                    
                    st.markdown(f"""
                    **RSI Analysis**: {ticker1_type} RSI at {current_rsi1:.1f} vs {ticker2_type} at {current_rsi2:.1f}. 
                    Divergence of {abs(rsi_diff):.1f} points suggests {'strong ' if abs(rsi_diff) > 20 else ''}
                    {'correlation' if abs(rsi_diff) < 10 else 'divergence'}. Current momentum: {forecast.lower()}.
                    """)
                
                # Plot RSI divergence for all timeframes
                st.subheader("📈 Multi-Timeframe RSI Analysis")
                for key, data in st.session_state.analysis_results.items():
                    tf, prd = key.split('_')
                    if 'rsi_divergence' in data and data['rsi_divergence']:
                        with st.expander(f"{tf} / {prd}", expanded=(key == current_key)):
                            rsi_data = data['rsi_divergence']['data']
                            
                            fig = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                subplot_titles=(f'{ticker1_type} Price', 'RSI Comparison'),
                                row_heights=[0.6, 0.4]
                            )
                            
                            # Price
                            fig.add_trace(go.Scatter(
                                x=rsi_data.index, y=rsi_data['Ticker1_Close'],
                                mode='lines', name=f'{ticker1_type} Price',
                                line=dict(color='blue', width=2)
                            ), row=1, col=1)
                            
                            # RSI comparison
                            fig.add_trace(go.Scatter(
                                x=rsi_data.index, y=rsi_data['Ticker1_RSI'],
                                mode='lines', name=f'{ticker1_type} RSI',
                                line=dict(color='green', width=2)
                            ), row=2, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=rsi_data.index, y=rsi_data['Ticker2_RSI'],
                                mode='lines', name=f'{ticker2_type} RSI',
                                line=dict(color='red', width=2)
                            ), row=2, col=1)
                            
                            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
                            
                            fig.update_layout(height=700, title=f'RSI Divergence Analysis - {tf}/{prd}', hovermode='x unified')
                            st.plotly_chart(fig, use_container_width=True)
        
        # TAB 3: Backtesting
        with tab3:
            st.header("🎯 Backtesting Results")
            
            data1 = current_data['data1']
            ratio_df = current_data['ratio']
            
            # Simple backtesting strategy based on ratio
            st.subheader(f"Strategy: Ratio Mean Reversion ({timeframe}/{period})")
            
            backtest_results = []
            mean_ratio = ratio_df['Ratio'].mean()
            std_ratio = ratio_df['Ratio'].std()
            
            position = None
            entry_price = 0
            entry_datetime = None
            
            for i in range(1, len(ratio_df)):
                current_ratio = ratio_df['Ratio'].iloc[i]
                current_price = ratio_df['Ticker1_Close'].iloc[i]
                current_datetime = ratio_df.index[i]
                
                # Entry logic
                if position is None:
                    if current_ratio < mean_ratio - std_ratio:  # Oversold
                        position = "LONG"
                        entry_price = current_price
                        entry_datetime = current_datetime
                    elif current_ratio > mean_ratio + std_ratio:  # Overbought
                        position = "SHORT"
                        entry_price = current_price
                        entry_datetime = current_datetime
                
                # Exit logic
                elif position == "LONG":
                    if current_ratio > mean_ratio:
                        pnl_points = current_price - entry_price
                        pnl_pct = (pnl_points / entry_price) * 100
                        backtest_results.append({
                            'Entry_DateTime': entry_datetime.strftime('%Y-%m-%d %H:%M:%S IST'),
                            'Exit_DateTime': current_datetime.strftime('%Y-%m-%d %H:%M:%S IST'),
                            'Position': position,
                            'Entry_Price': entry_price,
                            'Exit_Price': current_price,
                            'SL': entry_price * 0.98,
                            'Target': entry_price * 1.02,
                            'PnL_Points': pnl_points,
                            'PnL_Pct': pnl_pct,
                            'Reason': 'Ratio reverted to mean'
                        })
                        position = None
                
                elif position == "SHORT":
                    if current_ratio < mean_ratio:
                        pnl_points = entry_price - current_price
                        pnl_pct = (pnl_points / entry_price) * 100
                        backtest_results.append({
                            'Entry_DateTime': entry_datetime.strftime('%Y-%m-%d %H:%M:%S IST'),
                            'Exit_DateTime': current_datetime.strftime('%Y-%m-%d %H:%M:%S IST'),
                            'Position': position,
                            'Entry_Price': entry_price,
                            'Exit_Price': current_price,
                            'SL': entry_price * 1.02,
                            'Target': entry_price * 0.98,
                            'PnL_Points': pnl_points,
                            'PnL_Pct': pnl_pct,
                            'Reason': 'Ratio reverted to mean'
                        })
                        position = None
            
            if backtest_results:
                results_df = pd.DataFrame(backtest_results)
                
                # Summary metrics
                total_trades = len(results_df)
                winning_trades = len(results_df[results_df['PnL_Points'] > 0])
                accuracy = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                total_pnl = results_df['PnL_Points'].sum()
                avg_pnl = results_df['PnL_Points'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Trades", total_trades)
                col2.metric("Winning Trades", winning_trades)
                col3.metric("Accuracy", f"{accuracy:.1f}%")
                col4.metric("Total PnL", f"{total_pnl:.2f}")
                
                st.dataframe(results_df, use_container_width=True, height=500)
                
                # Forecast based on backtesting
                forecast_movements = results_df.tail(10)['PnL_Points'].tolist()
                forecast, confidence = get_majority_forecast(forecast_movements)
                st.markdown(f"### 🎯 Backtesting Forecast: **{forecast}** (Confidence: {confidence:.1f}%)")
            else:
                st.info("No trades executed in the selected timeframe with current strategy parameters.")
        
        # TAB 4: Statistics
        with tab4:
            st.header("📉 Statistical Analysis")
            
            data1 = current_data['data1']
            
            # Prepare statistics dataframe
            stats_df = data1.copy()
            stats_df['Points_Change'] = stats_df['Close'].diff()
            stats_df['Pct_Change'] = stats_df['Close'].pct_change() * 100
            stats_df['Day_of_Week'] = stats_df.index.day_name()
            stats_df['Hour'] = stats_df.index.hour
            stats_df['DateTime_IST'] = stats_df.index.strftime('%Y-%m-%d %H:%M:%S IST')
            
            # Sort in descending order
            stats_df_display = stats_df.sort_index(ascending=False).copy()
            
            st.subheader(f"Price Movement Analysis ({timeframe}/{period})")
            
            # Display table with color coding
            display_cols = ['DateTime_IST', 'Close', 'Points_Change', 'Pct_Change', 'Day_of_Week']
            if 'Hour' in stats_df_display.columns and timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']:
                display_cols.append('Hour')
            
            # Style function for dataframe
            def highlight_changes(val):
                if pd.isna(val):
                    return ''
                try:
                    color = 'background-color: lightgreen' if float(val) > 0 else 'background-color: lightcoral' if float(val) < 0 else ''
                    return color
                except:
                    return ''
            
            styled_df = stats_df_display[display_cols].head(n_bins).style.applymap(
                highlight_changes, 
                subset=['Points_Change', 'Pct_Change']
            )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Statistical summary
            with st.expander("📊 Statistical Summary", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Min Points Change", f"{stats_df['Points_Change'].min():.2f}")
                col2.metric("Max Points Change", f"{stats_df['Points_Change'].max():.2f}")
                col3.metric("Median Points Change", f"{stats_df['Points_Change'].median():.2f}")
                col4.metric("Mean Points Change", f"{stats_df['Points_Change'].mean():.2f}")
                
                # Day of week analysis
                st.markdown("### 📅 Day of Week Analysis")
                day_stats = stats_df.groupby('Day_of_Week').agg({
                    'Points_Change': ['mean', 'std', 'count'],
                    'Pct_Change': 'mean'
                }).round(2)
                
                # Reorder days
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_stats = day_stats.reindex([d for d in days_order if d in day_stats.index])
                
                st.dataframe(day_stats, use_container_width=True)
                
                # Best and worst days
                best_day = day_stats[('Points_Change', 'mean')].idxmax()
                worst_day = day_stats[('Points_Change', 'mean')].idxmin()
                
                col1, col2 = st.columns(2)
                col1.success(f"🟢 Best Day: **{best_day}** ({day_stats.loc[best_day, ('Points_Change', 'mean')]:.2f} pts)")
                col2.error(f"🔴 Worst Day: **{worst_day}** ({day_stats.loc[worst_day, ('Points_Change', 'mean')]:.2f} pts)")
                
                # Hour analysis (if intraday)
                if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']:
                    st.markdown("### ⏰ Hourly Analysis")
                    hour_stats = stats_df.groupby('Hour').agg({
                        'Points_Change': 'mean',
                        'Pct_Change': 'mean'
                    }).round(2)
                    
                    st.dataframe(hour_stats, use_container_width=True)
                    
                    best_hour = hour_stats['Points_Change'].idxmax()
                    worst_hour = hour_stats['Points_Change'].idxmin()
                    
                    col1, col2 = st.columns(2)
                    col1.success(f"🟢 Best Hour: **{best_hour}:00** ({hour_stats.loc[best_hour, 'Points_Change']:.2f} pts)")
                    col2.error(f"🔴 Worst Hour: **{worst_hour}:00** ({hour_stats.loc[worst_hour, 'Points_Change']:.2f} pts)")
                
                # Majority forecast
                all_movements = stats_df['Pct_Change'].dropna().tail(50).tolist()
                forecast, confidence = get_majority_forecast(all_movements)
                st.markdown(f"### 🎯 Statistical Forecast: **{forecast}** (Confidence: {confidence:.1f}%)")
            
            # Charts
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Points change chart
                fig_points = go.Figure()
                fig_points.add_trace(go.Scatter(
                    x=stats_df.index,
                    y=stats_df['Points_Change'],
                    mode='lines',
                    name='Points Change',
                    line=dict(color='blue', width=1.5),
                    fill='tozeroy'
                ))
                fig_points.update_layout(
                    title='Points Change Over Time',
                    xaxis_title='DateTime (IST)',
                    yaxis_title='Points',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_points, use_container_width=True)
            
            with col2:
                # Percentage change chart
                fig_pct = go.Figure()
                fig_pct.add_trace(go.Scatter(
                    x=stats_df.index,
                    y=stats_df['Pct_Change'],
                    mode='lines',
                    name='% Change',
                    line=dict(color='green', width=1.5),
                    fill='tozeroy'
                ))
                fig_pct.update_layout(
                    title='Percentage Change Over Time',
                    xaxis_title='DateTime (IST)',
                    yaxis_title='Percentage (%)',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_pct, use_container_width=True)
            
            # Day of week distribution
            if not day_stats.empty:
                fig_day = go.Figure()
                fig_day.add_trace(go.Bar(
                    x=day_stats.index,
                    y=day_stats[('Points_Change', 'mean')],
                    marker_color=['lightgreen' if x > 0 else 'lightcoral' for x in day_stats[('Points_Change', 'mean')]],
                    text=day_stats[('Points_Change', 'mean')].round(2),
                    textposition='auto'
                ))
                fig_day.update_layout(
                    title='Average Points Change by Day of Week',
                    xaxis_title='Day',
                    yaxis_title='Average Points Change',
                    height=400
                )
                st.plotly_chart(fig_day, use_container_width=True)
        
        # FINAL SUMMARY - Below all tabs
        st.markdown("---")
        st.header("🎯 FINAL COMPREHENSIVE FORECAST")
        
        # Collect forecasts from all analyses
        all_forecasts = []
        
        # From all timeframes
        for key, data in st.session_state.analysis_results.items():
            if 'movement' in data:
                movements = data['movement']['Next_1_Pct'].dropna().tail(20).tolist()
                if movements:
                    forecast, _ = get_majority_forecast(movements)
                    all_forecasts.append(forecast)
        
        # Count forecasts
        bullish_count = all_forecasts.count("BULLISH")
        bearish_count = all_forecasts.count("BEARISH")
        sideways_count = all_forecasts.count("SIDEWAYS")
        total_count = len(all_forecasts)
        
        # Determine final forecast
        if bullish_count > bearish_count and bullish_count > sideways_count:
            final_forecast = "BULLISH"
            final_confidence = (bullish_count / total_count) * 100
            color = "green"
        elif bearish_count > bullish_count and bearish_count > sideways_count:
            final_forecast = "BEARISH"
            final_confidence = (bearish_count / total_count) * 100
            color = "red"
        else:
            final_forecast = "SIDEWAYS"
            final_confidence = (sideways_count / total_count) * 100
            color = "gray"
        
        # Display final forecast
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Final Forecast", final_forecast)
        with col2:
            st.metric("Confidence", f"{final_confidence:.1f}%")
        with col3:
            st.metric("Bullish Signals", f"{bullish_count}/{total_count}")
        with col4:
            st.metric("Bearish Signals", f"{bearish_count}/{total_count}")
        with col5:
            st.metric("Sideways Signals", f"{sideways_count}/{total_count}")
        
        # Visual indicator
        if final_forecast == "BULLISH":
            st.success(f"🚀 **STRONG {final_forecast} SIGNAL** - {bullish_count} out of {total_count} timeframes show bullish momentum with {final_confidence:.0f}% confidence.")
        elif final_forecast == "BEARISH":
            st.error(f"📉 **STRONG {final_forecast} SIGNAL** - {bearish_count} out of {total_count} timeframes show bearish momentum with {final_confidence:.0f}% confidence.")
        else:
            st.info(f"↔️ **{final_forecast} MARKET** - Market consolidating with {final_confidence:.0f}% confidence. Wait for clear directional move.")
        
        # Recommendation
        st.markdown("### 💡 Trading Recommendation")
        current_price = current_data['data1']['Close'].iloc[-1]
        
        if final_forecast == "BULLISH" and final_confidence > 60:
            entry = current_price
            sl = entry * 0.985
            target1 = entry * 1.015
            target2 = entry * 1.03
            risk_reward = (target1 - entry) / (entry - sl)
            
            st.success(f"""
            **LONG SETUP** ({ticker1_type})
            - Entry: {entry:.2f}
            - Stop Loss: {sl:.2f} ({-1.5:.1f}%)
            - Target 1: {target1:.2f} ({1.5:.1f}%)
            - Target 2: {target2:.2f} ({3.0:.1f}%)
            - Risk:Reward = 1:{risk_reward:.1f}
            """)
        
        elif final_forecast == "BEARISH" and final_confidence > 60:
            entry = current_price
            sl = entry * 1.015
            target1 = entry * 0.985
            target2 = entry * 0.97
            risk_reward = (entry - target1) / (sl - entry)
            
            st.error(f"""
            **SHORT SETUP** ({ticker1_type})
            - Entry: {entry:.2f}
            - Stop Loss: {sl:.2f} ({1.5:.1f}%)
            - Target 1: {target1:.2f} ({-1.5:.1f}%)
            - Target 2: {target2:.2f} ({-3.0:.1f}%)
            - Risk:Reward = 1:{risk_reward:.1f}
            """)
        
        else:
            st.warning(f"""
            **NO CLEAR SETUP**
            - Current Price: {current_price:.2f}
            - Confidence: {final_confidence:.0f}% (below 60% threshold)
            - Recommendation: Wait for clearer signal or use range-bound strategies
            """)
    
    else:
        st.warning(f"No data available for {timeframe}/{period}. Please try a different combination.")

else:
    # Welcome screen
    st.info("""
    👋 **Welcome to the Professional AlgoTrading Platform!**
    
    **Getting Started:**
    1. Select your tickers from the sidebar (Default: NIFTY 50 / USDINR)
    2. Choose timeframe and period (validated combinations only)
    3. Set number of bins and next N candles to analyze
    4. Click **Fetch & Analyze** to start
    
    **Features:**
    - 📊 Multi-timeframe ratio analysis
    - 📈 RSI divergence detection
    - 🎯 Automated backtesting
    - 📉 Statistical analysis with day/hour patterns
    - 🎯 AI-powered forecasting with confidence levels
    
    **Note:** Data fetching respects yfinance API rate limits with 1-1.5 second delays between requests.
    All timestamps are converted to IST (Indian Standard Time) for consistency.
    """)
    
    # Show example combinations
    with st.expander("📚 Valid Timeframe/Period Combinations"):
        for tf, periods in TIMEFRAME_PERIODS.items():
            st.write(f"**{tf}:** {', '.join(periods)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Professional AlgoTrading Platform v1.0</strong></p>
    <p>⚠️ Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results. 
    Always perform your own analysis and risk management before trading.</p>
</div>
""", unsafe_allow_html=True)
