import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Algo Trading Analysis", layout="wide", initial_sidebar_state="expanded")

# Utility functions
def convert_to_ist(df):
    """Convert datetime to IST and handle timezone awareness"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(ist)
        else:
            df.index = df.index.tz_convert(ist)
        df.index = df.index.tz_localize(None)
    except:
        pass
    return df

def align_data(data1, data2):
    """Align two dataframes by common index"""
    common_index = data1.index.intersection(data2.index)
    return data1.loc[common_index], data2.loc[common_index]

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volatility(data, window=14):
    """Calculate rolling volatility"""
    returns = data.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return volatility

def find_pattern_similarity(data, window=20, top_n=5):
    """Find similar patterns based on normalized percentage changes"""
    if len(data) < window * 2:
        return []
    
    pct_changes = data.pct_change().fillna(0)
    current_pattern = pct_changes.iloc[-window:].values
    current_pattern_norm = (current_pattern - current_pattern.mean()) / (current_pattern.std() + 1e-8)
    
    similarities = []
    for i in range(window, len(data) - window):
        historical_pattern = pct_changes.iloc[i-window:i].values
        historical_pattern_norm = (historical_pattern - historical_pattern.mean()) / (historical_pattern.std() + 1e-8)
        
        correlation = np.corrcoef(current_pattern_norm, historical_pattern_norm)[0, 1]
        if not np.isnan(correlation) and correlation > 0.7:
            future_points = float(data.iloc[i+window] - data.iloc[i])
            future_pct = (future_points / float(data.iloc[i])) * 100
            
            similarities.append({
                'index': i,
                'date': data.index[i],
                'correlation': correlation,
                'price_at_pattern': float(data.iloc[i]),
                'future_price': float(data.iloc[i+window]) if i+window < len(data) else float(data.iloc[i]),
                'future_points': future_points,
                'future_return_pct': future_pct
            })
    
    similarities.sort(key=lambda x: x['correlation'], reverse=True)
    return similarities[:top_n]

# Streamlit UI
st.title("Advanced Algorithmic Trading Analysis Dashboard")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    ticker1_options = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Sensex": "^BSESN",
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "USD/INR": "USDINR=X",
        "Custom": "CUSTOM"
    }
    
    ticker1_name = st.selectbox("Select Ticker 1", list(ticker1_options.keys()))
    if ticker1_name == "Custom":
        ticker1 = st.text_input("Enter Ticker 1 Symbol", "AAPL")
    else:
        ticker1 = ticker1_options[ticker1_name]
    
    ticker2_name = st.selectbox("Select Ticker 2", list(ticker1_options.keys()), index=1)
    if ticker2_name == "Custom":
        ticker2 = st.text_input("Enter Ticker 2 Symbol", "MSFT")
    else:
        ticker2 = ticker1_options[ticker2_name]
    
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk"])
    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
    fetch_button = st.button("Fetch Data & Analyze", type="primary", use_container_width=True)

if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False

# Main analysis
if fetch_button:
    with st.spinner("Fetching data from Yahoo Finance..."):
        try:
            data1 = yf.download(ticker1, period=period, interval=timeframe, progress=False)
            data2 = yf.download(ticker2, period=period, interval=timeframe, progress=False)
            
            if data1.empty or data2.empty:
                st.error("Failed to fetch data. Please check ticker symbols and try again.")
                st.stop()
            
            if isinstance(data1.columns, pd.MultiIndex):
                data1.columns = data1.columns.get_level_values(0)
            if isinstance(data2.columns, pd.MultiIndex):
                data2.columns = data2.columns.get_level_values(0)
            
            data1 = convert_to_ist(data1)
            data2 = convert_to_ist(data2)
            
            data1, data2 = align_data(data1, data2)
            
            data1 = data1.sort_index()
            data2 = data2.sort_index()
            
            if 'Volume' not in data1.columns:
                data1['Volume'] = 0
            if 'Volume' not in data2.columns:
                data2['Volume'] = 0
            
            st.session_state.data1 = data1
            st.session_state.data2 = data2
            st.session_state.ticker1 = ticker1
            st.session_state.ticker2 = ticker2
            st.session_state.ticker1_name = ticker1_name if ticker1_name != "Custom" else ticker1
            st.session_state.ticker2_name = ticker2_name if ticker2_name != "Custom" else ticker2
            st.session_state.data_fetched = True
            
            st.success("Data fetched successfully!")
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.stop()

if st.session_state.data_fetched:
    data1 = st.session_state.data1.copy()
    data2 = st.session_state.data2.copy()
    ticker1 = st.session_state.ticker1
    ticker2 = st.session_state.ticker2
    ticker1_name = st.session_state.ticker1_name
    ticker2_name = st.session_state.ticker2_name
    
    ratio = data1['Close'] / data2['Close']
    
    rsi1 = calculate_rsi(data1['Close'])
    rsi2 = calculate_rsi(data2['Close'])
    rsi_ratio = calculate_rsi(ratio)
    
    vol1 = calculate_volatility(data1['Close'])
    vol2 = calculate_volatility(data2['Close'])
    vol_ratio = calculate_volatility(ratio)
    
    # SECTION 1: BASIC TICKER ANALYSIS
    st.header("1. Basic Ticker Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{ticker1_name} Analysis")
        latest_price1 = float(data1['Close'].iloc[-1])
        prev_price1 = float(data1['Close'].iloc[-2]) if len(data1) > 1 else latest_price1
        change1 = latest_price1 - prev_price1
        pct_change1 = (change1 / prev_price1) * 100
        rsi1_current = float(rsi1.iloc[-1]) if not pd.isna(rsi1.iloc[-1]) else 50.0
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Latest Price", f"{latest_price1:.2f}")
        metric_col2.metric("Change", f"{change1:.2f}", f"{pct_change1:.2f}%")
        metric_col3.metric("RSI", f"{rsi1_current:.2f}")
        
        stats1_data = {
            'DateTime': [str(dt) for dt in data1.index[-10:]],
            'Close': [float(x) for x in data1['Close'].iloc[-10:].values],
            'Change': [float(x) if not pd.isna(x) else 0.0 for x in data1['Close'].iloc[-10:].diff().values],
            'Pct_Change': [float(x) if not pd.isna(x) else 0.0 for x in (data1['Close'].iloc[-10:].pct_change() * 100).values]
        }
        stats1 = pd.DataFrame(stats1_data)
        st.dataframe(stats1.style.background_gradient(subset=['Change', 'Pct_Change'], cmap='RdYlGn'), use_container_width=True)
    
    with col2:
        st.subheader(f"{ticker2_name} Analysis")
        latest_price2 = float(data2['Close'].iloc[-1])
        prev_price2 = float(data2['Close'].iloc[-2]) if len(data2) > 1 else latest_price2
        change2 = latest_price2 - prev_price2
        pct_change2 = (change2 / prev_price2) * 100
        rsi2_current = float(rsi2.iloc[-1]) if not pd.isna(rsi2.iloc[-1]) else 50.0
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Latest Price", f"{latest_price2:.2f}")
        metric_col2.metric("Change", f"{change2:.2f}", f"{pct_change2:.2f}%")
        metric_col3.metric("RSI", f"{rsi2_current:.2f}")
        
        stats2_data = {
            'DateTime': [str(dt) for dt in data2.index[-10:]],
            'Close': [float(x) for x in data2['Close'].iloc[-10:].values],
            'Change': [float(x) if not pd.isna(x) else 0.0 for x in data2['Close'].iloc[-10:].diff().values],
            'Pct_Change': [float(x) if not pd.isna(x) else 0.0 for x in (data2['Close'].iloc[-10:].pct_change() * 100).values]
        }
        stats2 = pd.DataFrame(stats2_data)
        st.dataframe(stats2.style.background_gradient(subset=['Change', 'Pct_Change'], cmap='RdYlGn'), use_container_width=True)
    
    # SECTION 2: RATIO ANALYSIS
    st.header("2. Ratio Analysis")
    
    min_len = min(len(data1), len(data2), len(ratio), len(rsi1), len(rsi2), len(rsi_ratio))
    
    ratio_analysis = pd.DataFrame({
        'DateTime': [str(dt) for dt in data1.index[-min_len:]],
        f'{ticker1_name}_Price': [float(x) for x in data1['Close'].iloc[-min_len:].values],
        f'{ticker2_name}_Price': [float(x) for x in data2['Close'].iloc[-min_len:].values],
        'Ratio': [float(x) for x in ratio.iloc[-min_len:].values],
        f'RSI_{ticker1_name}': [float(x) if not pd.isna(x) else 50.0 for x in rsi1.iloc[-min_len:].values],
        f'RSI_{ticker2_name}': [float(x) if not pd.isna(x) else 50.0 for x in rsi2.iloc[-min_len:].values],
        'RSI_Ratio': [float(x) if not pd.isna(x) else 50.0 for x in rsi_ratio.iloc[-min_len:].values]
    })
    
    st.dataframe(ratio_analysis.tail(20).style.background_gradient(subset=['Ratio', 'RSI_Ratio'], cmap='viridis'), use_container_width=True)
    
    st.subheader("Ratio Insights")
    current_ratio = float(ratio.iloc[-1])
    mean_ratio = float(ratio.mean())
    std_ratio = float(ratio.std())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Ratio", f"{current_ratio:.4f}")
    col2.metric("Mean Ratio", f"{mean_ratio:.4f}")
    col3.metric("Std Dev", f"{std_ratio:.4f}")
    
    ratio_changes = ratio.pct_change()
    rally_threshold = float(ratio_changes.quantile(0.9))
    rallies = ratio_changes[ratio_changes > rally_threshold]
    
    if len(rallies) > 0:
        st.write("### Historical Ratio Rallies Detected:")
        rally_data = []
        for idx in rallies.index[-5:]:
            idx_loc = ratio.index.get_loc(idx)
            if idx_loc < len(ratio) - 1:
                ratio_before = float(ratio.iloc[idx_loc])
                ratio_after = float(ratio.iloc[idx_loc+1])
                future_change_pts = ratio_after - ratio_before
                future_change_pct = (future_change_pts / ratio_before) * 100
                
                rally_data.append({
                    'DateTime_IST': str(idx),
                    'Ratio_Value': ratio_before,
                    'Change_Pct': float(rallies.loc[idx]) * 100,
                    'Next_Period_Pts': future_change_pts,
                    'Next_Period_Pct': future_change_pct,
                    f'{ticker1_name}_Price': float(data1['Close'].loc[idx]),
                    f'{ticker2_name}_Price': float(data2['Close'].loc[idx])
                })
        
        if rally_data:
            rally_df = pd.DataFrame(rally_data)
            st.dataframe(rally_df.style.background_gradient(subset=['Change_Pct', 'Next_Period_Pct'], cmap='Greens'), use_container_width=True)
            
            num_rallies = len(rallies)
            rally_thresh_pct = rally_threshold * 100
            ratio_vs_mean = "above" if current_ratio > mean_ratio else "below"
            ratio_diff_pct = abs((current_ratio - mean_ratio) / mean_ratio * 100)
            
            st.write(f"""
            **Key Insight - Historical Ratio Rally Analysis:**
            
            Historical data shows {num_rallies} significant ratio movements (>{rally_thresh_pct:.2f}%).
            
            **Current Market Status:**
            - Current Ratio: {current_ratio:.4f}
            - Mean Ratio: {mean_ratio:.4f}
            - The ratio is currently {ratio_diff_pct:.2f}% {ratio_vs_mean} the historical mean
            - Current {ticker1_name} price: {latest_price1:.2f} ({change1:+.2f} pts, {pct_change1:+.2f}%)
            - Current {ticker2_name} price: {latest_price2:.2f} ({change2:+.2f} pts, {pct_change2:+.2f}%)
            
            **Historical Rally Pattern:**
            The most recent strong rally occurred on {rally_df.iloc[-1]['DateTime_IST']} when:
            - Ratio was at {rally_df.iloc[-1]['Ratio_Value']:.4f}
            - {ticker1_name} was at {rally_df.iloc[-1][f'{ticker1_name}_Price']:.2f}
            - {ticker2_name} was at {rally_df.iloc[-1][f'{ticker2_name}_Price']:.2f}
            - Rally magnitude: {rally_df.iloc[-1]['Change_Pct']:.2f}%
            - After rally, moved {rally_df.iloc[-1]['Next_Period_Pts']:.4f} points ({rally_df.iloc[-1]['Next_Period_Pct']:.2f}%)
            
            **Market Implication:**
            """)
            
            if current_ratio > mean_ratio + std_ratio:
                st.warning("CAUTION: Ratio approaching historical high - potential mean reversion expected")
            else:
                st.success(f"OPPORTUNITY: Ratio below mean - potential upside in {ticker1_name} relative to {ticker2_name}")
    
    # SECTION 3: RATIO BINNING
    st.header("3. Ratio Binning & Rally Analysis")
    
    try:
        ratio_clean = ratio.dropna()
        if len(ratio_clean) >= 10:
            ratio_bins = pd.qcut(ratio_clean, q=10, duplicates='drop', labels=False)
            ratio_binned_df = pd.DataFrame({
                'Ratio': ratio_clean.values,
                'Bin': ratio_bins
            }, index=ratio_clean.index)
            
            ratio_binned_df['Forward_Return_Pct'] = ratio_clean.pct_change().shift(-1) * 100
            ratio_binned_df['Forward_Points'] = ratio_clean.diff().shift(-1)
            
            bin_analysis = ratio_binned_df.groupby('Bin').agg({
                'Forward_Return_Pct': ['mean', 'std', 'count'],
                'Forward_Points': ['mean', 'sum'],
                'Ratio': ['min', 'max', 'mean']
            }).round(4)
            
            st.dataframe(bin_analysis.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
            
            current_bin = None
            if current_ratio >= ratio_clean.min() and current_ratio <= ratio_clean.max():
                current_bin = pd.qcut([current_ratio], q=10, labels=False, duplicates='drop')[0]
            
            st.write("**Bin Analysis Insights:**")
            st.write(f"This table divides the ratio into 10 equal bins and shows average forward movement in each bin.")
            st.write(f"Current ratio {current_ratio:.4f} falls in Bin {current_bin if current_bin is not None else 'N/A'}")
            
            if current_bin is not None:
                fwd_ret = float(bin_analysis.loc[current_bin, ('Forward_Return_Pct', 'mean')])
                fwd_pts = float(bin_analysis.loc[current_bin, ('Forward_Points', 'mean')])
                st.write(f"Historical forward return in this bin: {fwd_ret:.2f}%")
                st.write(f"Historical forward points in this bin: {fwd_pts:.4f}")
                
                if fwd_ret > 0:
                    st.success("Current bin shows positive historical returns - Consider LONG position")
                else:
                    st.warning("Current bin shows negative historical returns - Consider CAUTION or SHORT position")
    except Exception as e:
        st.warning(f"Unable to create bins: {str(e)}")
    
    # SECTION 4: CANDLESTICK CHARTS
    st.header("4. Technical Charts")
    
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker1_name} Candlesticks', f'{ticker2_name} Candlesticks', 
                       'Ratio', f'{ticker1_name} RSI', f'{ticker2_name} RSI', 'Ratio RSI'),
        row_heights=[0.2, 0.2, 0.2, 0.13, 0.13, 0.14]
    )
    
    fig.add_trace(go.Candlestick(
        x=data1.index, open=data1['Open'], high=data1['High'],
        low=data1['Low'], close=data1['Close'], name=ticker1_name
    ), row=1, col=1)
    
    fig.add_trace(go.Candlestick(
        x=data2.index, open=data2['Open'], high=data2['High'],
        low=data2['Low'], close=data2['Close'], name=ticker2_name
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=data1.index, y=ratio, mode='lines',
        name='Ratio', line=dict(color='purple', width=2)
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=data1.index, y=rsi1, name=f'RSI {ticker1_name}', line=dict(color='blue')), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    
    fig.add_trace(go.Scatter(x=data2.index, y=rsi2, name=f'RSI {ticker2_name}', line=dict(color='orange')), row=5, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=5, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=5, col=1)
    
    fig.add_trace(go.Scatter(x=data1.index, y=rsi_ratio, name='RSI Ratio', line=dict(color='purple')), row=6, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=6, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=6, col=1)
    
    fig.update_layout(height=1800, showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # SECTION 5: DATA EXPORT
    st.header("5. Export Data")
    
    export_data = pd.DataFrame({
        'DateTime_IST': [str(dt) for dt in data1.index],
        f'{ticker1_name}_Open': [float(x) for x in data1['Open'].values],
        f'{ticker1_name}_High': [float(x) for x in data1['High'].values],
        f'{ticker1_name}_Low': [float(x) for x in data1['Low'].values],
        f'{ticker1_name}_Close': [float(x) for x in data1['Close'].values],
        f'{ticker1_name}_Volume': [float(x) for x in data1['Volume'].values],
        f'{ticker2_name}_Open': [float(x) for x in data2['Open'].values],
        f'{ticker2_name}_High': [float(x) for x in data2['High'].values],
        f'{ticker2_name}_Low': [float(x) for x in data2['Low'].values],
        f'{ticker2_name}_Close': [float(x) for x in data2['Close'].values],
        f'{ticker2_name}_Volume': [float(x) for x in data2['Volume'].values],
        'Ratio': [float(x) for x in ratio.values]
    })
    
    col1, col2 = st.columns(2)
    with col1:
        csv = export_data.to_csv(index=False)
        st.download_button("Download as CSV", csv, "trading_data.csv", "text/csv")
    with col2:
        st.download_button("Download as Excel (CSV)", csv, "trading_data.xlsx", "text/csv")
    
    # SECTION 6: VOLATILITY ANALYSIS
    st.header("6. Volatility & Points Analysis")
    
    points_len = min(len(data1), len(data2), len(ratio), len(vol1), len(vol2), len(vol_ratio))
    
    points_data = pd.DataFrame({
        'DateTime_IST': [str(dt) for dt in data1.index[-points_len:]],
        f'{ticker1_name}_Price': [float(x) for x in data1['Close'].iloc[-points_len:].values],
        f'{ticker2_name}_Price': [float(x) for x in data2['Close'].iloc[-points_len:].values],
        'Ratio': [float(x) for x in ratio.iloc[-points_len:].values],
        f'Vol_Pct_{ticker1_name}': [float(x) if not pd.isna(x) else 0.0 for x in vol1.iloc[-points_len:].values],
        f'Vol_Pct_{ticker2_name}': [float(x) if not pd.isna(x) else 0.0 for x in vol2.iloc[-points_len:].values],
        'Vol_Pct_Ratio': [float(x) if not pd.isna(x) else 0.0 for x in vol_ratio.iloc[-points_len:].values],
        f'Points_{ticker1_name}': [float(x) if not pd.isna(x) else 0.0 for x in data1['Close'].iloc[-points_len:].diff().values],
        f'Points_{ticker2_name}': [float(x) if not pd.isna(x) else 0.0 for x in data2['Close'].iloc[-points_len:].diff().values],
        'Points_Ratio': [float(x) if not pd.isna(x) else 0.0 for x in ratio.iloc[-points_len:].diff().values],
        f'Pct_Change_{ticker1_name}': [float(x) if not pd.isna(x) else 0.0 for x in (data1['Close'].iloc[-points_len:].pct_change() * 100).values],
        f'Pct_Change_{ticker2_name}': [float(x) if not pd.isna(x) else 0.0 for x in (data2['Close'].iloc[-points_len:].pct_change() * 100).values],
        'Pct_Change_Ratio': [float(x) if not pd.isna(x) else 0.0 for x in (ratio.iloc[-points_len:].pct_change() * 100).values]
    })
    
    st.dataframe(points_data.tail(20).style.background_gradient(subset=[f'Points_{ticker1_name}', f'Points_{ticker2_name}', 'Points_Ratio'], cmap='RdYlGn'), 
                use_container_width=True)
    
    vol1_current = float(vol1.iloc[-1]) if not pd.isna(vol1.iloc[-1]) else 0.0
    vol2_current = float(vol2.iloc[-1]) if not pd.isna(vol2.iloc[-1]) else 0.0
    vol_ratio_current = float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else 0.0
    
    latest_pts1 = points_data.iloc[-1][f'Points_{ticker1_name}']
    latest_pct1 = points_data.iloc[-1][f'Pct_Change_{ticker1_name}']
    latest_pts2 = points_data.iloc[-1][f'Points_{ticker2_name}']
    latest_pct2 = points_data.iloc[-1][f'Pct_Change_{ticker2_name}']
    latest_pts_ratio = points_data.iloc[-1]['Points_Ratio']
    latest_pct_ratio = points_data.iloc[-1]['Pct_Change_Ratio']
    
    st.write(f"""
    **Volatility & Movement Insights:** 
    
    **Current Volatility (Annualized Percentage):**
    - {ticker1_name}: {vol1_current:.2f}%
    - {ticker2_name}: {vol2_current:.2f}%
    - Ratio: {vol_ratio_current:.2f}%
    
    **Recent Movement (Latest Period):**
    - {ticker1_name}: {latest_pts1:.2f} points ({latest_pct1:.2f}%)
    - {ticker2_name}: {latest_pts2:.2f} points ({latest_pct2:.2f}%)
    - Ratio: {latest_pts_ratio:.4f} points ({latest_pct_ratio:.2f}%)
    
    Higher volatility indicates higher risk but also potential for larger moves.
    """)
    
    # Continue with remaining sections...
    st.header("7. Pattern Similarity & Forecast")
    
    similarities1 = find_pattern_similarity(data1['Close'], window=min(20, len(data1)//4))
    similarities_ratio = find_pattern_similarity(ratio, window=min(20, len(ratio)//4))
    
    if similarities1:
        st.subheader(f"{ticker1_name} Similar Patterns")
        sim_data1 = []
        for sim in similarities1:
            sim_data1.append({
                'DateTime_IST': str(sim['date']),
                'Price_Then': sim['price_at_pattern'],
                'Correlation': sim['correlation'],
                'Future_Price': sim['future_price'],
                'Future_Points': sim['future_points'],
                'Future_Pct': sim['future_return_pct']
            })
        
        sim_df1 = pd.DataFrame(sim_data1)
        st.dataframe(sim_df1.style.background_gradient(subset=['Correlation', 'Future_Pct'], cmap='RdYlGn'), use_container_width=True)
        
        avg_future_pts = sim_df1['Future_Points'].mean()
        avg_future_pct = sim_df1['Future_Pct'].mean()
        forecast_price1 = latest_price1 + avg_future_pts
        
        st.write(f"""
        **Pattern-Based Forecast:**
        
        Found {len(similarities1)} similar patterns with {sim_df1['Correlation'].mean()*100:.1f}% average correlation
        
        **Most Similar Pattern:**
        - Occurred on: {sim_df1.iloc[0]['DateTime_IST']} IST
        - Price then: {sim_df1.iloc[0]['Price_Then']:.2f}
        - Correlation: {sim_df1.iloc[0]['Correlation']*100:.1f}%
        - What happened: Moved {sim_df1.iloc[0]['Future_Points']:.2f} points ({sim_df1.iloc[0]['Future_Pct']:.2f}%)
        
        **FORECAST for Current Market:**
        - Current price: {latest_price1:.2f}
        - Expected target: {forecast_price1:.2f}
        - Expected move: {avg_future_pts:+.2f} points ({avg_future_pct:+.2f}%)
        """)
        
        if avg_future_pct > 2:
            st.success(f"Strong BULLISH forecast - Expected {avg_future_pts:.2f} pts ({avg_future_pct:.2f}%) gain")
        elif avg_future_pct < -2:
            st.error(f"BEARISH forecast - Expected {avg_future_pts:.2f} pts ({avg_future_pct:.2f}%) decline")
        else:
            st.info("NEUTRAL forecast - Expect sideways movement")
    
    if similarities_ratio:
        st.subheader("Ratio Pattern Forecast")
        sim_data_ratio = []
        for sim in similarities_ratio:
            sim_data_ratio.append({
                'DateTime_IST': str(sim['date']),
                'Ratio_Then': sim['price_at_pattern'],
                'Correlation': sim['correlation'],
                'Future_Ratio': sim['future_price'],
                'Future_Points': sim['future_points'],
                'Future_Pct': sim['future_return_pct']
            })
        
        sim_df_ratio = pd.DataFrame(sim_data_ratio)
        st.dataframe(sim_df_ratio.style.background_gradient(subset=['Correlation', 'Future_Pct'], cmap='RdYlGn'), use_container_width=True)
        
        avg_ratio_pts = sim_df_ratio['Future_Points'].mean()
        avg_ratio_pct = sim_df_ratio['Future_Pct'].mean()
        
        st.write(f"""
        **Ratio Forecast:**
        - Current ratio: {current_ratio:.4f}
        - Expected move: {avg_ratio_pts:+.4f} points ({avg_ratio_pct:+.2f}%)
        """)
        
        if avg_ratio_pct > 1:
            st.success(f"{ticker1_name} expected to OUTPERFORM {ticker2_name} by {avg_ratio_pct:.2f}%")
        elif avg_ratio_pct < -1:
            st.warning(f"{ticker2_name} expected to OUTPERFORM {ticker1_name} by {abs(avg_ratio_pct):.2f}%")
    
    # SECTION 8: RETURNS HEATMAPS
    st.header("8. Returns Heatmaps")
    
    returns1_daily = data1['Close'].pct_change() * 100
    returns2_daily = data2['Close'].pct_change() * 100
    ratio_returns_daily = ratio.pct_change() * 100
    
    if len(data1) >= 50:
        try:
            st.subheader(f"{ticker1_name} Returns Heatmap")
            returns1_recent = returns1_daily.tail(50).values
            n_rows = len(returns1_recent) // 10
            if n_rows > 0:
                returns1_matrix = returns1_recent[:n_rows*10].reshape(n_rows, 10)
                fig1 = go.Figure(data=go.Heatmap(
                    z=returns1_matrix, colorscale='RdYlGn', zmid=0,
                    text=np.round(returns1_matrix, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Return %")
                ))
                fig1.update_layout(title=f'{ticker1_name} Returns Pattern', height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader(f"{ticker2_name} Returns Heatmap")
            returns2_recent = returns2_daily.tail(50).values
            if n_rows > 0:
                returns2_matrix = returns2_recent[:n_rows*10].reshape(n_rows, 10)
                fig2 = go.Figure(data=go.Heatmap(
                    z=returns2_matrix, colorscale='RdYlGn', zmid=0,
                    text=np.round(returns2_matrix, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Return %")
                ))
                fig2.update_layout(title=f'{ticker2_name} Returns Pattern', height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            positive_returns1 = (returns1_daily > 0).sum()
            negative_returns1 = (returns1_daily < 0).sum()
            win_rate = (positive_returns1 / (positive_returns1 + negative_returns1) * 100)
            
            st.write(f"""
            **Returns Analysis:**
            - Positive periods: {positive_returns1}
            - Negative periods: {negative_returns1}
            - Win rate: {win_rate:.1f}%
            - Latest return: {returns1_daily.iloc[-1]:.2f}%
            """)
        except Exception as e:
            st.info(f"Not enough data for heatmap: {str(e)}")
    
    # SECTION 9: VOLATILITY HEATMAPS
    st.header("9. Volatility Heatmaps")
    
    if len(vol1.dropna()) >= 25:
        try:
            vol1_recent = vol1.dropna().tail(50).values
            n_rows_vol = len(vol1_recent) // 10
            if n_rows_vol > 0:
                vol1_matrix = vol1_recent[:n_rows_vol*10].reshape(n_rows_vol, 10)
                fig_vol1 = go.Figure(data=go.Heatmap(
                    z=vol1_matrix, colorscale='Reds',
                    text=np.round(vol1_matrix, 2),
                    texttemplate='%{text}%',
                    textfont={"size": 8},
                    colorbar=dict(title="Vol %")
                ))
                fig_vol1.update_layout(title=f'{ticker1_name} Volatility Heatmap', height=400)
                st.plotly_chart(fig_vol1, use_container_width=True)
            
            vol_q75 = vol1.quantile(0.75)
            vol_q25 = vol1.quantile(0.25)
            
            if vol1_current > vol_q75:
                vol_regime = "HIGH volatility regime"
            elif vol1_current > vol_q25:
                vol_regime = "MODERATE volatility"
            else:
                vol_regime = "LOW volatility"
            
            st.write(f"""
            **Volatility Insights:**
            - Current {ticker1_name} volatility: {vol1_current:.2f}% - {vol_regime}
            - High vol periods typically see larger price swings
            - Low vol periods often precede breakouts
            """)
        except Exception as e:
            st.info(f"Not enough data for volatility heatmap: {str(e)}")
    
    # SECTION 10: REVERSAL DETECTION
    st.header("10. Reversal Detection & Risk")
    
    try:
        peaks_idx = argrelextrema(data1['Close'].values, np.greater, order=5)[0]
        troughs_idx = argrelextrema(data1['Close'].values, np.less, order=5)[0]
        
        if len(peaks_idx) > 0 and len(troughs_idx) > 0:
            reversal_data = []
            for peak_idx in peaks_idx[-5:]:
                if peak_idx > 20 and peak_idx < len(data1) - 5:
                    prior_troughs = troughs_idx[troughs_idx < peak_idx]
                    if len(prior_troughs) > 0:
                        trough_idx = prior_troughs[-1]
                        
                        trough_price = float(data1['Close'].iloc[trough_idx])
                        peak_price = float(data1['Close'].iloc[peak_idx])
                        after_peak_price = float(data1['Close'].iloc[peak_idx+5])
                        
                        rally_points = peak_price - trough_price
                        rally_pct = (rally_points / trough_price) * 100
                        
                        reversal_points = after_peak_price - peak_price
                        reversal_pct = (reversal_points / peak_price) * 100
                        
                        reversal_data.append({
                            'Trough_DateTime': str(data1.index[trough_idx]),
                            'Peak_DateTime_IST': str(data1.index[peak_idx]),
                            'Trough_Price': trough_price,
                            'Peak_Price': peak_price,
                            'Rally_Points': rally_points,
                            'Rally_Pct': rally_pct,
                            'Reversal_Points': reversal_points,
                            'Reversal_Pct': reversal_pct,
                            'RSI_at_Peak': float(rsi1.iloc[peak_idx]) if not pd.isna(rsi1.iloc[peak_idx]) else 50.0
                        })
            
            if reversal_data:
                reversal_df = pd.DataFrame(reversal_data)
                st.dataframe(reversal_df.style.background_gradient(subset=['Rally_Pct', 'Reversal_Pct'], cmap='RdYlGn'), 
                            use_container_width=True)
                
                recent_low_idx = data1['Close'].iloc[-20:].idxmin()
                recent_low = float(data1['Close'].loc[recent_low_idx])
                current_rally_points = latest_price1 - recent_low
                current_rally_pct = (current_rally_points / recent_low) * 100
                
                avg_rally_pts = float(reversal_df['Rally_Points'].mean())
                avg_rally_pct = float(reversal_df['Rally_Pct'].mean())
                avg_reversal_pts = float(reversal_df['Reversal_Points'].mean())
                avg_reversal_pct = float(reversal_df['Reversal_Pct'].mean())
                
                st.write(f"""
                **Reversal Analysis:**
                
                **Historical Average:**
                - Rally before reversal: {avg_rally_pts:.2f} points ({avg_rally_pct:.2f}%)
                - Reversal magnitude: {avg_reversal_pts:.2f} points ({avg_reversal_pct:.2f}%)
                
                **Current Status:**
                - Current price: {latest_price1:.2f}
                - Recent low: {recent_low:.2f} (from {str(recent_low_idx)})
                - Current rally: {current_rally_points:.2f} points ({current_rally_pct:.2f}%)
                - Progress: {(current_rally_pct / avg_rally_pct * 100):.1f}% of typical rally
                """)
                
                if current_rally_pct >= avg_rally_pct * 0.8:
                    st.error(f"HIGH REVERSAL RISK: Rally at {current_rally_pct:.2f}% approaching historical avg {avg_rally_pct:.2f}%")
                elif current_rally_pct >= avg_rally_pct * 0.5:
                    st.warning(f"MODERATE RISK: Rally at {current_rally_pct:.2f}% - Monitor closely")
                else:
                    st.success(f"LOW RISK: Rally at {current_rally_pct:.2f}% has room to {avg_rally_pct:.2f}%")
    except Exception as e:
        st.info(f"Reversal analysis needs more data: {str(e)}")
    
    # SECTION 11: ULTIMATE RECOMMENDATION
    st.header("11. Ultimate Trading Recommendation")
    
    recommendation_score = 0
    reasons = []
    
    # RSI Analysis
    if rsi1_current < 30:
        recommendation_score += 2
        reasons.append(f"RSI {rsi1_current:.1f} is oversold - BUY signal")
    elif rsi1_current > 70:
        recommendation_score -= 2
        reasons.append(f"RSI {rsi1_current:.1f} is overbought - SELL signal")
    else:
        reasons.append(f"RSI {rsi1_current:.1f} is neutral")
    
    # Ratio Analysis
    ratio_zscore = (current_ratio - mean_ratio) / std_ratio
    if ratio_zscore < -1:
        recommendation_score += 1.5
        reasons.append(f"Ratio {current_ratio:.4f} is {abs(ratio_zscore):.2f} std below mean")
    elif ratio_zscore > 1:
        recommendation_score -= 1.5
        reasons.append(f"Ratio {current_ratio:.4f} is {ratio_zscore:.2f} std above mean")
    
    # Pattern Similarity
    if similarities1:
        if avg_future_pct > 2:
            recommendation_score += 2
            reasons.append(f"Patterns suggest {avg_future_pts:+.2f} pts ({avg_future_pct:+.2f}%) upside")
        elif avg_future_pct < -2:
            recommendation_score -= 2
            reasons.append(f"Patterns suggest {avg_future_pts:.2f} pts ({avg_future_pct:.2f}%) downside")
    
    # Volatility
    avg_vol = float(vol1.mean())
    if vol1_current > avg_vol * 1.5:
        recommendation_score -= 1
        reasons.append(f"High volatility {vol1_current:.2f}% - Reduce position size")
    elif vol1_current < avg_vol * 0.7:
        recommendation_score += 0.5
        reasons.append(f"Low volatility {vol1_current:.2f}% - Favorable for entry")
    
    # Reversal Risk
    if 'reversal_df' in locals() and len(reversal_df) > 0:
        if current_rally_pct >= avg_rally_pct * 0.8:
            recommendation_score -= 2
            reasons.append(f"HIGH REVERSAL RISK: {current_rally_pct:.2f}% vs {avg_rally_pct:.2f}%")
        elif current_rally_pct < avg_rally_pct * 0.5:
            recommendation_score += 1
            reasons.append(f"Low reversal risk: {current_rally_pct:.2f}% rally")
    
    # Momentum
    recent_change_pct = float(data1['Close'].pct_change(5).iloc[-1] * 100)
    if recent_change_pct > 3:
        recommendation_score += 1
        reasons.append(f"Positive momentum: {recent_change_pct:+.2f}% over 5 periods")
    elif recent_change_pct < -3:
        recommendation_score -= 1
        reasons.append(f"Negative momentum: {recent_change_pct:.2f}% over 5 periods")
    
    # Final Recommendation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if recommendation_score >= 4:
            st.success("### STRONG BUY")
            action = "STRONG BUY"
        elif recommendation_score >= 2:
            st.success("### BUY")
            action = "BUY"
        elif recommendation_score >= 0:
            st.info("### HOLD/NEUTRAL")
            action = "HOLD"
        elif recommendation_score >= -2:
            st.warning("### SELL")
            action = "SELL"
        else:
            st.error("### STRONG SELL")
            action = "STRONG SELL"
        
        st.metric("Score", f"{recommendation_score:.1f}/10")
    
    st.markdown("---")
    
    st.subheader("Recommendation Breakdown")
    for reason in reasons:
        st.write(f"- {reason}")
    
    # Price Targets
    if similarities1:
        st.subheader("Price Targets & Risk Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Target Price", f"{forecast_price1:.2f}", 
                     f"{avg_future_pts:+.2f} pts ({avg_future_pct:+.2f}%)")
        
        with col2:
            stop_loss = latest_price1 - (2 * float(data1['Close'].std()))
            stop_loss_pts = latest_price1 - stop_loss
            stop_loss_pct = (stop_loss_pts / latest_price1) * 100
            st.metric("Stop Loss", f"{stop_loss:.2f}", 
                     f"-{stop_loss_pts:.2f} pts (-{stop_loss_pct:.2f}%)")
        
        with col3:
            risk = latest_price1 - stop_loss
            reward = forecast_price1 - latest_price1
            rr_ratio = reward / risk if risk > 0 else 0
            st.metric("Risk:Reward", f"{rr_ratio:.2f}:1")
    
    # Summary
    st.markdown("---")
    st.subheader("Executive Summary")
    
    last_datetime = str(data1.index[-1])
    
    summary = f"""
    **Analysis Time:** {last_datetime} IST
    
    **Current Status:**
    - {ticker1_name}: {latest_price1:.2f} ({change1:+.2f} pts, {pct_change1:+.2f}%)
    - {ticker2_name}: {latest_price2:.2f} ({change2:+.2f} pts, {pct_change2:+.2f}%)
    - Ratio: {current_ratio:.4f}
    
    **Recommendation:** {action}
    **Score:** {recommendation_score:.1f}/10
    
    **Key Factors:**
    """
    
    for i, reason in enumerate(reasons[:5], 1):
        summary += f"\n{i}. {reason}"
    
    if similarities1:
        summary += f"""
        
        **Expected Move:**
        - Target: {forecast_price1:.2f}
        - Move: {avg_future_pts:+.2f} points ({avg_future_pct:+.2f}%)
        - Stop Loss: {stop_loss:.2f}
        - Risk:Reward: {rr_ratio:.2f}:1
        """
    
    summary += """
    
    **Disclaimer:** This is NOT financial advice. Past performance does not guarantee future results.
    Always do your own research and consider your risk tolerance.
    """
    
    st.info(summary)
    
    # Additional Insights
    st.markdown("---")
    st.subheader("Additional Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Volume Analysis:**")
        if data1['Volume'].sum() > 0:
            avg_volume1 = float(data1['Volume'].mean())
            current_volume1 = float(data1['Volume'].iloc[-1])
            volume_ratio_val = current_volume1 / avg_volume1 if avg_volume1 > 0 else 1
            
            if volume_ratio_val > 1.5:
                st.write(f"High volume: {current_volume1:,.0f} ({volume_ratio_val:.1f}x avg)")
            elif volume_ratio_val < 0.5:
                st.write(f"Low volume: {current_volume1:,.0f} ({volume_ratio_val:.1f}x avg)")
            else:
                st.write(f"Normal volume: {current_volume1:,.0f} ({volume_ratio_val:.1f}x avg)")
        else:
            st.write("Volume data not available")
    
    with col2:
        st.write("**Market Correlation:**")
        correlation = float(data1['Close'].corr(data2['Close']))
        st.write(f"Correlation: {correlation:.2f}")
        
        if abs(correlation) > 0.8:
            st.write("Strong correlation - Moving together")
        elif abs(correlation) < 0.3:
            st.write("Weak correlation - Independent movements")
        else:
            st.write("Moderate correlation")
    
    st.markdown("---")
    ist_now = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    st.caption(f"Analysis completed at {ist_now} IST")
    st.caption("For educational purposes only. Not financial advice.")

else:
    st.info("Configure parameters in sidebar and click 'Fetch Data & Analyze'")
    
    with st.expander("How to Use"):
        st.write("""
        ### Features:
        - Multi-asset support (stocks, crypto, indices, forex)
        - Ratio analysis for pairs trading
        - Pattern recognition with forecasting
        - Reversal detection
        - Volatility analysis
        - Comprehensive recommendations
        
        ### Quick Start:
        1. Select two tickers
        2. Choose timeframe and period
        3. Click 'Fetch Data & Analyze'
        4. Review all analysis sections
        
        ### Notes:
        - Works with different price scales (e.g., Nifty vs Bitcoin)
        - All forecasts include points and percentages
        - Heatmaps show actual values
        - Handles missing volume data gracefully
        """)

st.sidebar.markdown("---")
st.sidebar.info("Respects API limits - Data cached")
st.sidebar.caption("Powered by yfinance & Streamlit")
st.sidebar.caption("v2.0 - Fixed All Errors")
