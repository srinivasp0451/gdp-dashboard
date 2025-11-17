import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Advanced Trading Pattern Analyzer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #ff7f0e; margin-top: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .positive {color: #00ff00; font-weight: bold;}
    .negative {color: #ff0000; font-weight: bold;}
    .neutral {color: #ffaa00; font-weight: bold;}
    .explanation-box {background-color: #e8f4f8; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = None

# Ticker mappings
TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD/INR": "INR=X",
    "EUR/USD": "EURUSD=X",
    "Custom": "CUSTOM"
}

def convert_to_ist(df):
    """Convert dataframe index to IST timezone and remove timezone info for Excel compatibility"""
    try:
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
        # Remove timezone info for Excel compatibility
        df.index = df.index.tz_localize(None)
    except:
        pass
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volatility(data, window=20):
    """Calculate rolling volatility"""
    return data.pct_change().rolling(window=window).std() * 100

def find_pattern_matches(df, window_size=10, top_n=20):
    """Find similar patterns in historical data"""
    if len(df) < window_size * 2:
        return pd.DataFrame()
    
    closes = df['Close'].values
    current_pattern = closes[-window_size:]
    current_normalized = (current_pattern - current_pattern.min()) / (current_pattern.max() - current_pattern.min() + 1e-10)
    current_volatility = df['Close'].tail(window_size).pct_change().std() * 100
    
    matches = []
    
    for i in range(window_size, len(df) - window_size - 15):
        historical_pattern = closes[i-window_size:i]
        hist_normalized = (historical_pattern - historical_pattern.min()) / (historical_pattern.max() - historical_pattern.min() + 1e-10)
        
        distance = euclidean(current_normalized, hist_normalized)
        correlation, _ = pearsonr(current_normalized, hist_normalized)
        hist_volatility = df['Close'].iloc[i-window_size:i].pct_change().std() * 100
        
        # Calculate future movements for 1-6 candles
        future_points = []
        future_pct = []
        for j in range(1, 7):
            if i + j < len(df):
                points = closes[i+j] - closes[i]
                pct = (points / closes[i]) * 100
                future_points.append(points)
                future_pct.append(pct)
            else:
                future_points.append(0)
                future_pct.append(0)
        
        matches.append({
            'match_date': df.index[i],
            'match_price': closes[i],
            'distance': distance,
            'correlation': correlation,
            'volatility': hist_volatility,
            'current_volatility': current_volatility,
            'volatility_diff': abs(current_volatility - hist_volatility),
            'candle_1_points': future_points[0],
            'candle_1_pct': future_pct[0],
            'candle_2_points': future_points[1],
            'candle_2_pct': future_pct[1],
            'candle_3_points': future_points[2],
            'candle_3_pct': future_pct[2],
            'candle_4_points': future_points[3],
            'candle_4_pct': future_pct[3],
            'candle_5_points': future_points[4],
            'candle_5_pct': future_pct[4],
            'candle_6_points': future_points[5],
            'candle_6_pct': future_pct[5],
        })
    
    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df['score'] = (1 / (1 + matches_df['distance'])) * (1 / (1 + matches_df['volatility_diff'])) * matches_df['correlation']
        matches_df = matches_df.sort_values('score', ascending=False).head(top_n)
    
    return matches_df

def create_detailed_forecast(matches_df, current_price):
    """Create detailed forecast with clear explanations"""
    if matches_df.empty:
        return "Insufficient data for pattern matching."
    
    summary = []
    summary.append("### 📊 DETAILED PATTERN MATCHING FORECAST\n")
    summary.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    summary.append(f"**Current Price:** ₹{current_price:.2f}")
    summary.append(f"**Patterns Found:** {len(matches_df)} similar historical occurrences\n")
    
    # Calculate averages for each candle
    predictions = []
    for i in range(1, 7):
        avg_points = matches_df[f'candle_{i}_points'].mean()
        avg_pct = matches_df[f'candle_{i}_pct'].mean()
        target_price = current_price + avg_points
        bullish_prob = (matches_df[f'candle_{i}_points'] > 0).sum() / len(matches_df) * 100
        
        predictions.append({
            'candle': i,
            'points': avg_points,
            'pct': avg_pct,
            'target': target_price,
            'bullish_prob': bullish_prob
        })
    
    summary.append("### 🎯 EXPECTED MOVEMENTS (Based on Historical Patterns):\n")
    
    for pred in predictions:
        direction = "📈 UP" if pred['points'] > 0 else "📉 DOWN" if pred['points'] < 0 else "➡️ FLAT"
        summary.append(f"**Candle {pred['candle']}:** {direction}")
        summary.append(f"  - Points: {pred['points']:+.2f} ({pred['pct']:+.2f}%)")
        summary.append(f"  - Target Price: ₹{pred['target']:.2f}")
        summary.append(f"  - Bullish Probability: {pred['bullish_prob']:.1f}%\n")
    
    # Overall recommendation
    avg_3_candle_pct = sum([p['pct'] for p in predictions[:3]]) / 3
    avg_bullish_prob = sum([p['bullish_prob'] for p in predictions[:3]]) / 3
    
    summary.append("### 🎲 TRADING RECOMMENDATION:\n")
    
    if avg_3_candle_pct > 1.5 and avg_bullish_prob > 60:
        summary.append("🟢 **STRONG BUY SIGNAL**")
        summary.append(f"- Expected gain in next 3 candles: {avg_3_candle_pct:+.2f}%")
        summary.append(f"- Historical success rate: {avg_bullish_prob:.1f}%")
        summary.append("- **Action:** Consider entering LONG position with stop-loss below recent support")
    elif avg_3_candle_pct > 0.5 and avg_bullish_prob > 50:
        summary.append("🟢 **BUY SIGNAL**")
        summary.append(f"- Expected gain in next 3 candles: {avg_3_candle_pct:+.2f}%")
        summary.append(f"- Historical success rate: {avg_bullish_prob:.1f}%")
        summary.append("- **Action:** Favorable for LONG, use appropriate risk management")
    elif avg_3_candle_pct < -1.5 and avg_bullish_prob < 40:
        summary.append("🔴 **STRONG SELL SIGNAL**")
        summary.append(f"- Expected loss in next 3 candles: {avg_3_candle_pct:+.2f}%")
        summary.append(f"- Historical success rate: {avg_bullish_prob:.1f}%")
        summary.append("- **Action:** Consider SHORT position or exit LONG positions")
    elif avg_3_candle_pct < -0.5 and avg_bullish_prob < 50:
        summary.append("🔴 **SELL SIGNAL**")
        summary.append(f"- Expected loss in next 3 candles: {avg_3_candle_pct:+.2f}%")
        summary.append(f"- Historical success rate: {avg_bullish_prob:.1f}%")
        summary.append("- **Action:** Caution advised, reduce exposure")
    else:
        summary.append("🟡 **HOLD / WAIT SIGNAL**")
        summary.append(f"- Expected movement in next 3 candles: {avg_3_candle_pct:+.2f}%")
        summary.append(f"- Historical success rate: {avg_bullish_prob:.1f}%")
        summary.append("- **Action:** No clear directional bias, wait for better setup")
    
    return "\n".join(summary)

def analyze_volatility_bins(df):
    """Analyze returns based on volatility bins"""
    df = df.copy()
    df['Returns'] = df['Close'].diff()
    df['Returns_Pct'] = df['Close'].pct_change() * 100
    df['Volatility'] = calculate_volatility(df['Close'], window=20)
    
    # Remove NaN values
    df_clean = df.dropna()
    
    if len(df_clean) < 50:
        return None, "Insufficient data for volatility analysis"
    
    # Create volatility bins
    df_clean['Volatility_Bin'] = pd.qcut(df_clean['Volatility'], q=10, labels=False, duplicates='drop')
    
    # Analyze by volatility bins
    volatility_analysis = df_clean.groupby('Volatility_Bin').agg({
        'Volatility': ['min', 'max', 'mean'],
        'Returns': ['mean', 'sum'],
        'Returns_Pct': ['mean', 'std'],
        'Close': ['mean', 'min', 'max', 'count']
    }).round(2)
    
    volatility_analysis.columns = ['Vol_Min', 'Vol_Max', 'Vol_Mean', 'Avg_Points', 'Total_Points', 
                                    'Avg_Return_Pct', 'Std_Return', 'Avg_Price', 'Min_Price', 'Max_Price', 'Count']
    
    # Get current volatility and bin
    current_vol = df['Volatility'].iloc[-1]
    current_bin = df_clean['Volatility_Bin'].iloc[-1] if not df_clean.empty else None
    
    # Create human-readable summary
    summary = []
    summary.append("### 📊 VOLATILITY-BASED ANALYSIS\n")
    summary.append(f"**Current Volatility:** {current_vol:.2f}%")
    summary.append(f"**Current Bin:** {current_bin}\n")
    
    summary.append("### 🔍 KEY INSIGHTS FROM VOLATILITY ANALYSIS:\n")
    
    # Find best and worst performing volatility ranges
    best_bin = volatility_analysis['Avg_Return_Pct'].idxmax()
    worst_bin = volatility_analysis['Avg_Return_Pct'].idxmin()
    
    best_data = volatility_analysis.loc[best_bin]
    worst_data = volatility_analysis.loc[worst_bin]
    
    summary.append(f"**🟢 BEST PERFORMANCE ZONE:**")
    summary.append(f"- Volatility Range: {best_data['Vol_Min']:.2f}% - {best_data['Vol_Max']:.2f}%")
    summary.append(f"- Average Return: {best_data['Avg_Return_Pct']:+.2f}%")
    summary.append(f"- Average Points Gained: {best_data['Avg_Points']:+.2f}")
    summary.append(f"- Price Range: ₹{best_data['Min_Price']:.2f} - ₹{best_data['Max_Price']:.2f}")
    summary.append(f"- Historical Occurrences: {int(best_data['Count'])}\n")
    
    summary.append(f"**🔴 WORST PERFORMANCE ZONE:**")
    summary.append(f"- Volatility Range: {worst_data['Vol_Min']:.2f}% - {worst_data['Vol_Max']:.2f}%")
    summary.append(f"- Average Return: {worst_data['Avg_Return_Pct']:+.2f}%")
    summary.append(f"- Average Points Lost: {worst_data['Avg_Points']:+.2f}")
    summary.append(f"- Price Range: ₹{worst_data['Min_Price']:.2f} - ₹{worst_data['Max_Price']:.2f}")
    summary.append(f"- Historical Occurrences: {int(worst_data['Count'])}\n")
    
    if current_bin is not None and current_bin in volatility_analysis.index:
        current_data = volatility_analysis.loc[current_bin]
        summary.append(f"**📍 CURRENT VOLATILITY ZONE PERFORMANCE:**")
        summary.append(f"- Volatility Range: {current_data['Vol_Min']:.2f}% - {current_data['Vol_Max']:.2f}%")
        summary.append(f"- Historical Average Return: {current_data['Avg_Return_Pct']:+.2f}%")
        summary.append(f"- Historical Average Points: {current_data['Avg_Points']:+.2f}")
        summary.append(f"- Price Range in this zone: ₹{current_data['Min_Price']:.2f} - ₹{current_data['Max_Price']:.2f}")
        summary.append(f"- Sample Size: {int(current_data['Count'])} periods\n")
        
        if current_data['Avg_Return_Pct'] > 0.5:
            summary.append("✅ **Interpretation:** Historically, this volatility level has been associated with POSITIVE returns")
        elif current_data['Avg_Return_Pct'] < -0.5:
            summary.append("⚠️ **Interpretation:** Historically, this volatility level has been associated with NEGATIVE returns")
        else:
            summary.append("➡️ **Interpretation:** Historically, this volatility level has shown NEUTRAL performance")
    
    return volatility_analysis, "\n".join(summary)

def create_comprehensive_table(df):
    """Create comprehensive data table with yesterday's comparison"""
    df = df.copy()
    df['Returns_Points'] = df['Close'].diff()
    df['Returns_Pct'] = df['Close'].pct_change() * 100
    df['Volatility'] = calculate_volatility(df['Close'], window=20)
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Get last 30 rows
    display_df = df.tail(30).copy()
    
    return display_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns_Points', 'Returns_Pct', 'Volatility', 'RSI']]

def create_ratio_analysis(df1, df2, ticker1, ticker2):
    """Enhanced ratio analysis with volatility"""
    df1 = df1.copy()
    df2 = df2.copy()
    
    common_index = df1.index.intersection(df2.index)
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    
    ratio_df = pd.DataFrame({
        'Ratio': df1['Close'] / df2['Close'],
        'Ticker1': df1['Close'],
        'Ticker2': df2['Close']
    })
    
    ratio_df['Returns'] = ratio_df['Ticker1'].pct_change() * 100
    ratio_df['Ratio_Volatility'] = ratio_df['Ratio'].pct_change().rolling(window=20).std() * 100
    
    # Create ratio bins
    ratio_df['Ratio_Bin'] = pd.qcut(ratio_df['Ratio'], q=10, labels=False, duplicates='drop')
    
    # Analyze by ratio bins
    bin_analysis = ratio_df.groupby('Ratio_Bin').agg({
        'Ratio': ['min', 'max', 'mean'],
        'Ratio_Volatility': ['mean'],
        'Returns': ['mean', 'std', 'count'],
        'Ticker1': ['mean', 'min', 'max']
    }).round(4)
    
    bin_analysis.columns = ['Ratio_Min', 'Ratio_Max', 'Ratio_Mean', 'Avg_Volatility', 
                            'Avg_Return', 'Std_Return', 'Count', 'Avg_Price_T1', 'Min_Price_T1', 'Max_Price_T1']
    
    current_ratio = ratio_df['Ratio'].iloc[-1]
    current_volatility = ratio_df['Ratio_Volatility'].iloc[-1]
    current_bin = ratio_df['Ratio_Bin'].iloc[-1]
    
    summary = []
    summary.append(f"### 🔄 RATIO ANALYSIS: {ticker1} / {ticker2}\n")
    summary.append(f"**Current Ratio:** {current_ratio:.4f}")
    summary.append(f"**Current Ratio Volatility:** {current_volatility:.2f}%")
    summary.append(f"**Current Bin:** {current_bin}\n")
    
    if not bin_analysis.empty and current_bin in bin_analysis.index:
        current_data = bin_analysis.loc[current_bin]
        
        summary.append(f"**📊 CURRENT RATIO BIN ANALYSIS:**")
        summary.append(f"- Ratio Range: {current_data['Ratio_Min']:.4f} - {current_data['Ratio_Max']:.4f}")
        summary.append(f"- Average Volatility in this range: {current_data['Avg_Volatility']:.2f}%")
        summary.append(f"- Historical Average Return: {current_data['Avg_Return']:+.2f}%")
        summary.append(f"- Return Volatility: {current_data['Std_Return']:.2f}%")
        summary.append(f"- {ticker1} Price Range: ₹{current_data['Min_Price_T1']:.2f} - ₹{current_data['Max_Price_T1']:.2f}")
        summary.append(f"- Average {ticker1} Price: ₹{current_data['Avg_Price_T1']:.2f}")
        summary.append(f"- Sample Size: {int(current_data['Count'])} periods\n")
        
        if current_data['Avg_Return'] > 2:
            summary.append(f"✅ **STRONG BULLISH** - When ratio was in this range historically, {ticker1} gained an average of {current_data['Avg_Return']:.2f}%")
        elif current_data['Avg_Return'] > 0.5:
            summary.append(f"🟢 **BULLISH** - When ratio was in this range historically, {ticker1} gained an average of {current_data['Avg_Return']:.2f}%")
        elif current_data['Avg_Return'] < -2:
            summary.append(f"🔴 **STRONG BEARISH** - When ratio was in this range historically, {ticker1} lost an average of {current_data['Avg_Return']:.2f}%")
        elif current_data['Avg_Return'] < -0.5:
            summary.append(f"⚠️ **BEARISH** - When ratio was in this range historically, {ticker1} lost an average of {current_data['Avg_Return']:.2f}%")
        else:
            summary.append(f"➡️ **NEUTRAL** - When ratio was in this range historically, {ticker1} showed neutral performance")
    
    return "\n".join(summary), ratio_df, bin_analysis

def plot_comprehensive_charts(df, ticker_name, matches_df=None):
    """Create comprehensive charts with volume at bottom"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(f'{ticker_name} - Price & Patterns', 'RSI (14)', 'Returns %', 'Volume'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Add pattern matches
    if matches_df is not None and not matches_df.empty:
        for idx, match in matches_df.head(10).iterrows():
            match_date = match['match_date']
            if match_date in df.index:
                avg_3_candle = (match['candle_1_pct'] + match['candle_2_pct'] + match['candle_3_pct']) / 3
                color = "#00ff00" if avg_3_candle > 0 else "#ff0000"
                fig.add_annotation(
                    x=match_date,
                    y=df.loc[match_date, 'High'],
                    text=f"Match<br>+{avg_3_candle:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    bgcolor=color,
                    opacity=0.7,
                    font=dict(color="white", size=9),
                    row=1, col=1
                )
    
    # RSI
    rsi = calculate_rsi(df['Close'])
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name='RSI',
        line=dict(color='#9c27b0', width=2)
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # Returns
    returns = df['Close'].pct_change() * 100
    colors_returns = ['green' if x > 0 else 'red' for x in returns]
    fig.add_trace(go.Bar(
        x=df.index, y=returns, name='Returns %',
        marker_color=colors_returns
    ), row=3, col=1)
    
    # Volume at bottom
    colors_vol = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker_color=colors_vol, showlegend=False
    ), row=4, col=1)
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=1200, hovermode='x unified', template='plotly_white')
    
    return fig

def explain_chart_analysis(df, matches_df):
    """Provide human-readable explanation of charts"""
    explanation = []
    explanation.append("### 📖 CHART ANALYSIS EXPLANATION\n")
    
    # Price action
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    high_20 = df['High'].tail(20).max()
    low_20 = df['Low'].tail(20).min()
    price_position = ((current_price - low_20) / (high_20 - low_20)) * 100
    
    explanation.append(f"**📈 PRICE ACTION:**")
    explanation.append(f"- Latest candle moved {price_change:+.2f}% from previous candle")
    explanation.append(f"- Current price (₹{current_price:.2f}) is at {price_position:.1f}% of the 20-candle range")
    if price_position > 70:
        explanation.append(f"- Price is near the TOP of recent range - potential resistance\n")
    elif price_position < 30:
        explanation.append(f"- Price is near the BOTTOM of recent range - potential support\n")
    else:
        explanation.append(f"- Price is in the MIDDLE of recent range - no clear boundary\n")
    
    # Pattern matches explanation
    if matches_df is not None and not matches_df.empty:
        explanation.append(f"**🎯 PATTERN MATCHES (Blue/Green/Red markers on chart):**")
        explanation.append(f"- Found {len(matches_df)} historical instances with similar price patterns")
        explanation.append(f"- Markers show where similar patterns occurred and their outcomes")
        explanation.append(f"- Green markers = Bullish outcome | Red markers = Bearish outcome")
        
        best_match = matches_df.iloc[0]
        explanation.append(f"- Best match: {best_match['match_date'].strftime('%Y-%m-%d')} (Similarity: {best_match['correlation']:.2f})\n")
    
    # RSI explanation
    current_rsi = calculate_rsi(df['Close']).iloc[-1]
    explanation.append(f"**📉 RSI INDICATOR:**")
    explanation.append(f"- Current RSI: {current_rsi:.2f}")
    if current_rsi > 70:
        explanation.append(f"- Market is OVERBOUGHT - high probability of correction")
    elif current_rsi < 30:
        explanation.append(f"- Market is OVERSOLD - high probability of bounce")
    elif 40 < current_rsi < 60:
        explanation.append(f"- Market is in NEUTRAL zone - no extreme conditions")
    else:
        explanation.append(f"- Market is in TRANSITIONAL zone\n")
    
    # Volume analysis
    avg_volume = df['Volume'].tail(20).mean()
    current_volume = df['Volume'].iloc[-1]
    volume_ratio = (current_volume / avg_volume) * 100
    
    explanation.append(f"\n**📊 VOLUME ANALYSIS:**")
    explanation.append(f"- Latest volume: {current_volume:,.0f}")
    explanation.append(f"- 20-period average: {avg_volume:,.0f}")
    explanation.append(f"- Current volume is {volume_ratio:.1f}% of average")
    if volume_ratio > 150:
        explanation.append(f"- HIGH VOLUME spike - indicates strong conviction in current move")
    elif volume_ratio < 70:
        explanation.append(f"- LOW VOLUME - indicates weak participation, move may not sustain")
    else:
        explanation.append(f"- NORMAL VOLUME - average market participation")
    
    return "\n".join(explanation)

def create_returns_volatility_heatmaps(df):
    """Create returns and volatility heatmaps"""
    df = df.copy()
    df['Returns'] = df['Close'].pct_change() * 100
    df['Volatility'] = calculate_volatility(df['Close'], window=20)
    
    # Add time components
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Hour'] = df.index.hour
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    explanations = []
    figures = []
    
    # Day vs Month Returns Heatmap
    if len(df) > 30:
        pivot_day_month_returns = df.groupby(['DayOfWeek', 'Month'])['Returns'].mean().reset_index()
        pivot_day_month_returns = pivot_day_month_returns.pivot(index='DayOfWeek', columns='Month', values='Returns')
        pivot_day_month_returns.index = [day_names[i] for i in pivot_day_month_returns.index]
        pivot_day_month_returns.columns = [month_names[i-1] for i in pivot_day_month_returns.columns]
        
        fig1 = go.Figure(data=go.Heatmap(
            z=pivot_day_month_returns.values,
            x=pivot_day_month_returns.columns,
            y=pivot_day_month_returns.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_day_month_returns.values, 2),
            texttemplate='%{text:.2f}%',
            textfont={"size": 10},
            colorbar=dict(title="Avg Return %")
        ))
        fig1.update_layout(
            title="Day of Week vs Month - Average Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Day of Week",
            height=500
        )
        figures.append(('day_month_returns', fig1))
        
        # Explanation
        best_combo = pivot_day_month_returns.stack().idxmax()
        worst_combo = pivot_day_month_returns.stack().idxmin()
        best_return = pivot_day_month_returns.stack().max()
        worst_return = pivot_day_month_returns.stack().min()
        
        explanations.append(f"**📅 DAY vs MONTH RETURNS ANALYSIS:**")
        explanations.append(f"- Best Combination: {best_combo[0]} in {best_combo[1]} with average return of {best_return:+.2f}%")
        explanations.append(f"- Worst Combination: {worst_combo[0]} in {worst_combo[1]} with average return of {worst_return:+.2f}%")
        explanations.append(f"- Interpretation: Historically, {best_combo[0]}s during {best_combo[1]} showed strongest performance\n")
    
    # Day vs Month Volatility Heatmap
    if len(df) > 30:
        pivot_day_month_vol = df.groupby(['DayOfWeek', 'Month'])['Volatility'].mean().reset_index()
        pivot_day_month_vol = pivot_day_month_vol.pivot(index='DayOfWeek', columns='Month', values='Volatility')
        pivot_day_month_vol.index = [day_names[i] for i in pivot_day_month_vol.index]
        pivot_day_month_vol.columns = [month_names[i-1] for i in pivot_day_month_vol.columns]
        
        fig2 = go.Figure(data=go.Heatmap(
            z=pivot_day_month_vol.values,
            x=pivot_day_month_vol.columns,
            y=pivot_day_month_vol.index,
            colorscale='Reds',
            text=np.round(pivot_day_month_vol.values, 2),
            texttemplate='%{text:.2f}%',
            textfont={"size": 10},
            colorbar=dict(title="Avg Volatility %")
        ))
        fig2.update_layout(
            title="Day of Week vs Month - Average Volatility Heatmap",
            xaxis_title="Month",
            yaxis_title="Day of Week",
            height=500
        )
        figures.append(('day_month_volatility', fig2))
        
        # Explanation
        highest_vol_combo = pivot_day_month_vol.stack().idxmax()
        lowest_vol_combo = pivot_day_month_vol.stack().idxmin()
        highest_vol = pivot_day_month_vol.stack().max()
        lowest_vol = pivot_day_month_vol.stack().min()
        
        explanations.append(f"**📊 DAY vs MONTH VOLATILITY ANALYSIS:**")
        explanations.append(f"- Highest Volatility: {highest_vol_combo[0]} in {highest_vol_combo[1]} ({highest_vol:.2f}%)")
        explanations.append(f"- Lowest Volatility: {lowest_vol_combo[0]} in {lowest_vol_combo[1]} ({lowest_vol:.2f}%)")
        explanations.append(f"- Interpretation: {highest_vol_combo[0]}s in {highest_vol_combo[1]} are most volatile - higher risk/reward\n")
    
    # Month vs Year Returns Heatmap
    if len(df) > 365:
        pivot_month_year_returns = df.groupby(['Year', 'Month'])['Returns'].sum().reset_index()
        pivot_month_year_returns = pivot_month_year_returns.pivot(index='Year', columns='Month', values='Returns')
        pivot_month_year_returns.columns = [month_names[i-1] for i in pivot_month_year_returns.columns]
        
        fig3 = go.Figure(data=go.Heatmap(
            z=pivot_month_year_returns.values,
            x=pivot_month_year_returns.columns,
            y=pivot_month_year_returns.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_month_year_returns.values, 2),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Monthly Return %")
        ))
        fig3.update_layout(
            title="Month vs Year - Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            height=500
        )
        figures.append(('month_year_returns', fig3))
        
        # Explanation
        monthly_avg = pivot_month_year_returns.mean()
        best_month = monthly_avg.idxmax()
        worst_month = monthly_avg.idxmin()
        
        explanations.append(f"**📈 MONTH vs YEAR RETURNS ANALYSIS:**")
        explanations.append(f"- Best Performing Month: {best_month} (Average: {monthly_avg[best_month]:+.2f}%)")
        explanations.append(f"- Worst Performing Month: {worst_month} (Average: {monthly_avg[worst_month]:+.2f}%)")
        
        yearly_returns = pivot_month_year_returns.sum(axis=1)
        best_year = yearly_returns.idxmax()
        worst_year = yearly_returns.idxmin()
        explanations.append(f"- Best Year: {best_year} ({yearly_returns[best_year]:+.2f}%)")
        explanations.append(f"- Worst Year: {worst_year} ({yearly_returns[worst_year]:+.2f}%)")
        explanations.append(f"- Interpretation: Historically, {best_month} has been the strongest month for trading\n")
    
    # Month vs Year Volatility Heatmap
    if len(df) > 365:
        pivot_month_year_vol = df.groupby(['Year', 'Month'])['Volatility'].mean().reset_index()
        pivot_month_year_vol = pivot_month_year_vol.pivot(index='Year', columns='Month', values='Volatility')
        pivot_month_year_vol.columns = [month_names[i-1] for i in pivot_month_year_vol.columns]
        
        fig4 = go.Figure(data=go.Heatmap(
            z=pivot_month_year_vol.values,
            x=pivot_month_year_vol.columns,
            y=pivot_month_year_vol.index,
            colorscale='Reds',
            text=np.round(pivot_month_year_vol.values, 2),
            texttemplate='%{text:.2f}%',
            textfont={"size": 10},
            colorbar=dict(title="Avg Volatility %")
        ))
        fig4.update_layout(
            title="Month vs Year - Average Volatility Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            height=500
        )
        figures.append(('month_year_volatility', fig4))
        
        # Explanation
        monthly_vol_avg = pivot_month_year_vol.mean()
        most_volatile_month = monthly_vol_avg.idxmax()
        least_volatile_month = monthly_vol_avg.idxmin()
        
        explanations.append(f"**📊 MONTH vs YEAR VOLATILITY ANALYSIS:**")
        explanations.append(f"- Most Volatile Month: {most_volatile_month} ({monthly_vol_avg[most_volatile_month]:.2f}%)")
        explanations.append(f"- Least Volatile Month: {least_volatile_month} ({monthly_vol_avg[least_volatile_month]:.2f}%)")
        
        yearly_vol = pivot_month_year_vol.mean(axis=1)
        most_volatile_year = yearly_vol.idxmax()
        explanations.append(f"- Most Volatile Year: {most_volatile_year} ({yearly_vol[most_volatile_year]:.2f}%)")
        explanations.append(f"- Interpretation: {most_volatile_month} typically shows highest volatility - use caution")
    
    return figures, "\n".join(explanations)

def plot_ratio_charts_enhanced(ratio_df, ticker1, ticker2):
    """Enhanced ratio charts with volume"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker1} Price', f'{ticker2} Price', f'Ratio: {ticker1}/{ticker2}', 'Ratio Volatility'),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Ticker 1
    fig.add_trace(go.Scatter(
        x=ratio_df.index, y=ratio_df['Ticker1'],
        name=ticker1, line=dict(color='#2196F3', width=2),
        fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.1)'
    ), row=1, col=1)
    
    # Ticker 2
    fig.add_trace(go.Scatter(
        x=ratio_df.index, y=ratio_df['Ticker2'],
        name=ticker2, line=dict(color='#FF9800', width=2),
        fill='tozeroy', fillcolor='rgba(255, 152, 0, 0.1)'
    ), row=2, col=1)
    
    # Ratio
    fig.add_trace(go.Scatter(
        x=ratio_df.index, y=ratio_df['Ratio'],
        name='Ratio', line=dict(color='#4CAF50', width=2),
        fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.1)'
    ), row=3, col=1)
    
    # Ratio Volatility
    fig.add_trace(go.Scatter(
        x=ratio_df.index, y=ratio_df['Ratio_Volatility'],
        name='Ratio Volatility', line=dict(color='#9C27B0', width=2),
        fill='tozeroy', fillcolor='rgba(156, 39, 176, 0.1)'
    ), row=4, col=1)
    
    fig.update_layout(height=1000, hovermode='x unified', template='plotly_white')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=3, col=1)
    fig.update_yaxes(title_text="Volatility %", row=4, col=1)
    
    return fig

def explain_ratio_charts(ratio_df, ticker1, ticker2):
    """Explain ratio charts in human-readable form"""
    explanation = []
    explanation.append(f"### 📖 RATIO CHART EXPLANATION\n")
    
    current_ratio = ratio_df['Ratio'].iloc[-1]
    avg_ratio = ratio_df['Ratio'].mean()
    ratio_std = ratio_df['Ratio'].std()
    
    explanation.append(f"**🔄 RATIO OVERVIEW:**")
    explanation.append(f"- Current Ratio: {current_ratio:.4f}")
    explanation.append(f"- Historical Average: {avg_ratio:.4f}")
    explanation.append(f"- Standard Deviation: {ratio_std:.4f}")
    
    if current_ratio > avg_ratio + ratio_std:
        explanation.append(f"- Status: Ratio is ABOVE average by >1 std dev - {ticker1} is relatively EXPENSIVE vs {ticker2}")
        explanation.append(f"- Implication: Consider favoring {ticker2} over {ticker1}, or wait for ratio to decrease\n")
    elif current_ratio < avg_ratio - ratio_std:
        explanation.append(f"- Status: Ratio is BELOW average by >1 std dev - {ticker1} is relatively CHEAP vs {ticker2}")
        explanation.append(f"- Implication: Consider favoring {ticker1} over {ticker2}, potential value opportunity\n")
    else:
        explanation.append(f"- Status: Ratio is near historical average - fair relative valuation\n")
    
    # Volatility explanation
    current_vol = ratio_df['Ratio_Volatility'].iloc[-1]
    avg_vol = ratio_df['Ratio_Volatility'].mean()
    
    explanation.append(f"**📊 RATIO VOLATILITY:**")
    explanation.append(f"- Current Volatility: {current_vol:.2f}%")
    explanation.append(f"- Average Volatility: {avg_vol:.2f}%")
    
    if current_vol > avg_vol * 1.5:
        explanation.append(f"- Status: HIGH volatility period - ratio is fluctuating significantly")
        explanation.append(f"- Implication: Increased uncertainty, higher risk but also potential for mean reversion")
    elif current_vol < avg_vol * 0.5:
        explanation.append(f"- Status: LOW volatility period - ratio is stable")
        explanation.append(f"- Implication: Lower risk environment, smaller expected movements")
    else:
        explanation.append(f"- Status: NORMAL volatility period")
    
    return "\n".join(explanation)

# Main App
st.markdown('<p class="main-header">📊 Advanced Trading Pattern Analyzer - Enhanced</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker_choice = st.selectbox("Select Asset", list(TICKER_MAP.keys()))
    
    if ticker_choice == "Custom":
        custom_ticker = st.text_input("Enter Ticker Symbol")
        ticker_symbol = custom_ticker.upper()
    else:
        ticker_symbol = TICKER_MAP[ticker_choice]
    
    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Timeframe", ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'], index=7)
    with col2:
        period = st.selectbox("Period", ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y'], index=6)
    
    fetch_button = st.button("🔄 Fetch Data & Analyze", type="primary", use_container_width=True)
    
    st.divider()
    
    st.subheader("🔄 Ratio Analysis")
    enable_ratio = st.checkbox("Enable Ratio Analysis")
    
    if enable_ratio:
        ticker2_choice = st.selectbox("Select Second Asset", list(TICKER_MAP.keys()), key='ticker2')
        if ticker2_choice == "Custom":
            custom_ticker2 = st.text_input("Enter Second Ticker", key='custom2')
            ticker2_symbol = custom_ticker2.upper()
        else:
            ticker2_symbol = TICKER_MAP[ticker2_choice]

if fetch_button:
    if not ticker_symbol:
        st.error("Please enter a valid ticker symbol!")
    else:
        with st.spinner(f"📥 Fetching data for {ticker_symbol}..."):
            try:
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    st.error("No data returned. Please check the ticker symbol.")
                else:
                    df = convert_to_ist(df)
                    st.session_state.df = df
                    st.session_state.ticker_symbol = ticker_symbol
                    st.session_state.data_fetched = True
                    st.success(f"✅ Data fetched! {len(df)} data points")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.data_fetched = False

if st.session_state.data_fetched and st.session_state.df is not None:
    df = st.session_state.df
    ticker_symbol = st.session_state.ticker_symbol
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
    with col2:
        st.metric("High", f"₹{df['High'].iloc[-1]:.2f}")
    with col3:
        st.metric("Low", f"₹{df['Low'].iloc[-1]:.2f}")
    with col4:
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Pattern Analysis", "📈 Data Table", "🔥 Volatility Analysis", 
        "🔄 Ratio Analysis", "📉 RSI Analysis", "📅 Returns & Vol Heatmaps", "💾 Export"
    ])
    
    with tab1:
        st.markdown('<p class="sub-header">Pattern Matching & Forecast</p>', unsafe_allow_html=True)
        
        with st.spinner("🔍 Searching patterns..."):
            matches_df = find_pattern_matches(df, window_size=10, top_n=20)
            
            if not matches_df.empty:
                forecast = create_detailed_forecast(matches_df, current_price)
                st.markdown(forecast)
                
                st.divider()
                
                st.subheader("🎯 Detailed Pattern Matches")
                display_cols = ['match_date', 'match_price', 'correlation', 'volatility',
                               'candle_1_points', 'candle_1_pct', 'candle_2_points', 'candle_2_pct',
                               'candle_3_points', 'candle_3_pct', 'candle_4_points', 'candle_4_pct',
                               'candle_5_points', 'candle_5_pct', 'candle_6_points', 'candle_6_pct']
                
                display_matches = matches_df[display_cols].head(15).copy()
                display_matches.columns = ['Date', 'Price', 'Correlation', 'Volatility%',
                                          'C1_Pts', 'C1_%', 'C2_Pts', 'C2_%',
                                          'C3_Pts', 'C3_%', 'C4_Pts', 'C4_%',
                                          'C5_Pts', 'C5_%', 'C6_Pts', 'C6_%']
                
                st.dataframe(
                    display_matches.style.format({
                        'Price': '₹{:.2f}',
                        'Correlation': '{:.3f}',
                        'Volatility%': '{:.2f}',
                        'C1_Pts': '{:+.2f}',
                        'C1_%': '{:+.2f}',
                        'C2_Pts': '{:+.2f}',
                        'C2_%': '{:+.2f}',
                        'C3_Pts': '{:+.2f}',
                        'C3_%': '{:+.2f}',
                        'C4_Pts': '{:+.2f}',
                        'C4_%': '{:+.2f}',
                        'C5_Pts': '{:+.2f}',
                        'C5_%': '{:+.2f}',
                        'C6_Pts': '{:+.2f}',
                        'C6_%': '{:+.2f}'
                    }).background_gradient(subset=['Correlation'], cmap='RdYlGn'),
                    use_container_width=True,
                    height=400
                )
                
                st.divider()
                
                st.subheader("📉 Interactive Charts")
                fig = plot_comprehensive_charts(df, ticker_symbol, matches_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Chart explanation
                chart_explanation = explain_chart_analysis(df, matches_df)
                with st.expander("📖 Detailed Chart Explanation", expanded=True):
                    st.markdown(f'<div class="explanation-box">{chart_explanation}</div>', unsafe_allow_html=True)
            else:
                st.warning("Not enough data for pattern matching.")
    
    with tab2:
        st.markdown('<p class="sub-header">Comprehensive Data Table</p>', unsafe_allow_html=True)
        
        data_table = create_comprehensive_table(df)
        
        st.dataframe(
            data_table.style.format({
                'Open': '₹{:.2f}',
                'High': '₹{:.2f}',
                'Low': '₹{:.2f}',
                'Close': '₹{:.2f}',
                'Volume': '{:,.0f}',
                'Returns_Points': '{:+.2f}',
                'Returns_Pct': '{:+.2f}%',
                'Volatility': '{:.2f}%',
                'RSI': '{:.2f}'
            }).background_gradient(subset=['Returns_Pct'], cmap='RdYlGn')
              .background_gradient(subset=['Volatility'], cmap='Reds'),
            use_container_width=True,
            height=600
        )
        
        # Summary
        with st.expander("📊 Data Summary", expanded=True):
            summary_text = f"""
            **Data Summary:**
            - Total Periods: {len(df)}
            - Highest Close: ₹{df['Close'].max():.2f}
            - Lowest Close: ₹{df['Close'].min():.2f}
            - Average Volume: {df['Volume'].mean():,.0f}
            - Current Volatility: {calculate_volatility(df['Close']).iloc[-1]:.2f}%
            - Current RSI: {calculate_rsi(df['Close']).iloc[-1]:.2f}
            """
            st.markdown(summary_text)
    
    with tab3:
        st.markdown('<p class="sub-header">Volatility-Based Analysis</p>', unsafe_allow_html=True)
        
        vol_analysis, vol_summary = analyze_volatility_bins(df)
        
        if vol_analysis is not None:
            st.markdown(vol_summary)
            
            st.divider()
            
            st.subheader("📊 Volatility Bins Performance Table")
            st.dataframe(
                vol_analysis.style.format({
                    'Vol_Min': '{:.2f}%',
                    'Vol_Max': '{:.2f}%',
                    'Vol_Mean': '{:.2f}%',
                    'Avg_Points': '{:+.2f}',
                    'Total_Points': '{:+.2f}',
                    'Avg_Return_Pct': '{:+.2f}%',
                    'Std_Return': '{:.2f}%',
                    'Avg_Price': '₹{:.2f}',
                    'Min_Price': '₹{:.2f}',
                    'Max_Price': '₹{:.2f}',
                    'Count': '{:.0f}'
                }).background_gradient(subset=['Avg_Return_Pct'], cmap='RdYlGn')
                  .background_gradient(subset=['Vol_Mean'], cmap='Reds'),
                use_container_width=True
            )
            
            with st.expander("📖 How to Read This Table", expanded=True):
                st.markdown("""
                **Understanding Volatility Bins:**
                - Each row represents a volatility range (bin)
                - **Vol_Min/Max/Mean:** The volatility range for this bin
                - **Avg_Points:** Average points gained/lost when volatility was in this range
                - **Avg_Return_Pct:** Average percentage return in this volatility range
                - **Price Range:** The price levels historically seen at this volatility
                - **Count:** Number of times this volatility level occurred
                
                **Key Insights:**
                - Green cells = Positive returns historically
                - Red cells = Negative returns historically
                - Higher volatility bins = More risk and potential reward
                - Current bin shows where we are now and expected behavior
                """)
        else:
            st.info("Insufficient data for volatility analysis")
    
    with tab4:
        st.markdown('<p class="sub-header">Ratio Analysis</p>', unsafe_allow_html=True)
        
        if enable_ratio and ticker2_symbol:
            with st.spinner(f"📥 Fetching {ticker2_symbol}..."):
                try:
                    ticker2 = yf.Ticker(ticker2_symbol)
                    df2 = ticker2.history(period=period, interval=interval)
                    df2 = convert_to_ist(df2)
                    
                    if not df2.empty:
                        ratio_summary, ratio_df, bin_analysis = create_ratio_analysis(df, df2, ticker_symbol, ticker2_symbol)
                        
                        st.markdown(ratio_summary)
                        
                        st.divider()
                        
                        st.subheader("📊 Ratio Bins Performance Table")
                        st.dataframe(
                            bin_analysis.style.format({
                                'Ratio_Min': '{:.4f}',
                                'Ratio_Max': '{:.4f}',
                                'Ratio_Mean': '{:.4f}',
                                'Avg_Volatility': '{:.2f}%',
                                'Avg_Return': '{:+.2f}%',
                                'Std_Return': '{:.2f}%',
                                'Count': '{:.0f}',
                                'Avg_Price_T1': '₹{:.2f}',
                                'Min_Price_T1': '₹{:.2f}',
                                'Max_Price_T1': '₹{:.2f}'
                            }).background_gradient(subset=['Avg_Return'], cmap='RdYlGn')
                              .background_gradient(subset=['Avg_Volatility'], cmap='Reds'),
                            use_container_width=True
                        )
                        
                        st.divider()
                        
                        st.subheader("📈 Ratio Charts")
                        fig_ratio = plot_ratio_charts_enhanced(ratio_df, ticker_symbol, ticker2_symbol)
                        st.plotly_chart(fig_ratio, use_container_width=True)
                        
                        ratio_chart_explanation = explain_ratio_charts(ratio_df, ticker_symbol, ticker2_symbol)
                        with st.expander("📖 Ratio Chart Explanation", expanded=True):
                            st.markdown(f'<div class="explanation-box">{ratio_chart_explanation}</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"No data for {ticker2_symbol}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("👈 Enable ratio analysis in sidebar")
    
    with tab5:
        st.markdown('<p class="sub-header">RSI Analysis</p>', unsafe_allow_html=True)
        
        df_rsi = df.copy()
        df_rsi['RSI'] = calculate_rsi(df_rsi['Close'])
        df_rsi['Future_Returns'] = df_rsi['Close'].pct_change(5).shift(-5) * 100
        
        rsi_bins = [0, 30, 40, 50, 60, 70, 100]
        rsi_labels = ['Oversold<30', 'Weak30-40', 'Neutral-40-50', 'Neutral+50-60', 'Strong60-70', 'Overbought>70']
        df_rsi['RSI_Bin'] = pd.cut(df_rsi['RSI'], bins=rsi_bins, labels=rsi_labels, include_lowest=True)
        
        rsi_analysis = df_rsi.groupby('RSI_Bin', observed=True)['Future_Returns'].agg(['mean', 'std', 'count']).round(2)
        
        current_rsi = df_rsi['RSI'].iloc[-1]
        current_bin = df_rsi['RSI_Bin'].iloc[-1]
        
        st.markdown(f"**Current RSI:** {current_rsi:.2f} ({current_bin})")
        
        st.dataframe(
            rsi_analysis.style.format({
                'mean': '{:+.2f}%',
                'std': '{:.2f}%',
                'count': '{:.0f}'
            }).background_gradient(subset=['mean'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # RSI chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_rsi.index, y=df_rsi['RSI'],
            name='RSI', line=dict(color='#9C27B0', width=2),
            fill='tozeroy', fillcolor='rgba(156, 39, 176, 0.1)'
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
        
        fig_rsi.update_layout(
            title="RSI (14) Timeline",
            xaxis_title="Date",
            yaxis_title="RSI",
            height=400,
            yaxis_range=[0, 100],
            template='plotly_white'
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        with st.expander("📖 RSI Explanation", expanded=True):
            st.markdown("""
            **RSI (Relative Strength Index):**
            - Momentum oscillator measuring speed and change of price movements
            - Scale: 0-100
            - **>70:** Overbought - potential sell signal
            - **<30:** Oversold - potential buy signal
            - **40-60:** Neutral zone
            
            **Table shows:** Historical returns 5 periods later based on RSI level at entry
            """)
    
    with tab6:
        st.markdown('<p class="sub-header">Returns & Volatility Heatmaps</p>', unsafe_allow_html=True)
        
        with st.spinner("Creating heatmaps..."):
            heatmap_figures, heatmap_explanations = create_returns_volatility_heatmaps(df)
            
            if heatmap_figures:
                st.markdown(heatmap_explanations)
                
                st.divider()
                
                for fig_name, fig in heatmap_figures:
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()
                
                with st.expander("📖 How to Read Heatmaps", expanded=True):
                    st.markdown("""
                    **Returns Heatmaps (Green/Red):**
                    - **Green cells:** Positive returns in that time period
                    - **Red cells:** Negative returns in that time period
                    - **Darker colors:** Stronger performance (positive or negative)
                    - Use to identify seasonal patterns and best trading times
                    
                    **Volatility Heatmaps (Red shades):**
                    - **Darker red:** Higher volatility (more risk/opportunity)
                    - **Lighter red:** Lower volatility (more stable)
                    - Use to identify risky periods and adjust position sizing
                    
                    **Practical Use:**
                    - Enter positions during historically positive periods
                    - Avoid/reduce exposure during negative periods
                    - Increase position size during low volatility
                    - Decrease position size during high volatility
                    """)
            else:
                st.info("Need more data for heatmap analysis (minimum 30 days)")
    
    with tab7:
        st.markdown('<p class="sub-header">Data Export & Statistics</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Summary Statistics")
            
            total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            max_price = df['High'].max()
            min_price = df['Low'].min()
            avg_volume = df['Volume'].mean()
            current_vol = calculate_volatility(df['Close']).iloc[-1]
            
            stats_text = f"""
            **Performance Metrics:**
            - Total Return: {total_return:+.2f}%
            - Maximum Price: ₹{max_price:.2f}
            - Minimum Price: ₹{min_price:.2f}
            - Current Volatility: {current_vol:.2f}%
            - Average Volume: {avg_volume:,.0f}
            - Data Points: {len(df):,}
            
            **Date Range:**
            - Start: {df.index[0].strftime('%Y-%m-%d %H:%M')}
            - End: {df.index[-1].strftime('%Y-%m-%d %H:%M')}
            """
            st.markdown(stats_text)
        
        with col2:
            st.subheader("📈 Key Metrics")
            
            positive_days = (df['Close'].pct_change() > 0).sum()
            negative_days = (df['Close'].pct_change() < 0).sum()
            win_rate = (positive_days / (positive_days + negative_days)) * 100 if (positive_days + negative_days) > 0 else 0
            
            avg_gain = df['Close'].pct_change()[df['Close'].pct_change() > 0].mean() * 100
            avg_loss = df['Close'].pct_change()[df['Close'].pct_change() < 0].mean() * 100
            
            metrics_text = f"""
            **Trading Metrics:**
            - Positive Periods: {positive_days}
            - Negative Periods: {negative_days}
            - Win Rate: {win_rate:.1f}%
            - Average Gain: {avg_gain:.2f}%
            - Average Loss: {avg_loss:.2f}%
            - Risk/Reward Ratio: {abs(avg_gain/avg_loss):.2f} if avg_loss != 0 else 0
            """
            st.markdown(metrics_text)
        
        st.divider()
        
        st.subheader("📋 Complete Data Preview")
        
        export_df = df.copy()
        export_df['Returns_Points'] = export_df['Close'].diff()
        export_df['Returns_Pct'] = export_df['Close'].pct_change() * 100
        export_df['Volatility'] = calculate_volatility(export_df['Close'])
        export_df['RSI'] = calculate_rsi(export_df['Close'])
        
        st.dataframe(
            export_df.tail(100).style.format({
                'Open': '₹{:.2f}',
                'High': '₹{:.2f}',
                'Low': '₹{:.2f}',
                'Close': '₹{:.2f}',
                'Volume': '{:,.0f}',
                'Returns_Points': '{:+.2f}',
                'Returns_Pct': '{:+.2f}%',
                'Volatility': '{:.2f}%',
                'RSI': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        st.divider()
        
        st.subheader("💾 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV Export
            csv_data = export_df.to_csv(index=True)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"{ticker_symbol}_{interval}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel Export (timezone-naive)
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main data sheet
                export_df.to_excel(writer, sheet_name='OHLCV Data', index=True)
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Ticker', 'Interval', 'Period', 'Total Return %', 
                        'Win Rate %', 'Volatility %', 'Max Price', 'Min Price',
                        'Avg Volume', 'Start Date', 'End Date', 'Data Points'
                    ],
                    'Value': [
                        ticker_symbol,
                        interval,
                        period,
                        f"{total_return:.2f}",
                        f"{win_rate:.1f}",
                        f"{current_vol:.2f}",
                        f"{max_price:.2f}",
                        f"{min_price:.2f}",
                        f"{avg_volume:.0f}",
                        df.index[0].strftime('%Y-%m-%d %H:%M'),
                        df.index[-1].strftime('%Y-%m-%d %H:%M'),
                        str(len(df))
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Pattern matches if available
                if 'matches_df' in locals() and not matches_df.empty:
                    matches_export = matches_df[['match_date', 'match_price', 'correlation', 'volatility',
                                                 'candle_1_pct', 'candle_2_pct', 'candle_3_pct']].head(20)
                    matches_export.to_excel(writer, sheet_name='Pattern Matches', index=False)
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"{ticker_symbol}_{interval}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Final Comprehensive Recommendation
    st.divider()
    st.markdown('<p class="sub-header">🎯 FINAL TRADING RECOMMENDATION</p>', unsafe_allow_html=True)
    
    # Calculate all signals
    rsi_current = calculate_rsi(df['Close']).iloc[-1]
    price_ma20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Close'].mean()
    price_current = df['Close'].iloc[-1]
    current_volatility = calculate_volatility(df['Close']).iloc[-1]
    
    # Pattern signal
    matches_df = find_pattern_matches(df, window_size=10, top_n=20)
    if not matches_df.empty:
        pattern_signal_1 = matches_df['candle_1_pct'].mean()
        pattern_signal_3 = matches_df[['candle_1_pct', 'candle_2_pct', 'candle_3_pct']].mean().mean()
    else:
        pattern_signal_1 = 0
        pattern_signal_3 = 0
    
    # RSI signal
    if rsi_current < 30:
        rsi_signal = 2  # Strong buy
        rsi_text = "🟢 RSI OVERSOLD - Strong buy signal"
    elif rsi_current < 40:
        rsi_signal = 1  # Buy
        rsi_text = "🟢 RSI LOW - Buy signal"
    elif rsi_current > 70:
        rsi_signal = -2  # Strong sell
        rsi_text = "🔴 RSI OVERBOUGHT - Strong sell signal"
    elif rsi_current > 60:
        rsi_signal = -1  # Sell
        rsi_text = "🔴 RSI HIGH - Sell signal"
    else:
        rsi_signal = 0
        rsi_text = "🟡 RSI NEUTRAL - No clear signal"
    
    # Trend signal
    if price_current > price_ma20:
        trend_signal = 1
        trend_text = "🟢 UPTREND - Price above 20-MA"
    else:
        trend_signal = -1
        trend_text = "🔴 DOWNTREND - Price below 20-MA"
    
    # Volatility assessment
    avg_volatility = calculate_volatility(df['Close']).mean()
    if current_volatility > avg_volatility * 1.5:
        vol_text = "⚠️ HIGH VOLATILITY - Reduce position size"
        vol_warning = True
    elif current_volatility < avg_volatility * 0.5:
        vol_text = "✅ LOW VOLATILITY - Safe for larger positions"
        vol_warning = False
    else:
        vol_text = "➡️ NORMAL VOLATILITY"
        vol_warning = False
    
    # Combined scoring
    pattern_score = 2 if pattern_signal_3 > 1.5 else (1 if pattern_signal_3 > 0.5 else (-1 if pattern_signal_3 < -0.5 else 0))
    combined_score = pattern_score + rsi_signal + trend_signal
    
    # Display signals
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📊 Signal Analysis:")
        st.markdown(f"**Pattern Signal:** {'🟢 Bullish' if pattern_signal_3 > 0.5 else ('🔴 Bearish' if pattern_signal_3 < -0.5 else '🟡 Neutral')} (Expected 3-candle move: {pattern_signal_3:+.2f}%)")
        st.markdown(f"**{rsi_text}** (Current: {rsi_current:.1f})")
        st.markdown(f"**{trend_text}**")
        st.markdown(f"**{vol_text}** (Current: {current_volatility:.2f}% vs Avg: {avg_volatility:.2f}%)")
        
        st.divider()
        
        st.markdown("### 💡 Reasoning:")
        
        reasoning = []
        if pattern_score > 0:
            reasoning.append(f"✅ Historical patterns suggest upward movement of ~{pattern_signal_3:.2f}% in next 3 periods")
        elif pattern_score < 0:
            reasoning.append(f"⚠️ Historical patterns suggest downward movement of ~{pattern_signal_3:.2f}% in next 3 periods")
        
        if rsi_signal > 0:
            reasoning.append(f"✅ RSI at {rsi_current:.1f} indicates oversold conditions, favoring reversal")
        elif rsi_signal < 0:
            reasoning.append(f"⚠️ RSI at {rsi_current:.1f} indicates overbought conditions, correction likely")
        
        if trend_signal > 0:
            reasoning.append(f"✅ Price is in uptrend (above 20-MA at ₹{price_ma20:.2f})")
        else:
            reasoning.append(f"⚠️ Price is in downtrend (below 20-MA at ₹{price_ma20:.2f})")
        
        if vol_warning:
            reasoning.append(f"⚠️ Current high volatility suggests increased risk - use tighter stops")
        
        for reason in reasoning:
            st.markdown(f"- {reason}")
    
    with col2:
        if combined_score >= 3:
            st.success("### 🟢 STRONG BUY")
            action = "Enter LONG position with conviction. Multiple confirming signals."
            confidence = "High"
        elif combined_score >= 1:
            st.success("### 🟢 BUY")
            action = "Favorable for LONG position. Use proper risk management."
            confidence = "Medium-High"
        elif combined_score <= -3:
            st.error("### 🔴 STRONG SELL")
            action = "Exit LONG or enter SHORT. Multiple bearish signals."
            confidence = "High"
        elif combined_score <= -1:
            st.error("### 🔴 SELL")
            action = "Reduce exposure or exit positions. Bearish signals dominate."
            confidence = "Medium-High"
        else:
            st.warning("### 🟡 HOLD")
            action = "Wait for clearer signals. No strong directional bias."
            confidence = "Low"
        
        st.markdown(f"**Action:** {action}")
        st.markdown(f"**Confidence:** {confidence}")
        
        if not matches_df.empty:
            bullish_prob = (matches_df['candle_1_pct'] > 0).sum() / len(matches_df) * 100
            st.metric("Bullish Probability", f"{bullish_prob:.1f}%")
    
    st.info("⚠️ **Risk Disclaimer:** This is an algorithmic analysis tool for educational purposes only. Always conduct your own research, use proper risk management, and never invest more than you can afford to lose. Past performance does not guarantee future results.")

else:
    # Welcome screen
    st.info("👆 Configure analysis in sidebar and click 'Fetch Data & Analyze'")
    
    st.markdown("""
    ## 🚀 Enhanced Features
    
    ### 📊 Advanced Pattern Matching
    - **Detailed 6-Candle Forecast:** See exact point and percentage predictions for next 6 periods
    - **Similarity Matching:** Finds historical patterns with correlation and volatility matching
    - **Visual Pattern Markers:** Chart annotations showing similar patterns and their outcomes
    - **Confidence Scores:** Bullish probability based on historical accuracy
    
    ### 🔥 Volatility Analysis
    - **Volatility Bins:** Performance analysis across different volatility ranges
    - **Price-Volatility Correlation:** See which price levels correspond to volatility ranges
    - **Current Zone Analysis:** Know what to expect based on current volatility
    - **Risk Assessment:** Understand when market is most/least risky
    
    ### 📈 Comprehensive Data Table
    - **Yesterday Comparison:** Points and percentage change from previous period
    - **Volatility Tracking:** Real-time volatility for each data point
    - **Color-Coded Returns:** Easy visual identification of gains/losses
    - **Complete OHLCV Data:** All standard market data plus calculated indicators
    
    ### 📅 Returns & Volatility Heatmaps
    - **Day vs Month Analysis:** Best/worst days and months for trading
    - **Month vs Year Trends:** Seasonal patterns across years
    - **Volatility Patterns:** When market is most/least volatile
    - **Human-Readable Insights:** Plain English explanation of patterns
    
    ### 🔄 Enhanced Ratio Analysis
    - **Detailed Ratio Bins:** Performance at different ratio levels
    - **Volatility Ranges:** Know the volatility at each ratio level
    - **Price Correlation:** See ticker1 prices within each ratio bin
    - **4-Chart Layout:** Complete visual analysis with dedicated volatility chart
    
    ### 💾 Fixed Excel Export
    - **Timezone Issue Resolved:** Proper IST conversion without Excel errors
    - **Multiple Sheets:** Data, Summary, and Pattern Matches
    - **Complete Statistics:** All metrics included in export
    - **CSV Option:** Alternative format for easy data manipulation
    
    ### 📖 Chart Explanations
    - **Detailed Interpretations:** Human-readable explanation of every chart
    - **Pattern Explanation:** What each pattern marker means
    - **Volume Analysis:** Understanding volume spikes and drops
    - **RSI Interpretation:** Clear buy/sell zone explanations
    
    ## 🎯 How Predictions Work
    
    1. **Pattern Search:** Algorithm scans entire history for similar 10-candle patterns
    2. **Similarity Score:** Uses Euclidean distance + correlation + volatility matching
    3. **Future Outcome:** Records what happened 1-6 candles after each match
    4. **Aggregation:** Averages all matches to predict current situation
    5. **Confidence:** Shows bullish probability and historical accuracy
    
    ## ⚡ Pro Tips
    
    - Use **1d timeframe with 1y period** for swing trading analysis
    - Use **15m/1h with 1mo period** for intraday patterns
    - Check **volatility bins** before position sizing
    - Use **heatmaps** to identify seasonal patterns
    - Enable **ratio analysis** to find relative value opportunities
    - Export data to Excel for custom backtesting
    
    **Ready to analyze?** Configure settings in sidebar and click Fetch! 🚀
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🤖 Advanced Trading Pattern Analyzer - Enhanced Edition</p>
    <p>📊 Pattern Matching • 🔥 Volatility Analysis • 📅 Seasonal Patterns • 🎯 Precise Forecasting</p>
</div>
""", unsafe_allow_html=True)
