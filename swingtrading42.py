import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import pytz
from scipy import stats
import io

# Page configuration
st.set_page_config(page_title="Professional Algo Trading Dashboard", layout="wide")

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .gain {
        color: #00ff00;
        font-weight: bold;
    }
    .loss {
        color: #ff0000;
        font-weight: bold;
    }
    .neutral {
        color: #ffaa00;
        font-weight: bold;
    }
    .dataframe {
        font-size: 0.9rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1rem;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None

# Ticker mappings
TICKER_MAP = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'Custom': 'Custom'
}

# Helper Functions
def calculate_rsi(data, period=14):
    """Calculate RSI manually"""
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(data), index=data.index)

def calculate_ema(data, period):
    """Calculate EMA manually"""
    try:
        return data.ewm(span=period, adjust=False).mean()
    except:
        return data

def calculate_sma(data, period):
    """Calculate SMA manually"""
    try:
        return data.rolling(window=period).mean()
    except:
        return data

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    return {
        '0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100%': low
    }

def calculate_support_resistance(df, num_levels=3):
    """Calculate support and resistance levels"""
    try:
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find pivot points
        pivots_high = []
        pivots_low = []
        
        for i in range(5, len(closes) - 5):
            if highs[i] == max(highs[i-5:i+5]):
                pivots_high.append(highs[i])
            if lows[i] == min(lows[i-5:i+5]):
                pivots_low.append(lows[i])
        
        # Get unique levels
        resistance = sorted(list(set(pivots_high)), reverse=True)[:num_levels]
        support = sorted(list(set(pivots_low)))[:num_levels]
        
        return support, resistance
    except:
        return [], []

def calculate_volatility(df):
    """Calculate volatility as standard deviation of returns"""
    try:
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        return round(volatility, 2)
    except:
        return 0.0

def fetch_data_with_delay(ticker, interval, period, delay=2):
    """Fetch data with delay to respect API limits"""
    try:
        time.sleep(delay)
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        # Handle multi-index dataframe
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Reset index and handle datetime
        data = data.reset_index()
        
        # Rename datetime column
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'Datetime'}, inplace=True)
        elif 'Datetime' not in data.columns and len(data.columns) > 0:
            data.rename(columns={data.columns[0]: 'Datetime'}, inplace=True)
        
        # Keep only required columns
        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in required_cols if col in data.columns]]
        
        # Convert to IST
        if 'Datetime' in data.columns:
            if data['Datetime'].dt.tz is None:
                data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize('UTC')
            data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')
            data['Datetime'] = data['Datetime'].dt.tz_localize(None)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def detect_divergence(price, indicator, window=5):
    """Detect divergence between price and indicator"""
    try:
        if len(price) < window * 2:
            return False
        
        price_trend = price.iloc[-window:].iloc[-1] > price.iloc[-window:].iloc[0]
        indicator_trend = indicator.iloc[-window:].iloc[-1] > indicator.iloc[-window:].iloc[0]
        
        return price_trend != indicator_trend
    except:
        return False

def create_ratio_binning_analysis(df1, df2, df_ratio):
    """Create ratio binning analysis"""
    try:
        st.subheader("📊 Ratio Binning Analysis")
        
        # Ensure all dataframes have same length
        min_len = min(len(df1), len(df2), len(df_ratio))
        df1 = df1.iloc[:min_len].reset_index(drop=True)
        df2 = df2.iloc[:min_len].reset_index(drop=True)
        df_ratio = df_ratio.iloc[:min_len].reset_index(drop=True)
        
        # Create bins
        ratio_values = df_ratio['Ratio'].dropna()
        
        if len(ratio_values) < 5:
            st.warning("Not enough data points for ratio binning analysis")
            return
        
        # Create bins with error handling
        try:
            df_ratio['Ratio_Bin'] = pd.qcut(df_ratio['Ratio'], q=5, duplicates='drop', labels=False)
        except:
            # If qcut fails, use cut instead
            df_ratio['Ratio_Bin'] = pd.cut(df_ratio['Ratio'], bins=5, labels=False)
        
        # Calculate returns - ensure same length
        df1_close = pd.Series(df1['Close'].values[:min_len])
        df2_close = pd.Series(df2['Close'].values[:min_len])
        
        # Merge with price data
        analysis_df = df_ratio.copy()
        analysis_df['Ticker1_Return'] = df1_close.pct_change()
        analysis_df['Ticker2_Return'] = df2_close.pct_change()
        
        # Calculate statistics per bin
        bin_stats = []
        for bin_num in sorted(analysis_df['Ratio_Bin'].dropna().unique()):
            bin_data = analysis_df[analysis_df['Ratio_Bin'] == bin_num]
            
            if len(bin_data) == 0:
                continue
                
            bin_range = f"{bin_data['Ratio'].min():.4f} - {bin_data['Ratio'].max():.4f}"
            
            bin_stats.append({
                'Bin': f"Bin {int(bin_num)+1} ({bin_range})",
                'Ticker1_Avg_Return_%': bin_data['Ticker1_Return'].mean() * 100,
                'Ticker2_Avg_Return_%': bin_data['Ticker2_Return'].mean() * 100,
                'Count': len(bin_data)
            })
        
        if not bin_stats:
            st.warning("Could not generate bin statistics")
            return
            
        bin_df = pd.DataFrame(bin_stats)
        
        # Style the dataframe
        def color_returns(val):
            if isinstance(val, (int, float)):
                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                return f'color: {color}; font-weight: bold'
            return ''
        
        styled_df = bin_df.style.applymap(color_returns, subset=['Ticker1_Avg_Return_%', 'Ticker2_Avg_Return_%'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Current bin analysis
        current_ratio = df_ratio['Ratio'].iloc[-1]
        current_bin_val = analysis_df[analysis_df['Ratio'].notna()].iloc[-1]['Ratio_Bin']
        
        if pd.notna(current_bin_val):
            current_bin = int(current_bin_val)
            
            st.markdown(f"""
            **Current Ratio: {current_ratio:.4f}**
            
            **Current Bin: Bin {current_bin+1}**
            
            **Insight:** Based on historical data, when ratio is in this range, 
            Ticker 1 typically shows {bin_df.iloc[current_bin]['Ticker1_Avg_Return_%']:.2f}% return 
            and Ticker 2 shows {bin_df.iloc[current_bin]['Ticker2_Avg_Return_%']:.2f}% return.
            """)
        
    except Exception as e:
        st.error(f"Error in ratio binning: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def create_multi_timeframe_analysis(ticker, ticker_name):
    """Create comprehensive multi-timeframe analysis"""
    st.subheader(f"🔍 Multi-Timeframe Analysis - {ticker_name}")
    
    timeframes = [
        ('1m', '1d'), ('5m', '5d'), ('15m', '5d'), ('30m', '1mo'),
        ('1h', '1mo'), ('2h', '3mo'), ('4h', '6mo'), ('1d', '1y'),
        ('1wk', '5y'), ('1mo', '10y')
    ]
    
    analysis_data = []
    progress_bar = st.progress(0)
    
    for idx, (interval, period) in enumerate(timeframes):
        try:
            df = fetch_data_with_delay(ticker, interval, period, delay=1.5)
            
            if df.empty or len(df) < 50:
                continue
            
            current_price = df['Close'].iloc[-1]
            max_price = df['High'].max()
            min_price = df['Low'].min()
            
            # Calculate indicators
            rsi = calculate_rsi(df['Close'])
            ema9 = calculate_ema(df['Close'], 9)
            ema20 = calculate_ema(df['Close'], 20)
            ema21 = calculate_ema(df['Close'], 21)
            ema33 = calculate_ema(df['Close'], 33)
            ema50 = calculate_ema(df['Close'], 50)
            ema100 = calculate_ema(df['Close'], 100)
            ema150 = calculate_ema(df['Close'], 150)
            ema200 = calculate_ema(df['Close'], 200)
            
            sma20 = calculate_sma(df['Close'], 20)
            sma50 = calculate_sma(df['Close'], 50)
            sma100 = calculate_sma(df['Close'], 100)
            sma150 = calculate_sma(df['Close'], 150)
            sma200 = calculate_sma(df['Close'], 200)
            
            # Trend determination
            trend = "Up" if df['Close'].iloc[-1] > df['Close'].iloc[0] else "Down"
            
            # Fibonacci levels
            fib = calculate_fibonacci_levels(max_price, min_price)
            
            # Support and Resistance
            support, resistance = calculate_support_resistance(df)
            
            # Volatility
            volatility = calculate_volatility(df)
            
            # Price changes
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
            price_change_pct = (price_change / df['Close'].iloc[0]) * 100
            
            # RSI status
            current_rsi = rsi.iloc[-1]
            rsi_status = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
            
            analysis_data.append({
                'Timeframe': f"{interval}/{period}",
                'Trend': trend,
                'Max_Close': f"{max_price:.2f}",
                'Min_Close': f"{min_price:.2f}",
                'Fib_23.6%': f"{fib['23.6%']:.2f}",
                'Fib_38.2%': f"{fib['38.2%']:.2f}",
                'Fib_50%': f"{fib['50%']:.2f}",
                'Fib_61.8%': f"{fib['61.8%']:.2f}",
                'Volatility': f"{volatility}%",
                'Change_%': f"{price_change_pct:.2f}%",
                'Points_Changed': f"{price_change:.2f}",
                'Support_Levels': ', '.join([f"{s:.2f}" for s in support[:3]]),
                'Resistance_Levels': ', '.join([f"{r:.2f}" for r in resistance[:3]]),
                'RSI': f"{current_rsi:.2f}",
                'RSI_Status': rsi_status,
                'EMA9': f"{ema9.iloc[-1]:.2f}",
                'EMA20': f"{ema20.iloc[-1]:.2f}",
                'EMA21': f"{ema21.iloc[-1]:.2f}",
                'EMA33': f"{ema33.iloc[-1]:.2f}",
                'EMA50': f"{ema50.iloc[-1]:.2f}",
                'EMA100': f"{ema100.iloc[-1]:.2f}",
                'EMA150': f"{ema150.iloc[-1]:.2f}",
                'EMA200': f"{ema200.iloc[-1]:.2f}",
                'vs_EMA20': 'Above' if current_price > ema20.iloc[-1] else 'Below',
                'vs_EMA50': 'Above' if current_price > ema50.iloc[-1] else 'Below',
                'vs_EMA200': 'Above' if current_price > ema200.iloc[-1] else 'Below',
                'SMA20': f"{sma20.iloc[-1]:.2f}",
                'SMA50': f"{sma50.iloc[-1]:.2f}",
                'SMA100': f"{sma100.iloc[-1]:.2f}",
                'SMA150': f"{sma150.iloc[-1]:.2f}",
                'SMA200': f"{sma200.iloc[-1]:.2f}",
                'vs_SMA20': 'Above' if current_price > sma20.iloc[-1] else 'Below',
                'vs_SMA50': 'Above' if current_price > sma50.iloc[-1] else 'Below',
                'vs_SMA200': 'Above' if current_price > sma200.iloc[-1] else 'Below',
            })
            
        except Exception as e:
            st.warning(f"Could not fetch {interval}/{period}: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(timeframes))
    
    if analysis_data:
        mtf_df = pd.DataFrame(analysis_data)
        
        # Style the dataframe
        def style_mtf(row):
            styles = [''] * len(row)
            if 'Trend' in row.index:
                idx = row.index.get_loc('Trend')
                styles[idx] = 'color: green; font-weight: bold' if row['Trend'] == 'Up' else 'color: red; font-weight: bold'
            if 'RSI_Status' in row.index:
                idx = row.index.get_loc('RSI_Status')
                if row['RSI_Status'] == 'Oversold':
                    styles[idx] = 'color: green; font-weight: bold'
                elif row['RSI_Status'] == 'Overbought':
                    styles[idx] = 'color: red; font-weight: bold'
            return styles
        
        st.dataframe(mtf_df.style.apply(style_mtf, axis=1), use_container_width=True)
        
        # Summary
        up_trends = sum([1 for d in analysis_data if d['Trend'] == 'Up'])
        total = len(analysis_data)
        
        st.markdown(f"""
        ### 📈 Multi-Timeframe Summary for {ticker_name}
        
        - **Uptrends:** {up_trends}/{total} timeframes
        - **Downtrends:** {total - up_trends}/{total} timeframes
        - **Overall Bias:** {'BULLISH' if up_trends > total/2 else 'BEARISH'}
        - **Current Price:** {current_price:.2f}
        - **Volatility Range:** {min([float(d['Volatility'].rstrip('%')) for d in analysis_data]):.1f}% - {max([float(d['Volatility'].rstrip('%')) for d in analysis_data]):.1f}%
        
        **Recommendation:** Based on multi-timeframe analysis, the {'upward' if up_trends > total/2 else 'downward'} trend 
        is dominant across timeframes. Consider this bias for trading decisions.
        """)

def create_volatility_bins_analysis(df, ticker_name):
    """Create volatility binning analysis"""
    st.subheader(f"📊 Volatility Bins Analysis - {ticker_name}")
    
    try:
        # Calculate volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # Create bins
        df['Vol_Bin'] = pd.qcut(df['Volatility'].dropna(), q=5, duplicates='drop', labels=False)
        
        # Analysis table
        vol_analysis = []
        for idx, row in df.dropna(subset=['Vol_Bin']).iterrows():
            vol_analysis.append({
                'Datetime': row['Datetime'],
                'Volatility_Bin': f"Bin {int(row['Vol_Bin'])+1}",
                'Volatility_%': f"{row['Volatility']:.2f}%",
                'Price': f"{row['Close']:.2f}",
                'Returns_Points': f"{row['Returns'] * row['Close']:.2f}",
                'Returns_%': f"{row['Returns'] * 100:.2f}%"
            })
        
        vol_df = pd.DataFrame(vol_analysis[-100:])  # Last 100 rows
        st.dataframe(vol_df, use_container_width=True)
        
        # Statistics
        st.markdown(f"""
        ### Volatility Statistics
        
        - **Highest Volatility:** {df['Volatility'].max():.2f}%
        - **Lowest Volatility:** {df['Volatility'].min():.2f}%
        - **Mean Volatility:** {df['Volatility'].mean():.2f}%
        - **Current Volatility:** {df['Volatility'].iloc[-1]:.2f}%
        - **Current Bin:** Bin {int(df['Vol_Bin'].iloc[-1])+1}
        
        **Forecast:** Based on current volatility levels, expect {'higher' if df['Volatility'].iloc[-1] > df['Volatility'].mean() else 'lower'} 
        than average price movements.
        """)
        
    except Exception as e:
        st.error(f"Error in volatility analysis: {str(e)}")

def detect_patterns(df, threshold=30):
    """Detect patterns before significant moves"""
    st.subheader("🔎 Advanced Pattern Recognition")
    
    try:
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['EMA20'] = calculate_ema(df['Close'], 20)
        df['EMA50'] = calculate_ema(df['Close'], 50)
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(10).std() * 100
        
        # Detect significant moves
        df['Price_Move'] = df['Close'].diff()
        significant_moves = df[abs(df['Price_Move']) > threshold].copy()
        
        patterns = []
        for idx in significant_moves.index:
            pos = df.index.get_loc(idx)
            if pos < 10:
                continue
            
            lookback = df.iloc[pos-10:pos]
            current = df.loc[idx]
            
            # Pattern detection
            vol_burst = lookback['Volatility'].iloc[-1] > lookback['Volatility'].mean() * 1.5
            rsi_divergence = detect_divergence(lookback['Close'], lookback['RSI'])
            ema_cross = (lookback['EMA20'].iloc[-2] < lookback['EMA50'].iloc[-2] and 
                        lookback['EMA20'].iloc[-1] > lookback['EMA50'].iloc[-1])
            
            # Large body candles
            body_size = abs(df['Close'] - df['Open']).iloc[pos-10:pos]
            large_body = (body_size.iloc[-1] > body_size.mean() * 1.5)
            
            # Consecutive moves
            consecutive_up = all(df['Returns'].iloc[pos-3:pos] > 0)
            consecutive_down = all(df['Returns'].iloc[pos-3:pos] < 0)
            
            patterns.append({
                'Datetime': current['Datetime'],
                'Move_Points': f"{current['Price_Move']:.2f}",
                'Move_%': f"{(current['Price_Move']/df['Close'].iloc[pos-1])*100:.2f}%",
                'Direction': 'Up' if current['Price_Move'] > 0 else 'Down',
                'Volatility_Burst': 'Yes' if vol_burst else 'No',
                'RSI_Divergence': 'Yes' if rsi_divergence else 'No',
                'EMA_Crossover': 'Yes' if ema_cross else 'No',
                'Large_Body': 'Yes' if large_body else 'No',
                'Consecutive_Moves': 'Yes' if (consecutive_up or consecutive_down) else 'No',
                'RSI_Before': f"{lookback['RSI'].iloc[0]:.2f}",
                'RSI_At_Move': f"{current['RSI']:.2f}"
            })
        
        if patterns:
            pattern_df = pd.DataFrame(patterns)
            st.dataframe(pattern_df, use_container_width=True)
            
            # Summary
            st.markdown(f"""
            ### Pattern Analysis Summary
            
            - **Total Significant Moves:** {len(patterns)}
            - **Volatility Bursts Detected:** {sum([1 for p in patterns if p['Volatility_Burst'] == 'Yes'])}
            - **RSI Divergences:** {sum([1 for p in patterns if p['RSI_Divergence'] == 'Yes'])}
            - **EMA Crossovers:** {sum([1 for p in patterns if p['EMA_Crossover'] == 'Yes'])}
            
            **Current Market Check:**
            - Current RSI: {df['RSI'].iloc[-1]:.2f}
            - Current Volatility: {df['Volatility'].iloc[-1]:.2f}%
            - RSI Divergence Present: {'Yes' if detect_divergence(df['Close'].tail(20), df['RSI'].tail(20)) else 'No'}
            
            ⚠️ **Warning:** {'Pattern similarity detected! Exercise caution.' if len(patterns) > 5 else 'Normal market conditions.'}
            """)
        else:
            st.info("No significant patterns detected with current threshold.")
            
    except Exception as e:
        st.error(f"Error in pattern detection: {str(e)}")

def create_charts(df1, df2, ratio_df, ticker1_name, ticker2_name, enable_ratio):
    """Create interactive charts"""
    st.subheader("📈 Interactive Charts")
    
    try:
        if enable_ratio and df2 is not None:
            # Ticker 1 Charts
            st.markdown(f"### {ticker1_name} Analysis")
            fig1 = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                                shared_xaxes=True, vertical_spacing=0.05)
            
            # Candlestick
            fig1.add_trace(go.Candlestick(x=df1['Datetime'], open=df1['Open'],
                                         high=df1['High'], low=df1['Low'],
                                         close=df1['Close'], name='Price'), row=1, col=1)
            
            # EMAs
            ema20 = calculate_ema(df1['Close'], 20)
            ema50 = calculate_ema(df1['Close'], 50)
            ema200 = calculate_ema(df1['Close'], 200)
            
            fig1.add_trace(go.Scatter(x=df1['Datetime'], y=ema20, name='EMA20',
                                     line=dict(color='orange', width=1)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df1['Datetime'], y=ema50, name='EMA50',
                                     line=dict(color='blue', width=1)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df1['Datetime'], y=ema200, name='EMA200',
                                     line=dict(color='purple', width=1)), row=1, col=1)
            
            # RSI
            rsi1 = calculate_rsi(df1['Close'])
            fig1.add_trace(go.Scatter(x=df1['Datetime'], y=rsi1, name='RSI',
                                     line=dict(color='green')), row=2, col=1)
            fig1.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig1.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig1.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Ticker 2 Charts
            st.markdown(f"### {ticker2_name} Analysis")
            fig2 = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                                shared_xaxes=True, vertical_spacing=0.05)
            
            fig2.add_trace(go.Candlestick(x=df2['Datetime'], open=df2['Open'],
                                         high=df2['High'], low=df2['Low'],
                                         close=df2['Close'], name='Price'), row=1, col=1)
            
            ema20_2 = calculate_ema(df2['Close'], 20)
            ema50_2 = calculate_ema(df2['Close'], 50)
            ema200_2 = calculate_ema(df2['Close'], 200)
            
            fig2.add_trace(go.Scatter(x=df2['Datetime'], y=ema20_2, name='EMA20',
                                     line=dict(color='orange', width=1)), row=1, col=1)
            fig2.add_trace(go.Scatter(x=df2['Datetime'], y=ema50_2, name='EMA50',
                                     line=dict(color='blue', width=1)), row=1, col=1)
            fig2.add_trace(go.Scatter(x=df2['Datetime'], y=ema200_2, name='EMA200',
                                     line=dict(color='purple', width=1)), row=1, col=1)
            
            rsi2 = calculate_rsi(df2['Close'])
            fig2.add_trace(go.Scatter(x=df2['Datetime'], y=rsi2, name='RSI',
                                     line=dict(color='green')), row=2, col=1)
            fig2.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig2.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig2.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Ratio Chart
            st.markdown("### Ratio Analysis")
            fig3 = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                                shared_xaxes=True, vertical_spacing=0.05)
            
            fig3.add_trace(go.Scatter(x=ratio_df['Datetime'], y=ratio_df['Ratio'],
                                     name='Ratio', line=dict(color='purple')), row=1, col=1)
            
            rsi_ratio = calculate_rsi(ratio_df['Ratio'])
            fig3.add_trace(go.Scatter(x=ratio_df['Datetime'], y=rsi_ratio, name='RSI',
                                     line=dict(color='orange')), row=2, col=1)
            fig3.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig3.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig3.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig3, use_container_width=True)
            
        else:
            # Single ticker charts
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                               shared_xaxes=True, vertical_spacing=0.05)
            
            fig.add_trace(go.Candlestick(x=df1['Datetime'], open=df1['Open'],
                                        high=df1['High'], low=df1['Low'],
                                        close=df1['Close'], name='Price'), row=1, col=1)
            
            ema20 = calculate_ema(df1['Close'], 20)
            ema50 = calculate_ema(df1['Close'], 50)
            ema200 = calculate_ema(df1['Close'], 200)
            
            fig.add_trace(go.Scatter(x=df1['Datetime'], y=ema20, name='EMA20',
                                    line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df1['Datetime'], y=ema50, name='EMA50',
                                    line=dict(color='blue', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df1['Datetime'], y=ema200, name='EMA200',
                                    line=dict(color='purple', width=1)), row=1, col=1)
            
            rsi = calculate_rsi(df1['Close'])
            fig.add_trace(go.Scatter(x=df1['Datetime'], y=rsi, name='RSI',
                                    line=dict(color='green')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")

def create_statistical_distribution(df, ticker_name):
    """Create statistical distribution analysis"""
    st.subheader(f"📊 Statistical Distribution Analysis - {ticker_name}")
    
    try:
        df['Returns'] = df['Close'].pct_change() * 100
        returns = df['Returns'].dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of returns
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns',
                                       marker_color='lightblue'))
            fig1.update_layout(title='Returns Distribution', xaxis_title='Returns (%)',
                             yaxis_title='Frequency', height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Histogram with normal curve
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns',
                                       histnorm='probability density',
                                       marker_color='lightgreen'))
            
            # Normal curve overlay
            mu, std = returns.mean(), returns.std()
            x_range = np.linspace(returns.min(), returns.max(), 100)
            normal_dist = stats.norm.pdf(x_range, mu, std)
            fig2.add_trace(go.Scatter(x=x_range, y=normal_dist, mode='lines',
                                     name='Normal Curve', line=dict(color='red', width=2)))
            
            fig2.update_layout(title='Returns with Normal Curve', xaxis_title='Returns (%)',
                             yaxis_title='Density', height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Bell Curve Visualization
        st.markdown("### 🔔 Bell Curve Analysis")
        
        fig3 = go.Figure()
        
        # Add shaded regions
        x_vals = np.linspace(mu - 4*std, mu + 4*std, 1000)
        y_vals = stats.norm.pdf(x_vals, mu, std)
        
        # Green zone (±1 std)
        mask1 = (x_vals >= mu - std) & (x_vals <= mu + std)
        fig3.add_trace(go.Scatter(x=x_vals[mask1], y=y_vals[mask1],
                                 fill='tozeroy', fillcolor='rgba(0,255,0,0.3)',
                                 line=dict(width=0), showlegend=True,
                                 name='±1σ (68%)'))
        
        # Yellow zone (±2 std)
        mask2 = ((x_vals >= mu - 2*std) & (x_vals < mu - std)) | ((x_vals > mu + std) & (x_vals <= mu + 2*std))
        fig3.add_trace(go.Scatter(x=x_vals[mask2], y=y_vals[mask2],
                                 fill='tozeroy', fillcolor='rgba(255,255,0,0.3)',
                                 line=dict(width=0), showlegend=True,
                                 name='±2σ (95%)'))
        
        # Red zone (beyond ±2 std)
        mask3 = (x_vals < mu - 2*std) | (x_vals > mu + 2*std)
        fig3.add_trace(go.Scatter(x=x_vals[mask3], y=y_vals[mask3],
                                 fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
                                 line=dict(width=0), showlegend=True,
                                 name='Beyond ±2σ'))
        
        # Current position
        current_return = returns.iloc[-1]
        current_y = stats.norm.pdf(current_return, mu, std)
        fig3.add_trace(go.Scatter(x=[current_return], y=[current_y],
                                 mode='markers', marker=dict(size=15, color='black'),
                                 name='Current Position'))
        
        fig3.update_layout(title='Bell Curve with Current Position',
                          xaxis_title='Returns (%)', yaxis_title='Probability Density',
                          height=500)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Z-Score Analysis
        st.markdown("### 📈 Z-Score Analysis")
        
        df['Z_Score'] = (df['Returns'] - mu) / std
        
        zscore_data = []
        for idx, row in df.dropna(subset=['Z_Score']).tail(50).iterrows():
            zscore_data.append({
                'Datetime': row['Datetime'],
                'Returns_%': f"{row['Returns']:.2f}%",
                'Returns_Points': f"{(row['Returns']/100) * row['Close']:.2f}",
                'Z_Score': f"{row['Z_Score']:.2f}"
            })
        
        zscore_df = pd.DataFrame(zscore_data)
        
        def color_zscore(val):
            if 'Z_Score' in str(val):
                try:
                    z = float(val.replace('Z_Score', '').strip())
                    if abs(z) > 2:
                        return 'color: red; font-weight: bold'
                    elif abs(z) > 1:
                        return 'color: orange; font-weight: bold'
                except:
                    pass
            return ''
        
        st.dataframe(zscore_df.style.applymap(color_zscore), use_container_width=True)
        
        # Statistical Summary
        current_z = df['Z_Score'].iloc[-1]
        percentile = stats.norm.cdf(current_z) * 100
        
        st.markdown(f"""
        ### 📊 Statistical Summary
        
        **Distribution Parameters:**
        - Mean Return: {mu:.4f}%
        - Standard Deviation: {std:.4f}%
        - Skewness: {stats.skew(returns):.4f}
        - Kurtosis: {stats.kurtosis(returns):.4f}
        
        **Current Position:**
        - Current Return: {current_return:.2f}%
        - Z-Score: {current_z:.2f}
        - Percentile: {percentile:.1f}%
        
        **Probability Ranges:**
        - Within ±1σ (68%): {mu - std:.2f}% to {mu + std:.2f}%
        - Within ±2σ (95%): {mu - 2*std:.2f}% to {mu + 2*std:.2f}%
        - Within ±3σ (99.7%): {mu - 3*std:.2f}% to {mu + 3*std:.2f}%
        
        **Interpretation:**
        {
        'Current position is EXTREME (beyond 2σ). This is a rare event occurring < 5% of the time.' 
        if abs(current_z) > 2 
        else 'Current position is UNUSUAL (beyond 1σ). This occurs ~32% of the time.' 
        if abs(current_z) > 1 
        else 'Current position is NORMAL (within 1σ). This is typical behavior.'
        }
        
        **Trading Implication:**
        {
        '⚠️ MEAN REVERSION LIKELY - Extreme moves tend to reverse. Consider counter-trend positions.' 
        if abs(current_z) > 2 
        else '⚡ MOMENTUM MAY CONTINUE - Unusual but not extreme. Monitor for continuation or reversal.' 
        if abs(current_z) > 1 
        else '✅ NORMAL MARKET CONDITIONS - No extreme statistical signal.'
        }
        """)
        
    except Exception as e:
        st.error(f"Error in statistical analysis: {str(e)}")

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    try:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else df['Close'].std()
    except:
        return df['Close'].std()

def generate_final_recommendation(df, ticker_name):
    """Generate comprehensive trading recommendation"""
    st.subheader("🎯 Final Trading Recommendation")
    
    try:
        # Calculate all required indicators
        current_price = df['Close'].iloc[-1]
        rsi = calculate_rsi(df['Close'])
        ema20 = calculate_ema(df['Close'], 20)
        ema50 = calculate_ema(df['Close'], 50)
        ema200 = calculate_ema(df['Close'], 200)
        
        # Calculate returns and z-score
        returns = df['Close'].pct_change() * 100
        mu, std = returns.mean(), returns.std()
        current_z = (returns.iloc[-1] - mu) / std if std > 0 else 0
        
        # ATR for position sizing
        atr = calculate_atr(df)
        
        # Signal Components
        signals = {}
        
        # 1. Multi-timeframe trend (30% weight)
        trend_score = 0
        if current_price > ema20.iloc[-1]:
            trend_score += 0.33
        if current_price > ema50.iloc[-1]:
            trend_score += 0.33
        if current_price > ema200.iloc[-1]:
            trend_score += 0.34
        
        if trend_score > 0.66:
            signals['trend'] = ('BULLISH', 1, 0.30)
        elif trend_score < 0.33:
            signals['trend'] = ('BEARISH', -1, 0.30)
        else:
            signals['trend'] = ('NEUTRAL', 0, 0.30)
        
        # 2. RSI (20% weight)
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            signals['rsi'] = ('OVERSOLD - BUY', 1, 0.20)
        elif current_rsi > 70:
            signals['rsi'] = ('OVERBOUGHT - SELL', -1, 0.20)
        else:
            signals['rsi'] = ('NEUTRAL', 0, 0.20)
        
        # 3. Z-Score (20% weight)
        if current_z > 2:
            signals['zscore'] = ('EXTREME HIGH - SELL', -1, 0.20)
        elif current_z < -2:
            signals['zscore'] = ('EXTREME LOW - BUY', 1, 0.20)
        elif current_z > 1:
            signals['zscore'] = ('HIGH - CAUTION', -0.5, 0.20)
        elif current_z < -1:
            signals['zscore'] = ('LOW - OPPORTUNITY', 0.5, 0.20)
        else:
            signals['zscore'] = ('NORMAL', 0, 0.20)
        
        # 4. EMA Alignment (30% weight)
        ema_alignment = 0
        if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
            ema_alignment = 1
            signals['ema'] = ('BULLISH ALIGNMENT', 1, 0.30)
        elif ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
            ema_alignment = -1
            signals['ema'] = ('BEARISH ALIGNMENT', -1, 0.30)
        else:
            signals['ema'] = ('MIXED ALIGNMENT', 0, 0.30)
        
        # Calculate combined signal
        total_score = sum([signal[1] * signal[2] for signal in signals.values()])
        
        # Determine final signal
        if total_score > 0.4:
            final_signal = 'BUY'
            signal_color = 'green'
            confidence = 'HIGH' if total_score > 0.6 else 'MODERATE'
        elif total_score < -0.4:
            final_signal = 'SELL'
            signal_color = 'red'
            confidence = 'HIGH' if total_score < -0.6 else 'MODERATE'
        else:
            final_signal = 'HOLD'
            signal_color = 'orange'
            confidence = 'LOW'
        
        # Trade Setup
        if final_signal == 'BUY':
            entry = current_price
            target = current_price + (2 * atr)
            stop_loss = current_price - atr
            direction = 1
        elif final_signal == 'SELL':
            entry = current_price
            target = current_price - (2 * atr)
            stop_loss = current_price + atr
            direction = -1
        else:
            entry = current_price
            target = current_price
            stop_loss = current_price
            direction = 0
        
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Position sizing (1-2% risk)
        account_size = 100000  # Example account size
        risk_per_trade = account_size * 0.02
        position_size = risk_per_trade / risk if risk > 0 else 0
        
        # Display recommendation
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;'>
            <h2 style='text-align: center; margin-bottom: 1rem;'>
                {final_signal} SIGNAL
            </h2>
            <h3 style='text-align: center; color: {signal_color};'>
                Confidence: {confidence}
            </h3>
            <h3 style='text-align: center;'>
                Combined Score: {total_score:.2f}
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            ### 📊 Signal Components
            
            **Trend Signal:** {signals['trend'][0]}  
            Weight: {signals['trend'][2]*100:.0f}% | Score: {signals['trend'][1]}
            
            **RSI Signal:** {signals['rsi'][0]}  
            Weight: {signals['rsi'][2]*100:.0f}% | Score: {signals['rsi'][1]}
            
            **Z-Score Signal:** {signals['zscore'][0]}  
            Weight: {signals['zscore'][2]*100:.0f}% | Score: {signals['zscore'][1]}
            
            **EMA Signal:** {signals['ema'][0]}  
            Weight: {signals['ema'][2]*100:.0f}% | Score: {signals['ema'][1]}
            """)
        
        with col2:
            st.markdown(f"""
            ### 💰 Trade Setup
            
            **Entry Price:** ₹{entry:.2f}
            
            **Target Price:** ₹{target:.2f}  
            Gain: {((target-entry)/entry)*100:.2f}%
            
            **Stop Loss:** ₹{stop_loss:.2f}  
            Loss: {((stop_loss-entry)/entry)*100:.2f}%
            
            **Risk/Reward Ratio:** {rr_ratio:.2f}
            
            **Position Size:** {position_size:.0f} units  
            (Based on 2% risk per trade)
            """)
        
        with col3:
            st.markdown(f"""
            ### 🎯 Risk Management
            
            **Risk per Trade:** 1-2% of capital
            
            **Exit Strategy:**
            - Take profit at target
            - Cut losses at stop loss
            - Trail stops in profit
            
            **Trade Management:**
            - Monitor RSI levels
            - Watch EMA support/resistance
            - Adjust stops as needed
            
            **Maximum Loss:** ₹{risk * position_size:.2f}  
            **Potential Profit:** ₹{reward * position_size:.2f}
            """)
        
        # Detailed Rationale
        st.markdown(f"""
        ### 📝 Signal Rationale
        
        **Why {final_signal}?**
        
        The {final_signal} signal is generated based on a weighted multi-factor analysis:
        
        1. **Trend Analysis (30%):** Price is currently {'above' if trend_score > 0.5 else 'below'} major EMAs, 
           indicating a {'bullish' if trend_score > 0.5 else 'bearish'} trend structure.
        
        2. **RSI Analysis (20%):** RSI at {current_rsi:.2f} suggests 
           {'oversold conditions - potential bounce' if current_rsi < 30 
            else 'overbought conditions - potential pullback' if current_rsi > 70 
            else 'neutral momentum'}.
        
        3. **Statistical Position (20%):** Z-score of {current_z:.2f} indicates 
           {'extreme deviation suggesting mean reversion' if abs(current_z) > 2
            else 'moderate deviation' if abs(current_z) > 1
            else 'normal distribution range'}.
        
        4. **EMA Alignment (30%):** EMAs show {signals['ema'][0].lower()}, 
           {'supporting trend continuation' if abs(ema_alignment) == 1 else 'suggesting consolidation'}.
        
        **Historical Context:**
        - Current Price: ₹{current_price:.2f}
        - 20-period Average: ₹{df['Close'].tail(20).mean():.2f}
        - Volatility (ATR): ₹{atr:.2f}
        - Recent Price Change: {returns.iloc[-1]:.2f}%
        
        **Expected Scenario:**
        {
        f'Bullish momentum expected to continue towards ₹{target:.2f}. Price above key EMAs suggests buyer strength.' 
        if final_signal == 'BUY'
        else f'Bearish pressure likely to push towards ₹{target:.2f}. Price below key EMAs indicates seller dominance.'
        if final_signal == 'SELL'
        else 'Mixed signals suggest consolidation. Wait for clearer directional bias before entering.'
        }
        
        **Market Structure:**
        - Support Zone: ₹{df['Low'].tail(20).min():.2f}
        - Resistance Zone: ₹{df['High'].tail(20).max():.2f}
        - Trend: {'Uptrend' if trend_score > 0.66 else 'Downtrend' if trend_score < 0.33 else 'Sideways'}
        
        ⚠️ **Disclaimer:** This is an algorithmic signal based on technical analysis. 
        Always consider fundamental factors, market conditions, and risk tolerance before trading.
        """)
        
    except Exception as e:
        st.error(f"Error generating recommendation: {str(e)}")

# Main Application
def main():
    st.markdown("<h1 class='main-header'>🚀 Professional Algo Trading Dashboard</h1>", 
                unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Ticker 1 Selection
    ticker1_preset = st.sidebar.selectbox("Select Ticker 1", list(TICKER_MAP.keys()))
    if ticker1_preset == 'Custom':
        ticker1 = st.sidebar.text_input("Enter Custom Ticker 1", "RELIANCE.NS")
    else:
        ticker1 = TICKER_MAP[ticker1_preset]
    
    # Ratio Analysis Option
    enable_ratio = st.sidebar.checkbox("Enable Ratio Analysis (Ticker 2)", value=False)
    
    ticker2 = None
    ticker2_preset = None
    if enable_ratio:
        ticker2_preset = st.sidebar.selectbox("Select Ticker 2", list(TICKER_MAP.keys()))
        if ticker2_preset == 'Custom':
            ticker2 = st.sidebar.text_input("Enter Custom Ticker 2", "TCS.NS")
        else:
            ticker2 = TICKER_MAP[ticker2_preset]
    
    # Timeframe Selection
    interval = st.sidebar.selectbox("Interval", 
                                    ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1wk', '1mo'])
    period = st.sidebar.selectbox("Period",
                                  ['1d', '5d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', 
                                   '10y', '15y', '20y', '25y', '30y'])
    
    # API Rate Limiting
    api_delay = st.sidebar.slider("API Delay (seconds)", 1.0, 5.0, 2.0, 0.5)
    
    # Pattern Detection Threshold
    pattern_threshold = st.sidebar.number_input("Pattern Detection Threshold (points)", 
                                                 min_value=1, max_value=100, value=30)
    
    # Fetch Data Button
    if st.sidebar.button("🔄 Fetch Data & Analyze"):
        with st.spinner("Fetching data... Please wait."):
            # Fetch Ticker 1
            df1 = fetch_data_with_delay(ticker1, interval, period, delay=api_delay)
            
            if df1.empty:
                st.error(f"Failed to fetch data for {ticker1}")
                return
            
            st.session_state.df1 = df1
            st.session_state.ticker1_name = ticker1_preset if ticker1_preset != 'Custom' else ticker1
            
            # Fetch Ticker 2 if enabled
            if enable_ratio and ticker2:
                df2 = fetch_data_with_delay(ticker2, interval, period, delay=api_delay)
                if df2.empty:
                    st.error(f"Failed to fetch data for {ticker2}")
                    return
                st.session_state.df2 = df2
                st.session_state.ticker2_name = ticker2_preset if ticker2_preset != 'Custom' else ticker2
            else:
                st.session_state.df2 = None
                st.session_state.ticker2_name = None
            
            st.session_state.data_fetched = True
            st.session_state.enable_ratio = enable_ratio
            st.success("✅ Data fetched successfully!")
    
            # Display Analysis if data is fetched
    if st.session_state.data_fetched and st.session_state.df1 is not None:
        df1 = st.session_state.df1.copy()
        df2 = st.session_state.df2.copy() if st.session_state.df2 is not None else None
        ticker1_name = st.session_state.ticker1_name
        ticker2_name = st.session_state.ticker2_name
        enable_ratio = st.session_state.enable_ratio
        
        # Basic Statistics
        st.header("📊 Market Overview")
        
        if enable_ratio and df2 is not None:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price1 = df1['Close'].iloc[-1]
            prev_price1 = df1['Close'].iloc[0]
            change1 = current_price1 - prev_price1
            change_pct1 = (change1 / prev_price1) * 100
            
            st.metric(
                label=f"{ticker1_name} Price",
                value=f"₹{current_price1:.2f}",
                delta=f"{change1:.2f} ({change_pct1:.2f}%)"
            )
        
        with col2:
            if enable_ratio and df2 is not None:
                current_price2 = df2['Close'].iloc[-1]
                prev_price2 = df2['Close'].iloc[0]
                change2 = current_price2 - prev_price2
                change_pct2 = (change2 / prev_price2) * 100
                
                st.metric(
                    label=f"{ticker2_name} Price",
                    value=f"₹{current_price2:.2f}",
                    delta=f"{change2:.2f} ({change_pct2:.2f}%)"
                )
            else:
                st.metric(
                    label="Period Start Price",
                    value=f"₹{prev_price1:.2f}"
                )
        
        with col3:
            if enable_ratio and df2 is not None:
                ratio = current_price1 / current_price2
                st.metric(
                    label="Current Ratio",
                    value=f"{ratio:.4f}"
                )
            else:
                high_price = df1['High'].max()
                st.metric(
                    label="Period High",
                    value=f"₹{high_price:.2f}"
                )
        
        with col4:
            if enable_ratio and df2 is not None:
                st.metric(
                    label="Data Points",
                    value=len(df1)
                )
            else:
                low_price = df1['Low'].min()
                st.metric(
                    label="Period Low",
                    value=f"₹{low_price:.2f}"
                )
        
        # Data Table
        st.subheader("📋 Price Data")
        display_df1 = df1.copy()
        display_df1['Returns_%'] = display_df1['Close'].pct_change() * 100
        st.dataframe(display_df1.tail(20), use_container_width=True)
        
        # Export Data
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='Ticker1', index=False)
            if enable_ratio and df2 is not None:
                df2.to_excel(writer, sheet_name='Ticker2', index=False)
        
        st.download_button(
            label="📥 Download Data (Excel)",
            data=buffer.getvalue(),
            file_name=f"trading_data_{ticker1_name}_{interval}_{period}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Ratio Analysis
        if enable_ratio and df2 is not None:
            st.header("📊 Ratio Analysis")
            
            # Align dataframes by finding common dates
            df1_copy = df1.copy()
            df2_copy = df2.copy()
            
            # Ensure both have Datetime column
            if 'Datetime' not in df1_copy.columns:
                df1_copy['Datetime'] = df1_copy.index
            if 'Datetime' not in df2_copy.columns:
                df2_copy['Datetime'] = df2_copy.index
            
            # Merge on datetime
            merged = pd.merge(df1_copy[['Datetime', 'Close']], 
                            df2_copy[['Datetime', 'Close']], 
                            on='Datetime', 
                            suffixes=('_T1', '_T2'))
            
            # Create ratio dataframe
            ratio_df = pd.DataFrame()
            ratio_df['Datetime'] = merged['Datetime']
            ratio_df['Ticker1_Price'] = merged['Close_T1']
            ratio_df['Ticker2_Price'] = merged['Close_T2']
            ratio_df['Ratio'] = ratio_df['Ticker1_Price'] / ratio_df['Ticker2_Price']
            
            # Calculate RSI for aligned data
            ratio_df['RSI_Ticker1'] = calculate_rsi(pd.Series(ratio_df['Ticker1_Price'].values))
            ratio_df['RSI_Ticker2'] = calculate_rsi(pd.Series(ratio_df['Ticker2_Price'].values))
            ratio_df['RSI_Ratio'] = calculate_rsi(pd.Series(ratio_df['Ratio'].values))
            
            st.dataframe(ratio_df.tail(20), use_container_width=True)
            
            # Ratio Binning Analysis - pass aligned dataframes
            df1_aligned = df1_copy[df1_copy['Datetime'].isin(merged['Datetime'])].reset_index(drop=True)
            df2_aligned = df2_copy[df2_copy['Datetime'].isin(merged['Datetime'])].reset_index(drop=True)
            create_ratio_binning_analysis(df1_aligned, df2_aligned, ratio_df)
        
        # Multi-Timeframe Analysis
        st.header("🔍 Multi-Timeframe Analysis")
        create_multi_timeframe_analysis(ticker1, ticker1_name)
        
        if enable_ratio and df2 is not None:
            create_multi_timeframe_analysis(ticker2, ticker2_name)
        
        # Volatility Bins Analysis
        st.header("📊 Volatility Analysis")
        create_volatility_bins_analysis(df1, ticker1_name)
        
        if enable_ratio and df2 is not None:
            create_volatility_bins_analysis(df2, ticker2_name)
        
        # Pattern Recognition
        st.header("🔎 Pattern Recognition")
        detect_patterns(df1, threshold=pattern_threshold)
        
        if enable_ratio and df2 is not None:
            detect_patterns(df2, threshold=pattern_threshold)
        
        # Interactive Charts
        st.header("📈 Technical Charts")
        if enable_ratio and df2 is not None:
            create_charts(df1, df2, ratio_df, ticker1_name, ticker2_name, True)
        else:
            create_charts(df1, None, None, ticker1_name, None, False)
        
        # Statistical Distribution Analysis
        st.header("📊 Statistical Distribution & Z-Score Analysis")
        create_statistical_distribution(df1, ticker1_name)
        
        if enable_ratio and df2 is not None:
            create_statistical_distribution(df2, ticker2_name)
        
        # Trading Signals with proper column handling
        st.header("🎯 Trading Signals")
        
        # Generate recommendations for both tickers
        if enable_ratio and df2 is not None:
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown(f"### {ticker1_name} Recommendation")
                generate_final_recommendation(df1, ticker1_name)
            
            with col_right:
                st.markdown(f"### {ticker2_name} Recommendation")
                generate_final_recommendation(df2, ticker2_name)
        else:
            st.markdown(f"### {ticker1_name} Recommendation")
            generate_final_recommendation(df1, ticker1_name)
        
        # Unified Recommendation Section
        st.markdown("---")
        st.markdown("## 🎯 Unified Trading Strategy")
        
        # Calculate signals for both tickers if ratio analysis is enabled
        if enable_ratio and df2 is not None:
            # Get signals from both tickers
            df1_signal = calculate_unified_signal(df1)
            df2_signal = calculate_unified_signal(df2)
            
            st.markdown(f"""
            ### Combined Analysis Summary
            
            **{ticker1_name} Signal:** {df1_signal['signal']} (Score: {df1_signal['score']:.2f})
            
            **{ticker2_name} Signal:** {df2_signal['signal']} (Score: {df2_signal['score']:.2f})
            
            **Ratio Strategy:**
            
            Based on the comparative analysis:
            
            - If both signals align (both BUY or both SELL), the trend is strong in that direction
            - If signals diverge, consider ratio-based pairs trading opportunities
            - Current ratio positioning suggests {'mean reversion opportunity' if abs(ratio_df['Ratio'].iloc[-1] - ratio_df['Ratio'].mean()) > ratio_df['Ratio'].std() else 'normal range behavior'}
            
            **Recommended Approach:**
            {
            'Focus on the stronger signal between the two tickers' if df1_signal['signal'] != df2_signal['signal']
            else f"Both tickers showing {df1_signal['signal']} - Consider correlated positions"
            }
            """)
        else:
            # Single ticker unified recommendation
            signal = calculate_unified_signal(df1)
            
            st.markdown(f"""
            ### Final Recommendation for {ticker1_name}
            
            **Signal:** {signal['signal']}  
            **Confidence:** {signal['confidence']}  
            **Combined Score:** {signal['score']:.2f}
            
            **Action Plan:**
            
            {
            f"✅ EXECUTE {signal['signal']} - Entry: ₹{signal['entry']:.2f}, Target: ₹{signal['target']:.2f}, Stop: ₹{signal['stop']:.2f}"
            if signal['confidence'] in ['HIGH', 'MODERATE']
            else "⚠️ WAIT - Conflicting signals suggest staying on sidelines until clearer picture emerges"
            }
            
            **Key Points:**
            - Risk/Reward: {signal['rr_ratio']:.2f}
            - Expected Return: {signal['expected_return']:.2f}%
            - Time Horizon: {interval} timeframe
            
            💡 **Pro Tip:** Always use proper position sizing and never risk more than 1-2% of your capital on a single trade.
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; padding: 2rem;'>
            <p><b>Disclaimer:</b> This dashboard provides technical analysis based on historical data. 
            It should not be considered as financial advice. Always do your own research and consult 
            with financial advisors before making investment decisions.</p>
            <p>📊 Professional Algo Trading Dashboard | Built with Streamlit & Python</p>
        </div>
        """, unsafe_allow_html=True)

def calculate_unified_signal(df):
    """Calculate unified signal for a ticker across all factors"""
    try:
        # Calculate indicators
        current_price = df['Close'].iloc[-1]
        rsi = calculate_rsi(df['Close'])
        ema20 = calculate_ema(df['Close'], 20)
        ema50 = calculate_ema(df['Close'], 50)
        ema200 = calculate_ema(df['Close'], 200)
        
        # Returns and statistics
        returns = df['Close'].pct_change() * 100
        mu, std = returns.mean(), returns.std()
        current_z = (returns.iloc[-1] - mu) / std if std > 0 else 0
        
        # ATR
        atr = calculate_atr(df)
        
        # Calculate component scores
        trend_score = 0
        if current_price > ema20.iloc[-1]:
            trend_score += 0.33
        if current_price > ema50.iloc[-1]:
            trend_score += 0.33
        if current_price > ema200.iloc[-1]:
            trend_score += 0.34
        
        # Normalize to -1 to 1
        trend_signal = (trend_score - 0.5) * 2
        
        # RSI signal
        current_rsi = rsi.iloc[-1]
        if current_rsi < 30:
            rsi_signal = 1
        elif current_rsi > 70:
            rsi_signal = -1
        else:
            rsi_signal = (50 - current_rsi) / 20  # Normalize
        
        # Z-score signal
        if current_z > 2:
            z_signal = -1
        elif current_z < -2:
            z_signal = 1
        else:
            z_signal = -current_z / 2  # Normalize
        
        # EMA alignment
        if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
            ema_signal = 1
        elif ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
            ema_signal = -1
        else:
            ema_signal = 0
        
        # Weighted combination
        total_score = (trend_signal * 0.30 + rsi_signal * 0.20 + 
                      z_signal * 0.20 + ema_signal * 0.30)
        
        # Determine signal
        if total_score > 0.4:
            signal = 'BUY'
            confidence = 'HIGH' if total_score > 0.6 else 'MODERATE'
            target = current_price + (2 * atr)
            stop = current_price - atr
        elif total_score < -0.4:
            signal = 'SELL'
            confidence = 'HIGH' if total_score < -0.6 else 'MODERATE'
            target = current_price - (2 * atr)
            stop = current_price + atr
        else:
            signal = 'HOLD'
            confidence = 'LOW'
            target = current_price
            stop = current_price
        
        risk = abs(current_price - stop)
        reward = abs(target - current_price)
        rr_ratio = reward / risk if risk > 0 else 0
        expected_return = ((target - current_price) / current_price) * 100
        
        return {
            'signal': signal,
            'confidence': confidence,
            'score': total_score,
            'entry': current_price,
            'target': target,
            'stop': stop,
            'rr_ratio': rr_ratio,
            'expected_return': expected_return
        }
    except Exception as e:
        return {
            'signal': 'ERROR',
            'confidence': 'NONE',
            'score': 0,
            'entry': 0,
            'target': 0,
            'stop': 0,
            'rr_ratio': 0,
            'expected_return': 0
        }

if __name__ == "__main__":
    main()
