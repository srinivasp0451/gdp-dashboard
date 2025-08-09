import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Universal Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Streamlit for large file uploads programmatically
if 'STREAMLIT_SERVER_MAX_UPLOAD_SIZE' not in os.environ:
    os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '2048'  # 2GB limit

try:
    from streamlit.config import set_option
    set_option('server.maxUploadSize', 2048)  # 2GB in MB
    set_option('server.maxMessageSize', 2048)
except:
    pass

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #007bff;
    margin-bottom: 10px;
}
.signal-buy { 
    background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%); 
    border-left: 4px solid #28a745; 
    padding: 1rem; 
    border-radius: 10px; 
    margin: 10px 0;
}
.signal-sell { 
    background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%); 
    border-left: 4px solid #dc3545; 
    padding: 1rem; 
    border-radius: 10px; 
    margin: 10px 0;
}
.signal-hold { 
    background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%); 
    border-left: 4px solid #ffc107; 
    padding: 1rem; 
    border-radius: 10px; 
    margin: 10px 0;
}
.profit-positive { color: #28a745; font-weight: bold; }
.profit-negative { color: #dc3545; font-weight: bold; }
.profit-neutral { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def detect_csv_format(df):
    """Enhanced column detection with fuzzy matching"""
    st.info(f"🔍 **Analyzing columns:** {list(df.columns)}")
    
    # Comprehensive column variations
    date_variations = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'datetime', 'DateTime', 'time', 'Time']
    open_variations = ['open', 'Open', 'OPEN', 'o', 'O', 'opening', 'Opening', 'open_price']
    high_variations = ['high', 'High', 'HIGH', 'h', 'H', 'maximum', 'Maximum', 'high_price', 'peak']
    low_variations = ['low', 'Low', 'LOW', 'l', 'L', 'minimum', 'Minimum', 'low_price', 'bottom']
    close_variations = ['close', 'Close', 'CLOSE', 'c', 'C', 'closing', 'adj_close', 'adj close', 'Adj Close', 'price', 'Price', 'value', 'Value', 'last', 'final']
    volume_variations = ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'shares', 'Shares', 'shares_traded', 'Shares Traded', 'quantity', 'Quantity']
    
    def fuzzy_match(column_name, variations):
        col_clean = str(column_name).strip().lower()
        for variation in variations:
            if col_clean == variation.lower() or variation.lower() in col_clean or col_clean in variation.lower():
                return True
        return False
    
    column_mapping = {}
    detected_types = []
    
    for col in df.columns:
        if fuzzy_match(col, date_variations):
            column_mapping[col] = 'Date'
            detected_types.append(f"📅 Date: '{col}'")
        elif fuzzy_match(col, open_variations):
            column_mapping[col] = 'Open'
            detected_types.append(f"🟢 Open: '{col}'")
        elif fuzzy_match(col, high_variations):
            column_mapping[col] = 'High'
            detected_types.append(f"📈 High: '{col}'")
        elif fuzzy_match(col, low_variations):
            column_mapping[col] = 'Low'
            detected_types.append(f"📉 Low: '{col}'")
        elif fuzzy_match(col, close_variations):
            column_mapping[col] = 'Close'
            detected_types.append(f"🔴 Close: '{col}'")
        elif fuzzy_match(col, volume_variations):
            column_mapping[col] = 'Volume'
            detected_types.append(f"📊 Volume: '{col}'")
    
    if detected_types:
        st.success("✅ **Detected columns:** " + " | ".join(detected_types))
    
    # Fallback for Close column
    if 'Close' not in column_mapping.values():
        st.warning("⚠️ No Close price column detected. Trying fallback...")
        numeric_columns = []
        for col in df.columns:
            try:
                sample_vals = df[col].dropna().head(10)
                if len(sample_vals) > 0:
                    cleaned_vals = sample_vals.astype(str).str.replace(',', '').str.replace('$', '').str.replace('₹', '')
                    pd.to_numeric(cleaned_vals, errors='raise')
                    numeric_vals = pd.to_numeric(cleaned_vals)
                    if numeric_vals.min() > 0 and numeric_vals.max() < 1000000:
                        numeric_columns.append(col)
            except:
                continue
        
        if numeric_columns:
            first_numeric = numeric_columns[0]
            column_mapping[first_numeric] = 'Close'
            st.success(f"✅ **Using '{first_numeric}' as Close price**")
        else:
            st.error("❌ **Cannot auto-detect Close price column.**")
            price_column = st.selectbox("Select price column:", [""] + list(df.columns), key="manual_price")
            if price_column:
                column_mapping[price_column] = 'Close'
                st.success(f"✅ **Manually selected '{price_column}' as Close price**")
    
    return column_mapping

def load_and_process_data(uploaded_file):
    """Universal data loader with enhanced processing"""
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📁 Processing: **{uploaded_file.name}** ({file_size_mb:.1f} MB)")
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_size_mb > 500:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB)")
        
        progress_bar = st.progress(0) if file_size_mb > 100 else None
        
        # Read file based on extension
        if file_extension == 'csv':
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    if file_size_mb > 500:
                        chunks = pd.read_csv(uploaded_file, encoding=encoding, chunksize=10000)
                        df = pd.concat(chunks, ignore_index=True)
                    else:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"✅ CSV loaded with **{encoding}** encoding")
                    break
                except:
                    continue
            else:
                st.error("Failed to read CSV file")
                return None
                
        elif file_extension in ['xlsx', 'xls']:
            try:
                if file_extension == 'xlsx':
                    import openpyxl
                else:
                    import xlrd
            except ImportError as e:
                st.error(f"📦 Missing dependency: {str(e)}")
                st.code(f"pip install {'openpyxl' if file_extension == 'xlsx' else 'xlrd'}")
                return None
            
            excel_file = pd.ExcelFile(uploaded_file)
            if len(excel_file.sheet_names) > 1:
                selected_sheet = st.selectbox("Select sheet:", excel_file.sheet_names)
            else:
                selected_sheet = excel_file.sheet_names[0]
            
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            st.success(f"✅ Excel loaded from sheet **'{selected_sheet}'**")
        else:
            st.error(f"❌ Unsupported format: **{file_extension}**")
            return None
        
        if progress_bar:
            progress_bar.progress(50)
        
        st.success(f"📊 **Raw data:** {len(df):,} rows × {len(df.columns)} columns")
        
        # Clean and detect columns
        df.columns = df.columns.str.strip()
        column_mapping = detect_csv_format(df)
        
        st.subheader("📋 Data Preview:")
        st.dataframe(df.head(), use_container_width=True)
        
        if not column_mapping or 'Close' not in column_mapping.values():
            st.error("❌ Could not detect required columns")
            return None
        
        # Process data
        df_renamed = df.rename(columns=column_mapping)
        
        # Fill missing OHLC with Close
        for col in ['Open', 'High', 'Low']:
            if col not in df_renamed.columns:
                df_renamed[col] = df_renamed['Close']
                st.warning(f"⚠️ Using Close price for missing **{col}** column")
        
        if 'Volume' not in df_renamed.columns:
            df_renamed['Volume'] = 1000000
            st.info("💹 Added default volume values")
        
        # Convert data types
        date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%d-%b-%Y']
        for fmt in date_formats:
            try:
                df_renamed['Date'] = pd.to_datetime(df_renamed['Date'], format=fmt)
                st.success(f"✅ **Date format:** {fmt}")
                break
            except:
                continue
        else:
            df_renamed['Date'] = pd.to_datetime(df_renamed['Date'], errors='coerce')
            st.info("✅ Date format auto-detected")
        
        # Convert numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_renamed.columns:
                if df_renamed[col].dtype == 'object':
                    df_renamed[col] = df_renamed[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('₹', '')
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
        
        # Clean and validate data
        df_renamed = df_renamed.sort_values('Date').reset_index(drop=True)
        initial_rows = len(df_renamed)
        df_renamed = df_renamed.dropna(subset=['Date', 'Close'])
        df_renamed = df_renamed[(df_renamed['Open'] > 0) & (df_renamed['High'] > 0) & 
                               (df_renamed['Low'] > 0) & (df_renamed['Close'] > 0)]
        
        if len(df_renamed) < initial_rows:
            st.warning(f"⚠️ Removed {initial_rows - len(df_renamed)} invalid rows")
        
        if len(df_renamed) < 10:
            st.error(f"❌ Insufficient data: {len(df_renamed)} rows")
            return None
        
        # Fix OHLC relationships
        ohlc_issues = ((df_renamed['High'] < df_renamed['Low']) | 
                      (df_renamed['High'] < df_renamed['Open']) | 
                      (df_renamed['High'] < df_renamed['Close']) |
                      (df_renamed['Low'] > df_renamed['Open']) | 
                      (df_renamed['Low'] > df_renamed['Close'])).sum()
        
        if ohlc_issues > 0:
            df_renamed['High'] = df_renamed[['Open', 'High', 'Low', 'Close']].max(axis=1)
            df_renamed['Low'] = df_renamed[['Open', 'High', 'Low', 'Close']].min(axis=1)
            st.info(f"✅ **Auto-corrected** {ohlc_issues} OHLC inconsistencies")
        
        if progress_bar:
            progress_bar.progress(100)
            progress_bar.empty()
        
        date_range = f"{df_renamed['Date'].min().strftime('%Y-%m-%d')} to {df_renamed['Date'].max().strftime('%Y-%m-%d')}"
        price_range = f"₹{df_renamed['Close'].min():.2f} - ₹{df_renamed['Close'].max():.2f}"
        st.success(f"🎉 **Processing complete!**")
        st.info(f"📊 **Final:** {len(df_renamed):,} rows | **Range:** {date_range} | **Prices:** {price_range}")
        
        return df_renamed[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        st.error(f"❌ **Error:** {str(e)}")
        return None

def add_technical_indicators(df):
    """Add comprehensive technical indicators"""
    df = df.copy()
    
    # Basic indicators
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Support/Resistance
    df['Resistance'] = df['High'].rolling(20).max()
    df['Support'] = df['Low'].rolling(20).min()
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(20).std()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    return df

def generate_strategy_signals(df, strategy_name, params=None):
    """Generate detailed buy/sell signals with reasoning"""
    df = df.copy()
    df['Signal'] = 'HOLD'
    df['Signal_Strength'] = 0
    df['Entry_Reason'] = ''
    df['Exit_Reason'] = ''
    df['Stop_Loss'] = np.nan
    df['Take_Profit'] = np.nan
    
    if params is None:
        params = {}
    
    if strategy_name == "Buy & Hold":
        df.loc[0, 'Signal'] = 'BUY'
        df.loc[0, 'Entry_Reason'] = 'Buy and Hold Strategy - Initial Purchase'
        df.loc[0, 'Signal_Strength'] = 100
        
    elif strategy_name == "Enhanced Contrarian":
        drop_threshold = params.get('drop_threshold', -2.0)
        gain_threshold = params.get('gain_threshold', 1.5)
        
        # Buy conditions
        volume_spike = df['Volume_Ratio'] > 1.2
        oversold_rsi = df['RSI'] < 35
        big_drop = df['Daily_Return'] <= drop_threshold
        near_support = df['Close'] <= df['Support'] * 1.02
        
        buy_condition = big_drop & (oversold_rsi | volume_spike | near_support)
        
        # Sell conditions
        overbought_rsi = df['RSI'] > 65
        good_gain = df['Daily_Return'] >= gain_threshold
        near_resistance = df['Close'] >= df['Resistance'] * 0.98
        
        sell_condition = good_gain & (overbought_rsi | near_resistance)
        
        df.loc[buy_condition, 'Signal'] = 'BUY'
        df.loc[buy_condition, 'Entry_Reason'] = df.loc[buy_condition].apply(
            lambda row: f"Big drop ({row['Daily_Return']:.1f}%) + RSI({row['RSI']:.0f}) + Volume({row['Volume_Ratio']:.1f}x)", axis=1
        )
        df.loc[buy_condition, 'Signal_Strength'] = (
            50 + abs(df.loc[buy_condition, 'Daily_Return']) * 10 + 
            (70 - df.loc[buy_condition, 'RSI'].fillna(50))
        ).clip(0, 100)
        df.loc[buy_condition, 'Stop_Loss'] = df.loc[buy_condition, 'Close'] * 0.95
        df.loc[buy_condition, 'Take_Profit'] = df.loc[buy_condition, 'Close'] * 1.03
        
        df.loc[sell_condition, 'Signal'] = 'SELL'
        df.loc[sell_condition, 'Exit_Reason'] = df.loc[sell_condition].apply(
            lambda row: f"Good gain ({row['Daily_Return']:.1f}%) + RSI({row['RSI']:.0f})", axis=1
        )
        
    elif strategy_name == "Smart Momentum":
        # Momentum conditions
        sma_bullish = df['SMA_5'] > df['SMA_20']
        macd_bullish = df['MACD'] > df['MACD_Signal']
        rsi_momentum = (df['RSI'] > 50) & (df['RSI'] < 80)
        volume_confirm = df['Volume_Ratio'] > 1.1
        
        buy_condition = sma_bullish & macd_bullish & rsi_momentum & volume_confirm
        
        # Exit conditions
        sma_bearish = df['SMA_5'] < df['SMA_20']
        macd_bearish = df['MACD'] < df['MACD_Signal']
        rsi_overbought = df['RSI'] > 75
        
        sell_condition = sma_bearish | macd_bearish | rsi_overbought
        
        df.loc[buy_condition, 'Signal'] = 'BUY'
        df.loc[buy_condition, 'Entry_Reason'] = 'Momentum: SMA+MACD+RSI+Volume aligned'
        df.loc[buy_condition, 'Signal_Strength'] = (
            (df.loc[buy_condition, 'RSI'].fillna(50) - 50) + 
            df.loc[buy_condition, 'Volume_Ratio'] * 20 + 30
        ).clip(0, 100)
        df.loc[buy_condition, 'Stop_Loss'] = df.loc[buy_condition, 'SMA_20']
        
        df.loc[sell_condition, 'Signal'] = 'SELL'
        df.loc[sell_condition, 'Exit_Reason'] = 'Momentum weakening'
        
    elif strategy_name == "Mean Reversion Pro":
        # Oversold conditions
        rsi_oversold = df['RSI'] < 30
        stoch_oversold = df['Stoch_K'] < 20
        bb_oversold = df['BB_Position'] < 0.1
        
        buy_condition = (rsi_oversold & stoch_oversold) | bb_oversold
        
        # Overbought conditions
        rsi_overbought = df['RSI'] > 70
        stoch_overbought = df['Stoch_K'] > 80
        bb_overbought = df['BB_Position'] > 0.9
        
        sell_condition = (rsi_overbought & stoch_overbought) | bb_overbought
        
        df.loc[buy_condition, 'Signal'] = 'BUY'
        df.loc[buy_condition, 'Entry_Reason'] = df.loc[buy_condition].apply(
            lambda row: f"Oversold: RSI({row['RSI']:.0f}) Stoch({row['Stoch_K']:.0f})", axis=1
        )
        df.loc[buy_condition, 'Signal_Strength'] = (100 - df.loc[buy_condition, 'RSI'].fillna(50)).clip(0, 100)
        
        df.loc[sell_condition, 'Signal'] = 'SELL'
        df.loc[sell_condition, 'Exit_Reason'] = df.loc[sell_condition].apply(
            lambda row: f"Overbought: RSI({row['RSI']:.0f}) Stoch({row['Stoch_K']:.0f})", axis=1
        )
    
    return df




