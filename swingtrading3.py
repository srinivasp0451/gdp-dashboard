import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Universal Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
.trade-entry { background-color: #e8f5e8; }
.trade-exit { background-color: #ffe8e8; }
</style>
""", unsafe_allow_html=True)

def detect_csv_format(df):
    """Detect and standardize CSV format for universal compatibility"""
    # Common column name variations
    date_cols = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'time', 'Time']
    open_cols = ['open', 'Open', 'OPEN', 'o', 'O']
    high_cols = ['high', 'High', 'HIGH', 'h', 'H']
    low_cols = ['low', 'Low', 'LOW', 'l', 'L']
    close_cols = ['close', 'Close', 'CLOSE', 'c', 'C', 'adj close', 'Adj Close', 'adjclose']
    volume_cols = ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'shares traded', 'Shares Traded', 'quantity']
    
    # Create mapping
    column_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(d.lower() in col_lower for d in date_cols):
            column_mapping[col] = 'Date'
        elif any(o.lower() in col_lower for o in open_cols):
            column_mapping[col] = 'Open'
        elif any(h.lower() in col_lower for h in high_cols):
            column_mapping[col] = 'High'
        elif any(l.lower() in col_lower for l in low_cols):
            column_mapping[col] = 'Low'
        elif any(c.lower() in col_lower for c in close_cols):
            column_mapping[col] = 'Close'
        elif any(v.lower() in col_lower for v in volume_cols):
            column_mapping[col] = 'Volume'
    
    return column_mapping

def load_and_process_data(uploaded_file):
    """Universal data loader for any stock CSV format"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except:
                continue
        
        if df is None:
            st.error("Could not read the CSV file with any encoding")
            return None
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Detect column format
        column_mapping = detect_csv_format(df)
        
        if not column_mapping:
            st.error("Could not detect standard OHLC columns. Please check your CSV format.")
            st.write("Available columns:", df.columns.tolist())
            st.write("Expected: Date, Open, High, Low, Close (and optionally Volume)")
            return None
        
        # Rename columns to standard format
        df_renamed = df.rename(columns=column_mapping)
        
        # Ensure we have minimum required columns
        required_cols = ['Date', 'Close']
        missing_cols = [col for col in required_cols if col not in df_renamed.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Fill missing OHLC columns with Close price if not available
        for col in ['Open', 'High', 'Low']:
            if col not in df_renamed.columns:
                df_renamed[col] = df_renamed['Close']
                st.warning(f"Column '{col}' not found, using Close price as substitute")
        
        # Add default volume if not present
        if 'Volume' not in df_renamed.columns:
            df_renamed['Volume'] = 1000000  # Default volume
            st.info("Volume column not found, using default values")
        
        # Convert data types
        try:
            # Try multiple date formats
            date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%d-%b-%Y', '%Y/%m/%d']
            
            for fmt in date_formats:
                try:
                    df_renamed['Date'] = pd.to_datetime(df_renamed['Date'], format=fmt, errors='raise')
                    break
                except:
                    continue
            else:
                # If no format works, try pandas auto-detection
                df_renamed['Date'] = pd.to_datetime(df_renamed['Date'], errors='coerce')
                
        except Exception as e:
            st.error(f"Error parsing dates: {str(e)}")
            return None
        
        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df_renamed.columns:
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
        
        # Sort by date and remove NaN values
        df_renamed = df_renamed.sort_values('Date').reset_index(drop=True)
        df_renamed = df_renamed.dropna(subset=['Date', 'Close'])
        
        # Validate data
        if len(df_renamed) < 10:
            st.error("Insufficient data. Need at least 10 data points.")
            return None
            
        return df_renamed[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def add_technical_indicators(df):
    """Add comprehensive technical indicators"""
    df = df.copy()
    
    # Price-based indicators
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Price_Change'] = df['Close'].diff()
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    
    # Moving Averages (various periods)
    for period in [5, 10, 20, 50]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Support and Resistance levels
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    
    # Price patterns
    df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df['ATR'] = ((df['High'] - df['Low']).rolling(window=14).mean())  # Simplified ATR
    
    return df

def generate_strategy_signals(df, strategy_name, params=None):
    """Generate detailed buy/sell signals with reasoning"""
    df = df.copy()
    df['Signal'] = 'HOLD'
    df['Signal_Strength'] = 0  # 0-100 scale
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
        # Enhanced contrarian with multiple conditions
        drop_threshold = params.get('drop_threshold', -2.0)
        gain_threshold = params.get('gain_threshold', 1.5)
        
        # Buy conditions: Big drops + oversold conditions + volume spike
        volume_spike = df['Volume_Ratio'] > 1.2
        oversold_rsi = df['RSI'] < 35
        big_drop = df['Daily_Return'] <= drop_threshold
        near_support = df['Close'] <= df['Support'] * 1.02
        
        buy_condition = big_drop & (oversold_rsi | volume_spike | near_support)
        
        # Sell conditions: Good gains + overbought conditions
        overbought_rsi = df['RSI'] > 65
        good_gain = df['Daily_Return'] >= gain_threshold
        near_resistance = df['Close'] >= df['Resistance'] * 0.98
        
        sell_condition = good_gain & (overbought_rsi | near_resistance)
        
        df.loc[buy_condition, 'Signal'] = 'BUY'
        df.loc[buy_condition, 'Entry_Reason'] = df.loc[buy_condition].apply(
            lambda row: f"Big drop ({row['Daily_Return']:.1f}%) + " + 
            f"RSI({row['RSI']:.0f}) " +
            f"Volume({row['Volume_Ratio']:.1f}x)", axis=1
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
        df.loc[sell_condition, 'Signal_Strength'] = (
            50 + df.loc[sell_condition, 'Daily_Return'] * 10 + 
            (df.loc[sell_condition, 'RSI'].fillna(50) - 50)
        ).clip(0, 100)
        
    elif strategy_name == "Smart Momentum":
        # Multi-timeframe momentum with volume confirmation
        
        # Momentum conditions
        sma_bullish = df['SMA_5'] > df['SMA_20']
        macd_bullish = df['MACD'] > df['MACD_Signal']
        rsi_momentum = (df['RSI'] > 50) & (df['RSI'] < 80)
        volume_confirm = df['Volume_Ratio'] > 1.1
        breakout = df['Close'] > df['Resistance'].shift(1)
        
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
        # Advanced mean reversion with multiple oscillators
        
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
            lambda row: f"Oversold: RSI({row['RSI']:.0f}) Stoch({row['Stoch_K']:.0f}) BB({row['BB_Position']:.2f})", axis=1
        )
        df.loc[buy_condition, 'Signal_Strength'] = (100 - df.loc[buy_condition, 'RSI'].fillna(50)).clip(0, 100)
        
        df.loc[sell_condition, 'Signal'] = 'SELL'
        df.loc[sell_condition, 'Exit_Reason'] = df.loc[sell_condition].apply(
            lambda row: f"Overbought: RSI({row['RSI']:.0f}) Stoch({row['Stoch_K']:.0f})", axis=1
        )
    
    return df

def advanced_backtest(df, initial_capital=100000, transaction_cost_pct=0.1, stop_loss_pct=5, take_profit_pct=10):
    """Advanced backtesting with detailed trade analysis"""
    df = df.copy()
    
    # Initialize tracking variables
    cash = initial_capital
    shares = 0
    portfolio_value = []
    trades = []
    position_history = []
    
    current_position = None
    entry_date = None
    entry_price = None
    entry_reason = None
    days_in_position = 0
    
    for i, row in df.iterrows():
        current_price = row['Close']
        signal = row['Signal']
        signal_strength = row.get('Signal_Strength', 0)
        
        # Calculate current portfolio value
        current_portfolio_value = cash + (shares * current_price)
        portfolio_value.append(current_portfolio_value)
        
        # Track position
        days_in_position = days_in_position + 1 if shares > 0 else 0
        
        # Check stop loss and take profit for existing positions
        if shares > 0 and current_position:
            stop_loss_price = current_position.get('stop_loss', 0)
            take_profit_price = current_position.get('take_profit', float('inf'))
            
            if current_price <= stop_loss_price:
                signal = 'SELL'
                exit_reason = f"Stop Loss triggered at {current_price:.2f}"
            elif current_price >= take_profit_price:
                signal = 'SELL' 
                exit_reason = f"Take Profit triggered at {current_price:.2f}"
            else:
                exit_reason = row.get('Exit_Reason', 'Strategy exit signal')
        
        # Execute trades
        if signal == 'BUY' and shares == 0 and cash > current_price:
            # Calculate position size (use percentage of capital)
            position_size_pct = min(0.95, signal_strength / 100) if signal_strength > 0 else 0.95
            available_cash = cash * position_size_pct
            shares_to_buy = int(available_cash / current_price)
            
            if shares_to_buy > 0:
                cost_before_fees = shares_to_buy * current_price
                transaction_fee = cost_before_fees * (transaction_cost_pct / 100)
                total_cost = cost_before_fees + transaction_fee
                
                if cash >= total_cost:
                    shares = shares_to_buy
                    cash -= total_cost
                    entry_date = row['Date']
                    entry_price = current_price
                    entry_reason = row.get('Entry_Reason', 'Buy signal')
                    
                    # Set stop loss and take profit
                    stop_loss_price = current_price * (1 - stop_loss_pct/100)
                    take_profit_price = current_price * (1 + take_profit_pct/100)
                    if not pd.isna(row.get('Stop_Loss')):
                        stop_loss_price = row['Stop_Loss']
                    if not pd.isna(row.get('Take_Profit')):
                        take_profit_price = row['Take_Profit']
                    
                    current_position = {
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'entry_reason': entry_reason
                    }
                    
                    trades.append({
                        'Date': row['Date'],
                        'Type': 'BUY',
                        'Price': current_price,
                        'Shares': shares,
                        'Cost': total_cost,
                        'Transaction_Fee': transaction_fee,
                        'Reason': entry_reason,
                        'Signal_Strength': signal_strength,
                        'Stop_Loss': stop_loss_price,
                        'Take_Profit': take_profit_price,
                        'Portfolio_Value': current_portfolio_value
                    })
                    
                    days_in_position = 0
        
        elif signal == 'SELL' and shares > 0:
            # Sell position
            proceeds_before_fees = shares * current_price
            transaction_fee = proceeds_before_fees * (transaction_cost_pct / 100)
            net_proceeds = proceeds_before_fees - transaction_fee
            
            cash += net_proceeds
            
            # Calculate trade performance
            if current_position:
                profit_loss = net_proceeds - current_position['entry_price'] * shares
                profit_loss_pct = (profit_loss / (current_position['entry_price'] * shares)) * 100
                holding_days = (row['Date'] - current_position['entry_date']).days
                
                trades.append({
                    'Date': row['Date'],
                    'Type': 'SELL',
                    'Price': current_price,
                    'Shares': shares,
                    'Proceeds': net_proceeds,
                    'Transaction_Fee': transaction_fee,
                    'Profit_Loss': profit_loss,
                    'Profit_Loss_Pct': profit_loss_pct,
                    'Holding_Days': holding_days,
                    'Entry_Price': current_position['entry_price'],
                    'Entry_Date': current_position['entry_date'],
                    'Entry_Reason': current_position['entry_reason'],
                    'Exit_Reason': locals().get('exit_reason', row.get('Exit_Reason', 'Sell signal')),
                    'Portfolio_Value': current_portfolio_value
                })
            
            shares = 0
            current_position = None
            days_in_position = 0
        
        # Record position status
        position_history.append({
            'Date': row['Date'],
            'Portfolio_Value': current_portfolio_value,
            'Cash': cash,
            'Shares': shares,
            'Price': current_price,
            'Signal': signal,
            'Days_In_Position': days_in_position,
            'Unrealized_PL': (shares * current_price - shares * entry_price) if shares > 0 and entry_price else 0
        })
    
    # Final sell if still holding
    if shares > 0:
        final_price = df.iloc[-1]['Close']
        proceeds_before_fees = shares * final_price
        transaction_fee = proceeds_before_fees * (transaction_cost_pct / 100)
        net_proceeds = proceeds_before_fees - transaction_fee
        cash += net_proceeds
        
        if current_position:
            profit_loss = net_proceeds - current_position['entry_price'] * shares
            profit_loss_pct = (profit_loss / (current_position['entry_price'] * shares)) * 100
            holding_days = (df.iloc[-1]['Date'] - current_position['entry_date']).days
            
            trades.append({
                'Date': df.iloc[-1]['Date'],
                'Type': 'FINAL_SELL',
                'Price': final_price,
                'Shares': shares,
                'Proceeds': net_proceeds,
                'Transaction_Fee': transaction_fee,
                'Profit_Loss': profit_loss,
                'Profit_Loss_Pct': profit_loss_pct,
                'Holding_Days': holding_days,
                'Entry_Price': current_position['entry_price'],
                'Entry_Date': current_position['entry_date'],
                'Entry_Reason': current_position['entry_reason'],
                'Exit_Reason': 'Final liquidation',
                'Portfolio_Value': cash
            })
    
    df['Portfolio_Value'] = portfolio_value
    
    # Calculate comprehensive metrics
    final_value = cash
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
    
    # Detailed trade analysis
    completed_trades = [t for t in trades if t['Type'] in ['SELL', 'FINAL_SELL']]
    
    if completed_trades:
        profits = [t['Profit_Loss'] for t in completed_trades if 'Profit_Loss' in t]
        profit_pcts = [t['Profit_Loss_Pct'] for t in completed_trades if 'Profit_Loss_Pct' in t]
        holding_days = [t['Holding_Days'] for t in completed_trades if 'Holding_Days' in t]
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        total_trades = len(completed_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_profit_per_trade = np.mean(profits) if profits else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        avg_holding_days = np.mean(holding_days) if holding_days else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
        
        max_win = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0
        
        total_fees = sum([t.get('Transaction_Fee', 0) for t in trades])
        
    else:
        total_trades = win_count = loss_count = 0
        win_rate = avg_profit_per_trade = avg_win = avg_loss = 0
        avg_holding_days = profit_factor = max_win = max_loss = total_fees = 0
    
    # Risk metrics
    returns = pd.Series(portfolio_value).pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
    
    # Maximum drawdown
    peak = pd.Series(portfolio_value).expanding().max()
    drawdown = (pd.Series(portfolio_value) - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (simplified)
    excess_returns = returns.mean() * 252  # Annualized
    sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    metrics = {
        'Total_Return': total_return,
        'Buy_Hold_Return': buy_hold_return,
        'Outperformance': total_return - buy_hold_return,
        'Final_Value': final_value,
        'Total_Trades': total_trades,
        'Win_Count': win_count,
        'Loss_Count': loss_count,
        'Win_Rate': win_rate,
        'Avg_Profit_Per_Trade': avg_profit_per_trade,
        'Avg_Win': avg_win,
        'Avg_Loss': avg_loss,
        'Max_Win': max_win,
        'Max_Loss': max_loss,
        'Profit_Factor': profit_factor,
        'Avg_Holding_Days': avg_holding_days,
        'Total_Fees': total_fees,
        'Max_Drawdown': max_drawdown,
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio
    }
    
    return df, trades, metrics, position_history

def get_current_signals(df, strategies):
    """Generate current market signals based on latest data"""
    if len(df) < 50:
        return {}
    
    latest_data = df.iloc[-1]
    signals = {}
    
    for strategy_name in strategies:
        df_with_signals = generate_strategy_signals(df, strategy_name)
        latest_signal = df_with_signals.iloc[-1]
        
        # Get signal details
        current_signal = latest_signal['Signal']
        signal_strength = latest_signal.get('Signal_Strength', 0)
        entry_reason = latest_signal.get('Entry_Reason', '')
        exit_reason = latest_signal.get('Exit_Reason', '')
        stop_loss = latest_signal.get('Stop_Loss', None)
        take_profit = latest_signal.get('Take_Profit', None)
        
        # Add market context
        price = latest_signal['Close']
        rsi = latest_signal.get('RSI', 50)
        volume_ratio = latest_signal.get('Volume_Ratio', 1)
        bb_position = latest_signal.get('BB_Position', 0.5)
        
        signals[strategy_name] = {
            'signal': current_signal,
            'strength': signal_strength,
            'price': price,
            'entry_reason': entry_reason,
            'exit_reason': exit_reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'bb_position': bb_position,
            'date': latest_signal['Date']
        }
    
    return signals

def create_detailed_charts(df, trades, strategy_name):
    """Create comprehensive charts with trade markers"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'Price Action with Trade Signals',
            'Portfolio Value vs Buy & Hold',
            'Technical Indicators (RSI, MACD)',
            'Volume Analysis'
        ],
        row_heights=[0.4, 0.25, 0.2, 0.15],
        vertical_spacing=0.05
    )
    
    # 1. Price chart with trades
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', 
                  line=dict(color='blue', width=1), opacity=0.7),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', 
                  line=dict(color='gray', width=1, dash='dash'), opacity=0.5),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', 
                  line=dict(color='gray', width=1, dash='dash'), opacity=0.5),
        row=1, col=1
    )
    
    # Add trade markers
    buy_trades = [t for t in trades if t['Type'] == 'BUY']
    sell_trades = [t for t in trades if t['Type'] in ['SELL', 'FINAL_SELL']]
    
    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[t['Date'] for t in buy_trades],
                y=[t['Price'] for t in buy_trades],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy Signals',
                hovertemplate='<b>BUY</b><br>Date: %{x}<br>Price: ₹%{y}<br>Reason: %{text}',
                text=[t.get('Reason', '') for t in buy_trades]
            ),
            row=1, col=1
        )
    
    if sell_trades:
        fig.add_trace(
            go.Scatter(
                x=[t['Date'] for t in sell_trades],
                y=[t['Price'] for t in sell_trades],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell Signals',
                hovertemplate='<b>SELL</b><br>Date: %{x}<br>Price: ₹%{y}<br>P&L: ₹%{text}',
                text=[f"{t.get('Profit_Loss', 0):.0f}" for t in sell_trades]
            ),
            row=1, col=1
        )
    
    # 2. Portfolio performance
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Portfolio_Value'], name='Portfolio Value', 
                  line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    # Normalized buy & hold for comparison
    initial_price = df.iloc[0]['Close']
    normalized_bh = (df['Close'] / initial_price) * 100000
    fig.add_trace(
        go.Scatter(x=df['Date'], y=normalized_bh, name='Buy & Hold (Normalized)', 
                  line=dict(color='red', width=2, dash='dash')),
        row=2, col=1
    )
    
    # 3. Technical indicators
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', 
                  line=dict(color='purple', width=1)),
        row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # 4. Volume
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', 
               marker_color=colors, opacity=0.7),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Volume_MA'], name='Volume MA', 
                  line=dict(color='orange', width=2)),
        row=4, col=1
    )
    
    fig.update_layout(
        title=f'Comprehensive Analysis: {strategy_name}',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value (₹)", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def display_live_signals(signals):
    """Display current market signals and recommendations"""
    st.subheader("🔴 LIVE Market Signals & Recommendations")
    
    if not signals:
        st.warning("Insufficient data for signal generation (need at least 50 data points)")
        return
    
    # Get latest date
    latest_date = list(signals.values())[0]['date']
    st.info(f"📅 **Signals based on latest data:** {latest_date.strftime('%Y-%m-%d')}")
    
    cols = st.columns(len(signals))
    
    for i, (strategy, signal_data) in enumerate(signals.items()):
        with cols[i]:
            signal_type = signal_data['signal']
            strength = signal_data['strength']
            price = signal_data['price']
            
            # Signal styling
            if signal_type == 'BUY':
                signal_class = 'signal-buy'
                signal_emoji = '🟢'
                signal_text = 'BUY SIGNAL'
            elif signal_type == 'SELL':
                signal_class = 'signal-sell'
                signal_emoji = '🔴'
                signal_text = 'SELL SIGNAL'
            else:
                signal_class = 'signal-hold'
                signal_emoji = '🟡'
                signal_text = 'HOLD'
            
            st.markdown(f"""
            <div class="{signal_class}">
                <h4>{signal_emoji} {strategy}</h4>
                <h3>{signal_text}</h3>
                <p><strong>Current Price:</strong> ₹{price:.2f}</p>
                <p><strong>Signal Strength:</strong> {strength:.0f}%</p>
                <p><strong>RSI:</strong> {signal_data['rsi']:.1f}</p>
                <p><strong>Volume Ratio:</strong> {signal_data['volume_ratio']:.1f}x</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show entry/exit reasoning
            if signal_data['entry_reason']:
                st.write(f"**Entry Logic:** {signal_data['entry_reason']}")
            if signal_data['exit_reason']:
                st.write(f"**Exit Logic:** {signal_data['exit_reason']}")
            
            # Show stop loss and take profit levels
            if signal_data['stop_loss'] and not pd.isna(signal_data['stop_loss']):
                st.write(f"**Stop Loss:** ₹{signal_data['stop_loss']:.2f}")
            if signal_data['take_profit'] and not pd.isna(signal_data['take_profit']):
                st.write(f"**Take Profit:** ₹{signal_data['take_profit']:.2f}")

def display_trade_analysis(trades):
    """Display detailed trade analysis"""
    if not trades:
        st.warning("No trades executed by this strategy")
        return
    
    st.subheader("🔍 Detailed Trade Analysis")
    
    # Separate buy and sell trades
    buy_trades = [t for t in trades if t['Type'] == 'BUY']
    sell_trades = [t for t in trades if t['Type'] in ['SELL', 'FINAL_SELL']]
    
    if sell_trades:
        trades_df = pd.DataFrame(sell_trades)
        
        # Trade summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            profitable_trades = len([t for t in sell_trades if t.get('Profit_Loss', 0) > 0])
            st.metric("Profitable Trades", f"{profitable_trades}/{len(sell_trades)}")
        
        with col2:
            avg_holding = np.mean([t.get('Holding_Days', 0) for t in sell_trades])
            st.metric("Avg Holding Days", f"{avg_holding:.1f}")
        
        with col3:
            total_profit = sum([t.get('Profit_Loss', 0) for t in sell_trades])
            st.metric("Total P&L", f"₹{total_profit:,.0f}")
        
        with col4:
            avg_return = np.mean([t.get('Profit_Loss_Pct', 0) for t in sell_trades])
            st.metric("Avg Return %", f"{avg_return:.1f}%")
        
        # Detailed trade table
        st.subheader("📊 Individual Trade Results")
        
        display_columns = [
            'Entry_Date', 'Date', 'Entry_Price', 'Price', 
            'Shares', 'Holding_Days', 'Profit_Loss', 'Profit_Loss_Pct',
            'Entry_Reason', 'Exit_Reason'
        ]
        
        available_columns = [col for col in display_columns if col in trades_df.columns]
        display_df = trades_df[available_columns].copy()
        
        # Format the display
        if 'Entry_Date' in display_df.columns:
            display_df['Entry_Date'] = pd.to_datetime(display_df['Entry_Date']).dt.strftime('%Y-%m-%d')
        if 'Date' in display_df.columns:
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Color code profitable/unprofitable trades
        def highlight_profit_loss(val):
            if pd.isna(val):
                return ''
            if val > 0:
                return 'background-color: #d4edda; color: #155724'
            elif val < 0:
                return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        if 'Profit_Loss' in display_df.columns:
            styled_df = display_df.style.applymap(highlight_profit_loss, subset=['Profit_Loss'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
        # Trade distribution chart
        if 'Profit_Loss' in trades_df.columns:
            st.subheader("📈 Trade P&L Distribution")
            fig_hist = px.histogram(
                trades_df, 
                x='Profit_Loss', 
                title='Distribution of Trade Profits/Losses',
                labels={'Profit_Loss': 'Profit/Loss (₹)', 'count': 'Number of Trades'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even")
            st.plotly_chart(fig_hist, use_container_width=True)

def main():
    st.title("📈 Universal Trading Strategy Backtester")
    st.markdown("**Upload any stock data CSV and get comprehensive backtest results + live signals!**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload with larger size limit
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Stock Data CSV",
        type=['*'],
        help="Supports any OHLC format. Auto-detects columns. Max size: 200MB"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("🔄 Loading and processing data..."):
            df = load_and_process_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ **Data loaded successfully!**")
            st.info(f"📊 **{len(df)} data points** from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            
            # Add technical indicators
            with st.spinner("📈 Calculating technical indicators..."):
                df = add_technical_indicators(df)
            
            # Strategy and parameter selection
            st.sidebar.markdown("### 🎯 Strategy Selection")
            
            strategies = [
                "Buy & Hold",
                "Enhanced Contrarian",
                "Smart Momentum", 
                "Mean Reversion Pro"
            ]
            
            selected_strategy = st.sidebar.selectbox("📊 Choose Strategy", strategies, index=1)
            
            # Trading parameters
            st.sidebar.markdown("### 💰 Trading Parameters")
            initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, min_value=10000, step=10000)
            transaction_cost = st.sidebar.slider("Transaction Cost (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
            stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
            take_profit_pct = st.sidebar.slider("Take Profit (%)", min_value=2.0, max_value=50.0, value=10.0, step=1.0)
            
            # Strategy descriptions
            strategy_descriptions = {
                "Buy & Hold": "📊 **Benchmark Strategy**: Buy on first day and hold till the end. Your performance baseline.",
                "Enhanced Contrarian": "🔄 **Advanced Mean Reversion**: Buys on big drops (2%+) with oversold conditions + volume confirmation. Sells on gains (1.5%+) with overbought signals.",
                "Smart Momentum": "🚀 **Multi-Timeframe Momentum**: Uses SMA, MACD, RSI, and volume alignment for trend-following entries. Exits on momentum weakness.",
                "Mean Reversion Pro": "🎯 **Professional Mean Reversion**: Combines RSI, Stochastic, and Bollinger Bands for precise oversold/overbought identification."
            }
            
            st.info(strategy_descriptions[selected_strategy])
            
            # Generate signals and backtest
            with st.spinner("🔄 Running comprehensive backtest..."):
                df_with_signals = generate_strategy_signals(df, selected_strategy)
                df_result, trades, metrics, position_history = advanced_backtest(
                    df_with_signals, initial_capital, transaction_cost, stop_loss_pct, take_profit_pct
                )
            
            # LIVE SIGNALS SECTION
            with st.spinner("📡 Generating live market signals..."):
                live_signals = get_current_signals(df, strategies)
                display_live_signals(live_signals)
            
            # PERFORMANCE RESULTS
            st.header("🎯 Comprehensive Performance Results")
            
            # Enhanced metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                return_class = "profit-positive" if metrics['Total_Return'] > 0 else "profit-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💰 Strategy Return</h4>
                    <h2 class="{return_class}">{metrics['Total_Return']:.2f}%</h2>
                    <p>Final Value: ₹{metrics['Final_Value']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                outperf_class = "profit-positive" if metrics['Outperformance'] > 0 else "profit-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 vs Buy & Hold</h4>
                    <h2 class="{outperf_class}">{metrics['Outperformance']:+.2f}%</h2>
                    <p>B&H Return: {metrics['Buy_Hold_Return']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                win_rate_class = "profit-positive" if metrics['Win_Rate'] >= 50 else "profit-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Win Rate</h4>
                    <h2 class="{win_rate_class}">{metrics['Win_Rate']:.1f}%</h2>
                    <p>{metrics['Win_Count']}W / {metrics['Loss_Count']}L</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚡ Performance Metrics</h4>
                    <p><strong>Max Drawdown:</strong> {metrics['Max_Drawdown']:.1f}%</p>
                    <p><strong>Sharpe Ratio:</strong> {metrics['Sharpe_Ratio']:.2f}</p>
                    <p><strong>Avg Hold:</strong> {metrics['Avg_Holding_Days']:.0f} days</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("💹 Total Trades", metrics['Total_Trades'])
            with col6:
                profit_class = "profit-positive" if metrics['Avg_Profit_Per_Trade'] > 0 else "profit-negative"
                st.markdown(f"**Avg P&L/Trade:** <span class='{profit_class}'>₹{metrics['Avg_Profit_Per_Trade']:,.0f}</span>", unsafe_allow_html=True)
            with col7:
                st.metric("💸 Total Fees", f"₹{metrics['Total_Fees']:,.0f}")
            with col8:
                pf_display = f"{metrics['Profit_Factor']:.2f}" if metrics['Profit_Factor'] != float('inf') else "∞"
                st.metric("🎲 Profit Factor", pf_display)
            
            # Comprehensive charts
            st.subheader("📈 Detailed Performance Analysis")
            detailed_chart = create_detailed_charts(df_result, trades, selected_strategy)
            st.plotly_chart(detailed_chart, use_container_width=True)
            
            # Detailed trade analysis
            display_trade_analysis(trades)
            
            # Performance summary and insights
            st.subheader("🧠 Strategy Insights & Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Performance Analysis")
                
                if metrics['Outperformance'] > 5:
                    st.success(f"🎉 **Excellent Performance!** Strategy beats buy-and-hold by {metrics['Outperformance']:.1f}%")
                elif metrics['Outperformance'] > 1:
                    st.info(f"✅ **Good Performance!** Strategy outperforms by {metrics['Outperformance']:.1f}%")
                elif metrics['Outperformance'] > -2:
                    st.warning(f"⚠️ **Similar Performance** to buy-and-hold ({metrics['Outperformance']:+.1f}%)")
                else:
                    st.error(f"📉 **Underperformance** by {abs(metrics['Outperformance']):.1f}%. Consider buy-and-hold.")
                
                # Risk analysis
                if metrics['Max_Drawdown'] < -20:
                    st.error("🚨 **High Risk**: Maximum drawdown exceeds 20%")
                elif metrics['Max_Drawdown'] < -10:
                    st.warning("⚠️ **Moderate Risk**: Drawdown between 10-20%")
                else:
                    st.success("✅ **Low Risk**: Drawdown under 10%")
            
            with col2:
                st.markdown("### 💡 Recommendations")
                
                recommendations = []
                
                if metrics['Win_Rate'] < 40:
                    recommendations.append("🎯 **Low win rate detected** - Consider refining entry criteria or using different timeframes")
                
                if metrics['Total_Fees'] > (initial_capital * 0.02):  # More than 2% of capital
                    recommendations.append("💸 **High transaction costs** - Strategy trades too frequently. Consider lower frequency approaches")
                
                if metrics['Avg_Holding_Days'] < 3:
                    recommendations.append("⏱️ **Very short holding periods** - May be affected by noise. Consider longer timeframes")
                
                if metrics['Profit_Factor'] < 1.5 and metrics['Profit_Factor'] != float('inf'):
                    recommendations.append("⚖️ **Low profit factor** - Average wins not significantly larger than average losses")
                
                if metrics['Sharpe_Ratio'] < 0.5:
                    recommendations.append("📊 **Low risk-adjusted returns** - Consider risk management improvements")
                
                if not recommendations:
                    recommendations.append("✅ **Good overall performance** - Strategy shows promising results")
                    recommendations.append("📈 **Continue monitoring** - Test on different market conditions")
                
                for rec in recommendations:
                    st.write(rec)
            
            # Download comprehensive results
            st.subheader("📥 Export Results")
            
            # Prepare comprehensive results
            results_summary = {
                'Strategy': selected_strategy,
                'Data_Period': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                'Initial_Capital': initial_capital,
                'Transaction_Cost_Pct': transaction_cost,
                'Stop_Loss_Pct': stop_loss_pct,
                'Take_Profit_Pct': take_profit_pct,
                **metrics
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance summary download
                summary_df = pd.DataFrame([results_summary])
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Performance Summary",
                    data=csv_summary,
                    file_name=f"performance_summary_{selected_strategy.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Detailed trades download
                if trades:
                    trades_df = pd.DataFrame(trades)
                    csv_trades = trades_df.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Trade Details",
                        data=csv_trades,
                        file_name=f"trade_details_{selected_strategy.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
    
    else:
        # Landing page with instructions
        st.markdown("""
        ## 🚀 Features
        
        ### 📊 **Universal Data Support**
        - Auto-detects CSV column formats
        - Supports any stock/forex/crypto data
        - Handles various date formats
        - Works with incomplete data (fills missing OHLC)
        
        ### 🎯 **Advanced Strategies**
        - **Enhanced Contrarian**: Smart mean reversion with multiple confirmations
        - **Smart Momentum**: Multi-timeframe trend following
        - **Mean Reversion Pro**: Professional oscillator combinations
        - **Custom parameters** for each strategy
        
        ### 🔴 **Live Market Signals**
        - Current BUY/SELL/HOLD recommendations
        - Signal strength indicators
        - Entry/exit reasoning
        - Stop loss & take profit levels
        
        ### 📈 **Comprehensive Analysis**
        - **Detailed trade breakdown** with P&L per trade
        - **Risk metrics** (drawdown, Sharpe ratio, volatility)
        - **Transaction cost impact** analysis
        - **Visual charts** with trade markers
        - **Performance comparison** vs buy-and-hold
        
        ---
        
        ## 📋 **Supported CSV Formats**
        
        The tool auto-detects these column variations:
        """)
        
        format_examples = {
            'Date': ['Date', 'date', 'Timestamp', 'Time'],
            'Open': ['Open', 'open', 'OPEN', 'O'],
            'High': ['High', 'high', 'HIGH', 'H'], 
            'Low': ['Low', 'low', 'LOW', 'L'],
            'Close': ['Close', 'close', 'Adj Close', 'C'],
            'Volume': ['Volume', 'volume', 'Shares Traded', 'Vol']
        }
        
        for standard, variations in format_examples.items():
            st.write(f"**{standard}:** {', '.join(variations)}")
        
        st.markdown("""
        ---
        
        ## 📁 **Sample Data Format**
        """)
        
        sample_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [100.0, 101.5, 99.8],
            'High': [102.0, 103.0, 101.2],
            'Low': [99.5, 100.8, 98.9],
            'Close': [101.5, 99.8, 100.5],
            'Volume': [1000000, 1200000, 950000]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        st.info("👆 **Upload your CSV file above to get started!** The tool will auto-detect your data format.")

if __name__ == "__main__":
    main()  
        
