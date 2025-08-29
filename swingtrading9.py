import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dynamic Nifty Backtesting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class TechnicalIndicators:
    """Custom technical indicators without external libraries"""
    
    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data, window=20, std_dev=2):
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high, low, close, k_window=14, d_window=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def vwap(high, low, close, volume):
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def atr(high, low, close, window=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(window=window).mean()

class SupportResistanceCalculator:
    """Dynamic Support and Resistance calculation"""
    
    @staticmethod
    def calculate_support_resistance(data, window=20, min_touches=2):
        """
        Calculate dynamic support and resistance levels
        
        Method:
        1. Find local maxima and minima using rolling windows
        2. Cluster nearby levels within tolerance
        3. Count touches for each level
        4. Return strongest levels with minimum touches
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Find local maxima (resistance) and minima (support)
        local_max = high[(high.shift(1) < high) & (high.shift(-1) < high)]
        local_min = low[(low.shift(1) > low) & (low.shift(-1) > low)]
        
        # Tolerance for clustering levels (0.5% of current price)
        tolerance = close.iloc[-1] * 0.005
        
        # Function to cluster levels
        def cluster_levels(levels, tolerance):
            if len(levels) == 0:
                return []
            
            levels_sorted = sorted(levels)
            clusters = []
            current_cluster = [levels_sorted[0]]
            
            for level in levels_sorted[1:]:
                if level - current_cluster[-1] <= tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [level]
            clusters.append(current_cluster)
            
            # Return average of each cluster with touch count
            result = []
            for cluster in clusters:
                avg_level = sum(cluster) / len(cluster)
                touch_count = len(cluster)
                result.append((avg_level, touch_count))
            
            return result
        
        # Cluster resistance and support levels
        resistance_levels = cluster_levels(local_max.tolist(), tolerance)
        support_levels = cluster_levels(local_min.tolist(), tolerance)
        
        # Filter by minimum touches and sort by strength
        strong_resistance = [(level, touches) for level, touches in resistance_levels 
                           if touches >= min_touches]
        strong_support = [(level, touches) for level, touches in support_levels 
                        if touches >= min_touches]
        
        # Sort by number of touches (strength)
        strong_resistance.sort(key=lambda x: x[1], reverse=True)
        strong_support.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'resistance': strong_resistance[:3],  # Top 3 resistance levels
            'support': strong_support[:3],       # Top 3 support levels
            'current_price': close.iloc[-1]
        }

class StrategyBacktester:
    """Dynamic strategy backtesting class"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.ti = TechnicalIndicators()
        self.trades = []
        
    def rsi_mean_reversion_strategy(self, rsi_oversold=30, rsi_overbought=70, 
                                  rsi_period=14, stop_loss_pct=0.6, 
                                  trade_direction='both'):
        """
        RSI Mean Reversion Strategy
        Parameters:
        - rsi_oversold: RSI level to trigger buy (default: 30)
        - rsi_overbought: RSI level to trigger sell (default: 70)
        - rsi_period: RSI calculation period (default: 14)
        - stop_loss_pct: Stop loss percentage (default: 0.6%)
        - trade_direction: 'long', 'short', or 'both'
        """
        
        # Calculate RSI
        rsi = self.ti.rsi(self.data['Close'], rsi_period)
        self.data['RSI'] = rsi
        
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        
        for i in range(1, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_rsi = rsi.iloc[i]
            prev_rsi = rsi.iloc[i-1]
            current_date = self.data.index[i]
            
            # Exit conditions first
            if position is not None:
                stop_loss = entry_price * (1 - stop_loss_pct/100) if position == 'long' else entry_price * (1 + stop_loss_pct/100)
                
                # Stop loss hit
                if (position == 'long' and current_price <= stop_loss) or \
                   (position == 'short' and current_price >= stop_loss):
                    
                    pnl = (current_price - entry_price) if position == 'long' else (entry_price - current_price)
                    pnl_pct = (pnl / entry_price) * 100
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_rsi': self.data['RSI'].loc[entry_date],
                        'exit_rsi': current_rsi,
                        'exit_reason': 'Stop Loss',
                        'holding_period': (current_date - entry_date).days
                    })
                    
                    position = None
                
                # Profit target hit (RSI reversal)
                elif (position == 'long' and current_rsi >= rsi_overbought) or \
                     (position == 'short' and current_rsi <= rsi_oversold):
                    
                    pnl = (current_price - entry_price) if position == 'long' else (entry_price - current_price)
                    pnl_pct = (pnl / entry_price) * 100
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_rsi': self.data['RSI'].loc[entry_date],
                        'exit_rsi': current_rsi,
                        'exit_reason': 'Target Hit',
                        'holding_period': (current_date - entry_date).days
                    })
                    
                    position = None
            
            # Entry conditions
            if position is None:
                # Long entry
                if trade_direction in ['long', 'both'] and current_rsi <= rsi_oversold and prev_rsi > rsi_oversold:
                    position = 'long'
                    entry_price = current_price
                    entry_date = current_date
                
                # Short entry  
                elif trade_direction in ['short', 'both'] and current_rsi >= rsi_overbought and prev_rsi < rsi_overbought:
                    position = 'short'
                    entry_price = current_price
                    entry_date = current_date
        
        return trades
    
    def momentum_breakout_strategy(self, lookback_period=20, volume_threshold=1.5,
                                 stop_loss_pct=0.8, trade_direction='both'):
        """
        Momentum Breakout Strategy
        Parameters:
        - lookback_period: Period for high/low breakout (default: 20)
        - volume_threshold: Volume multiplier for confirmation (default: 1.5x)
        - stop_loss_pct: Stop loss percentage (default: 0.8%)
        - trade_direction: 'long', 'short', or 'both'
        """
        
        # Calculate indicators
        self.data['HighestHigh'] = self.data['High'].rolling(lookback_period).max()
        self.data['LowestLow'] = self.data['Low'].rolling(lookback_period).min()
        self.data['VolumeMA'] = self.data['Volume'].rolling(lookback_period).mean()
        
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        
        for i in range(lookback_period, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_high = self.data['High'].iloc[i]
            current_low = self.data['Low'].iloc[i]
            current_volume = self.data['Volume'].iloc[i]
            current_date = self.data.index[i]
            
            prev_highest_high = self.data['HighestHigh'].iloc[i-1]
            prev_lowest_low = self.data['LowestLow'].iloc[i-1]
            avg_volume = self.data['VolumeMA'].iloc[i]
            
            # Exit conditions
            if position is not None:
                stop_loss = entry_price * (1 - stop_loss_pct/100) if position == 'long' else entry_price * (1 + stop_loss_pct/100)
                
                if (position == 'long' and current_price <= stop_loss) or \
                   (position == 'short' and current_price >= stop_loss):
                    
                    pnl = (current_price - entry_price) if position == 'long' else (entry_price - current_price)
                    pnl_pct = (pnl / entry_price) * 100
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_breakout': prev_highest_high if position == 'long' else prev_lowest_low,
                        'exit_reason': 'Stop Loss',
                        'holding_period': (current_date - entry_date).days,
                        'entry_volume_ratio': self.data['Volume'].loc[entry_date] / self.data['VolumeMA'].loc[entry_date]
                    })
                    
                    position = None
            
            # Entry conditions
            if position is None and current_volume >= avg_volume * volume_threshold:
                # Long entry (breakout above highest high)
                if trade_direction in ['long', 'both'] and current_high > prev_highest_high:
                    position = 'long'
                    entry_price = current_price
                    entry_date = current_date
                
                # Short entry (breakdown below lowest low)
                elif trade_direction in ['short', 'both'] and current_low < prev_lowest_low:
                    position = 'short'
                    entry_price = current_price
                    entry_date = current_date
        
        return trades
    
    def vwap_bounce_strategy(self, vwap_deviation=0.2, volume_threshold=1.2,
                           profit_target=2, stop_loss_pct=0.4, trade_direction='both'):
        """
        VWAP Bounce Strategy
        Parameters:
        - vwap_deviation: Percentage deviation from VWAP for entry (default: 0.2%)
        - volume_threshold: Volume multiplier for confirmation (default: 1.2x)
        - profit_target: Profit target as percentage (default: 2%)
        - stop_loss_pct: Stop loss percentage (default: 0.4%)
        - trade_direction: 'long', 'short', or 'both'
        """
        
        # Calculate VWAP and volume MA
        self.data['VWAP'] = self.ti.vwap(self.data['High'], self.data['Low'], 
                                       self.data['Close'], self.data['Volume'])
        self.data['VolumeMA'] = self.data['Volume'].rolling(20).mean()
        
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        
        for i in range(20, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_vwap = self.data['VWAP'].iloc[i]
            current_volume = self.data['Volume'].iloc[i]
            current_date = self.data.index[i]
            avg_volume = self.data['VolumeMA'].iloc[i]
            
            deviation_pct = abs(current_price - current_vwap) / current_vwap * 100
            
            # Exit conditions
            if position is not None:
                stop_loss = entry_price * (1 - stop_loss_pct/100) if position == 'long' else entry_price * (1 + stop_loss_pct/100)
                profit_target_price = entry_price * (1 + profit_target/100) if position == 'long' else entry_price * (1 - profit_target/100)
                
                # Stop loss or profit target
                if (position == 'long' and (current_price <= stop_loss or current_price >= profit_target_price)) or \
                   (position == 'short' and (current_price >= stop_loss or current_price <= profit_target_price)):
                    
                    pnl = (current_price - entry_price) if position == 'long' else (entry_price - current_price)
                    pnl_pct = (pnl / entry_price) * 100
                    exit_reason = 'Profit Target' if pnl > 0 else 'Stop Loss'
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_vwap': self.data['VWAP'].loc[entry_date],
                        'exit_vwap': current_vwap,
                        'exit_reason': exit_reason,
                        'holding_period': (current_date - entry_date).days,
                        'entry_volume_ratio': self.data['Volume'].loc[entry_date] / self.data['VolumeMA'].loc[entry_date]
                    })
                    
                    position = None
            
            # Entry conditions
            if position is None and current_volume >= avg_volume * volume_threshold and deviation_pct >= vwap_deviation:
                # Long entry (price below VWAP)
                if trade_direction in ['long', 'both'] and current_price < current_vwap:
                    position = 'long'
                    entry_price = current_price
                    entry_date = current_date
                
                # Short entry (price above VWAP)
                elif trade_direction in ['short', 'both'] and current_price > current_vwap:
                    position = 'short'
                    entry_price = current_price
                    entry_date = current_date
        
        return trades
    
    def bollinger_bands_strategy(self, bb_period=20, bb_std=2, rsi_period=14,
                               stop_loss_pct=0.5, trade_direction='both'):
        """
        Bollinger Bands Mean Reversion Strategy
        Parameters:
        - bb_period: Bollinger Bands period (default: 20)
        - bb_std: Standard deviation multiplier (default: 2)
        - rsi_period: RSI period for confirmation (default: 14)
        - stop_loss_pct: Stop loss percentage (default: 0.5%)
        - trade_direction: 'long', 'short', or 'both'
        """
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(self.data['Close'], bb_period, bb_std)
        rsi = self.ti.rsi(self.data['Close'], rsi_period)
        
        self.data['BB_Upper'] = bb_upper
        self.data['BB_Middle'] = bb_middle  
        self.data['BB_Lower'] = bb_lower
        self.data['RSI'] = rsi
        
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        
        for i in range(max(bb_period, rsi_period), len(self.data)):
            current_price = self.data['Close'].iloc[i]
            current_date = self.data.index[i]
            current_rsi = rsi.iloc[i]
            
            bb_upper_val = bb_upper.iloc[i]
            bb_lower_val = bb_lower.iloc[i]
            bb_middle_val = bb_middle.iloc[i]
            
            # Exit conditions
            if position is not None:
                stop_loss = entry_price * (1 - stop_loss_pct/100) if position == 'long' else entry_price * (1 + stop_loss_pct/100)
                
                # Stop loss or mean reversion to middle band
                if (position == 'long' and (current_price <= stop_loss or current_price >= bb_middle_val)) or \
                   (position == 'short' and (current_price >= stop_loss or current_price <= bb_middle_val)):
                    
                    pnl = (current_price - entry_price) if position == 'long' else (entry_price - current_price)
                    pnl_pct = (pnl / entry_price) * 100
                    exit_reason = 'Mean Reversion' if pnl > 0 else 'Stop Loss'
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_rsi': self.data['RSI'].loc[entry_date],
                        'exit_rsi': current_rsi,
                        'exit_reason': exit_reason,
                        'holding_period': (current_date - entry_date).days,
                        'bb_squeeze': (bb_upper_val - bb_lower_val) / bb_middle_val * 100
                    })
                    
                    position = None
            
            # Entry conditions
            if position is None:
                # Long entry (price touches lower band + RSI oversold)
                if trade_direction in ['long', 'both'] and current_price <= bb_lower_val and current_rsi <= 30:
                    position = 'long'
                    entry_price = current_price
                    entry_date = current_date
                
                # Short entry (price touches upper band + RSI overbought)
                elif trade_direction in ['short', 'both'] and current_price >= bb_upper_val and current_rsi >= 70:
                    position = 'short'
                    entry_price = current_price
                    entry_date = current_date
        
        return trades

def calculate_performance_metrics(trades):
    """Calculate comprehensive performance metrics"""
    if not trades:
        return None
    
    df_trades = pd.DataFrame(trades)
    
    # Basic metrics
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
    losing_trades = len(df_trades[df_trades['pnl_pct'] <= 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # P&L metrics
    total_return_pct = df_trades['pnl_pct'].sum()
    avg_win_pct = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
    avg_loss_pct = df_trades[df_trades['pnl_pct'] <= 0]['pnl_pct'].mean() if losing_trades > 0 else 0
    
    # Risk metrics
    max_win_pct = df_trades['pnl_pct'].max()
    max_loss_pct = df_trades['pnl_pct'].min()
    profit_factor = abs(df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].sum() / 
                       df_trades[df_trades['pnl_pct'] <= 0]['pnl_pct'].sum()) if losing_trades > 0 else float('inf')
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + df_trades['pnl_pct'] / 100).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max * 100
    max_drawdown_pct = drawdown.min()
    
    # Sharpe ratio (assuming 252 trading days)
    returns_std = df_trades['pnl_pct'].std()
    if returns_std != 0:
        sharpe_ratio = (df_trades['pnl_pct'].mean() / returns_std) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Other metrics
    avg_holding_period = df_trades['holding_period'].mean()
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_return_pct': total_return_pct,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'max_win_pct': max_win_pct,
        'max_loss_pct': max_loss_pct,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'avg_holding_period': avg_holding_period
    }

@st.cache_data
def load_nifty_data(period='1y'):
    """Load Nifty 50 data"""
    try:
        nifty = yf.Ticker("^NSEI")
        data = nifty.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.markdown("<h1 class='main-header'>🚀 Dynamic Nifty Backtesting System</h1>", unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("📊 Strategy Configuration")
    
    # Data loading
    data_period = st.sidebar.selectbox("Data Period", ['6mo', '1y', '2y', '5y'], index=1)
    
    with st.spinner('Loading Nifty data...'):
        data = load_nifty_data(data_period)
    
    if data is None or data.empty:
        st.error("Failed to load data. Please try again.")
        return
    
    st.success(f"✅ Loaded {len(data)} days of Nifty data")
    
    # Strategy selection
    strategy_name = st.sidebar.selectbox(
        "Strategy", 
        ["RSI Mean Reversion", "Momentum Breakout", "VWAP Bounce", "Bollinger Bands"]
    )
    
    # Trade direction
    trade_direction = st.sidebar.selectbox(
        "Trade Direction", 
        ["both", "long", "short"]
    )
    
    # Expected returns input
    expected_returns = st.sidebar.number_input(
        "Expected Returns (%)", 
        min_value=0.0, 
        max_value=1000.0, 
        value=120.0, 
        step=10.0
    )
    
    # Strategy-specific parameters
    st.sidebar.subheader("📋 Strategy Parameters")
    
    if strategy_name == "RSI Mean Reversion":
        rsi_period = st.sidebar.slider("RSI Period", 5, 50, 14)
        rsi_oversold = st.sidebar.slider("RSI Oversold Level", 10, 40, 30)
        rsi_overbought = st.sidebar.slider("RSI Overbought Level", 60, 90, 70)
        stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.1, 2.0, 0.6, 0.1)
        
        params = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'stop_loss_pct': stop_loss_pct,
            'trade_direction': trade_direction
        }
        
    elif strategy_name == "Momentum Breakout":
        lookback_period = st.sidebar.slider("Breakout Lookback Period", 5, 50, 20)
        volume_threshold = st.sidebar.slider("Volume Threshold (x)", 1.0, 3.0, 1.5, 0.1)
        stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.1, 2.0, 0.8, 0.1)
        
        params = {
            'lookback_period': lookback_period,
            'volume_threshold': volume_threshold,
            'stop_loss_pct': stop_loss_pct,
            'trade_direction': trade_direction
        }
        
    elif strategy_name == "VWAP Bounce":
        vwap_deviation = st.sidebar.slider("VWAP Deviation (%)", 0.1, 1.0, 0.2, 0.05)
        volume_threshold = st.sidebar.slider("Volume Threshold (x)", 1.0, 2.5, 1.2, 0.1)
        profit_target = st.sidebar.slider("Profit Target (%)", 0.5, 5.0, 2.0, 0.1)
        stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.1, 1.0, 0.4, 0.1)
        
        params = {
            'vwap_deviation': vwap_deviation,
            'volume_threshold': volume_threshold,
            'profit_target': profit_target,
            'stop_loss_pct': stop_loss_pct,
            'trade_direction': trade_direction
        }
        
    elif strategy_name == "Bollinger Bands":
        bb_period = st.sidebar.slider("BB Period", 10, 50, 20)
        bb_std = st.sidebar.slider("BB Standard Deviation", 1.0, 3.0, 2.0, 0.1)
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
        stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.1, 1.0, 0.5, 0.1)
        
        params = {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'rsi_period': rsi_period,
            'stop_loss_pct': stop_loss_pct,
            'trade_direction': trade_direction
        }
    
    # Run backtest button
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        
        with st.spinner('Running backtest...'):
            backtester = StrategyBacktester(data)
            
            # Execute selected strategy
            if strategy_name == "RSI Mean Reversion":
                trades = backtester.rsi_mean_reversion_strategy(**params)
            elif strategy_name == "Momentum Breakout":
                trades = backtester.momentum_breakout_strategy(**params)
            elif strategy_name == "VWAP Bounce":
                trades = backtester.vwap_bounce_strategy(**params)
            elif strategy_name == "Bollinger Bands":
                trades = backtester.bollinger_bands_strategy(**params)
            
            # Calculate performance metrics
            metrics = calculate_performance_metrics(trades)
            
            if metrics is None:
                st.warning("⚠️ No trades generated with current parameters. Try adjusting the settings.")
                return
            
            # Check if strategy meets expected returns
            meets_expectation = metrics['total_return_pct'] >= expected_returns
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card {"success-metric" if meets_expectation else "error-metric"}'>
                    <h3>📈 Total Returns</h3>
                    <h2>{metrics['total_return_pct']:.2f}%</h2>
                    <p>Target: {expected_returns}%</p>
                    <p>{'✅ Target Met!' if meets_expectation else '❌ Below Target'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>🎯 Win Rate</h3>
                    <h2>{metrics['win_rate']:.1f}%</h2>
                    <p>{metrics['winning_trades']}/{metrics['total_trades']} trades</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>⚡ Profit Factor</h3>
                    <h2>{metrics['profit_factor']:.2f}</h2>
                    <p>Risk-Reward Ratio</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.subheader("📊 Detailed Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", metrics['total_trades'])
                st.metric("Average Win", f"{metrics['avg_win_pct']:.2f}%")
                st.metric("Max Win", f"{metrics['max_win_pct']:.2f}%")
            
            with col2:
                st.metric("Winning Trades", metrics['winning_trades'])
                st.metric("Average Loss", f"{metrics['avg_loss_pct']:.2f}%")
                st.metric("Max Loss", f"{metrics['max_loss_pct']:.2f}%")
            
            with col3:
                st.metric("Losing Trades", metrics['losing_trades'])
                st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            
            with col4:
                st.metric("Avg Holding Period", f"{metrics['avg_holding_period']:.1f} days")
                st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                st.metric("Return/DD Ratio", f"{abs(metrics['total_return_pct']/metrics['max_drawdown_pct']):.2f}")
            
            # Strategy Parameters Used
            st.subheader("🔧 Strategy Parameters Used")
            param_cols = st.columns(len(params))
            for i, (param, value) in enumerate(params.items()):
                with param_cols[i % len(param_cols)]:
                    st.metric(param.replace('_', ' ').title(), value)
            
            # Trades table
            if trades:
                st.subheader("📋 Trade Details")
                
                df_trades = pd.DataFrame(trades)
                
                # Format the dataframe for display
                display_df = df_trades.copy()
                display_df['Entry Date'] = pd.to_datetime(display_df['entry_date']).dt.strftime('%Y-%m-%d')
                display_df['Exit Date'] = pd.to_datetime(display_df['exit_date']).dt.strftime('%Y-%m-%d')
                display_df['Entry Price'] = display_df['entry_price'].round(2)
                display_df['Exit Price'] = display_df['exit_price'].round(2)
                display_df['P&L %'] = display_df['pnl_pct'].round(2)
                display_df['Position'] = display_df['position'].str.upper()
                display_df['Holding Days'] = display_df['holding_period']
                
                # Select columns based on strategy
                base_cols = ['Entry Date', 'Exit Date', 'Position', 'Entry Price', 'Exit Price', 'P&L %', 'Holding Days']
                
                if strategy_name == "RSI Mean Reversion":
                    display_df['Entry RSI'] = display_df['entry_rsi'].round(1)
                    display_df['Exit RSI'] = display_df['exit_rsi'].round(1)
                    cols_to_show = base_cols + ['Entry RSI', 'Exit RSI', 'exit_reason']
                elif strategy_name == "Momentum Breakout":
                    display_df['Breakout Level'] = display_df['entry_breakout'].round(2)
                    display_df['Volume Ratio'] = display_df['entry_volume_ratio'].round(2)
                    cols_to_show = base_cols + ['Breakout Level', 'Volume Ratio', 'exit_reason']
                elif strategy_name == "VWAP Bounce":
                    display_df['Entry VWAP'] = display_df['entry_vwap'].round(2)
                    display_df['Volume Ratio'] = display_df['entry_volume_ratio'].round(2)
                    cols_to_show = base_cols + ['Entry VWAP', 'Volume Ratio', 'exit_reason']
                elif strategy_name == "Bollinger Bands":
                    display_df['Entry RSI'] = display_df['entry_rsi'].round(1)
                    display_df['BB Squeeze'] = display_df['bb_squeeze'].round(2)
                    cols_to_show = base_cols + ['Entry RSI', 'BB Squeeze', 'exit_reason']
                
                # Color code profitable trades
                def highlight_trades(row):
                    if row['P&L %'] > 0:
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)
                
                styled_df = display_df[cols_to_show].style.apply(highlight_trades, axis=1)
                st.dataframe(styled_df, use_container_width=True)
            
            # Charts
            st.subheader("📈 Performance Visualization")
            
            if trades:
                # Create equity curve
                df_trades = pd.DataFrame(trades)
                df_trades['cumulative_return'] = (1 + df_trades['pnl_pct']/100).cumprod() - 1
                
                # Equity curve chart
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=list(range(len(df_trades))),
                    y=df_trades['cumulative_return'] * 100,
                    mode='lines',
                    name='Cumulative Returns',
                    line=dict(color='green', width=2)
                ))
                
                fig_equity.update_layout(
                    title='Equity Curve',
                    xaxis_title='Trade Number',
                    yaxis_title='Cumulative Return (%)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Trade distribution
                fig_dist = px.histogram(
                    df_trades, 
                    x='pnl_pct', 
                    nbins=20, 
                    title='Trade Returns Distribution',
                    color_discrete_sequence=['skyblue']
                )
                fig_dist.update_layout(xaxis_title='Return (%)', yaxis_title='Frequency')
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # Support and Resistance Analysis
    st.subheader("🎯 Support & Resistance Analysis")
    
    if st.button("Calculate S&R Levels"):
        with st.spinner('Calculating support and resistance levels...'):
            sr_calc = SupportResistanceCalculator()
            levels = sr_calc.calculate_support_resistance(data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔺 Resistance Levels:**")
                for i, (level, touches) in enumerate(levels['resistance'], 1):
                    distance = ((level - levels['current_price']) / levels['current_price']) * 100
                    st.write(f"{i}. ₹{level:.2f} ({touches} touches, {distance:+.2f}%)")
            
            with col2:
                st.write("**🔻 Support Levels:**")
                for i, (level, touches) in enumerate(levels['support'], 1):
                    distance = ((level - levels['current_price']) / levels['current_price']) * 100
                    st.write(f"{i}. ₹{level:.2f} ({touches} touches, {distance:+.2f}%)")
            
            st.info(f"Current Price: ₹{levels['current_price']:.2f}")
            
            # Plot S&R levels
            fig_sr = go.Figure()
            
            # Price data
            fig_sr.add_trace(go.Candlestick(
                x=data.index[-100:],  # Last 100 days
                open=data['Open'][-100:],
                high=data['High'][-100:],
                low=data['Low'][-100:],
                close=data['Close'][-100:],
                name='Nifty 50'
            ))
            
            # Resistance levels
            for level, touches in levels['resistance']:
                fig_sr.add_hline(
                    y=level,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"R: {level:.0f} ({touches})",
                    annotation_position="top right"
                )
            
            # Support levels
            for level, touches in levels['support']:
                fig_sr.add_hline(
                    y=level,
                    line_dash="dash", 
                    line_color="green",
                    annotation_text=f"S: {level:.0f} ({touches})",
                    annotation_position="bottom right"
                )
            
            fig_sr.update_layout(
                title='Nifty 50 with Support & Resistance Levels',
                yaxis_title='Price (₹)',
                xaxis_title='Date'
            )
            
            st.plotly_chart(fig_sr, use_container_width=True)
    
    # Live Trading Recommendations
    st.subheader("🔴 Live Trading Recommendations")
    
    if st.button("Generate Live Signals"):
        with st.spinner('Analyzing current market conditions...'):
            # Get latest data
            latest_data = data.tail(50)  # Last 50 days for analysis
            
            # Initialize indicators
            ti = TechnicalIndicators()
            
            # Calculate current indicators
            current_price = latest_data['Close'].iloc[-1]
            current_rsi = ti.rsi(latest_data['Close']).iloc[-1]
            
            # VWAP
            current_vwap = ti.vwap(latest_data['High'], latest_data['Low'], 
                                 latest_data['Close'], latest_data['Volume']).iloc[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ti.bollinger_bands(latest_data['Close'])
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            current_bb_middle = bb_middle.iloc[-1]
            
            # MACD
            macd_line, signal_line, histogram = ti.macd(latest_data['Close'])
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            
            # 20-day high/low
            high_20 = latest_data['High'].rolling(20).max().iloc[-1]
            low_20 = latest_data['Low'].rolling(20).min().iloc[-1]
            
            # Volume analysis
            avg_volume = latest_data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = latest_data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📊 Current Market Data")
                st.metric("Nifty Price", f"₹{current_price:.2f}")
                st.metric("RSI (14)", f"{current_rsi:.1f}")
                st.metric("VWAP", f"₹{current_vwap:.2f}")
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
            
            with col2:
                st.markdown("### 🎯 Key Levels")
                st.metric("20-Day High", f"₹{high_20:.2f}")
                st.metric("20-Day Low", f"₹{low_20:.2f}")
                st.metric("BB Upper", f"₹{current_bb_upper:.2f}")
                st.metric("BB Lower", f"₹{current_bb_lower:.2f}")
            
            with col3:
                st.markdown("### 📈 Signals")
                
                signals = []
                
                # RSI signals
                if current_rsi <= 30:
                    signals.append("🟢 RSI Oversold - BUY Signal")
                elif current_rsi >= 70:
                    signals.append("🔴 RSI Overbought - SELL Signal")
                
                # VWAP signals
                if current_price < current_vwap * 0.998:  # 0.2% below VWAP
                    signals.append("🟢 Below VWAP - BUY Signal")
                elif current_price > current_vwap * 1.002:  # 0.2% above VWAP
                    signals.append("🔴 Above VWAP - SELL Signal")
                
                # Bollinger Bands signals
                if current_price <= current_bb_lower:
                    signals.append("🟢 BB Lower Touch - BUY Signal")
                elif current_price >= current_bb_upper:
                    signals.append("🔴 BB Upper Touch - SELL Signal")
                
                # Breakout signals
                if current_price > high_20 and volume_ratio > 1.5:
                    signals.append("🚀 Breakout HIGH - BUY Signal")
                elif current_price < low_20 and volume_ratio > 1.5:
                    signals.append("💥 Breakdown LOW - SELL Signal")
                
                # MACD signals
                if current_macd > current_signal:
                    signals.append("📈 MACD Bullish")
                else:
                    signals.append("📉 MACD Bearish")
                
                if signals:
                    for signal in signals:
                        st.write(signal)
                else:
                    st.write("⏳ No clear signals at the moment")
            
            # Recommended trades based on signals
            st.markdown("### 🎯 Recommended Trades")
            
            recommendations = []
            
            # Generate specific trade recommendations
            if current_rsi <= 30 and current_price < current_vwap:
                entry_price = current_price
                stop_loss = entry_price * 0.994  # 0.6% SL
                target = current_bb_middle if current_price < current_bb_lower else entry_price * 1.02
                
                recommendations.append({
                    'Direction': 'BUY',
                    'Entry': f"₹{entry_price:.2f}",
                    'Stop Loss': f"₹{stop_loss:.2f}",
                    'Target': f"₹{target:.2f}",
                    'Risk': f"{((entry_price - stop_loss)/entry_price)*100:.1f}%",
                    'Reward': f"{((target - entry_price)/entry_price)*100:.1f}%",
                    'Logic': 'RSI Oversold + Below VWAP'
                })
            
            elif current_rsi >= 70 and current_price > current_vwap:
                entry_price = current_price
                stop_loss = entry_price * 1.006  # 0.6% SL
                target = current_bb_middle if current_price > current_bb_upper else entry_price * 0.98
                
                recommendations.append({
                    'Direction': 'SELL',
                    'Entry': f"₹{entry_price:.2f}",
                    'Stop Loss': f"₹{stop_loss:.2f}",
                    'Target': f"₹{target:.2f}",
                    'Risk': f"{((stop_loss - entry_price)/entry_price)*100:.1f}%",
                    'Reward': f"{((entry_price - target)/entry_price)*100:.1f}%",
                    'Logic': 'RSI Overbought + Above VWAP'
                })
            
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                st.dataframe(rec_df, use_container_width=True)
            else:
                st.info("⏳ No high-probability trade setups identified at current levels.")
    
    # Strategy Comparison
    st.subheader("⚖️ Strategy Comparison")
    
    if st.button("Compare All Strategies"):
        with st.spinner('Running comprehensive strategy comparison...'):
            comparison_results = []
            
            strategies = [
                ("RSI Mean Reversion", 'rsi_mean_reversion_strategy', {
                    'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 
                    'stop_loss_pct': 0.6, 'trade_direction': 'both'
                }),
                ("Momentum Breakout", 'momentum_breakout_strategy', {
                    'lookback_period': 20, 'volume_threshold': 1.5, 
                    'stop_loss_pct': 0.8, 'trade_direction': 'both'
                }),
                ("VWAP Bounce", 'vwap_bounce_strategy', {
                    'vwap_deviation': 0.2, 'volume_threshold': 1.2, 
                    'profit_target': 2, 'stop_loss_pct': 0.4, 'trade_direction': 'both'
                }),
                ("Bollinger Bands", 'bollinger_bands_strategy', {
                    'bb_period': 20, 'bb_std': 2, 'rsi_period': 14, 
                    'stop_loss_pct': 0.5, 'trade_direction': 'both'
                })
            ]
            
            for strategy_name, method_name, params in strategies:
                backtester = StrategyBacktester(data)
                method = getattr(backtester, method_name)
                trades = method(**params)
                metrics = calculate_performance_metrics(trades)
                
                if metrics:
                    comparison_results.append({
                        'Strategy': strategy_name,
                        'Total Return (%)': f"{metrics['total_return_pct']:.2f}",
                        'Win Rate (%)': f"{metrics['win_rate']:.1f}",
                        'Total Trades': metrics['total_trades'],
                        'Profit Factor': f"{metrics['profit_factor']:.2f}",
                        'Max Drawdown (%)': f"{metrics['max_drawdown_pct']:.2f}",
                        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                        'Meets Target': '✅' if metrics['total_return_pct'] >= expected_returns else '❌'
                    })
            
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                
                # Highlight best performing strategy
                def highlight_best(row):
                    if float(row['Total Return (%)']) == max([float(r['Total Return (%)']) for r in comparison_results]):
                        return ['background-color: #90EE90'] * len(row)
                    return [''] * len(row)
                
                styled_comparison = comparison_df.style.apply(highlight_best, axis=1)
                st.dataframe(styled_comparison, use_container_width=True)
                
                # Find best strategy
                best_strategy = max(comparison_results, key=lambda x: float(x['Total Return (%)']))
                
                if float(best_strategy['Total Return (%)']) >= expected_returns:
                    st.success(f"🎉 **{best_strategy['Strategy']}** meets your target of {expected_returns}% with {best_strategy['Total Return (%)']}% returns!")
                else:
                    st.warning(f"⚠️ Best strategy is **{best_strategy['Strategy']}** with {best_strategy['Total Return (%)']}% returns, but it's below your target of {expected_returns}%")
    
    # Footer with methodology
    st.markdown("---")
    st.subheader("📚 Methodology & Calculations")
    
    with st.expander("Support & Resistance Calculation"):
        st.markdown("""
        **Support & Resistance Calculation Method:**
        
        1. **Local Extremes Identification:**
           - Find local maxima: `high[i] > high[i-1] AND high[i] > high[i+1]`
           - Find local minima: `low[i] < low[i-1] AND low[i] < low[i+1]`
        
        2. **Level Clustering:**
           - Group nearby levels within 0.5% of current price
           - Tolerance = Current Price × 0.005
        
        3. **Strength Calculation:**
           - Count number of touches for each level
           - Minimum 2 touches required for valid S&R
           - Higher touch count = stronger level
        
        4. **Final Selection:**
           - Top 3 support and resistance levels
           - Sorted by number of touches (strength)
        """)
    
    with st.expander("Performance Metrics Definitions"):
        st.markdown("""
        **Key Performance Metrics:**
        
        - **Total Return (%)**: Cumulative percentage return from all trades
        - **Win Rate (%)**: Percentage of profitable trades
        - **Profit Factor**: Gross profit ÷ Gross loss
        - **Maximum Drawdown (%)**: Largest peak-to-trough decline
        - **Sharpe Ratio**: Risk-adjusted return metric
        - **Average Holding Period**: Mean days per trade
        - **R-Multiple**: Total return divided by average risk per trade
        """)
    
    with st.expander("Strategy Parameters"):
        st.markdown("""
        **RSI Mean Reversion:**
        - RSI Period: 14 (default)
        - Oversold Level: 30
        - Overbought Level: 70
        - Stop Loss: 0.6% of entry price
        
        **Momentum Breakout:**
        - Lookback Period: 20 days for high/low
        - Volume Threshold: 1.5x average volume
        - Stop Loss: 0.8% of entry price
        
        **VWAP Bounce:**
        - VWAP Deviation: 0.2% from VWAP
        - Volume Threshold: 1.2x average volume
        - Profit Target: 2% from entry
        - Stop Loss: 0.4% of entry price
        
        **Bollinger Bands:**
        - BB Period: 20 days
        - Standard Deviation: 2.0
        - RSI Confirmation: 14 period
        - Stop Loss: 0.5% of entry price
        """)

if __name__ == "__main__":
    main()
