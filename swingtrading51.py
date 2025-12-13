import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
from typing import Dict, List, Tuple, Optional
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Multi-Strategy Live Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .profit { color: #00cc00; font-weight: bold; }
    .loss { color: #ff0000; font-weight: bold; }
    .status-running { color: #00cc00; font-weight: bold; font-size: 1.2rem; }
    .status-stopped { color: #ff6b6b; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'current_position' not in st.session_state:
    st.session_state.current_position = None
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

# ============================================================================
# BASE STRATEGY CLASS
# ============================================================================

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.ist_tz = pytz.timezone('Asia/Kolkata')
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """Generate trading signals: (bullish, bearish, signal_data)"""
        return False, False, {}

# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

class EMASMACrossoverStrategy(BaseStrategy):
    """EMA/SMA Crossover Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        fast_type = self.params['fast_type']
        slow_type = self.params['slow_type']
        
        if fast_type == 'EMA':
            data['fast_ind'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
        else:
            data['fast_ind'] = data['Close'].rolling(window=fast_period).mean()
        
        if slow_type == 'EMA':
            data['slow_ind'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
        else:
            data['slow_ind'] = data['Close'].rolling(window=slow_period).mean()
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        fast = data['fast_ind']
        slow = data['slow_ind']
        
        bullish = (fast.iloc[-2] <= slow.iloc[-2]) and (fast.iloc[-1] > slow.iloc[-1])
        bearish = (fast.iloc[-2] >= slow.iloc[-2]) and (fast.iloc[-1] < slow.iloc[-1])
        
        signal_data = {
            'fast_value': float(fast.iloc[-1]),
            'slow_value': float(slow.iloc[-1]),
            'type': 'EMA/SMA Crossover'
        }
        
        return bullish, bearish, signal_data


class ZScoreStrategy(BaseStrategy):
    """Z-Score Mean Reversion Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        
        data['price_ma'] = data['Close'].rolling(window=window).mean()
        data['price_std'] = data['Close'].rolling(window=window).std()
        data['z_score'] = (data['Close'] - data['price_ma']) / data['price_std']
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        z_score = data['z_score'].iloc[-1]
        entry_threshold = self.params.get('entry_threshold', 2.0)
        
        bullish = z_score < -entry_threshold
        bearish = z_score > entry_threshold
        
        signal_data = {
            'z_score': float(z_score),
            'mean': float(data['price_ma'].iloc[-1]),
            'type': 'Z-Score Mean Reversion'
        }
        
        return bullish, bearish, signal_data


class RSIStrategy(BaseStrategy):
    """RSI-based Strategy"""
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['rsi'] = self.calculate_rsi(data['Close'])
        data['price_ma'] = data['Close'].rolling(window=20).mean()
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        rsi = data['rsi'].iloc[-1]
        rsi_prev = data['rsi'].iloc[-2]
        
        # RSI oversold reversal
        bullish = rsi < 30 and rsi > rsi_prev
        
        # RSI overbought reversal
        bearish = rsi > 70 and rsi < rsi_prev
        
        signal_data = {
            'rsi': float(rsi),
            'type': 'RSI Strategy'
        }
        
        return bullish, bearish, signal_data


class MACDStrategy(BaseStrategy):
    """MACD Crossover Strategy"""
    
    def calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        macd, signal, hist = self.calculate_macd(data['Close'])
        data['macd'] = macd
        data['signal'] = signal
        data['histogram'] = hist
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        macd_curr = data['macd'].iloc[-1]
        signal_curr = data['signal'].iloc[-1]
        macd_prev = data['macd'].iloc[-2]
        signal_prev = data['signal'].iloc[-2]
        
        bullish = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
        bearish = (macd_prev >= signal_prev) and (macd_curr < signal_curr)
        
        signal_data = {
            'macd': float(macd_curr),
            'signal': float(signal_curr),
            'type': 'MACD Crossover'
        }
        
        return bullish, bearish, signal_data


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get('period', 20)
        std_dev = self.params.get('std_dev', 2)
        
        data['bb_middle'] = data['Close'].rolling(window=period).mean()
        data['bb_std'] = data['Close'].rolling(window=period).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * std_dev)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * std_dev)
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        close = data['Close'].iloc[-1]
        close_prev = data['Close'].iloc[-2]
        upper = data['bb_upper'].iloc[-1]
        lower = data['bb_lower'].iloc[-1]
        lower_prev = data['bb_lower'].iloc[-2]
        upper_prev = data['bb_upper'].iloc[-2]
        
        # Price bounces from lower band
        bullish = close_prev <= lower_prev and close > lower
        
        # Price bounces from upper band
        bearish = close_prev >= upper_prev and close < upper
        
        signal_data = {
            'bb_upper': float(upper),
            'bb_lower': float(lower),
            'bb_middle': float(data['bb_middle'].iloc[-1]),
            'type': 'Bollinger Bands'
        }
        
        return bullish, bearish, signal_data


class BreakoutStrategy(BaseStrategy):
    """Breakout Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        
        data['high_band'] = data['High'].rolling(window=window).max()
        data['low_band'] = data['Low'].rolling(window=window).min()
        data['volume_ma'] = data['Volume'].rolling(window=window).mean()
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        close = data['Close'].iloc[-1]
        close_prev = data['Close'].iloc[-2]
        high_band = data['high_band'].iloc[-2]  # Previous high
        low_band = data['low_band'].iloc[-2]    # Previous low
        volume = data['Volume'].iloc[-1]
        volume_ma = data['volume_ma'].iloc[-1]
        
        # Bullish breakout with volume
        bullish = close > high_band and close_prev <= high_band and volume > volume_ma * 1.5
        
        # Bearish breakdown with volume
        bearish = close < low_band and close_prev >= low_band and volume > volume_ma * 1.5
        
        signal_data = {
            'high_band': float(high_band),
            'low_band': float(low_band),
            'volume_ratio': float(volume / volume_ma),
            'type': 'Breakout'
        }
        
        return bullish, bearish, signal_data


class VolumeStrategy(BaseStrategy):
    """Volume-based Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        
        data['volume_ma'] = data['Volume'].rolling(window=window).mean()
        data['volume_std'] = data['Volume'].rolling(window=window).std()
        data['price_change'] = data['Close'].pct_change()
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        volume = data['Volume'].iloc[-1]
        volume_ma = data['volume_ma'].iloc[-1]
        volume_std = data['volume_std'].iloc[-1]
        price_change = data['price_change'].iloc[-1]
        
        # Volume spike threshold
        volume_spike = volume > (volume_ma + 2 * volume_std)
        
        # Bullish: High volume + price increase
        bullish = volume_spike and price_change > 0.01
        
        # Bearish: High volume + price decrease
        bearish = volume_spike and price_change < -0.01
        
        signal_data = {
            'volume_ratio': float(volume / volume_ma),
            'price_change': float(price_change * 100),
            'type': 'Volume Spike'
        }
        
        return bullish, bearish, signal_data


class SupportResistanceStrategy(BaseStrategy):
    """Support/Resistance Strategy"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 20)
        
        data['resistance'] = data['High'].rolling(window=window).max()
        data['support'] = data['Low'].rolling(window=window).min()
        data['middle'] = (data['resistance'] + data['support']) / 2
        
        return data
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        data = self.calculate_indicators(data)
        
        if len(data) < 2:
            return False, False, {}
        
        close = data['Close'].iloc[-1]
        support = data['support'].iloc[-1]
        resistance = data['resistance'].iloc[-1]
        tolerance = self.params.get('tolerance', 0.02)
        
        # Near support (within tolerance)
        near_support = abs(close - support) / support < tolerance
        bullish = near_support and close > data['Close'].iloc[-2]
        
        # Near resistance (within tolerance)
        near_resistance = abs(close - resistance) / resistance < tolerance
        bearish = near_resistance and close < data['Close'].iloc[-2]
        
        signal_data = {
            'support': float(support),
            'resistance': float(resistance),
            'distance_to_support': float((close - support) / support * 100),
            'type': 'Support/Resistance'
        }
        
        return bullish, bearish, signal_data

# ============================================================================
# TRADING SYSTEM
# ============================================================================

class TradingSystem:
    def __init__(self):
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.strategy = None
    
    def set_strategy(self, strategy: BaseStrategy):
        self.strategy = strategy
    
    def fetch_data(self, ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        try:
            time.sleep(2)
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert(self.ist_tz)
            
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_sl_target(self, entry_price: float, position_type: str,
                           sl_config: Dict, target_config: Dict) -> Tuple[Optional[float], Optional[float]]:
        # Stop Loss
        if sl_config['type'] == 'Custom Points':
            sl_points = sl_config['value']
            sl_price = entry_price - sl_points if position_type == 'LONG' else entry_price + sl_points
        elif sl_config['type'] == 'Trail SL':
            sl_points = sl_config['value']
            sl_price = entry_price - sl_points if position_type == 'LONG' else entry_price + sl_points
        else:
            sl_price = None
        
        # Target
        if target_config['type'] == 'Custom Points':
            target_points = target_config['value']
            target_price = entry_price + target_points if position_type == 'LONG' else entry_price - target_points
        else:
            target_price = None
        
        return sl_price, target_price
    
    def update_trailing_sl(self, position: Dict, current_price: float, sl_config: Dict) -> Optional[float]:
        if sl_config['type'] != 'Trail SL':
            return position['sl_price']
        
        trail_points = sl_config['value']
        position_type = position['type']
        current_sl = position['sl_price']
        
        if position_type == 'LONG':
            new_sl = current_price - trail_points
            return max(current_sl, new_sl) if current_sl else new_sl
        else:
            new_sl = current_price + trail_points
            return min(current_sl, new_sl) if current_sl else new_sl
    
    def check_exit_conditions(self, position: Dict, current_price: float,
                             data: pd.DataFrame, sl_config: Dict, 
                             target_config: Dict) -> Tuple[bool, str]:
        # Check SL
        if position['sl_price'] is not None:
            if position['type'] == 'LONG' and current_price <= position['sl_price']:
                return True, 'Stop Loss Hit'
            elif position['type'] == 'SHORT' and current_price >= position['sl_price']:
                return True, 'Stop Loss Hit'
        
        # Check Target
        if position['target_price'] is not None:
            if position['type'] == 'LONG' and current_price >= position['target_price']:
                return True, 'Target Achieved'
            elif position['type'] == 'SHORT' and current_price <= position['target_price']:
                return True, 'Target Achieved'
        
        # Check signal-based exit
        if sl_config['type'] == 'Signal Based' or target_config['type'] == 'Signal Based':
            bullish_signal, bearish_signal, _ = self.strategy.generate_signal(data)
            
            if position['type'] == 'LONG' and bearish_signal:
                return True, 'Bearish Signal Exit'
            elif position['type'] == 'SHORT' and bullish_signal:
                return True, 'Bullish Signal Exit'
        
        return False, ''
    
    def analyze_trade_performance(self, trade: Dict) -> str:
        pnl = trade['pnl']
        pnl_pct = trade['pnl_percent']
        duration = trade['duration']
        exit_reason = trade['exit_reason']
        strategy_type = trade.get('strategy', 'Unknown')
        
        analysis = []
        
        if pnl > 0:
            analysis.append(f"✅ **Profitable Trade**: +{pnl_pct:.2f}% ({pnl:.2f} points)")
            analysis.append(f"📊 **Strategy**: {strategy_type}")
            if exit_reason == 'Target Achieved':
                analysis.append("🎯 **Perfect Exit**: Target hit")
            if pnl_pct > 1.5:
                analysis.append("💪 **Excellent**: Strong profit capture")
        else:
            analysis.append(f"❌ **Loss Trade**: {pnl_pct:.2f}% ({pnl:.2f} points)")
            analysis.append(f"📊 **Strategy**: {strategy_type}")
            if exit_reason == 'Stop Loss Hit':
                analysis.append("🛡️ **Protected**: SL prevented larger loss")
        
        if duration < 300:
            analysis.append("⚡ **Quick Trade**: < 5 minutes")
        elif duration > 3600:
            analysis.append("⏰ **Long Hold**: > 1 hour")
        
        analysis.append("\n**Recommendations:**")
        if pnl > 0:
            analysis.append("• Maintain strategy discipline")
            analysis.append("• Consider position sizing")
        else:
            analysis.append("• Review entry conditions")
            analysis.append("• Check risk-reward ratio")
        
        return "\n".join(analysis)
    
    def get_market_status(self, position: Dict, current_price: float, signal_data: Dict) -> str:
        if not position:
            return "No active position"
        
        entry_price = position['entry_price']
        position_type = position['type']
        pnl = current_price - entry_price if position_type == 'LONG' else entry_price - current_price
        pnl_pct = (pnl / entry_price) * 100
        
        status = []
        
        if position_type == 'LONG':
            status.append("📈 **LONG Position Active**")
        else:
            status.append("📉 **SHORT Position Active**")
        
        if pnl > 0:
            status.append(f"✅ In Profit: +{pnl:.2f} pts (+{pnl_pct:.2f}%)")
        elif pnl < 0:
            status.append(f"⚠️ In Loss: {pnl:.2f} pts ({pnl_pct:.2f}%)")
        else:
            status.append("➡️ At entry price")
        
        if position['sl_price']:
            sl_dist = abs(current_price - position['sl_price'])
            status.append(f"🛡️ SL Distance: {sl_dist:.2f} pts")
        
        if position['target_price']:
            target_dist = abs(position['target_price'] - current_price)
            status.append(f"🎯 Target Distance: {target_dist:.2f} pts")
        
        return "\n".join(status)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.markdown('<h1 class="main-header">🚀 Multi-Strategy Live Trading System</h1>', unsafe_allow_html=True)

trading_system = TradingSystem()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Strategy Selection
    strategy_type = st.selectbox(
        "📊 Select Strategy",
        [
            "EMA/SMA Crossover",
            "Z-Score Mean Reversion",
            "RSI Strategy",
            "MACD Crossover",
            "Bollinger Bands",
            "Breakout Strategy",
            "Volume Strategy",
            "Support/Resistance"
        ]
    )
    
    st.markdown("---")
    
    # Asset Selection
    st.subheader("📈 Asset")
    asset_type = st.selectbox(
        "Type",
        ["Indian Indices", "Crypto", "Forex", "Custom"]
    )
    
    ticker_map = {
        "Indian Indices": {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN"},
        "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"},
        "Forex": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/INR": "INR=X"}
    }
    
    if asset_type == "Custom":
        ticker = st.text_input("Ticker", "RELIANCE.NS")
    else:
        asset_name = st.selectbox("Asset", list(ticker_map[asset_type].keys()))
        ticker = ticker_map[asset_type][asset_name]
    
    # Timeframe
    st.subheader("⏰ Timeframe")
    col1, col2 = st.columns(2)
    with col1:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])
    with col2:
        period_map = {
            "1m": ["1d", "5d"],
            "5m": ["1d", "5d", "1mo"],
            "15m": ["1d", "5d", "1mo"],
            "30m": ["1d", "5d", "1mo"],
            "1h": ["1d", "5d", "1mo", "3mo"],
            "1d": ["1mo", "3mo", "6mo", "1y"]
        }
        period = st.selectbox("Period", period_map.get(interval, ["1d"]))
    
    st.markdown("---")
    
    # Strategy Parameters
    st.subheader("🎯 Strategy Params")
    strategy_params = {}
    
    if strategy_type == "EMA/SMA Crossover":
        col1, col2 = st.columns(2)
        with col1:
            fast_type = st.selectbox("Fast", ["EMA", "SMA"])
            fast_period = st.number_input("Fast Period", 1, 200, 9)
        with col2:
            slow_type = st.selectbox("Slow", ["EMA", "SMA"])
            slow_period = st.number_input("Slow Period", 1, 200, 20)
        strategy_params = {
            'fast_type': fast_type, 'fast_period': fast_period,
            'slow_type': slow_type, 'slow_period': slow_period
        }
    
    elif strategy_type == "Z-Score Mean Reversion":
        window = st.number_input("Window", 10, 100, 20)
        threshold = st.slider("Z-Score Threshold", 1.0, 3.0, 2.0, 0.1)
        strategy_params = {'window': window, 'entry_threshold': threshold}
    
    elif strategy_type == "Bollinger Bands":
        period = st.number_input("Period", 10, 50, 20)
        std_dev = st.slider("Std Dev", 1.0, 3.0, 2.0, 0.1)
        strategy_params = {'period': period, 'std_dev': std_dev}
    
    elif strategy_type == "Breakout Strategy":
        window = st.number_input("Window", 10, 50, 20)
        strategy_params = {'window': window}
    
    elif strategy_type == "Volume Strategy":
        window = st.number_input("Window", 10, 50, 20)
        strategy_params = {'window': window}
    
    elif strategy_type == "Support/Resistance":
        window = st.number_input("Window", 10, 50, 20)
        tolerance = st.slider("Tolerance %", 0.5, 5.0, 2.0) / 100
        strategy_params = {'window': window, 'tolerance': tolerance}
    
    st.markdown("---")
    
    # Risk Management
    st.subheader("🛡️ Risk Management")
    
    sl_type = st.selectbox("Stop Loss", ["Custom Points", "Trail SL", "Signal Based"])
    sl_value = None
    if sl_type in ["Custom Points", "Trail SL"]:
        sl_value = st.number_input("SL Points", 0.0, 1000.0, 10.0, 0.5)
    sl_config = {"type": sl_type, "value": sl_value}
    
    target_type = st.selectbox("Target", ["Custom Points", "Trail Target", "Signal Based"])
    target_value = None
    if target_type in ["Custom Points", "Trail Target"]:
        target_value = st.number_input("Target Points", 0.0, 1000.0, 20.0, 0.5)
    target_config = {"type": target_type, "value": target_value}
    
    position_size = st.number_input("Position Size", 1, 1000, 1)

# Initialize Strategy
strategy_map = {
    "EMA/SMA Crossover": EMASMACrossoverStrategy,
    "Z-Score Mean Reversion": ZScoreStrategy,
    "RSI Strategy": RSIStrategy,
    "MACD Crossover": MACDStrategy,
    "Bollinger Bands": BollingerBandsStrategy,
    "Breakout Strategy": BreakoutStrategy,
    "Volume Strategy": VolumeStrategy,
    "Support/Resistance": SupportResistanceStrategy
}

selected_strategy = strategy_map[strategy_type](strategy_params)
trading_system.set_strategy(selected_strategy)

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Trading", "📈 History", "📋 Log"])

with tab1:
    # Control Buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("▶️ Start", type="primary", disabled=st.session_state.trading_active):
            st.session_state.trading_active = True
            st.session_state.trade_log.append({
                'time': datetime.now(trading_system.ist_tz),
                'message': f'Started: {strategy_type} on {ticker} ({interval})'
            })
            st.rerun()
    
    with col2:
        if st.button("⏸️ Stop", disabled=not st.session_state.trading_active):
            st.session_state.trading_active = False
            st.rerun()
    
    with col3:
        if st.session_state.trading_active:
            st.markdown('<p class="status-running">🟢 Active</p>', unsafe_allow_html=True)
        else:
