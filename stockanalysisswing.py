import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Swing Trading Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class SwingSignal:
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    target_price: float
    stop_loss: float
    expected_points: float
    confidence: str
    timeframe: str
    reason: str
    risk_reward: float

class SwingTradingCore:
    """Core logic for swing trading analysis"""
    
    def __init__(self):
        self.nifty_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
            'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'LT.NS',
            'HCLTECH.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NESTLEIND.NS',
            'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS', 'SUNPHARMA.NS', 'INDUSINDBK.NS',
            'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'ONGC.NS', 'COALINDIA.NS', 'JSWSTEEL.NS',
            'TATASTEEL.NS', 'GRASIM.NS', 'HINDALCO.NS', 'DRREDDY.NS', 'CIPLA.NS',
            'HEROMOTOCO.NS', 'EICHERMOT.NS', 'BRITANNIA.NS', 'DIVISLAB.NS', 'ADANIPORTS.NS'
        ]
        
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_data(_self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch stock data with caching"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_swing_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for swing trading using manual calculations"""
        df = data.copy()
        
        # Moving Averages
        df['EMA_20'] = self._calculate_ema(df['Close'], 20)
        df['EMA_50'] = self._calculate_ema(df['Close'], 50)
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        ema_12 = self._calculate_ema(df['Close'], 12)
        ema_26 = self._calculate_ema(df['Close'], 26)
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = self._calculate_ema(df['MACD'], 9)
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma_20 + (std_20 * 2)
        df['BB_Lower'] = sma_20 - (std_20 * 2)
        df['BB_Middle'] = sma_20
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        # Average True Range for volatility
        df['ATR'] = self._calculate_atr(df['High'], df['Low'], df['Close'], 14)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def identify_swing_patterns(self, df: pd.DataFrame, symbol: str) -> List[SwingSignal]:
        """Identify swing trading opportunities"""
        signals = []
        
        if len(df) < 50:
            return signals
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = latest['Close']
        
        # Pattern 1: Bullish Breakout from consolidation
        breakout_signal = self._check_breakout_pattern(df, symbol, current_price)
        if breakout_signal:
            signals.append(breakout_signal)
        
        # Pattern 2: RSI Oversold Bounce
        oversold_signal = self._check_oversold_bounce(df, symbol, current_price)
        if oversold_signal:
            signals.append(oversold_signal)
        
        # Pattern 3: Moving Average Crossover
        ma_signal = self._check_ma_crossover(df, symbol, current_price)
        if ma_signal:
            signals.append(ma_signal)
        
        # Pattern 4: Bollinger Band Squeeze breakout
        bb_signal = self._check_bb_squeeze_breakout(df, symbol, current_price)
        if bb_signal:
            signals.append(bb_signal)
        
        # Pattern 5: Volume breakout
        volume_signal = self._check_volume_breakout(df, symbol, current_price)
        if volume_signal:
            signals.append(volume_signal)
        
        return signals
    
    def _check_breakout_pattern(self, df: pd.DataFrame, symbol: str, price: float) -> Optional[SwingSignal]:
        """Check for breakout from consolidation pattern"""
        latest = df.iloc[-1]
        
        # Check if price is breaking above resistance with volume
        resistance = latest['Resistance']
        volume_surge = latest['Volume_Ratio'] > 1.5
        
        if price > resistance * 1.01 and volume_surge:
            target = price + (price - df['Support'].iloc[-1]) * 0.8
            stop_loss = resistance * 0.98
            expected_points = target - price
            
            if expected_points >= 200:  # Minimum 200 points gain
                return SwingSignal(
                    symbol=symbol,
                    action="BUY",
                    entry_price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    expected_points=expected_points,
                    confidence="High" if expected_points > 300 else "Medium",
                    timeframe="3-5 days",
                    reason="Breakout from resistance with volume surge",
                    risk_reward=(target - price) / (price - stop_loss)
                )
        return None
    
    def _check_oversold_bounce(self, df: pd.DataFrame, symbol: str, price: float) -> Optional[SwingSignal]:
        """Check for oversold bounce opportunity"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # RSI oversold and turning up
        rsi_oversold = latest['RSI'] < 35 and latest['RSI'] > prev['RSI']
        near_support = price <= latest['Support'] * 1.02
        
        if rsi_oversold and near_support:
            target = latest['EMA_20']
            stop_loss = latest['Support'] * 0.98
            expected_points = target - price
            
            if expected_points >= 200:
                return SwingSignal(
                    symbol=symbol,
                    action="BUY",
                    entry_price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    expected_points=expected_points,
                    confidence="Medium",
                    timeframe="2-4 days",
                    reason="RSI oversold bounce from support",
                    risk_reward=(target - price) / (price - stop_loss)
                )
        return None
    
    def _check_ma_crossover(self, df: pd.DataFrame, symbol: str, price: float) -> Optional[SwingSignal]:
        """Check for moving average crossover"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # EMA 20 crossing above EMA 50
        golden_cross = (latest['EMA_20'] > latest['EMA_50'] and 
                       prev['EMA_20'] <= prev['EMA_50'])
        
        above_200sma = price > latest['SMA_200']
        
        if golden_cross and above_200sma:
            target = price + (latest['ATR'] * 3)
            stop_loss = latest['EMA_50'] * 0.98
            expected_points = target - price
            
            if expected_points >= 200:
                return SwingSignal(
                    symbol=symbol,
                    action="BUY",
                    entry_price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    expected_points=expected_points,
                    confidence="High",
                    timeframe="5-7 days",
                    reason="Golden cross with trend confirmation",
                    risk_reward=(target - price) / (price - stop_loss)
                )
        return None
    
    def _check_bb_squeeze_breakout(self, df: pd.DataFrame, symbol: str, price: float) -> Optional[SwingSignal]:
        """Check for Bollinger Band squeeze breakout"""
        latest = df.iloc[-1]
        
        # Bollinger Bands are narrow (squeeze) and price breaking out
        bb_width = (latest['BB_Upper'] - latest['BB_Lower']) / latest['BB_Middle']
        bb_squeeze = bb_width < 0.1  # Narrow bands
        
        breakout_up = price > latest['BB_Upper']
        
        if bb_squeeze and breakout_up:
            target = price + (latest['BB_Upper'] - latest['BB_Lower'])
            stop_loss = latest['BB_Middle']
            expected_points = target - price
            
            if expected_points >= 200:
                return SwingSignal(
                    symbol=symbol,
                    action="BUY",
                    entry_price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    expected_points=expected_points,
                    confidence="Medium",
                    timeframe="3-5 days",
                    reason="Bollinger Band squeeze breakout",
                    risk_reward=(target - price) / (price - stop_loss)
                )
        return None
    
    def _check_volume_breakout(self, df: pd.DataFrame, symbol: str, price: float) -> Optional[SwingSignal]:
        """Check for volume breakout pattern"""
        latest = df.iloc[-1]
        
        # High volume with price movement
        volume_breakout = latest['Volume_Ratio'] > 2.0
        price_momentum = (price - df['Close'].iloc[-5]) / df['Close'].iloc[-5] > 0.03
        
        if volume_breakout and price_momentum:
            target = price + (latest['ATR'] * 4)
            stop_loss = df['Low'].iloc[-5:].min()
            expected_points = target - price
            
            if expected_points >= 200:
                return SwingSignal(
                    symbol=symbol,
                    action="BUY",
                    entry_price=price,
                    target_price=target,
                    stop_loss=stop_loss,
                    expected_points=expected_points,
                    confidence="High" if expected_points > 400 else "Medium",
                    timeframe="2-4 days",
                    reason="High volume breakout with momentum",
                    risk_reward=(target - price) / (price - stop_loss)
                )
        return None

# Initialize the core analyzer
@st.cache_resource
def get_analyzer():
    return SwingTradingCore()

def create_swing_chart(df: pd.DataFrame, symbol: str, signals: List[SwingSignal] = None):
    """Create interactive swing trading chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Indicators', 'RSI', 'MACD', 'Volume'),
        row_width=[0.2, 0.1, 0.1, 0.1]
    )
    
    # Price chart with candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA_20'],
        line=dict(color='orange', width=2),
        name='EMA 20'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA_50'],
        line=dict(color='blue', width=2),
        name='EMA 50'
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_Upper'],
        line=dict(color='gray', width=1, dash='dash'),
        name='BB Upper'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_Lower'],
        line=dict(color='gray', width=1, dash='dash'),
        name='BB Lower',
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)'
    ), row=1, col=1)
    
    # Support and Resistance
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Support'],
        line=dict(color='green', width=1, dash='dot'),
        name='Support'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Resistance'],
        line=dict(color='red', width=1, dash='dot'),
        name='Resistance'
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        line=dict(color='orange', width=2),
        name='RSI'
    ), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        line=dict(color='blue', width=2),
        name='MACD'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'],
        line=dict(color='red', width=2),
        name='MACD Signal'
    ), row=3, col=1)
    
    fig.add_trace(go.Bar(
        x=df.index, y=df['MACD_Histogram'],
        name='MACD Histogram',
        marker_color='gray'
    ), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume',
        marker_color='lightblue'
    ), row=4, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Volume_SMA'],
        line=dict(color='red', width=2),
        name='Volume SMA'
    ), row=4, col=1)
    
    # Add signal markers if provided
    if signals:
        for signal in signals:
            fig.add_trace(go.Scatter(
                x=[df.index[-1]],
                y=[signal.entry_price],
                mode='markers',
                marker=dict(size=15, color='lime', symbol='triangle-up'),
                name=f'{signal.action} Signal',
                text=f'{signal.reason}'
            ), row=1, col=1)
    
    fig.update_layout(
        title=f'{symbol} - Swing Trading Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📈 Swing Trading Analyzer")
    st.markdown("### Target: 200-500 Points Gain | Live Market Scanner")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Mode selection
        mode = st.selectbox(
            "Select Mode",
            ["Market Scanner", "Individual Analysis", "Live Dashboard"]
        )
        
        # Stock selection for individual analysis
        if mode == "Individual Analysis":
            analyzer = get_analyzer()
            selected_stock = st.selectbox(
                "Select Stock",
                analyzer.nifty_stocks,
                format_func=lambda x: x.replace('.NS', '')
            )
        
        # Filters
        st.subheader("📊 Filters")
        min_points = st.slider("Minimum Expected Points", 200, 1000, 200, 50)
        confidence_filter = st.multiselect(
            "Confidence Level",
            ["High", "Medium"],
            default=["High", "Medium"]
        )
        
        # Auto-refresh for live mode
        if mode == "Live Dashboard":
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
            if auto_refresh:
                st.rerun()
    
    analyzer = get_analyzer()
    
    if mode == "Market Scanner":
        st.header("🔍 Market Scanner Results")
        
        if st.button("🚀 Scan Market", type="primary"):
            with st.spinner("Scanning NIFTY 50 stocks for swing opportunities..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_signals = []
                
                for i, symbol in enumerate(analyzer.nifty_stocks):
                    status_text.text(f"Analyzing {symbol.replace('.NS', '')}...")
                    progress_bar.progress((i + 1) / len(analyzer.nifty_stocks))
                    
                    try:
                        data = analyzer.fetch_data(symbol)
                        if data is not None and len(data) > 50:
                            df = analyzer.calculate_swing_indicators(data)
                            signals = analyzer.identify_swing_patterns(df, symbol)
                            
                            # Apply filters
                            filtered_signals = [
                                s for s in signals 
                                if s.expected_points >= min_points and s.confidence in confidence_filter
                            ]
                            all_signals.extend(filtered_signals)
                    except Exception as e:
                        continue
                
                progress_bar.empty()
                status_text.empty()
                
                # Sort by expected points
                all_signals.sort(key=lambda x: x.expected_points, reverse=True)
                
                if all_signals:
                    st.success(f"✅ Found {len(all_signals)} swing opportunities!")
                    
                    # Display results in cards
                    for i, signal in enumerate(all_signals[:10]):  # Top 10
                        with st.container():
                            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                            
                            with col1:
                                confidence_color = "🟢" if signal.confidence == "High" else "🟡"
                                st.metric(
                                    f"{confidence_color} {signal.symbol.replace('.NS', '')}",
                                    f"₹{signal.entry_price:.2f}",
                                    f"{signal.action}"
                                )
                            
                            with col2:
                                st.metric(
                                    "Expected Gain",
                                    f"{signal.expected_points:.0f} pts",
                                    f"₹{signal.target_price:.2f}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Risk:Reward",
                                    f"1:{signal.risk_reward:.1f}",
                                    f"SL: ₹{signal.stop_loss:.2f}"
                                )
                            
                            with col4:
                                st.metric(
                                    "Timeframe",
                                    signal.timeframe,
                                    signal.confidence
                                )
                            
                            st.caption(f"📝 {signal.reason}")
                            st.divider()
                else:
                    st.warning("❌ No swing opportunities found matching your criteria")
    
    elif mode == "Individual Analysis":
        st.header(f"📊 Detailed Analysis: {selected_stock.replace('.NS', '')}")
        
        with st.spinner("Loading stock data and analysis..."):
            data = analyzer.fetch_data(selected_stock, period="3mo")
            
            if data is not None:
                df = analyzer.calculate_swing_indicators(data)
                signals = analyzer.identify_swing_patterns(df, selected_stock)
                
                # Apply filters
                filtered_signals = [
                    s for s in signals 
                    if s.expected_points >= min_points and s.confidence in confidence_filter
                ]
                
                # Display current metrics
                latest = df.iloc[-1]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Current Price", f"₹{latest['Close']:.2f}")
                
                with col2:
                    rsi_color = "inverse" if latest['RSI'] > 70 or latest['RSI'] < 30 else "normal"
                    st.metric("RSI", f"{latest['RSI']:.1f}", delta_color=rsi_color)
                
                with col3:
                    st.metric("EMA 20", f"₹{latest['EMA_20']:.2f}")
                
                with col4:
                    st.metric("Volume Ratio", f"{latest['Volume_Ratio']:.1f}x")
                
                with col5:
                    st.metric("ATR", f"₹{latest['ATR']:.2f}")
                
                # Display signals
                if filtered_signals:
                    st.subheader("🎯 Swing Opportunities")
                    for signal in filtered_signals:
                        with st.expander(f"📈 {signal.action} Signal - {signal.expected_points:.0f} points"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Entry Price:** ₹{signal.entry_price:.2f}")
                                st.write(f"**Target Price:** ₹{signal.target_price:.2f}")
                                st.write(f"**Stop Loss:** ₹{signal.stop_loss:.2f}")
                            
                            with col2:
                                st.write(f"**Expected Points:** {signal.expected_points:.0f}")
                                st.write(f"**Risk:Reward:** 1:{signal.risk_reward:.1f}")
                                st.write(f"**Timeframe:** {signal.timeframe}")
                                st.write(f"**Confidence:** {signal.confidence}")
                            
                            st.write(f"**Strategy:** {signal.reason}")
                
                # Display chart
                fig = create_swing_chart(df, selected_stock, filtered_signals)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Failed to load stock data")
    
    elif mode == "Live Dashboard":
        st.header("🔴 Live Dashboard")
        st.markdown("*Real-time swing trading opportunities*")
        
        # Auto-refresh mechanism
        placeholder = st.empty()
        
        with placeholder.container():
            with st.spinner("Loading live data..."):
                all_signals = []
                
                # Quick scan of top 20 stocks
                top_stocks = analyzer.nifty_stocks[:20]
                
                for symbol in top_stocks:
                    try:
                        data = analyzer.fetch_data(symbol)
                        if data is not None and len(data) > 50:
                            df = analyzer.calculate_swing_indicators(data)
                            signals = analyzer.identify_swing_patterns(df, symbol)
                            
                            # Apply filters
                            filtered_signals = [
                                s for s in signals 
                                if s.expected_points >= min_points and s.confidence in confidence_filter
                            ]
                            all_signals.extend(filtered_signals)
                    except Exception:
                        continue
                
                all_signals.sort(key=lambda x: x.expected_points, reverse=True)
                
                # Display dashboard
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Active Signals", len(all_signals))
                
                with col2:
                    high_conf_signals = [s for s in all_signals if s.confidence == "High"]
                    st.metric("High Confidence", len(high_conf_signals))
                
                with col3:
                    avg_points = np.mean([s.expected_points for s in all_signals]) if all_signals else 0
                    st.metric("Avg Expected Points", f"{avg_points:.0f}")
                
                # Top 5 opportunities
                if all_signals:
                    st.subheader("🚀 Top 5 Opportunities")
                    
                    for signal in all_signals[:5]:
                        with st.container():
                            col1, col2, col3 = st.columns([2, 2, 3])
                            
                            with col1:
                                confidence_emoji = "🟢" if signal.confidence == "High" else "🟡"
                                st.write(f"**{confidence_emoji} {signal.symbol.replace('.NS', '')}**")
                                st.write(f"Entry: ₹{signal.entry_price:.2f}")
                            
                            with col2:
                                st.write(f"**{signal.expected_points:.0f} points**")
                                st.write(f"Target: ₹{signal.target_price:.2f}")
                            
                            with col3:
                                st.write(f"*{signal.reason}*")
                                st.write(f"RR: 1:{signal.risk_reward:.1f} | {signal.timeframe}")
                            
                            st.divider()
                else:
                    st.info("⏳ No opportunities found. Market conditions may not be favorable for swing trading.")
                
                st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:** This is for educational purposes only. Always do your own research before trading.")

if __name__ == "__main__":
    main()
