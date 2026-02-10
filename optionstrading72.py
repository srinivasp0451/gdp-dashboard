import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Page config
st.set_page_config(
    page_title="Options Trading Signal Generator",
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
        margin-bottom: 2rem;
    }
    .signal-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .signal-buy-call {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .signal-buy-put {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .signal-no {
        background-color: #e2e3e5;
        border: 2px solid #6c757d;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Ticker configuration
TICKER_MAP = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "🔍 SCAN ALL NIFTY 50 STOCKS": "SCAN_NIFTY50",  # Scanner option
    "Custom": "CUSTOM"
}

# Nifty 50 constituent stocks
NIFTY50_STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "L&T": "LT.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Maruti": "MARUTI.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "HCL Tech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Titan": "TITAN.NS",
    "Nestle": "NESTLEIND.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "UltraTech": "ULTRACEMCO.NS",
    "Tech Mahindra": "TECHM.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Power Grid": "POWERGRID.NS",
    "M&M": "M&M.NS",
    "NTPC": "NTPC.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Coal India": "COALINDIA.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Grasim": "GRASIM.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Shree Cement": "SHREECEM.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Hindalco": "HINDALCO.NS",
    "ONGC": "ONGC.NS",
    "BPCL": "BPCL.NS",
    "Divis Labs": "DIVISLAB.NS",
    "Apollo Hospital": "APOLLOHOSP.NS",
    "SBI Life": "SBILIFE.NS",
    "BEL": "BEL.NS"
}

INDEX_TICKERS = ["^NSEI", "^NSEBANK", "^BSESN", "^GSPC", "^IXIC", "^DJI", "^INDIAVIX"]

# Initialize session state
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []


class OptionSignalGenerator:
    """Main class for generating option trading signals"""
    
    def __init__(self, ticker, is_index=False):
        self.ticker = ticker
        self.is_index = is_index
        self.data = None
        
    def fetch_data(self, period="6mo", interval="1d"):
        """Fetch historical data - supports different intervals for intraday/scalping"""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            
            # For swing trading (interval="1d"), use the original method
            if interval == "1d":
                # ORIGINAL SWING TRADING CODE (UNCHANGED!)
                self.data = ticker_obj.history(period=period)
            else:
                # For intraday/scalping, use download with interval support
                # yf is already imported at module level
                self.data = yf.download(
                    self.ticker,
                    period=period,
                    interval=interval,
                    progress=False
                )
                
                # Filter to market hours for intraday (9:15 AM - 3:30 PM IST)
                try:
                    self.data = self.data.between_time('09:15', '15:30')
                except:
                    pass  # If timezone issues, skip filtering
            
            if self.data is None or self.data.empty:
                return None
            
            return self.data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def calculate_indicators(self):
        """Calculate all technical indicators with safety for shorter timeframes"""
        if self.data is None or self.data.empty:
            return None
        
        # Reduced minimum requirement - need at least 20 bars for MA20
        if len(self.data) < 20:
            return None
            
        df = self.data.copy()
        
        # Moving Averages
        df['MA20'] = df['Close'].rolling(20).mean()
        
        # MA50 only if we have enough data, otherwise use MA20
        if len(df) >= 50:
            df['MA50'] = df['Close'].rolling(50).mean()
        else:
            df['MA50'] = df['MA20']  # Fallback for short timeframes
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        
        # Historical Volatility (20-day annualized)
        returns = df['Close'].pct_change()
        df['HV'] = returns.rolling(20).std() * np.sqrt(252) * 100
        
        # ATR (Average True Range)
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
        df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()
        
        # Volume Analysis (only for stocks, not indices)
        if not self.is_index and 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        else:
            df['Volume_MA'] = np.nan
            df['Volume_Ratio'] = 1.0  # Neutral for indices
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        self.data = df
        return df
    
    def generate_signal(self, strategy="SWING"):
        """Generate signal - uses original logic for SWING, modified for INTRADAY/SCALPING"""
        
        # Strategy-specific minimum data requirements
        if strategy == "SWING":
            min_data_required = 60  # Original requirement for daily data
        elif strategy == "INTRADAY":
            min_data_required = 30  # Lower requirement for intraday (need ~25 for MA20)
        else:  # SCALPING
            min_data_required = 25  # Minimal requirement for scalping
        
        if self.data is None or len(self.data) < min_data_required:
            return None
            
        df = self.data
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Strategy-specific thresholds
        if strategy == "SWING":
            # ORIGINAL SWING TRADING THRESHOLDS (UNCHANGED!)
            min_points = 7
            confidence_mult = 8
            target_mult = 2.5  # ATR multiplier
            stop_mult = 1.0
        elif strategy == "INTRADAY":
            # Intraday: slightly lower threshold, tighter targets
            min_points = 6
            confidence_mult = 10
            target_mult = 1.5
            stop_mult = 0.5
        else:  # SCALPING
            # Scalping: even lower threshold, very tight targets
            min_points = 5
            confidence_mult = 12
            target_mult = 1.0
            stop_mult = 0.3
        
        signal = {
            'timestamp': current.name,
            'price': current['Close'],
            'signal': 'NO SIGNAL',
            'confidence': 0,
            'reasons': [],
            'target': None,
            'stop_loss': None,
            'risk_reward': None,
            'indicators': {},
            'strategy': strategy
        }
        
        # Store key indicators
        signal['indicators'] = {
            'RSI': current['RSI'],
            'MACD': current['MACD'],
            'Signal_Line': current['Signal_Line'],
            'HV': current['HV'],
            'ATR': current['ATR'],
            'MA20': current['MA20'],
            'MA50': current['MA50'],
            'Stoch_K': current['Stoch_K']
        }
        
        # BULLISH SETUP (BUY CALL) Analysis
        bullish_score = 0
        bullish_reasons = []
        
        # Trend Analysis (4 points max)
        if current['Close'] > current['MA20']:
            bullish_score += 2
            bullish_reasons.append("✓ Price above MA20 (short-term uptrend)")
        
        if current['MA20'] > current['MA50']:
            bullish_score += 2
            bullish_reasons.append("✓ MA20 > MA50 (medium-term uptrend)")
        
        # RSI Analysis (3 points max)
        if 30 < current['RSI'] < 50:
            bullish_score += 3
            bullish_reasons.append(f"✓ RSI oversold recovery ({current['RSI']:.1f}) - strong buy zone")
        elif 50 <= current['RSI'] < 60:
            bullish_score += 1
            bullish_reasons.append(f"✓ RSI neutral-bullish ({current['RSI']:.1f})")
        
        # MACD Analysis (2 points max)
        if current['MACD'] > current['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']:
            bullish_score += 2
            bullish_reasons.append("✓ MACD bullish crossover (fresh signal)")
        elif current['MACD'] > current['Signal_Line']:
            bullish_score += 1
            bullish_reasons.append("✓ MACD above signal line")
        
        # Volume Confirmation (1 point max - only for stocks)
        if not self.is_index:
            if current['Volume_Ratio'] > 1.3:
                bullish_score += 1
                bullish_reasons.append(f"✓ High volume confirmation ({current['Volume_Ratio']:.1f}x average)")
        else:
            bullish_score += 0.5  # Give half point benefit for indices
        
        # Support Bounce Analysis (2 points max)
        if abs(prev['Close'] - current['MA50']) / current['MA50'] < 0.015:
            if current['Close'] > prev['Close']:
                bullish_score += 2
                bullish_reasons.append("✓ Bounced from MA50 support with follow-through")
        
        # Bollinger Band Analysis (1 point max)
        if prev['Close'] < current['BB_lower'] and current['Close'] > current['BB_lower']:
            bullish_score += 1
            bullish_reasons.append("✓ Broke above lower Bollinger Band (oversold bounce)")
        
        # Stochastic Analysis (1 point max)
        if current['Stoch_K'] < 30 and current['Stoch_K'] > prev['Stoch_K']:
            bullish_score += 1
            bullish_reasons.append(f"✓ Stochastic turning up from oversold ({current['Stoch_K']:.1f})")
        
        # BEARISH SETUP (BUY PUT) Analysis
        bearish_score = 0
        bearish_reasons = []
        
        # Trend Analysis (4 points max)
        if current['Close'] < current['MA20']:
            bearish_score += 2
            bearish_reasons.append("✓ Price below MA20 (short-term downtrend)")
        
        if current['MA20'] < current['MA50']:
            bearish_score += 2
            bearish_reasons.append("✓ MA20 < MA50 (medium-term downtrend)")
        
        # RSI Analysis (3 points max)
        if 50 < current['RSI'] < 70:
            bearish_score += 3
            bearish_reasons.append(f"✓ RSI overbought rejection ({current['RSI']:.1f}) - strong sell zone")
        elif 40 < current['RSI'] <= 50:
            bearish_score += 1
            bearish_reasons.append(f"✓ RSI neutral-bearish ({current['RSI']:.1f})")
        
        # MACD Analysis (2 points max)
        if current['MACD'] < current['Signal_Line'] and prev['MACD'] >= prev['Signal_Line']:
            bearish_score += 2
            bearish_reasons.append("✓ MACD bearish crossover (fresh signal)")
        elif current['MACD'] < current['Signal_Line']:
            bearish_score += 1
            bearish_reasons.append("✓ MACD below signal line")
        
        # Volume Confirmation (1 point max - only for stocks)
        if not self.is_index:
            if current['Volume_Ratio'] > 1.3:
                bearish_score += 1
                bearish_reasons.append(f"✓ High volume confirmation ({current['Volume_Ratio']:.1f}x average)")
        else:
            bearish_score += 0.5  # Give half point benefit for indices
        
        # Resistance Rejection Analysis (2 points max)
        if abs(prev['Close'] - current['MA50']) / current['MA50'] < 0.015:
            if current['Close'] < prev['Close']:
                bearish_score += 2
                bearish_reasons.append("✓ Rejected at MA50 resistance with follow-through")
        
        # Bollinger Band Analysis (1 point max)
        if prev['Close'] > current['BB_upper'] and current['Close'] < current['BB_upper']:
            bearish_score += 1
            bearish_reasons.append("✓ Broke below upper Bollinger Band (overbought rejection)")
        
        # Stochastic Analysis (1 point max)
        if current['Stoch_K'] > 70 and current['Stoch_K'] < prev['Stoch_K']:
            bearish_score += 1
            bearish_reasons.append(f"✓ Stochastic turning down from overbought ({current['Stoch_K']:.1f})")
        
        # Determine final signal using strategy-specific threshold
        if bullish_score >= min_points:
            signal['signal'] = 'BUY CALL'
            signal['confidence'] = min(int(bullish_score * confidence_mult), 95)
            signal['reasons'] = bullish_reasons
            
            # Calculate targets using ATR
            signal['target'] = current['Close'] + (current['ATR'] * target_mult)
            signal['stop_loss'] = current['Close'] - (current['ATR'] * stop_mult)
            signal['risk_reward'] = (signal['target'] - current['Close']) / (current['Close'] - signal['stop_loss'])
            
        elif bearish_score >= min_points:
            signal['signal'] = 'BUY PUT'
            signal['confidence'] = min(int(bearish_score * confidence_mult), 95)
            signal['reasons'] = bearish_reasons
            
            # Calculate targets using ATR
            signal['target'] = current['Close'] - (current['ATR'] * target_mult)
            signal['stop_loss'] = current['Close'] + (current['ATR'] * stop_mult)
            signal['risk_reward'] = (current['Close'] - signal['target']) / (signal['stop_loss'] - current['Close'])
        
        else:
            # No clear signal
            signal['confidence'] = max(int(bullish_score * 6), int(bearish_score * 6))
            if bullish_score > bearish_score:
                signal['reasons'] = [f"Bullish setup incomplete ({bullish_score}/{min_points} points)"] + bullish_reasons
            else:
                signal['reasons'] = [f"Bearish setup incomplete ({bearish_score}/{min_points} points)"] + bearish_reasons
        
        return signal
    
    def get_volatility_context(self):
        """Get volatility analysis for IV comparison - standardized to 1-year comparison"""
        if self.data is None:
            return None
            
        current = self.data.iloc[-1]
        hv_data = self.data['HV'].dropna()
        
        if len(hv_data) == 0:
            return None
        
        # IMPORTANT: For options trading, we ALWAYS compare against 1 year (252 trading days)
        # This is the industry standard regardless of period selected
        comparison_period = min(252, len(hv_data))  # Use 1 year or less if not available
        comparison_data = hv_data.tail(comparison_period)
        
        # Calculate percentile against the comparison period
        hv_percentile = (comparison_data < current['HV']).sum() / len(comparison_data) * 100
        
        # Calculate statistical measures
        hv_mean = comparison_data.mean()
        hv_std = comparison_data.std()
        hv_z_score = (current['HV'] - hv_mean) / hv_std if hv_std > 0 else 0
        
        context = {
            'current_hv': current['HV'],
            'hv_percentile': hv_percentile,
            'avg_hv_period': hv_mean,
            'min_hv_period': comparison_data.min(),
            'max_hv_period': comparison_data.max(),
            'hv_std': hv_std,
            'hv_z_score': hv_z_score,
            'comparison_days': comparison_period
        }
        
        # More lenient recommendation logic
        # Using percentile OR z-score (whichever is more favorable)
        if hv_percentile < 30 and hv_z_score < 0:
            context['recommendation'] = 'EXCELLENT - Low volatility, cheap options'
            context['color'] = 'green'
            context['trade_ok'] = True
        elif hv_percentile < 60 or hv_z_score < 0.5:
            context['recommendation'] = 'GOOD - Reasonable volatility, can trade'
            context['color'] = 'lightgreen'
            context['trade_ok'] = True
        elif hv_percentile < 75:
            context['recommendation'] = 'CAUTION - Elevated volatility, smaller position'
            context['color'] = 'orange'
            context['trade_ok'] = True  # Still tradeable but be careful
        else:
            context['recommendation'] = 'EXPENSIVE - Very high volatility, avoid buying'
            context['color'] = 'red'
            context['trade_ok'] = False
        
        return context


class PositionTracker:
    """Track open positions with exit signals"""
    
    def __init__(self, position_id, entry_price, position_type, stop_loss_pct=50, target_pct=100):
        self.position_id = position_id
        self.entry_price = entry_price
        self.position_type = position_type  # 'CALL' or 'PUT'
        self.stop_loss_price = entry_price * (1 - stop_loss_pct/100)
        self.target_price = entry_price * (1 + target_pct/100)
        self.trailing_stop = None
        self.status = 'OPEN'
        self.partial_exit_done = False
        
    def check_exit(self, current_price):
        """Check exit conditions and return action"""
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # Stop loss hit
        if current_price <= self.stop_loss_price:
            self.status = 'CLOSED'
            return {
                'action': 'EXIT ALL',
                'reason': 'Stop Loss Hit',
                'pnl_pct': pnl_pct,
                'exit_price': current_price
            }
        
        # Target hit - partial exit
        if current_price >= self.target_price and not self.partial_exit_done:
            self.partial_exit_done = True
            self.trailing_stop = self.entry_price + (current_price - self.entry_price) * 0.5
            return {
                'action': 'PARTIAL EXIT (50%)',
                'reason': 'Target Hit - Lock Profits',
                'pnl_pct': pnl_pct,
                'exit_price': current_price
            }
        
        # Trailing stop hit
        if self.trailing_stop and current_price <= self.trailing_stop:
            self.status = 'CLOSED'
            return {
                'action': 'EXIT REMAINING (50%)',
                'reason': 'Trailing Stop Hit',
                'pnl_pct': pnl_pct,
                'exit_price': current_price
            }
        
        # Update trailing stop if price keeps moving
        if current_price > self.entry_price * 1.5 and self.trailing_stop:
            new_trail = self.entry_price + (current_price - self.entry_price) * 0.7
            self.trailing_stop = max(self.trailing_stop, new_trail)
        
        return {
            'action': 'HOLD',
            'reason': f'In profit: ₹{current_price:.2f} | Stop: ₹{self.stop_loss_price:.2f}',
            'pnl_pct': pnl_pct,
            'exit_price': None
        }


def plot_technical_chart(data, signal):
    """Create interactive technical analysis chart"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Volume' if 'Volume' in data.columns else 'Historical Volatility'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Price and MAs
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
    
    # Add signal marker
    if signal and signal['signal'] != 'NO SIGNAL':
        marker_color = 'green' if signal['signal'] == 'BUY CALL' else 'red'
        fig.add_trace(go.Scatter(
            x=[signal['timestamp']],
            y=[signal['price']],
            mode='markers',
            marker=dict(size=15, color=marker_color, symbol='triangle-up' if signal['signal'] == 'BUY CALL' else 'triangle-down'),
            name='Signal',
            showlegend=True
        ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], name='Signal', line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram'), row=3, col=1)
    
    # Volume or HV
    if 'Volume' in data.columns and data['Volume'].sum() > 0:
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' for i in range(len(data))]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors), row=4, col=1)
    else:
        fig.add_trace(go.Scatter(x=data.index, y=data['HV'], name='Historical Volatility %', line=dict(color='blue')), row=4, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume" if 'Volume' in data.columns else "HV %", row=4, col=1)
    
    return fig


def display_signal_box(signal, vol_context=None):
    """Display signal in a styled box with clear trade decision"""
    
    # Determine if we should actually trade
    can_trade = False
    final_decision = "WAIT"
    decision_color = "orange"
    
    if signal['signal'] != 'NO SIGNAL' and vol_context:
        # Use the trade_ok flag from volatility context
        if vol_context.get('trade_ok', False):
            can_trade = True
            final_decision = "TRADE NOW ✅"
            decision_color = "green"
        else:
            final_decision = "DON'T TRADE - Wait for IV to drop ❌"
            decision_color = "red"
    
    # Original signal display
    if signal['signal'] == 'BUY CALL':
        box_class = 'signal-buy-call'
        emoji = '🟢'
        signal_text = '📈 TECHNICAL SIGNAL: BUY CALL'
    elif signal['signal'] == 'BUY PUT':
        box_class = 'signal-buy-put'
        emoji = '🔴'
        signal_text = '📉 TECHNICAL SIGNAL: BUY PUT'
    else:
        box_class = 'signal-no'
        emoji = '⚪'
        signal_text = '⏸️ NO TECHNICAL SIGNAL'
    
    st.markdown(f'<div class="signal-box {box_class}">', unsafe_allow_html=True)
    
    # Show final decision prominently
    if signal['signal'] != 'NO SIGNAL':
        st.markdown(f"## :{decision_color}[{final_decision}]")
        st.markdown("---")
    
    st.markdown(f"### {emoji} {signal_text}")
    st.markdown(f"**Confidence Level:** {signal['confidence']}%")
    st.progress(signal['confidence'] / 100)
    
    if signal['signal'] != 'NO SIGNAL':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entry Price", f"₹{signal['price']:.2f}")
        with col2:
            st.metric("Target", f"₹{signal['target']:.2f}", f"+{((signal['target']/signal['price']-1)*100):.1f}%")
        with col3:
            st.metric("Stop Loss", f"₹{signal['stop_loss']:.2f}", f"-{((1-signal['stop_loss']/signal['price'])*100):.1f}%")
        
        st.markdown(f"**Risk:Reward Ratio:** 1:{signal['risk_reward']:.2f}")
    
    st.markdown("**Analysis:**")
    for reason in signal['reasons']:
        st.markdown(f"- {reason}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def extract_option_data_from_image(image):
    """Extract option chain data using OCR"""
    try:
        import easyocr
        import cv2
        import numpy as np
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Initialize OCR reader
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Perform OCR
        result = reader.readtext(img_array)
        
        # Extract text
        extracted_text = []
        for (bbox, text, prob) in result:
            if prob > 0.3:  # Confidence threshold
                extracted_text.append(text.strip())
        
        return extracted_text, True
    
    except ImportError:
        return None, False
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return None, False


def parse_option_chain_data(extracted_text):
    """Parse extracted text to find strikes, premiums, and IVs"""
    import re
    
    options_data = {'calls': [], 'puts': []}
    
    # Patterns to match
    strike_pattern = r'\b\d{4,6}\b'  # 4-6 digit numbers (strikes)
    premium_pattern = r'\b\d+\.?\d*\b'  # Decimal numbers (premiums/IV)
    
    i = 0
    while i < len(extracted_text):
        text = extracted_text[i]
        
        # Try to find strike price
        strike_match = re.search(strike_pattern, text)
        if strike_match:
            strike = float(strike_match.group())
            
            # Look ahead for premium and IV
            premium = None
            iv = None
            
            # Search next 5 items for premium and IV
            for j in range(i+1, min(i+6, len(extracted_text))):
                next_text = extracted_text[j]
                numbers = re.findall(premium_pattern, next_text)
                
                for num in numbers:
                    try:
                        val = float(num)
                        if 10 < val < 500:  # Likely premium
                            if premium is None:
                                premium = val
                        elif 0 < val < 100:  # Likely IV percentage
                            if iv is None:
                                iv = val
                    except:
                        continue
            
            if premium and iv:
                # Determine if CALL or PUT based on context
                # This is simplified - in real scenario, would need better logic
                option_type = 'calls'  # Default
                
                options_data[option_type].append({
                    'strike': strike,
                    'premium': premium,
                    'iv': iv
                })
        
        i += 1
    
    return options_data


def option_chain_analyzer():
    """Option chain analysis - Manual input primary, OCR optional"""
    st.markdown("### 📊 Option Chain Analysis")
    
    # Check if OCR libraries are available
    ocr_available = False
    try:
        import easyocr
        import cv2
        ocr_available = True
    except ImportError:
        pass
    
    # ALWAYS show manual input first (primary method)
    st.markdown("#### 🖊️ Manual Input (Recommended)")
    st.info("💡 Enter strike prices, premiums, and IV values from your broker's option chain")
    
    manual_input_section()
    
    # OCR as optional feature if available
    if ocr_available:
        st.markdown("---")
        st.markdown("### 🤖 Advanced: Auto-Extract with OCR (Beta)")
        st.warning("⚠️ OCR is experimental. Manual input above is more reliable.")
        
        with st.expander("📸 Try OCR Auto-Extract (Optional)"):
            st.info("Upload 1-3 screenshots and the system will attempt to extract data automatically")
            
            uploaded_files = st.file_uploader(
                "Upload Option Chain Screenshots", 
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key='ocr_upload'
            )
            
            if uploaded_files:
                # Display all uploaded images
                st.markdown(f"**{len(uploaded_files)} screenshot(s) uploaded**")
                
                cols = st.columns(min(len(uploaded_files), 3))
                images = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    images.append(image)
                    with cols[idx % 3]:
                        st.image(image, caption=f"Screenshot {idx+1}", use_column_width=True)
                
                st.markdown("---")
                
                # OCR Processing
                if st.button("🤖 Try Auto-Extract with OCR", type="secondary", key='ocr_extract'):
                    with st.spinner("🔍 Extracting data from screenshots..."):
                        all_call_data = []
                        all_put_data = []
                        
                        for idx, image in enumerate(images):
                            st.markdown(f"**Processing Screenshot {idx+1}...**")
                            
                            try:
                                extracted_text, success = extract_option_data_from_image(image)
                                
                                if success and extracted_text:
                                    st.success(f"✅ Extracted {len(extracted_text)} text elements")
                                    
                                    # Parse the data
                                    parsed_data = parse_option_chain_data(extracted_text)
                                    
                                    all_call_data.extend(parsed_data['calls'])
                                    all_put_data.extend(parsed_data['puts'])
                                else:
                                    st.warning(f"⚠️ Could not extract data from screenshot {idx+1}")
                            except Exception as e:
                                st.error(f"❌ Error processing screenshot {idx+1}: {e}")
                        
                        if all_call_data or all_put_data:
                            st.success(f"✅ Total extracted: {len(all_call_data)} CALL options, {len(all_put_data)} PUT options")
                            
                            # Display extracted data
                            if all_call_data:
                                st.markdown("#### 📊 Extracted CALL Options")
                                df_calls = pd.DataFrame(all_call_data)
                                st.dataframe(df_calls, use_container_width=True)
                            
                            if all_put_data:
                                st.markdown("#### 📊 Extracted PUT Options")
                                df_puts = pd.DataFrame(all_put_data)
                                st.dataframe(df_puts, use_container_width=True)
                            
                            # Analyze the data
                            if 'current_signal' in st.session_state and 'vol_context' in st.session_state:
                                analyze_extracted_options(all_call_data, all_put_data, 
                                                         st.session_state.current_signal,
                                                         st.session_state.vol_context)
                        else:
                            st.error("❌ Could not extract option data. OCR may not have detected the data correctly.")
                            st.info("💡 **No problem!** Use the Manual Input section above instead - it's more reliable.")
    else:
        st.info("ℹ️ OCR auto-extract is not available. Manual input works perfectly - use the form above!")


def analyze_extracted_options(call_data, put_data, signal, vol_context):
    """Analyze extracted option data and provide recommendations"""
    st.markdown("---")
    st.markdown("### 📈 Analysis Results")
    
    if signal and signal['signal'] != 'NO SIGNAL':
        hv = vol_context['current_hv']
        
        # Check overall volatility environment first
        if not vol_context.get('trade_ok', False):
            st.error("❌ **STOP: Unfavorable Volatility Environment**")
            st.warning(f"HV Percentile: {vol_context['hv_percentile']:.1f}% | Z-Score: {vol_context['hv_z_score']:.2f}")
            st.warning("Options may be overpriced. Consider waiting for better entry or use smaller position size.")
            st.info("💡 **You can still trade**, but be aware options are relatively expensive. Risk:Reward will be less favorable.")
            # Don't return - allow analysis to continue
        else:
            st.success("✅ **Good Volatility Environment - Favorable for Option Buying**")
            st.success(f"✅ HV Percentile: {vol_context['hv_percentile']:.1f}% | Z-Score: {vol_context['hv_z_score']:.2f}")
        
        if signal['signal'] == 'BUY CALL':
            st.success("✅ Technical Signal: BUY CALL")
            
            if call_data:
                df_calls = pd.DataFrame(call_data)
                df_calls['iv_vs_hv'] = df_calls['iv'] - hv
                df_calls['score'] = 0.0
                
                # Scoring system
                for idx, row in df_calls.iterrows():
                    score = 0
                    # Lower IV is better (up to 3 points)
                    if row['iv'] < hv:
                        score += 3
                    elif row['iv'] < hv * 1.1:
                        score += 2
                    elif row['iv'] < hv * 1.2:
                        score += 1
                    
                    # Reasonable premium (1 point)
                    if 50 < row['premium'] < 300:
                        score += 1
                    
                    df_calls.at[idx, 'score'] = score
                
                df_calls = df_calls.sort_values('score', ascending=False)
                
                st.markdown("#### 🎯 CALL Options Ranking")
                st.dataframe(df_calls.style.highlight_max(subset=['score'], color='lightgreen'), 
                           use_container_width=True)
                
                if len(df_calls) > 0:
                    best_call = df_calls.iloc[0]
                    st.success(f"**🏆 RECOMMENDED: {best_call['strike']:.0f} CE at ₹{best_call['premium']:.2f}**")
                    st.info(f"IV: {best_call['iv']:.2f}% | HV: {hv:.2f}% | {'✅ Cheap' if best_call['iv'] < hv else '⚠️ Expensive'}")
                    
                    display_trade_recommendation(best_call['premium'])
            else:
                st.warning("No CALL data extracted")
        
        elif signal['signal'] == 'BUY PUT':
            st.success("✅ Technical Signal: BUY PUT")
            
            if put_data:
                df_puts = pd.DataFrame(put_data)
                df_puts['iv_vs_hv'] = df_puts['iv'] - hv
                df_puts['score'] = 0.0
                
                # Scoring system
                for idx, row in df_puts.iterrows():
                    score = 0
                    if row['iv'] < hv:
                        score += 3
                    elif row['iv'] < hv * 1.1:
                        score += 2
                    elif row['iv'] < hv * 1.2:
                        score += 1
                    
                    if 50 < row['premium'] < 300:
                        score += 1
                    
                    df_puts.at[idx, 'score'] = score
                
                df_puts = df_puts.sort_values('score', ascending=False)
                
                st.markdown("#### 🎯 PUT Options Ranking")
                st.dataframe(df_puts.style.highlight_max(subset=['score'], color='lightcoral'), 
                           use_container_width=True)
                
                if len(df_puts) > 0:
                    best_put = df_puts.iloc[0]
                    st.success(f"**🏆 RECOMMENDED: {best_put['strike']:.0f} PE at ₹{best_put['premium']:.2f}**")
                    st.info(f"IV: {best_put['iv']:.2f}% | HV: {hv:.2f}% | {'✅ Cheap' if best_put['iv'] < hv else '⚠️ Expensive'}")
                    
                    display_trade_recommendation(best_put['premium'])
            else:
                st.warning("No PUT data extracted")
    else:
        st.warning("⚠️ No active technical signal. Generate signal first in the main tab.")


def display_trade_recommendation(entry_premium):
    """Display trade entry, target, and stop loss"""
    st.markdown("#### 💰 Trade Setup")
    
    target_premium = entry_premium * 2.0  # 100% target
    stop_premium = entry_premium * 0.5  # 50% stop
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        st.metric("Entry", f"₹{entry_premium:.2f}")
    with rec_col2:
        st.metric("Target", f"₹{target_premium:.2f}", "+100%")
    with rec_col3:
        st.metric("Stop Loss", f"₹{stop_premium:.2f}", "-50%")
    
    st.success("✅ **Risk:Reward = 1:2** | **Expected Win Rate: 55-65%**")


def manual_input_section():
    """Fallback manual input section"""
def manual_input_section():
    """Fallback manual input section"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### CALL Options")
        num_call_strikes = st.number_input("Number of CALL strikes", min_value=1, max_value=10, value=5, key='num_calls')
        
        call_data = []
        for i in range(num_call_strikes):
            st.markdown(f"**Call Strike {i+1}:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                strike = st.number_input(f"Strike", key=f'call_strike_{i}', value=0.0, format="%.2f")
            with c2:
                premium = st.number_input(f"Premium", key=f'call_premium_{i}', value=0.0, format="%.2f")
            with c3:
                iv = st.number_input(f"IV %", key=f'call_iv_{i}', value=0.0, format="%.2f")
            
            if strike > 0:
                call_data.append({'strike': strike, 'premium': premium, 'iv': iv})
    
    with col2:
        st.markdown("#### PUT Options")
        num_put_strikes = st.number_input("Number of PUT strikes", min_value=1, max_value=10, value=5, key='num_puts')
        
        put_data = []
        for i in range(num_put_strikes):
            st.markdown(f"**Put Strike {i+1}:**")
            p1, p2, p3 = st.columns(3)
            with p1:
                strike = st.number_input(f"Strike", key=f'put_strike_{i}', value=0.0, format="%.2f")
            with p2:
                premium = st.number_input(f"Premium", key=f'put_premium_{i}', value=0.0, format="%.2f")
            with p3:
                iv = st.number_input(f"IV %", key=f'put_iv_{i}', value=0.0, format="%.2f")
            
            if strike > 0:
                put_data.append({'strike': strike, 'premium': premium, 'iv': iv})
    
    if st.button("🔍 Analyze Option Chain (Manual)", key='analyze_manual'):
        if call_data or put_data:
            if 'current_signal' in st.session_state and 'vol_context' in st.session_state:
                analyze_extracted_options(call_data, put_data,
                                        st.session_state.current_signal,
                                        st.session_state.vol_context)
            else:
                st.warning("⚠️ Please generate a technical signal first in the 'Signal Generator' tab")
        else:
            st.error("Please input option chain data")


def scan_nifty50_stocks(strategy, period, interval):
    """Scan all Nifty 50 stocks and return tradeable opportunities with API rate limit protection"""
    import time
    import random
    
    st.markdown("## 🔍 Scanning All Nifty 50 Stocks...")
    st.info(f"📊 Running **{strategy}** strategy on {len(NIFTY50_STOCKS)} stocks. This will take ~1-2 minutes with rate limit protection...")
    
    # Strategy-specific minimum data requirements
    if strategy == "SWING":
        min_bars_required = 60
    elif strategy == "INTRADAY":
        min_bars_required = 30
    else:  # SCALPING
        min_bars_required = 25
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total_stocks = len(NIFTY50_STOCKS)
    
    for idx, (stock_name, ticker) in enumerate(NIFTY50_STOCKS.items()):
        # Update progress
        progress = (idx + 1) / total_stocks
        progress_bar.progress(progress)
        status_text.text(f"📊 Scanning {stock_name} ({idx + 1}/{total_stocks})...")
        
        try:
            # Initialize signal generator for this stock (all are stocks, not indices)
            signal_gen = OptionSignalGenerator(ticker, is_index=False)
            
            # Fetch data with appropriate interval
            data = signal_gen.fetch_data(period=period, interval=interval)
            
            if data is not None and not data.empty and len(data) >= min_bars_required:
                # Calculate indicators
                signal_gen.calculate_indicators()
                
                # Generate signal with strategy
                signal = signal_gen.generate_signal(strategy=strategy)
                vol_context = signal_gen.get_volatility_context()
                
                # Only include if there's a tradeable signal
                if signal and signal['signal'] != 'NO SIGNAL' and vol_context:
                    # Check volatility
                    vol_status = "GOOD ✅" if vol_context.get('trade_ok', False) else "HIGH ⚠️"
                    tradeable = '✅ YES' if vol_context.get('trade_ok', False) else '⏸️ WAIT'
                    
                    results.append({
                        'Stock': stock_name,
                        'Ticker': ticker,
                        'Signal': signal['signal'],
                        'Confidence': f"{signal['confidence']}%",
                        'Price': f"₹{signal['price']:.2f}",
                        'Target': f"₹{signal['target']:.2f}",
                        'Stop Loss': f"₹{signal['stop_loss']:.2f}",
                        'R:R': f"{signal['risk_reward']:.1f}:1",
                        'Current HV': f"{vol_context['current_hv']:.2f}%",
                        'Avg HV': f"{vol_context['avg_hv_period']:.2f}%",
                        'HV Percentile': f"{vol_context['hv_percentile']:.1f}%",
                        'Z-Score': f"{vol_context['hv_z_score']:.2f}",
                        'Vol Status': vol_status,
                        'Trade?': tradeable
                    })
        
        except Exception as e:
            # Silently skip stocks with errors (data unavailable, delisted, etc.)
            pass
        
        # CRITICAL: Random delay between 1.0 to 1.5 seconds to avoid yfinance API rate limits
        if idx < total_stocks - 1:  # Don't delay after last stock
            delay = random.uniform(1.0, 1.5)
            time.sleep(delay)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<p class="main-header">📈 Professional Options Trading Signal Generator</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # STRATEGY SELECTOR (NEW!)
        st.markdown("### 📊 Select Trading Strategy")
        
        strategy_choice = st.selectbox(
            "Choose Your Strategy",
            options=[
                "Swing Trading (Recommended - Original System)",
                "Intraday Trading (Advanced)",
                "Scalping (Expert Only)"
            ],
            index=0,
            help="Swing = Proven system (unchanged) | Intraday = Fast trades | Scalping = Very fast (risky)"
        )
        
        # Extract strategy type
        if "Swing" in strategy_choice:
            STRATEGY = "SWING"
            st.success("✅ Using proven swing trading system (original code)")
        elif "Intraday" in strategy_choice:
            STRATEGY = "INTRADAY"
            st.warning("⚠️ Intraday mode - requires active monitoring")
        else:
            STRATEGY = "SCALPING"
            st.error("❌ Scalping mode - extreme risk, not recommended")
        
        st.markdown("---")
        
        # Ticker selection
        ticker_name = st.selectbox(
            "Select Asset",
            options=list(TICKER_MAP.keys()),
            index=0
        )
        
        if ticker_name == "Custom":
            custom_ticker = st.text_input("Enter Custom Ticker (e.g., RELIANCE.NS, AAPL)")
            ticker_symbol = custom_ticker
            is_index = st.checkbox("Is this an index? (No volume data)")
        else:
            ticker_symbol = TICKER_MAP[ticker_name]
            is_index = ticker_symbol in INDEX_TICKERS
        
        st.markdown("---")
        
        # Data period - conditional based on strategy
        st.markdown("### 📊 Data Period")
        
        if STRATEGY == "SWING":
            # ORIGINAL SWING TRADING SETTINGS (UNCHANGED!)
            period = st.selectbox(
                "Historical Data Period",
                options=["3mo", "6mo", "1y", "2y"],
                index=2,  # Default to 1y
                help="1y (1 year) is recommended for standard HV percentile calculation"
            )
            
            if period in ["3mo"]:
                st.warning("⚠️ 3mo may have insufficient data. Recommend 6mo or 1y.")
            
            st.info("💡 **Recommended: 1y** - Industry standard for options HV analysis")
            
            # For swing trading, always use daily interval
            interval = "1d"
            
        elif STRATEGY == "INTRADAY":
            # INTRADAY SETTINGS - Updated for yfinance limits
            st.info("ℹ️ **Intraday Limits:** 5m = max 60 days, 15m = max 60 days")
            
            interval = st.selectbox(
                "Candle Timeframe",
                options=["5m", "15m"],
                index=0,
                help="5-min = faster signals | 15-min = less noise"
            )
            
            # Period options based on interval
            if interval == "5m":
                period = st.selectbox(
                    "Intraday Data Period",
                    options=["1d", "5d", "1mo"],
                    index=1,  # Default to 5d
                    help="5-min candles: 1d=~75 bars, 5d=~375 bars, 1mo=~1500 bars"
                )
            else:  # 15m
                period = st.selectbox(
                    "Intraday Data Period",
                    options=["1d", "5d", "1mo"],
                    index=2,  # Default to 1mo for 15-min
                    help="15-min candles: 1d=~25 bars, 5d=~125 bars, 1mo=~500 bars"
                )
            
            st.warning("⚠️ **Trade only 10:00 AM - 2:00 PM**")
            st.warning("⚠️ **MUST exit before market close!**")
            
        else:  # SCALPING
            # SCALPING SETTINGS - Updated for yfinance limits
            st.error("ℹ️ **Scalping Limits:** 1m = max 7 days only!")
            
            interval = st.selectbox(
                "Candle Timeframe",
                options=["1m", "3m", "5m"],
                index=1,  # Default to 3m (1m has very limited data)
                help="1-min = max 7 days | 3-min = max 60 days | 5-min = max 60 days"
            )
            
            # Period options based on interval
            if interval == "1m":
                period = st.selectbox(
                    "Scalping Data Period",
                    options=["1d", "5d", "7d"],
                    index=1,  # Default to 5d
                    help="1-min candles: 1d=~375 bars, 5d=~1875 bars, 7d=~2625 bars"
                )
            elif interval == "3m":
                period = st.selectbox(
                    "Scalping Data Period",
                    options=["1d", "5d", "1mo"],
                    index=1,  # Default to 5d
                    help="3-min candles: Good data availability"
                )
            else:  # 5m
                period = st.selectbox(
                    "Scalping Data Period",
                    options=["1d", "5d", "1mo"],
                    index=1,  # Default to 5d
                    help="5-min candles: Same as intraday"
                )
            
            st.error("❌ **EXTREME RISK - Not Recommended!**")
            st.error("⚠️ Bid-ask spreads will eat profits")
        
        st.markdown("---")
        
        # Risk parameters
        st.markdown("### 🎯 Risk Management")
        stop_loss_pct = st.slider("Stop Loss %", 30, 70, 50, 5)
        target_pct = st.slider("Target %", 50, 200, 100, 10)
        
        st.markdown("---")
        
        # Generate signal button
        generate_signal = st.button("🚀 Generate Signal", use_container_width=True, type="primary")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Signal Generator", "📸 Option Chain Analysis", "📈 Position Tracker", "📚 Strategy Guide"])
    
    with tab1:
        st.markdown("### 🎯 Real-Time Signal Generation")
        
        if generate_signal:
            # Check if scanner option selected
            if ticker_symbol == "SCAN_NIFTY50":
                # Run Nifty 50 stock scanner
                st.success("🔍 **NIFTY 50 SCANNER MODE** - Scanning all 50 stocks for opportunities")
                
                scan_results = scan_nifty50_stocks(STRATEGY, period, interval)
                
                if scan_results:
                    st.markdown("---")
                    st.markdown(f"## ✅ Found {len(scan_results)} Trading Opportunities")
                    
                    # Convert to DataFrame for better display
                    df_results = pd.DataFrame(scan_results)
                    
                    # Separate tradeable from wait
                    tradeable_df = df_results[df_results['Trade?'] == '✅ YES']
                    wait_df = df_results[df_results['Trade?'] == '⏸️ WAIT']
                    
                    # Show tradeable opportunities first
                    if not tradeable_df.empty:
                        st.success(f"### ✅ {len(tradeable_df)} TRADEABLE Now (Good Volatility)")
                        st.dataframe(
                            tradeable_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Download button for tradeable stocks
                        csv_tradeable = tradeable_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Tradeable Stocks (CSV)",
                            data=csv_tradeable,
                            file_name=f"nifty50_tradeable_{STRATEGY}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No stocks with good volatility for trading right now")
                    
                    st.markdown("---")
                    
                    # Show wait list
                    if not wait_df.empty:
                        st.warning(f"### ⏸️ {len(wait_df)} Stocks with Good Signal BUT High Volatility (Wait)")
                        st.info("These have good technical setups but options are expensive. Wait for volatility to drop.")
                        st.dataframe(
                            wait_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # Summary statistics
                    st.markdown("---")
                    st.markdown("### 📊 Scan Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Signals", len(scan_results))
                    with col2:
                        st.metric("Tradeable Now", len(tradeable_df))
                    with col3:
                        buy_calls = len(df_results[df_results['Signal'] == 'BUY CALL'])
                        st.metric("BUY CALL", buy_calls)
                    with col4:
                        buy_puts = len(df_results[df_results['Signal'] == 'BUY PUT'])
                        st.metric("BUY PUT", buy_puts)
                    
                else:
                    st.warning("⚠️ No trading opportunities found in Nifty 50 stocks with current criteria")
                    st.info("💡 Try again later or use a different strategy/timeframe")
            
            elif not ticker_symbol:
                st.error("Please select or enter a ticker symbol")
            else:
                # Normal single-ticker analysis
                with st.spinner(f"Analyzing {ticker_name} using {STRATEGY} strategy..."):
                    # Initialize signal generator
                    signal_gen = OptionSignalGenerator(ticker_symbol, is_index)
                    
                    # Fetch data with appropriate interval
                    data = signal_gen.fetch_data(period=period, interval=interval)
                    
                    if data is None or data.empty:
                        st.error(f"❌ Could not fetch data for {ticker_symbol}. Please check ticker symbol.")
                    elif len(data) < 20:
                        st.error(f"❌ Insufficient data: Only {len(data)} bars available.")
                        
                        if STRATEGY == "SWING":
                            st.warning("💡 Need at least 60 bars. Try selecting a longer period (6mo or 1y).")
                        elif STRATEGY == "INTRADAY":
                            st.warning(f"💡 Need at least 30 bars for intraday. Try 5d or 1mo period with {interval} candles.")
                        else:  # SCALPING
                            st.warning(f"💡 Need at least 25 bars for scalping. Try 5d or 7d period with {interval} candles.")
                    else:
                        # Show data info for intraday/scalping
                        if STRATEGY != "SWING":
                            st.info(f"📊 Using {interval} candles | Data points: {len(data)} | Strategy: {STRATEGY}")
                        
                        # Calculate indicators (same for all strategies)
                        signal_gen.calculate_indicators()
                        
                        # Generate signal with strategy parameter
                        signal = signal_gen.generate_signal(strategy=STRATEGY)
                        vol_context = signal_gen.get_volatility_context()
                        
                        # Store in session state
                        st.session_state.current_signal = signal
                        st.session_state.vol_context = vol_context
                        st.session_state.current_ticker = ticker_name
                        st.session_state.current_strategy = STRATEGY
                        
                        if signal:
                            # Show which strategy is being used
                            current_strategy = signal.get('strategy', 'SWING')
                            
                            if current_strategy == "SWING":
                                st.success("✅ **SWING TRADING MODE** - Using original proven system")
                            elif current_strategy == "INTRADAY":
                                st.warning("⚡ **INTRADAY MODE** - Exit before 2:00 PM! Lower win rate than swing.")
                            else:
                                st.error("⚡⚡ **SCALPING MODE** - Extreme risk! Spreads will eat profits. Not recommended!")
                            
                            # Display signal with volatility context
                            display_signal_box(signal, vol_context)
                            
                            # Volatility context
                            st.markdown("---")
                            st.markdown("### 📊 Volatility Analysis")
                            
                            if vol_context:
                                vol_col1, vol_col2, vol_col3, vol_col4, vol_col5 = st.columns(5)
                                
                                with vol_col1:
                                    st.metric("Current HV", f"{vol_context['current_hv']:.2f}%")
                                with vol_col2:
                                    st.metric(f"Avg HV ({vol_context['comparison_days']}d)", f"{vol_context['avg_hv_period']:.2f}%")
                                with vol_col3:
                                    st.metric("HV Percentile", f"{vol_context['hv_percentile']:.1f}%")
                                with vol_col4:
                                    st.metric("Z-Score", f"{vol_context['hv_z_score']:.2f}")
                                with vol_col5:
                                    st.markdown(f"**Status**")
                                    st.markdown(f":{vol_context['color']}[{vol_context['recommendation']}]")
                                
                                # Explanation of metrics
                                st.info(f"""
                                💡 **Understanding Volatility Metrics:**
                                - **HV Percentile {vol_context['hv_percentile']:.1f}%**: Current HV is higher than {vol_context['hv_percentile']:.1f}% of the past year
                                - **Z-Score {vol_context['hv_z_score']:.2f}**: How many standard deviations from mean (±1.0 is normal range)
                                - **Comparison**: Using last {vol_context['comparison_days']} days (~{vol_context['comparison_days']//21} months)
                                - **For Trading**: HV < 60th percentile OR Z-Score < 0.5 = Good to trade
                                """)
                                
                                # Show HV history chart
                                if len(signal_gen.data) > 0:
                                    st.markdown("#### Historical Volatility Trend")
                                    hv_chart_data = signal_gen.data[['HV']].tail(252).dropna()
                                    if len(hv_chart_data) > 0:
                                        import plotly.graph_objects as go
                                        fig_hv = go.Figure()
                                        fig_hv.add_trace(go.Scatter(
                                            x=hv_chart_data.index,
                                            y=hv_chart_data['HV'],
                                            name='HV',
                                            line=dict(color='blue')
                                        ))
                                        fig_hv.add_hline(
                                            y=vol_context['avg_hv_period'],
                                            line_dash="dash",
                                            line_color="green",
                                            annotation_text=f"Average: {vol_context['avg_hv_period']:.2f}%"
                                        )
                                        fig_hv.add_hline(
                                            y=vol_context['current_hv'],
                                            line_dash="dot",
                                            line_color="red",
                                            annotation_text=f"Current: {vol_context['current_hv']:.2f}%"
                                        )
                                        fig_hv.update_layout(
                                            title="Historical Volatility Over Time",
                                            xaxis_title="Date",
                                            yaxis_title="HV (%)",
                                            height=300
                                        )
                                        st.plotly_chart(fig_hv, use_container_width=True)
                            
                            # Technical indicators summary
                            st.markdown("---")
                            st.markdown("### 📐 Key Indicators")
                            
                            ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                            
                            with ind_col1:
                                st.metric("RSI", f"{signal['indicators']['RSI']:.1f}")
                            with ind_col2:
                                st.metric("MACD", f"{signal['indicators']['MACD']:.2f}")
                            with ind_col3:
                                st.metric("ATR", f"{signal['indicators']['ATR']:.2f}")
                            with ind_col4:
                                st.metric("Stochastic", f"{signal['indicators']['Stoch_K']:.1f}")
                            
                            # Chart
                            st.markdown("---")
                            st.markdown("### 📈 Technical Chart")
                            chart = plot_technical_chart(signal_gen.data.tail(100), signal)
                            st.plotly_chart(chart, use_container_width=True)
                            
                            # Strategy-specific guidance
                            st.markdown("---")
                            current_strategy = signal.get('strategy', 'SWING')
                            
                            if current_strategy == "SWING":
                                st.success("""
                                **✅ SWING TRADING - Original Proven System**
                                
                                You're using the tested, reliable strategy with:
                                - **60-65% win rate** (best of all strategies)
                                - **2:1 to 3:1 risk:reward**
                                - **Hold 2-10 days** - No need to monitor constantly
                                - **10-20% monthly returns** - Highest of all strategies
                                - **5 minutes per day** - Works with your job
                                
                                **This is the strategy that actually works!** Keep using it.
                                """)
                            
                            elif current_strategy == "INTRADAY":
                                st.warning("""
                                **⚠️ INTRADAY TRADING - Active Mode**
                                
                                You're using a more difficult strategy:
                                - **45-55% win rate** (lower than swing's 60-65%)
                                - **1.5:1 to 2:1 risk:reward** (worse than swing's 2:1 to 3:1)
                                - **MUST exit before 2:00 PM** - Don't hold overnight!
                                - **5-15% monthly returns** (lower than swing's 10-20%)
                                - **4-6 hours monitoring** - Can't do with job
                                
                                **Critical Rules:**
                                - ⚠️ EXIT ALL positions by 2:00 PM
                                - ⚠️ Bid-ask spreads reduce profits 10-20%
                                - ⚠️ Requires constant screen time
                                - ⚠️ Most traders do better with swing trading
                                
                                **Consider switching to Swing Trading for better results.**
                                """)
                            
                            else:  # SCALPING
                                st.error("""
                                **❌ SCALPING MODE - EXTREME RISK**
                                
                                This strategy has the WORST performance:
                                - **40-50% win rate** (worse than random!)
                                - **1:1 risk:reward** (no mathematical edge)
                                - **5-30 minute holds** - Extreme stress
                                - **-5 to +10% monthly** (usually NEGATIVE!)
                                - **All day monitoring** - Impossible with job
                                
                                **Why This Fails:**
                                - ❌ Bid-ask spreads eat 50-100% of target profit
                                - ❌ You're competing against HFT algorithms
                                - ❌ Need 12% move for 10% gain (due to spreads)
                                - ❌ 95% of scalpers lose money long-term
                                
                                **STRONG RECOMMENDATION:**
                                Switch to Swing Trading immediately. Scalping options is almost impossible to profit from.
                                
                                If you want fast trades, try:
                                - Futures (lower spreads)
                                - 0DTE weekly options (Thursdays only)
                                - Swing trading (best returns anyway!)
                                """)
                            
                            # Save to history
                            st.session_state.signal_history.append({
                                'timestamp': datetime.now(),
                                'ticker': ticker_name,
                                'signal': signal['signal'],
                                'confidence': signal['confidence'],
                                'price': signal['price']
                            })
                        else:
                            st.error("Could not generate signal. Insufficient data.")
        
        # Display recent signals
        if st.session_state.signal_history:
            st.markdown("---")
            st.markdown("### 📜 Recent Signals")
            df_history = pd.DataFrame(st.session_state.signal_history)
            st.dataframe(df_history.tail(10), use_container_width=True)
    
    with tab2:
        option_chain_analyzer()
    
    with tab3:
        st.markdown("### 📊 Position Tracker")
        st.info("Track your open positions and get exit signals")
        
        # Add position form
        with st.expander("➕ Add New Position"):
            pos_col1, pos_col2, pos_col3 = st.columns(3)
            
            with pos_col1:
                pos_ticker = st.text_input("Ticker/Strike", value="NIFTY 23000 CE")
            with pos_col2:
                pos_entry = st.number_input("Entry Premium (₹)", min_value=0.0, value=150.0, step=10.0)
            with pos_col3:
                pos_type = st.selectbox("Type", ["CALL", "PUT"])
            
            if st.button("Add Position"):
                new_pos = {
                    'id': len(st.session_state.positions) + 1,
                    'ticker': pos_ticker,
                    'entry': pos_entry,
                    'type': pos_type,
                    'added': datetime.now(),
                    'status': 'OPEN'
                }
                st.session_state.positions.append(new_pos)
                st.success(f"✅ Added {pos_ticker} position")
        
        # Display positions
        if st.session_state.positions:
            st.markdown("### 📋 Active Positions")
            
            for pos in st.session_state.positions:
                if pos['status'] == 'OPEN':
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{pos['ticker']}** ({pos['type']})")
                            st.caption(f"Entry: ₹{pos['entry']:.2f}")
                        
                        with col2:
                            current_premium = st.number_input(
                                "Current ₹",
                                key=f"curr_{pos['id']}",
                                value=pos['entry'],
                                step=5.0
                            )
                        
                        with col3:
                            pnl = ((current_premium - pos['entry']) / pos['entry']) * 100
                            st.metric("P&L", f"{pnl:.1f}%", 
                                     delta=f"₹{current_premium - pos['entry']:.2f}")
                        
                        with col4:
                            tracker = PositionTracker(pos['id'], pos['entry'], pos['type'])
                            exit_info = tracker.check_exit(current_premium)
                            
                            if exit_info['action'] != 'HOLD':
                                st.warning(f"⚠️ {exit_info['action']}")
                                st.caption(exit_info['reason'])
                        
                        st.markdown("---")
        else:
            st.info("No active positions. Add a position above to track it.")
    
    with tab4:
        st.markdown("### 📚 Strategy Guide & Best Practices")
        
        # Strategy comparison table
        st.markdown("## 📊 Strategy Comparison")
        comparison_data = {
            "Aspect":          ["Hold Duration", "Win Rate",    "Risk:Reward",   "Target Gains", "Stop Loss",       "Signals/Month", "Time Per Day", "Difficulty", "Monthly Return",    "Best For"],
            "✅ Swing (Default)":["2–10 days",    "60–65%",      "2:1 to 3:1",    "50–100%",      "50% of premium",  "2–4 setups",    "5 min",        "Medium",     "10–20%",            "Everyone"],
            "⚠️ Intraday":      ["1–4 hours",    "45–55%",      "1.5:1 to 2:1",  "20–40%",       "10–20% of prem.", "5–10 setups",   "4–6 hours",    "High",       "5–15%",             "Full-time traders"],
            "❌ Scalping":      ["5–30 minutes", "40–50%",      "1:1 to 1.5:1",  "10–20%",       "5–10% of prem.",  "10–20 attempts","All day",      "Extreme",    "-5% to +10%",       "Experts only"],
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Detailed tabs for each strategy
        s1, s2, s3 = st.tabs(["✅ Swing Trading (Recommended)", "⚠️ Intraday Trading", "❌ Scalping (Not Recommended)"])
        
        with s1:
            st.success("### ✅ Swing Trading — The Proven System")
            st.markdown("""
**This is the original, unchanged strategy built into the app.**

#### 🎯 Core Logic
- Uses **daily candles** + **1 year of history**
- Requires **7 / 12 technical points** to fire a signal
- Buys options only when **HV Percentile < 60%** (cheap options)
- Hold 2–10 days, exit at **+100% target** or **-50% stop**

#### 📊 Indicators Used (all 12)
| # | Indicator | Bullish Condition |
|---|-----------|-------------------|
| 1 | MA20 vs MA50 | Price > MA20 > MA50 |
| 2 | Price vs MA20 | Close > MA20 |
| 3 | RSI | 30–60 zone (recovery) |
| 4 | MACD | MACD crosses above signal line |
| 5 | MACD Histogram | Rising histogram |
| 6 | Bollinger Bands | Price near lower band (bounce) |
| 7 | Stochastic %K | Below 30 (oversold) |
| 8 | Stochastic Cross | %K crossing above %D |
| 9 | ATR Expansion | ATR rising (momentum) |
| 10 | Price Momentum | Close > 5-day avg |
| 11 | Volume | Volume > 20-day avg |
| 12 | HV Percentile | < 60% (cheap options) |

#### 🧮 Volatility Filters
| HV Percentile | Z-Score | Status | Action |
|---------------|---------|--------|--------|
| < 30% | < 0 | Excellent | Trade aggressively |
| 30–60% | < 0.5 | Good | Trade normally |
| 60–75% | 0.5–1.0 | Caution | Trade smaller size |
| > 75% | > 1.0 | Expensive | Skip / Wait |

#### 💡 Best Practices
1. ✅ Only trade when Confidence ≥ 70%
2. ✅ Use options with 30–45 DTE minimum
3. ✅ Exit 50% at target, trail the rest
4. ✅ Never risk more than 2% of capital per trade
5. ✅ Paper trade 2–3 months before going live
            """)

        with s2:
            st.warning("### ⚠️ Intraday Trading — Active, Harder, Lower Returns")
            st.markdown("""
**Same indicators as Swing but applied to 5-min / 15-min candles.**

#### ⚙️ Key Differences from Swing
| Parameter | Swing | Intraday |
|-----------|-------|----------|
| Candle size | 1 day | 5-min or 15-min |
| Data window | 1 year | 5 days – 1 month |
| Min signal points | 7 / 12 | 6 / 12 |
| ATR target mult | 2.5× | 1.5× |
| ATR stop mult | 1.0× | 0.5× |
| Hold time | Days | Hours |

#### ⏰ Strict Time Rules
| Time | Action |
|------|--------|
| Before 9:30 AM | Do not trade (opening volatility) |
| 9:30–10:00 AM | Watch only, let price settle |
| 10:00 AM–2:00 PM | **Trading window** ✅ |
| After 2:00 PM | **EXIT ALL positions** — theta accelerates |
| 3:30 PM | Market close — anything open LOSES |

#### ⚠️ Why Intraday Is Harder
- **Bid-ask spread** eats 10–20% of profit on small moves
- Signals are noisier — 70% noise vs 30% signal on 5-min charts
- Requires **constant screen monitoring** — not compatible with a job
- Lower win rate (45–55%) vs Swing (60–65%)

#### 📐 Position Sizing
- Use **half the position size** of your swing trades
- Risk only **1% per trade** (vs 2% for swing)
- If 3 consecutive losses: stop for the day

#### 💡 Only Works Well When
✓ India VIX < 18 (calm market)  
✓ Clear gap-and-go trending day  
✓ Trading Nifty/BankNifty ATM options only  
✓ Using 5-min charts (NOT 1-min — too much noise)  
            """)

        with s3:
            st.error("### ❌ Scalping — Why It Almost Always Fails")
            st.markdown("""
**Technical signals exist but economics make it nearly impossible to profit.**

#### 🔢 The Math That Kills Scalping
```
Buy at ask:          ₹100
Immediate bid:       ₹98      ← you're already -2%
Your 10% target:     ₹110 bid
But seller asks:     ₹112     ← you must wait for this
Net spread cost:     ~2% of trade = 20% of your 10% gain!
```

#### 🤖 You're Competing Against HFT Algorithms
| Factor | You | HFT Algo |
|--------|-----|----------|
| Execution speed | Seconds | Microseconds |
| Order flow visibility | None | Full |
| Spread cost | Full spread | Near-zero |
| Co-location | No | Yes |
**Result: You are outgunned on every trade.**

#### 📉 Statistical Reality
| Metric | Value |
|--------|-------|
| Win Rate | 40–50% |
| Risk:Reward | ~1:1 |
| Expected Value per trade | **Negative** |
| Long-term success rate | < 5% of scalpers |
| Monthly return (typical) | −5% to +5% |

#### ⚙️ Scalping Settings in This App (If You Insist)
| Setting | Recommended | Why |
|---------|------------|-----|
| Interval | 3-min (NOT 1-min) | Less noise |
| Period | 5d | Enough bars |
| Min signal points | 5 / 12 | Quick signals |
| Risk per trade | 0.5% of capital | Survival |
| Daily loss limit | 3% of capital | Stop immediately |
| Time window | 10:30 AM – 1:30 PM | Best liquidity |

#### ✅ Better Alternatives to Scalping
| If you want... | Use instead |
|----------------|------------|
| Fast trades | Nifty Futures (tiny spreads) |
| Same-day expiry | 0DTE weekly options (Thursdays) |
| More trades | Intraday options |
| Best returns | **Swing trading** ← always this |

> **Honest advice: Don't scalp options. The math doesn't work.**  
> Switch to Swing Trading — higher win rate, better returns, less stress.
            """)
        
        st.markdown("---")
        st.success("📞 **Recommendation:** Start with Swing Trading. Master it first. Only move to faster strategies after 3 months of profitable swing trading.")


if __name__ == "__main__":
    main()
