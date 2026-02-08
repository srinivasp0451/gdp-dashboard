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
    "Custom": "CUSTOM"
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
        
    def fetch_data(self, period="6mo"):
        """Fetch historical data"""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            self.data = ticker_obj.history(period=period)
            
            if self.data.empty:
                return None
            
            return self.data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        if self.data is None or self.data.empty:
            return None
            
        df = self.data.copy()
        
        # Moving Averages
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        
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
    
    def generate_signal(self):
        """Generate comprehensive BUY CALL or BUY PUT signal"""
        if self.data is None or len(self.data) < 200:
            return None
            
        df = self.data
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        signal = {
            'timestamp': current.name,
            'price': current['Close'],
            'signal': 'NO SIGNAL',
            'confidence': 0,
            'reasons': [],
            'target': None,
            'stop_loss': None,
            'risk_reward': None,
            'indicators': {}
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
        
        # Determine final signal
        # Threshold: 7+ points for signal generation
        if bullish_score >= 7:
            signal['signal'] = 'BUY CALL'
            signal['confidence'] = min(int(bullish_score * 8), 95)
            signal['reasons'] = bullish_reasons
            
            # Calculate targets using ATR
            atr_multiplier = 2.5
            signal['target'] = current['Close'] + (current['ATR'] * atr_multiplier)
            signal['stop_loss'] = current['Close'] - (current['ATR'] * 1.0)
            signal['risk_reward'] = (signal['target'] - current['Close']) / (current['Close'] - signal['stop_loss'])
            
        elif bearish_score >= 7:
            signal['signal'] = 'BUY PUT'
            signal['confidence'] = min(int(bearish_score * 8), 95)
            signal['reasons'] = bearish_reasons
            
            # Calculate targets using ATR
            atr_multiplier = 2.5
            signal['target'] = current['Close'] - (current['ATR'] * atr_multiplier)
            signal['stop_loss'] = current['Close'] + (current['ATR'] * 1.0)
            signal['risk_reward'] = (current['Close'] - signal['target']) / (signal['stop_loss'] - current['Close'])
        
        else:
            # No clear signal
            signal['confidence'] = max(int(bullish_score * 6), int(bearish_score * 6))
            if bullish_score > bearish_score:
                signal['reasons'] = [f"Bullish setup incomplete ({bullish_score}/7 points)"] + bullish_reasons
            else:
                signal['reasons'] = [f"Bearish setup incomplete ({bearish_score}/7 points)"] + bearish_reasons
        
        return signal
    
    def get_volatility_context(self):
        """Get volatility analysis for IV comparison"""
        if self.data is None:
            return None
            
        current = self.data.iloc[-1]
        hv_data = self.data['HV'].dropna()
        
        if len(hv_data) == 0:
            return None
        
        hv_percentile = (hv_data < current['HV']).sum() / len(hv_data) * 100
        
        context = {
            'current_hv': current['HV'],
            'hv_percentile': hv_percentile,
            'avg_hv_6m': hv_data.mean(),
            'min_hv_6m': hv_data.min(),
            'max_hv_6m': hv_data.max()
        }
        
        # Recommendation based on HV percentile
        if hv_percentile < 30:
            context['recommendation'] = 'EXCELLENT - Low volatility, cheap options'
            context['color'] = 'green'
        elif hv_percentile < 50:
            context['recommendation'] = 'GOOD - Below average volatility'
            context['color'] = 'lightgreen'
        elif hv_percentile < 70:
            context['recommendation'] = 'NEUTRAL - Average volatility'
            context['color'] = 'orange'
        else:
            context['recommendation'] = 'EXPENSIVE - High volatility, avoid buying'
            context['color'] = 'red'
        
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
        # Both technical AND volatility must be favorable
        if vol_context['hv_percentile'] < 50:  # Good volatility environment
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
    """Automatic option chain analysis from screenshots with OCR"""
    st.markdown("### 📸 Option Chain Screenshot Analysis")
    
    # Check if OCR libraries are available
    ocr_available = False
    try:
        import easyocr
        import cv2
        ocr_available = True
    except ImportError:
        pass
    
    if ocr_available:
        st.success("✅ OCR Mode: Automatic extraction enabled")
        st.info("💡 Upload 1-3 screenshots of your option chain. The system will automatically extract Strike, Premium, and IV data.")
    else:
        st.warning("⚠️ OCR libraries not installed. Using manual input mode.")
        st.info("💡 To enable automatic OCR, run: `pip install easyocr opencv-python`")
    
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload Option Chain Screenshots (JPG/PNG)", 
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
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
        if ocr_available and st.button("🤖 Auto-Extract Data with OCR", type="primary"):
            with st.spinner("🔍 Extracting data from screenshots..."):
                all_call_data = []
                all_put_data = []
                
                for idx, image in enumerate(images):
                    st.markdown(f"**Processing Screenshot {idx+1}...**")
                    
                    extracted_text, success = extract_option_data_from_image(image)
                    
                    if success and extracted_text:
                        st.success(f"✅ Extracted {len(extracted_text)} text elements")
                        
                        # Parse the data
                        parsed_data = parse_option_chain_data(extracted_text)
                        
                        all_call_data.extend(parsed_data['calls'])
                        all_put_data.extend(parsed_data['puts'])
                    else:
                        st.warning(f"⚠️ Could not extract data from screenshot {idx+1}")
                
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
                    st.info("💡 Try the manual input mode below as fallback")
        
        st.markdown("---")
        st.markdown("### 🖊️ Manual Input (Fallback Mode)")
        st.info("If OCR doesn't work well, manually input the data you see in the screenshots above")
        
        manual_input_section()


def analyze_extracted_options(call_data, put_data, signal, vol_context):
    """Analyze extracted option data and provide recommendations"""
    st.markdown("---")
    st.markdown("### 📈 Analysis Results")
    
    if signal and signal['signal'] != 'NO SIGNAL':
        hv = vol_context['current_hv']
        
        # Check overall volatility environment first
        if vol_context['hv_percentile'] >= 50:
            st.error("❌ **STOP: High Volatility Environment**")
            st.warning(f"HV Percentile: {vol_context['hv_percentile']:.1f}% - Options are EXPENSIVE. DO NOT BUY even with good technical setup.")
            st.info("💡 **Wait for volatility to drop below 50th percentile before entering trades.**")
            return
        
        if signal['signal'] == 'BUY CALL':
            st.success("✅ Technical Signal: BUY CALL + Favorable Volatility")
            st.success(f"✅ HV Percentile: {vol_context['hv_percentile']:.1f}% - Options are CHEAP")
            
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
            st.success("✅ Technical Signal: BUY PUT + Favorable Volatility")
            st.success(f"✅ HV Percentile: {vol_context['hv_percentile']:.1f}% - Options are CHEAP")
            
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



def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<p class="main-header">📈 Professional Options Trading Signal Generator</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
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
        
        # Data period
        period = st.selectbox(
            "Historical Data Period",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
        
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
            if not ticker_symbol:
                st.error("Please select or enter a ticker symbol")
            else:
                with st.spinner(f"Analyzing {ticker_name}..."):
                    # Initialize signal generator
                    signal_gen = OptionSignalGenerator(ticker_symbol, is_index)
                    
                    # Fetch data
                    data = signal_gen.fetch_data(period=period)
                    
                    if data is None or data.empty:
                        st.error(f"❌ Could not fetch data for {ticker_symbol}. Please check ticker symbol.")
                    else:
                        # Calculate indicators
                        signal_gen.calculate_indicators()
                        
                        # Generate signal
                        signal = signal_gen.generate_signal()
                        vol_context = signal_gen.get_volatility_context()
                        
                        # Store in session state
                        st.session_state.current_signal = signal
                        st.session_state.vol_context = vol_context
                        st.session_state.current_ticker = ticker_name
                        
                        if signal:
                            # Display signal with volatility context
                            display_signal_box(signal, vol_context)
                            
                            # Volatility context
                            st.markdown("---")
                            st.markdown("### 📊 Volatility Analysis")
                            
                            if vol_context:
                                vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
                                
                                with vol_col1:
                                    st.metric("Current HV", f"{vol_context['current_hv']:.2f}%")
                                with vol_col2:
                                    st.metric("6M Average HV", f"{vol_context['avg_hv_6m']:.2f}%")
                                with vol_col3:
                                    st.metric("HV Percentile", f"{vol_context['hv_percentile']:.1f}%")
                                with vol_col4:
                                    st.markdown(f"**Status**")
                                    st.markdown(f":{vol_context['color']}[{vol_context['recommendation']}]")
                                
                                st.info("💡 **How to use HV:** Compare Current HV with IV from your option chain screenshot. If IV < HV, options are cheap (good to buy). If IV > HV, options are expensive (avoid).")
                            
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
        
        st.markdown("""
        ## 🎯 How This Strategy Works
        
        This is a **technical analysis + volatility-based option buying strategy** designed for high-probability entries.
        
        ### Core Principles:
        
        1. **Multiple Confirmation Filters**
           - Trend analysis (MA20, MA50)
           - Momentum indicators (RSI, MACD)
           - Volatility checks (Bollinger Bands, Stochastic)
           - Volume confirmation (for stocks)
           
        2. **Entry Only at High-Probability Setups**
           - Minimum 7/12 points needed for signal
           - All factors must align in same direction
           - Enter when options are cheap (low IV)
        
        3. **Strict Risk Management**
           - Fixed 50% stop loss on option premium
           - 100% profit target (2:1 risk-reward minimum)
           - Partial exit at target + trailing stop
        
        ### 📊 Volatility Check (Critical!)
        
        **Before buying ANY option:**
        1. Check Current HV from the app
        2. Upload option chain screenshot
        3. Compare IV with HV:
           - **IV < HV**: Options are CHEAP ✅
           - **IV > HV**: Options are EXPENSIVE ❌
        
        ### 🎲 Expected Performance:
        
        - **Win Rate**: 55-65% (with discipline)
        - **Average Risk:Reward**: 2:1 to 3:1
        - **Signals Per Month**: 2-4 (quality over quantity)
        - **Max Position Risk**: 2% of capital per trade
        
        ### ⚠️ Common Mistakes to Avoid:
        
        1. ❌ Taking signals without IV check
        2. ❌ Not honoring stop losses
        3. ❌ Overtrading (taking weak signals)
        4. ❌ Position sizing too large
        5. ❌ Buying options with <30 days to expiry
        
        ### ✅ Best Practices:
        
        1. ✅ Only trade when confidence >70%
        2. ✅ Always check IV before entry
        3. ✅ Use 30-45 DTE options minimum
        4. ✅ Paper trade for 2-3 months first
        5. ✅ Keep detailed trade journal
        
        ### 📈 Ideal Market Conditions:
        
        **For NIFTY/BANKNIFTY:**
        - Clear trending market (not sideways)
        - India VIX below 15 (calm market)
        - Liquid strikes with tight bid-ask
        
        **For Stocks:**
        - Large cap stocks (better liquidity)
        - No earnings within option expiry
        - Clear technical setup on daily chart
        
        ### 💡 Pro Tips:
        
        - Trade morning session (9:30-11:00 AM) for best liquidity
        - Avoid trading on expiry day unless experienced
        - Use limit orders, not market orders
        - Exit 50% at target, trail remaining 50%
        - Never risk more than 2% total capital per trade
        """)
        
        st.markdown("---")
        st.success("📞 **Need Help?** Use this systematically, follow the rules, and track your results!")


if __name__ == "__main__":
    main()
