import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import time
from typing import Dict, List, Tuple, Optional
import ta
from dataclasses import dataclass

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

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
        
    def fetch_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch stock data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_swing_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for swing trading"""
        df = data.copy()
        
        # Moving Averages
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        # Average True Range for volatility
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
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

class SwingTradingAnalyzer:
    """Main application class for swing trading analysis"""
    
    def __init__(self):
        self.core = SwingTradingCore()
        self.last_update = None
        
    def scan_market(self) -> List[SwingSignal]:
        """Scan the market for swing trading opportunities"""
        print("🔍 Scanning market for swing trading opportunities...")
        print("=" * 60)
        
        all_signals = []
        processed = 0
        
        for symbol in self.core.nifty_stocks:
            try:
                print(f"Analyzing {symbol}... ({processed + 1}/{len(self.core.nifty_stocks)})")
                
                data = self.core.fetch_data(symbol)
                if data is None:
                    continue
                
                df = self.core.calculate_swing_indicators(data)
                signals = self.core.identify_swing_patterns(df, symbol)
                
                all_signals.extend(signals)
                processed += 1
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by expected points (highest first)
        all_signals.sort(key=lambda x: x.expected_points, reverse=True)
        
        print(f"\n✅ Scan completed! Analyzed {processed} stocks")
        print(f"🎯 Found {len(all_signals)} swing opportunities")
        
        self.last_update = datetime.now()
        return all_signals
    
    def display_signals(self, signals: List[SwingSignal]):
        """Display swing trading signals in a formatted way"""
        if not signals:
            print("\n❌ No swing trading opportunities found at the moment")
            print("💡 Try again later or consider different market conditions")
            return
        
        print(f"\n🚀 TOP SWING TRADING OPPORTUNITIES")
        print("=" * 80)
        
        for i, signal in enumerate(signals[:10], 1):  # Show top 10
            print(f"\n#{i} 📈 {signal.symbol.replace('.NS', '')}")
            print(f"   Action: {signal.action}")
            print(f"   Entry: ₹{signal.entry_price:.2f}")
            print(f"   Target: ₹{signal.target_price:.2f}")
            print(f"   Stop Loss: ₹{signal.stop_loss:.2f}")
            print(f"   Expected Gain: {signal.expected_points:.0f} points")
            print(f"   Risk:Reward: 1:{signal.risk_reward:.1f}")
            print(f"   Confidence: {signal.confidence}")
            print(f"   Timeframe: {signal.timeframe}")
            print(f"   Reason: {signal.reason}")
            print("-" * 50)
    
    def create_detailed_analysis(self, symbol: str):
        """Create detailed analysis for a specific stock"""
        print(f"\n📊 DETAILED ANALYSIS: {symbol}")
        print("=" * 50)
        
        data = self.core.fetch_data(symbol, period="3mo")
        if data is None:
            print("❌ Unable to fetch data")
            return
        
        df = self.core.calculate_swing_indicators(data)
        
        # Current metrics
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        print(f"Current Price: ₹{current_price:.2f}")
        print(f"RSI: {latest['RSI']:.1f}")
        print(f"EMA 20: ₹{latest['EMA_20']:.2f}")
        print(f"EMA 50: ₹{latest['EMA_50']:.2f}")
        print(f"Support: ₹{latest['Support']:.2f}")
        print(f"Resistance: ₹{latest['Resistance']:.2f}")
        print(f"Volume Ratio: {latest['Volume_Ratio']:.1f}x")
        
        # Plot analysis
        self._plot_swing_analysis(df, symbol)
    
    def _plot_swing_analysis(self, df: pd.DataFrame, symbol: str):
        """Create comprehensive swing trading chart"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price chart with indicators
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        ax1.plot(df.index, df['EMA_20'], label='EMA 20', alpha=0.7)
        ax1.plot(df.index, df['EMA_50'], label='EMA 50', alpha=0.7)
        ax1.plot(df.index, df['Support'], label='Support', linestyle='--', alpha=0.5)
        ax1.plot(df.index, df['Resistance'], label='Resistance', linestyle='--', alpha=0.5)
        ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.1, label='Bollinger Bands')
        
        ax1.set_title(f'{symbol} - Swing Trading Analysis', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        ax2.plot(df.index, df['RSI'], color='orange', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax2.fill_between(df.index, 30, 70, alpha=0.1)
        ax2.set_title('RSI (14)', fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Volume
        ax3.bar(df.index, df['Volume'], alpha=0.6, color='blue')
        ax3.plot(df.index, df['Volume_SMA'], color='red', linewidth=2, label='Volume SMA')
        ax3.set_title('Volume Analysis', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_live_scanner(self):
        """Run continuous market scanner"""
        print("🔴 LIVE SWING TRADING SCANNER")
        print("=" * 40)
        print("⏰ Scanning every 5 minutes...")
        print("📊 Press Ctrl+C to stop")
        
        try:
            while True:
                signals = self.scan_market()
                self.display_signals(signals)
                
                if signals:
                    print(f"\n⏰ Last updated: {self.last_update.strftime('%H:%M:%S')}")
                    print("💰 Top recommendations above target 200-500 points!")
                
                print("\n⏳ Next scan in 5 minutes...")
                print("=" * 60)
                
                time.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            print("\n🛑 Scanner stopped by user")

def main():
    """Main function to run the swing trading analyzer"""
    analyzer = SwingTradingAnalyzer()
    
    while True:
        print("\n🎯 SWING TRADING ANALYZER")
        print("=" * 40)
        print("1. 🔍 Quick Market Scan")
        print("2. 📊 Detailed Stock Analysis")
        print("3. 🔴 Live Scanner (Auto-refresh)")
        print("4. 📈 Market Overview")
        print("5. ❌ Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            signals = analyzer.scan_market()
            analyzer.display_signals(signals)
            
        elif choice == "2":
            symbol = input("Enter stock symbol (e.g., RELIANCE.NS): ").strip().upper()
            if not symbol.endswith('.NS'):
                symbol += '.NS'
            analyzer.create_detailed_analysis(symbol)
            
        elif choice == "3":
            analyzer.run_live_scanner()
            
        elif choice == "4":
            print("\n📈 NIFTY 50 SWING OPPORTUNITIES")
            signals = analyzer.scan_market()
            if signals:
                high_confidence = [s for s in signals if s.confidence == "High"]
                print(f"🎯 High Confidence Signals: {len(high_confidence)}")
                print(f"📊 Total Opportunities: {len(signals)}")
                print(f"💰 Average Expected Points: {np.mean([s.expected_points for s in signals]):.0f}")
            
        elif choice == "5":
            print("👋 Thank you for using Swing Trading Analyzer!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
