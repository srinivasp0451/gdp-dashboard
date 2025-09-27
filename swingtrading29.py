import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import itertools
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced Trading System - Pure Pandas",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedTradingStrategy:
    def __init__(self, strategy_type="Momentum Breakout", ema_fast=9, ema_slow=21, 
                 rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                 bb_period=20, bb_std=2, macd_fast=12, macd_slow=26, macd_signal=9,
                 stop_loss_pct=2.0, take_profit_pct=6.0, use_volume=True):
        
        self.strategy_type = strategy_type
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_volume = use_volume
    
    def calculate_ema(self, data, window):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()
    
    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    def calculate_stochastic(self, high, low, close, window=14, smooth_k=3, smooth_d=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, high, low, close, window=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_cci(self, high, low, close, window=20):
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = self.calculate_sma(typical_price, window)
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def calculate_adx(self, high, low, close, window=14):
        """Calculate Average Directional Index"""
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff.abs()) & (high_diff > 0), 0)
        minus_dm = low_diff.abs().where((low_diff.abs() > high_diff) & (low_diff < 0), 0)
        
        atr = self.calculate_atr(high, low, close, window)
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    def calculate_obv(self, close, volume):
        """Calculate On Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def prepare_data(self, df):
        """Prepare data with all technical indicators"""
        df = df.copy()
        df = df.sort_index()
        df = df.dropna()
        
        if df.empty:
            return df
        
        # Check volume validity
        if 'Volume' in df.columns:
            if df['Volume'].sum() == 0 or df['Volume'].std() == 0:
                self.use_volume = False
                df['Volume'] = df['Close'] * 1000  # Fake volume for indices
        else:
            df['Volume'] = df['Close'] * 1000
            self.use_volume = False
        
        return self.calculate_all_indicators(df)
    
    def calculate_all_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        try:
            # Moving Averages
            df['EMA_Fast'] = self.calculate_ema(df['Close'], self.ema_fast)
            df['EMA_Slow'] = self.calculate_ema(df['Close'], self.ema_slow)
            df['SMA_20'] = self.calculate_sma(df['Close'], 20)
            df['SMA_50'] = self.calculate_sma(df['Close'], 50)
            
            # RSI
            df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(
                df['Close'], self.macd_fast, self.macd_slow, self.macd_signal
            )
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = self.calculate_bollinger_bands(
                df['Close'], self.bb_period, self.bb_std
            )
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'] * 100
            
            # Stochastic
            df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(
                df['High'], df['Low'], df['Close']
            )
            
            # Williams %R
            df['Williams_R'] = self.calculate_williams_r(
                df['High'], df['Low'], df['Close']
            )
            
            # CCI
            df['CCI'] = self.calculate_cci(df['High'], df['Low'], df['Close'])
            
            # ATR
            df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
            
            # ADX
            df['ADX'] = self.calculate_adx(df['High'], df['Low'], df['Close'])
            
            # Volume indicators
            if self.use_volume:
                df['Volume_SMA'] = self.calculate_sma(df['Volume'], 20)
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
                df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
            else:
                # Price-based volume substitute
                df['Price_Change'] = df['Close'].pct_change().abs()
                df['Volume_Ratio'] = df['Price_Change'].rolling(20).rank(pct=True)
                df['OBV'] = (df['Close'].diff() > 0).astype(int).cumsum()
            
            # Price position in range
            df['Price_Position'] = ((df['Close'] - df['Low'].rolling(20).min()) / 
                                   (df['High'].rolling(20).max() - df['Low'].rolling(20).min()) * 100)
            
            # Trend detection
            df['Trend_Up'] = (df['Close'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_50'])
            df['Trend_Down'] = (df['Close'] < df['SMA_20']) & (df['SMA_20'] < df['SMA_50'])
            
            return df.fillna(method='ffill').fillna(method='bfill')
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def generate_signals(self, df):
        """Generate trading signals based on selected strategy"""
        
        if self.strategy_type == "Momentum Breakout":
            return self.momentum_breakout_signals(df)
        elif self.strategy_type == "Mean Reversion":
            return self.mean_reversion_signals(df)
        elif self.strategy_type == "Trend Following":
            return self.trend_following_signals(df)
        elif self.strategy_type == "Multi-Indicator":
            return self.multi_indicator_signals(df)
        elif self.strategy_type == "Scalping":
            return self.scalping_signals(df)
        else:
            return self.momentum_breakout_signals(df)
    
    def momentum_breakout_signals(self, df):
        """Momentum breakout strategy - High frequency, trend catching"""
        # Long signals
        long_conditions = [
            df['Close'] > df['EMA_Fast'],
            df['EMA_Fast'] > df['EMA_Slow'],
            df['RSI'] > 45,
            df['RSI'] < 80,
            df['MACD'] > df['MACD_Signal'],
            df['Volume_Ratio'] > 1.1,
            df['Close'] > df['Close'].shift(1),
            df['ADX'] > 20
        ]
        
        # Short signals  
        short_conditions = [
            df['Close'] < df['EMA_Fast'],
            df['EMA_Fast'] < df['EMA_Slow'],
            df['RSI'] < 55,
            df['RSI'] > 20,
            df['MACD'] < df['MACD_Signal'],
            df['Volume_Ratio'] > 1.1,
            df['Close'] < df['Close'].shift(1),
            df['ADX'] > 20
        ]
        
        # Count satisfied conditions
        df['Long_Score'] = sum([cond.astype(int) for cond in long_conditions])
        df['Short_Score'] = sum([cond.astype(int) for cond in short_conditions])
        
        # Generate signals (need 6 out of 8 conditions)
        df['Long_Signal'] = (df['Long_Score'] >= 6) & (df['Long_Score'].shift(1) < 6)
        df['Short_Signal'] = (df['Short_Score'] >= 6) & (df['Short_Score'].shift(1) < 6)
        
        return df
    
    def mean_reversion_signals(self, df):
        """Mean reversion strategy - Buy low, sell high"""
        # Long signals (oversold conditions)
        long_conditions = [
            df['RSI'] < self.rsi_oversold,
            df['Close'] < df['BB_Lower'],
            df['Stoch_K'] < 20,
            df['Williams_R'] < -80,
            df['CCI'] < -100,
            df['Price_Position'] < 20,
            df['Close'] > df['SMA_50']  # Still in uptrend
        ]
        
        # Short signals (overbought conditions)
        short_conditions = [
            df['RSI'] > self.rsi_overbought,
            df['Close'] > df['BB_Upper'],
            df['Stoch_K'] > 80,
            df['Williams_R'] > -20,
            df['CCI'] > 100,
            df['Price_Position'] > 80,
            df['Close'] < df['SMA_50']  # Still in downtrend
        ]
        
        # Count satisfied conditions
        df['Long_Score'] = sum([cond.astype(int) for cond in long_conditions])
        df['Short_Score'] = sum([cond.astype(int) for cond in short_conditions])
        
        # Generate signals (need 4 out of 7 conditions)
        df['Long_Signal'] = (df['Long_Score'] >= 4) & (df['Long_Score'].shift(1) < 4)
        df['Short_Signal'] = (df['Short_Score'] >= 4) & (df['Short_Score'].shift(1) < 4)
        
        return df
    
    def trend_following_signals(self, df):
        """Trend following strategy - Ride the trends"""
        # Strong uptrend conditions
        long_conditions = [
            df['Close'] > df['SMA_20'],
            df['SMA_20'] > df['SMA_50'],
            df['EMA_Fast'] > df['EMA_Slow'],
            df['MACD'] > 0,
            df['MACD'] > df['MACD_Signal'],
            df['ADX'] > 25,
            df['RSI'] > 50,
            df['Close'] > df['Close'].shift(5)
        ]
        
        # Strong downtrend conditions
        short_conditions = [
            df['Close'] < df['SMA_20'],
            df['SMA_20'] < df['SMA_50'],
            df['EMA_Fast'] < df['EMA_Slow'],
            df['MACD'] < 0,
            df['MACD'] < df['MACD_Signal'],
            df['ADX'] > 25,
            df['RSI'] < 50,
            df['Close'] < df['Close'].shift(5)
        ]
        
        # Count satisfied conditions
        df['Long_Score'] = sum([cond.astype(int) for cond in long_conditions])
        df['Short_Score'] = sum([cond.astype(int) for cond in short_conditions])
        
        # Generate signals (need 6 out of 8 conditions)
        df['Long_Signal'] = (df['Long_Score'] >= 6) & (df['Long_Score'].shift(1) < 6)
        df['Short_Signal'] = (df['Short_Score'] >= 6) & (df['Short_Score'].shift(1) < 6)
        
        return df
    
    def multi_indicator_signals(self, df):
        """Multi-indicator confluence strategy"""
        # Score-based system
        df['Bull_Score'] = 0
        df['Bear_Score'] = 0
        
        # Add points for bullish conditions
        df.loc[df['Close'] > df['EMA_Fast'], 'Bull_Score'] += 1
        df.loc[df['EMA_Fast'] > df['EMA_Slow'], 'Bull_Score'] += 1
        df.loc[df['RSI'] > 50, 'Bull_Score'] += 1
        df.loc[df['MACD'] > df['MACD_Signal'], 'Bull_Score'] += 1
        df.loc[df['Stoch_K'] > 50, 'Bull_Score'] += 1
        df.loc[df['CCI'] > 0, 'Bull_Score'] += 1
        df.loc[df['Williams_R'] > -50, 'Bull_Score'] += 1
        df.loc[df['Volume_Ratio'] > 1.2, 'Bull_Score'] += 1
        df.loc[df['ADX'] > 25, 'Bull_Score'] += 1
        df.loc[df['Close'] > df['BB_Mid'], 'Bull_Score'] += 1
        
        # Add points for bearish conditions
        df.loc[df['Close'] < df['EMA_Fast'], 'Bear_Score'] += 1
        df.loc[df['EMA_Fast'] < df['EMA_Slow'], 'Bear_Score'] += 1
        df.loc[df['RSI'] < 50, 'Bear_Score'] += 1
        df.loc[df['MACD'] < df['MACD_Signal'], 'Bear_Score'] += 1
        df.loc[df['Stoch_K'] < 50, 'Bear_Score'] += 1
        df.loc[df['CCI'] < 0, 'Bear_Score'] += 1
        df.loc[df['Williams_R'] < -50, 'Bear_Score'] += 1
        df.loc[df['Volume_Ratio'] > 1.2, 'Bear_Score'] += 1
        df.loc[df['ADX'] > 25, 'Bear_Score'] += 1
        df.loc[df['Close'] < df['BB_Mid'], 'Bear_Score'] += 1
        
        # Generate signals (need 7 out of 10 points)
        df['Long_Signal'] = (df['Bull_Score'] >= 7) & (df['Bull_Score'].shift(1) < 7)
        df['Short_Signal'] = (df['Bear_Score'] >= 7) & (df['Bear_Score'].shift(1) < 7)
        
        return df
    
    def scalping_signals(self, df):
        """High-frequency scalping strategy"""
        # Very sensitive signals for quick trades
        long_conditions = [
            df['Close'] > df['EMA_Fast'],
            df['RSI'] > 45,
            df['RSI'] < 65,
            df['MACD_Hist'] > df['MACD_Hist'].shift(1),
            df['Stoch_K'] > df['Stoch_D'],
            df['Close'] > df['Close'].shift(1),
            df['Volume_Ratio'] > 0.8
        ]
        
        short_conditions = [
            df['Close'] < df['EMA_Fast'],
            df['RSI'] < 55,
            df['RSI'] > 35,
            df['MACD_Hist'] < df['MACD_Hist'].shift(1),
            df['Stoch_K'] < df['Stoch_D'],
            df['Close'] < df['Close'].shift(1),
            df['Volume_Ratio'] > 0.8
        ]
        
        # Count satisfied conditions
        df['Long_Score'] = sum([cond.astype(int) for cond in long_conditions])
        df['Short_Score'] = sum([cond.astype(int) for cond in short_conditions])
        
        # Generate signals (need 5 out of 7 conditions)
        df['Long_Signal'] = (df['Long_Score'] >= 5) & (df['Long_Score'].shift(1) < 5)
        df['Short_Signal'] = (df['Short_Score'] >= 5) & (df['Short_Score'].shift(1) < 5)
        
        return df
    
    def backtest(self, df, initial_balance=100000):
        """Comprehensive backtesting"""
        df = self.prepare_data(df)
        df = self.generate_signals(df)
        
        trades = []
        balance = initial_balance
        current_trade =         return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_points': total_points,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_win_points': avg_win_points,
            'avg_loss_points': avg_loss_points,
            'profit_factor': profit_factor,
            'max_drawdown': drawdown,
            'avg_trade_duration': avg_duration,
            'best_trade_pct': trades_df['pnl_percentage'].max() if len(trades_df) > 0 else 0,
            'worst_trade_pct': trades_df['pnl_percentage'].min() if len(trades_df) > 0 else 0,
            'buy_hold_return': buy_hold_return,
            'buy_hold_points': buy_hold_points,
            'strategy_vs_buyhold_pct': total_return - buy_hold_return,
            'strategy_vs_buyhold_points': total_points - buy_hold_points
        }

class StrategyOptimizer:
    def __init__(self, df, strategy_type, use_volume=True):
        self.df = df
        self.strategy_type = strategy_type
        self.use_volume = use_volume
    
    def get_parameter_ranges(self):
        """Parameter ranges for different strategies"""
        base_ranges = {
            'ema_fast': [5, 9, 12, 15],
            'ema_slow': [18, 21, 26, 30],
            'rsi_period': [10, 14, 18, 21],
            'stop_loss_pct': [1.0, 1.5, 2.0, 2.5, 3.0],
            'take_profit_pct': [3.0, 4.5, 6.0, 9.0, 12.0]
        }
        
        if self.strategy_type == "Mean Reversion":
            base_ranges.update({
                'rsi_oversold': [25, 30, 35],
                'rsi_overbought': [65, 70, 75],
                'bb_std': [1.5, 2.0, 2.5]
            })
        
        return base_ranges
    
    def optimize_parameters(self, max_combinations=50):
        """Optimize strategy parameters"""
        param_ranges = self.get_parameter_ranges()
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        # Generate combinations
        all_combinations = list(itertools.product(*values))
        
        if len(all_combinations) > max_combinations:
            # Randomly sample combinations
            import random
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        st.info(f"🔍 Optimizing {len(combinations)} parameter combinations...")
        
        progress_bar = st.progress(0)
        results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            try:
                strategy = AdvancedTradingStrategy(
                    strategy_type=self.strategy_type,
                    ema_fast=params.get('ema_fast', 9),
                    ema_slow=params.get('ema_slow', 21),
                    rsi_period=params.get('rsi_period', 14),
                    rsi_oversold=params.get('rsi_oversold', 30),
                    rsi_overbought=params.get('rsi_overbought', 70),
                    bb_std=params.get('bb_std', 2.0),
                    stop_loss_pct=params['stop_loss_pct'],
                    take_profit_pct=params['take_profit_pct'],
                    use_volume=self.use_volume
                )
                
                _, trades_df, performance = strategy.backtest(self.df)
                
                # Calculate optimization score
                score = 0
                if performance['total_return'] > 0:
                    score += 40
                if performance['win_rate'] >= 50:
                    score += 20
                elif performance['win_rate'] >= 40:
                    score += 10
                if performance['profit_factor'] >= 1.5:
                    score += 20
                elif performance['profit_factor'] >= 1.0:
                    score += 10
                if performance['total_trades'] >= 10:
                    score += 10
                if performance['strategy_vs_buyhold_pct'] > 0:
                    score += 10
                
                result = performance.copy()
                result['parameters'] = params
                result['score'] = score
                results.append(result)
                
            except Exception as e:
                results.append({
                    'score': 0, 'total_return': -999, 'parameters': params,
                    'error': str(e), 'total_trades': 0, 'win_rate': 0,
                    'profit_factor': 0, 'max_drawdown': 0
                })
            
            progress_bar.progress((i + 1) / len(combinations))
        
        progress_bar.empty()
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results

def load_data_yfinance(symbol, period="1y", interval="1d"):
    """Enhanced data loading with multiple timeframes"""
    try:
        # Indian market symbols
        indian_symbols = {
            'NIFTY': '^NSEI', 'BANKNIFTY': '^NSEBANK', 'SENSEX': '^BSESN',
            'NIFTY50': '^NSEI', 'NIFTYBANK': '^NSEBANK', 'FINNIFTY': '^CNXFIN'
        }
        
        yf_symbol = indian_symbols.get(symbol.upper(), symbol)
        ticker = yf.Ticker(yf_symbol)
        
        # Try to get data with error handling
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            st.error(f"No data found for {symbol}")
            return         with col1:
            return_color = "normal" if performance['total_return'] > 0 else "inverse"
            st.metric(
                "Total Return",
                f"{performance['total_return']:.2f}%",
                delta=f"{performance['strategy_vs_buyhold_pct']:+.2f}% vs B&H",
                delta_color="normal" if performance['strategy_vs_buyhold_pct'] > 0 else "inverse"
            )
        
        with col2:
            st.metric(
                "Total Trades",
                f"{performance['total_trades']}",
                delta=f"Win Rate: {performance['win_rate']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Profit Factor",
                f"{performance['profit_factor']:.2f}",
                delta=f"Avg Duration: {performance['avg_trade_duration']:.0f} bars"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{performance['max_drawdown']:.2f}%",
                delta=f"Total Points: {performance['total_points']:+.1f}"
            )
        
        # Strategy performance assessment
        st.subheader("🎯 Strategy Performance Assessment")
        
        score = 0
        assessment = []
        
        if performance['total_return'] > 0:
            score += 25
            assessment.append("✅ Positive Returns")
        else:
            assessment.append("❌ Negative Returns")
        
        if performance['total_trades'] >= 20:
            score += 20
            assessment.append("✅ Good Trade Frequency")
        elif performance['total_trades'] >= 10:
            score += 15
            assessment.append("⚠️ Moderate Trade Frequency")
        elif performance['total_trades'] >= 5:
            score += 10
            assessment.append("⚠️ Low Trade Frequency")
        else:
            assessment.append("❌ Very Low Trade Frequency")
        
        if performance['win_rate'] >= 60:
            score += 25
            assessment.append("✅ High Win Rate (≥60%)")
        elif performance['win_rate'] >= 50:
            score += 20
            assessment.append("✅ Good Win Rate (≥50%)")
        elif performance['win_rate'] >= 40:
            score += 15
            assessment.append("⚠️ Moderate Win Rate (≥40%)")
        else:
            assessment.append("❌ Low Win Rate (<40%)")
        
        if performance['profit_factor'] >= 2.0:
            score += 20
            assessment.append("✅ Excellent Profit Factor (≥2.0)")
        elif performance['profit_factor'] >= 1.5:
            score += 15
            assessment.append("✅ Good Profit Factor (≥1.5)")
        elif performance['profit_factor'] >= 1.0:
            score += 10
            assessment.append("⚠️ Break-even Profit Factor")
        else:
            assessment.append("❌ Poor Profit Factor (<1.0)")
        
        if performance['strategy_vs_buyhold_pct'] > 5:
            score += 10
            assessment.append("✅ Significantly Beats Buy & Hold")
        elif performance['strategy_vs_buyhold_pct'] > 0:
            score += 5
            assessment.append("✅ Beats Buy & Hold")
        else:
            assessment.append("❌ Underperforms Buy & Hold")
        
        # Display assessment
        if score >= 80:
            st.success(f"🏆 Excellent Strategy Performance ({score}/100)")
            st.success("🚀 **Ready for live trading!** This configuration shows strong performance.")
        elif score >= 60:
            st.warning(f"⚠️ Good Strategy Performance ({score}/100)")
            st.info("📊 **Consider optimization** to improve performance further.")
        elif score >= 40:
            st.info(f"📊 Fair Strategy Performance ({score}/100)")
            st.warning("🔧 **Optimization recommended** before live trading.")
        else:
            st.error(f"❌ Poor Strategy Performance ({score}/100)")
            st.error("🛠️ **Requires optimization** or different strategy/parameters.")
        
        for item in assessment:
            st.write(f"  {item}")
        
        # Detailed performance breakdown
        st.subheader("📊 Detailed Performance Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📈 Winning Trades:**")
            st.write(f"- Count: {performance['winning_trades']}")
            st.write(f"- Win Rate: {performance['win_rate']:.1f}%")
            st.write(f"- Avg Win: {performance['avg_win_pct']:.2f}%")
            st.write(f"- Avg Win Points: {performance['avg_win_points']:.1f}")
            st.write(f"- Best Trade: {performance['best_trade_pct']:.2f}%")
        
        with col2:
            st.write("**📉 Losing Trades:**")
            st.write(f"- Count: {performance['losing_trades']}")
            st.write(f"- Loss Rate: {100-performance['win_rate']:.1f}%")
            st.write(f"- Avg Loss: {performance['avg_loss_pct']:.2f}%")
            st.write(f"- Avg Loss Points: {performance['avg_loss_points']:.1f}")
            st.write(f"- Worst Trade: {performance['worst_trade_pct']:.2f}%")
        
        with col3:
            st.write("**⚡ Strategy Metrics:**")
            st.write(f"- Total Return: {performance['total_return']:.2f}%")
            st.write(f"- Buy & Hold: {performance['buy_hold_return']:.2f}%")
            st.write(f"- Outperformance: {performance['strategy_vs_buyhold_pct']:+.2f}%")
            st.write(f"- Profit Factor: {performance['profit_factor']:.2f}")
            st.write(f"- Max Drawdown: {performance['max_drawdown']:.2f}%")
        
        # Trading chart
        st.subheader("📈 Trading Chart & Signals")
        fig = create_advanced_chart(processed_df, trades_df, strategy_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        if not trades_df.empty:
            st.subheader("💼 Trade History")
            
            # Show last 20 trades
            recent_trades = trades_df.tail(20).copy()
            recent_trades['Entry Date'] = pd.to_datetime(recent_trades['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
            recent_trades['Exit Date'] = pd.to_datetime(recent_trades['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
            recent_trades['Entry Price'] = recent_trades['entry_price'].round(2)
            recent_trades['Exit Price'] = recent_trades['exit_price'].round(2)
            recent_trades['P&L %'] = recent_trades['pnl_percentage'].round(2)
            recent_trades['P&L Points'] = recent_trades['pnl_points'].round(1)
            recent_trades['Type'] = recent_trades['type']
            recent_trades['Duration'] = recent_trades['duration_bars'].astype(int)
            recent_trades['Exit Reason'] = recent_trades['exit_reason']
            recent_trades['Score'] = recent_trades['score'].astype(int)
            
            # Color coding
            def highlight_pnl(row):
                colors = []
                for col in row.index:
                    if col == 'P&L %' or col == 'P&L Points':
                        if row[col] > 0:
                            colors.append('background-color: rgba(34, 197, 94, 0.3)')
                        else:
                            colors.append('background-color: rgba(239, 68, 68, 0.3)')
                    else:
                        colors.append('')
                return colors
            
            display_cols = ['Entry Date', 'Type', 'Entry Price', 'Exit Price', 'P&L %', 'P&L Points', 'Duration', 'Score', 'Exit Reason']
            styled_df = recent_trades[display_cols].style.apply(highlight_pnl, axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Trade statistics
            st.subheader("📊 Trade Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Performance by Exit Reason:**")
                exit_stats = trades_df.groupby('exit_reason').agg({
                    'pnl_percentage': ['count', 'mean'],
                    'pnl_points': 'sum'
                }).round(2)
                
                for reason in exit_stats.index:
                    count = exit_stats.loc[reason, ('pnl_percentage', 'count')]
                    avg_pct = exit_stats.loc[reason, ('pnl_percentage', 'mean')]
                    total_points = exit_stats.loc[reason, ('pnl_points', 'sum')]
                    st.write(f"- **{reason}**: {count} trades, {avg_pct:.1f}% avg, {total_points:.1f} pts total")
            
            with col2:
                st.write("**⏱️ Performance by Duration:**")
                trades_df['duration_category'] = pd.cut(trades_df['duration_bars'], 
                    bins=[0, 5, 15, 50, 1000], labels=['Very Short', 'Short', 'Medium', 'Long'])
                
                duration_stats = trades_df.groupby('duration_category').agg({
                    'pnl_percentage': ['count', 'mean']
                }).round(2)
                
                for duration in duration_stats.index:
                    if pd.isna(duration):
                        continue
                    count = duration_stats.loc[duration, ('pnl_percentage', 'count')]
                    avg_pct = duration_stats.loc[duration, ('pnl_percentage', 'mean')]
                    st.write(f"- **{duration}**: {count} trades, {avg_pct:.1f}% avg")
        
        else:
            st.warning("⚠️ No trades generated!")
            st.info("**Try these solutions:**")
            st.info("- Switch to shorter timeframe (5m, 15m)")
            st.info("- Try different strategy (Scalping, Mean Reversion)")
            st.info("- Use longer data period")
            st.info("- Run optimization to find better parameters")
        
        # Strategy recommendations
        st.subheader("💡 Strategy Recommendations")
        
        if performance['total_return'] > 5 and performance['win_rate'] >= 50:
            st.success("🚀 **Excellent Results!** Strategy is ready for live trading.")
            st.success("💡 **Next Steps**: Start with small position sizes and monitor performance.")
        elif performance['total_trades'] < 10:
            st.warning("⚠️ **Increase Trade Frequency:**")
            st.info("- Switch to shorter timeframes (1m, 5m, 15m)")
            st.info("- Try Scalping or Mean Reversion strategies")
            st.info("- Reduce stop-loss percentage for more entries")
        elif performance['win_rate'] < 45:
            st.error("❌ **Improve Win Rate:**")
            st.info("- Try Trend Following strategy")
            st.info("- Use longer timeframes (1h, 4h, 1d)")
            st.info("- Run optimization to find better parameters")
        elif performance['profit_factor'] < 1.2:
            st.warning("⚠️ **Improve Risk-Reward:**")
            st.info("- Increase take-profit percentage")
            st.info("- Decrease stop-loss percentage")
            st.info("- Use Multi-Indicator strategy for better entries")
        else:
            st.info("📊 **Good Foundation!** Consider optimization for better results.")
    
    elif mode == "🎯 Optimization":
        st.header(f"🎯 {strategy_type} Strategy Optimization")
        
        if optimize_button:
            st.subheader("🔍 Running Parameter Optimization")
            
            optimizer = StrategyOptimizer(df, strategy_type, has_volume)
            
            with st.spinner(f"Optimizing {strategy_type} strategy parameters..."):
                results = optimizer.optimize_parameters(max_combinations=100)
            
            if results and len(results) > 0 and results[0].get('total_return', -999) != -999:
                st.success("✅ Optimization completed!")
                
                # Show optimization summary
                best_result = results[0]
                st.subheader("🏆 Best Configuration Found")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Return", f"{best_result['total_return']:.2f}%")
                with col2:
                    st.metric("Win Rate", f"{best_result['win_rate']:.1f}%")
                with col3:
                    st.metric("Trades", f"{best_result['total_trades']}")
                with col4:
                    st.metric("Score", f"{best_result['score']:.0f}/100")
                
                # Show top 5 results
                st.subheader("🏆 Top 5 Optimized Configurations")
                
                for i, result in enumerate(results[:5]):
                    if result.get('total_return', -999) == -999:
                        continue
                        
                    with st.expander(f"#{i+1} - Return: {result['total_return']:.2f}% | Win Rate: {result['win_rate']:.1f}% | Trades: {result['total_trades']} | Score: {result['score']:.0f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📊 Performance:**")
                            st.write(f"- Total Return: {result['total_return']:.2f}%")
                            st.write(f"- Win Rate: {result['win_rate']:.1f}%")
                            st.write(f"- Total Trades: {result['total_trades']}")
                            st.write(f"- Profit Factor: {result['profit_factor']:.2f}")
                            st.write(f"- Max Drawdown: {result['max_drawdown']:.2f}%")
                            st.write(f"- vs Buy & Hold: {result['strategy_vs_buyhold_pct']:+.2f}%")
                        
                        with col2:
                            st.write("**⚙️ Parameters:**")
                            params = result['parameters']
                            for key, value in params.items():
                                display_key = key.replace('_', ' ').title()
                                if isinstance(value, float):
                                    st.write(f"- {display_key}: {value:.2f}")
                                else:
                                    st.write(f"- {display_key}: {value}")
                
                # Optimization insights
                st.subheader("💡 Optimization Insights")
                
                # Analyze parameter patterns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 Best Parameter Ranges:**")
                    top_10 = results[:10]
                    
                    if len(top_10) > 1:
                        stop_losses = [r['parameters']['stop_loss_pct'] for r in top_10]
                        take_profits = [r['parameters']['take_profit_pct'] for r in top_10]
                        
                        st.write(f"- Stop Loss: {min(stop_losses):.1f}% - {max(stop_losses):.1f}%")
                        st.write(f"- Take Profit: {min(take_profits):.1f}% - {max(take_profits):.1f}%")
                        
                        if 'ema_fast' in results[0]['parameters']:
                            ema_fasts = [r['parameters']['ema_fast'] for r in top_10]
                            ema_slows = [r['parameters']['ema_slow'] for r in top_10]
                            st.write(f"- EMA Fast: {min(ema_fasts)} - {max(ema_fasts)}")
                            st.write(f"- EMA Slow: {min(ema_slows)} - {max(ema_slows)}")
                
                with col2:
                    st.write("**📈 Performance Improvement:**")
                    if len(results) > 1:
                        worst = min(results, key=lambda x: x.get('score', 0))
                        improvement = best_result['total_return'] - worst['total_return']
                        st.write(f"- Return Improvement: +{improvement:.2f}%")
                        
                        win_rate_improvement = best_result['win_rate'] - worst['win_rate']
                        st.write(f"- Win Rate Improvement: +{win_rate_improvement:.1f}%")
                        
                        trade_improvement = best_result['total_trades'] - worst['total_trades']
                        st.write(f"- Trade Frequency: +{trade_improvement} trades")
                
                # Use best parameters button
                if st.button("🎯 Apply Best Parameters", type="primary"):
                    st.session_state.optimized_params = best_result['parameters']
                    st.success("✅ Best parameters applied! Go to Backtesting to see optimized results.")
                    st.balloons()
                
            else:
                st.error("❌ Optimization failed to find profitable configurations.")
                st.error("**Try these solutions:**")
                st.error("- Use different strategy type")
                st.error("- Use different timeframe")
                st.error("- Get more historical data")
        
        else:
            st.info("👆 Click 'Optimize Strategy' to automatically find the best parameters.")
            
            # Show optimization preview
            st.subheader("🔧 What Will Be Optimized")
            
            param_ranges = StrategyOptimizer(df, strategy_type, has_volume).get_parameter_ranges()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Parameters to Optimize:**")
                for param, values in param_ranges.items():
                    param_name = param.replace('_', ' ').title()
                    st.write(f"- **{param_name}**: {values}")
            
            with col2:
                st.write("**🎯 Optimization Goals:**")
                st.write("- **Positive returns** (40 points)")
                st.write("- **Win rate ≥50%** (20 points)")
                st.write("- **Profit factor ≥1.5** (20 points)")
                st.write("- **Trade count ≥10** (10 points)")
                st.write("- **Beat buy & hold** (10 points)")
                st.write("- **Maximum score: 100 points**")
            
            total_combinations = 1
            for values in param_ranges.values():
                total_combinations *= len(values)
            
            st.info(f"🔍 **Will test up to {min(100, total_combinations)} parameter combinations**")
    
    else:  # Live Analysis
        st.header(f"🔍 {strategy_type} Live Analysis")
        
        with st.spinner("Processing live market data..."):
            processed_df = strategy.prepare_data(df)
            processed_df = strategy.generate_signals(processed_df)
        
        if len(processed_df) < 50:
            st.error("❌ Insufficient data for reliable analysis (need 50+ bars)")
            st.error("**Solutions:**")
            st.error("- Use longer time period")
            st.error("- Use shorter timeframe")
            return
        
        # Current market status
        latest = processed_df.iloc[-1]
        prev = processed_df.iloc[-2]
        
        st.subheader("📊 Current Market Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = latest['Close'] - prev['Close']
            price_change_pct = (price_change / prev['Close']) * 100
            st.metric(
                "Current Price",
                f"{latest['Close']:.2f}",
                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            rsi = latest.get('RSI', 50)
            rsi_status = "🟢 Normal" if 30 <= rsi <= 70 else "🔴 Extreme"
            st.metric(
                "RSI",
                f"{rsi:.1f}",
                delta=rsi_status
            )
        
        with col3:
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            macd_trend = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
            st.metric(
                "MACD Trend",
                f"{macd:.3f}",
                delta=macd_trend
            )
        
        with col4:
            vol_ratio = latest.get('Volume_Ratio', 1)
            vol_status = "🟢 High" if vol_ratio > 1.2 else "⚪ Normal" if vol_ratio > 0.8 else "🔴 Low"
            st.metric(
                "Volume Activity",
                f"{vol_ratio:.2f}x",
                delta=vol_status
            )
        
        # Strategy-specific indicators
        st.subheader(f"🎯 {strategy_type} Strategy Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if strategy_type in ["Momentum Breakout", "Trend Following", "Multi-Indicator"]:
            with col1:
                ema_fast_val = latest.get('EMA_Fast', 0)
                ema_trend = "🟢 Bullish" if latest['Close'] > ema_fast_val else "🔴 Bearish"
                st.metric(f"EMA {ema_fast}", f"{ema_fast_val:.2f}", delta=ema_trend)
            
            with col2:
                ema_slow_val = latest.get('EMA_Slow', 0)
                ema_alignment = "🟢 Aligned" if ema_fast_val > ema_slow_val else "🔴 Crossed"
                st.metric(f"EMA {ema_slow}", f"{ema_slow_val:.2f}", delta=ema_alignment)
        
        if strategy_type == "Mean Reversion":
            with col1:
                bb_upper = latest.get('BB_Upper', 0)
                st.metric("BB Upper", f"{bb_upper:.2f}")
            
            with col2:
                bb_lower = latest.get('BB_Lower', 0)
                st.metric("BB Lower", f"{bb_lower:.2f}")
        
        with col3:
            adx = latest.get('ADX', 0)
            adx_trend = "🟢 Trending" if adx > 25 else "⚪ Ranging"
            st.metric("ADX", f"{adx:.1f}", delta=adx_trend)
        
        with col4:
            if strategy_type == "Multi-Indicator":
                bull_score = latest.get('Bull_Score', 0)
                bear_score = latest.get('Bear_Score', 0)
                st.metric("Bull Score", f"{bull_score}/10")
                st.metric("Bear Score", f"{bear_score}/10")
            else:
                long_score = latest.get('Long_Score', 0)
                short_score = latest.get('Short_Score', 0)
                st.metric("Long Score", f"{long_score}/8")
                st.metric("Short Score", f"{short_score}/8")
        
        # Live signal detection
        st.subheader("🚨 Live Signal Detection")
        
        # Check recent signals (last 3 bars)
        recent_data = processed_df.tail(3)
        long_signals = recent_data[recent_data.get('Long_Signal', False)] if 'Long_Signal' in recent_data.columns else pd.DataFrame()
        short_signals = recent_data[recent_data.get('Short_Signal', False)] if 'Short_Signal' in recent_data.columns else pd.DataFrame()
        
        if not long_signals.empty or not short_signals.empty:
            if not long_signals.empty:
                signal = long_signals.iloc[-1]
                st.success("🟢 **LONG SIGNAL DETECTED!**")
                
                entry_price = signal['Close']
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
                take_profit = entry_price * (1 + take_profit_pct / 100)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📋 Trade Setup:**")
                    st.write(f"- **Entry:** {entry_price:.2f}")
                    st.write(f"- **Stop Loss:** {stop_loss:.2f} ({stop_loss_pct:.1f}%)")
                    st.write(f"- **Take Profit:** {take_profit:.2f} ({take_profit_pct:.1f}%)")
                    st.write(f"- **Risk:Reward:** 1:{take_profit_pct/stop_loss_pct:.1f}")
                    st.write(f"- **Risk Points:** {entry_price - stop_loss:.2f}")
                    st.write(f"- **Reward Points:** {take_profit - entry_price:.2f}")
                
                with col2:
                    st.write("**📊 Signal Quality:**")
                    st.write(f"- **Strategy:** {strategy_type}")
                    st.write(f"- **RSI:** {signal.get('RSI', 50):.1f}")
                    st.write(f"- **MACD:** {signal.get('MACD', 0):.3f}")
                    st.write(f"- **Volume Ratio:** {signal.get('Volume_Ratio', 1):.2f}x")
                    st.write(f"- **Signal Score:** {signal.get('Long_Score', 0)}")
                    st.write(f"- **Confidence:** {'High' if signal.get('Long_Score', 0) >= 6 else 'Medium'}")
            
            if not short_signals.empty:
                signal = short_signals.iloc[-1]
                st.error("🔴 **SHORT SIGNAL DETECTED!**")
                
                entry_price = signal['Close']
                stop_loss = entry_price * (1 + stop_loss_pct / 100)
                take_profit = entry_price * (1 - take_profit_pct / 100)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📋 Trade Setup:**")
                    st.write(f"- **Entry:** {entry_price:.2f}")
                    st.write(f"- **Stop Loss:** {stop_loss:.2f} ({stop_loss_pct:.1f}%)")
                    st.write(f"- **Take Profit:** {take_profit:.2f} ({take_profit_pct:.1f}%)")
                    st.write(f"- **Risk:Reward:** 1:{take_profit_pct/stop_loss_pct:.1f}")
                    st.write(f"- **Risk Points:** {stop_loss - entry_price:.2f}")
                    st.write(f"- **Reward Points:** {entry_price - take_profit:.2f}")
                
                with col2:
                    st.write("**📊 Signal Quality:**")
                    st.write(f"- **Strategy:** {strategy_type}")
                    st.write(f"- **RSI:** {signal.get('RSI', 50):.1f}")
                    st.write(f"- **MACD:** {signal.get('MACD', 0):.3f}")
                    st.write(f"- **Volume Ratio:** {signal.get('Volume_Ratio', 1):.2f}x")
                    st.write(f"- **Signal Score:** {signal.get('Short_Score', 0)}")
                    st.write(f"- **Confidence:** {'High' if signal.get('Short_Score', 0) >= 6 else 'Medium'}")
        
        else:
            st.info("⏳ **No Active Signals - Market Analysis**")
            
            # Show what's needed for signals
            st.write("**🔍 Signal Requirements Analysis:**")
            
            if strategy_type == "Momentum Breakout":
                required_score = 6
                current_long = latest.get('Long_Score', 0)
                current_short = latest.get('Short_Score', 0)
                
                st.write(f"**Long Signal Requirements:**")
                st.write(f"- Current Score: {current_long}/8 (need ≥{required_score})")
                st.write(f"**Short Signal Requirements:**")
                st.write(f"- Current Score: {current_short}/8 (need ≥{required_score})")
            
            elif strategy_type == "Mean Reversion":
                required_score = 4
                current_long = latest.get('Long_Score', 0)
                current_short = latest.get('Short_Score', 0)
                
                st.write(f"**Long Signal Requirements (Oversold):**")
                st.write(f"- Current Score: {current_long}/7 (need ≥{required_score})")
                st.write(f"**Short Signal Requirements (Overbought):**")
                st.write(f"- Current Score: {current_short}/7 (need ≥{required_score})")
            
            elif strategy_type == "Multi-Indicator":
                required_score = 7
                current_bull = latest.get('Bull_Score', 0)
                current_bear = latest.get('Bear_Score', 0)
                
                st.write(f"**Bull Signal Requirements:**")
                st.write(f"- Current Score: {current_bull}/10 (need ≥{required_score})")
                st.write(f"**Bear Signal Requirements:**")
                st.write(f"- Current Score: {current_bear}/10 (need ≥{required_score})")
            
            # Show individual conditions
            st.write("**📊 Current Market Conditions:**")
            
            conditions = []
            
            # Price vs EMAs
            if latest['Close'] > latest.get('EMA_Fast', 0):
                conditions.append("✅ Price > EMA Fast")
            else:
                conditions.append("❌ Price < EMA Fast")
            
            if latest.get('EMA_Fast', 0) > latest.get('EMA_Slow', 0):
                conditions.append("✅ EMA Fast > EMA Slow")
            else:
                conditions.append("❌ EMA Fast < EMA Slow")
            
            # RSI conditions
            rsi = latest.get('RSI', 50)
            if strategy_type == "Mean Reversion":
                if rsi < rsi_oversold:
                    conditions.append(f"✅ RSI Oversold ({rsi:.1f} < {rsi_oversold})")
                elif rsi > rsi_overbought:
                    conditions.append(f"✅ RSI Overbought ({rsi:.1f} > {rsi_overbought})")
                else:
                    conditions.append(f"❌ RSI Neutral ({rsi:.1f})")
            else:
                if 45 < rsi < 65:
                    conditions.append(f"✅ RSI Normal ({rsi:.1f})")
                else:
                    conditions.append(f"❌ RSI Extreme ({rsi:.1f})")
            
            # MACD
            if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
                conditions.append("✅ MACD Bullish")
            else:
                conditions.append("❌ MACD Bearish")
            
            # Volume
            if latest.get('Volume_Ratio', 1) > 1.1:
                conditions.append("✅ Volume Above Average")
            else:
                conditions.append("❌ Volume Below Average")
            
            # ADX
            if latest.get('ADX', 0) > 20:
                conditions.append("✅ Market Trending")
            else:
                conditions.append("❌ Market Ranging")
            
            for condition in conditions:
                st.write(f"  {condition}")
        
        # Live chart
        st.subheader("📈 Live Market Chart")
        fig = create_advanced_chart(processed_df, strategy_type=strategy_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Market timing advice
        st.subheader("💡 Trading Recommendations")
        
        # Calculate market readiness score
        market_score = 0
        scoring_factors = []
        
        # Price trend
        if latest['Close'] > latest.get('EMA_Fast', 0) > latest.get('EMA_Slow', 0):
            market_score += 25
            scoring_factors.append("✅ Strong Uptrend")
        elif latest['Close'] < latest.get('EMA_Fast', 0) < latest.get('EMA_Slow', 0):
            market_score += 25
            scoring_factors.append("✅ Strong Downtrend")
        else:
            scoring_factors.append("⚠️ Sideways Market")
        
        # RSI positioning
        rsi = latest.get('RSI', 50)
        if 30 < rsi < 70:
            market_score += 20
            scoring_factors.append("✅ RSI Normal Range")
        else:
            scoring_factors.append("⚠️ RSI Extreme")
        
        # Volume activity
        if latest.get('Volume_Ratio', 1) > 1.2:
            market_score += 20
            scoring_factors.append("✅ High Volume")
        elif latest.get('Volume_Ratio', 1) > 0.8:
            market_score += 10
            scoring_factors.append("⚪ Normal Volume")
        else:
            scoring_factors.append("🔴 Low Volume")
        
        # Trending vs ranging
        if latest.get('ADX', 0) > 25:
            market_score += 20
            scoring_factors.append("✅ Strong Trend")
        elif latest.get('ADX', 0) > 20:
            market_score += 10
            scoring_factors.append("⚪ Weak Trend")
        else:
            scoring_factors.append("🔴 Ranging Market")
        
        # MACD alignment
        if abs(latest.get('MACD', 0)) > abs(latest.get('MACD_Signal', 0)):
            market_score += 15
            scoring_factors.append("✅ MACD Momentum")
        else:
            scoring_factors.append("⚠️ MACD Weak")
        
        # Display market timing recommendation
        if market_score >= 80:
            st.success(f"🟢 **EXCELLENT TRADING CONDITIONS** ({market_score}/100)")
            st.success("🚀 **Recommendation:** Prime time for trading! Watch for signals closely.")
        elif market_score >= 60:
            st.warning(f"⚠️ **GOOD TRADING CONDITIONS** ({market_score}/100)")
            st.info("📊 **Recommendation:** Favorable conditions. Trade with standard position size.")
        elif market_score >= 40:
            st.info(f"🔵 **FAIR TRADING CONDITIONS** ({market_score}/100)")
            st.warning("💡 **Recommendation:** Cautious trading. Consider smaller positions.")
        else:
            st.error(f"🔴 **POOR TRADING CONDITIONS** ({market_score}/100)")
            st.error("⚠️ **Recommendation:** Avoid trading. Wait for better setup.")
        
        # Show scoring breakdown
        with st.expander("📊 Market Scoring Breakdown"):
            for factor in scoring_factors:
                st.write(f"• {factor}")
        
        # Strategy-specific advice
        st.subheader(f"🎯 {strategy_type} Strategy Guidance")
        
        if strategy_type == "Momentum Breakout":
            st.write("**🚀 Momentum Breakout Tips:**")
            st.write("- Look for price breaking above recent resistance with volume")
            st.write("- Best in trending markets (ADX > 25)")
            st.write("- Enter on EMA crossover confirmations")
            st.write("- Avoid during low volume periods")
            
        elif strategy_type == "Mean Reversion":
            st.write("**🎯 Mean Reversion Tips:**")
            st.write("- Wait for RSI < 30 (oversold) for longs")
            st.write("- Wait for RSI > 70 (overbought) for shorts")
            st.write("- Works best in ranging markets (ADX < 25)")
            st.write("- Use Bollinger Band touches as confirmation")
            
        elif strategy_type == "Trend Following":
            st.write("**📈 Trend Following Tips:**")
            st.write("- Only trade in direction of major trend")
            st.write("- Wait for pullbacks to EMAs for better entries")
            st.write("- Requires strong trending conditions (ADX > 25)")
            st.write("- Be patient - fewer but higher quality trades")
            
        elif strategy_type == "Multi-Indicator":
            st.write("**📊 Multi-Indicator Tips:**")
            st.write("- Wait for confluence of multiple indicators")
            st.write("- Higher scores = higher probability trades")
            st.write("- More selective but generally more accurate")
            st.write("- Good for all market conditions")
            
        elif strategy_type == "Scalping":
            st.write("**⚡ Scalping Tips:**")
            st.write("- Use shorter timeframes (1m, 5m)")
            st.write("- Quick entries and exits")
            st.write("- Requires active monitoring")
            st.write("- Best during high volume periods")
    
    # Footer with system info
    st.markdown("---")
    st.markdown("### 🚀 Pure Pandas Trading System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**✅ No Dependencies:**")
        st.markdown("- Pure pandas + numpy")
        st.markdown("- No TA-Lib required")
        st.markdown("- Works everywhere")
        st.markdown("- Easy deployment")
    
    with col2:
        st.markdown("**📊 Complete Features:**")
        st.markdown("- All timeframes (1m-1mo)")
        st.markdown("- 5 proven strategies")
        st.markdown("- Auto-optimization")
        st.markdown("- Live signal detection")
    
    with col3:
        st.markdown("**🎯 Built for Results:**")
        st.markdown("- High trade frequency")
        st.markdown("- Positive returns focus")
        st.markdown("- Proper risk management")
        st.markdown("- Real-time analysis")
    
    # Current configuration summary
    if st.session_state.data_loaded:
        st.markdown("**📋 Current Configuration:**")
        config_info = f"Strategy: {strategy_type} | Timeframe: {interval} | Period: {period} | "
        config_info += f"Risk: {stop_loss_pct:.1f}% / {take_profit_pct:.1f}% | "
        config_info += f"Records: {len(st.session_state.df)}"
        st.markdown(f"*{config_info}*")
    
    st.markdown("⚠️ *This system generates signals based on technical analysis. Always use proper risk management and never risk more than you can afford to lose.*")

if __name__ == "__main__":
    main(), False
        
        # Process data
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df['Date'] = df['Datetime']
            df = df.drop('Datetime', axis=1)
        
        # Ensure OHLCV columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Missing OHLCV data")
            return None, False
        
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Convert to numeric and clean
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if df.empty:
            st.error("No valid data after cleaning")
            return None, False
        
        # Check volume validity
        has_volume = df['Volume'].sum() > 0 and df['Volume'].std() > 0
        
        return df, has_volume
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, False

def load_data_csv(uploaded_file):
    """Load CSV with enhanced error handling"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Find date column
        date_columns = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'timestamp'])]
        if date_columns:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
            df = df.set_index(date_columns[0])
            df = df.sort_index()
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Found: {list(df.columns)}")
            return None, False
        
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        # Clean and validate
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        has_volume = df['Volume'].sum() > 0 and df['Volume'].std() > 0
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']], has_volume
        
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None, False

def create_advanced_chart(df, trades_df=None, strategy_type=""):
    """Create comprehensive trading chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'Price Chart - {strategy_type}',
            'RSI & Stochastic',
            'MACD',
            'Volume & Indicators'
        ),
        row_heights=[0.5, 0.2, 0.2, 0.1]
    )
    
    # 1. Price chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name="Price"
    ), row=1, col=1)
    
    # Add EMAs
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_Fast'], mode='lines',
            name='EMA Fast', line=dict(color='orange', width=1)
        ), row=1, col=1)
    
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_Slow'], mode='lines',
            name='EMA Slow', line=dict(color='blue', width=1)
        ), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'], mode='lines',
            name='BB Upper', line=dict(color='gray', width=1, dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'], mode='lines',
            name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
        ), row=1, col=1)
    
    # Entry signals
    if trades_df is not None and not trades_df.empty:
        long_entries = trades_df[trades_df['type'] == 'LONG']
        short_entries = trades_df[trades_df['type'] == 'SHORT']
        
        if not long_entries.empty:
            fig.add_trace(go.Scatter(
                x=long_entries['entry_date'], y=long_entries['entry_price'],
                mode='markers', name='Long Entry',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ), row=1, col=1)
        
        if not short_entries.empty:
            fig.add_trace(go.Scatter(
                x=short_entries['entry_date'], y=short_entries['entry_price'],
                mode='markers', name='Short Entry',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ), row=1, col=1)
    
    # 2. RSI & Stochastic
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], mode='lines',
            name='RSI', line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", row=2, col=1)
    
    if 'Stoch_K' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Stoch_K'], mode='lines',
            name='Stoch %K', line=dict(color='orange')
        ), row=2, col=1)
        
        if 'Stoch_D' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Stoch_D'], mode='lines',
                name='Stoch %D', line=dict(color='red')
            ), row=2, col=1)
    
    # 3. MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], mode='lines',
            name='MACD', line=dict(color='blue')
        ), row=3, col=1)
        
        if 'MACD_Signal' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_Signal'], mode='lines',
                name='MACD Signal', line=dict(color='red')
            ), row=3, col=1)
        
        if 'MACD_Hist' in df.columns:
            fig.add_trace(go.Bar(
                x=df.index, y=df['MACD_Hist'],
                name='MACD Histogram', marker_color='gray'
            ), row=3, col=1)
        
        fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)
    
    # 4. Volume
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            name='Volume', marker_color='lightblue'
        ), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"Advanced Trading System Analysis - {strategy_type}",
        height=800,
        showlegend=True,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    st.title("🚀 Advanced Trading System - Pure Pandas")
    st.markdown("**High-Frequency Profitable Strategies** using only pandas/numpy (no TA-Lib required)")
    
    # Initialize session state
    for key in ['data_loaded', 'df', 'has_volume', 'optimized_params', 'last_results']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'data_loaded' else False
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Strategy Configuration")
    
    # Strategy selection
    st.sidebar.subheader("🎯 Strategy Selection")
    strategy_type = st.sidebar.selectbox(
        "Trading Strategy:",
        [
            "Momentum Breakout",
            "Mean Reversion", 
            "Trend Following",
            "Multi-Indicator",
            "Scalping"
        ],
        index=0,
        help="Choose the trading strategy type"
    )
    
    # Show strategy description
    strategy_descriptions = {
        "Momentum Breakout": "High-frequency trend catching with EMA crossovers and momentum",
        "Mean Reversion": "Buy oversold, sell overbought using RSI and Bollinger Bands",
        "Trend Following": "Ride strong trends with multiple confirmations",
        "Multi-Indicator": "Confluence-based signals using 10 different indicators",
        "Scalping": "Very high frequency with quick entries and exits"
    }
    
    st.sidebar.info(f"📊 {strategy_descriptions[strategy_type]}")
    
    # Data source
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio("Select Source:", ["Yahoo Finance", "File Upload"], index=0)
    
    # Enhanced data loading options
    uploaded_file = None
    symbol = ""
    period = "1y"
    interval = "1d"
    
    if data_source == "File Upload":
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
    else:
        st.sidebar.subheader("🏛️ Market & Symbol")
        market_type = st.sidebar.selectbox(
            "Market Type:",
            ["Indian Indices", "Indian Stocks", "US Stocks", "Crypto", "Forex", "Custom"]
        )
        
        if market_type == "Indian Indices":
            symbol = st.sidebar.selectbox(
                "Select Index:",
                ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"],
                index=0
            )
        elif market_type == "Indian Stocks":
            popular_stocks = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
            symbol_choice = st.sidebar.selectbox("Popular Stocks:", popular_stocks + ["Custom"])
            if symbol_choice == "Custom":
                symbol = st.sidebar.text_input("Enter Stock Symbol:", "")
            else:
                symbol = symbol_choice
        elif market_type == "US Stocks":
            popular_us = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
            symbol_choice = st.sidebar.selectbox("Popular US Stocks:", popular_us + ["Custom"])
            if symbol_choice == "Custom":
                symbol = st.sidebar.text_input("Enter US Stock Symbol:", "")
            else:
                symbol = symbol_choice
        elif market_type == "Crypto":
            crypto_pairs = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "DOGE-USD"]
            symbol = st.sidebar.selectbox("Crypto Pairs:", crypto_pairs)
        elif market_type == "Forex":
            forex_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
            symbol = st.sidebar.selectbox("Forex Pairs:", forex_pairs)
        else:
            symbol = st.sidebar.text_input("Custom Symbol:", "")
        
        # Enhanced period options
        st.sidebar.subheader("📅 Time Period")
        period_type = st.sidebar.radio("Period Type:", ["Predefined", "Custom Days"])
        
        if period_type == "Predefined":
            period = st.sidebar.selectbox(
                "Period:",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "ytd", "max"],
                index=5
            )
        else:
            custom_days = st.sidebar.number_input("Custom Days:", min_value=1, max_value=3650, value=365)
            period = f"{custom_days}d"
        
        # Enhanced interval options
        st.sidebar.subheader("⏱️ Timeframe")
        interval_category = st.sidebar.selectbox(
            "Interval Category:",
            ["Intraday", "Daily/Weekly", "Custom"]
        )
        
        if interval_category == "Intraday":
            interval = st.sidebar.selectbox(
                "Intraday Intervals:",
                ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
                index=2,
                help="Higher frequency = more trades"
            )
        elif interval_category == "Daily/Weekly":
            interval = st.sidebar.selectbox(
                "Daily/Weekly Intervals:",
                ["1d", "5d", "1wk", "1mo", "3mo"],
                index=0
            )
        else:
            interval = st.sidebar.text_input("Custom Interval (e.g., 4h):", "1h")
    
    # Strategy parameters
    st.sidebar.subheader("⚙️ Strategy Parameters")
    
    if strategy_type in ["Momentum Breakout", "Trend Following", "Multi-Indicator"]:
        ema_fast = st.sidebar.slider("EMA Fast", 5, 20, 9)
        ema_slow = st.sidebar.slider("EMA Slow", 15, 50, 21)
    else:
        ema_fast, ema_slow = 9, 21
    
    if strategy_type == "Mean Reversion":
        rsi_oversold = st.sidebar.slider("RSI Oversold", 20, 35, 30)
        rsi_overbought = st.sidebar.slider("RSI Overbought", 65, 80, 70)
    else:
        rsi_oversold, rsi_overbought = 30, 70
    
    # Risk management
    st.sidebar.subheader("🛡️ Risk Management")
    stop_loss_pct = st.sidebar.slider("Stop Loss %", 0.5, 5.0, 2.0, 0.1)
    take_profit_pct = st.sidebar.slider("Take Profit %", 2.0, 15.0, 6.0, 0.5)
    
    st.sidebar.info(f"📊 Risk:Reward = 1:{take_profit_pct/stop_loss_pct:.1f}")
    
    # Load data button
    fetch_button = st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True)
    
    # Optimize button
    optimize_button = st.sidebar.button("🎯 Optimize Strategy", type="secondary", use_container_width=True)
    
    # Data status
    if st.session_state.data_loaded:
        volume_status = "✅ With Volume" if st.session_state.has_volume else "📊 Price Only"
        st.sidebar.success(f"✅ Data loaded: {volume_status}")
        if st.session_state.df is not None:
            st.sidebar.info(f"📊 Records: {len(st.session_state.df)}")
            data_range = f"{st.session_state.df.index[0].strftime('%Y-%m-%d')} to {st.session_state.df.index[-1].strftime('%Y-%m-%d')}"
            st.sidebar.info(f"📅 Range: {data_range}")
    else:
        st.sidebar.info("👆 Configure and load data")
    
    # Mode selection
    st.sidebar.subheader("📈 Analysis Mode")
    mode = st.sidebar.radio("Select Mode:", ["📊 Backtesting", "🎯 Optimization", "🔍 Live Analysis"])
    
    # Handle data loading
    if fetch_button or (data_source == "File Upload" and uploaded_file is not None):
        if data_source == "File Upload" and uploaded_file is not None:
            with st.spinner("Loading CSV data..."):
                df, has_volume = load_data_csv(uploaded_file)
        else:
            if symbol:
                with st.spinner(f"Loading {symbol} data ({period}, {interval})..."):
                    df, has_volume = load_data_yfinance(symbol, period, interval)
            else:
                st.error("Please enter a symbol")
                return
        
        if df is not None and not df.empty:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.has_volume = has_volume
            st.session_state.optimized_params = None
            
            volume_msg = "with volume data" if has_volume else "price-only mode"
            st.success(f"✅ Loaded {len(df)} records in {volume_msg}")
            st.info(f"📅 Period: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.session_state.data_loaded = False
            return
    
    # Check data availability
    if not st.session_state.data_loaded:
        st.info("👆 Configure settings and load data to start analysis")
        
        # Show benefits of no external dependencies
        st.markdown("## ✅ Pure Pandas Implementation Benefits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 No External Dependencies")
            st.markdown("**✅ Works everywhere:**")
            st.markdown("- No TA-Lib installation needed")
            st.markdown("- No pandas-ta required")
            st.markdown("- Only uses pandas + numpy")
            st.markdown("- Compatible with all environments")
            
            st.markdown("**📊 All Indicators Built-in:**")
            st.markdown("- RSI, MACD, Bollinger Bands")
            st.markdown("- Stochastic, Williams %R, CCI")
            st.markdown("- EMAs, SMAs, ATR, ADX")
            st.markdown("- Custom volume indicators")
        
        with col2:
            st.markdown("### 🚀 High Performance Strategies")
            st.markdown("**🎯 Strategy Highlights:**")
            st.markdown("- **Momentum Breakout**: 6/8 conditions")
            st.markdown("- **Mean Reversion**: 4/7 conditions")
            st.markdown("- **Trend Following**: 6/8 conditions")
            st.markdown("- **Multi-Indicator**: 7/10 points")
            st.markdown("- **Scalping**: 5/7 conditions")
            
            st.markdown("**⚡ More Trades Expected:**")
            st.markdown("- Relaxed entry conditions")
            st.markdown("- Multiple strategy types")
            st.markdown("- High frequency capability")
            st.markdown("- All timeframes supported")
        
        # Show timeframe examples
        st.markdown("### ⏱️ Complete Timeframe Support")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 Intraday Trading:**")
            st.markdown("- 1m, 2m, 5m (scalping)")
            st.markdown("- 15m, 30m (day trading)")
            st.markdown("- 60m, 90m (swing intraday)")
            st.markdown("- **Best for**: High frequency")
        
        with col2:
            st.markdown("**📊 Swing Trading:**")
            st.markdown("- 1d (daily swings)")
            st.markdown("- 1wk (weekly swings)")
            st.markdown("- 1mo (monthly trends)")
            st.markdown("- **Best for**: Medium term")
        
        with col3:
            st.markdown("**🎯 Flexible Periods:**")
            st.markdown("- 1d to 10+ years")
            st.markdown("- Custom day periods")
            st.markdown("- YTD, Max available")
            st.markdown("- **Best for**: Any timeframe")
        
        return
    
    df = st.session_state.df
    has_volume = st.session_state.has_volume
    
    # Initialize strategy with current parameters
    strategy = AdvancedTradingStrategy(
        strategy_type=strategy_type,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        use_volume=has_volume
    )
    
    # Analysis based on mode
    if mode == "📊 Backtesting":
        st.header(f"📊 {strategy_type} Strategy Backtesting")
        
        # Show current parameters
        st.subheader("⚙️ Current Configuration")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Strategy", strategy_type)
        with col2:
            st.metric("Stop Loss", f"{stop_loss_pct:.1f}%")
        with col3:
            st.metric("Take Profit", f"{take_profit_pct:.1f}%")
        with col4:
            st.metric("Risk:Reward", f"1:{take_profit_pct/stop_loss_pct:.1f}")
        with col5:
            timeframe_text = f"{interval} / {period}"
            st.metric("Timeframe", timeframe_text)
        
        with st.spinner("Running comprehensive backtest..."):
            processed_df, trades_df, performance = strategy.backtest(df)
            st.session_state.last_results = (processed_df, trades_df, performance)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            return_color = "normal" if performance['total_return'] > 0 else "inverse"
            st.metric(
                "Total
        
        # Buy and hold comparison
        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        buy_hold_points = df['Close'].iloc[-1] - df['Close'].iloc[0]
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Entry signals
            if current_row['Long_Signal'] or current_row['Short_Signal']:
                # Close existing opposite trade
                if current_trade:
                    if ((current_row['Long_Signal'] and current_trade['type'] == 'SHORT') or 
                        (current_row['Short_Signal'] and current_trade['type'] == 'LONG')):
                        
                        exit_price = current_row['Close']
                        pnl_pct = self.calculate_pnl_percentage(current_trade, exit_price)
                        pnl_points = self.calculate_pnl_points(current_trade, exit_price)
                        
                        current_trade.update({
                            'exit_price': exit_price,
                            'exit_date': current_row.name,
                            'exit_reason': 'Opposite Signal',
                            'pnl_points': pnl_points,
                            'pnl_percentage': pnl_pct,
                            'duration_bars': i - current_trade['entry_index']
                        })
                        trades.append(current_trade)
                        current_trade = None
                
                # Open new trade
                if not current_trade:
                    trade_type = 'LONG' if current_row['Long_Signal'] else 'SHORT'
                    entry_price = current_row['Close']
                    
                    # Calculate dynamic stops based on ATR
                    atr_multiplier = 1.5
                    dynamic_stop = max(self.stop_loss_pct, (current_row.get('ATR', 0) / entry_price * 100 * atr_multiplier))
                    dynamic_target = dynamic_stop * (self.take_profit_pct / self.stop_loss_pct)
                    
                    if trade_type == 'LONG':
                        stop_loss = entry_price * (1 - dynamic_stop / 100)
                        take_profit = entry_price * (1 + dynamic_target / 100)
                    else:
                        stop_loss = entry_price * (1 + dynamic_stop / 100)
                        take_profit = entry_price * (1 - dynamic_target / 100)
                    
                    current_trade = {
                        'entry_date': current_row.name,
                        'entry_price': entry_price,
                        'entry_index': i,
                        'type': trade_type,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'stop_pct': dynamic_stop,
                        'target_pct': dynamic_target,
                        'rsi': current_row.get('RSI', 50),
                        'macd': current_row.get('MACD', 0),
                        'score': current_row.get('Long_Score' if trade_type == 'LONG' else 'Short_Score', 0)
                    }
            
            # Check for stop/target hits
            if current_trade:
                hit_stop = False
                hit_target = False
                exit_price = 0
                
                if current_trade['type'] == 'LONG':
                    if current_row['Low'] <= current_trade['stop_loss']:
                        hit_stop = True
                        exit_price = current_trade['stop_loss']
                    elif current_row['High'] >= current_trade['take_profit']:
                        hit_target = True
                        exit_price = current_trade['take_profit']
                else:  # SHORT
                    if current_row['High'] >= current_trade['stop_loss']:
                        hit_stop = True
                        exit_price = current_trade['stop_loss']
                    elif current_row['Low'] <= current_trade['take_profit']:
                        hit_target = True
                        exit_price = current_trade['take_profit']
                
                if hit_stop or hit_target:
                    pnl_pct = self.calculate_pnl_percentage(current_trade, exit_price)
                    pnl_points = self.calculate_pnl_points(current_trade, exit_price)
                    
                    current_trade.update({
                        'exit_price': exit_price,
                        'exit_date': current_row.name,
                        'exit_reason': 'Stop Loss Hit' if hit_stop else 'Take Profit Hit',
                        'pnl_points': pnl_points,
                        'pnl_percentage': pnl_pct,
                        'duration_bars': i - current_trade['entry_index']
                    })
                    trades.append(current_trade)
                    current_trade = None
        
        # Close any remaining trade
        if current_trade:
            exit_price = df['Close'].iloc[-1]
            pnl_pct = self.calculate_pnl_percentage(current_trade, exit_price)
            pnl_points = self.calculate_pnl_points(current_trade, exit_price)
            
            current_trade.update({
                'exit_price': exit_price,
                'exit_date': df.index[-1],
                'exit_reason': 'End of Data',
                'pnl_points': pnl_points,
                'pnl_percentage': pnl_pct,
                'duration_bars': len(df) - current_trade['entry_index']
            })
            trades.append(current_trade)
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        performance = self.calculate_performance_metrics(trades_df, initial_balance, buy_hold_return, buy_hold_points)
        
        return df, trades_df, performance
    
    def calculate_pnl_points(self, trade, exit_price):
        """Calculate P&L in points"""
        if trade['type'] == 'LONG':
            return exit_price - trade['entry_price']
        else:
            return trade['entry_price'] - exit_price
    
    def calculate_pnl_percentage(self, trade, exit_price):
        """Calculate P&L as percentage"""
        if trade['type'] == 'LONG':
            return ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
        else:
            return ((trade['entry_price'] - exit_price) / trade['entry_price']) * 100
    
    def calculate_performance_metrics(self, trades_df, initial_balance, buy_hold_return, buy_hold_points):
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_return': 0, 'total_points': 0,
                'avg_win_pct': 0, 'avg_loss_pct': 0, 'avg_win_points': 0, 'avg_loss_points': 0,
                'profit_factor': 0, 'max_drawdown': 0, 'avg_trade_duration': 0,
                'best_trade_pct': 0, 'worst_trade_pct': 0,
                'buy_hold_return': buy_hold_return, 'buy_hold_points': buy_hold_points,
                'strategy_vs_buyhold_pct': -buy_hold_return,
                'strategy_vs_buyhold_points': -buy_hold_points
            }
        
        winning_trades = trades_df[trades_df['pnl_percentage'] > 0]
        losing_trades = trades_df[trades_df['pnl_percentage'] <= 0]
        
        total_return = trades_df['pnl_percentage'].sum()
        total_points = trades_df['pnl_points'].sum()
        win_rate = len(winning_trades) / len(trades_df) * 100
        
        avg_win_pct = winning_trades['pnl_percentage'].mean() if len(winning_trades) > 0 else 0
        avg_loss_pct = losing_trades['pnl_percentage'].mean() if len(losing_trades) > 0 else 0
        avg_win_points = winning_trades['pnl_points'].mean() if len(winning_trades) > 0 else 0
        avg_loss_points = losing_trades['pnl_points'].mean() if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades['pnl_points'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl_points'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Simple drawdown calculation
        cumulative_returns = trades_df['pnl_percentage'].cumsum()
        peak = cumulative_returns.expanding().max()
        drawdown = ((peak - cumulative_returns) / peak * 100).max() if len(cumulative_returns) > 0 else 0
        
        avg_duration = trades_df['duration_bars'].mean() if len(trades_df) > 0 else 0
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning
