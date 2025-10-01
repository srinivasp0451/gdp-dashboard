import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Trading System", page_icon="🚀", layout="wide")

class TechnicalIndicators:
    """Pure pandas/numpy implementation of technical indicators"""
    
    @staticmethod
    def ema(data, window):
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def atr(high, low, close, window=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

class TradingStrategy:
    def __init__(self, strategy_type="Momentum", ema_fast=9, ema_slow=21, rsi_period=14,
                 stop_loss_pct=2.0, take_profit_pct=6.0):
        self.strategy_type = strategy_type
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.ti = TechnicalIndicators()
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        df = df.copy()
        df = df.sort_index()
        
        # Moving averages
        df['EMA_Fast'] = self.ti.ema(df['Close'], self.ema_fast)
        df['EMA_Slow'] = self.ti.ema(df['Close'], self.ema_slow)
        df['SMA_20'] = self.ti.sma(df['Close'], 20)
        df['SMA_50'] = self.ti.sma(df['Close'], 50)
        
        # RSI
        df['RSI'] = self.ti.rsi(df['Close'], self.rsi_period)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.ti.macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = self.ti.bollinger_bands(df['Close'])
        
        # ATR
        df['ATR'] = self.ti.atr(df['High'], df['Low'], df['Close'])
        
        # Volume analysis
        if df['Volume'].sum() > 0:
            df['Volume_SMA'] = self.ti.sma(df['Volume'], 20)
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = 1.0
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change() * 100
        df['Momentum'] = df['Close'].diff(5)
        
        return df.fillna(method='bfill')
    
    def generate_signals(self, df):
        """Generate trading signals"""
        df = self.calculate_indicators(df)
        
        if self.strategy_type == "Momentum":
            # Momentum strategy - simpler conditions for more trades
            df['Long_Cond1'] = df['Close'] > df['EMA_Fast']
            df['Long_Cond2'] = df['EMA_Fast'] > df['EMA_Slow']
            df['Long_Cond3'] = df['RSI'] > 40
            df['Long_Cond4'] = df['RSI'] < 80
            df['Long_Cond5'] = df['MACD'] > df['MACD_Signal']
            
            df['Short_Cond1'] = df['Close'] < df['EMA_Fast']
            df['Short_Cond2'] = df['EMA_Fast'] < df['EMA_Slow']
            df['Short_Cond3'] = df['RSI'] < 60
            df['Short_Cond4'] = df['RSI'] > 20
            df['Short_Cond5'] = df['MACD'] < df['MACD_Signal']
            
            df['Long_Score'] = (df['Long_Cond1'].astype(int) + df['Long_Cond2'].astype(int) + 
                              df['Long_Cond3'].astype(int) + df['Long_Cond4'].astype(int) + 
                              df['Long_Cond5'].astype(int))
            
            df['Short_Score'] = (df['Short_Cond1'].astype(int) + df['Short_Cond2'].astype(int) + 
                               df['Short_Cond3'].astype(int) + df['Short_Cond4'].astype(int) + 
                               df['Short_Cond5'].astype(int))
            
            # Signal when score >= 4 out of 5
            df['Long_Signal'] = (df['Long_Score'] >= 4) & (df['Long_Score'].shift(1) < 4)
            df['Short_Signal'] = (df['Short_Score'] >= 4) & (df['Short_Score'].shift(1) < 4)
        
        elif self.strategy_type == "Mean Reversion":
            # Mean reversion - buy oversold, sell overbought
            df['Long_Cond1'] = df['RSI'] < 35
            df['Long_Cond2'] = df['Close'] < df['BB_Lower']
            df['Long_Cond3'] = df['Price_Change'] < -1
            
            df['Short_Cond1'] = df['RSI'] > 65
            df['Short_Cond2'] = df['Close'] > df['BB_Upper']
            df['Short_Cond3'] = df['Price_Change'] > 1
            
            df['Long_Score'] = (df['Long_Cond1'].astype(int) + df['Long_Cond2'].astype(int) + 
                              df['Long_Cond3'].astype(int))
            
            df['Short_Score'] = (df['Short_Cond1'].astype(int) + df['Short_Cond2'].astype(int) + 
                               df['Short_Cond3'].astype(int))
            
            # Signal when score >= 2 out of 3
            df['Long_Signal'] = (df['Long_Score'] >= 2) & (df['Long_Score'].shift(1) < 2)
            df['Short_Signal'] = (df['Short_Score'] >= 2) & (df['Short_Score'].shift(1) < 2)
        
        elif self.strategy_type == "Trend":
            # Trend following
            df['Long_Cond1'] = df['Close'] > df['SMA_20']
            df['Long_Cond2'] = df['SMA_20'] > df['SMA_50']
            df['Long_Cond3'] = df['EMA_Fast'] > df['EMA_Slow']
            df['Long_Cond4'] = df['MACD'] > 0
            
            df['Short_Cond1'] = df['Close'] < df['SMA_20']
            df['Short_Cond2'] = df['SMA_20'] < df['SMA_50']
            df['Short_Cond3'] = df['EMA_Fast'] < df['EMA_Slow']
            df['Short_Cond4'] = df['MACD'] < 0
            
            df['Long_Score'] = (df['Long_Cond1'].astype(int) + df['Long_Cond2'].astype(int) + 
                              df['Long_Cond3'].astype(int) + df['Long_Cond4'].astype(int))
            
            df['Short_Score'] = (df['Short_Cond1'].astype(int) + df['Short_Cond2'].astype(int) + 
                               df['Short_Cond3'].astype(int) + df['Short_Cond4'].astype(int))
            
            # Signal when score >= 3 out of 4
            df['Long_Signal'] = (df['Long_Score'] >= 3) & (df['Long_Score'].shift(1) < 3)
            df['Short_Signal'] = (df['Short_Score'] >= 3) & (df['Short_Score'].shift(1) < 3)
        
        return df
    
    def backtest(self, df):
        """Enhanced backtesting with proper trade tracking"""
        df = self.generate_signals(df)
        
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        entry_type = None
        stop_loss = 0
        take_profit = 0
        
        # Buy and hold metrics
        initial_price = df['Close'].iloc[0]
        final_price = df['Close'].iloc[-1]
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        buy_hold_points = final_price - initial_price
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            
            # Check for exit conditions first
            if position is not None:
                exit_triggered = False
                exit_price = 0
                exit_reason = ""
                
                if position == 'LONG':
                    if current['Low'] <= stop_loss:
                        exit_triggered = True
                        exit_price = stop_loss
                        exit_reason = "Stop Loss"
                    elif current['High'] >= take_profit:
                        exit_triggered = True
                        exit_price = take_profit
                        exit_reason = "Take Profit"
                    elif current['Short_Signal']:
                        exit_triggered = True
                        exit_price = current['Close']
                        exit_reason = "Opposite Signal"
                
                elif position == 'SHORT':
                    if current['High'] >= stop_loss:
                        exit_triggered = True
                        exit_price = stop_loss
                        exit_reason = "Stop Loss"
                    elif current['Low'] <= take_profit:
                        exit_triggered = True
                        exit_price = take_profit
                        exit_reason = "Take Profit"
                    elif current['Long_Signal']:
                        exit_triggered = True
                        exit_price = current['Close']
                        exit_reason = "Opposite Signal"
                
                if exit_triggered:
                    # Calculate P&L
                    if position == 'LONG':
                        pnl_points = exit_price - entry_price
                        pnl_pct = (pnl_points / entry_price) * 100
                    else:
                        pnl_points = entry_price - exit_price
                        pnl_pct = (pnl_points / entry_price) * 100
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current.name,
                        'type': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'pnl_points': pnl_points,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'bars_held': i - df.index.get_loc(entry_date)
                    })
                    
                    position = None
            
            # Check for new entry signals
            if position is None:
                if current['Long_Signal']:
                    position = 'LONG'
                    entry_price = current['Close']
                    entry_date = current.name
                    entry_type = 'LONG'
                    
                    # Calculate stops and targets
                    atr_stop = current['ATR'] * 1.5
                    pct_stop = entry_price * (self.stop_loss_pct / 100)
                    stop_distance = max(atr_stop, pct_stop)
                    
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + (stop_distance * (self.take_profit_pct / self.stop_loss_pct))
                
                elif current['Short_Signal']:
                    position = 'SHORT'
                    entry_price = current['Close']
                    entry_date = current.name
                    entry_type = 'SHORT'
                    
                    # Calculate stops and targets
                    atr_stop = current['ATR'] * 1.5
                    pct_stop = entry_price * (self.stop_loss_pct / 100)
                    stop_distance = max(atr_stop, pct_stop)
                    
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - (stop_distance * (self.take_profit_pct / self.stop_loss_pct))
        
        # Close any open position at end
        if position is not None:
            exit_price = df['Close'].iloc[-1]
            if position == 'LONG':
                pnl_points = exit_price - entry_price
                pnl_pct = (pnl_points / entry_price) * 100
            else:
                pnl_points = entry_price - exit_price
                pnl_pct = (pnl_points / entry_price) * 100
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[-1],
                'type': position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pnl_points': pnl_points,
                'pnl_pct': pnl_pct,
                'exit_reason': 'End of Data',
                'bars_held': len(df) - df.index.get_loc(entry_date)
            })
        
        # Calculate performance metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            winning = trades_df[trades_df['pnl_pct'] > 0]
            losing = trades_df[trades_df['pnl_pct'] <= 0]
            
            total_return = trades_df['pnl_pct'].sum()
            total_points = trades_df['pnl_points'].sum()
            win_rate = (len(winning) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
            
            avg_win = winning['pnl_pct'].mean() if len(winning) > 0 else 0
            avg_loss = losing['pnl_pct'].mean() if len(losing) > 0 else 0
            
            gross_profit = winning['pnl_points'].sum() if len(winning) > 0 else 0
            gross_loss = abs(losing['pnl_points'].sum()) if len(losing) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate max drawdown
            cumulative = trades_df['pnl_pct'].cumsum()
            peak = cumulative.expanding().max()
            drawdown = ((peak - cumulative) / peak * 100).max() if len(cumulative) > 0 else 0
            
            performance = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'win_rate': win_rate,
                'total_return': total_return,
                'total_points': total_points,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': drawdown,
                'best_trade': trades_df['pnl_pct'].max(),
                'worst_trade': trades_df['pnl_pct'].min(),
                'avg_bars_held': trades_df['bars_held'].mean(),
                'buy_hold_return': buy_hold_return,
                'buy_hold_points': buy_hold_points,
                'vs_buy_hold_pct': total_return - buy_hold_return,
                'vs_buy_hold_points': total_points - buy_hold_points
            }
        else:
            performance = {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_return': 0, 'total_points': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0, 'max_drawdown': 0,
                'best_trade': 0, 'worst_trade': 0, 'avg_bars_held': 0,
                'buy_hold_return': buy_hold_return, 'buy_hold_points': buy_hold_points,
                'vs_buy_hold_pct': -buy_hold_return, 'vs_buy_hold_points': -buy_hold_points
            }
            trades_df = pd.DataFrame()
        
        return df, trades_df, performance
    
    def get_live_recommendation(self, df):
        """Get live trading recommendation based on last candle"""
        df = self.generate_signals(df)
        
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        recommendation = {
            'timestamp': latest.name,
            'current_price': latest['Close'],
            'signal': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_points': None,
            'reward_points': None,
            'risk_reward_ratio': None,
            'confidence': 0,
            'reasons': []
        }
        
        # Check for signals
        if latest['Long_Signal']:
            recommendation['signal'] = 'LONG'
            recommendation['entry_price'] = latest['Close']
            
            # Calculate stops and targets
            atr_stop = latest['ATR'] * 1.5
            pct_stop = latest['Close'] * (self.stop_loss_pct / 100)
            stop_distance = max(atr_stop, pct_stop)
            
            recommendation['stop_loss'] = latest['Close'] - stop_distance
            recommendation['take_profit'] = latest['Close'] + (stop_distance * (self.take_profit_pct / self.stop_loss_pct))
            recommendation['risk_points'] = stop_distance
            recommendation['reward_points'] = recommendation['take_profit'] - recommendation['entry_price']
            recommendation['risk_reward_ratio'] = recommendation['reward_points'] / recommendation['risk_points']
            
            # Confidence and reasons
            recommendation['confidence'] = int((latest['Long_Score'] / 5) * 100) if self.strategy_type == "Momentum" else 70
            
            if latest['Close'] > latest['EMA_Fast']:
                recommendation['reasons'].append(f"✅ Price above EMA{self.ema_fast} ({latest['Close']:.2f} > {latest['EMA_Fast']:.2f})")
            if latest['EMA_Fast'] > latest['EMA_Slow']:
                recommendation['reasons'].append(f"✅ EMA crossover bullish (Fast > Slow)")
            if latest['RSI'] > 40 and latest['RSI'] < 80:
                recommendation['reasons'].append(f"✅ RSI in bullish zone ({latest['RSI']:.1f})")
            if latest['MACD'] > latest['MACD_Signal']:
                recommendation['reasons'].append(f"✅ MACD bullish crossover")
            if latest['Volume_Ratio'] > 1.0:
                recommendation['reasons'].append(f"✅ Volume above average ({latest['Volume_Ratio']:.2f}x)")
            
            recommendation['reasons'].append(f"📊 Signal Score: {latest['Long_Score']}/5")
        
        elif latest['Short_Signal']:
            recommendation['signal'] = 'SHORT'
            recommendation['entry_price'] = latest['Close']
            
            # Calculate stops and targets
            atr_stop = latest['ATR'] * 1.5
            pct_stop = latest['Close'] * (self.stop_loss_pct / 100)
            stop_distance = max(atr_stop, pct_stop)
            
            recommendation['stop_loss'] = latest['Close'] + stop_distance
            recommendation['take_profit'] = latest['Close'] - (stop_distance * (self.take_profit_pct / self.stop_loss_pct))
            recommendation['risk_points'] = stop_distance
            recommendation['reward_points'] = recommendation['entry_price'] - recommendation['take_profit']
            recommendation['risk_reward_ratio'] = recommendation['reward_points'] / recommendation['risk_points']
            
            # Confidence and reasons
            recommendation['confidence'] = int((latest['Short_Score'] / 5) * 100) if self.strategy_type == "Momentum" else 70
            
            if latest['Close'] < latest['EMA_Fast']:
                recommendation['reasons'].append(f"✅ Price below EMA{self.ema_fast} ({latest['Close']:.2f} < {latest['EMA_Fast']:.2f})")
            if latest['EMA_Fast'] < latest['EMA_Slow']:
                recommendation['reasons'].append(f"✅ EMA crossover bearish (Fast < Slow)")
            if latest['RSI'] < 60 and latest['RSI'] > 20:
                recommendation['reasons'].append(f"✅ RSI in bearish zone ({latest['RSI']:.1f})")
            if latest['MACD'] < latest['MACD_Signal']:
                recommendation['reasons'].append(f"✅ MACD bearish crossover")
            if latest['Volume_Ratio'] > 1.0:
                recommendation['reasons'].append(f"✅ Volume above average ({latest['Volume_Ratio']:.2f}x)")
            
            recommendation['reasons'].append(f"📊 Signal Score: {latest['Short_Score']}/5")
        
        else:
            # No signal - show market status
            recommendation['signal'] = 'WAIT'
            recommendation['reasons'].append("⏳ No clear signal - waiting for setup")
            
            if latest['RSI'] > 70:
                recommendation['reasons'].append(f"⚠️ RSI overbought ({latest['RSI']:.1f})")
            elif latest['RSI'] < 30:
                recommendation['reasons'].append(f"⚠️ RSI oversold ({latest['RSI']:.1f})")
            else:
                recommendation['reasons'].append(f"📊 RSI neutral ({latest['RSI']:.1f})")
            
            if latest['Close'] > latest['EMA_Fast'] > latest['EMA_Slow']:
                recommendation['reasons'].append("📈 Trend: Bullish alignment")
            elif latest['Close'] < latest['EMA_Fast'] < latest['EMA_Slow']:
                recommendation['reasons'].append("📉 Trend: Bearish alignment")
            else:
                recommendation['reasons'].append("📊 Trend: Sideways/Mixed")
            
            recommendation['reasons'].append(f"💹 MACD: {'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish'}")
        
        return recommendation

def load_data_yfinance(symbol, period="1y", interval="1d"):
    """Load data from Yahoo Finance"""
    try:
        indian_symbols = {
            'NIFTY': '^NSEI', 'BANKNIFTY': '^NSEBANK', 'SENSEX': '^BSESN'
        }
        
        yf_symbol = indian_symbols.get(symbol.upper(), symbol)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None, False
        
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df['Date'] = df['Datetime']
            df = df.drop('Datetime', axis=1)
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        has_volume = df['Volume'].sum() > 0 and df['Volume'].std() > 0
        
        return df, has_volume
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, False

def load_data_csv(uploaded_file):
    """Load CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time'])]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df = df.set_index(date_cols[0])
        
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'open' in cl:
                col_map[col] = 'Open'
            elif 'high' in cl:
                col_map[col] = 'High'
            elif 'low' in cl:
                col_map[col] = 'Low'
            elif 'close' in cl:
                col_map[col] = 'Close'
            elif 'volume' in cl:
                col_map[col] = 'Volume'
        
        df = df.rename(columns=col_map)
        
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        has_volume = df['Volume'].sum() > 0
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']], has_volume
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, False

def create_chart(df, trades_df=None):
    """Create trading chart"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                       subplot_titles=('Price & Signals', 'RSI', 'MACD'),
                       row_heights=[0.6, 0.2, 0.2])
    
    # Price
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    
    if 'EMA_Fast' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Fast'], name='EMA Fast',
                                line=dict(color='orange', width=1)), row=1, col=1)
    if 'EMA_Slow' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_Slow'], name='EMA Slow',
                                line=dict(color='blue', width=1)), row=1, col=1)
    
    # Signals
    if trades_df is not None and not trades_df.empty:
        longs = trades_df[trades_df['type'] == 'LONG']
        shorts = trades_df[trades_df['type'] == 'SHORT']
        
        if not longs.empty:
            fig.add_trace(go.Scatter(x=longs['entry_date'], y=longs['entry_price'],
                                    mode='markers', name='Long', 
                                    marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
        if not shorts.empty:
            fig.add_trace(go.Scatter(x=shorts['entry_date'], y=shorts['entry_price'],
                                    mode='markers', name='Short',
                                    marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                line=dict(color='blue')), row=3, col=1)
        if 'MACD_Signal' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                                    line=dict(color='red')), row=3, col=1)
        if 'MACD_Hist' in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
                                marker_color='gray'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, template="plotly_dark",
                     xaxis_rangeslider_visible=False)
    
    return fig

def main():
    st.title("🚀 Advanced Trading System")
    st.markdown("**Pure Pandas Implementation** - No External Dependencies")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'has_volume' not in st.session_state:
        st.session_state.has_volume = True
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Strategy selection
    st.sidebar.subheader("🎯 Strategy")
    strategy_type = st.sidebar.selectbox("Select Strategy:", 
                                         ["Momentum", "Mean Reversion", "Trend"])
    
    strategy_info = {
        "Momentum": "Catches trending moves with EMA crossovers",
        "Mean Reversion": "Buys oversold, sells overbought",
        "Trend": "Follows strong established trends"
    }
    st.sidebar.info(strategy_info[strategy_type])
    
    # Data source
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio("Source:", ["Yahoo Finance", "CSV Upload"])
    
    uploaded_file = None
    symbol = ""
    period = "1y"
    interval = "1d"
    
    if data_source == "CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    else:
        market = st.sidebar.selectbox("Market:", 
                                     ["Indian Indices", "US Stocks", "Crypto"])
        
        if market == "Indian Indices":
            symbol = st.sidebar.selectbox("Index:", ["NIFTY", "BANKNIFTY", "SENSEX"])
        elif market == "US Stocks":
            symbol = st.sidebar.selectbox("Stock:", 
                                         ["AAPL", "MSFT", "GOOGL", "TSLA", "Custom"])
            if symbol == "Custom":
                symbol = st.sidebar.text_input("Symbol:")
        else:
            symbol = st.sidebar.selectbox("Crypto:", 
                                         ["BTC-USD", "ETH-USD", "BNB-USD"])
        
        period = st.sidebar.selectbox("Period:", 
                                     ['1d','5d',"1mo", "3mo", "6mo", "1y", "2y", "5y",'10y','15y','20y','30y'], index=2)
        interval = st.sidebar.selectbox("Interval:", 
                                       ["1m", "5m", '10m',"15m", "30m", "1h",'4h', "1d",'1wk'], index=5)
    
    # Strategy parameters
    st.sidebar.subheader("⚙️ Parameters")
    ema_fast = st.sidebar.slider("EMA Fast", 5, 20, 9)
    ema_slow = st.sidebar.slider("EMA Slow", 15, 50, 21)
    rsi_period = st.sidebar.slider("RSI Period", 10, 21, 14)
    
    # Risk management
    st.sidebar.subheader("🛡️ Risk Management")
    stop_loss_pct = st.sidebar.slider("Stop Loss %", 0.5, 5.0, 2.0, 0.1)
    take_profit_pct = st.sidebar.slider("Take Profit %", 2.0, 15.0, 6.0, 0.5)
    
    st.sidebar.info(f"Risk:Reward = 1:{take_profit_pct/stop_loss_pct:.1f}")
    
    # Load data button
    fetch_button = st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True)
    
    # Status
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data loaded")
        if st.session_state.df is not None:
            st.sidebar.info(f"📊 Records: {len(st.session_state.df)}")
    else:
        st.sidebar.info("👆 Load data to start")
    
    # Mode
    st.sidebar.subheader("📈 Mode")
    mode = st.sidebar.radio("Select:", ["📊 Backtest", "🎯 Live Signals"])
    
    # Load data
    if fetch_button or (data_source == "CSV Upload" and uploaded_file is not None):
        if data_source == "CSV Upload" and uploaded_file is not None:
            with st.spinner("Loading CSV..."):
                df, has_volume = load_data_csv(uploaded_file)
        else:
            if symbol:
                with st.spinner(f"Loading {symbol}..."):
                    df, has_volume = load_data_yfinance(symbol, period, interval)
            else:
                st.error("Enter a symbol")
                return
        
        if df is not None and not df.empty:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.has_volume = has_volume
            
            st.success(f"✅ Loaded {len(df)} records")
            st.info(f"📅 {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            st.session_state.data_loaded = False
            return
    
    # Check data
    if not st.session_state.data_loaded:
        st.info("👆 Configure and load data to begin")
        
        st.markdown("## 🎯 System Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**✅ Pure Pandas**")
            st.markdown("- No TA-Lib needed")
            st.markdown("- No external dependencies")
            st.markdown("- Works everywhere")
            st.markdown("- Easy to deploy")
        
        with col2:
            st.markdown("**📊 Strategies**")
            st.markdown("- Momentum trading")
            st.markdown("- Mean reversion")
            st.markdown("- Trend following")
            st.markdown("- All timeframes")
        
        with col3:
            st.markdown("**🎯 Features**")
            st.markdown("- Live recommendations")
            st.markdown("- Entry/Stop/Target")
            st.markdown("- Signal reasons")
            st.markdown("- Full backtesting")
        
        return
    
    df = st.session_state.df
    
    # Initialize strategy
    strategy = TradingStrategy(
        strategy_type=strategy_type,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )
    
    if mode == "📊 Backtest":
        st.header(f"📊 {strategy_type} Strategy Backtest")
        
        with st.spinner("Running backtest..."):
            processed_df, trades_df, perf = strategy.backtest(df)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{perf['total_return']:.2f}%",
                     delta=f"{perf['vs_buy_hold_pct']:+.2f}% vs B&H")
        with col2:
            st.metric("Total Trades", f"{perf['total_trades']}",
                     delta=f"Win Rate: {perf['win_rate']:.1f}%")
        with col3:
            st.metric("Profit Factor", f"{perf['profit_factor']:.2f}",
                     delta=f"Avg Hold: {perf['avg_bars_held']:.0f} bars")
        with col4:
            st.metric("Total Points", f"{perf['total_points']:+.1f}",
                     delta=f"Max DD: {perf['max_drawdown']:.1f}%")
        
        # Assessment
        st.subheader("🎯 Performance Assessment")
        
        score = 0
        assessment = []
        
        if perf['total_return'] > 0:
            score += 30
            assessment.append("✅ Positive returns")
        else:
            assessment.append("❌ Negative returns")
        
        if perf['total_trades'] >= 20:
            score += 25
            assessment.append("✅ Good trade frequency")
        elif perf['total_trades'] >= 10:
            score += 15
            assessment.append("⚠️ Moderate frequency")
        else:
            assessment.append("❌ Low frequency")
        
        if perf['win_rate'] >= 60:
            score += 25
            assessment.append("✅ High win rate")
        elif perf['win_rate'] >= 50:
            score += 20
            assessment.append("✅ Good win rate")
        elif perf['win_rate'] >= 40:
            score += 10
            assessment.append("⚠️ Fair win rate")
        else:
            assessment.append("❌ Low win rate")
        
        if perf['profit_factor'] >= 2.0:
            score += 20
            assessment.append("✅ Excellent profit factor")
        elif perf['profit_factor'] >= 1.5:
            score += 15
            assessment.append("✅ Good profit factor")
        elif perf['profit_factor'] >= 1.0:
            score += 5
            assessment.append("⚠️ Break-even")
        else:
            assessment.append("❌ Poor profit factor")
        
        if score >= 75:
            st.success(f"🏆 Excellent Performance ({score}/100)")
        elif score >= 50:
            st.warning(f"⚠️ Good Performance ({score}/100)")
        else:
            st.error(f"❌ Poor Performance ({score}/100)")
        
        for item in assessment:
            st.write(f"  {item}")
        
        # Detailed metrics
        st.subheader("📊 Detailed Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Winning Trades:**")
            st.write(f"- Count: {perf['winning_trades']}")
            st.write(f"- Win Rate: {perf['win_rate']:.1f}%")
            st.write(f"- Avg Win: {perf['avg_win']:.2f}%")
            st.write(f"- Best: {perf['best_trade']:.2f}%")
        
        with col2:
            st.write("**Losing Trades:**")
            st.write(f"- Count: {perf['losing_trades']}")
            st.write(f"- Loss Rate: {100-perf['win_rate']:.1f}%")
            st.write(f"- Avg Loss: {perf['avg_loss']:.2f}%")
            st.write(f"- Worst: {perf['worst_trade']:.2f}%")
        
        with col3:
            st.write("**Overall:**")
            st.write(f"- Total Return: {perf['total_return']:.2f}%")
            st.write(f"- Buy & Hold: {perf['buy_hold_return']:.2f}%")
            st.write(f"- Profit Factor: {perf['profit_factor']:.2f}")
            st.write(f"- Max Drawdown: {perf['max_drawdown']:.1f}%")
        
        # Chart
        st.subheader("📈 Trading Chart")
        fig = create_chart(processed_df, trades_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        if not trades_df.empty:
            st.subheader("💼 Trade History (Last 20)")
            
            recent = trades_df.tail(20).copy()
            recent['Entry'] = pd.to_datetime(recent['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
            recent['Exit'] = pd.to_datetime(recent['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
            recent['Type'] = recent['type']
            recent['Entry $'] = recent['entry_price'].round(2)
            recent['Exit $'] = recent['exit_price'].round(2)
            recent['P&L %'] = recent['pnl_pct'].round(2)
            recent['P&L Pts'] = recent['pnl_points'].round(2)
            recent['Bars'] = recent['bars_held'].astype(int)
            recent['Exit Reason'] = recent['exit_reason']
            
            display_cols = ['Entry', 'Type', 'Entry $', 'Exit $', 'P&L %', 'P&L Pts', 'Bars', 'Exit Reason']
            st.dataframe(recent[display_cols], use_container_width=True)
        
        # Recommendations
        if perf['total_return'] > 5 and perf['win_rate'] >= 50:
            st.success("🚀 **Excellent!** Ready for live trading")
        elif perf['total_trades'] < 10:
            st.warning("⚠️ **Low frequency** - Try shorter timeframes")
        elif perf['win_rate'] < 40:
            st.error("❌ **Low win rate** - Try different strategy or parameters")
    
    else:  # Live Signals
        st.header(f"🎯 {strategy_type} Live Signals")
        
        with st.spinner("Analyzing live market..."):
            recommendation = strategy.get_live_recommendation(df)
        
        if recommendation is None:
            st.error("❌ Insufficient data for analysis")
            return
        
        # Current market status
        st.subheader("📊 Current Market Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"{recommendation['current_price']:.2f}")
        
        with col2:
            processed_df = strategy.generate_signals(df)
            latest = processed_df.iloc[-1]
            st.metric("RSI", f"{latest['RSI']:.1f}")
        
        with col3:
            macd_status = "Bullish" if latest['MACD'] > latest['MACD_Signal'] else "Bearish"
            st.metric("MACD", macd_status)
        
        with col4:
            trend = "Up" if latest['Close'] > latest['EMA_Fast'] > latest['EMA_Slow'] else "Down" if latest['Close'] < latest['EMA_Fast'] < latest['EMA_Slow'] else "Sideways"
            st.metric("Trend", trend)
        
        # Live recommendation
        st.subheader("🚨 Live Trading Recommendation")
        
        if recommendation['signal'] == 'LONG':
            st.success("🟢 **LONG SIGNAL DETECTED**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Trade Setup")
                st.write(f"**Entry Price:** {recommendation['entry_price']:.2f}")
                st.write(f"**Stop Loss:** {recommendation['stop_loss']:.2f}")
                st.write(f"**Take Profit:** {recommendation['take_profit']:.2f}")
                st.write(f"**Risk Points:** {recommendation['risk_points']:.2f}")
                st.write(f"**Reward Points:** {recommendation['reward_points']:.2f}")
                st.write(f"**Risk:Reward:** 1:{recommendation['risk_reward_ratio']:.2f}")
            
            with col2:
                st.markdown("### 📊 Signal Quality")
                st.write(f"**Confidence:** {recommendation['confidence']}%")
                st.write(f"**Timestamp:** {recommendation['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Strategy:** {strategy_type}")
                
                # Progress bar for confidence
                st.progress(recommendation['confidence'] / 100)
            
            st.markdown("### 💡 Entry Reasons")
            for reason in recommendation['reasons']:
                st.write(f"  {reason}")
            
            # Action plan
            st.markdown("### 🎯 Action Plan")
            st.info("1️⃣ Enter LONG position at current price")
            st.info(f"2️⃣ Set stop loss at {recommendation['stop_loss']:.2f}")
            st.info(f"3️⃣ Set take profit at {recommendation['take_profit']:.2f}")
            st.info("4️⃣ Monitor position and adjust if needed")
        
        elif recommendation['signal'] == 'SHORT':
            st.error("🔴 **SHORT SIGNAL DETECTED**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Trade Setup")
                st.write(f"**Entry Price:** {recommendation['entry_price']:.2f}")
                st.write(f"**Stop Loss:** {recommendation['stop_loss']:.2f}")
                st.write(f"**Take Profit:** {recommendation['take_profit']:.2f}")
                st.write(f"**Risk Points:** {recommendation['risk_points']:.2f}")
                st.write(f"**Reward Points:** {recommendation['reward_points']:.2f}")
                st.write(f"**Risk:Reward:** 1:{recommendation['risk_reward_ratio']:.2f}")
            
            with col2:
                st.markdown("### 📊 Signal Quality")
                st.write(f"**Confidence:** {recommendation['confidence']}%")
                st.write(f"**Timestamp:** {recommendation['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Strategy:** {strategy_type}")
                
                st.progress(recommendation['confidence'] / 100)
            
            st.markdown("### 💡 Entry Reasons")
            for reason in recommendation['reasons']:
                st.write(f"  {reason}")
            
            st.markdown("### 🎯 Action Plan")
            st.info("1️⃣ Enter SHORT position at current price")
            st.info(f"2️⃣ Set stop loss at {recommendation['stop_loss']:.2f}")
            st.info(f"3️⃣ Set take profit at {recommendation['take_profit']:.2f}")
            st.info("4️⃣ Monitor position and adjust if needed")
        
        else:
            st.info("⏳ **NO SIGNAL - WAIT FOR SETUP**")
            
            st.markdown("### 📊 Market Analysis")
            for reason in recommendation['reasons']:
                st.write(f"  {reason}")
            
            st.markdown("### 💡 What to Look For")
            if strategy_type == "Momentum":
                st.write("- Wait for EMA crossover")
                st.write("- RSI should be 40-80 for longs, 20-60 for shorts")
                st.write("- MACD crossover confirmation")
                st.write("- Volume above average")
            elif strategy_type == "Mean Reversion":
                st.write("- RSI < 35 for long signals")
                st.write("- RSI > 65 for short signals")
                st.write("- Price touching Bollinger Bands")
            else:
                st.write("- Strong trend alignment")
                st.write("- Multiple EMAs in order")
                st.write("- MACD confirming trend")
        
        # Live chart
        st.subheader("📈 Live Chart")
        processed_df = strategy.generate_signals(df)
        fig = create_chart(processed_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Advanced Trading System** - Pure Pandas Implementation")
    st.markdown("✅ No external dependencies | 📊 All timeframes | 🎯 Live recommendations")
    st.markdown("⚠️ *Past performance doesn't guarantee future results. Always use proper risk management.*")

if __name__ == "__main__":
    main()
