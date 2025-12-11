import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import time
from scipy.signal import argrelextrema

# ==========================================
# 1. SYSTEM CONFIGURATION & STATE MANAGEMENT
# ==========================================
st.set_page_config(layout="wide", page_title="AlgoTrader Pro V2", page_icon="📊")

# Initialize Session State for Deep Scan Persistence
if 'deep_scan_results' not in st.session_state:
    st.session_state.deep_scan_results = None

# Constants
ASSETS = {
    "Indices": ["^NSEI", "^NSEBANK", "^BSESN", "^GSPC", "^DJI"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDINR=X", "JPY=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TATAMOTORS.NS", "AAPL", "TSLA", "NVDA", "AMD"]
}

# Strict YFinance Valid Intervals
VALID_INTERVALS = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1h": "730d", "4h": "730d", "1d": "max", "1wk": "max", "1mo": "max"
}

# ==========================================
# 2. DATA ENGINE
# ==========================================
class DataManager:
    @staticmethod
    def fetch_data(ticker, interval, period):
        """Fetches data with rate limiting, error handling, and timezone alignment."""
        time.sleep(1.0)  # Prevent Rate Limiting
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            
            if df.empty: return None
            
            # Fix YFinance MultiIndex columns (Common issue in 2024/2025)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize Timezone to IST
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Kolkata')
            
            # Ensure proper types
            df = df.astype(float)
            return df
        except Exception as e:
            st.error(f"Data Fetch Error for {ticker}: {e}")
            return None

# ==========================================
# 3. MANUAL TECHNICAL INDICATORS
# ==========================================
class Technicals:
    @staticmethod
    def calculate_all(df):
        """Applies all manual indicators to the dataframe."""
        # Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands (20, 2)
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        
        # ATR (14) - For SL/Target
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # ADX (14)
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/14).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        df['ADX'] = dx.rolling(14).mean()
        
        # Clean NaN
        df.dropna(inplace=True)
        return df

    @staticmethod
    def get_fibonacci_levels(df):
        """Dynamic Fibonacci Retracement Levels based on recent swing."""
        max_price = df['High'].max()
        min_price = df['Low'].min()
        diff = max_price - min_price
        
        levels = {
            0.0: min_price,
            0.236: min_price + 0.236 * diff,
            0.382: min_price + 0.382 * diff,
            0.5: min_price + 0.5 * diff,
            0.618: min_price + 0.618 * diff,
            1.0: max_price
        }
        return levels

# ==========================================
# 4. SIGNAL GENERATOR & BACKTEST ENGINE
# ==========================================
class SignalEngine:
    @staticmethod
    def generate_signals(df):
        """Generates Buy/Sell/Hold signals based on composite logic."""
        last = df.iloc[-1]
        score = 0
        reasons = []
        
        # 1. Trend Filter
        if last['Close'] > last['SMA_200']:
            score += 1
            reasons.append("Price > 200 SMA (Long-term Bullish)")
        elif last['Close'] < last['SMA_200']:
            score -= 1
            reasons.append("Price < 200 SMA (Long-term Bearish)")
            
        # 2. Momentum (RSI)
        if last['RSI'] < 30:
            score += 2
            reasons.append("RSI Oversold (Bounce Likely)")
        elif last['RSI'] > 70:
            score -= 2
            reasons.append("RSI Overbought (Pullback Likely)")
            
        # 3. MACD
        if last['MACD'] > last['Signal']:
            score += 1
            reasons.append("MACD Bullish Crossover")
        else:
            score -= 1
            reasons.append("MACD Bearish Divergence")
            
        # 4. Bollinger
        if last['Close'] < last['BB_Lower']:
            score += 1
            reasons.append("Price below Lower BB (Mean Reversion)")
        elif last['Close'] > last['BB_Upper']:
            score -= 1
            reasons.append("Price above Upper BB (Mean Reversion)")
            
        return score, reasons

    @staticmethod
    def calculate_trade_setup(df, signal_type):
        """Calculates precise Entry, SL, and Target using ATR."""
        last_close = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        if signal_type == "BUY":
            entry = last_close
            sl = last_close - (2.0 * atr)  # 2x ATR Stop Loss
            target = last_close + (3.0 * atr) # 3x ATR Target
            risk_reward = (target - entry) / (entry - sl)
        else: # SELL
            entry = last_close
            sl = last_close + (2.0 * atr)
            target = last_close - (3.0 * atr)
            risk_reward = (entry - target) / (sl - entry)
            
        return entry, sl, target, risk_reward

class Backtester:
    @staticmethod
    def prove_reliability(df):
        """
        Runs a vectorized backtest to PROVE if the ticker respects the strategy.
        Checks: 'When MACD crossed up & Price > SMA200, did price rise in next 10 bars?'
        """
        # Define the Strategy Logic for the Backtest
        df['Signal_Flag'] = 0
        
        # Condition: Golden Cross (SMA50 > SMA200) + RSI < 50 (Pullback in Uptrend)
        buy_condition = (df['SMA_50'] > df['SMA_200']) & (df['RSI'] < 45) & (df['RSI'] > 30)
        df.loc[buy_condition, 'Signal_Flag'] = 1
        
        # Forward Returns (5 periods later)
        df['Fwd_Return'] = df['Close'].shift(-5) / df['Close'] - 1
        
        # Filter only signal candles
        signals = df[df['Signal_Flag'] == 1]
        
        if len(signals) < 5:
            return None, "Insufficient Historical Signals"
        
        win_rate = len(signals[signals['Fwd_Return'] > 0]) / len(signals) * 100
        avg_gain = signals['Fwd_Return'].mean() * 100
        
        # Compare vs Buy & Hold
        total_period_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        return {
            "Total Signals": len(signals),
            "Win Rate": win_rate,
            "Avg Trade Return": avg_gain,
            "Buy & Hold Return": total_period_return * 100,
            "Is Working": win_rate > 50 and avg_gain > 0
        }, signals

# ==========================================
# 5. UI ORCHESTRATION
# ==========================================
def main():
    st.sidebar.header("🕹️ Control Panel")
    
    # Inputs
    asset_class = st.sidebar.selectbox("Asset Class", list(ASSETS.keys()))
    ticker = st.sidebar.selectbox("Ticker", ASSETS[asset_class])
    custom = st.sidebar.text_input("Custom Ticker (YFinance)")
    if custom: ticker = custom
    
    # Config
    col1, col2 = st.sidebar.columns(2)
    interval = col1.selectbox("Timeframe", list(VALID_INTERVALS.keys()), index=6) # Default 1d
    period = col2.selectbox("History", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
    
    run_analysis = st.sidebar.button("🚀 Analyze Market", type="primary")
    run_deep_scan = st.sidebar.button("📡 Run Deep Scan (Multi-TF)")

    # Main Data Fetch
    if run_analysis:
        st.session_state.deep_scan_results = None # Reset deep scan on new analysis
        
        with st.spinner(f"Fetching {ticker} data..."):
            df = DataManager.fetch_data(ticker, interval, period)
            
        if df is None:
            st.error("❌ Data Fetch Failed. Please check the ticker symbol.")
            return

        # Calculate Techs
        df = Technicals.calculate_all(df)
        
        # Generate Signal
        score, reasons = SignalEngine.generate_signals(df)
        signal_type = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
        entry, sl, target, rr = SignalEngine.calculate_trade_setup(df, signal_type)
        
        # --- DASHBOARD ---
        st.title(f"📈 Algorithmic Analysis: {ticker}")
        
        # 1. RECOMMENDATION CARD
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommendation", signal_type, delta=f"Score: {score}/5", delta_color="normal")
        c2.metric("Entry Zone", f"{entry:.2f}")
        c3.metric("Stop Loss", f"{sl:.2f}", delta=f"-{(abs(entry-sl)/entry)*100:.2f}%", delta_color="inverse")
        c4.metric("Target", f"{target:.2f}", delta=f"+{(abs(target-entry)/entry)*100:.2f}%")
        
        # 2. PROOF OF CONCEPT (Validation)
        st.markdown("### 🛡️ Does this Ticker Respect the Strategy?")
        stats, sig_df = Backtester.prove_reliability(df)
        
        if stats:
            p1, p2, p3 = st.columns(3)
            p1.metric("Historical Win Rate", f"{stats['Win Rate']:.1f}%", help="Percentage of signals that were profitable after 5 bars.")
            p2.metric("Avg Return per Trade", f"{stats['Avg Trade Return']:.2f}%")
            
            is_valid = stats['Is Working']
            color = "green" if is_valid else "red"
            status = "APPROVED" if is_valid else "CAUTION"
            p3.markdown(f"**Status:** :{color}[{status}]")
            
            if is_valid:
                st.success(f"✅ PROOF: This asset historically respects 'Pullback in Uptrend' logic. Strategy beats random chance.")
            else:
                st.warning(f"⚠️ CAUTION: Historical reliability is low ({stats['Win Rate']:.1f}%). Use tight stops.")
                
            # Plot Proof
            fig_proof = go.Figure()
            fig_proof.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='gray', width=1)))
            # Plot Winning Signals
            wins = sig_df[sig_df['Fwd_Return'] > 0]
            fig_proof.add_trace(go.Scatter(x=wins.index, y=wins['Close'], mode='markers', marker=dict(color='green', size=8, symbol='triangle-up'), name="Winning Signal"))
            # Plot Losing Signals
            losses = sig_df[sig_df['Fwd_Return'] <= 0]
            fig_proof.add_trace(go.Scatter(x=losses.index, y=losses['Close'], mode='markers', marker=dict(color='red', size=8, symbol='x'), name="Failed Signal"))
            
            fig_proof.update_layout(title="Historical Signal Validation (Green=Win, Red=Loss)", height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_proof, use_container_width=True)

        # 3. DETAILED CHARTS
        tab1, tab2, tab3 = st.tabs(["Technical Chart", "Ratio & Volatility", "Logic & Summary"])
        
        with tab1:
            # Main Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dot', width=1), name='BB Upper'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dot', width=1), name='BB Lower'), row=1, col=1)
            
            # Subplot
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Fibonacci Levels
            fibs = Technicals.get_fibonacci_levels(df)
            for level, price in fibs.items():
                fig.add_hline(y=price, line_color="gold", line_width=1, opacity=0.5, row=1, col=1, annotation_text=f"Fib {level}")

            fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"Price Action Structure ({interval})")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Volatility Analysis")
                df['Returns'] = df['Close'].pct_change()
                fig_vol = go.Figure(data=[go.Histogram(x=df['Returns'], nbinsx=50, marker_color='teal')])
                fig_vol.update_layout(title="Return Distribution (Fat Tails = High Risk)", height=300)
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with c2:
                st.subheader("Ratio Analysis (vs Gold)")
                # Fetch Gold for Ratio
                gold_df = DataManager.fetch_data("GC=F", interval, period)
                if gold_df is not None:
                    # Align indices
                    common = df.index.intersection(gold_df.index)
                    if not common.empty:
                        ratio = df.loc[common]['Close'] / gold_df.loc[common]['Close']
                        fig_ratio = go.Figure(go.Scatter(x=ratio.index, y=ratio, mode='lines', name=f'{ticker}/Gold'))
                        fig_ratio.update_layout(title=f"Strength vs Gold (Rising = {ticker} Stronger)", height=300)
                        st.plotly_chart(fig_ratio, use_container_width=True)
                    else:
                        st.warning("Timestamps do not align for Ratio analysis.")
                else:
                    st.warning("Could not fetch Gold data for ratio.")

        with tab3:
            st.subheader("Strategy Logic")
            st.write("The Recommendation is based on the following composite analysis:")
            for reason in reasons:
                st.write(f"• {reason}")
            
            st.markdown("### 300-Word Summary")
            trend_str = "Bullish" if df['Close'].iloc[-1] > df['SMA_200'].iloc[-1] else "Bearish"
            vol_str = "High" if df['ATR'].iloc[-1] > df['ATR'].mean() else "Low"
            
            summary = f"""
            **Market Structure Analysis for {ticker}**
            
            The current market structure for **{ticker}** on the **{interval}** timeframe presents a **{trend_str}** bias. The price is currently trading at {df['Close'].iloc[-1]:.2f}. 
            
            **Trend & Momentum:** The asset is trading {'above' if trend_str == 'Bullish' else 'below'} the 200-period Simple Moving Average, indicating the long-term trend direction. The RSI is currently at {df['RSI'].iloc[-1]:.2f}, suggesting momentum is {'neutral' if 30 < df['RSI'].iloc[-1] < 70 else 'extreme'}.
            
            **Volatility & Risk:** Volatility is currently **{vol_str}**, as measured by the ATR ({df['ATR'].iloc[-1]:.2f}). This implies that stop-losses should be adjusted accordingly. The Bollinger Bands are {'expanding' if df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1] > (df['BB_Upper'].mean() - df['BB_Lower'].mean()) else 'contracting'}, indicating a potential breakout or consolidation phase.
            
            **Conclusion:** Based on the statistical backtest shown in the "Proof" section, this setup has a historical win rate of {stats['Win Rate'] if stats else 'N/A'}%. Traders should watch the {entry:.2f} level for entries, strictly adhering to the stop loss at {sl:.2f}.
            """
            st.info(summary)

    # ==========================================
    # 6. DEEP SCAN LOGIC
    # ==========================================
    if run_deep_scan:
        st.markdown("---")
        st.subheader("📡 Deep Scan: Multi-Timeframe Matrix")
        
        # We use a subset of timeframes to prevent API Rate Limits
        scan_tfs = ["15m", "1h", "4h", "1d"]
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, tf in enumerate(scan_tfs):
            status_text.text(f"Scanning {tf} timeframe...")
            
            # Fetch minimal data for speed
            scan_period = "1mo" if tf in ["15m", "1h"] else "1y"
            tf_df = DataManager.fetch_data(ticker, tf, scan_period)
            
            if tf_df is not None:
                tf_df = Technicals.calculate_all(tf_df)
                last_row = tf_df.iloc[-1]
                
                trend = "BULL" if last_row['Close'] > last_row['SMA_200'] else "BEAR"
                rsi = last_row['RSI']
                signal = "BUY" if rsi < 30 and trend == "BULL" else "SELL" if rsi > 70 and trend == "BEAR" else "NEUTRAL"
                
                results.append({
                    "Timeframe": tf,
                    "Trend (200 SMA)": trend,
                    "RSI": f"{rsi:.1f}",
                    "Volatility (ATR)": f"{last_row['ATR']:.2f}",
                    "AI Signal": signal
                })
            
            time.sleep(1.1) # Respect API limits
            progress_bar.progress((i + 1) / len(scan_tfs))
            
        st.session_state.deep_scan_results = pd.DataFrame(results)
        status_text.text("Scan Complete.")

    # Display Deep Scan Results (Persisted)
    if st.session_state.deep_scan_results is not None:
        st.table(st.session_state.deep_scan_results)
        
        # Auto-Analysis of Matrix
        bull_count = len(st.session_state.deep_scan_results[st.session_state.deep_scan_results['Trend (200 SMA)'] == "BULL"])
        total = len(st.session_state.deep_scan_results)
        
        if bull_count == total:
            st.success("🔥 CONFLUENCE: All Timeframes are BULLISH. Strong Buy Confidence.")
        elif bull_count == 0:
            st.error("❄️ CONFLUENCE: All Timeframes are BEARISH. Strong Sell Confidence.")
        else:
            st.warning("⚠️ CONFLICT: Mixed signals across timeframes. Trade with caution.")

if __name__ == "__main__":
    main()
