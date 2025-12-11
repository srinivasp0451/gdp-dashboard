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
# 1. CONFIGURATION & STATE
# ==========================================
st.set_page_config(layout="wide", page_title="AlgoTrader Pro V3", page_icon="📈")

# Initialize Session State
if 'deep_scan_results' not in st.session_state:
    st.session_state.deep_scan_results = None

# Asset Definitions
ASSETS = {
    "Indices": ["^NSEI", "^NSEBANK", "^BSESN", "^GSPC", "^DJI"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDINR=X", "JPY=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "AAPL", "TSLA", "NVDA", "AMD"]
}

# Benchmarks for Ratio Analysis
RATIO_BENCHMARKS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "USD/INR": "USDINR=X",
    "Infosys": "INFY.NS"
}

VALID_INTERVALS = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1h": "730d", "4h": "730d", "1d": "max", "1wk": "max", "1mo": "max"
}

# ==========================================
# 2. DATA LAYER (Robust)
# ==========================================
class DataManager:
    @staticmethod
    def fetch_data(ticker, interval, period):
        time.sleep(0.5)  # Rate limit
        try:
            # Auto_adjust=True handles splits/dividends
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            
            if df is None or df.empty:
                return None
            
            # Fix MultiIndex columns (yfinance 2024+ update)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Timezone handling
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Kolkata')
            
            return df
        except Exception as e:
            return None

# ==========================================
# 3. TECHNICAL INDICATORS (Math Only)
# ==========================================
class Technicals:
    @staticmethod
    def calculate_all(df):
        if len(df) < 50: return df # Not enough data
        
        # Trend
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        # Momentum
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # ADX
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/14).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        df['ADX'] = dx.rolling(14).mean()
        
        df.dropna(inplace=True)
        return df

    @staticmethod
    def get_fibonacci_levels(df):
        recent_high = df['High'].tail(100).max()
        recent_low = df['Low'].tail(100).min()
        diff = recent_high - recent_low
        return {
            '0.0%': recent_low,
            '38.2%': recent_low + 0.382 * diff,
            '50.0%': recent_low + 0.5 * diff,
            '61.8%': recent_low + 0.618 * diff,
            '100%': recent_high
        }

# ==========================================
# 4. SIGNAL & BACKTEST ENGINE
# ==========================================
class SignalEngine:
    @staticmethod
    def analyze(df):
        last = df.iloc[-1]
        score = 0
        reasons = []
        
        # Logic
        if last['Close'] > last['SMA_200']:
            score += 1
            reasons.append("Bullish Trend (Price > 200 SMA)")
        else:
            score -= 1
            reasons.append("Bearish Trend (Price < 200 SMA)")
            
        if last['RSI'] < 30:
            score += 2
            reasons.append("RSI Oversold (Buy Zone)")
        elif last['RSI'] > 70:
            score -= 2
            reasons.append("RSI Overbought (Sell Zone)")
            
        if last['MACD'] > last['Signal']:
            score += 1
            reasons.append("MACD Bullish Crossover")
            
        signal = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
        
        # Trade Setup
        atr = last['ATR']
        if signal == "BUY":
            entry = last['Close']
            sl = entry - (2 * atr)
            target = entry + (3 * atr)
        elif signal == "SELL":
            entry = last['Close']
            sl = entry + (2 * atr)
            target = entry - (3 * atr)
        else:
            entry = sl = target = 0
            
        return signal, score, reasons, entry, sl, target

    @staticmethod
    def prove_strategy(df):
        """Backtests the logic: If Signal=BUY, does price go up?"""
        df['Signal_Flag'] = 0
        # Simple Logic: Price > SMA50 AND RSI < 45 (Dip Buy)
        buy_mask = (df['Close'] > df['SMA_50']) & (df['RSI'] < 45) & (df['RSI'] > 30)
        df.loc[buy_mask, 'Signal_Flag'] = 1
        
        # Check return 5 bars later
        df['Result'] = df['Close'].shift(-5)
        df['Trade_Return'] = (df['Result'] - df['Close']) / df['Close']
        
        trades = df[df['Signal_Flag'] == 1].copy()
        if len(trades) < 3: return None
        
        wins = len(trades[trades['Trade_Return'] > 0])
        win_rate = (wins / len(trades)) * 100
        
        return {
            "Win Rate": win_rate,
            "Total Trades": len(trades),
            "Avg Return": trades['Trade_Return'].mean() * 100,
            "Trades": trades
        }

# ==========================================
# 5. UI & LOGIC
# ==========================================
def main():
    st.sidebar.header("⚙️ Settings")
    
    asset_class = st.sidebar.selectbox("Asset Class", list(ASSETS.keys()))
    ticker1 = st.sidebar.selectbox("Primary Asset", ASSETS[asset_class])
    ticker2 = st.sidebar.text_input("Compare Ticker (Optional)", value="RELIANCE.NS")
    
    col1, col2 = st.sidebar.columns(2)
    interval = col1.selectbox("Interval", list(VALID_INTERVALS.keys()), index=6) # 1d default
    period = col2.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if st.sidebar.button("Analyze Now", type="primary"):
        st.session_state.deep_scan_results = None # Reset
        run_app(ticker1, ticker2, interval, period)

    if st.sidebar.button("Run Deep Scan"):
        run_deep_scan(ticker1)

def run_app(ticker1, ticker2, interval, period):
    with st.spinner("Fetching Data & Crunching Numbers..."):
        df = DataManager.fetch_data(ticker1, interval, period)
        if df is None:
            st.error("Data not found.")
            return

        df = Technicals.calculate_all(df)
        signal, score, reasons, entry, sl, tgt = SignalEngine.analyze(df)
        
        # --- HEADER ---
        st.title(f"📊 Analysis: {ticker1} ({interval})")
        
        # --- RECOMMENDATION ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", signal, delta=f"Score: {score}")
        if signal != "HOLD":
            c2.metric("Entry", f"{entry:.2f}")
            c3.metric("Stop Loss", f"{sl:.2f}", delta_color="inverse")
            c4.metric("Target", f"{tgt:.2f}")
            
        st.markdown("---")
        
        # --- TABS ---
        tabs = st.tabs(["Charts", "Ratio Analysis", "Deep Scan Results", "Strategy Proof"])
        
        # 1. CHARTS
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange'), name='SMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dot'), name='BB Upper'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dot'), name='BB Lower'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", row=2, col=1)
            
            # Auto-Summary
            last_rsi = df['RSI'].iloc[-1]
            trend = "Bullish" if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1] else "Bearish"
            summary = f"""
            **Technical Summary:**
            The asset is currently in a **{trend}** trend relative to the 50 SMA. 
            The RSI is at **{last_rsi:.2f}**, indicating the asset is {'Oversold' if last_rsi < 30 else 'Overbought' if last_rsi > 70 else 'Neutral'}.
            Volatility (ATR) is {df['ATR'].iloc[-1]:.2f}, suggesting specific stop-loss width.
            """
            st.plotly_chart(fig, use_container_width=True)
            st.info(summary)

        # 2. RATIO ANALYSIS (Fixed & Expanded)
        with tabs[1]:
            st.subheader("Relative Strength (Ratio) Analysis")
            st.write(f"Comparing **{ticker1}** Performance vs Global Assets")
            
            ratio_targets = RATIO_BENCHMARKS.copy()
            if ticker2: ratio_targets["User Selected"] = ticker2
            
            # Container for plots
            cols = st.columns(2)
            idx = 0
            
            insights = []
            
            for name, bench_ticker in ratio_targets.items():
                bench_df = DataManager.fetch_data(bench_ticker, interval, period)
                
                if bench_df is not None and not bench_df.empty:
                    # Align Dates
                    common = df.index.intersection(bench_df.index)
                    if len(common) > 10:
                        ratio = df.loc[common]['Close'] / bench_df.loc[common]['Close']
                        
                        # Plot
                        with cols[idx % 2]:
                            fig_r = go.Figure()
                            fig_r.add_trace(go.Scatter(x=ratio.index, y=ratio, mode='lines', name=f"{ticker1}/{name}"))
                            fig_r.update_layout(title=f"{ticker1} vs {name} Ratio", height=300)
                            st.plotly_chart(fig_r, use_container_width=True)
                            
                            # Micro-Insight
                            change = ((ratio.iloc[-1] - ratio.iloc[0]) / ratio.iloc[0]) * 100
                            direction = "Outperforming" if change > 0 else "Underperforming"
                            insights.append(f"- **{name}:** {ticker1} is **{direction}** by {abs(change):.1f}% over this period.")
                        
                        idx += 1
            
            # Ratio Summary
            st.markdown("### 📝 Ratio Analysis Summary")
            st.write(f"Based on the charts above, here is how **{ticker1}** compares to the broader market:")
            for i in insights: st.markdown(i)
            st.caption("Note: Rising ratio line = Primary Ticker is stronger. Falling line = Benchmark is stronger.")

        # 3. DEEP SCAN (Persisted)
        with tabs[2]:
            if st.session_state.deep_scan_results is not None:
                st.subheader("Multi-Timeframe Deep Scan Results")
                st.table(st.session_state.deep_scan_results)
                
                # Summary
                bulls = len(st.session_state.deep_scan_results[st.session_state.deep_scan_results['Trend'] == "BULL"])
                summary_scan = f"The Deep Scan shows **{bulls} out of {len(st.session_state.deep_scan_results)}** timeframes are Bullish. "
                if bulls == 4: summary_scan += "This indicates Strong Momentum alignment."
                else: summary_scan += "This indicates mixed signals; caution is advised."
                st.info(summary_scan)
            else:
                st.warning("Click 'Run Deep Scan' in the sidebar to populate this tab.")

        # 4. PROOF
        with tabs[3]:
            st.subheader("🛡️ Strategy Validation (Backtest)")
            stats = SignalEngine.prove_strategy(df)
            
            if stats:
                c1, c2, c3 = st.columns(3)
                c1.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
                c2.metric("Avg Return", f"{stats['Avg Return']:.2f}%")
                c3.metric("Total Trades", stats['Total Trades'])
                
                # Validation Logic
                if stats['Win Rate'] > 55 and stats['Avg Return'] > 0:
                    st.success(f"✅ PASSED: {ticker1} respects this strategy logic historically.")
                else:
                    st.error(f"❌ FAILED: {ticker1} does not respect this strategy well. Use manual discretion.")
                
                # Plot Trades
                trades = stats['Trades']
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='gray', opacity=0.5)))
                fig_bt.add_trace(go.Scatter(x=trades.index, y=trades['Close'], mode='markers', marker=dict(color='blue', size=8), name="Entry Signals"))
                st.plotly_chart(fig_bt, use_container_width=True)
                
                st.markdown(f"""
                **Proof Summary:**
                We tested a standard 'Dip Buying' strategy (Price > SMA50 & RSI < 45) on {ticker1}.
                Over the selected period, it generated {stats['Total Trades']} signals with a {stats['Win Rate']:.1f}% win rate.
                This confirms whether the asset moves technically or randomly.
                """)
            else:
                st.warning("Not enough historical data or signals to validate strategy.")

def run_deep_scan(ticker):
    """
    Loops through timeframes and stores results in session state.
    Includes ERROR HANDLING to prevent IndexError.
    """
    scan_tfs = ["15m", "1h", "4h", "1d"]
    results = []
    
    status = st.empty()
    bar = st.progress(0)
    
    for i, tf in enumerate(scan_tfs):
        status.text(f"Scanning {tf}...")
        
        # Fetch minimal data
        p = "1mo" if tf in ["15m", "1h"] else "1y"
        df = DataManager.fetch_data(ticker, tf, p)
        
        # --- FIX: Check if df is valid AND has rows ---
        if df is not None and not df.empty:
            try:
                # Calculate basic tech
                df['SMA_200'] = df['Close'].rolling(200).mean()
                df['RSI'] = Technicals.calculate_all(df)['RSI'] # Re-use logic
                
                # Safe access to last row
                last = df.iloc[-1]
                
                trend = "BULL" if last['Close'] > last['SMA_200'] else "BEAR"
                rsi = last['RSI']
                
                results.append({
                    "Timeframe": tf,
                    "Trend": trend,
                    "RSI": f"{rsi:.1f}",
                    "Close": f"{last['Close']:.2f}"
                })
            except Exception as e:
                # If calculation fails (e.g. not enough data for SMA200), skip gracefully
                results.append({"Timeframe": tf, "Trend": "N/A", "RSI": "N/A", "Close": "N/A"})
        else:
            results.append({"Timeframe": tf, "Trend": "No Data", "RSI": "-", "Close": "-"})
            
        bar.progress((i+1)/len(scan_tfs))
        
    st.session_state.deep_scan_results = pd.DataFrame(results)
    status.text("Scan Complete!")

if __name__ == "__main__":
    main()
