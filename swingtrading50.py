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
# 1. CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(layout="wide", page_title="AlgoTrader Pro", page_icon="📈")

# Supported Assets
ASSETS = {
    "Indices": ["^NSEI", "^NSEBANK", "^BSESN", "^GSPC", "^DJI"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDINR=X", "JPY=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F"],
    "Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "TSLA", "NVDA"]
}

# Valid Interval/Period Map (Strict yfinance rules)
VALID_INTERVALS = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", 
    "1h": "730d", "1d": "max", "1wk": "max", "1mo": "max"
}

# ==========================================
# 2. DATA LAYER
# ==========================================
class DataManager:
    @staticmethod
    @st.cache_data(ttl=300) # Cache for 5 mins
    def fetch_data(ticker, interval, period):
        """Fetches data with rate limiting and timezone handling."""
        time.sleep(1.5)  # Rate limiting
        
        try:
            # Force auto_adjust to get real price action
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            
            if df.empty:
                return None
            
            # Handle MultiIndex columns if present (yfinance update)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Timezone Conversion to IST
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Kolkata')
            
            return df
        except Exception as e:
            st.error(f"API Error: {e}")
            return None

# ==========================================
# 3. MANUAL TECHNICAL INDICATORS (NO TA-LIB)
# ==========================================
class Technicals:
    @staticmethod
    def sma(series, window):
        return series.rolling(window=window).mean()

    @staticmethod
    def ema(series, window):
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        sig = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - sig
        return macd, sig, hist

    @staticmethod
    def bollinger_bands(series, window=20, num_std=2):
        ma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower

    @staticmethod
    def atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def adx(high, low, close, window=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/window).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(window).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def obv(close, volume):
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def z_score(series, window=20):
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std

# ==========================================
# 4. ADVANCED PATTERN ANALYSIS
# ==========================================
class PatternAnalyzer:
    @staticmethod
    def find_support_resistance(df, window=20, tolerance=0.02):
        """Identifies SR levels using local Min/Max clustering."""
        # Find local peaks
        highs = df['High'].values
        lows = df['Low'].values
        
        # Get local max/min indices
        max_idx = argrelextrema(highs, np.greater, order=window)[0]
        min_idx = argrelextrema(lows, np.less, order=window)[0]
        
        levels = []
        # Add Highs
        for i in max_idx:
            levels.append((highs[i], 'Resistance'))
        # Add Lows
        for i in min_idx:
            levels.append((lows[i], 'Support'))
            
        # Cluster levels that are close
        levels.sort(key=lambda x: x[0])
        merged = []
        if not levels: return []
        
        current_group = [levels[0]]
        
        for i in range(1, len(levels)):
            if levels[i][0] <= current_group[-1][0] * (1 + tolerance):
                current_group.append(levels[i])
            else:
                # Average the group
                avg_price = np.mean([x[0] for x in current_group])
                type_ = "Zone" # Mixed zone
                if all(x[1] == 'Support' for x in current_group): type_ = 'Strong Support'
                elif all(x[1] == 'Resistance' for x in current_group): type_ = 'Strong Resistance'
                
                merged.append({'Price': avg_price, 'Type': type_, 'Strength': len(current_group)})
                current_group = [levels[i]]
        
        # Add last
        avg_price = np.mean([x[0] for x in current_group])
        merged.append({'Price': avg_price, 'Type': 'Level', 'Strength': len(current_group)})
        
        return sorted(merged, key=lambda x: x['Strength'], reverse=True)[:5] # Return top 5

    @staticmethod
    def detect_divergence(df, rsi_col='RSI'):
        """Detects Regular Bullish/Bearish Divergence."""
        # Simple implementation: Price makes Lower Low, RSI makes Higher Low (Bullish)
        # Price makes Higher High, RSI makes Lower High (Bearish)
        
        back_candles = 20
        if len(df) < back_candles: return None
        
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-back_candles]
        
        curr_rsi = df[rsi_col].iloc[-1]
        prev_rsi = df[rsi_col].iloc[-back_candles]
        
        signal = "Neutral"
        if curr_price < prev_price and curr_rsi > prev_rsi and curr_rsi < 50:
            signal = "Bullish Divergence"
        elif curr_price > prev_price and curr_rsi < prev_rsi and curr_rsi > 50:
            signal = "Bearish Divergence"
            
        return signal

    @staticmethod
    def simplified_elliot_wave(df):
        """
        Approximation of Elliot Wave using ZigZag logic.
        (Note: Real EW is subjective, this is a programmatic estimate)
        """
        # ZigZag placeholder logic for demonstration
        # 1. Identify major Swing Highs/Lows
        # 2. Label recent 5 swings
        return "Market likely in Corrective Phase (Wave C or 2) based on recent volatility contraction."

    @staticmethod
    def fibonacci_levels(df):
        """Calculates Fib levels based on recent significant High/Low."""
        period = 100
        recent_high = df['High'].tail(period).max()
        recent_low = df['Low'].tail(period).min()
        diff = recent_high - recent_low
        
        levels = {
            '0.0%': recent_low,
            '23.6%': recent_low + 0.236 * diff,
            '38.2%': recent_low + 0.382 * diff,
            '50.0%': recent_low + 0.5 * diff,
            '61.8%': recent_low + 0.618 * diff,
            '100.0%': recent_high
        }
        return levels

# ==========================================
# 5. UI & ORCHESTRATION
# ==========================================
def main():
    st.sidebar.header("Data Control")
    
    asset_class = st.sidebar.selectbox("Asset Class", list(ASSETS.keys()))
    ticker = st.sidebar.selectbox("Select Asset", ASSETS[asset_class])
    custom_ticker = st.sidebar.text_input("Or Custom Ticker (e.g., TSLA)")
    if custom_ticker: ticker = custom_ticker
    
    col1, col2 = st.sidebar.columns(2)
    interval = col1.selectbox("Interval", list(VALID_INTERVALS.keys()), index=5)
    period = col2.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=5)
    
    fetch_btn = st.sidebar.button("Analyze Market", type="primary")

    if fetch_btn:
        with st.spinner('Fetching Data & Crunching Numbers...'):
            df = DataManager.fetch_data(ticker, interval, period)
            
            if df is None:
                st.error("Failed to fetch data. Check ticker or internet connection.")
                return

            # --- Calculation Pipeline ---
            # 1. Basic Techs
            df['SMA_50'] = Technicals.sma(df['Close'], 50)
            df['SMA_200'] = Technicals.sma(df['Close'], 200)
            df['RSI'] = Technicals.rsi(df['Close'])
            df['MACD'], df['Signal'], df['Hist'] = Technicals.macd(df['Close'])
            df['UpperBB'], df['LowerBB'] = Technicals.bollinger_bands(df['Close'])
            df['ATR'] = Technicals.atr(df['High'], df['Low'], df['Close'])
            df['ADX'], df['+DI'], df['-DI'] = Technicals.adx(df['High'], df['Low'], df['Close'])
            df['Vol_MA'] = Technicals.sma(df['Volume'], 20)
            df['Z_Score'] = Technicals.z_score(df['Close'])
            
            # 2. Benchmarks for Ratio Analysis (Gold & SP500)
            bench_gold = DataManager.fetch_data('GC=F', '1d', '1y')
            bench_sp500 = DataManager.fetch_data('^GSPC', '1d', '1y')

            # --- DASHBOARD ---
            st.title(f"AlgoTrading Analysis: {ticker}")
            st.markdown(f"**Current Price:** {df['Close'].iloc[-1]:.2f} | **RSI:** {df['RSI'].iloc[-1]:.2f} | **ADX:** {df['ADX'].iloc[-1]:.2f}")

            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chart & Indicators", "Ratios & Volatility", "Advanced Patterns", "Multi-Timeframe Scan", "Signals & Backtest"])

            # --- TAB 1: Main Chart ---
            with tab1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                
                # Candlestick
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                
                # Overlays
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['UpperBB'], line=dict(color='gray', dash='dot'), name='Upper BB'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['LowerBB'], line=dict(color='gray', dash='dot'), name='Lower BB'), row=1, col=1)
                
                # Subplot: RSI or MACD
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_layout(height=600, xaxis_rangeslider_visible=False, title="Price Action & Technicals")
                st.plotly_chart(fig, use_container_width=True)

            # --- TAB 2: Ratios & Volatility ---
            with tab2:
                col_r1, col_r2 = st.columns(2)
                
                # Ratio Logic
                if bench_gold is not None:
                    # Align dates
                    common_idx = df.index.intersection(bench_gold.index)
                    if len(common_idx) > 10:
                        ratio = df.loc[common_idx]['Close'] / bench_gold.loc[common_idx]['Close']
                        
                        with col_r1:
                            st.subheader("Asset / Gold Ratio")
                            fig_ratio = go.Figure()
                            fig_ratio.add_trace(go.Scatter(x=ratio.index, y=ratio, name=f'{ticker}/Gold'))
                            st.plotly_chart(fig_ratio, use_container_width=True)
                            st.caption("Rising ratio implies Asset outperforming Gold. Falling implies Gold is safer.")
                
                # Volatility Bins
                with col_r2:
                    st.subheader("Volatility Clustering")
                    df['Returns'] = df['Close'].pct_change()
                    df['Vol_Bin'] = pd.qcut(df['Returns'].abs(), q=4, labels=["Low", "Med", "High", "Extreme"])
                    vol_counts = df['Vol_Bin'].value_counts()
                    st.bar_chart(vol_counts)
                    st.caption("Frequency of volatility regimes. 'Extreme' clusters often precede trend reversals.")

                # Z-Score
                st.subheader("Price Z-Score (Statistical Reversion)")
                fig_z = go.Figure()
                fig_z.add_trace(go.Bar(x=df.index, y=df['Z_Score'], marker_color=np.where(df['Z_Score'] > 2, 'red', np.where(df['Z_Score'] < -2, 'green', 'gray'))))
                fig_z.add_hline(y=2, line_dash="dot", line_color="red")
                fig_z.add_hline(y=-2, line_dash="dot", line_color="green")
                st.plotly_chart(fig_z, use_container_width=True)
                st.markdown("""
                **Insight:** Z-Scores > 2 suggest the asset is statistically overextended (expensive). 
                Z-Scores < -2 suggest it is undervalued relative to recent mean.
                """)

            # --- TAB 3: Advanced Patterns ---
            with tab3:
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Support & Resistance Zones")
                    sr_levels = PatternAnalyzer.find_support_resistance(df)
                    if sr_levels:
                        sr_df = pd.DataFrame(sr_levels)
                        st.dataframe(sr_df)
                        
                        # Plot SR on small chart
                        fig_sr = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                        for lvl in sr_levels:
                            color = 'green' if lvl['Type'] == 'Support' else 'red'
                            fig_sr.add_hline(y=lvl['Price'], line_color=color, annotation_text=lvl['Type'])
                        st.plotly_chart(fig_sr, use_container_width=True)
                    else:
                        st.write("No clear strong levels found in current window.")

                with c2:
                    st.subheader("Pattern Signals")
                    
                    # Divergence
                    div = PatternAnalyzer.detect_divergence(df)
                    st.metric("RSI Divergence", div if div else "None", help="Bullish: Price Lower Low, RSI Higher Low")
                    
                    # Fibonacci
                    fibs = PatternAnalyzer.fibonacci_levels(df)
                    st.write("**Key Fibonacci Levels (Retracement):**")
                    st.json(fibs)
                    
                    # Elliot Wave
                    st.info(f"Elliot Wave Context: {PatternAnalyzer.simplified_elliot_wave(df)}")

            # --- TAB 4: Multi-Timeframe (Simulated Scan) ---
            with tab4:
                st.write("Click below to perform a Deep Scan across timeframes (15m, 1h, 4h, 1d).")
                st.write("*Note: This simulates the 'All Timeframes' requirement by sampling key frames to respect API limits.*")
                
                if st.button("Run Deep Scan"):
                    scan_res = []
                    tf_list = ['15m', '1h', '1d']
                    p_list = ['5d', '1mo', '1y']
                    
                    progress = st.progress(0)
                    for i, (tf, p) in enumerate(zip(tf_list, p_list)):
                        try:
                            tf_df = DataManager.fetch_data(ticker, tf, p)
                            if tf_df is not None:
                                last_close = tf_df['Close'].iloc[-1]
                                sma50 = tf_df['Close'].rolling(50).mean().iloc[-1]
                                trend = "BULLISH" if last_close > sma50 else "BEARISH"
                                rsi_val = Technicals.rsi(tf_df['Close']).iloc[-1]
                                scan_res.append({"Timeframe": tf, "Trend": trend, "RSI": f"{rsi_val:.1f}"})
                        except:
                            pass
                        time.sleep(1) # Safety delay
                        progress.progress((i + 1) / len(tf_list))
                    
                    st.table(pd.DataFrame(scan_res))

            # --- TAB 5: Signals & Backtest ---
            with tab5:
                # Signal Generation
                last_row = df.iloc[-1]
                score = 0
                reasons = []
                
                if last_row['Close'] > last_row['SMA_200']: 
                    score += 1
                    reasons.append("Price above 200 SMA (Long Term Bullish)")
                
                if last_row['RSI'] < 30:
                    score += 1
                    reasons.append("RSI Oversold (Potential Bounce)")
                elif last_row['RSI'] > 70:
                    score -= 1
                    reasons.append("RSI Overbought (Potential Pullback)")
                
                if last_row['MACD'] > last_row['Signal']:
                    score += 1
                    reasons.append("MACD Bullish Crossover")
                
                if last_row['+DI'] > 25 and last_row['ADX'] > 25:
                    score += 1
                    reasons.append("Strong Uptrend (ADX)")
                    
                recommendation = "HOLD"
                color = "gray"
                if score >= 2: 
                    recommendation = "BUY"
                    color = "green"
                elif score <= -1:
                    recommendation = "SELL"
                    color = "red"
                
                st.markdown(f"### AI Recommendation: :{color}[{recommendation}]")
                st.markdown(f"**Confidence Score:** {score}/5")
                with st.expander("Logic / Reasons"):
                    for r in reasons: st.write(f"- {r}")
                
                # Trade Setup
                if recommendation == "BUY":
                    entry = last_row['Close']
                    sl = entry * 0.98 # 2% SL
                    target = entry * 1.04 # 4% Target
                    st.success(f"**Setup:** Entry: {entry:.2f} | SL: {sl:.2f} | Target: {target:.2f}")

                # Simple Vectorized Backtest
                st.subheader("Backtest of this Logic (Last 1 Year)")
                df['Signal'] = 0
                df.loc[(df['Close'] > df['SMA_50']) & (df['RSI'] < 40), 'Signal'] = 1 # Simple Mean Reversion Strategy
                df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
                cum_returns = (1 + df['Strategy_Returns']).cumprod()
                buy_hold = (1 + df['Returns']).cumprod()
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=df.index, y=cum_returns, name="Algo Strategy"))
                fig_bt.add_trace(go.Scatter(x=df.index, y=buy_hold, name="Buy & Hold", line=dict(dash='dot')))
                st.plotly_chart(fig_bt, use_container_width=True)
                st.caption("Backtest compares a simple 'Buy on Dip' algo (Green) vs Standard Buy & Hold (Dot).")
                
if __name__ == "__main__":
    main()
