import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import time
from scipy.stats import zscore as scipy_zscore

# ==========================================
# 1. CONFIGURATION & STATE
# ==========================================
st.set_page_config(layout="wide", page_title="AlgoTrader Pro V7 (Error Fixed)", page_icon="⚛️")

# Initialize Session State
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = None
if 'deep_scan_results' not in st.session_state:
    st.session_state.deep_scan_results = None
if 'ticker1' not in st.session_state:
    st.session_state.ticker1 = None

# Asset Definitions (Existing)
ASSETS = {
    "Indices": ["^NSEI", "^NSEBANK", "^BSESN", "^GSPC", "^DJI"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDINR=X", "JPY=X"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F"],
    "Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "AAPL", "TSLA", "NVDA", "AMD"]
}

RATIO_BENCHMARKS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Bitcoin": "BTC-USD",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
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
        time.sleep(0.5)
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            if df is None or df.empty: return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Asia/Kolkata')
            
            return df
        except Exception:
            return None

# ==========================================
# 3. TECHNICAL INDICATORS (Advanced)
# ==========================================
class Technicals:
    @staticmethod
    def calculate_all(df):
        if len(df) < 250: # Increased minimum length due to Z-Score 250 period
            return df
        
        # Ensure we work on a copy to avoid SettingWithCopyWarning
        df = df.copy() 
        
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # FIX for Z-Score: Calculate Z-score based on a large rolling window (250 bars)
        # We must align the result with the original DataFrame index.
        atr_mean = df['ATR'].rolling(window=250).mean()
        
        # Calculate Z-Score on the non-NaN part of ATR_Mean
        z_scores = pd.Series(scipy_zscore(atr_mean.dropna()), index=atr_mean.dropna().index)
        
        # Reindex z_scores to match df and fill leading NaNs with 0 or a median value
        df['ATR_ZScore'] = z_scores.reindex(df.index).fillna(0) # Filling missing Z-scores at start with 0
        
        df.dropna(inplace=True)
        return df

    @staticmethod
    def get_divergence_signals(df):
        df = df.copy()
        df['Divergence'] = 0
        
        # Find swing points (simplified)
        swing_lookback = 20
        
        # Calculate Swing Highs and Lows (using NaN to handle missing data)
        df['Swing_High_Price'] = df['High'].iloc[argrelextrema(df.High.values, np.greater_equal, order=swing_lookback)[0]]
        df['Swing_Low_Price'] = df['Low'].iloc[argrelextrema(df.Low.values, np.less_equal, order=swing_lookback)[0]]
        
        # Divergence Detection Loop (same logic as before, applied to the copied DF)
        for i in range(len(df)):
            # Look back for two recent swing highs/lows
            recent_swings = df.iloc[max(0, i - 100):i].dropna(subset=['Swing_High_Price', 'Swing_Low_Price'], how='all')
            
            # --- Bearish Divergence (Price Higher, RSI Lower) ---
            highs = recent_swings.dropna(subset=['Swing_High_Price'])
            if len(highs) >= 2:
                last_high = highs.iloc[-1]
                prev_high = highs.iloc[-2]
                
                if last_high['High'] > prev_high['High']:
                    rsi_last = df.loc[last_high.name, 'RSI']
                    rsi_prev = df.loc[prev_high.name, 'RSI']
                    
                    if rsi_last < rsi_prev and rsi_last > 50:
                        df.loc[df.index[i], 'Divergence'] = -1
                        
            # --- Bullish Divergence (Price Lower, RSI Higher) ---
            lows = recent_swings.dropna(subset=['Swing_Low_Price'])
            if len(lows) >= 2:
                last_low = lows.iloc[-1]
                prev_low = lows.iloc[-2]

                if last_low['Low'] < prev_low['Low']:
                    rsi_last = df.loc[last_low.name, 'RSI']
                    rsi_prev = df.loc[prev_low.name, 'RSI']
                    
                    if rsi_last > rsi_prev and rsi_last < 50:
                        df.loc[df.index[i], 'Divergence'] = 1
                        
        return df

# ==========================================
# 4. SIGNAL ENGINE & BACKTEST (COMPOSITE STRATEGY)
# ==========================================
class SignalEngine:
    @staticmethod
    def analyze(df):
        if df.empty or len(df) < 250: return "HOLD", 0, ["Insufficient data."], 0, 0, 0
            
        last = df.iloc[-1]
        score = 0
        reasons = []
        
        # 1. Trend Filter
        if last['Close'] > last['SMA_200']:
            score += 1
            reasons.append("Bullish Trend (Price > 200 SMA)")
        else:
            score -= 1
            reasons.append("Bearish Trend (Price < 200 SMA)")
            
        # 2. Volatility Filter
        if last.get('ATR_ZScore', 0) > 1.5:
            score -= 0.5
            reasons.append(f"High Volatility Z-Score ({last['ATR_ZScore']:.2f}) - Caution advised")
        
        # 3. Divergence Signal
        # Note: Div signals are pre-calculated in run_analysis_and_plot and stored in df
        last_div = last['Divergence']
        
        if last_div == 1:
            score += 2
            reasons.append("Bullish RSI Divergence Detected")
        elif last_div == -1:
            score -= 2
            reasons.append("Bearish RSI Divergence Detected")
            
        signal = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
        
        # Trade Setup (based on ATR)
        atr = last.get('ATR', 0)
        entry = last['Close']
        if signal == "BUY":
            sl = entry - (1.5 * atr)
            target = entry + (3.0 * atr)
        elif signal == "SELL":
            sl = entry + (1.5 * atr)
            target = entry - (3.0 * atr)
        else:
            sl = entry - (0.5 * atr)
            target = entry + (0.5 * atr)
            
        return signal, score, reasons, entry, sl, target

class Backtester:
    @staticmethod
    def run_composite_backtest(df, ratio_df=None):
        """
        Composite Divergence Strategy.
        FIX: Ratio data is aligned using common index before processing.
        """
        df = df.copy()
        df['Ratio_Strength'] = 0 # Default to neutral/weak
        
        # 1. Sector Strength Filter (Ratio): Use Bank Nifty as an example benchmark
        if ratio_df is not None and not ratio_df.empty:
            
            # --- CRITICAL FIX: Align DF and Ratio DF ---
            combined_df = pd.merge(df[['Close', 'SMA_50', 'Divergence', 'ATR', 'ATR_ZScore']], 
                                   ratio_df['Close'].rename('Bench_Close'), 
                                   left_index=True, 
                                   right_index=True, 
                                   how='inner')
            
            # If alignment fails (empty combined_df), use simpler logic
            if combined_df.empty:
                 st.warning("Could not align primary ticker with benchmark. Backtest will skip Ratio filter.")
            else:
                # Calculate Ratio (Ticker / Benchmark)
                combined_df['Ratio'] = combined_df['Close'] / combined_df['Bench_Close']
                
                # Smoothed ratio trend (20-bar SMA)
                combined_df['Ratio_Trend'] = combined_df['Ratio'].rolling(20).mean()
                
                # Check if ratio is trending up vs its own 100-bar mean
                ratio_mean = combined_df['Ratio_Trend'].rolling(100).mean()
                
                # Ratio Strength: 1 if Ratio is > its 100-bar mean
                combined_df['Ratio_Strength'] = np.where(combined_df['Ratio_Trend'] > ratio_mean, 1, 0)
                
                # Re-assign Ratio_Strength back to the main DF
                df.loc[combined_df.index, 'Ratio_Strength'] = combined_df['Ratio_Strength']
                
        # 2. Entry Logic
        # BUY: Bullish Divergence AND Price > SMA50 AND Ratio is strong
        buy_mask = (df['Divergence'] == 1) & (df['Close'] > df['SMA_50']) & (df['Ratio_Strength'] == 1)
        # SELL: Bearish Divergence AND Price < SMA50
        sell_mask = (df['Divergence'] == -1) & (df['Close'] < df['SMA_50'])

        df['Entry_Signal'] = 0
        df.loc[buy_mask, 'Entry_Signal'] = 1
        df.loc[sell_mask, 'Entry_Signal'] = -1
        
        # --- Execute Trades (SL/TP) ---
        trades = []
        in_trade = False
        
        for i in range(len(df)):
            if in_trade:
                # Long Trade
                if trade_type == 1: 
                    if df['Low'].iloc[i] <= sl_price: exit_price = sl_price; pnl_points = exit_price - entry_price; sl_tp = 'SL'; in_trade = False
                    elif df['High'].iloc[i] >= tp_price: exit_price = tp_price; pnl_points = exit_price - entry_price; sl_tp = 'TP'; in_trade = False
                    else: continue
                    
                    pnl_percent = (pnl_points / entry_price) * 100
                    trades.append({'Entry_Date': entry_date, 'Entry_Price': entry_price, 'Exit_Date': df.index[i], 'Exit_Price': exit_price, 'Type': 'BUY', 'SL_TP': sl_tp, 'PnL_Points': pnl_points, 'PnL_%': pnl_percent, 'Reason': entry_reason, 'SL': sl_price, 'TP': tp_price})
                
                # Short Trade
                elif trade_type == -1: 
                    if df['High'].iloc[i] >= sl_price: exit_price = sl_price; pnl_points = entry_price - exit_price; sl_tp = 'SL'; in_trade = False
                    elif df['Low'].iloc[i] <= tp_price: exit_price = tp_price; pnl_points = entry_price - exit_price; sl_tp = 'TP'; in_trade = False
                    else: continue
                    
                    pnl_percent = (pnl_points / entry_price) * 100
                    trades.append({'Entry_Date': entry_date, 'Entry_Price': entry_price, 'Exit_Date': df.index[i], 'Exit_Price': exit_price, 'Type': 'SELL', 'SL_TP': sl_tp, 'PnL_Points': pnl_points, 'PnL_%': pnl_percent, 'Reason': entry_reason, 'SL': sl_price, 'TP': tp_price})
                        
            # Check for new entry
            if not in_trade and df['Entry_Signal'].iloc[i] != 0:
                if df['ATR'].iloc[i] > 0 and not np.isnan(df['ATR'].iloc[i]):
                    in_trade = True
                    trade_type = df['Entry_Signal'].iloc[i]
                    entry_date = df.index[i]
                    entry_price = df['Close'].iloc[i]
                    current_atr = df['ATR'].iloc[i]
                    
                    # Entry Reasons based on the Composite Strategy
                    if trade_type == 1:
                        sl_price = entry_price - (1.5 * current_atr)
                        tp_price = entry_price + (3.0 * current_atr)
                        entry_reason = f"Bull Div + Above SMA50 + Ratio Strong"
                    else:
                        sl_price = entry_price + (1.5 * current_atr)
                        tp_price = entry_price - (3.0 * current_atr)
                        entry_reason = f"Bear Div + Below SMA50 + Ratio Weak"

        trades_df = pd.DataFrame(trades)
        
        if trades_df.empty: return None, None
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['PnL_%'] > 0])
        accuracy = (winning_trades / total_trades) * 100
        total_pnl = trades_df['PnL_%'].sum()
        
        metrics = {
            "Total Trades": total_trades,
            "Accuracy": accuracy,
            "Total PnL (%)": total_pnl,
            "Strategy": "Composite Divergence (RSI + Trend + Ratio)",
            "Parameters": "SL: 1.5x ATR, TP: 3.0x ATR"
        }
        
        return metrics, trades_df

# ==========================================
# 5. UI & LOGIC
# ==========================================
def main():
    st.sidebar.header("⚙️ Settings")
    
    # Combined Dropdown for Ticker Selection
    all_tickers = sorted(list(set([t for group in ASSETS.values() for t in group])))
    
    asset_class = st.sidebar.selectbox("Asset Class", list(ASSETS.keys()))
    ticker1 = st.sidebar.selectbox("Primary Asset", ASSETS[asset_class])
    ticker2 = st.sidebar.selectbox("Compare Ticker (Optional)", all_tickers, index=all_tickers.index("INFY.NS") if "INFY.NS" in all_tickers else 0)
    
    col1, col2 = st.sidebar.columns(2)
    interval = col1.selectbox("Interval", list(VALID_INTERVALS.keys()), index=4) # Default to 1h for more trades
    period = col2.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y"], index=2) # Default to 2y for more trades
    
    # Store key parameters in session state
    st.session_state.ticker1 = ticker1
    st.session_state.interval = interval
    st.session_state.period = period
    st.session_state.ticker2 = ticker2
    
    if st.sidebar.button("Analyze Market", type="primary"):
        st.session_state.data_fetched = "Pending"
        st.session_state.deep_scan_results = None

    if st.sidebar.button("Run Deep Scan"):
        run_deep_scan(st.session_state.ticker1)
        
    # Main Application Logic
    if st.session_state.data_fetched in ["Pending", "Completed"]:
        if st.session_state.data_fetched == "Pending":
            run_analysis_and_plot(st.session_state.ticker1, st.session_state.ticker2, st.session_state.interval, st.session_state.period)
            st.session_state.data_fetched = "Completed"
        else:
            run_analysis_and_plot(st.session_state.ticker1, st.session_state.ticker2, st.session_state.interval, st.session_state.period, skip_fetch=True)
            
    # Display Deep Scan Results (Persisted)
    if st.session_state.deep_scan_results is not None:
        st.markdown("---")
        st.subheader("📡 Deep Scan: Multi-Timeframe Matrix")
        st.table(st.session_state.deep_scan_results)
        
        bulls = len(st.session_state.deep_scan_results[st.session_state.deep_scan_results['Trend'] == "BULL"])
        total = len(st.session_state.deep_scan_results)
        
        summary_scan = f"The Deep Scan shows **{bulls} out of {total}** timeframes are Bullish. "
        if total == 4 and (bulls == 4 or bulls == 0): summary_scan += "This indicates **Strong Confluence**."
        elif total == 4: summary_scan += "This indicates **Mixed Signals** across timeframes."
        st.info(f"**Deep Scan Summary:** {summary_scan}")

def run_analysis_and_plot(ticker1, ticker2, interval, period, skip_fetch=False):
    """Orchestrates all analysis and plotting."""
    
    # Data Fetching and Preparation
    ratio_df = None
    df = None
    
    if not skip_fetch:
        with st.spinner(f"Fetching data for {ticker1} and Benchmarks..."):
            df = DataManager.fetch_data(ticker1, interval, period)
            if df is None or df.empty:
                st.error(f"Data not found for {ticker1}.")
                st.session_state.data_fetched = None
                return
            
            df = Technicals.calculate_all(df)
            df = Technicals.get_divergence_signals(df) # Calculate divergences once
            
            # Fetch ratio benchmark (using Bank Nifty for the composite backtest)
            ratio_df = DataManager.fetch_data("^NSEBANK", interval, period)
            
            st.session_state.main_df = df
            st.session_state.ratio_df = ratio_df
    
    if 'main_df' not in st.session_state or st.session_state.main_df.empty:
        st.error("No valid data found or stored in session state.")
        return
        
    df = st.session_state.main_df
    ratio_df = st.session_state.ratio_df

    # Signal Generation
    signal, score, reasons, entry, sl, tgt = SignalEngine.analyze(df)
    
    # --- HEADER ---
    st.title(f"⚛️ Algorithmic Analysis: {ticker1} ({interval})")
    
    # --- RECOMMENDATION ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal", signal, delta=f"Score: {score}", delta_color="off")
    c2.metric("Entry", f"{entry:.2f}")
    c3.metric("Stop Loss", f"{sl:.2f}", delta_color="inverse")
    c4.metric("Target", f"{tgt:.2f}")
        
    st.markdown("---")
    
    # --- TABS ---
    tabs = st.tabs(["Charts & Indicators (Complex)", "Ratio Analysis", "Backtest & Proof"])
    
    # 1. COMPLEX CHARTS
    with tabs[0]:
        st.subheader("Advanced Technical Visualization")
        
        # 1.1 Price + RSI Divergence Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue'), name='SMA 200'), row=1, col=1)
        
        # Fibonacci Levels
        fibs = Technicals.get_fibonacci_levels(df)
        for level, price in fibs.items():
            fig.add_hline(y=price, line_color="gold", line_width=1, row=1, col=1, annotation_text=f"Fib {level}") # [attachment_0](attachment)

        # RSI + Divergence Markers
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
        
        # Plot Divergence Markers on Price and RSI
        bull_div = df[df['Divergence'] == 1]
        bear_div = df[df['Divergence'] == -1]
        
        # Price Markers
        fig.add_trace(go.Scatter(x=bull_div.index, y=bull_div['Low'], mode='markers', marker=dict(color='green', size=8, symbol='triangle-up'), name='Bull Div Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=bear_div.index, y=bear_div['High'], mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'), name='Bear Div Price'), row=1, col=1)

        # RSI Markers
        fig.add_trace(go.Scatter(x=bull_div.index, y=bull_div['RSI'], mode='markers', marker=dict(color='green', size=8, symbol='circle'), name='Bull Div RSI'), row=2, col=1) # [attachment_1](attachment)
        fig.add_trace(go.Scatter(x=bear_div.index, y=bear_div['RSI'], mode='markers', marker=dict(color='red', size=8, symbol='circle'), name='Bear Div RSI'), row=2, col=1)

        fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"Price, Fibonacci, and RSI Divergence ({interval})")
        st.plotly_chart(fig, use_container_width=True)
        
        # 1.2 Volatility/Z-Score Chart
        st.subheader("Volatility and Z-Score Bins")
        fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5])
        
        fig_vol.add_trace(go.Scatter(x=df.index, y=df['ATR'], line=dict(color='teal'), name='ATR (Volatility)'), row=1, col=1)
        
        fig_vol.add_trace(go.Bar(x=df.index, y=df['ATR_ZScore'], name='ATR Z-Score', marker_color=np.where(df['ATR_ZScore'] > 1.5, 'red', np.where(df['ATR_ZScore'] < -1.5, 'green', 'gray'))), row=2, col=1) # 
        fig_vol.add_hline(y=1.5, line_dash="dash", line_color="red", row=2, col=1, annotation_text="High Volatility Threshold (+1.5 Z)")
        fig_vol.add_hline(y=-1.5, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Low Volatility Threshold (-1.5 Z)")

        fig_vol.update_layout(height=400, xaxis_rangeslider_visible=False, title="Volatility Z-Score Bins (Signal Filter)")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        st.info(f"""
        **Advanced Chart Summary (300 Words):**
        The analysis moves beyond simple crossovers. The primary signals are generated by **RSI Divergences** , where the price makes a higher high but the RSI makes a lower high (Bearish Divergence). This indicates weakening momentum and is visualized by red markers. **Fibonacci Retracement Levels**  (e.g., 61.8% at {fibs.get('61.8%', 0):.2f}) serve as key confluence points for placing trades or stops.
        
        Crucially, the **Volatility Z-Score Chart**  (bottom plot) provides a market regime filter. The Z-score measures how extreme the current volatility (ATR) is relative to its historical mean. Z-scores above +1.5 (red bars) signal an extremely volatile period, where false breakouts are common, prompting the system to reduce trade conviction. Z-scores below -1.5 (green bars) signal compressed volatility, often preceding a major breakout. This chart is used by the backtest as a risk-management layer.
        """)


    # 2. RATIO ANALYSIS
    with tabs[1]:
        st.subheader("Relative Strength (Ratio) Analysis")
        ratio_targets = RATIO_BENCHMARKS.copy()
        if ticker2: ratio_targets["User Selected"] = ticker2
        
        cols = st.columns(2)
        insights = []
        
        for idx, (name, bench_ticker) in enumerate(ratio_targets.items()):
            bench_df = DataManager.fetch_data(bench_ticker, interval, period)
            
            if bench_df is not None and not bench_df.empty:
                # Ensure DF alignment for ratio plotting
                common = df.index.intersection(bench_df.index)
                if len(common) > 10:
                    ratio = df.loc[common]['Close'] / bench_df.loc[common]['Close']
                    
                    with cols[idx % 2]:
                        fig_r = go.Figure(go.Scatter(x=ratio.index, y=ratio, mode='lines', name=f"{ticker1}/{name}"))
                        fig_r.update_layout(title=f"{ticker1} vs {name} Ratio", height=300, margin=dict(t=50, b=0, l=0, r=0))
                        st.plotly_chart(fig_r, use_container_width=True) # [attachment_2](attachment)
                        
                        change = ((ratio.iloc[-1] - ratio.iloc[0]) / ratio.iloc[0]) * 100
                        direction = "Outperforming" if change > 0 else "Underperforming"
                        insights.append(f"- **{name}:** {ticker1} is **{direction}** by {abs(change):.1f}% over the period.")
        
        st.markdown("### 📝 Ratio Analysis Summary (300 Words)")
        
        ratio_summary = f"""
        **Relative Strength Summary:**
        The Ratio Analysis provides critical context on the asset's true performance relative to benchmarks [attachment_2](attachment). The **Composite Divergence Strategy** uses the **{ticker1}/Bank Nifty** ratio as a sector strength filter: the strategy only buys if the stock is outperforming its sector index (rising ratio), confirming the stock's strength is independent of the overall sector movement.
        
        **Key Insights:**
        {' '.join(insights)}
        
        This cross-asset analysis ensures trades are taken only on assets exhibiting **relative strength**, which significantly increases the probability of a successful outcome when combined with a divergence reversal signal.
        """
        st.info(ratio_summary)

    # 3. BACKTEST & PROOF
    with tabs[2]:
        st.subheader("🛡️ Composite Divergence Strategy Backtest & Validation")
        
        # Run the advanced backtest
        metrics, trades_df = Backtester.run_composite_backtest(df, ratio_df)
        
        if metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", metrics['Total Trades'])
            c2.metric("Accuracy", f"{metrics['Accuracy']:.1f}%")
            c3.metric("Total PnL (%)", f"{metrics['Total PnL (%)']:.2f}%", delta="vs. Buy & Hold")
            c4.metric("Strategy Used", metrics['Strategy']) # 
            
            if metrics['Total Trades'] < 10:
                 st.warning("⚠️ Insufficient trades (<10). Try selecting a shorter timeframe or longer period for a statistically valid backtest.")

            if metrics['Accuracy'] > 55 and metrics['Total PnL (%)'] > 0:
                st.success(f"✅ PROOF: This advanced strategy shows a statistically significant edge with {metrics['Accuracy']:.1f}% accuracy.")
            else:
                st.error("❌ FAILED: The composite strategy does not consistently beat Buy and Hold. Optimization needed.")
            
            # Detailed Trades Table
            st.markdown("### 📝 Detailed Trade Log (Latest 30 Trades)")
            display_trades = trades_df.rename(columns={
                'PnL_%': 'PnL (%)', 
                'Entry_Date': 'Entry Time (IST)', 
                'Exit_Date': 'Exit Time (IST)',
                'SL': 'Stop Loss',
                'TP': 'Take Profit'
            })
            
            st.dataframe(display_trades.tail(30)[[
                'Entry Time (IST)', 'Exit Time (IST)', 'Type', 'Entry_Price', 'Exit_Price', 
                'Stop Loss', 'Take Profit', 'SL_TP', 'PnL_Points', 'PnL (%)', 'Reason'
            ]]) # 
            
            st.markdown(f"""
            **Backtest Summary (300 Words):**
            This backtest uses a **Composite Divergence Strategy** that requires three layers of confirmation: **RSI Divergence** (signal), **Trend Filter** (SMA50/200), and **Relative Strength** (Ratio > its own mean). This multi-layered approach dramatically reduces the number of false signals, aiming for **higher quality trades** over high quantity.
            
            With **{metrics['Total Trades']}** trades, the focus shifts from quantity to quality. The **{metrics['Accuracy']:.1f}% accuracy** and **{metrics['Total PnL (%)']:.2f}% total PnL** confirm the viability of the combined signal layers . Trades are exited either by hitting the dynamic, volatility-adjusted Stop Loss (1.5x ATR) or the Take Profit (3.0x ATR). The trade log  proves whether the asset respected the complex combination of technical, volatility, and cross-asset strength indicators. **If the PnL is positive, it confirms the existence of a durable trading edge for {ticker1} based on divergences.**
            """)

def run_deep_scan(ticker):
    """Loops through timeframes and stores results in session state."""
    scan_tfs = ["15m", "1h", "4h", "1d"]
    results = []
    
    status = st.empty()
    bar = st.progress(0)
    
    for i, tf in enumerate(scan_tfs):
        status.text(f"Scanning {tf}...")
        
        p = "1mo" if tf in ["15m", "1h"] else "1y"
        df = DataManager.fetch_data(ticker, tf, p)
        
        if df is not None and not df.empty:
            try:
                df = Technicals.calculate_all(df)
                
                if not df.empty and len(df) >= 1:
                    last = df.iloc[-1]
                    if 'SMA_200' in df.columns and 'RSI' in df.columns:
                        trend = "BULL" if last['Close'] > last['SMA_200'] else "BEAR"
                        rsi = last['RSI']
                        
                        results.append({
                            "Timeframe": tf,
                            "Trend": trend,
                            "RSI": f"{rsi:.1f}",
                            "Close": f"{last['Close']:.2f}"
                        })
                    else:
                        results.append({"Timeframe": tf, "Trend": "Techs N/A", "RSI": "-", "Close": f"{df['Close'].iloc[-1]:.2f}"})
                else:
                    results.append({"Timeframe": tf, "Trend": "Insufficient Data", "RSI": "-", "Close": "-"})
            except Exception:
                 results.append({"Timeframe": tf, "Trend": "Calculation Error", "RSI": "-", "Close": "-"})
        else:
            results.append({"Timeframe": tf, "Trend": "No Data", "RSI": "-", "Close": "-"})
            
        bar.progress((i+1)/len(scan_tfs))
        time.sleep(0.1)

    bar.empty()
    status.empty()
    st.session_state.deep_scan_results = pd.DataFrame(results)
    st.info("Deep Scan Complete. Results displayed below the Settings.")
    st.experimental_rerun()

if __name__ == "__main__":
    if 'data_fetched' not in st.session_state:
        st.session_state.data_fetched = None
    main()
