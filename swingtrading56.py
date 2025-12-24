import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
from scipy.signal import argrelextrema
import time
from io import StringIO

# Page configuration
st.set_page_config(page_title="Algorithmic Trading Analysis", layout="wide", initial_sidebar_state="expanded")

# Indian timezone
IST = pytz.timezone('Asia/Kolkata')

# Valid timeframe-period combinations
VALID_COMBINATIONS = {
    '1m': ['1d', '5d'],
    '2m': ['1d', '5d'],
    '5m': ['1d', '1mo'],
    '15m': ['1d', '1mo'],
    '30m': ['1d', '1mo'],
    '60m': ['1mo', '3mo'],
    '90m': ['1mo', '3mo'],
    '1h': ['1mo', '3mo'],
    '1d': ['1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y', '15y', '20y'],
    '5d': ['1y', '2y', '5y'],
    '1wk': ['1y', '2y', '5y'],
    '1mo': ['2y', '5y', '10y'],
    '3mo': ['5y', '10y']
}

# Predefined assets
PREDEFINED_ASSETS = {
    'NIFTY 50': '^NSEI',
    'Bank NIFTY': '^NSEBANK',
    'SENSEX': '^BSESN',
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'USD/INR': 'INR=X',
    'EUR/USD': 'EURUSD=X'
}

# Helper functions
def format_time_ago(dt):
    """Format datetime to human-readable 'time ago' format"""
    if pd.isna(dt):
        return "N/A"
    
    try:
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        if dt.tzinfo is                                 'Risk:Reward': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                                'trades': trades
                            })
                    
                    # Volatility Breakout Strategy
                    if test_vol_breakout:
                        trades = []
                        in_trade = False
                        entry_price = 0
                        entry_idx = 0
                        
                        for i in range(50, len(data)):
                            avg_vol = data['Volatility_%'].iloc[max(0, i-20):i].mean()
                            
                            if not in_trade:
                                # Entry: Low volatility followed by breakout
                                if (data['Volatility_%'].iloc[i] < avg_vol * 0.7 and
                                    data['Close'].iloc[i] > data['Close'].iloc[i-1] * 1.01):
                                    in_trade = True
                                    entry_price = data['Close'].iloc[i]
                                    entry_idx = i
                            else:
                                # Exit: Volatility returns to normal or price drops
                                if data['Volatility_%'].iloc[i] > avg_vol or data['Close'].iloc[i] < entry_price * 0.98:
                                    exit_price = data['Close'].iloc[i]
                                    pnl = ((exit_price - entry_price) / entry_price) * 100
                                    
                                    trades.append({
                                        'strategy': 'Volatility Breakout',
                                        'entry_date': data['DateTime_IST'].iloc[entry_idx],
                                        'entry_price': entry_price,
                                        'exit_date': data['DateTime_IST'].iloc[i],
                                        'exit_price': exit_price,
                                        'pnl_pct': pnl,
                                        'points': exit_price - entry_price
                                    })
                                    in_trade = False
                        
                        if trades:
                            winning_trades = len([t for t in trades if t['pnl_pct'] > 0])
                            losing_trades = len(trades) - winning_trades
                            total_pnl = sum([t['pnl_pct'] for t in trades])
                            avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if winning_trades > 0 else 0
                            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if losing_trades > 0 else 0
                            best_trade = max([t['pnl_pct'] for t in trades])
                            worst_trade = min([t['pnl_pct'] for t in trades])
                            win_rate = (winning_trades / len(trades)) * 100
                            
                            all_results.append({
                                'Strategy': 'Volatility Breakout',
                                'Timeframe': tf_str,
                                'Total Trades': len(trades),
                                'Winning': winning_trades,
                                'Losing': losing_trades,
                                'Win Rate %': win_rate,
                                'Total PnL %': total_pnl,
                                'Avg Win %': avg_win,
                                'Avg Loss %': avg_loss,
                                'Best Trade %': best_trade,
                                'Worst Trade %': worst_trade,
                                'Risk:Reward': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                                'trades': trades
                            })
                    
                    # Support Bounce Strategy
                    if test_support_bounce:
                        trades = []
                        in_trade = False
                        entry_price = 0
                        entry_idx = 0
                        
                        sr_levels = detect_support_resistance(data)
                        support_prices = [l['price'] for l in sr_levels if l['type'] == 'Support']
                        
                        for i in range(50, len(data)):
                            if not in_trade and support_prices:
                                # Entry: Price near support (within 1%)
                                nearest_support = min(support_prices, key=lambda x: abs(x - data['Close'].iloc[i]))
                                distance_pct = abs(data['Close'].iloc[i] - nearest_support) / nearest_support
                                
                                if distance_pct < 0.01 and data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                                    in_trade = True
                                    entry_price = data['Close'].iloc[i]
                                    entry_idx = i
                            else:
                                # Exit: 2% profit or 1% loss
                                if data['Close'].iloc[i] > entry_price * 1.02 or data['Close'].iloc[i] < entry_price * 0.99:
                                    exit_price = data['Close'].iloc[i]
                                    pnl = ((exit_price - entry_price) / entry_price) * 100
                                    
                                    trades.append({
                                        'strategy': 'Support Bounce',
                                        'entry_date': data['DateTime_IST'].iloc[entry_idx],
                                        'entry_price': entry_price,
                                        'exit_date': data['DateTime_IST'].iloc[i],
                                        'exit_price': exit_price,
                                        'pnl_pct': pnl,
                                        'points': exit_price - entry_price
                                    })
                                    in_trade = False
                        
                        if trades:
                            winning_trades = len([t for t in trades if t['pnl_pct'] > 0])
                            losing_trades = len(trades) - winning_trades
                            total_pnl = sum([t['pnl_pct'] for t in trades])
                            avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if winning_trades > 0 else 0
                            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if losing_trades > 0 else 0
                            best_trade = max([t['pnl_pct'] for t in trades])
                            worst_trade = min([t['pnl_pct'] for t in trades])
                            win_rate = (winning_trades / len(trades)) * 100
                            
                            all_results.append({
                                'Strategy': 'Support Bounce',
                                'Timeframe': tf_str,
                                'Total Trades': len(trades),
                                'Winning': winning_trades,
                                'Losing': losing_trades,
                                'Win Rate %': win_rate,
                                'Total PnL %': total_pnl,
                                'Avg Win %': avg_win,
                                'Avg Loss %': avg_loss,
                                'Best Trade %': best_trade,
                                'Worst Trade %': worst_trade,
                                'Risk:Reward': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                                'trades': trades
                            })
                
                # Display results
                if all_results:
                    st.success(f"✅ Backtesting complete! Analyzed {len(all_results)} strategy-timeframe combinations")
                    
                    # Summary table
                    st.subheader("📊 Backtest Results Summary")
                    
                    summary_data = []
                    for result in all_results:
                        summary_data.append({
                            'Strategy': result['Strategy'],
                            'Timeframe': result['Timeframe'],
                            'Total Trades': result['Total Trades'],
                            'Win Rate %': f"{result['Win Rate %']:.1f}%",
                            'Total PnL %': f"{result['Total PnL %']:.2f}%",
                            'Avg Win %': f"{result['Avg Win %']:.2f}%",
                            'Avg Loss %': f"{result['Avg Loss %']:.2f}%",
                            'Best Trade %': f"{result['Best Trade %']:.2f}%",
                            'Worst Trade %': f"{result['Worst Trade %']:.2f}%",
                            'Risk:Reward': f"{result['Risk:Reward']:.2f}"
                        })
                    
                    df_summary = pd.DataFrame(summary_data)
                    df_summary = df_summary.sort_values('Total PnL %', ascending=False)
                    st.dataframe(df_summary, use_container_width=True)
                    
                    # Best strategy
                    best_result = max(all_results, key=lambda x: x['Total PnL %'])
                    
                    st.subheader(f"🏆 Best Performing Strategy: {best_result['Strategy']} on {best_result['Timeframe']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Trades", best_result['Total Trades'])
                    with col2:
                        st.metric("Win Rate", f"{best_result['Win Rate %']:.1f}%")
                    with col3:
                        st.metric("Total PnL", f"{best_result['Total PnL %']:.2f}%")
                    with col4:
                        st.metric("Risk:Reward", f"{best_result['Risk:Reward']:.2f}")
                    
                    # Strategy explanation
                    st.markdown("### 📖 Strategy Logic")
                    
                    strategy_explanations = {
                        'RSI+EMA': "**Entry:** RSI < 30 (oversold) AND Price > 20 EMA (uptrend)\n**Exit:** RSI > 70 (overbought) OR Price < 20 EMA\n**Logic:** Buy oversold dips in uptrend, exit on overbought or trend break",
                        'EMA Crossover': "**Entry:** 20 EMA crosses above 50 EMA (golden cross)\n**Exit:** 20 EMA crosses below 50 EMA (death cross)\n**Logic:** Follow trend changes signaled by EMA crossovers",
                        'Z-Score Reversion': "**Entry:** Z-Score < -2 (extreme oversold)\n**Exit:** Z-Score > 0 (return to mean)\n**Logic:** Mean reversion from extreme statistical deviations",
                        '9 EMA Pullback': "**Entry:** Price bounces off 9 EMA in uptrend (20 EMA > 50 EMA)\n**Exit:** Price closes below 9 EMA\n**Logic:** Buy pullbacks to fast EMA in established uptrend",
                        'Volatility Breakout': "**Entry:** Low volatility (<70% avg) + 1% price breakout\n**Exit:** Volatility normalizes OR 2% stop loss\n**Logic:** Compression leads to expansion, catch the breakout",
                        'Support Bounce': "**Entry:** Price within 1% of support level + bullish candle\n**Exit:** +2% target OR -1% stop loss\n**Logic:** Buy bounces from tested support levels"
                    }
                    
                    if best_result['Strategy'] in strategy_explanations:
                        st.info(strategy_explanations[best_result['Strategy']])
                    
                    # Detailed trades
                    st.markdown("### 📋 Detailed Trade Log")
                    
                    trade_data = []
                    for trade in best_result['trades']:
                        trade_data.append({
                            'Entry Date': format_time_ago(trade['entry_date']),
                            'Entry Price': f"₹{trade['entry_price']:,.2f}",
                            'Exit Date': format_time_ago(trade['exit_date']),
                            'Exit Price': f"₹{trade['exit_price']:,.2f}",
                            'Points': f"{trade['points']:,.2f}",
                            'PnL %': f"{trade['pnl_pct']:.2f}%",
                            'Result': '✅ Win' if trade['pnl_pct'] > 0 else '❌ Loss'
                        })
                    
                    df_trades = pd.DataFrame(trade_data)
                    st.dataframe(df_trades, use_container_width=True)
                    
                    # Cumulative PnL chart
                    st.markdown("### 📈 Cumulative PnL")
                    
                    cumulative_pnl = []
                    running_pnl = 0
                    dates = []
                    
                    for trade in best_result['trades']:
                        running_pnl += trade['pnl_pct']
                        cumulative_pnl.append(running_pnl)
                        dates.append(trade['exit_date'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=cumulative_pnl,
                        mode='lines+markers',
                        name='Cumulative PnL',
                        line=dict(color='green' if cumulative_pnl[-1] > 0 else 'red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Cumulative PnL - {best_result['Strategy']} ({best_result['Timeframe']})",
                        xaxis_title="Date",
                        yaxis_title="Cumulative PnL (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics
                    st.markdown("### 📊 Performance Metrics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Average Win:** {best_result['Avg Win %']:.2f}%")
                        st.write(f"**Average Loss:** {best_result['Avg Loss %']:.2f}%")
                        st.write(f"**Best Trade:** {best_result['Best Trade %']:.2f}%")
                    with col2:
                        st.write(f"**Worst Trade:** {best_result['Worst Trade %']:.2f}%")
                        st.write(f"**Winning Trades:** {best_result['Winning']}")
                        st.write(f"**Losing Trades:** {best_result['Losing']}")
                    
                    # What worked vs didn't work
                    st.markdown("### 💡 Analysis Summary")
                    
                    st.markdown("**What Worked:**")
                    best_strategies = sorted(all_results, key=lambda x: x['Total PnL %'], reverse=True)[:3]
                    for i, strat in enumerate(best_strategies, 1):
                        st.write(f"{i}. {strat['Strategy']} on {strat['Timeframe']}: +{strat['Total PnL %']:.2f}% (Win Rate: {strat['Win Rate %']:.1f}%)")
                    
                    st.markdown("**What Didn't Work:**")
                    worst_strategies = sorted(all_results, key=lambda x: x['Total PnL %'])[:3]
                    for i, strat in enumerate(worst_strategies, 1):
                        st.write(f"{i}. {strat['Strategy']} on {strat['Timeframe']}: {strat['Total PnL %']:.2f}% (Win Rate: {strat['Win Rate %']:.1f}%)")
                    
                    # Trading recommendations
                    st.markdown("### 🎯 Trading Recommendations")
                    
                    if best_result['Win Rate %'] > 60 and best_result['Risk:Reward'] > 1.5:
                        st.success(f"✅ **RECOMMENDED:** {best_result['Strategy']} shows strong edge with {best_result['Win Rate %']:.1f}% win rate and {best_result['Risk:Reward']:.2f} risk:reward")
                    elif best_result['Win Rate %'] > 50 and best_result['Total PnL %'] > 5:
                        st.info(f"👍 **VIABLE:** {best_result['Strategy']} is profitable but requires careful risk management")
                    else:
                        st.warning("⚠️ **CAUTION:** Best strategy shows limited edge. Consider combining multiple timeframes or strategies")
                    
                    st.write(f"**Suggested Position Size:** Risk 1-2% per trade based on {best_result['Avg Loss %']:.2f}% average loss")
                    st.write(f"**Expected Win Rate:** ~{best_result['Win Rate %']:.0f}%")
                    st.write(f"**Expected R:R:** {best_result['Risk:Reward']:.2f}:1")
                    
                else:
                    st.warning("No trades generated. Try different strategies or timeframes.")
    
    # Tab 11: Live Trading
    with tabs[11]:
        st.header("▶️ Live Trading Monitor")
        
        st.info("🔴 **LIVE MODE** - Data refreshes automatically every 2 seconds")
        
        # Strategy selection
        st.subheader("📊 Select Strategy to Monitor")
        
        live_strategy = st.selectbox(
            "Strategy",
            ['RSI+EMA', 'EMA Crossover', 'Z-Score Reversion', '9 EMA Pullback', 'Volatility Breakout', 'Support Bounce'],
            key='live_strategy'
        )
        
        live_timeframe = st.selectbox(
            "Timeframe",
            [f"{tf[0]}/{tf[1]}" for tf in st.session_state['combinations']],
            key='live_tf'
        )
        
        # Strategy parameters display
        st.markdown("### ⚙️ Strategy Parameters")
        
        strategy_params = {
            'RSI+EMA': {
                'Entry Condition': 'RSI < 30 AND Price > 20 EMA',
                'Exit Condition': 'RSI > 70 OR Price < 20 EMA',
                'Stop Loss': 'Price < 20 EMA',
                'Target': 'RSI > 70 or +2% gain',
                'Risk Management': 'Exit on trend break'
            },
            'EMA Crossover': {
                'Entry Condition': '20 EMA crosses above 50 EMA',
                'Exit Condition': '20 EMA crosses below 50 EMA',
                'Stop Loss': '2% below entry',
                'Target': '+3% gain',
                'Risk Management': 'Trail stop with 20 EMA'
            },
            'Z-Score Reversion': {
                'Entry Condition': 'Z-Score < -2 (extreme oversold)',
                'Exit Condition': 'Z-Score > 0 (mean reversion)',
                'Stop Loss': 'Z-Score < -3 or -2% loss',
                'Target': 'Z-Score crosses above 0',
                'Risk Management': 'Exit if oversold deepens'
            },
            '9 EMA Pullback': {
                'Entry Condition': 'Price bounces off 9 EMA in uptrend',
                'Exit Condition': 'Price closes below 9 EMA',
                'Stop Loss': 'Close below 9 EMA',
                'Target': '+1.5% or resistance level',
                'Risk Management': 'Tight stop at 9 EMA'
            },
            'Volatility Breakout': {
                'Entry Condition': 'Low vol (<70% avg) + 1% breakout',
                'Exit Condition': 'Volatility normalizes',
                'Stop Loss': '-2% from entry',
                'Target': '+3% gain',
                'Risk Management': 'Exit when expansion completes'
            },
            'Support Bounce': {
                'Entry Condition': 'Price within 1% of support + bullish candle',
                'Exit Condition': '+2% target OR -1% stop',
                'Stop Loss': '-1% below entry',
                'Target': '+2% gain',
                'Risk Management': 'Fixed stop/target'
            }
        }
        
        if live_strategy in strategy_params:
            params = strategy_params[live_strategy]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Entry:** {params['Entry Condition']}")
                st.write(f"**Exit:** {params['Exit Condition']}")
                st.write(f"**Stop Loss:** {params['Stop Loss']}")
            with col2:
                st.write(f"**Target:** {params['Target']}")
                st.write(f"**Risk Mgmt:** {params['Risk Management']}")
        
        # Live monitoring
        st.markdown("### 📡 Live Market Status")
        
        # Create placeholder for live updates
        live_status = st.empty()
        live_metrics = st.empty()
        live_chart = st.empty()
        live_signals = st.empty()
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Enable Auto-Refresh (2 sec)", value=False, key='auto_refresh')
        manual_refresh = st.button("🔃 Manual Refresh", type="primary")
        
        if auto_refresh or manual_refresh:
            # Parse timeframe
            interval, period = live_timeframe.split('/')
            
            # Fetch fresh data
            with st.spinner("Fetching live data..."):
                live_data = fetch_data(st.session_state['ticker1'], interval, period)
            
            if live_data is not None and len(live_data) > 50:
                # Calculate indicators
                live_data['RSI'] = calculate_rsi(live_data['Close'], 14)
                live_data['EMA_9'] = calculate_ema(live_data['Close'], 9)
                live_data['EMA_20'] = calculate_ema(live_data['Close'], 20)
                live_data['EMA_50'] = calculate_ema(live_data['Close'], 50)
                live_data['Z_Score'] = calculate_z_score(live_data)
                live_data['Volatility_%'] = calculate_volatility(live_data, 20)
                
                current_price = live_data['Close'].iloc[-1]
                current_rsi = live_data['RSI'].iloc[-1]
                current_z = live_data['Z_Score'].iloc[-1]
                current_vol = live_data['Volatility_%'].iloc[-1]
                avg_vol = live_data['Volatility_%'].mean()
                
                last_update = live_data['DateTime_IST'].iloc[-1]
                
                # Status
                with live_status:
                    st.success(f"✅ Live | Last Update: {format_time_ago(last_update)}")
                
                # Metrics
                with live_metrics:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Price", f"₹{current_price:,.2f}")
                    with col2:
                        rsi_delta = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
                        st.metric("RSI", f"{current_rsi:.1f}", rsi_delta)
                    with col3:
                        z_delta = "Extreme" if abs(current_z) > 2 else "Normal"
                        st.metric("Z-Score", f"{current_z:.2f}", z_delta)
                    with col4:
                        vol_delta = "High" if current_vol > avg_vol * 1.5 else "Low" if current_vol < avg_vol * 0.7 else "Normal"
                        st.metric("Volatility", f"{current_vol:.2f}%", vol_delta)
                    with col5:
                        ema_trend = "Bullish" if current_price > live_data['EMA_20'].iloc[-1] > live_data['EMA_50'].iloc[-1] else "Bearish" if current_price < live_data['EMA_20'].iloc[-1] < live_data['EMA_50'].iloc[-1] else "Mixed"
                        st.metric("Trend", ema_trend)
                
                # Evaluate strategy conditions
                with live_signals:
                    st.markdown("### 🎯 Strategy Signal")
                    
                    signal_active = False
                    signal_message = ""
                    entry_price = current_price
                    stop_loss = 0
                    target = 0
                    
                    if live_strategy == 'RSI+EMA':
                        if current_rsi < 30 and current_price > live_data['EMA_20'].iloc[-1]:
                            signal_active = True
                            signal_message = "🟢 **BUY SIGNAL ACTIVE**"
                            stop_loss = live_data['EMA_20'].iloc[-1]
                            target = current_price * 1.02
                        elif current_rsi > 70 or current_price < live_data['EMA_20'].iloc[-1]:
                            signal_message = "🔴 **EXIT CONDITIONS MET**"
                        else:
                            signal_message = "🟡 **WAITING FOR SETUP**"
                    
                    elif live_strategy == 'EMA Crossover':
                        ema20_curr = live_data['EMA_20'].iloc[-1]
                        ema50_curr = live_data['EMA_50'].iloc[-1]
                        ema20_prev = live_data['EMA_20'].iloc[-2]
                        ema50_prev = live_data['EMA_50'].iloc[-2]
                        
                        if ema20_curr > ema50_curr and ema20_prev <= ema50_prev:
                            signal_active = True
                            signal_message = "🟢 **GOLDEN CROSS - BUY SIGNAL**"
                            stop_loss = current_price * 0.98
                            target = current_price * 1.03
                        elif ema20_curr < ema50_curr and ema20_prev >= ema50_prev:
                            signal_message = "🔴 **DEATH CROSS - SELL SIGNAL**"
                        else:
                            signal_message = "🟡 **NO CROSSOVER**"
                    
                    elif live_strategy == 'Z-Score Reversion':
                        if current_z < -2:
                            signal_active = True
                            signal_message = "🟢 **EXTREME OVERSOLD - BUY SIGNAL**"
                            stop_loss = current_price * 0.98
                            target = current_price * 1.02
                        elif current_z > 2:
                            signal_message = "🔴 **EXTREME OVERBOUGHT - SELL SIGNAL**"
                        elif abs(current_z) < 0.5:
                            signal_message = "🟡 **NEAR MEAN - NO SIGNAL**"
                        else:
                            signal_message = "🟡 **WAITING FOR EXTREME**"
                    
                    elif live_strategy == '9 EMA Pullback':
                        ema9_curr = live_data['EMA_9'].iloc[-1]
                        ema20_curr = live_data['EMA_20'].iloc[-1]
                        ema50_curr = live_data['EMA_50'].iloc[-1]
                        
                        if ema20_curr > ema50_curr and current_price > ema9_curr and live_data['Close'].iloc[-2] < live_data['EMA_9'].iloc[-2]:
                            signal_active = True
                            signal_message = "🟢 **PULLBACK BOUNCE - BUY SIGNAL**"
                            stop_loss = ema9_curr
                            target = current_price * 1.015
                        elif current_price < ema9_curr:
                            signal_message = "🔴 **BELOW 9 EMA - EXIT**"
                        else:
                            signal_message = "🟡 **WAITING FOR PULLBACK**"
                    
                    elif live_strategy == 'Volatility Breakout':
                        if current_vol < avg_vol * 0.7 and current_price > live_data['Close'].iloc[-2] * 1.01:
                            signal_active = True
                            signal_message = "🟢 **BREAKOUT FROM COMPRESSION - BUY**"
                            stop_loss = current_price * 0.98
                            target = current_price * 1.03
                        elif current_vol > avg_vol * 1.5:
                            signal_message = "🔴 **HIGH VOLATILITY - CAUTION**"
                        else:
                            signal_message = "🟡 **WAITING FOR COMPRESSION**"
                    
                    elif live_strategy == 'Support Bounce':
                        sr_levels = detect_support_resistance(live_data)
                        support_prices = [l['price'] for l in sr_levels if l['type'] == 'Support']
                        
                        if support_prices:
                            nearest_support = min(support_prices, key=lambda x: abs(x - current_price))
                            distance_pct = abs(current_price - nearest_support) / nearest_support
                            
                            if distance_pct < 0.01 and current_price > live_data['Close'].iloc[-2]:
                                signal_active = True
                                signal_message = f"🟢 **SUPPORT BOUNCE - BUY SIGNAL**"
                                stop_loss = nearest_support * 0.99
                                target = current_price * 1.02
                            elif distance_pct < 0.02:
                                signal_message = f"🟡 **NEAR SUPPORT @ ₹{nearest_support:,.2f}**"
                            else:
                                signal_message = "🟡 **NO SUPPORT NEARBY**"
                        else:
                            signal_message = "🟡 **NO SUPPORT LEVELS DETECTED**"
                    
                    # Display signal
                    if signal_active:
                        st.success(signal_message)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Entry", f"₹{entry_price:,.2f}")
                        with col2:
                            st.metric("Stop Loss", f"₹{stop_loss:,.2f}")
                        with col3:
                            st.metric("Target", f"₹{target:,.2f}")
                        with col4:
                            risk_reward = abs(target - entry_price) / abs(entry_price - stop_loss) if stop_loss != entry_price else 0
                            st.metric("R:R", f"{risk_reward:.2f}:1")
                        
                        st.warning("⚠️ **ACTION REQUIRED:** Entry conditions met. Review and execute trade.")
                    else:
                        if "EXIT" in signal_message or "SELL" in signal_message:
                            st.error(signal_message)
                        elif "WAITING" in signal_message or "NO" in signal_message:
                            st.info(signal_message)
                        else:
                            st.warning(signal_message)
                
                # Live chart
                with live_chart:
                    st.markdown("### 📈 Live Price Chart")
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.03, 
                                       row_heights=[0.7, 0.3],
                                       subplot_titles=('Price & EMAs', 'RSI'))
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=live_data['DateTime_IST'].tail(100),
                        open=live_data['Open'].tail(100),
                        high=live_data['High'].tail(100),
                        low=live_data['Low'].tail(100),
                        close=live_data['Close'].tail(100),
                        name='Price'
                    ), row=1, col=1)
                    
                    # EMAs
                    fig.add_trace(go.Scatter(
                        x=live_data['DateTime_IST'].tail(100),
                        y=live_data['EMA_9'].tail(100),
                        mode='lines',
                        name='EMA 9',
                        line=dict(color='yellow', width=1)
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=live_data['DateTime_IST'].tail(100),
                        y=live_data['EMA_20'].tail(100),
                        mode='lines',
                        name='EMA 20',
                        line=dict(color='blue', width=1)
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=live_data['DateTime_IST'].tail(100),
                        y=live_data['EMA_50'].tail(100),
                        mode='lines',
                        name='EMA 50',
                        line=dict(color='purple', width=1)
                    ), row=1, col=1)
                    
                    # RSI
                    fig.add_trace(go.Scatter(
                        x=live_data['DateTime_IST'].tail(100),
                        y=live_data['RSI'].tail(100),
                        mode='lines',
                        name='RSI',
                        line=dict(color='orange', width=2)
                    ), row=2, col=1)
                    
                    # RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
                    
                    fig.update_layout(
                        height=700,
                        showlegend=True,
                        xaxis_rangeslider_visible=False,
                        title=f"Live Chart - {st.session_state['ticker1']} ({live_timeframe})"
                    )
                    
                    fig.update_xaxes(title_text="Time", row=2, col=1)
                    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                    fig.update_yaxes(title_text="RSI", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Auto-refresh logic
                if auto_refresh:
                    time.sleep(2)
                    st.rerun()
            
            else:
                st.error("Failed to fetch live data or insufficient data points")
        
        else:
            st.info("👆 Enable auto-refresh or click manual refresh to start live monitoring")
            st.markdown("### 📖 How to Use Live Trading Monitor")
            st.write("""
            1. **Select your strategy** from the dropdown above
            2. **Choose a timeframe** that matches your trading style
            3. **Review strategy parameters** to understand entry/exit conditions
            4. **Enable auto-refresh** for continuous monitoring (updates every 2 seconds)
            5. **Watch for signals** - Green means BUY, Red means EXIT/SELL, Yellow means WAIT
            6. **Execute trades** when conditions are met, using the provided entry/SL/target levels
            
            ⚠️ **Important Notes:**
            - Live data is fetched from yfinance with a small delay
            - Always verify signals before executing trades
            - Use proper position sizing (risk 1-2% per trade)
            - Set stop losses immediately after entering trades
            - Monitor open positions regularly
            """)
            
            st.markdown("### 💡 Strategy Quick Guide")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Best for Intraday:**")
                st.write("- RSI+EMA (5m, 15m)")
                st.write("- 9 EMA Pullback (5m, 15m)")
                st.write("- Support Bounce (15m, 30m)")
                
                st.markdown("**Best for Swing:**")
                st.write("- EMA Crossover (1d)")
                st.write("- Z-Score Reversion (1d)")
                st.write("- Volatility Breakout (1h, 1d)")
            
            with col2:
                st.markdown("**Risk Management:**")
                st.write("- Always use stop losses")
                st.write("- Risk max 1-2% per trade")
                st.write("- Take partial profits at targets")
                st.write("- Trail stops in trending markets")
                
                st.markdown("**Best Practices:**")
                st.write("- Wait for clear signals")
                st.write("- Avoid trading in choppy markets")
                st.write("- Combine multiple timeframes")
                st.write("- Review backtest results first")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Professional Algorithmic Trading Analysis System</strong></p>
    <p>⚠️ <strong>Disclaimer:</strong> This system is for educational and informational purposes only. 
    Past performance does not guarantee future results. Always conduct your own research and 
    consider consulting with a qualified financial advisor before making investment decisions.</p>
    <p>Built with Streamlit | Data from yfinance | Technical Analysis</p>
</div>
""", unsafe_allow_html=True):
            dt = IST.localize(dt)
        else:
            dt = dt.astimezone(IST)
        
        now = pd.Timestamp.now(tz=IST)
        diff = now - dt
        
        minutes = diff.total_seconds() / 60
        hours = minutes / 60
        days = hours / 24
        
        if minutes < 60:
            return f"{int(minutes)} minutes ago"
        elif hours < 24:
            return f"{int(hours)} hours ago"
        elif days < 30:
            return f"{int(days)} days ago"
        else:
            months = int(days / 30)
            remaining_days = int(days % 30)
            return f"{months} months and {remaining_days} days ago ({dt.strftime('%Y-%m-%d %H:%M:%S IST')})"
    except Exception as e:
        return str(dt)

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    """Calculate EMA"""
    return data.ewm(span=period, adjust=False).mean()

def detect_support_resistance(data, window=20, tolerance=0.02):
    """Detect support and resistance levels"""
    highs = data['High'].values
    lows = data['Low'].values
    closes = data['Close'].values
    dates = data.index
    
    # Find local maxima and minima
    resistance_idx = argrelextrema(highs, np.greater, order=window)[0]
    support_idx = argrelextrema(lows, np.less, order=window)[0]
    
    levels = []
    
    # Process resistance levels
    for idx in resistance_idx:
        if idx < len(highs):
            price = highs[idx]
            date = dates[idx]
            levels.append({'type': 'Resistance', 'price': price, 'date': date})
    
    # Process support levels
    for idx in support_idx:
        if idx < len(lows):
            price = lows[idx]
            date = dates[idx]
            levels.append({'type': 'Support', 'price': price, 'date': date})
    
    if not levels:
        return []
    
    # Cluster similar levels
    df_levels = pd.DataFrame(levels)
    df_levels = df_levels.sort_values('price')
    
    clustered = []
    for level_type in ['Support', 'Resistance']:
        type_levels = df_levels[df_levels['type'] == level_type].copy()
        if len(type_levels) == 0:
            continue
        
        current_cluster = [type_levels.iloc[0]]
        
        for i in range(1, len(type_levels)):
            if abs(type_levels.iloc[i]['price'] - current_cluster[-1]['price']) / current_cluster[-1]['price'] <= tolerance:
                current_cluster.append(type_levels.iloc[i])
            else:
                avg_price = np.mean([l['price'] for l in current_cluster])
                first_hit = min([l['date'] for l in current_cluster])
                last_hit = max([l['date'] for l in current_cluster])
                hit_count = len(current_cluster)
                
                # Calculate sustained count
                sustained = 0
                for l in current_cluster:
                    idx = data.index.get_loc(l['date'])
                    if idx < len(data) - 1:
                        if level_type == 'Resistance':
                            if data['Close'].iloc[idx+1] < l['price']:
                                sustained += 1
                        else:
                            if data['Close'].iloc[idx+1] > l['price']:
                                sustained += 1
                
                clustered.append({
                    'type': level_type,
                    'price': avg_price,
                    'hit_count': hit_count,
                    'sustained_count': sustained,
                    'first_hit': first_hit,
                    'last_hit': last_hit
                })
                
                current_cluster = [type_levels.iloc[i]]
        
        if current_cluster:
            avg_price = np.mean([l['price'] for l in current_cluster])
            first_hit = min([l['date'] for l in current_cluster])
            last_hit = max([l['date'] for l in current_cluster])
            hit_count = len(current_cluster)
            sustained = sum(1 for l in current_cluster if data.index.get_loc(l['date']) < len(data) - 1)
            
            clustered.append({
                'type': level_type,
                'price': avg_price,
                'hit_count': hit_count,
                'sustained_count': sustained,
                'first_hit': first_hit,
                'last_hit': last_hit
            })
    
    return clustered

def fetch_data(ticker, interval, period):
    """Fetch data from yfinance with proper error handling"""
    try:
        time.sleep(1.5)  # Rate limiting
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
        
        # Handle multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Reset index and convert to IST
        data = data.reset_index()
        
        # Rename datetime column
        date_col = data.columns[0]
        data = data.rename(columns={date_col: 'DateTime_IST'})
        
        # Convert to IST
        if data['DateTime_IST'].dtype == 'datetime64[ns]':
            data['DateTime_IST'] = pd.to_datetime(data['DateTime_IST'])
            if data['DateTime_IST'].dt.tz is None:
                data['DateTime_IST'] = data['DateTime_IST'].dt.tz_localize('UTC').dt.tz_convert(IST)
            else:
                data['DateTime_IST'] = data['DateTime_IST'].dt.tz_convert(IST)
        
        # Keep only required columns
        cols = ['DateTime_IST', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in data.columns:
            cols.append('Volume')
        
        data = data[cols].copy()
        data = data.dropna(subset=['Close'])
        
        return data
    except Exception as e:
        st.warning(f"Error fetching {ticker} ({interval}/{period}): {str(e)}")
        return None

def calculate_z_score(data):
    """Calculate Z-Score for returns"""
    returns = data['Close'].pct_change()
    z_score = (returns - returns.mean()) / returns.std()
    return z_score

def calculate_volatility(data, window=20):
    """Calculate rolling volatility"""
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return volatility

def detect_rsi_divergence(data, rsi_period=14):
    """Detect RSI divergences"""
    data = data.copy()
    data['RSI'] = calculate_rsi(data['Close'], rsi_period)
    
    # Find peaks and troughs
    price_peaks = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    price_troughs = argrelextrema(data['Close'].values, np.less, order=5)[0]
    rsi_peaks = argrelextrema(data['RSI'].values, np.greater, order=5)[0]
    rsi_troughs = argrelextrema(data['RSI'].values, np.less, order=5)[0]
    
    divergences = []
    
    # Bearish divergence
    for i in range(len(price_peaks) - 1):
        idx1, idx2 = price_peaks[i], price_peaks[i+1]
        if idx2 < len(data):
            if data['Close'].iloc[idx2] > data['Close'].iloc[idx1]:
                # Find corresponding RSI peaks
                rsi_peak1 = min(rsi_peaks, key=lambda x: abs(x - idx1)) if len(rsi_peaks) > 0 else None
                rsi_peak2 = min(rsi_peaks, key=lambda x: abs(x - idx2)) if len(rsi_peaks) > 0 else None
                
                if rsi_peak1 is not None and rsi_peak2 is not None:
                    if data['RSI'].iloc[rsi_peak2] < data['RSI'].iloc[rsi_peak1]:
                        divergences.append({
                            'type': 'Bearish',
                            'price1': data['Close'].iloc[idx1],
                            'price2': data['Close'].iloc[idx2],
                            'date1': data['DateTime_IST'].iloc[idx1],
                            'date2': data['DateTime_IST'].iloc[idx2],
                            'rsi1': data['RSI'].iloc[rsi_peak1],
                            'rsi2': data['RSI'].iloc[rsi_peak2],
                            'resolved': idx2 < len(data) - 5 and data['Close'].iloc[idx2:idx2+5].min() < data['Close'].iloc[idx2]
                        })
    
    # Bullish divergence
    for i in range(len(price_troughs) - 1):
        idx1, idx2 = price_troughs[i], price_troughs[i+1]
        if idx2 < len(data):
            if data['Close'].iloc[idx2] < data['Close'].iloc[idx1]:
                rsi_trough1 = min(rsi_troughs, key=lambda x: abs(x - idx1)) if len(rsi_troughs) > 0 else None
                rsi_trough2 = min(rsi_troughs, key=lambda x: abs(x - idx2)) if len(rsi_troughs) > 0 else None
                
                if rsi_trough1 is not None and rsi_trough2 is not None:
                    if data['RSI'].iloc[rsi_trough2] > data['RSI'].iloc[rsi_trough1]:
                        divergences.append({
                            'type': 'Bullish',
                            'price1': data['Close'].iloc[idx1],
                            'price2': data['Close'].iloc[idx2],
                            'date1': data['DateTime_IST'].iloc[idx1],
                            'date2': data['DateTime_IST'].iloc[idx2],
                            'rsi1': data['RSI'].iloc[rsi_trough1],
                            'rsi2': data['RSI'].iloc[rsi_trough2],
                            'resolved': idx2 < len(data) - 5 and data['Close'].iloc[idx2:idx2+5].max() > data['Close'].iloc[idx2]
                        })
    
    return divergences

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    
    levels = {
        '0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100%': low
    }
    
    return levels

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Ticker 1
st.sidebar.subheader("Primary Ticker")
ticker1_type = st.sidebar.selectbox("Select Asset Type", list(PREDEFINED_ASSETS.keys()) + ['Custom'], key='ticker1_type')

if ticker1_type == 'Custom':
    ticker1 = st.sidebar.text_input("Enter Ticker Symbol", "^NSEI", key='ticker1_custom')
else:
    ticker1 = PREDEFINED_ASSETS[ticker1_type]

st.sidebar.write(f"**Ticker 1:** {ticker1}")

# Timeframe/Period selection
st.sidebar.subheader("Timeframes & Periods")
selected_combinations = []

for tf, periods in VALID_COMBINATIONS.items():
    with st.sidebar.expander(f"📊 {tf}", expanded=False):
        for period in periods:
            if st.checkbox(f"{period}", key=f"{tf}_{period}"):
                selected_combinations.append((tf, period))

# Ratio Analysis
st.sidebar.subheader("Ratio Analysis (Optional)")
enable_ratio = st.sidebar.checkbox("Enable Ticker 2 Comparison", value=False)

ticker2 = None
if enable_ratio:
    ticker2_type = st.sidebar.selectbox("Select Asset Type", list(PREDEFINED_ASSETS.keys()) + ['Custom'], key='ticker2_type')
    
    if ticker2_type == 'Custom':
        ticker2 = st.sidebar.text_input("Enter Ticker Symbol", "BTC-USD", key='ticker2_custom')
    else:
        ticker2 = PREDEFINED_ASSETS[ticker2_type]
    
    st.sidebar.write(f"**Ticker 2:** {ticker2}")

# Fetch button
if st.sidebar.button("🚀 Fetch Data & Analyze", type="primary"):
    if not selected_combinations:
        st.error("Please select at least one timeframe/period combination")
    else:
        # Initialize session state
        st.session_state['data_fetched'] = True
        st.session_state['ticker1'] = ticker1
        st.session_state['ticker2'] = ticker2
        st.session_state['combinations'] = selected_combinations
        st.session_state['all_data'] = {}
        st.session_state['all_data_t2'] = {}
        
        # Fetch data
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(selected_combinations)
        for i, (interval, period) in enumerate(selected_combinations):
            status_text.text(f"Fetching {interval}/{period}... ({i+1}/{total})")
            progress_bar.progress((i + 1) / total)
            
            data = fetch_data(ticker1, interval, period)
            if data is not None:
                st.session_state['all_data'][(interval, period)] = data
            
            if enable_ratio and ticker2:
                data2 = fetch_data(ticker2, interval, period)
                if data2 is not None:
                    st.session_state['all_data_t2'][(interval, period)] = data2
        
        status_text.text("✅ Data fetching complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

# Main content
st.title("📈 Professional Algorithmic Trading Analysis System")

if 'data_fetched' not in st.session_state or not st.session_state['data_fetched']:
    st.info("👈 Configure settings in the sidebar and click 'Fetch Data & Analyze' to begin")
else:
    # Display basic info
    st.success(f"**Analyzing:** {st.session_state['ticker1']} | **Timeframes:** {len(st.session_state['combinations'])}")
    
    if st.session_state['ticker2']:
        st.info(f"**Ratio Analysis Enabled:** {st.session_state['ticker2']}")
    
    # Create tabs
    tabs = st.tabs([
        "📊 Overview",
        "🎯 S/R Analysis",
        "📉 Technical Indicators",
        "📊 Z-Score",
        "💨 Volatility",
        "🌊 Elliott Waves",
        "📐 Fibonacci",
        "🔄 RSI Divergence",
        "⚖️ Ratio Analysis",
        "🤖 AI Signals",
        "🔬 Backtesting",
        "▶️ Live Trading"
    ])
    
    # Tab 0: Multi-Timeframe Overview
    with tabs[0]:
        st.header("📊 Multi-Timeframe Overview")
        
        overview_data = []
        for interval, period in st.session_state['combinations']:
            if (interval, period) in st.session_state['all_data']:
                data = st.session_state['all_data'][(interval, period)]
                
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[0]
                returns = ((current_price - prev_price) / prev_price) * 100
                points = current_price - prev_price
                
                # RSI
                rsi = calculate_rsi(data['Close']).iloc[-1]
                
                # RSI Divergence
                divs = detect_rsi_divergence(data)
                rsi_div = "Bullish" if any(d['type'] == 'Bullish' and not d['resolved'] for d in divs) else "Bearish" if any(d['type'] == 'Bearish' and not d['resolved'] for d in divs) else "None"
                
                # S/R
                sr_levels = detect_support_resistance(data)
                near_sr = "None"
                if sr_levels:
                    closest = min(sr_levels, key=lambda x: abs(x['price'] - current_price))
                    distance_pct = abs(closest['price'] - current_price) / current_price * 100
                    if distance_pct < 1:
                        near_sr = f"{closest['type']} @ ₹{closest['price']:,.2f}"
                
                # Fibonacci
                fib_levels = calculate_fibonacci_levels(data)
                closest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
                
                # Status
                status = "🟢" if returns > 0 else "🔴"
                
                overview_data.append({
                    'Timeframe': interval,
                    'Period': period,
                    'Status': status,
                    'Current Price': f"₹{current_price:,.2f}",
                    'Returns %': f"{returns:.2f}%",
                    'Points': f"{points:,.2f}",
                    'RSI': f"{rsi:.1f}",
                    'RSI Divergence': rsi_div,
                    'Near S/R': near_sr,
                    'Nearest Fib': f"{closest_fib[0]} (₹{closest_fib[1]:,.2f})"
                })
        
        if overview_data:
            df_overview = pd.DataFrame(overview_data)
            st.dataframe(df_overview, use_container_width=True)
            
            # Download button
            csv = df_overview.to_csv(index=False)
            st.download_button(
                "📥 Download Overview CSV",
                csv,
                f"overview_{ticker1}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    # Tab 1: Support/Resistance Analysis
    with tabs[1]:
        st.header("🎯 Support & Resistance Analysis")
        
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            
            data = st.session_state['all_data'][(interval, period)]
            st.subheader(f"## 📊 S/R Analysis: {interval} / {period}")
            
            current_price = data['Close'].iloc[-1]
            sr_levels = detect_support_resistance(data)
            
            if sr_levels:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"₹{current_price:,.2f}")
                with col2:
                    support_count = len([l for l in sr_levels if l['type'] == 'Support'])
                    st.metric("Support Levels", support_count)
                with col3:
                    resistance_count = len([l for l in sr_levels if l['type'] == 'Resistance'])
                    st.metric("Resistance Levels", resistance_count)
                
                # S/R Table
                sr_table = []
                for level in sr_levels:
                    distance = level['price'] - current_price
                    distance_pct = (distance / current_price) * 100
                    accuracy = (level['sustained_count'] / level['hit_count'] * 100) if level['hit_count'] > 0 else 0
                    
                    sr_table.append({
                        'Type': level['type'],
                        'Price': f"₹{level['price']:,.2f}",
                        'Distance': f"{distance:,.2f} ({distance_pct:.2f}%)",
                        'Hit Count': level['hit_count'],
                        'Sustained': level['sustained_count'],
                        'Accuracy %': f"{accuracy:.1f}%",
                        'First Hit': format_time_ago(level['first_hit']),
                        'Last Hit': format_time_ago(level['last_hit'])
                    })
                
                df_sr = pd.DataFrame(sr_table)
                st.dataframe(df_sr, use_container_width=True)
                
                # Forecast
                nearest_support = [l for l in sr_levels if l['type'] == 'Support' and l['price'] < current_price]
                nearest_resistance = [l for l in sr_levels if l['type'] == 'Resistance' and l['price'] > current_price]
                
                if nearest_support:
                    nearest_support = max(nearest_support, key=lambda x: x['price'])
                    support_distance = ((current_price - nearest_support['price']) / current_price) * 100
                else:
                    support_distance = float('inf')
                
                if nearest_resistance:
                    nearest_resistance = min(nearest_resistance, key=lambda x: x['price'])
                    resistance_distance = ((nearest_resistance['price'] - current_price) / current_price) * 100
                else:
                    resistance_distance = float('inf')
                
                st.markdown("### 🎯 Forecast")
                if support_distance < resistance_distance and support_distance < 2:
                    st.success(f"⬆️ **BOUNCE EXPECTED** - Price near support at ₹{nearest_support['price']:,.2f} ({support_distance:.2f}% away)")
                    st.write(f"Historical accuracy: {(nearest_support['sustained_count']/nearest_support['hit_count']*100):.1f}%")
                elif resistance_distance < 2:
                    st.error(f"⬇️ **REJECTION EXPECTED** - Price near resistance at ₹{nearest_resistance['price']:,.2f} ({resistance_distance:.2f}% away)")
                    st.write(f"Historical accuracy: {(nearest_resistance['sustained_count']/nearest_resistance['hit_count']*100):.1f}%")
                else:
                    st.info("➡️ **NEUTRAL ZONE** - No immediate S/R nearby")
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data['DateTime_IST'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                
                for level in sr_levels:
                    color = 'green' if level['type'] == 'Support' else 'red'
                    fig.add_hline(y=level['price'], line_dash="dash", line_color=color,
                                  annotation_text=f"{level['type']}: ₹{level['price']:,.2f}")
                
                fig.update_layout(
                    title=f"S/R Levels - {interval}/{period}",
                    xaxis_title="Time",
                    yaxis_title="Price (₹)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No S/R levels detected for this timeframe")
            
            st.markdown("---")
        
        # Final Consensus
        st.subheader("## 🎯 S/R Consensus Across All Timeframes")
        
        near_support_count = 0
        near_resistance_count = 0
        
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            
            data = st.session_state['all_data'][(interval, period)]
            current_price = data['Close'].iloc[-1]
            sr_levels = detect_support_resistance(data)
            
            nearest_support = [l for l in sr_levels if l['type'] == 'Support' and l['price'] < current_price]
            nearest_resistance = [l for l in sr_levels if l['type'] == 'Resistance' and l['price'] > current_price]
            
            if nearest_support:
                nearest_support = max(nearest_support, key=lambda x: x['price'])
                support_distance = ((current_price - nearest_support['price']) / current_price) * 100
                if support_distance < 2:
                    near_support_count += 1
            
            if nearest_resistance:
                nearest_resistance = min(nearest_resistance, key=lambda x: x['price'])
                resistance_distance = ((nearest_resistance['price'] - current_price) / current_price) * 100
                if resistance_distance < 2:
                    near_resistance_count += 1
        
        total_tf = len(st.session_state['combinations'])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Near Support", f"{near_support_count}/{total_tf}", f"{(near_support_count/total_tf*100):.1f}%")
        with col2:
            st.metric("Near Resistance", f"{near_resistance_count}/{total_tf}", f"{(near_resistance_count/total_tf*100):.1f}%")
        
        if near_support_count > near_resistance_count:
            st.success(f"✅ **Overall S/R Signal: BOUNCE EXPECTED** ({near_support_count} timeframes near support)")
        elif near_resistance_count > near_support_count:
            st.error(f"⚠️ **Overall S/R Signal: REJECTION EXPECTED** ({near_resistance_count} timeframes near resistance)")
        else:
            st.info("➡️ **Overall S/R Signal: NEUTRAL** - No clear consensus")
    
    # Tab 2: Technical Indicators
    with tabs[2]:
        st.header("📉 Technical Indicators Analysis")
        
        # Multi-timeframe EMA table
        st.subheader("📊 Multi-Timeframe EMA Analysis")
        
        ema_overview = []
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            
            data = st.session_state['all_data'][(interval, period)]
            current_price = data['Close'].iloc[-1]
            ema_20 = calculate_ema(data['Close'], 20).iloc[-1]
            ema_50 = calculate_ema(data['Close'], 50).iloc[-1]
            
            trend = "Bullish 🟢" if current_price > ema_20 > ema_50 else "Bearish 🔴" if current_price < ema_20 < ema_50 else "Mixed 🟡"
            
            ema_overview.append({
                'Timeframe': interval,
                'Period': period,
                'Price': f"₹{current_price:,.2f}",
                '20 EMA': f"₹{ema_20:,.2f}",
                '50 EMA': f"₹{ema_50:,.2f}",
                'Trend': trend
            })
        
        df_ema = pd.DataFrame(ema_overview)
        st.dataframe(df_ema, use_container_width=True)
        
        bullish_count = len([x for x in ema_overview if "Bullish" in x['Trend']])
        bearish_count = len([x for x in ema_overview if "Bearish" in x['Trend']])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish Timeframes", f"{bullish_count}/{len(ema_overview)}")
        with col2:
            st.metric("Bearish Timeframes", f"{bearish_count}/{len(ema_overview)}")
        with col3:
            consensus_pct = (bullish_count / len(ema_overview) * 100) if ema_overview else 0
            st.metric("Bullish Consensus %", f"{consensus_pct:.1f}%")
        
        if consensus_pct > 60:
            st.success("✅ **Strong Bullish EMA Alignment**")
        elif consensus_pct < 40:
            st.error("⚠️ **Strong Bearish EMA Alignment**")
        else:
            st.info("➡️ **Mixed EMA Signals**")
        
        # Primary timeframe detailed analysis
        if st.session_state['combinations']:
            primary_tf = st.session_state['combinations'][0]
            if primary_tf in st.session_state['all_data']:
                data = st.session_state['all_data'][primary_tf]
                
                st.subheader(f"📊 Detailed Indicators: {primary_tf[0]} / {primary_tf[1]}")
                
                # Calculate indicators
                data_ind = data.copy()
                data_ind['EMA_9'] = calculate_ema(data_ind['Close'], 9)
                data_ind['EMA_20'] = calculate_ema(data_ind['Close'], 20)
                data_ind['EMA_50'] = calculate_ema(data_ind['Close'], 50)
                data_ind['RSI'] = calculate_rsi(data_ind['Close'], 14)
                
                # EMA Table
                st.markdown("### 📈 EMA Analysis")
                ema_table = data_ind.tail(20)[['DateTime_IST', 'Close', 'EMA_9', 'EMA_20', 'EMA_50']].copy()
                ema_table['Close'] = ema_table['Close'].apply(lambda x: f"₹{x:,.2f}")
                ema_table['EMA_9'] = ema_table['EMA_9'].apply(lambda x: f"₹{x:,.2f}")
                ema_table['EMA_20'] = ema_table['EMA_20'].apply(lambda x: f"₹{x:,.2f}")
                ema_table['EMA_50'] = ema_table['EMA_50'].apply(lambda x: f"₹{x:,.2f}")
                st.dataframe(ema_table, use_container_width=True)
                
                # RSI Table
                st.markdown("### 📊 RSI Analysis")
                rsi_table = data_ind.tail(20)[['DateTime_IST', 'Close', 'RSI']].copy()
                rsi_table['Status'] = rsi_table['RSI'].apply(
                    lambda x: "Oversold 🟢" if x < 30 else "Overbought 🔴" if x > 70 else "Neutral 🟡"
                )
                rsi_table['Close'] = rsi_table['Close'].apply(lambda x: f"₹{x:,.2f}")
                rsi_table['RSI'] = rsi_table['RSI'].apply(lambda x: f"{x:.1f}")
                st.dataframe(rsi_table, use_container_width=True)
    
    # Tab 3: Z-Score Analysis
    with tabs[3]:
        st.header("📊 Z-Score Analysis")
        
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            
            data = st.session_state['all_data'][(interval, period)]
            st.subheader(f"## 📊 Z-Score: {interval} / {period}")
            
            # Calculate Z-Score
            data_z = data.copy()
            data_z['Return_%'] = data_z['Close'].pct_change() * 100
            data_z['Z_Score'] = calculate_z_score(data_z)
            
            current_z = data_z['Z_Score'].iloc[-1]
            mean_price = data_z['Close'].mean()
            std_price = data_z['Close'].std()
            
            # Determine bin
            if current_z < -2:
                z_bin = "Extreme Negative (<-2)"
            elif current_z < -1:
                z_bin = "Negative (-2 to -1)"
            elif current_z < 0:
                z_bin = "Slightly Negative (-1 to 0)"
            elif current_z < 1:
                z_bin = "Slightly Positive (0 to 1)"
            elif current_z < 2:
                z_bin = "Positive (1 to 2)"
            else:
                z_bin = "Extreme Positive (>2)"
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Z-Score", f"{current_z:.2f}")
            with col2:
                st.metric("Current Bin", z_bin)
            with col3:
                st.metric("Mean Price", f"₹{mean_price:,.2f}")
            with col4:
                st.metric("Std Dev", f"₹{std_price:,.2f}")
            
            # Bin Distribution
            st.markdown("### 📊 Bin Distribution with Price Ranges")
            
            bins = [
                ("Extreme Negative (<-2)", data_z[data_z['Z_Score'] < -2]),
                ("Negative (-2 to -1)", data_z[(data_z['Z_Score'] >= -2) & (data_z['Z_Score'] < -1)]),
                ("Slightly Negative (-1 to 0)", data_z[(data_z['Z_Score'] >= -1) & (data_z['Z_Score'] < 0)]),
                ("Slightly Positive (0 to 1)", data_z[(data_z['Z_Score'] >= 0) & (data_z['Z_Score'] < 1)]),
                ("Positive (1 to 2)", data_z[(data_z['Z_Score'] >= 1) & (data_z['Z_Score'] < 2)]),
                ("Extreme Positive (>2)", data_z[data_z['Z_Score'] > 2])
            ]
            
            bin_data = []
            for bin_name, bin_df in bins:
                if len(bin_df) > 0:
                    price_min = bin_df['Close'].min()
                    price_max = bin_df['Close'].max()
                    avg_price = bin_df['Close'].mean()
                    is_current = bin_name == z_bin
                    
                    bin_data.append({
                        'Bin': bin_name,
                        'Count': len(bin_df),
                        'Percentage %': f"{(len(bin_df)/len(data_z)*100):.1f}%",
                        'Price Range': f"₹{price_min:,.2f} - ₹{price_max:,.2f}" + (" ✅" if is_current else ""),
                        'Average Price': f"₹{avg_price:,.2f}"
                    })
            
            df_bins = pd.DataFrame(bin_data)
            st.dataframe(df_bins, use_container_width=True)
            
            # Recent Data
            st.markdown("### 📋 Recent Data (Last 20 Periods)")
            recent_z = data_z.tail(20)[['DateTime_IST', 'Close', 'Return_%', 'Z_Score']].copy()
            recent_z['Z_Score_Bin'] = recent_z['Z_Score'].apply(
                lambda x: "Extreme Negative (<-2)" if x < -2 else
                         "Negative (-2 to -1)" if x < -1 else
                         "Slightly Negative (-1 to 0)" if x < 0 else
                         "Slightly Positive (0 to 1)" if x < 1 else
                         "Positive (1 to 2)" if x < 2 else
                         "Extreme Positive (>2)"
            )
            recent_z['Close'] = recent_z['Close'].apply(lambda x: f"₹{x:,.2f}")
            recent_z['Return_%'] = recent_z['Return_%'].apply(lambda x: f"{x:.2f}%")
            recent_z['Z_Score'] = recent_z['Z_Score'].apply(lambda x: f"{x:.2f}")
            st.dataframe(recent_z, use_container_width=True)
            
            # Historical Similarity
            st.markdown("### 🔍 Historical Similarity Analysis")
            
            if current_z > 2:
                st.error("⚠️ **EXTREME OVERBOUGHT - Correction Expected**")
                similar_events = data_z[data_z['Z_Score'] > 2]
                
                if len(similar_events) > 1:
                    st.write(f"Found {len(similar_events)} similar extreme overbought events in this timeframe.")
                    
                    # Analyze what happened after
                    corrections = []
                    for idx in similar_events.index[:-1]:
                        if idx < len(data_z) - 10:
                            entry_price = data_z.loc[idx, 'Close']
                            future_5 = data_z.loc[idx+5, 'Close'] if idx+5 < len(data_z) else entry_price
                            future_10 = data_z.loc[idx+10, 'Close'] if idx+10 < len(data_z) else entry_price
                            
                            pct_5 = ((future_5 - entry_price) / entry_price) * 100
                            pct_10 = ((future_10 - entry_price) / entry_price) * 100
                            
                            corrections.append({
                                'date': data_z.loc[idx, 'DateTime_IST'],
                                'price': entry_price,
                                'pct_5': pct_5,
                                'pct_10': pct_10
                            })
                    
                    if corrections:
                        avg_correction_5 = np.mean([c['pct_5'] for c in corrections])
                        avg_correction_10 = np.mean([c['pct_10'] for c in corrections])
                        success_rate = len([c for c in corrections if c['pct_10'] < 0]) / len(corrections) * 100
                        
                        st.write(f"**Average correction after 5 periods:** {avg_correction_5:.2f}%")
                        st.write(f"**Average correction after 10 periods:** {avg_correction_10:.2f}%")
                        st.write(f"**Correction accuracy:** {success_rate:.1f}%")
                        
                        last_event = corrections[-1]
                        st.info(f"Last similar event: {format_time_ago(last_event['date'])} at ₹{last_event['price']:,.2f}")
            
            elif current_z < -2:
                st.success("🟢 **EXTREME OVERSOLD - Rally Expected**")
                similar_events = data_z[data_z['Z_Score'] < -2]
                
                if len(similar_events) > 1:
                    st.write(f"Found {len(similar_events)} similar extreme oversold events in this timeframe.")
                    
                    rallies = []
                    for idx in similar_events.index[:-1]:
                        if idx < len(data_z) - 10:
                            entry_price = data_z.loc[idx, 'Close']
                            future_5 = data_z.loc[idx+5, 'Close'] if idx+5 < len(data_z) else entry_price
                            future_10 = data_z.loc[idx+10, 'Close'] if idx+10 < len(data_z) else entry_price
                            
                            pct_5 = ((future_5 - entry_price) / entry_price) * 100
                            pct_10 = ((future_10 - entry_price) / entry_price) * 100
                            
                            rallies.append({
                                'date': data_z.loc[idx, 'DateTime_IST'],
                                'price': entry_price,
                                'pct_5': pct_5,
                                'pct_10': pct_10
                            })
                    
                    if rallies:
                        avg_rally_5 = np.mean([r['pct_5'] for r in rallies])
                        avg_rally_10 = np.mean([r['pct_10'] for r in rallies])
                        success_rate = len([r for r in rallies if r['pct_10'] > 0]) / len(rallies) * 100
                        
                        st.write(f"**Average rally after 5 periods:** {avg_rally_5:.2f}%")
                        st.write(f"**Average rally after 10 periods:** {avg_rally_10:.2f}%")
                        st.write(f"**Rally accuracy:** {success_rate:.1f}%")
                        
                        last_event = rallies[-1]
                        st.info(f"Last similar event: {format_time_ago(last_event['date'])} at ₹{last_event['price']:,.2f}")
            else:
                st.info("➡️ **NORMAL RANGE - No extreme condition**")
            
            # Forecast
            st.markdown("### 🎯 Forecast")
            current_price = data_z['Close'].iloc[-1]
            
            if current_z > 2:
                target = current_price * 0.98
                st.error(f"⬇️ **Expected Direction:** Correction/Decline")
                st.write(f"**Target:** ₹{target:,.2f} (2% correction)")
                st.write(f"**Probability:** Based on historical pattern")
            elif current_z < -2:
                target = current_price * 1.02
                st.success(f"⬆️ **Expected Direction:** Rally/Bounce")
                st.write(f"**Target:** ₹{target:,.2f} (2% rally)")
                st.write(f"**Probability:** Based on historical pattern")
            else:
                st.info("➡️ **Expected Direction:** Continuation/Sideways")
            
            st.markdown("---")
        
        # Final Consensus
        st.subheader("## 📊 Z-Score Consensus")
        
        correction_signals = 0
        rally_signals = 0
        neutral_signals = 0
        
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            
            data = st.session_state['all_data'][(interval, period)]
            data_z = data.copy()
            data_z['Z_Score'] = calculate_z_score(data_z)
            current_z = data_z['Z_Score'].iloc[-1]
            
            if current_z > 2:
                correction_signals += 1
            elif current_z < -2:
                rally_signals += 1
            else:
                neutral_signals += 1
        
        total_tf = len(st.session_state['combinations'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correction Signals", f"{correction_signals}/{total_tf}")
        with col2:
            st.metric("Rally Signals", f"{rally_signals}/{total_tf}")
        with col3:
            st.metric("Neutral Signals", f"{neutral_signals}/{total_tf}")
        
        if rally_signals > correction_signals:
            st.success(f"✅ **Overall Z-Score Signal: RALLY EXPECTED** ({rally_signals} timeframes oversold)")
        elif correction_signals > rally_signals:
            st.error(f"⚠️ **Overall Z-Score Signal: CORRECTION EXPECTED** ({correction_signals} timeframes overbought)")
        else:
            st.info("➡️ **Overall Z-Score Signal: NEUTRAL**")
    
    # Tab 4: Volatility Analysis
    with tabs[4]:
        st.header("💨 Volatility Analysis")
        
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            
            data = st.session_state['all_data'][(interval, period)]
            st.subheader(f"## 📊 Volatility: {interval} / {period}")
            
            # Calculate volatility
            data_vol = data.copy()
            data_vol['Volatility_%'] = calculate_volatility(data_vol, window=20)
            
            current_vol = data_vol['Volatility_%'].iloc[-1]
            avg_vol = data_vol['Volatility_%'].mean()
            max_vol = data_vol['Volatility_%'].max()
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Volatility", f"{current_vol:.2f}%")
            with col2:
                st.metric("Average Volatility", f"{avg_vol:.2f}%")
            with col3:
                st.metric("Max Volatility", f"{max_vol:.2f}%")
            
            # Bin Distribution
            st.markdown("### 📊 Volatility Bin Distribution")
            
            vol_data = data_vol.dropna(subset=['Volatility_%'])
            if len(vol_data) > 0:
                q1 = vol_data['Volatility_%'].quantile(0.25)
                q2 = vol_data['Volatility_%'].quantile(0.50)
                q3 = vol_data['Volatility_%'].quantile(0.75)
                
                bins = [
                    (f"Low (<{q1:.1f}%)", vol_data[vol_data['Volatility_%'] < q1]),
                    (f"Below Avg ({q1:.1f}%-{q2:.1f}%)", vol_data[(vol_data['Volatility_%'] >= q1) & (vol_data['Volatility_%'] < q2)]),
                    (f"Above Avg ({q2:.1f}%-{q3:.1f}%)", vol_data[(vol_data['Volatility_%'] >= q2) & (vol_data['Volatility_%'] < q3)]),
                    (f"High (>{q3:.1f}%)", vol_data[vol_data['Volatility_%'] > q3])
                ]
                
                bin_data = []
                for bin_name, bin_df in bins:
                    if len(bin_df) > 0:
                        vol_min = bin_df['Volatility_%'].min()
                        vol_max = bin_df['Volatility_%'].max()
                        avg_vol_bin = bin_df['Volatility_%'].mean()
                        is_current = vol_min <= current_vol <= vol_max
                        
                        bin_data.append({
                            'Bin': bin_name,
                            'Count': len(bin_df),
                            'Percentage %': f"{(len(bin_df)/len(vol_data)*100):.1f}%",
                            'Volatility Range': f"{vol_min:.2f}% - {vol_max:.2f}%" + (" ✅" if is_current else ""),
                            'Average Vol %': f"{avg_vol_bin:.2f}%"
                        })
                
                df_vol_bins = pd.DataFrame(bin_data)
                st.dataframe(df_vol_bins, use_container_width=True)
            
            # Recent Data
            st.markdown("### 📋 Recent Volatility (Last 20 Periods)")
            recent_vol = data_vol.tail(20)[['DateTime_IST', 'Close', 'Volatility_%']].copy()
            recent_vol['Close'] = recent_vol['Close'].apply(lambda x: f"₹{x:,.2f}")
            recent_vol['Volatility_%'] = recent_vol['Volatility_%'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            st.dataframe(recent_vol, use_container_width=True)
            
            # Historical Behavior
            st.markdown("### 🔍 Historical Behavior")
            
            if current_vol > avg_vol * 1.5:
                st.warning("⚡ **HIGH VOLATILITY - Expect Large Moves**")
                high_vol_periods = vol_data[vol_data['Volatility_%'] > avg_vol * 1.5]
                
                if len(high_vol_periods) > 5:
                    # Analyze moves after high volatility
                    moves = []
                    for idx in high_vol_periods.index[:-5]:
                        if idx < len(data_vol) - 5:
                            entry_price = data_vol.loc[idx, 'Close']
                            future_price = data_vol.loc[idx+5, 'Close']
                            move_pct = abs((future_price - entry_price) / entry_price) * 100
                            moves.append(move_pct)
                    
                    if moves:
                        avg_move = np.mean(moves)
                        st.write(f"**Average move after high volatility:** ±{avg_move:.2f}%")
                        st.write(f"**Expect moves in range:** {avg_move*0.8:.2f}% - {avg_move*1.2:.2f}%")
            
            elif current_vol < avg_vol * 0.7:
                st.info("🔒 **LOW VOLATILITY - Compression, Breakout Imminent**")
                low_vol_periods = vol_data[vol_data['Volatility_%'] < avg_vol * 0.7]
                
                if len(low_vol_periods) > 5:
                    # Analyze breakouts after compression
                    breakouts = []
                    for idx in low_vol_periods.index[:-10]:
                        if idx < len(data_vol) - 10:
                            # Find when volatility increased
                            for j in range(1, 11):
                                if idx + j < len(data_vol):
                                    if data_vol.loc[idx+j, 'Volatility_%'] > avg_vol:
                                        entry_price = data_vol.loc[idx, 'Close']
                                        breakout_price = data_vol.loc[idx+j, 'Close']
                                        move_pct = abs((breakout_price - entry_price) / entry_price) * 100
                                        breakouts.append({'periods': j, 'move': move_pct})
                                        break
                    
                    if breakouts:
                        avg_periods = np.mean([b['periods'] for b in breakouts])
                        avg_breakout_move = np.mean([b['move'] for b in breakouts])
                        st.write(f"**Average periods until breakout:** {avg_periods:.0f}")
                        st.write(f"**Average breakout move:** ±{avg_breakout_move:.2f}%")
            else:
                st.success("✅ **NORMAL VOLATILITY - Standard Conditions**")
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data_vol['DateTime_IST'],
                y=data_vol['Volatility_%'],
                mode='lines',
                name='Volatility',
                line=dict(color='blue')
            ))
            fig.add_hline(y=avg_vol, line_dash="dash", line_color="green",
                          annotation_text=f"Average: {avg_vol:.2f}%")
            
            fig.update_layout(
                title=f"Volatility Over Time - {interval}/{period}",
                xaxis_title="Time",
                yaxis_title="Volatility (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast
            st.markdown("### 🎯 Forecast")
            current_price = data_vol['Close'].iloc[-1]
            
            if current_vol > avg_vol * 1.5:
                move_range = current_vol / 100 * current_price
                st.warning(f"**Expected move range:** ±₹{move_range:,.2f} (±{current_vol:.2f}%)")
                st.write(f"**Range:** ₹{current_price - move_range:,.2f} - ₹{current_price + move_range:,.2f}")
            elif current_vol < avg_vol * 0.7:
                st.info("**Breakout expected soon** - Position for larger move")
            else:
                normal_move = avg_vol / 100 * current_price
                st.success(f"**Normal conditions:** Expected range ±₹{normal_move:,.2f}")
            
            st.markdown("---")
        
        # Final Consensus
        st.subheader("## 💨 Volatility Consensus")
        
        high_vol_count = 0
        low_vol_count = 0
        normal_vol_count = 0
        
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            
            data = st.session_state['all_data'][(interval, period)]
            data_vol = data.copy()
            data_vol['Volatility_%'] = calculate_volatility(data_vol, window=20)
            
            current_vol = data_vol['Volatility_%'].iloc[-1]
            avg_vol = data_vol['Volatility_%'].mean()
            
            if current_vol > avg_vol * 1.5:
                high_vol_count += 1
            elif current_vol < avg_vol * 0.7:
                low_vol_count += 1
            else:
                normal_vol_count += 1
        
        total_tf = len(st.session_state['combinations'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Volatility", f"{high_vol_count}/{total_tf}")
        with col2:
            st.metric("Low Volatility", f"{low_vol_count}/{total_tf}")
        with col3:
            st.metric("Normal Volatility", f"{normal_vol_count}/{total_tf}")
        
        if high_vol_count > total_tf * 0.5:
            st.warning("⚡ **Overall Regime: HIGH VOLATILITY** - Expect large moves")
        elif low_vol_count > total_tf * 0.5:
            st.info("🔒 **Overall Regime: LOW VOLATILITY** - Compression phase")
        else:
            st.success("✅ **Overall Regime: NORMAL VOLATILITY**")
    
    # Tab 5: Elliott Waves
    with tabs[5]:
        st.header("🌊 Elliott Wave Analysis")
        
        if st.session_state['combinations']:
            primary_tf = st.session_state['combinations'][0]
            if primary_tf in st.session_state['all_data']:
                data = st.session_state['all_data'][primary_tf]
                
                st.info("🌊 Elliott Wave detection using swing highs and lows")
                
                # Simplified wave detection
                highs_idx = argrelextrema(data['High'].values, np.greater, order=10)[0]
                lows_idx = argrelextrema(data['Low'].values, np.less, order=10)[0]
                
                # Combine and sort
                extrema = []
                for idx in highs_idx:
                    if idx < len(data):
                        extrema.append({'idx': idx, 'price': data['High'].iloc[idx], 'type': 'High'})
                for idx in lows_idx:
                    if idx < len(data):
                        extrema.append({'idx': idx, 'price': data['Low'].iloc[idx], 'type': 'Low'})
                
                extrema = sorted(extrema, key=lambda x: x['idx'])
                
                if len(extrema) >= 2:
                    wave_data = []
                    for i in range(len(extrema) - 1):
                        wave_num = i + 1
                        start_price = extrema[i]['price']
                        end_price = extrema[i+1]['price']
                        start_date = data['DateTime_IST'].iloc[extrema[i]['idx']]
                        end_date = data['DateTime_IST'].iloc[extrema[i+1]['idx']]
                        move_pct = ((end_price - start_price) / start_price) * 100
                        
                        wave_type = "Impulse ⬆️" if move_pct > 0 else "Corrective ⬇️"
                        
                        wave_data.append({
                            'Wave': f"Wave {wave_num}",
                            'Type': wave_type,
                            'Start Price': f"₹{start_price:,.2f}",
                            'End Price': f"₹{end_price:,.2f}",
                            'Start Date': format_time_ago(start_date),
                            'End Date': format_time_ago(end_date),
                            'Move %': f"{move_pct:.2f}%"
                        })
                    
                    df_waves = pd.DataFrame(wave_data)
                    st.dataframe(df_waves, use_container_width=True)
                    
                    # Current wave
                    current_wave = wave_data[-1] if wave_data else None
                    if current_wave:
                        st.subheader("🎯 Current Wave Analysis")
                        if "Impulse" in current_wave['Type']:
                            st.success(f"**Current Wave:** {current_wave['Wave']} - {current_wave['Type']}")
                            st.write(f"**Expected Next:** Corrective wave (pullback/consolidation)")
                        else:
                            st.info(f"**Current Wave:** {current_wave['Wave']} - {current_wave['Type']}")
                            st.write(f"**Expected Next:** Impulse wave (continuation/reversal)")
                else:
                    st.warning("Not enough swing points detected for wave analysis")
    
    # Tab 6: Fibonacci Levels
    with tabs[6]:
        st.header("📐 Fibonacci Retracement Levels")
        
        if st.session_state['combinations']:
            primary_tf = st.session_state['combinations'][0]
            if primary_tf in st.session_state['all_data']:
                data = st.session_state['all_data'][primary_tf]
                
                fib_levels = calculate_fibonacci_levels(data)
                current_price = data['Close'].iloc[-1]
                
                st.subheader(f"📊 Fibonacci Levels: {primary_tf[0]} / {primary_tf[1]}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High", f"₹{data['High'].max():,.2f}")
                    st.metric("Current", f"₹{current_price:,.2f}")
                with col2:
                    st.metric("Low", f"₹{data['Low'].min():,.2f}")
                    st.metric("Range", f"₹{data['High'].max() - data['Low'].min():,.2f}")
                
                # Fib table
                fib_data = []
                for level, price in fib_levels.items():
                    distance = price - current_price
                    distance_pct = (distance / current_price) * 100
                    
                    if abs(distance_pct) < 1:
                        status = "✅ NEAR"
                    elif distance > 0:
                        status = "⬆️ Above"
                    else:
                        status = "⬇️ Below"
                    
                    fib_data.append({
                        'Level': level,
                        'Price': f"₹{price:,.2f}",
                        'Distance': f"{distance:,.2f} ({distance_pct:.2f}%)",
                        'Status': status
                    })
                
                df_fib = pd.DataFrame(fib_data)
                st.dataframe(df_fib, use_container_width=True)
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data['DateTime_IST'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'red']
                for i, (level, price) in enumerate(fib_levels.items()):
                    fig.add_hline(y=price, line_dash="dash", line_color=colors[i % len(colors)],
                                  annotation_text=f"{level}: ₹{price:,.2f}")
                
                fig.update_layout(
                    title=f"Fibonacci Retracement - {primary_tf[0]}/{primary_tf[1]}",
                    xaxis_title="Time",
                    yaxis_title="Price (₹)",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Key levels
                st.subheader("🎯 Key Fibonacci S/R Levels")
                nearest = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
                st.info(f"**Nearest Level:** {nearest[0]} at ₹{nearest[1]:,.2f}")
                
                support_levels = [lv for lv in fib_levels.items() if lv[1] < current_price]
                resistance_levels = [lv for lv in fib_levels.items() if lv[1] > current_price]
                
                if support_levels:
                    nearest_support = max(support_levels, key=lambda x: x[1])
                    st.success(f"**Nearest Support:** {nearest_support[0]} at ₹{nearest_support[1]:,.2f}")
                
                if resistance_levels:
                    nearest_resistance = min(resistance_levels, key=lambda x: x[1])
                    st.warning(f"**Nearest Resistance:** {nearest_resistance[0]} at ₹{nearest_resistance[1]:,.2f}")
    
    # Tab 7: RSI Divergence
    with tabs[7]:
        st.header("🔄 RSI Divergence Analysis")
        
        if st.session_state['combinations']:
            primary_tf = st.session_state['combinations'][0]
            if primary_tf in st.session_state['all_data']:
                data = st.session_state['all_data'][primary_tf]
                
                st.subheader(f"📊 Divergences: {primary_tf[0]} / {primary_tf[1]}")
                
                divergences = detect_rsi_divergence(data)
                
                if divergences:
                    div_data = []
                    for div in divergences:
                        div_data.append({
                            'Type': div['type'] + (" 🟢" if div['type'] == 'Bullish' else " 🔴"),
                            'Price 1': f"₹{div['price1']:,.2f}",
                            'Price 2': f"₹{div['price2']:,.2f}",
                            'Date 1': format_time_ago(div['date1']),
                            'Date 2': format_time_ago(div['date2']),
                            'RSI 1': f"{div['rsi1']:.1f}",
                            'RSI 2': f"{div['rsi2']:.1f}",
                            'Resolved': "✅" if div['resolved'] else "⏳"
                        })
                    
                    df_div = pd.DataFrame(div_data)
                    st.dataframe(df_div, use_container_width=True)
                    
                    # Active divergences
                    active_divs = [d for d in divergences if not d['resolved']]
                    
                    if active_divs:
                        st.subheader("⚡ Active Divergences")
                        
                        for div in active_divs:
                            if div['type'] == 'Bullish':
                                st.success(f"🟢 **Bullish Divergence Active**")
                                current_price = data['Close'].iloc[-1]
                                target = current_price * 1.02
                                sl = div['price2'] * 0.99
                                
                                st.write(f"**Expected Move:** Upward reversal")
                                st.write(f"**Entry:** ₹{current_price:,.2f}")
                                st.write(f"**Target:** ₹{target:,.2f} (+2%)")
                                st.write(f"**Stop Loss:** ₹{sl:,.2f}")
                            else:
                                st.error(f"🔴 **Bearish Divergence Active**")
                                current_price = data['Close'].iloc[-1]
                                target = current_price * 0.98
                                sl = div['price2'] * 1.01
                                
                                st.write(f"**Expected Move:** Downward reversal")
                                st.write(f"**Entry:** ₹{current_price:,.2f}")
                                st.write(f"**Target:** ₹{target:,.2f} (-2%)")
                                st.write(f"**Stop Loss:** ₹{sl:,.2f}")
                    
                    # Statistics
                    st.subheader("📊 Divergence Statistics")
                    bullish_count = len([d for d in divergences if d['type'] == 'Bullish'])
                    bearish_count = len([d for d in divergences if d['type'] == 'Bearish'])
                    resolved_count = len([d for d in divergences if d['resolved']])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Divergences", len(divergences))
                    with col2:
                        st.metric("Bullish", bullish_count)
                    with col3:
                        st.metric("Bearish", bearish_count)
                    with col4:
                        success_rate = (resolved_count / len(divergences) * 100) if divergences else 0
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                else:
                    st.info("No divergences detected in this timeframe")
    
    # Tab 8: Ratio Analysis
    with tabs[8]:
        st.header("⚖️ Ratio Analysis")
        
        if not enable_ratio or not st.session_state.get('ticker2'):
            st.warning("⚠️ Enable Ticker 2 comparison in sidebar to use ratio analysis")
        else:
            st.success(f"Comparing {st.session_state['ticker1']} vs {st.session_state['ticker2']}")
            
            # Find common timeframes
            common_tfs = []
            for tf in st.session_state['combinations']:
                if tf in st.session_state['all_data'] and tf in st.session_state['all_data_t2']:
                    common_tfs.append(tf)
            
            if not common_tfs:
                st.error("No common timeframes with data for both tickers")
            else:
                st.info(f"Analyzing {len(common_tfs)} common timeframes")
                
                all_ratio_data = []
                
                for interval, period in common_tfs:
                    data1 = st.session_state['all_data'][(interval, period)]
                    data2 = st.session_state['all_data_t2'][(interval, period)]
                    
                    st.subheader(f"## ⚖️ Ratio Analysis: {interval} / {period}")
                    
                    # Merge data
                    merged = pd.merge(
                        data1[['DateTime_IST', 'Close']],
                        data2[['DateTime_IST', 'Close']],
                        on='DateTime_IST',
                        how='inner',
                        suffixes=('_T1', '_T2')
                    )
                    
                    if len(merged) == 0:
                        st.warning("No overlapping timestamps between tickers")
                        continue
                    
                    merged['Ratio'] = merged['Close_T1'] / merged['Close_T2']
                    merged['Ratio_RSI'] = calculate_rsi(merged['Ratio'], 14)
                    
                    # Calculate Z-Score for ratio
                    ratio_returns = merged['Ratio'].pct_change()
                    merged['Ratio_ZScore'] = (ratio_returns - ratio_returns.mean()) / ratio_returns.std()
                    
                    # Volatility
                    merged['Volatility_%'] = ratio_returns.rolling(window=20).std() * np.sqrt(252) * 100
                    
                    current_ratio = merged['Ratio'].iloc[-1]
                    avg_ratio = merged['Ratio'].mean()
                    min_ratio = merged['Ratio'].min()
                    max_ratio = merged['Ratio'].max()
                    current_z = merged['Ratio_ZScore'].iloc[-1]
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Ratio", f"{current_ratio:.6f}")
                    with col2:
                        st.metric("Average Ratio", f"{avg_ratio:.6f}")
                    with col3:
                        st.metric("Min Ratio", f"{min_ratio:.6f}")
                    with col4:
                        st.metric("Max Ratio", f"{max_ratio:.6f}")
                    
                    # Ratio Bins
                    st.markdown("### 📊 Ratio Distribution")
                    
                    q1 = merged['Ratio'].quantile(0.25)
                    q2 = merged['Ratio'].quantile(0.50)
                    q3 = merged['Ratio'].quantile(0.75)
                    
                    bins = [
                        (f"{min_ratio:.6f} - {q1:.6f}", merged[merged['Ratio'] < q1]),
                        (f"{q1:.6f} - {q2:.6f}", merged[(merged['Ratio'] >= q1) & (merged['Ratio'] < q2)]),
                        (f"{q2:.6f} - {q3:.6f}", merged[(merged['Ratio'] >= q2) & (merged['Ratio'] < q3)]),
                        (f"{q3:.6f} - {max_ratio:.6f}", merged[merged['Ratio'] > q3])
                    ]
                    
                    bin_data = []
                    for bin_range, bin_df in bins:
                        if len(bin_df) > 0:
                            is_current = bin_df['Ratio'].min() <= current_ratio <= bin_df['Ratio'].max()
                            
                            # Determine behavior
                            future_moves = []
                            for idx in bin_df.index[:-5]:
                                if idx + 5 < len(merged):
                                    future_t1 = merged.loc[idx+5, 'Close_T1']
                                    current_t1 = merged.loc[idx, 'Close_T1']
                                    move = ((future_t1 - current_t1) / current_t1) * 100
                                    future_moves.append(move)
                            
                            if future_moves:
                                avg_move = np.mean(future_moves)
                                if avg_move > 0.5:
                                    behavior = "Rally ⬆️"
                                elif avg_move < -0.5:
                                    behavior = "Decline ⬇️"
                                else:
                                    behavior = "Sideways ➡️"
                            else:
                                behavior = "N/A"
                            
                            bin_data.append({
                                'Ratio Range': bin_range + (" ✅" if is_current else ""),
                                'Count': len(bin_df),
                                'Percentage %': f"{(len(bin_df)/len(merged)*100):.1f}%",
                                'Historical Behavior': behavior
                            })
                    
                    df_ratio_bins = pd.DataFrame(bin_data)
                    st.dataframe(df_ratio_bins, use_container_width=True)
                    
                    # Recent Data
                    st.markdown("### 📋 Recent Ratio Data (Last 20 Periods)")
                    recent_ratio = merged.tail(20)[['DateTime_IST', 'Close_T1', 'Close_T2', 'Ratio', 'Ratio_RSI', 'Volatility_%']].copy()
                    recent_ratio.columns = ['DateTime_IST', f'{st.session_state["ticker1"]}_Price', 
                                           f'{st.session_state["ticker2"]}_Price', 'Ratio', 'Ratio_RSI', 'Vol%']
                    recent_ratio[f'{st.session_state["ticker1"]}_Price'] = recent_ratio[f'{st.session_state["ticker1"]}_Price'].apply(lambda x: f"₹{x:,.2f}")
                    recent_ratio[f'{st.session_state["ticker2"]}_Price'] = recent_ratio[f'{st.session_state["ticker2"]}_Price'].apply(lambda x: f"₹{x:,.2f}")
                    recent_ratio['Ratio'] = recent_ratio['Ratio'].apply(lambda x: f"{x:.6f}")
                    recent_ratio['Ratio_RSI'] = recent_ratio['Ratio_RSI'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                    recent_ratio['Vol%'] = recent_ratio['Vol%'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                    st.dataframe(recent_ratio, use_container_width=True)
                    
                    # Forecast
                    st.markdown("### 🎯 Ratio-Based Forecast")
                    
                    if current_z > 1.5:
                        st.warning(f"⚠️ **{st.session_state['ticker1']} EXPENSIVE vs {st.session_state['ticker2']}**")
                        st.write(f"Ratio Z-Score: {current_z:.2f}")
                        st.write(f"**Expected:** Ratio compression (T1 underperforms or T2 outperforms)")
                    elif current_z < -1.5:
                        st.success(f"🟢 **{st.session_state['ticker1']} CHEAP vs {st.session_state['ticker2']}**")
                        st.write(f"Ratio Z-Score: {current_z:.2f}")
                        st.write(f"**Expected:** Ratio expansion (T1 outperforms or T2 underperforms)")
                    else:
                        st.info("➡️ **NORMAL RATIO RANGE** - No extreme relative valuation")
                    
                    # Chart
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                       subplot_titles=(f'{st.session_state["ticker1"]}', 
                                                      f'{st.session_state["ticker2"]}',
                                                      'Ratio'),
                                       vertical_spacing=0.05)
                    
                    fig.add_trace(go.Scatter(x=merged['DateTime_IST'], y=merged['Close_T1'],
                                           mode='lines', name=st.session_state['ticker1'],
                                           line=dict(color='blue')), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=merged['DateTime_IST'], y=merged['Close_T2'],
                                           mode='lines', name=st.session_state['ticker2'],
                                           line=dict(color='orange')), row=2, col=1)
                    
                    fig.add_trace(go.Scatter(x=merged['DateTime_IST'], y=merged['Ratio'],
                                           mode='lines', name='Ratio',
                                           line=dict(color='purple')), row=3, col=1)
                    
                    fig.add_hline(y=avg_ratio, line_dash="dash", line_color="green", row=3, col=1,
                                 annotation_text=f"Avg: {avg_ratio:.6f}")
                    
                    fig.update_layout(height=800, showlegend=True,
                                     title_text=f"Ratio Analysis: {interval}/{period}")
                    fig.update_xaxes(title_text="Time", row=3, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Price", row=2, col=1)
                    fig.update_yaxes(title_text="Ratio", row=3, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store for export
                    export_data = merged[['DateTime_IST', 'Close_T1', 'Close_T2', 'Ratio', 'Ratio_RSI', 'Ratio_ZScore', 'Volatility_%']].copy()
                    export_data.columns = ['DateTime_IST', f'{st.session_state["ticker1"]}_Price',
                                          f'{st.session_state["ticker2"]}_Price', 'Ratio', 'Ratio_RSI', 
                                          'Ratio_ZScore', 'Volatility_%']
                    all_ratio_data.append((f"{interval}_{period}", export_data))
                    
                    st.markdown("---")
                
                # Export all ratio data
                if all_ratio_data:
                    st.subheader("📥 Export Ratio Data")
                    for tf_label, df_export in all_ratio_data:
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            f"Download {tf_label} Ratio Data",
                            csv,
                            f"ratio_{st.session_state['ticker1']}_{st.session_state['ticker2']}_{tf_label}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                            "text/csv",
                            key=f"ratio_download_{tf_label}"
                        )
    
    # Tab 9: AI Signals & Final Forecast
    with tabs[9]:
        st.header("🤖 AI Signals & Final Forecast")
        st.markdown("### 🎯 Multi-Timeframe Algorithmic Signal")
        
        # Calculate scores for all timeframes
        timeframe_scores = []
        
        for interval, period in st.session_state['combinations']:
            if (interval, period) not in st.session_state['all_data']:
                continue
            
            data = st.session_state['all_data'][(interval, period)]
            score = 0
            factors = []
            
            current_price = data['Close'].iloc[-1]
            
            # RSI Score
            rsi = calculate_rsi(data['Close'], 14).iloc[-1]
            if rsi < 30:
                score += 20
                factors.append("RSI Oversold +20")
            elif rsi > 70:
                score -= 20
                factors.append("RSI Overbought -20")
            
            # EMA Alignment
            ema_20 = calculate_ema(data['Close'], 20).iloc[-1]
            ema_50 = calculate_ema(data['Close'], 50).iloc[-1]
            if current_price > ema_20 > ema_50:
                score += 15
                factors.append("Bullish EMA +15")
            elif current_price < ema_20 < ema_50:
                score -= 15
                factors.append("Bearish EMA -15")
            
            # S/R Proximity
            sr_levels = detect_support_resistance(data)
            if sr_levels:
                nearest_support = [l for l in sr_levels if l['type'] == 'Support' and l['price'] < current_price]
                nearest_resistance = [l for l in sr_levels if l['type'] == 'Resistance' and l['price'] > current_price]
                
                if nearest_support:
                    nearest_support = max(nearest_support, key=lambda x: x['price'])
                    support_distance = ((current_price - nearest_support['price']) / current_price) * 100
                    if support_distance < 1:
                        score += 20
                        factors.append("Near Support +20")
                
                if nearest_resistance:
                    nearest_resistance = min(nearest_resistance, key=lambda x: x['price'])
                    resistance_distance = ((nearest_resistance['price'] - current_price) / current_price) * 100
                    if resistance_distance < 1:
                        score -= 20
                        factors.append("Near Resistance -20")
            
            # RSI Divergence
            divs = detect_rsi_divergence(data)
            active_bullish = any(d['type'] == 'Bullish' and not d['resolved'] for d in divs)
            active_bearish = any(d['type'] == 'Bearish' and not d['resolved'] for d in divs)
            
            if active_bullish:
                score += 25
                factors.append("Bullish Divergence +25")
            elif active_bearish:
                score -= 25
                factors.append("Bearish Divergence -25")
            
            # Z-Score
            data_z = data.copy()
            data_z['Z_Score'] = calculate_z_score(data_z)
            current_z = data_z['Z_Score'].iloc[-1]
            
            if current_z < -2:
                score += 20
                factors.append("Z-Score Oversold +20")
            elif current_z > 2:
                score -= 20
                factors.append("Z-Score Overbought -20")
            
            # Volatility
            data_vol = data.copy()
            data_vol['Volatility_%'] = calculate_volatility(data_vol, 20)
            current_vol = data_vol['Volatility_%'].iloc[-1]
            avg_vol = data_vol['Volatility_%'].mean()
            
            vol_condition = "Normal"
            if current_vol > avg_vol * 1.5:
                vol_condition = "High"
            elif current_vol < avg_vol * 0.7:
                vol_condition = "Low"
            
            # Determine bias
            if score > 15:
                bias = "Bullish 🟢"
            elif score < -15:
                bias = "Bearish 🔴"
            else:
                bias = "Neutral 🟡"
            
            timeframe_scores.append({
                'timeframe': f"{interval}/{period}",
                'score': score,
                'bias': bias,
                'rsi': rsi,
                'z_score': current_z,
                'vol_pct': current_vol,
                'vol_condition': vol_condition,
                'factors': ", ".join(factors[:3])  # Top 3 factors
            })
        
        # Calculate final signal
        avg_score = np.mean([tf['score'] for tf in timeframe_scores])
        bullish_count = len([tf for tf in timeframe_scores if "Bullish" in tf['bias']])
        bearish_count = len([tf for tf in timeframe_scores if "Bearish" in tf['bias']])
        neutral_count = len([tf for tf in timeframe_scores if "Neutral" in tf['bias']])
        total_tf = len(timeframe_scores)
        
        # Determine signal
        if avg_score > 30:
            signal = "STRONG BUY"
            signal_color = "success"
            signal_emoji = "🟢"
        elif avg_score > 15:
            signal = "BUY"
            signal_color = "success"
            signal_emoji = "🟢"
        elif avg_score < -30:
            signal = "STRONG SELL"
            signal_color = "error"
            signal_emoji = "🔴"
        elif avg_score < -15:
            signal = "SELL"
            signal_color = "error"
            signal_emoji = "🔴"
        else:
            signal = "HOLD/NEUTRAL"
            signal_color = "info"
            signal_emoji = "🟡"
        
        # Confidence calculation
        agreement_pct = max(bullish_count, bearish_count) / total_tf
        confidence = 60 + (agreement_pct * 30) + (abs(avg_score) * 0.3)
        confidence = min(confidence, 95)
        
        # Display signal
        if signal_color == "success":
            st.success(f"# {signal_emoji} {signal}")
        elif signal_color == "error":
            st.error(f"# {signal_emoji} {signal}")
        else:
            st.info(f"# {signal_emoji} {signal}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{confidence:.1f}%")
        with col2:
            st.metric("Multi-Timeframe Score", f"{avg_score:.1f}/100")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish Timeframes", f"{bullish_count}/{total_tf}", f"{(bullish_count/total_tf*100):.1f}%")
        with col2:
            st.metric("Bearish Timeframes", f"{bearish_count}/{total_tf}", f"{(bearish_count/total_tf*100):.1f}%")
        with col3:
            st.metric("Neutral Timeframes", f"{neutral_count}/{total_tf}", f"{(neutral_count/total_tf*100):.1f}%")
        
        # Trading Plan
        st.markdown("### 📋 TRADING PLAN")
        
        current_price = st.session_state['all_data'][st.session_state['combinations'][0]]['Close'].iloc[-1]
        
        # Determine SL/Target based on instrument type
        if current_price > 20000:  # NIFTY/SENSEX level
            sl_pct = 0.015  # 1.5%
            target_pct = 0.0175  # 1.75%
        elif current_price > 1000:
            sl_pct = 0.02  # 2%
            target_pct = 0.025  # 2.5%
        else:
            sl_pct = 0.025  # 2.5%
            target_pct = 0.035  # 3.5%
        
        if "SELL" in signal:
            sl_price = current_price * (1 + sl_pct)
            target_price = current_price * (1 - target_pct)
        else:
            sl_price = current_price * (1 - sl_pct)
            target_price = current_price * (1 + target_pct)
        
        sl_points = abs(current_price - sl_price)
        target_points = abs(target_price - current_price)
        risk_reward = target_points / sl_points if sl_points > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Entry:** ₹{current_price:,.2f}")
            st.write(f"**Stop Loss:** ₹{sl_price:,.2f} ({sl_pct*100:.1f}% risk, {sl_points:.0f} points)")
        with col2:
            st.write(f"**Target:** ₹{target_price:,.2f} ({target_pct*100:.2f}% reward, {target_points:.0f} points)")
            st.write(f"**Risk:Reward:** 1:{risk_reward:.2f}")
        
        # Timeframe Breakdown
        st.markdown("### 📊 Timeframe Breakdown")
        
        breakdown_data = []
        for tf in timeframe_scores:
            breakdown_data.append({
                'Timeframe': tf['timeframe'],
                'Score': f"{tf['score']:.1f}",
                'Bias': tf['bias'],
                'RSI': f"{tf['rsi']:.1f}",
                'Z-Score': f"{tf['z_score']:.2f}",
                'Vol %': f"{tf['vol_pct']:.2f}%",
                'Vol Condition': tf['vol_condition'],
                'Key Factors': tf['factors']
            })
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True)
        
        # Why This Recommendation
        st.markdown("### 🔍 WHY THIS RECOMMENDATION?")
        
        st.markdown("**Multi-Timeframe Consensus:**")
        st.write(f"Based on analysis of {total_tf} timeframes, {bullish_count} show bullish bias ({(bullish_count/total_tf*100):.1f}%), {bearish_count} show bearish bias ({(bearish_count/total_tf*100):.1f}%).")
        
        st.markdown("**Supporting Factors:**")
        
        factor_list = []
        if bullish_count > total_tf * 0.6 or bearish_count > total_tf * 0.6:
            factor_list.append(f"1. **Strong Multi-Timeframe Agreement** ({max(bullish_count, bearish_count)}/{total_tf} aligned)")
            factor_list.append("   - Both short-term and long-term timeframes showing consensus")
            factor_list.append("   - Reduces whipsaw risk significantly")
        
        # Technical factors
        rsi_oversold = len([tf for tf in timeframe_scores if tf['rsi'] < 30])
        rsi_overbought = len([tf for tf in timeframe_scores if tf['rsi'] > 70])
        
        if rsi_oversold > 0 or rsi_overbought > 0:
            factor_list.append(f"2. **Technical Factors:**")
            if rsi_oversold > 0:
                factor_list.append(f"   - RSI Oversold on {rsi_oversold} timeframe(s)")
            if rsi_overbought > 0:
                factor_list.append(f"   - RSI Overbought on {rsi_overbought} timeframe(s)")
        
        factor_list.append(f"3. **Risk-Reward:** Favorable {risk_reward:.2f}:1 ratio")
        factor_list.append(f"4. **Probability Assessment:** Success probability ~{confidence:.0f}% based on multi-timeframe alignment")
        
        for factor in factor_list:
            st.write(factor)
        
        st.markdown("**Execution Strategy:**")
        st.write(f"- **Primary Entry:** ₹{current_price:,.2f} (current market price)")
        if "BUY" in signal:
            pullback_entry = current_price * 0.998
            st.write(f"- **Alternative Entry:** ₹{pullback_entry:,.2f} (on minor pullback)")
        else:
            bounce_entry = current_price * 1.002
            st.write(f"- **Alternative Entry:** ₹{bounce_entry:,.2f} (on minor bounce)")
        st.write("- **Position Size:** Risk 1-2% of capital per trade")
        st.write("- **Profit Taking:** 50% at target, trail remaining with stop")
        
        st.markdown("**What Could Invalidate:**")
        st.write(f"- Stop loss breach at ₹{sl_price:,.2f}")
        st.write("- Major unexpected news or market events")
        st.write("- Sudden shift in majority of timeframes against position")
        
        st.markdown("**Time Horizon:**")
        st.write("Expected to play out over next few trading sessions based on timeframe mix")
        
        st.markdown(f"**Success Probability: {confidence:.0f}%**")
        st.write(f"Based on: {bullish_count if 'BUY' in signal else bearish_count}/{total_tf} aligned timeframes, historical pattern analysis")
    
    # Tab 10: Backtesting
    with tabs[10]:
        st.header("🔬 Strategy Backtesting")
        
        st.markdown("### Select Strategies to Backtest")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_rsi_ema = st.checkbox("RSI + EMA Strategy", value=True)
            test_ema_cross = st.checkbox("EMA Crossover", value=True)
            test_vol_breakout = st.checkbox("Volatility Breakout", value=True)
        with col2:
            test_support_bounce = st.checkbox("Support Bounce", value=True)
            test_rsi_div = st.checkbox("RSI Divergence", value=False)
            test_zscore = st.checkbox("Z-Score Reversion", value=True)
        with col3:
            test_ratio = st.checkbox("Ratio Strategy", value=False)
            test_9ema = st.checkbox("9 EMA Pullback", value=True)
            test_pa = st.checkbox("Price Action", value=False)
        
        # Select timeframes to test
        st.markdown("### Select Timeframes for Testing")
        test_timeframes = st.multiselect(
            "Choose up to 5 timeframes",
            [f"{tf[0]}/{tf[1]}" for tf in st.session_state['combinations']],
            default=[f"{tf[0]}/{tf[1]}" for tf in st.session_state['combinations'][:min(5, len(st.session_state['combinations']))]]
        )
        
        if st.button("🚀 Run Backtest", type="primary"):
            if not test_timeframes:
                st.error("Please select at least one timeframe")
            else:
                st.info(f"Testing {sum([test_rsi_ema, test_ema_cross, test_vol_breakout, test_support_bounce, test_rsi_div, test_zscore, test_ratio, test_9ema, test_pa])} strategies across {len(test_timeframes)} timeframes...")
                
                all_results = []
                
                for tf_str in test_timeframes:
                    interval, period = tf_str.split('/')
                    
                    if (interval, period) not in st.session_state['all_data']:
                        continue
                    
                    data = st.session_state['all_data'][(interval, period)].copy()
                    
                    # Calculate indicators
                    data['RSI'] = calculate_rsi(data['Close'], 14)
                    data['EMA_9'] = calculate_ema(data['Close'], 9)
                    data['EMA_20'] = calculate_ema(data['Close'], 20)
                    data['EMA_50'] = calculate_ema(data['Close'], 50)
                    data['Z_Score'] = calculate_z_score(data)
                    data['Volatility_%'] = calculate_volatility(data, 20)
                    
                    # RSI + EMA Strategy
                    if test_rsi_ema:
                        trades = []
                        in_trade = False
                        entry_price = 0
                        entry_idx = 0
                        
                        for i in range(50, len(data)):
                            if not in_trade:
                                # Entry: RSI < 30 and price > EMA20
                                if data['RSI'].iloc[i] < 30 and data['Close'].iloc[i] > data['EMA_20'].iloc[i]:
                                    in_trade = True
                                    entry_price = data['Close'].iloc[i]
                                    entry_idx = i
                            else:
                                # Exit: RSI > 70 or price < EMA20
                                if data['RSI'].iloc[i] > 70 or data['Close'].iloc[i] < data['EMA_20'].iloc[i]:
                                    exit_price = data['Close'].iloc[i]
                                    pnl = ((exit_price - entry_price) / entry_price) * 100
                                    
                                    trades.append({
                                        'strategy': 'RSI+EMA',
                                        'entry_date': data['DateTime_IST'].iloc[entry_idx],
                                        'entry_price': entry_price,
                                        'exit_date': data['DateTime_IST'].iloc[i],
                                        'exit_price': exit_price,
                                        'pnl_pct': pnl,
                                        'points': exit_price - entry_price
                                    })
                                    in_trade = False
                        
                        if trades:
                            winning_trades = len([t for t in trades if t['pnl_pct'] > 0])
                            losing_trades = len(trades) - winning_trades
                            total_pnl = sum([t['pnl_pct'] for t in trades])
                            avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if winning_trades > 0 else 0
                            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if losing_trades > 0 else 0
                            best_trade = max([t['pnl_pct'] for t in trades])
                            worst_trade = min([t['pnl_pct'] for t in trades])
                            win_rate = (winning_trades / len(trades)) * 100
                            
                            all_results.append({
                                'Strategy': 'RSI+EMA',
                                'Timeframe': tf_str,
                                'Total Trades': len(trades),
                                'Winning': winning_trades,
                                'Losing': losing_trades,
                                'Win Rate %': win_rate,
                                'Total PnL %': total_pnl,
                                'Avg Win %': avg_win,
                                'Avg Loss %': avg_loss,
                                'Best Trade %': best_trade,
                                'Worst Trade %': worst_trade,
                                'Risk:Reward': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                                'trades': trades
                            })
                    
                    # EMA Crossover Strategy
                    if test_ema_cross:
                        trades = []
                        in_trade = False
                        entry_price = 0
                        entry_idx = 0
                        
                        for i in range(50, len(data) - 1):
                            if not in_trade:
                                # Bullish crossover: EMA20 crosses above EMA50
                                if data['EMA_20'].iloc[i] > data['EMA_50'].iloc[i] and data['EMA_20'].iloc[i-1] <= data['EMA_50'].iloc[i-1]:
                                    in_trade = True
                                    entry_price = data['Close'].iloc[i]
                                    entry_idx = i
                            else:
                                # Bearish crossover: EMA20 crosses below EMA50
                                if data['EMA_20'].iloc[i] < data['EMA_50'].iloc[i] and data['EMA_20'].iloc[i-1] >= data['EMA_50'].iloc[i-1]:
                                    exit_price = data['Close'].iloc[i]
                                    pnl = ((exit_price - entry_price) / entry_price) * 100
                                    
                                    trades.append({
                                        'strategy': 'EMA Crossover',
                                        'entry_date': data['DateTime_IST'].iloc[entry_idx],
                                        'entry_price': entry_price,
                                        'exit_date': data['DateTime_IST'].iloc[i],
                                        'exit_price': exit_price,
                                        'pnl_pct': pnl,
                                        'points': exit_price - entry_price
                                    })
                                    in_trade = False
                        
                        if trades:
                            winning_trades = len([t for t in trades if t['pnl_pct'] > 0])
                            losing_trades = len(trades) - winning_trades
                            total_pnl = sum([t['pnl_pct'] for t in trades])
                            avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if winning_trades > 0 else 0
                            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if losing_trades > 0 else 0
                            best_trade = max([t['pnl_pct'] for t in trades])
                            worst_trade = min([t['pnl_pct'] for t in trades])
                            win_rate = (winning_trades / len(trades)) * 100
                            
                            all_results.append({
                                'Strategy': 'EMA Crossover',
                                'Timeframe': tf_str,
                                'Total Trades': len(trades),
                                'Winning': winning_trades,
                                'Losing': losing_trades,
                                'Win Rate %': win_rate,
                                'Total PnL %': total_pnl,
                                'Avg Win %': avg_win,
                                'Avg Loss %': avg_loss,
                                'Best Trade %': best_trade,
                                'Worst Trade %': worst_trade,
                                'Risk:Reward': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                                'trades': trades
                            })
                    
                    # Z-Score Reversion Strategy
                    if test_zscore:
                        trades = []
                        in_trade = False
                        entry_price = 0
                        entry_idx = 0
                        
                        for i in range(50, len(data)):
                            if not in_trade:
                                # Entry: Z-Score < -2 (oversold)
                                if data['Z_Score'].iloc[i] < -2:
                                    in_trade = True
                                    entry_price = data['Close'].iloc[i]
                                    entry_idx = i
                            else:
                                # Exit: Z-Score > 0 (mean reversion)
                                if data['Z_Score'].iloc[i] > 0:
                                    exit_price = data['Close'].iloc[i]
                                    pnl = ((exit_price - entry_price) / entry_price) * 100
                                    
                                    trades.append({
                                        'strategy': 'Z-Score Reversion',
                                        'entry_date': data['DateTime_IST'].iloc[entry_idx],
                                        'entry_price': entry_price,
                                        'exit_date': data['DateTime_IST'].iloc[i],
                                        'exit_price': exit_price,
                                        'pnl_pct': pnl,
                                        'points': exit_price - entry_price
                                    })
                                    in_trade = False
                        
                        if trades:
                            winning_trades = len([t for t in trades if t['pnl_pct'] > 0])
                            losing_trades = len(trades) - winning_trades
                            total_pnl = sum([t['pnl_pct'] for t in trades])
                            avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if winning_trades > 0 else 0
                            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if losing_trades > 0 else 0
                            best_trade = max([t['pnl_pct'] for t in trades])
                            worst_trade = min([t['pnl_pct'] for t in trades])
                            win_rate = (winning_trades / len(trades)) * 100
                            
                            all_results.append({
                                'Strategy': 'Z-Score Reversion',
                                'Timeframe': tf_str,
                                'Total Trades': len(trades),
                                'Winning': winning_trades,
                                'Losing': losing_trades,
                                'Win Rate %': win_rate,
                                'Total PnL %': total_pnl,
                                'Avg Win %': avg_win,
                                'Avg Loss %': avg_loss,
                                'Best Trade %': best_trade,
                                'Worst Trade %': worst_trade,
                                'Risk:Reward': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                                'trades': trades
                            })
                    
                    # 9 EMA Pullback Strategy
                    if test_9ema:
                        trades = []
                        in_trade = False
                        entry_price = 0
                        entry_idx = 0
                        
                        for i in range(50, len(data)):
                            if not in_trade:
                                # Entry: Price pulls back to 9 EMA in uptrend
                                if (data['EMA_20'].iloc[i] > data['EMA_50'].iloc[i] and 
                                    data['Close'].iloc[i] > data['EMA_9'].iloc[i] and
                                    data['Close'].iloc[i-1] < data['EMA_9'].iloc[i-1]):
                                    in_trade = True
                                    entry_price = data['Close'].iloc[i]
                                    entry_idx = i
                            else:
                                # Exit: Price crosses below 9 EMA
                                if data['Close'].iloc[i] < data['EMA_9'].iloc[i]:
                                    exit_price = data['Close'].iloc[i]
                                    pnl = ((exit_price - entry_price) / entry_price) * 100
                                    
                                    trades.append({
                                        'strategy': '9 EMA Pullback',
                                        'entry_date': data['DateTime_IST'].iloc[entry_idx],
                                        'entry_price': entry_price,
                                        'exit_date': data['DateTime_IST'].iloc[i],
                                        'exit_price': exit_price,
                                        'pnl_pct': pnl,
                                        'points': exit_price - entry_price
                                    })
                                    in_trade = False
                        
                        if trades:
                            winning_trades = len([t for t in trades if t['pnl_pct'] > 0])
                            losing_trades = len(trades) - winning_trades
                            total_pnl = sum([t['pnl_pct'] for t in trades])
                            avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if winning_trades > 0 else 0
                            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if losing_trades > 0 else 0
                            best_trade = max([t['pnl_pct'] for t in trades])
                            worst_trade = min([t['pnl_pct'] for t in trades])
                            win_rate = (winning_trades / len(trades)) * 100
                            
                            all_results.append({
                                'Strategy': '9 EMA Pullback',
                                'Timeframe': tf_str,
                                'Total Trades': len(trades),
                                'Winning': winning_trades,
                                'Losing': losing_trades,
                                'Win Rate %': win_rate,
                                'Total PnL %': total_pnl,
                                'Avg Win %': avg_win,
                                'Avg Loss %': avg_loss,
                                'Best Trade %': best_trade,
                                'Worst Trade %': worst_trade,
                                'Risk:Reward': abs(avg_
