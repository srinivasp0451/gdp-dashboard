import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz

st.set_page_config(layout="wide", page_title="Advanced Swing Trading Platform with Patterns")

st.title("🔮 Advanced Swing Trading with Chart Patterns and Psychology")

# ------------------------
# File Upload & Column Mapping
# ------------------------
uploaded_file = st.file_uploader("Upload OHLCV Data (CSV/Excel)", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Column mapping
    col_map = {}
    for col in df.columns:
        c = col.lower()
        if 'date' in c:
            col_map['date'] = col
        elif 'open' in c:
            col_map['open'] = col
        elif 'high' in c:
            col_map['high'] = col
        elif 'low' in c:
            col_map['low'] = col
        elif 'close' in c:
            col_map['close'] = col
        elif 'volume' in c or 'shares' in c or 'traded' in c:
            col_map['volume'] = col

    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(rc in col_map for rc in required_cols):
        st.error(f"Uploaded file missing required columns. Detected columns: {list(col_map.keys())}")
        st.stop()

    # Rename columns
    df = df.rename(columns={v:k for k,v in col_map.items()})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    df = df.sort_values('date').reset_index(drop=True)

    st.subheader("✅ Data Overview")
    st.write(df.head(5))
    st.write(df.tail(5))
    st.write(f"Start Date: {df['date'].min()}, End Date: {df['date'].max()}")
    st.write(f"Max Price: {df['high'].max()}, Min Price: {df['low'].min()}")

    st.subheader("Raw Price Chart")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df['date'], df['close'], label='Close Price')
    ax.set_title("Price Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # ------------------------
    # End date selection for testing past data
    # ------------------------
    default_end = df['date'].max()
    end_date = st.date_input("Select End Date for Backtesting/Recommendation", value=default_end)
    end_date = pd.Timestamp(end_date).tz_localize('Asia/Kolkata')
    df_bt = df[df['date'] <= end_date].copy()

    # ------------------------
    # Exploratory Data Analysis
    # ------------------------
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    df_bt['returns'] = df_bt['close'].pct_change()
    st.write(df_bt.describe())

    st.write("### Returns Heatmap (Year vs Month)")
    df_bt['year'] = df_bt['date'].dt.year
    df_bt['month'] = df_bt['date'].dt.month
    pivot = df_bt.pivot_table(index='year', columns='month', values='returns', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12,4))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap='RdYlGn', ax=ax)
    st.pyplot(fig)

    # Human-readable summary of data
    summary = f"The uploaded stock has data from {df_bt['date'].min().date()} to {df_bt['date'].max().date()}. Prices ranged between {df_bt['low'].min()} and {df_bt['high'].max()}. Average returns are {df_bt['returns'].mean():.2%}. Opportunities may exist at support/resistance zones, breakout patterns, and high liquidity levels for swing trades."
    st.info(summary)

    # ------------------------
    # Strategy Parameters
    # ------------------------
    st.subheader("⚙️ Strategy Settings")
    side = st.selectbox("Select Side", options=['Long', 'Short', 'Both'])
    search_type = st.selectbox("Optimization Search Type", options=['Random Search','Grid Search'])
    desired_accuracy = st.slider("Desired Accuracy %", 50, 100, 80)
    min_points = st.number_input("Minimum Points for Strategy", min_value=1, value=50)

    # ------------------------
    # Progress Bar
    # ------------------------
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # ------------------------
    # Advanced Price Action & Pattern Detection
    # ------------------------
    def detect_support_resistance(df, window=20):
        df['sup'] = df['low'].rolling(window).min()
        df['res'] = df['high'].rolling(window).max()
        return df

    def detect_chart_patterns(df):
        patterns = []
        for i in range(2,len(df)-2):
            # M/W patterns
            if df['high'][i-1] < df['high'][i] > df['high'][i+1] and df['low'][i-1] > df['low'][i] < df['low'][i+1]:
                patterns.append((df['date'][i], 'M/W Pattern'))
            # Head & Shoulders detection
            if i>3:
                if (df['high'][i-3]<df['high'][i-2]>df['high'][i-1]<df['high'][i] and df['low'][i-3]>df['low'][i-2]<df['low'][i-1]>df['low'][i]):
                    patterns.append((df['date'][i-2],'Head & Shoulders'))
            # Cup & Handle detection (simplified)
            if i>5:
                cup_low = df['low'][i-5:i+1].min()
                cup_high = df['high'][i-5:i+1].max()
                if df['close'][i] > cup_high*0.95 and df['low'][i-5] > cup_low:
                    patterns.append((df['date'][i],'Cup & Handle'))
            # Triangles, Wedges, Flags can be added similarly
        return patterns

    def detect_trap_zones(df):
        trap = []
        for i in range(2,len(df)-2):
            if df['close'][i] < df['low'][i-1] and df['close'][i+1] > df['high'][i]:
                trap.append((df['date'][i],'Bull Trap Zone'))
            if df['close'][i] > df['high'][i-1] and df['close'][i+1] < df['low'][i]:
                trap.append((df['date'][i],'Bear Trap Zone'))
        return trap

    def compute_trade_signals(df, side='Both'):
        df['signal'] = 0
        df['target'] = np.nan
        df['sl'] = np.nan
        df['reason'] = ''
        for i in range(1,len(df)):
            # Long breakout
            if side in ['Long','Both'] and df['close'][i] > df['high'][i-1]:
                df.at[i,'signal'] = 1
                df.at[i,'target'] = df['close'][i] + (df['high'][i-1]-df['low'][i-1])
                df.at[i,'sl'] = df['low'][i-1]
                df.at[i,'reason'] = f"Breakout above resistance {df['high'][i-1]}"
            # Short breakout
            elif side in ['Short','Both'] and df['close'][i] < df['low'][i-1]:
                df.at[i,'signal'] = -1
                df.at[i,'target'] = df['close'][i] - (df['high'][i-1]-df['low'][i-1])
                df.at[i,'sl'] = df['high'][i-1]
                df.at[i,'reason'] = f"Breakdown below support {df['low'][i-1]}"
        return df

    # ------------------------
    # Optimization
    # ------------------------
    def optimize_strategy(df, search_type='Random Search', desired_accuracy=80, min_points=50):
        best_params = {'window':20}
        if search_type=='Random Search':
            windows = np.random.randint(5,50,10)
        else:
            windows = range(5,50,5)
        best_score=0
        for w in windows:
            df_test = detect_support_resistance(df.copy(), window=w)
            df_test = compute_trade_signals(df_test, side=side)
            trades = df_test[df_test['signal']!=0]
            if len(trades)==0:
                continue
            score = (trades['signal']*np.sign(trades['close'].diff().shift(-1))).mean()*100
            if score>best_score:
                best_score=score
                best_params['window']=w
        return best_params,best_score

    progress_text.text("Running Strategy Optimization...")
    progress_bar.progress(20)
    best_params,best_score = optimize_strategy(df_bt, search_type, desired_accuracy, min_points)
    progress_bar.progress(50)
    st.success(f"Best Strategy Parameters: {best_params}, Expected Accuracy: {best_score:.2f}%")

    # ------------------------
    # Backtest
    # ------------------------
    progress_text.text("Running Backtest...")
    df_bt = detect_support_resistance(df_bt, window=best_params['window'])
    df_bt = compute_trade_signals(df_bt, side=side)
    patterns = detect_chart_patterns(df_bt)
    traps = detect_trap_zones(df_bt)
    progress_bar.progress(80)

    trades = df_bt[df_bt['signal']!=0].copy()
    trades['pnl'] = np.where(trades['signal']==1,trades['target']-trades['close'],trades['close']-trades['target'])
    total_trades = len(trades)
    positive_trades = (trades['pnl']>0).sum()
    negative_trades = (trades['pnl']<=0).sum()
    accuracy = positive_trades/total_trades*100 if total_trades>0 else 0
    total_points = trades['pnl'].sum()

    progress_bar.progress(100)
    progress_text.text("Backtest Complete!")

    st.subheader("📈 Backtest Results")
    st.write(trades[['date','signal','close','target','sl','reason','pnl']])
    st.write(f"Total Trades: {total_trades}, Positive Trades: {positive_trades}, Negative Trades: {negative_trades}")
    st.write(f"Accuracy: {accuracy:.2f}%, Total Points: {total_points:.2f}")

    st.write("### Detected Patterns in Backtest")
    st.write(patterns)
    st.write("### Detected Trap Zones")
    st.write(traps)

    # Human readable summary
    summary_bt = f"Backtesting shows {total_trades} trades with {accuracy:.2f}% accuracy and total points {total_points:.2f}. Detected patterns: {len(patterns)}. Trap zones detected: {len(traps)}. Long trades were triggered on breakouts above resistance and short trades on breakdowns below support. SLs were set at swing levels. For live recommendation, use the same strategy on the latest candle close."
    st.info(summary_bt)

    # ------------------------
    # Live Recommendation
    # ------------------------
    st.subheader("🚀 Live Recommendation")
    live_df = compute_trade_signals(df_bt.tail(2), side=side).iloc[-1:]
    if live_df['signal'].values[0]!=0:
        signal_type = 'Long' if live_df['signal'].values[0]==1 else 'Short'
        live_summary = f"Live Recommendation: {signal_type} Entry at {live_df['close'].values[0]}, Target {live_df['target'].values[0]}, SL {live_df['sl'].values[0]}. Reason: {live_df['reason'].values[0]}"
        st.success(live_summary)
    else:
        st.info("No live trade signal on the last candle.")

    st.balloons()
