import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Swing Trading Screener", layout="wide")

st.title("📊 Swing Trading Strategy Screener with Multi-Confluence Analysis")

# ========== FILE UPLOAD ==========
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # ========== TECHNICAL INDICATORS ==========
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # MACD
    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['Close'].rolling(20).std())
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['Close'].rolling(20).std())

    # Volume surge
    df['AvgVol20'] = df['Shares Traded'].rolling(20).mean()
    df['Vol_Surge'] = df['Shares Traded'] > (1.5 * df['AvgVol20'])

    # ========== SIGNAL GENERATION ==========
    trades = []
    position = None
    entry_price = sl = target = entry_date = reason = None

    for i in range(50, len(df)):
        row = df.iloc[i]
        date_str = row['Date'].strftime('%Y-%m-%d')
        confluences = []

        # Indicators for Buy
        if row['SMA20'] > row['SMA50']: confluences.append("SMA Bullish")
        if row['RSI'] > 30 and row['RSI'] < 70: confluences.append("RSI Healthy")
        if row['MACD'] > row['Signal']: confluences.append("MACD Bullish")
        if row['Close'] > row['BB_Mid']: confluences.append("BB Breakout")
        if row['Vol_Surge']: confluences.append("Volume Surge")

        # Entry Condition: 3 or more bullish confluences → Long
        if position is None and len(confluences) >= 3:
            position = "Long"
            entry_price = row['Close']
            sl = entry_price - row['ATR']
            target = entry_price + 2 * row['ATR']
            entry_date = row['Date']
            reason = ", ".join(confluences)
        
        # Exit Condition for Long
        if position == "Long":
            if row['Close'] <= sl or row['Close'] >= target or row['SMA20'] < row['SMA50']:
                exit_price = row['Close']
                trades.append({
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Stop Loss": sl,
                    "Target": target,
                    "Exit Date": row['Date'],
                    "Exit Price": exit_price,
                    "Reason": reason,
                    "P/L": exit_price - entry_price
                })
                position = None

    trades_df = pd.DataFrame(trades)

    # ========== PERFORMANCE LOGS ==========
    total_trades = len(trades_df)
    wins = trades_df[trades_df['P/L'] > 0].shape[0]
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_profit = trades_df['P/L'].sum()

    st.subheader("📈 Backtesting Summary")
    st.write(f"Total Trades: {total_trades}")
    st.write(f"Win Rate: {win_rate:.2f}%")
    st.write(f"Total P/L (points): {total_profit:.2f}")

    st.subheader("📝 Detailed Trade Logs")
    st.dataframe(trades_df)

    # ========== LIVE RECOMMENDATION ==========
    st.subheader("📢 Live Recommendation")
    last = df.iloc[-1]
    live_confluences = []
    if last['SMA20'] > last['SMA50']: live_confluences.append("SMA Bullish")
    if last['RSI'] > 30 and last['RSI'] < 70: live_confluences.append("RSI Healthy")
    if last['MACD'] > last['Signal']: live_confluences.append("MACD Bullish")
    if last['Close'] > last['BB_Mid']: live_confluences.append("BB Breakout")
    if last['Vol_Surge']: live_confluences.append("Volume Surge")

    if len(live_confluences) >= 3:
        st.success(f"📈 LONG Recommended at {last['Close']} | SL: {last['Close'] - last['ATR']:.2f} | TP: {last['Close'] + 2*last['ATR']:.2f}\nReasons: {', '.join(live_confluences)}")
    elif len(live_confluences) <= 1:
        st.error("❌ No strong trade signal.")
    else:
        st.warning(f"⚠️ Wait for confirmation. Currently {len(live_confluences)} bullish confluences.")
