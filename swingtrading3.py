import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Swing Trading Screener", layout="wide")
st.title("📊 Swing Trading Screener — Multi-Confluence with Long & Short Trades")

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Fix possible BOM issue in first column!
    df.columns = [col.encode('utf-8').decode('utf-8-sig') for col in df.columns]
    df.columns = [col.strip() for col in df.columns]
    df.rename(columns={'oldName1': 'newName1', 'ï»¿Date': 'Date'}, inplace=True)
    st.write("Columns:", df.columns.tolist())
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    
    # ===== INDICATORS =====
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    # RSI
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
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
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

    # Volume Surge
    df['AvgVol20'] = df['Shares Traded'].rolling(20).mean()
    df['Vol_Surge'] = df['Shares Traded'] > (1.5 * df['AvgVol20'])

    # EMA200 Trend Filter
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # ===== BACKTEST =====
    trades = []
    position = None
    direction = None
    entry_price = sl = target = entry_date = reason = None

    for i in range(200, len(df)):
        row = df.iloc[i]
        confluences_long = []
        confluences_short = []

        # --- Long Setup ---
        if row['SMA20'] > row['SMA50']: confluences_long.append("SMA Bullish")
        if 30 < row['RSI'] < 70: confluences_long.append("RSI Healthy")
        if row['MACD'] > row['Signal']: confluences_long.append("MACD Bullish")
        if row['Close'] > row['BB_Mid']: confluences_long.append("BB Breakout")
        if row['Vol_Surge']: confluences_long.append("Vol Surge")

        # --- Short Setup ---
        if row['SMA20'] < row['SMA50']: confluences_short.append("SMA Bearish")
        if 30 < row['RSI'] < 70: confluences_short.append("RSI Healthy")
        if row['MACD'] < row['Signal']: confluences_short.append("MACD Bearish")
        if row['Close'] < row['BB_Mid']: confluences_short.append("BB Breakdown")
        if row['Vol_Surge']: confluences_short.append("Vol Surge")

        # Entry Long
        if position is None and row['Close'] > row['EMA200'] and len(confluences_long) >= 2:
            position = "Open"
            direction = "Long"
            entry_price = row['Close']
            sl = entry_price - 1.5 * row['ATR']
            target = entry_price + 2.5 * row['ATR']
            entry_date = row['Date']
            reason = ", ".join(confluences_long)

        # Entry Short
        elif position is None and row['Close'] < row['EMA200'] and len(confluences_short) >= 2:
            position = "Open"
            direction = "Short"
            entry_price = row['Close']
            sl = entry_price + 1.5 * row['ATR']
            target = entry_price - 2.5 * row['ATR']
            entry_date = row['Date']
            reason = ", ".join(confluences_short)

        # Manage Long Exit
        if position == "Open" and direction == "Long":
            if row['Close'] <= sl or row['Close'] >= target or row['SMA20'] < row['SMA50']:
                exit_price = row['Close']
                trades.append({
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Stop Loss": sl,
                    "Target": target,
                    "Exit Date": row['Date'],
                    "Exit Price": exit_price,
                    "Direction": direction,
                    "Reason": reason,
                    "P/L": exit_price - entry_price
                })
                position = None

        # Manage Short Exit
        if position == "Open" and direction == "Short":
            if row['Close'] >= sl or row['Close'] <= target or row['SMA20'] > row['SMA50']:
                exit_price = row['Close']
                trades.append({
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Stop Loss": sl,
                    "Target": target,
                    "Exit Date": row['Date'],
                    "Exit Price": exit_price,
                    "Direction": direction,
                    "Reason": reason,
                    "P/L": entry_price - exit_price
                })
                position = None

    trades_df = pd.DataFrame(trades)

    # ===== PERFORMANCE METRICS =====
    total_trades = len(trades_df)
    wins = trades_df[trades_df['P/L'] > 0].shape[0]
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_profit = trades_df['P/L'].sum()

    st.subheader("📈 Backtest Summary")
    st.write(f"Total Trades: {total_trades}")
    st.write(f"Win Rate: {win_rate:.2f}%")
    st.write(f"Total P/L (points): {total_profit:.2f}")

    st.subheader("📝 Detailed Trade Logs")
    st.dataframe(trades_df)

    # ===== Equity Curve =====
    if not trades_df.empty:
        trades_df['Cum_PnL'] = trades_df['P/L'].cumsum()
        fig, ax = plt.subplots()
        ax.plot(trades_df['Exit Date'], trades_df['Cum_PnL'], marker='o')
        ax.set_title("Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P/L")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ===== LIVE SIGNAL SECTION =====
    st.subheader("📢 Live Recommendation")
    last = df.iloc[-1]
    live_confluences_long = []
    live_confluences_short = []

    if last['SMA20'] > last['SMA50']: live_confluences_long.append("SMA Bullish")
    if 30 < last['RSI'] < 70: live_confluences_long.append("RSI Healthy")
    if last['MACD'] > last['Signal']: live_confluences_long.append("MACD Bullish")
    if last['Close'] > last['BB_Mid']: live_confluences_long.append("BB Breakout")
    if last['Vol_Surge']: live_confluences_long.append("Vol Surge")

    if last['SMA20'] < last['SMA50']: live_confluences_short.append("SMA Bearish")
    if 30 < last['RSI'] < 70: live_confluences_short.append("RSI Healthy")
    if last['MACD'] < last['Signal']: live_confluences_short.append("MACD Bearish")
    if last['Close'] < last['BB_Mid']: live_confluences_short.append("BB Breakdown")
    if last['Vol_Surge']: live_confluences_short.append("Vol Surge")

    if last['Close'] > last['EMA200'] and len(live_confluences_long) >= 2:
        st.success(f"📈 LONG at {last['Close']:.2f} | SL: {last['Close'] - 1.5*last['ATR']:.2f} | TP: {last['Close'] + 2.5*last['ATR']:.2f} | Reasons: {', '.join(live_confluences_long)}")
    elif last['Close'] < last['EMA200'] and len(live_confluences_short) >= 2:
        st.warning(f"📉 SHORT at {last['Close']:.2f} | SL: {last['Close'] + 1.5*last['ATR']:.2f} | TP: {last['Close'] - 2.5*last['ATR']:.2f} | Reasons: {', '.join(live_confluences_short)}")
    else:
        st.error("❌ No strong trade setup currently.")
