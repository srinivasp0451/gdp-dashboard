import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Swing Trading Screener — Optimizer", layout="wide")
st.title("📊 Swing Trading Screener — Interactive Optimizer")

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # --- FIX: Clean BOM & whitespace ---
    df = pd.read_csv(uploaded_file)
    df.columns = [col.encode('utf-8').decode('utf-8-sig') for col in df.columns]
    df.columns = [col.strip() for col in df.columns]
    df.rename(columns={'ï»¿Date': 'Date'}, inplace=True)

    st.write("Columns detected:", df.columns.tolist())
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # ===== SIDEBAR: PARAMETERS =====
    st.sidebar.header("Strategy Parameters")
    atr_sl = st.sidebar.slider("ATR StopLoss multiplier", 1.0, 3.0, 1.5, 0.1)
    atr_tp = st.sidebar.slider("ATR Target multiplier", 1.0, 4.0, 2.5, 0.1)
    ema_trend_period = st.sidebar.slider("EMA Trend Filter Period", 50, 300, 200, 10)
    min_conf = st.sidebar.slider("Min Confluences", 1, 5, 2, 1)
    trade_mode = st.sidebar.selectbox("Trade Direction", ["Both", "Long Only", "Short Only"])

    # ===== INDICATORS =====
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['Close'].rolling(20).std())
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['Close'].rolling(20).std())
    df['AvgVol20'] = df['Shares Traded'].rolling(20).mean()
    df['Vol_Surge'] = df['Shares Traded'] > (1.5 * df['AvgVol20'])
    df['EMA_TREND'] = df['Close'].ewm(span=ema_trend_period, adjust=False).mean()

    # ===== STRATEGY =========================================
    trades = []
    position = None
    direction = None

    for i in range(ema_trend_period, len(df)):
        row = df.iloc[i]
        confluences_long = []
        confluences_short = []

        # Long confluences
        if row['SMA20'] > row['SMA50']: confluences_long.append("SMA Bullish")
        if 30 < row['RSI'] < 70: confluences_long.append("RSI Healthy")
        if row['MACD'] > row['Signal']: confluences_long.append("MACD Bullish")
        if row['Close'] > row['BB_Mid']: confluences_long.append("BB Breakout")
        if row['Vol_Surge']: confluences_long.append("Volume Surge")

        # Short confluences
        if row['SMA20'] < row['SMA50']: confluences_short.append("SMA Bearish")
        if 30 < row['RSI'] < 70: confluences_short.append("RSI Healthy")
        if row['MACD'] < row['Signal']: confluences_short.append("MACD Bearish")
        if row['Close'] < row['BB_Mid']: confluences_short.append("BB Breakdown")
        if row['Vol_Surge']: confluences_short.append("Volume Surge")

        # Long entry
        if (position is None
            and row['Close'] > row['EMA_TREND'] 
            and len(confluences_long) >= min_conf 
            and (trade_mode in ["Both", "Long Only"])):
            position = "Open"
            direction = "Long"
            entry_price = row['Close']
            sl = entry_price - atr_sl * row['ATR']
            target = entry_price + atr_tp * row['ATR']
            entry_date = row['Date']
            reason = ", ".join(confluences_long)

        # Short entry
        elif (position is None
              and row['Close'] < row['EMA_TREND'] 
              and len(confluences_short) >= min_conf 
              and (trade_mode in ["Both", "Short Only"])):
            position = "Open"
            direction = "Short"
            entry_price = row['Close']
            sl = entry_price + atr_sl * row['ATR']
            target = entry_price - atr_tp * row['ATR']
            entry_date = row['Date']
            reason = ", ".join(confluences_short)

        # Long exit
        if position == "Open" and direction == "Long":
            if row['Close'] <= sl or row['Close'] >= target or row['SMA20'] < row['SMA50']:
                trades.append({
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Stop Loss": sl,
                    "Target": target,
                    "Exit Date": row['Date'],
                    "Exit Price": row['Close'],
                    "Direction": direction,
                    "Reason": reason,
                    "P/L": row['Close'] - entry_price,
                })
                position = None

        # Short exit
        if position == "Open" and direction == "Short":
            if row['Close'] >= sl or row['Close'] <= target or row['SMA20'] > row['SMA50']:
                trades.append({
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Stop Loss": sl,
                    "Target": target,
                    "Exit Date": row['Date'],
                    "Exit Price": row['Close'],
                    "Direction": direction,
                    "Reason": reason,
                    "P/L": entry_price - row['Close'],
                })
                position = None

    trades_df = pd.DataFrame(trades)

    # ===== PERFORMANCE =====
    if not trades_df.empty:
        total_trades = len(trades_df)
        wins = trades_df[trades_df['P/L'] > 0].shape[0]
        win_rate = wins / total_trades * 100
        total_profit = trades_df['P/L'].sum()
    else:
        total_trades = wins = total_profit = win_rate = 0

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

    # ===== LIVE TRADE SIGNAL =====
    st.subheader("📢 Live Recommendation")
    last = df.iloc[-1]
    live_long = []
    live_short = []
    if last['SMA20'] > last['SMA50']: live_long.append("SMA Bullish")
    if 30 < last['RSI'] < 70: live_long.append("RSI Healthy")
    if last['MACD'] > last['Signal']: live_long.append("MACD Bullish")
    if last['Close'] > last['BB_Mid']: live_long.append("BB Breakout")
    if last['Vol_Surge']: live_long.append("Volume Surge")
    if last['SMA20'] < last['SMA50']: live_short.append("SMA Bearish")
    if 30 < last['RSI'] < 70: live_short.append("RSI Healthy")
    if last['MACD'] < last['Signal']: live_short.append("MACD Bearish")
    if last['Close'] < last['BB_Mid']: live_short.append("BB Breakdown")
    if last['Vol_Surge']: live_short.append("Volume Surge")

    if last['Close'] > last['EMA_TREND'] and len(live_long) >= min_conf and (trade_mode in ["Both","Long Only"]):
        st.success(f"📈 LONG at {last['Close']:.2f} | SL: {last['Close'] - atr_sl*last['ATR']:.2f} | TP: {last['Close'] + atr_tp*last['ATR']:.2f} | Reasons: {', '.join(live_long)}")
    elif last['Close'] < last['EMA_TREND'] and len(live_short) >= min_conf and (trade_mode in ["Both","Short Only"]):
        st.warning(f"📉 SHORT at {last['Close']:.2f} | SL: {last['Close'] + atr_sl*last['ATR']:.2f} | TP: {last['Close'] - atr_tp*last['ATR']:.2f} | Reasons: {', '.join(live_short)}")
    else:
        st.error("❌ No strong trade setup currently.")

    # ===== Instructions =====
    st.info("↖️ Change parameters in sidebar to instantly optimize trade count, win rate, P/L, and live signals.")
