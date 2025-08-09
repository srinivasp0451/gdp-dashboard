import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from itertools import product

st.set_page_config(page_title="Swing Trading Screener — Auto Optimizer", layout="wide")
st.title("📊 Swing Trading Screener — Auto Parameter Optimization (Any CSV Format)")

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Clean BOM & whitespace ---
    df.columns = [col.encode('utf-8').decode('utf-8-sig') for col in df.columns]
    df.columns = [col.strip() for col in df.columns]

    # --- Standardise to lowercase for mapping ---
    df.columns = [col.lower() for col in df.columns]

    # --- Map known variations to standard names ---
    col_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'ltp': 'Close',                 # use LTP if close is absent
        'volume': 'Shares Traded',
        'vwap': 'VWAP',
        'no of trades': 'Trades',
        'shares traded': 'Shares Traded',
        'date': 'Date'
    }
    for src, dest in col_map.items():
        if src in df.columns and dest.lower() not in df.columns:
            df.rename(columns={src: dest}, inplace=True)

    # --- Final expected columns check ---
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Shares Traded']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns after mapping: {missing}")
        st.stop()

    # --- Make sure Date is datetime ---
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # ===== PARAMETER GRID =====
    atr_sl_choices = [1.0, 1.5, 2.0]
    atr_tp_choices = [2.0, 2.5, 3.0]
    ema_choices = [100, 150, 200]
    min_conf_choices = [1, 2, 3]
    trade_modes = ["Both", "Long Only", "Short Only"]
    param_grid = list(product(atr_sl_choices, atr_tp_choices, ema_choices, min_conf_choices, trade_modes))

    # ===== INDICATORS (calculated once) =====
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    df['RSI'] = 100 - (100 / (1 + pd.Series(gain).rolling(14).mean() / pd.Series(loss).rolling(14).mean()))

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
    df['AvgVol20'] = df['Shares Traded'].rolling(20).mean()
    df['Vol_Surge'] = df['Shares Traded'] > (1.5 * df['AvgVol20'])

    # ===== AUTO OPTIMIZER =====
    best_stats = None
    best_trades = None
    total_runs = len(param_grid)
    run_count = 0
    progress = st.progress(0)

    for (atr_sl, atr_tp, ema_trend_period, min_conf, trade_mode) in param_grid:
        df['EMA_TREND'] = df['Close'].ewm(span=ema_trend_period, adjust=False).mean()

        trades = []
        position = None
        direction = None

        for i in range(ema_trend_period, len(df)):
            row = df.iloc[i]
            confluences_long, confluences_short = [], []

            # Long signals
            if row['SMA20'] > row['SMA50']: confluences_long.append("SMA Bullish")
            if 30 < row['RSI'] < 70: confluences_long.append("RSI Healthy")
            if row['MACD'] > row['Signal']: confluences_long.append("MACD Bullish")
            if row['Close'] > row['BB_Mid']: confluences_long.append("BB Breakout")
            if row['Vol_Surge']: confluences_long.append("Volume Surge")

            # Short signals
            if row['SMA20'] < row['SMA50']: confluences_short.append("SMA Bearish")
            if 30 < row['RSI'] < 70: confluences_short.append("RSI Healthy")
            if row['MACD'] < row['Signal']: confluences_short.append("MACD Bearish")
            if row['Close'] < row['BB_Mid']: confluences_short.append("BB Breakdown")
            if row['Vol_Surge']: confluences_short.append("Volume Surge")

            # Buy entry
            if (position is None and row['Close'] > row['EMA_TREND'] and
                len(confluences_long) >= min_conf and (trade_mode in ["Both", "Long Only"])):
                position = "Open"; direction = "Long"
                entry_price, sl, target = row['Close'], row['Close'] - atr_sl*row['ATR'], row['Close'] + atr_tp*row['ATR']
                entry_date, reason = row['Date'], ", ".join(confluences_long)

            # Sell entry
            elif (position is None and row['Close'] < row['EMA_TREND'] and
                  len(confluences_short) >= min_conf and (trade_mode in ["Both", "Short Only"])):
                position = "Open"; direction = "Short"
                entry_price, sl, target = row['Close'], row['Close'] + atr_sl*row['ATR'], row['Close'] - atr_tp*row['ATR']
                entry_date, reason = row['Date'], ", ".join(confluences_short)

            # Exit Long
            if position == "Open" and direction == "Long":
                if row['Close'] <= sl or row['Close'] >= target or row['SMA20'] < row['SMA50']:
                    trades.append({
                        "Entry Date": entry_date, "Entry Price": entry_price,
                        "Stop Loss": sl, "Target": target,
                        "Exit Date": row['Date'], "Exit Price": row['Close'],
                        "Direction": direction, "Reason": reason,
                        "P/L": row['Close'] - entry_price
                    })
                    position = None

            # Exit Short
            if position == "Open" and direction == "Short":
                if row['Close'] >= sl or row['Close'] <= target or row['SMA20'] > row['SMA50']:
                    trades.append({
                        "Entry Date": entry_date, "Entry Price": entry_price,
                        "Stop Loss": sl, "Target": target,
                        "Exit Date": row['Date'], "Exit Price": row['Close'],
                        "Direction": direction, "Reason": reason,
                        "P/L": entry_price - row['Close']
                    })
                    position = None

        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        total_profit = trades_df['P/L'].sum() if not trades_df.empty else 0
        win_rate = (trades_df['P/L'] > 0).mean()*100 if total_trades > 0 else 0

        run_count += 1
        progress.progress(run_count/total_runs)

        if (best_stats is None) or (total_profit > best_stats['total_profit']):
            best_stats = dict(
                atr_sl=atr_sl, atr_tp=atr_tp, ema_trend_period=ema_trend_period,
                min_conf=min_conf, trade_mode=trade_mode,
                total_trades=total_trades, win_rate=win_rate, total_profit=total_profit
            )
            best_trades = trades_df.copy()

    progress.empty()

    # ===== DISPLAY BEST RESULT =====
    if best_stats:
        st.header("🚀 Best Parameters Found (Auto-Optimized)")
        st.success(f"""
        ATR SL: {best_stats['atr_sl']} | ATR TP: {best_stats['atr_tp']}  
        EMA Trend Period: {best_stats['ema_trend_period']} | Min Confluences: {best_stats['min_conf']}  
        Trade Mode: {best_stats['trade_mode']}  
        Total Trades: {best_stats['total_trades']}  
        Win Rate: {best_stats['win_rate']:.2f}%  
        Total P/L: {best_stats['total_profit']:.2f} points
        """)

        st.subheader("📝 Trade Log (Best Parameters)")
        st.dataframe(best_trades)

        if not best_trades.empty:
            best_trades['Cum_PnL'] = best_trades['P/L'].cumsum()
            fig, ax = plt.subplots()
            ax.plot(best_trades['Exit Date'], best_trades['Cum_PnL'], marker='o')
            ax.set_title("Equity Curve — Best Parameters")
            ax.set_xlabel("Date"); ax.set_ylabel("Cumulative P/L")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # ===== LIVE RECOMMENDATION =====
        st.subheader("📢 Live Recommendation (Best Parameters)")
        df['EMA_TREND'] = df['Close'].ewm(span=best_stats['ema_trend_period'], adjust=False).mean()
        last = df.iloc[-1]
        live_long, live_short = [], []

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

        if last['Close'] > last['EMA_TREND'] and len(live_long) >= best_stats['min_conf'] and (best_stats['trade_mode'] in ["Both","Long Only"]):
            st.success(f"📈 LONG at {last['Close']:.2f} | SL: {last['Close'] - best_stats['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] + best_stats['atr_tp']*last['ATR']:.2f} | Reasons: {', '.join(live_long)}")
        elif last['Close'] < last['EMA_TREND'] and len(live_short) >= best_stats['min_conf'] and (best_stats['trade_mode'] in ["Both","Short Only"]):
            st.warning(f"📉 SHORT at {last['Close']:.2f} | SL: {last['Close'] + best_stats['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] - best_stats['atr_tp']*last['ATR']:.2f} | Reasons: {', '.join(live_short)}")
        else:
            st.error("❌ No strong trade setup currently for best parameters.")
    else:
        st.error("No profitable parameter set found.")
