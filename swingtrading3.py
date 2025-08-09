import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from itertools import product

st.set_page_config(page_title="Swing Trading Screener — Multi-Source", layout="wide")
st.title("📊 Swing Trading Screener — Auto-Optimizer with Universal CSV Loader")

# --- Column mapping helper ---
def map_columns(df, col_map):
    col_lower = {c.lower(): c for c in df.columns}
    res = {}
    for target, choices in col_map.items():
        for choice in choices:
            if choice in df.columns:
                res[target] = choice
                break
            elif choice.lower() in col_lower:
                res[target] = col_lower[choice.lower()]
                break
    return res

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:

    # --- Read & clean BOM/whitespace ---
    df = pd.read_csv(uploaded_file)
    df.columns = [col.encode('utf-8').decode('utf-8-sig') for col in df.columns]
    df.columns = [col.strip() for col in df.columns]
    df.rename(columns={'ï»¿Date': 'Date'}, inplace=True)

    st.write("Columns detected:", df.columns.tolist())

    # --- Flexible mapping ---
    col_map = {
        'Date': ['Date'],
        'Open': ['Open', 'OPEN'],
        'High': ['High', 'HIGH'],
        'Low': ['Low', 'LOW'],
        'Close': ['Close', 'CLOSE', 'ltp', 'LTP', 'close'],
        'Volume': ['Shares Traded', 'VOLUME']
    }
    mapping = map_columns(df, col_map)

    # --- Check missing mapped columns ---
    missing = [k for k in col_map.keys() if k not in mapping]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        st.stop()

    # --- Build standardized frame ---
    df_std = pd.DataFrame()
    for k, v in mapping.items():
        df_std[k] = df[v]

    # --- Date handling ---
    df_std['Date'] = pd.to_datetime(df_std['Date'])
    df_std.sort_values('Date', inplace=True)

    # --- Numeric conversion fix (strip commas, force float) ---
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_std.columns:
            df_std[col] = pd.to_numeric(df_std[col].astype(str).str.replace(',', ''), errors='coerce')

    # ===== INDICATORS =====
    df_std['SMA20'] = df_std['Close'].rolling(20).mean()
    df_std['SMA50'] = df_std['Close'].rolling(50).mean()

    delta = df_std['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    df_std['RSI'] = 100 - (100 / (1 + pd.Series(gain).rolling(14).mean() /
                                  pd.Series(loss).rolling(14).mean()))

    df_std['H-L'] = df_std['High'] - df_std['Low']
    df_std['H-PC'] = abs(df_std['High'] - df_std['Close'].shift(1))
    df_std['L-PC'] = abs(df_std['Low'] - df_std['Close'].shift(1))
    df_std['TR'] = df_std[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_std['ATR'] = df_std['TR'].rolling(14).mean()

    EMA12 = df_std['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df_std['Close'].ewm(span=26, adjust=False).mean()
    df_std['MACD'] = EMA12 - EMA26
    df_std['Signal'] = df_std['MACD'].ewm(span=9, adjust=False).mean()

    df_std['BB_Mid'] = df_std['Close'].rolling(20).mean()
    df_std['AvgVol20'] = df_std['Volume'].rolling(20).mean()
    df_std['Vol_Surge'] = df_std['Volume'] > (1.5 * df_std['AvgVol20'])

    # ===== PARAMETER GRID =====
    atr_sl_choices = [1.0, 1.5, 2.0]
    atr_tp_choices = [2.0, 2.5, 3.0]
    ema_choices = [100, 150, 200]
    min_conf_choices = [1, 2, 3]
    trade_modes = ["Both", "Long Only", "Short Only"]
    param_grid = list(product(atr_sl_choices, atr_tp_choices, ema_choices,
                              min_conf_choices, trade_modes))

    # ===== OPTIMIZER =====
    best_stats, best_trades = None, None
    progress = st.progress(0)
    total_runs = len(param_grid)

    for idx, (atr_sl, atr_tp, ema_trend_period, min_conf, trade_mode) in enumerate(param_grid, start=1):
        df_std['EMA_TREND'] = df_std['Close'].ewm(span=ema_trend_period, adjust=False).mean()
        trades = []
        position = direction = None

        for i in range(ema_trend_period, len(df_std)):
            row = df_std.iloc[i]
            long_conf, short_conf = [], []

            if row['SMA20'] > row['SMA50']: long_conf.append("SMA Bullish")
            if 30 < row['RSI'] < 70: long_conf.append("RSI Healthy")
            if row['MACD'] > row['Signal']: long_conf.append("MACD Bullish")
            if row['Close'] > row['BB_Mid']: long_conf.append("BB Breakout")
            if row['Vol_Surge']: long_conf.append("Volume Surge")

            if row['SMA20'] < row['SMA50']: short_conf.append("SMA Bearish")
            if 30 < row['RSI'] < 70: short_conf.append("RSI Healthy")
            if row['MACD'] < row['Signal']: short_conf.append("MACD Bearish")
            if row['Close'] < row['BB_Mid']: short_conf.append("BB Breakdown")
            if row['Vol_Surge']: short_conf.append("Volume Surge")

            # entry rules
            if position is None and row['Close'] > row['EMA_TREND'] and len(long_conf) >= min_conf and trade_mode in ["Both", "Long Only"]:
                position, direction = "Open", "Long"
                entry_price = row['Close']
                sl = entry_price - atr_sl * row['ATR']
                target = entry_price + atr_tp * row['ATR']
                entry_date, reason = row['Date'], ", ".join(long_conf)
            elif position is None and row['Close'] < row['EMA_TREND'] and len(short_conf) >= min_conf and trade_mode in ["Both", "Short Only"]:
                position, direction = "Open", "Short"
                entry_price = row['Close']
                sl = entry_price + atr_sl * row['ATR']
                target = entry_price - atr_tp * row['ATR']
                entry_date, reason = row['Date'], ", ".join(short_conf)

            # exit rules
            if position == "Open" and direction == "Long":
                if row['Close'] <= sl or row['Close'] >= target or row['SMA20'] < row['SMA50']:
                    trades.append({"Entry Date": entry_date, "Entry Price": entry_price, "Stop Loss": sl,
                                   "Target": target, "Exit Date": row['Date'], "Exit Price": row['Close'],
                                   "Direction": direction, "Reason": reason, "P/L": row['Close'] - entry_price})
                    position = None
            elif position == "Open" and direction == "Short":
                if row['Close'] >= sl or row['Close'] <= target or row['SMA20'] > row['SMA50']:
                    trades.append({"Entry Date": entry_date, "Entry Price": entry_price, "Stop Loss": sl,
                                   "Target": target, "Exit Date": row['Date'], "Exit Price": row['Close'],
                                   "Direction": direction, "Reason": reason, "P/L": entry_price - row['Close']})
                    position = None

        trades_df = pd.DataFrame(trades)
        total_profit = trades_df['P/L'].sum() if not trades_df.empty else 0
        win_rate = (trades_df['P/L'] > 0).mean() * 100 if not trades_df.empty else 0

        if best_stats is None or total_profit > best_stats['total_profit']:
            best_stats = dict(atr_sl=atr_sl, atr_tp=atr_tp, ema_trend_period=ema_trend_period,
                              min_conf=min_conf, trade_mode=trade_mode, total_trades=len(trades_df),
                              win_rate=win_rate, total_profit=total_profit)
            best_trades = trades_df.copy()

        progress.progress(idx / total_runs)

    progress.empty()

    # ===== SHOW RESULTS =====
    if best_stats:
        st.header("🚀 Best Parameters Found")
        st.success(f"""
        ATR SL: {best_stats['atr_sl']} | ATR TP: {best_stats['atr_tp']}
        EMA Trend: {best_stats['ema_trend_period']} | Min Confluences: {best_stats['min_conf']}
        Trade Mode: {best_stats['trade_mode']}
        Trades: {best_stats['total_trades']} | Win Rate: {best_stats['win_rate']:.2f}%
        Total P/L: {best_stats['total_profit']:.2f} points
        """)
        st.subheader("📜 Trade Log")
        st.dataframe(best_trades)

        if not best_trades.empty:
            best_trades['Cum_PnL'] = best_trades['P/L'].cumsum()
            fig, ax = plt.subplots()
            ax.plot(best_trades['Exit Date'], best_trades['Cum_PnL'], marker='o')
            ax.set_title("Equity Curve — Best Params")
            ax.set_xlabel("Date"); ax.set_ylabel("Cumulative P/L")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # ===== LIVE RECOMMENDATION =====
        st.subheader("📢 Live Recommendation (Best Parameters)")
        df_std['EMA_TREND'] = df_std['Close'].ewm(span=best_stats['ema_trend_period'], adjust=False).mean()
        last = df_std.iloc[-1]
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
            st.error("❌ No strong trade setup currently (best params).")
    else:
        st.error("No valid parameter combination produced trades/profit on this data.")
