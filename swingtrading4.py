import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import product

st.set_page_config(page_title="Swing Trading Screener — Auto Optimizer with yfinance", layout="wide")
st.title("📊 Swing Trading Screener — Auto Parameter Optimization (Live Data from yfinance)")

# ===== INPUTS =====
ticker = st.sidebar.text_input("Enter Ticker Symbol", "^NSEBANK")  # ^NSEBANK for Nifty Bank
period = st.sidebar.selectbox("Select Period", ["6mo", "1y", "2y", "5y", "max"], index=1)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk"], index=0)

if st.sidebar.button("Fetch & Run Optimization"):
    st.write(f"📥 Downloading {ticker} data from yfinance...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    if df.empty:
        st.error("No data retrieved. Check ticker or period.")
    else:
        # ===== Fixing yfinance multiindex issue =====
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
        df.reset_index(inplace=True)

        # Map columns to expected names
        if 'Date' not in df.columns:
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        if 'Volume' in df.columns and 'Shares Traded' not in df.columns:
            df.rename(columns={'Volume': 'Shares Traded'}, inplace=True)

        # Keep only required columns
        col_map = {
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Shares Traded': 'Shares Traded'
        }
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Shares Traded']].copy()
        df.sort_values("Date", inplace=True)

        # ===== Parameter Ranges =====
        atr_sl_choices = [1.0, 1.5, 2.0]
        atr_tp_choices = [2.0, 2.5, 3.0]
        ema_choices = [100, 150, 200]
        min_conf_choices = [1, 2, 3]
        trade_modes = ["Both", "Long Only", "Short Only"]

        param_grid = list(product(atr_sl_choices, atr_tp_choices, ema_choices, min_conf_choices, trade_modes))

        # ===== Indicators =====
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

        # ===== Auto Optimization =====
        best_stats, best_params, best_trades = None, None, None

        total_runs = len(param_grid)
        run_count = 0
        progress = st.progress(0)

        for (atr_sl, atr_tp, ema_trend_period, min_conf, trade_mode) in param_grid:
            df['EMA_TREND'] = df['Close'].ewm(span=ema_trend_period, adjust=False).mean()
            trades = []
            position, direction = None, None

            for i in range(ema_trend_period, len(df)):
                row = df.iloc[i]
                long_conf, short_conf = [], []

                # Long signals
                if row['SMA20'] > row['SMA50']: long_conf.append("SMA Bullish")
                if 30 < row['RSI'] < 70: long_conf.append("RSI Healthy")
                if row['MACD'] > row['Signal']: long_conf.append("MACD Bullish")
                if row['Close'] > row['BB_Mid']: long_conf.append("BB Breakout")
                if row['Vol_Surge']: long_conf.append("Volume Surge")

                # Short signals
                if row['SMA20'] < row['SMA50']: short_conf.append("SMA Bearish")
                if 30 < row['RSI'] < 70: short_conf.append("RSI Healthy")
                if row['MACD'] < row['Signal']: short_conf.append("MACD Bearish")
                if row['Close'] < row['BB_Mid']: short_conf.append("BB Breakdown")
                if row['Vol_Surge']: short_conf.append("Volume Surge")

                # Long entry
                if (position is None and row['Close'] > row['EMA_TREND'] and len(long_conf) >= min_conf
                    and trade_mode in ["Both", "Long Only"]):
                    position, direction = "Open", "Long"
                    entry_price = row['Close']
                    sl = entry_price - atr_sl * row['ATR']
                    target = entry_price + atr_tp * row['ATR']
                    entry_date, reason = row['Date'], ", ".join(long_conf)

                # Short entry
                elif (position is None and row['Close'] < row['EMA_TREND'] and len(short_conf) >= min_conf
                      and trade_mode in ["Both", "Short Only"]):
                    position, direction = "Open", "Short"
                    entry_price = row['Close']
                    sl = entry_price + atr_sl * row['ATR']
                    target = entry_price - atr_tp * row['ATR']
                    entry_date, reason = row['Date'], ", ".join(short_conf)

                # Long exit
                if position == "Open" and direction == "Long":
                    if (row['Close'] <= sl or row['Close'] >= target or row['SMA20'] < row['SMA50']):
                        trades.append({"Entry Date": entry_date, "Entry Price": entry_price,
                                       "Stop Loss": sl, "Target": target,
                                       "Exit Date": row['Date'], "Exit Price": row['Close'],
                                       "Direction": direction, "Reason": reason,
                                       "P/L": row['Close'] - entry_price})
                        position = None

                # Short exit
                if position == "Open" and direction == "Short":
                    if (row['Close'] >= sl or row['Close'] <= target or row['SMA20'] > row['SMA50']):
                        trades.append({"Entry Date": entry_date, "Entry Price": entry_price,
                                       "Stop Loss": sl, "Target": target,
                                       "Exit Date": row['Date'], "Exit Price": row['Close'],
                                       "Direction": direction, "Reason": reason,
                                       "P/L": entry_price - row['Close']})
                        position = None

            trades_df = pd.DataFrame(trades)
            total_profit = trades_df['P/L'].sum() if not trades_df.empty else 0
            win_rate = (trades_df[trades_df['P/L'] > 0].shape[0] / len(trades_df) * 100) if not trades_df.empty else 0

            # Best by Total Profit
            if best_stats is None or total_profit > best_stats['total_profit']:
                best_stats = {
                    'total_profit': total_profit,
                    'trade_count': len(trades_df),
                    'win_rate': win_rate,
                    'atr_sl': atr_sl,
                    'atr_tp': atr_tp,
                    'ema_trend': ema_trend_period,
                    'min_conf': min_conf,
                    'trade_mode': trade_mode
                }
                best_params, best_trades = (atr_sl, atr_tp, ema_trend_period, min_conf, trade_mode), trades_df

            run_count += 1
            progress.progress(run_count / total_runs)

        # ===== Show Results =====
        st.success(f"""
        **Best Parameters**
        ATR SL: {best_stats['atr_sl']} | ATR TP: {best_stats['atr_tp']}  
        EMA Trend: {best_stats['ema_trend']} | Min Confluences: {best_stats['min_conf']}  
        Trade Mode: {best_stats['trade_mode']}  
        Trades: {best_stats['trade_count']} | Win Rate: {best_stats['win_rate']:.2f}% | 
        Total Profit: {best_stats['total_profit']:.2f}
        """)

        st.subheader("Trade Log (Best Params)")
        st.dataframe(best_trades)

        if not best_trades.empty:
            best_trades['Cum_PnL'] = best_trades['P/L'].cumsum()
            fig, ax = plt.subplots()
            ax.plot(best_trades['Exit Date'], best_trades['Cum_PnL'], marker='o')
            ax.set_title("Equity Curve")
            st.pyplot(fig)

        # ===== Live Recommendation =====
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

        if last['Close'] > last['EMA_TREND'] and len(live_long) >= best_stats['min_conf'] and best_stats['trade_mode'] in ["Both","Long Only"]:
            st.success(f"📈 LONG {last['Close']:.2f} | SL: {last['Close'] - best_stats['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] + best_stats['atr_tp']*last['ATR']:.2f}")
        elif last['Close'] < last['EMA_TREND'] and len(live_short) >= best_stats['min_conf'] and best_stats['trade_mode'] in ["Both","Short Only"]:
            st.warning(f"📉 SHORT {last['Close']:.2f} | SL: {last['Close'] + best_stats['atr_sl']*last['ATR']:.2f} | TP: {last['Close'] - best_stats['atr_tp']*last['ATR']:.2f}")
        else:
            st.error("❌ No strong trade setup now.")
