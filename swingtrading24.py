# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Swing Strategy + Backtest + Live Rec")

# -----------------------------
# Indicator helpers (no talib)
# -----------------------------
def ema(series: pd.Series, period: int = 20) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, period: int = 20, std_mul: float = 2.0):
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + std_mul * sd
    lower = ma - std_mul * sd
    return ma, upper, lower

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l = df['High'] - df['Low']
    h_pc = (df['High'] - df['Close'].shift(1)).abs()
    l_pc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

# -----------------------------
# Strategy: signals (lightweight)
# -----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA200'] = ema(df['Close'], 200)
    df['RSI'] = rsi(df['Close'], 14)
    df['BB_MID'], df['BB_UP'], df['BB_LO'] = bollinger_bands(df['Close'], 20, 2)
    df['ATR'] = atr(df, 14)
    return df

def generate_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Signal'] = None  # 'BUY' / 'SELL' / None
    position = None

    for i in range(len(df)):
        # skip until indicators are available
        if pd.isna(df['EMA20'].iat[i]) or pd.isna(df['EMA200'].iat[i]) or pd.isna(df['RSI'].iat[i]) or pd.isna(df['ATR'].iat[i]):
            continue

        close = float(df['Close'].iat[i])
        ema20 = float(df['EMA20'].iat[i])
        ema200 = float(df['EMA200'].iat[i])
        rsi_val = float(df['RSI'].iat[i])

        # Long setup: above 200EMA, pullback to <= EMA20, RSI > 45 (trend continuation)
        if (close > ema200) and (close <= ema20) and (rsi_val > 45):
            if position != "LONG":
                df.at[df.index[i], 'Signal'] = "BUY"
                position = "LONG"

        # Short setup: below 200EMA, pullback to >= EMA20, RSI < 55
        elif (close < ema200) and (close >= ema20) and (rsi_val < 55):
            if position != "SHORT":
                df.at[df.index[i], 'Signal'] = "SELL"
                position = "SHORT"

        # otherwise no new signal
    return df

# -----------------------------
# Backtest engine: trade-by-trade
# -----------------------------
def backtest_trades(df: pd.DataFrame, capital: float = 100000, risk_per_trade: float = 0.01, sl_atr_mul: float = 2.0, target_risk_mul: float = 2.0):
    """
    For each BUY/SELL signal we:
    - enter at close of signal bar
    - set stop loss at +/- sl_atr_mul * ATR
    - set target at entry +/- target_risk_mul * (entry - stop)
    - exit when target hit, stop hit, or opposite signal occurs
    Returns trades list and equity series.
    """
    df = df.copy().reset_index()
    trades = []
    equity = []
    cash = capital
    position = None
    entry_idx = None
    entry_price = None
    qty = 0
    stop = None
    target = None

    # track mark-to-market equity for every row
    mtm = np.full(len(df), capital, dtype=float)

    for i in range(len(df)):
        row = df.loc[i]

        # If new signal and no open position -> open trade
        if pd.notna(row['Signal']) and position is None:
            entry_price = float(row['Close'])
            atr_val = float(row['ATR']) if not pd.isna(row['ATR']) else 0.0
            if atr_val <= 0:
                # cannot compute stops properly, skip
                continue

            if row['Signal'] == "BUY":
                stop = entry_price - sl_atr_mul * atr_val
                if stop <= 0:
                    continue
                # position size by risk%: risk_amount / (entry - stop)
                per_trade_risk = cash * risk_per_trade
                qty = int(per_trade_risk // (entry_price - stop)) if (entry_price - stop) > 0 else 0
                if qty <= 0:
                    # too small to take meaningful qty
                    continue
                target = entry_price + target_risk_mul * (entry_price - stop)
                position = "LONG"
                entry_idx = i

            elif row['Signal'] == "SELL":
                stop = entry_price + sl_atr_mul * atr_val
                per_trade_risk = cash * risk_per_trade
                qty = int(per_trade_risk // (stop - entry_price)) if (stop - entry_price) > 0 else 0
                if qty <= 0:
                    continue
                target = entry_price - target_risk_mul * (stop - entry_price)
                position = "SHORT"
                entry_idx = i

            # reserve capital is not deducted (we use cash as capital; this is a simple backtest)
            trades.append({
                "entry_idx": entry_idx,
                "entry_date": df.at[entry_idx, df.columns[0]],
                "type": position,
                "entry_price": entry_price,
                "qty": qty,
                "stop": stop,
                "target": target,
                "exit_idx": None,
                "exit_date": None,
                "exit_price": None,
                "pnl": None
            })

        # If a position is open, check for stop/target/opposite-signal exit
        if position is not None:
            cur_price = float(row['Close'])
            # check LONG
            trade = trades[-1] if trades else None
            if position == "LONG":
                # stop hit
                if cur_price <= stop:
                    exit_price = stop
                    trade['exit_idx'] = i
                    trade['exit_date'] = df.at[i, df.columns[0]]
                    trade['exit_price'] = exit_price
                    trade['pnl'] = (exit_price - trade['entry_price']) * trade['qty']
                    position = None
                    entry_idx = None
                    qty = 0
                # target hit
                elif cur_price >= trade['target']:
                    exit_price = trade['target']
                    trade['exit_idx'] = i
                    trade['exit_date'] = df.at[i, df.columns[0]]
                    trade['exit_price'] = exit_price
                    trade['pnl'] = (exit_price - trade['entry_price']) * trade['qty']
                    position = None
                    entry_idx = None
                    qty = 0
                # opposite signal occurs -> exit at close
                elif pd.notna(row['Signal']) and row['Signal'] == "SELL":
                    exit_price = cur_price
                    trade['exit_idx'] = i
                    trade['exit_date'] = df.at[i, df.columns[0]]
                    trade['exit_price'] = exit_price
                    trade['pnl'] = (exit_price - trade['entry_price']) * trade['qty']
                    position = None
                    entry_idx = None
                    qty = 0

            elif position == "SHORT":
                if cur_price >= stop:
                    exit_price = stop
                    trade['exit_idx'] = i
                    trade['exit_date'] = df.at[i, df.columns[0]]
                    trade['exit_price'] = exit_price
                    trade['pnl'] = (trade['entry_price'] - exit_price) * trade['qty']
                    position = None
                    entry_idx = None
                    qty = 0
                elif cur_price <= trade['target']:
                    exit_price = trade['target']
                    trade['exit_idx'] = i
                    trade['exit_date'] = df.at[i, df.columns[0]]
                    trade['exit_price'] = exit_price
                    trade['pnl'] = (trade['entry_price'] - exit_price) * trade['qty']
                    position = None
                    entry_idx = None
                    qty = 0
                elif pd.notna(row['Signal']) and row['Signal'] == "BUY":
                    exit_price = cur_price
                    trade['exit_idx'] = i
                    trade['exit_date'] = df.at[i, df.columns[0]]
                    trade['exit_price'] = exit_price
                    trade['pnl'] = (trade['entry_price'] - exit_price) * trade['qty']
                    position = None
                    entry_idx = None
                    qty = 0

        # mark-to-market equity
        if position is None:
            mtm[i] = capital  # we keep flat capital as baseline (simpler)
        else:
            # if long
            if trades:
                t = trades[-1]
                if t['exit_price'] is None:
                    if t['type'] == 'LONG':
                        mtm[i] = capital + (float(row['Close']) - t['entry_price']) * t['qty']
                    else:
                        mtm[i] = capital + (t['entry_price'] - float(row['Close'])) * t['qty']
                else:
                    mtm[i] = capital + t['pnl']  # closed trade
            else:
                mtm[i] = capital

    # close any open trade at last close
    if trades and trades[-1]['exit_price'] is None:
        last = trades[-1]
        last_price = float(df.at[len(df)-1, 'Close'])
        if last['type'] == 'LONG':
            last['exit_idx'] = len(df)-1
            last['exit_date'] = df.at[len(df)-1, df.columns[0]]
            last['exit_price'] = last_price
            last['pnl'] = (last_price - last['entry_price']) * last['qty']
        else:
            last['exit_idx'] = len(df)-1
            last['exit_date'] = df.at[len(df)-1, df.columns[0]]
            last['exit_price'] = last_price
            last['pnl'] = (last['entry_price'] - last_price) * last['qty']

    equity_series = pd.Series(mtm, index=df[df.columns[0]])
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_series

# -----------------------------
# Performance metrics
# -----------------------------
def compute_metrics(trades_df: pd.DataFrame, equity: pd.Series):
    results = {}
    if trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "expectancy": None,
            "max_drawdown": None,
            "cagr": None,
            "sharpe": None,
            "net_profit": 0
        }

    trades_df = trades_df.copy()
    trades_df['pnl'] = trades_df['pnl'].astype(float)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    total_trades = len(trades_df)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0

    # expectancy per trade = (win_rate * avg_win + loss_rate * avg_loss) / average_risk
    expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss)

    # net profit
    net_profit = trades_df['pnl'].sum()

    # equity returns for daily frequency (approx)
    daily_equity = equity.resample('D').last().ffill().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    if not daily_returns.empty:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else np.nan
    else:
        sharpe = np.nan

    # CAGR
    try:
        start_val = daily_equity.iloc[0]
        end_val = daily_equity.iloc[-1]
        days = (daily_equity.index[-1] - daily_equity.index[0]).days
        years = days / 365.25 if days > 0 else 1/365.25
        cagr = ((end_val / start_val) ** (1 / years) - 1) if start_val > 0 else np.nan
    except Exception:
        cagr = np.nan

    # max drawdown
    roll_max = daily_equity.cummax()
    drawdown = (daily_equity - roll_max) / roll_max
    max_dd = drawdown.min() if not drawdown.empty else 0

    results.update({
        "total_trades": int(total_trades),
        "win_rate": float(win_rate) if total_trades > 0 else None,
        "avg_win": float(avg_win) if not np.isnan(avg_win) else None,
        "avg_loss": float(avg_loss) if not np.isnan(avg_loss) else None,
        "expectancy": float(expectancy) if not np.isnan(expectancy) else None,
        "net_profit": float(net_profit),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "cagr": float(cagr) if not np.isnan(cagr) else None,
        "max_drawdown": float(max_dd) if not np.isnan(max_dd) else None
    })
    return results

# -----------------------------
# UI layout & pickers
# -----------------------------
st.title("📊 Swing Strategy Backtest + Live Recommendation (No TA libs)")

left_col, right_col = st.columns([1, 2])
with left_col:
    # tickers list (some common names) and ability to enter custom
    tickers = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "FIN NIFTY": "NIFTY_FIN_SERVICE.NS",
        "SENSEX": "^BSESN",
        "RELIANCE": "RELIANCE.NS",
        "INFY": "INFY.NS",
        "TCS": "TCS.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "SBIN": "SBIN.NS"
    }
    choice = st.selectbox("Select Ticker", list(tickers.keys()) + ["Other"], index=0)
    if choice == "Other":
        ticker = st.text_input("Yahoo ticker (e.g. SBIN.NS)", "SBIN.NS")
    else:
        ticker = tickers[choice]

    # period/interval selection (yfinance constraints apply)
    st.markdown("**Select Period & Interval**")
    period_options = ["1d","5d","1mo","3mo","6mo","1y","2y","3y","5y","10y","ytd","max"]
    intraday_intervals = ["1m","2m","5m","15m","30m","60m","90m"]
    higher_intervals = ["1h","1d","5d","1wk","1mo"]  # note: '1h' maps close to 60m or 1h usage
    interval_options = intraday_intervals + ["15m","30m","60m","1h","4h","1d","1wk","1mo"]

    period = st.selectbox("Period", period_options, index=5)
    interval = st.selectbox("Interval", ["1m","3m","5m","10m","15m","30m","60m","90m","1h","4h","1d","1wk","1mo"], index=9)

    st.write("Note: `yfinance` has limitations for intraday data (1m typically available up to 7 days). If you pick incompatible Period+Interval you may get empty data.")

    capital = st.number_input("Capital (₹)", value=100000.0, step=10000.0)
    risk_per_trade = st.slider("Risk per trade (% of capital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100.0

    run_button = st.button("Run Backtest & Get Live Recommendation")

with right_col:
    st.markdown("### Output")
    status_area = st.empty()
    chart_area = st.empty()
    metrics_area = st.empty()
    trades_area = st.empty()
    rec_area = st.empty()

# -----------------------------
# Run: download, compute, backtest, metrics
# -----------------------------
if run_button:
    status_area.info(f"Fetching {ticker} for period={period}, interval={interval} ...")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            status_area.error("No data returned by yfinance — try a shorter period for intraday intervals or a different interval.")
        else:
            status_area.success(f"Data fetched: {df.shape[0]} rows.")
            # compute indicators & signals
            df = compute_indicators(df)
            df = generate_entry_signals(df)

            # backtest
            trades_df, equity_ser = backtest_trades(df, capital=capital, risk_per_trade=risk_per_trade)
            metrics = compute_metrics(trades_df, equity_ser)

            # plot price + signals
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df.index, df['Close'], label='Close')
            # plot EMAs
            if 'EMA20' in df.columns:
                ax.plot(df.index, df['EMA20'], label='EMA20', linewidth=0.8)
            if 'EMA200' in df.columns:
                ax.plot(df.index, df['EMA200'], label='EMA200', linewidth=0.8)
            # signals
            buys = df[df['Signal'] == 'BUY']
            sells = df[df['Signal'] == 'SELL']
            if not buys.empty:
                ax.scatter(buys.index, buys['Close'], marker='^', s=80, label='BUY', zorder=5)
            if not sells.empty:
                ax.scatter(sells.index, sells['Close'], marker='v', s=80, label='SELL', zorder=5)
            ax.legend()
            ax.set_title(f"{ticker} Price & Signals")
            chart_area.pyplot(fig)

            # equity curve
            fig2, ax2 = plt.subplots(figsize=(12,4))
            ax2.plot(equity_ser.index, equity_ser.values, label='Equity')
            ax2.set_title("Equity Curve (mark-to-market)")
            ax2.legend()
            chart_area.pyplot(fig2)

            # metrics
            md = metrics
            metrics_text = f"""
            **Total trades:** {md['total_trades']}
            **Net profit:** ₹{md['net_profit']:.2f}
            **Win rate:** {md['win_rate']*100:.2f}%  
            **Avg win:** ₹{md['avg_win']:.2f}
            **Avg loss:** ₹{md['avg_loss']:.2f}
            **Expectancy (₹ per trade):** ₹{md['expectancy']:.2f}
            **CAGR:** {md['cagr']*100:.2f}%  
            **Sharpe:** {md['sharpe']:.2f}
            **Max Drawdown:** {md['max_drawdown']*100:.2f}%
            """
            metrics_area.markdown(metrics_text)

            # trades table
            if not trades_df.empty:
                display_trades = trades_df.copy()
                display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date'])
                display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date'])
                trades_area.dataframe(display_trades[['entry_date','exit_date','type','entry_price','exit_price','qty','pnl']].sort_values('entry_date', ascending=False).reset_index(drop=True))
            else:
                trades_area.info("No trades generated by the signals for selected period/interval.")

            # live recommendation logic
            #  - require recent valid signal (last non-null Signal)
            #  - require backtest metrics to be reasonably positive to recommend (simple heuristic)
            last_signal_row = df[df['Signal'].notna()].tail(1)
            if not last_signal_row.empty:
                last_signal = last_signal_row['Signal'].values[0]
                last_signal_time = last_signal_row.index[-1]
                # Safety thresholds to allow 'GO' for live trading
                # (These thresholds are suggestions - tune them)
                min_trades_for_confidence = 10
                min_expectancy = 0  # positive expectation
                min_sharpe = 0.2
                max_dd_allowed = -0.6  # -60% (very loose), you can tighten to -0.25 etc.

                # compute conditions
                cond_trades = md['total_trades'] >= min_trades_for_confidence
                cond_expectancy = (md['expectancy'] is not None) and (md['expectancy'] > min_expectancy)
                cond_sharpe = (md['sharpe'] is not None) and (md['sharpe'] >= min_sharpe)
                cond_dd = (md['max_drawdown'] is not None) and (md['max_drawdown'] >= max_dd_allowed)  # max_drawdown is negative or zero

                # final recommend
                if cond_trades and cond_expectancy and cond_sharpe and cond_dd:
                    rec_text = f"✅ **RECOMMEND** {last_signal} {ticker} (Signal time: {last_signal_time}). Backtest metrics meet thresholds."
                    rec_style = "success"
                else:
                    rec_text = f"⚠️ **NO LIVE RECOMMENDATION** for {ticker} now. Latest signal: {last_signal} at {last_signal_time}. Backtest metrics insufficient."
                    rec_style = "warning"

                rec_area.markdown(f"### Live Recommendation\n**Latest signal:** {last_signal} at {last_signal_time}")
                if rec_style == "success":
                    rec_area.success(rec_text)
                else:
                    rec_area.warning(rec_text)

                # show the most recent bar and context
                recent = df.tail(5)[['Close','EMA20','EMA200','RSI','ATR','Signal']].fillna("-")
                rec_area.markdown("**Recent bars (last 5 rows)**")
                rec_area.dataframe(recent)
            else:
                rec_area.info("No signals found in this dataset. Cannot provide live recommendation.")

    except Exception as e:
        status_area.error(f"Error fetching or processing data: {e}")
