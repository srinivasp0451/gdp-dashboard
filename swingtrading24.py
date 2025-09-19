# streamlit_swing_trader.py
# Robust Swing Trading + "Smart Trading" Facility
# No use of pandas_ta or talib. All indicators implemented from scratch.
# Requirements: streamlit, yfinance, pandas, numpy, matplotlib
# Run: streamlit run streamlit_swing_trader.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Robust Swing Trader (No TA-Lib)")

st.title("Robust Swing Trading + Smart Trading Facility (India-focused)")
st.markdown(
    """
    This Streamlit app implements a robust swing-trading framework (trend + mean-reversion hybrid) **without** pandas_ta or talib.
    It also includes "smart" signals that attempt to identify: **momentum about to start**, **big-player entries**, and **high-volume support/resistance zones** (volume profile HVNs).

    **Notes / disclaimers**: This is an algorithmic template for research & backtesting. No guarantees. Always forward-test on small capital or paper trade.
    """
)

# ------------------------------
# Sidebar: user inputs
# ------------------------------
with st.sidebar:
    st.header("Strategy Inputs")
    ticker_default = st.selectbox(
        "Choose ticker / index (examples)",
        (
            "^NSEI",
            "^NSEBANK",
            "RELIANCE.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "INFY.NS",
            "ICICIBANK.NS",
            "KOTAKBANK.NS",
            "LT.NS",
            "AXISBANK.NS",
            "HDFC.NS",
            "SBIN.NS",
            "Other...",
        ),
    )
    if ticker_default == "Other...":
        ticker = st.text_input("Enter custom ticker (Yahoo format)", value="RELIANCE.NS")
    else:
        ticker = ticker_default

    interval = st.selectbox("Interval (yfinance)", ("1d", "60m", "30m", "15m", "5m"))
    period = st.selectbox("Period to fetch", ("6mo", "1y", "2y", "3mo"))

    st.markdown("---")
    st.subheader("Strategy Params")
    risk_percent = st.number_input("Risk per trade (% of capital)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
    starting_capital = st.number_input("Starting capital (INR)", value=100000.0, step=1000.0)
    max_hold_bars = st.number_input("Max holding bars (if TP/SL not hit)", value=20, min_value=1, step=1)
    spike_mult = st.number_input("Volume spike multiplier (big-player detector)", value=3.0, step=0.1)
    atr_mult_sl = st.number_input("Stop-loss ATR multiplier", value=2.0, step=0.1)
    rr_target = st.number_input("Target R:R (e.g., 2 means 2R)", value=2.0, step=0.1)

    st.markdown("---")
    st.write("Data / Misc")
    show_indicators = st.checkbox("Show indicators on chart", value=True)
    run_backtest = st.button("Fetch & Run Backtest")

# ------------------------------
# Utility indicator functions
# ------------------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's EMA
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = upper - lower
    return ma, upper, lower, width


def vwap(df):
    # daily VWAP (works across index with multi-day data)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume']
    df = df.copy()
    df['typ_vol'] = tp * vol
    # compute VWAP per day
    df['date'] = df.index.normalize()
    df['cum_tp_vol'] = df.groupby('date')['typ_vol'].cumsum()
    df['cum_vol'] = df.groupby('date')['Volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    return df['vwap']


def obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iat[i] > df['Close'].iat[i - 1]:
            obv.append(obv[-1] + df['Volume'].iat[i])
        elif df['Close'].iat[i] < df['Close'].iat[i - 1]:
            obv.append(obv[-1] - df['Volume'].iat[i])
        else:
            obv.append(obv[-1])
    s = pd.Series(obv, index=df.index)
    return s


def vpt(df):
    # Volume Price Trend
    pct = df['Close'].pct_change().fillna(0)
    vpt = (pct * df['Volume']).cumsum()
    return vpt

# Volume profile (simple bin based)
def volume_profile(df, bins=24):
    prices = df['Close']
    vols = df['Volume']
    low = prices.min()
    high = prices.max()
    bin_edges = np.linspace(low, high, bins + 1)
    inds = np.digitize(prices, bin_edges) - 1
    vol_by_bin = np.zeros(bins)
    for i, v in zip(inds, vols):
        if 0 <= i < bins:
            vol_by_bin[i] += v
    # top bins descending
    top_idx = np.argsort(vol_by_bin)[-5:][::-1]
    zones = [(bin_edges[i], bin_edges[i + 1], vol_by_bin[i]) for i in top_idx]
    return zones, bin_edges, vol_by_bin

# Pivot detection (simple center rolling window)
def pivots(df, window=5):
    highs = df['High']
    lows = df['Low']
    ph = highs[(highs == highs.rolling(window, center=True).max())]
    pl = lows[(lows == lows.rolling(window, center=True).min())]
    return ph.dropna(), pl.dropna()

# ------------------------------
# Signal generation and 'smart' detectors
# ------------------------------

def compute_indicators(df):
    df = df.copy()
    df['ema20'] = ema(df['Close'], 20)
    df['ema200'] = ema(df['Close'], 200)
    df['rsi14'] = rsi(df['Close'], 14)
    df['atr14'] = atr(df, 14)
    ma20, bb_upper, bb_lower, bb_width = bollinger_bands(df['Close'], 20, 2)
    df['bb_mid'] = ma20
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = bb_width
    # Keltner for squeeze-like detection
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['kc_mid'] = tp.ewm(span=20, adjust=False).mean()
    df['kc_upper'] = df['kc_mid'] + 1.5 * df['atr14']
    df['kc_lower'] = df['kc_mid'] - 1.5 * df['atr14']
    df['vwap'] = vwap(df)
    df['obv'] = obv(df)
    df['vpt'] = vpt(df)
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    # Squeeze detection: BB width less than KC width => squeeze on
    df['squeeze_on'] = (df['bb_width'] < (df['kc_upper'] - df['kc_lower']))
    return df


def generate_signals(df, spike_mult=3.0, atr_mult_sl=2.0, rr_target=2.0):
    df = df.copy()
    signals = []
    df['signal'] = 0
    df['signal_reason'] = ''
    df['momentum_imminent'] = False
    df['big_player'] = False

    # Determine bollinger width increase baseline
    bb_w_ma = df['bb_width'].rolling(20).mean()

    in_signal = False

    for i in range(201, len(df) - 1):  # start after ema200 stabilizes
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Momentum imminent: squeeze release + volume pickup + bb width expansion
        squeeze_release = prev['squeeze_on'] and (row['bb_width'] > max(1e-9, bb_w_ma.iloc[i - 1]) * 1.5)
        vol_pickup = row['Volume'] > 1.2 * row['vol_ma20'] if not np.isnan(row['vol_ma20']) else False
        momentum_flag = squeeze_release and vol_pickup
        if momentum_flag:
            df.at[row.name, 'momentum_imminent'] = True

        # Detect big player entry: big volume spike & directional move
        vol_spike = (row['Volume'] > spike_mult * row['vol_ma20']) if not np.isnan(row['vol_ma20']) else False
        strong_move = abs(row['Close'] - prev['Close']) > 0.8 * row['atr14'] if not np.isnan(row['atr14']) else False
        if vol_spike and strong_move:
            df.at[row.name, 'big_player'] = True

        # LONG condition (trend + pullback + confirmation)
        long_cond = (
            (row['Close'] > row['ema200'])
            and (row['Close'] <= row['ema20'])
            and (row['rsi14'] > 45)
            and (row['rsi14'] > prev['rsi14'])
            and (row['Close'] > row['Open'])
        )

        # SHORT condition
        short_cond = (
            (row['Close'] < row['ema200'])
            and (row['Close'] >= row['ema20'])
            and (row['rsi14'] < 55)
            and (row['rsi14'] < prev['rsi14'])
            and (row['Close'] < row['Open'])
        )

        reason = []
        if long_cond:
            reason.append('Trend+Pullback to EMA20')
        if short_cond:
            reason.append('Trend+Pullback to EMA20')

        # prefer trades with momentum or big_player entries, but allow without if conditions strict
        if long_cond and (momentum_flag or not row['squeeze_on'] or row['Volume'] > row['vol_ma20'] * 0.6):
            df.at[row.name, 'signal'] = 1
            df.at[row.name, 'signal_reason'] = "; ".join(reason)
        elif short_cond and (momentum_flag or not row['squeeze_on'] or row['Volume'] > row['vol_ma20'] * 0.6):
            df.at[row.name, 'signal'] = -1
            df.at[row.name, 'signal_reason'] = "; ".join(reason)

    return df

# ------------------------------
# Backtest engine (simple discrete trades)
# ------------------------------

def backtest(df, capital=100000, risk_pct=1.0, max_hold=20, atr_mult=2.0, rr_target=2.0):
    trades = []
    cash = capital
    equity_curve = [capital]
    current_idx = 0

    for i in range(len(df)):
        if df['signal'].iat[i] == 0:
            equity_curve.append(equity_curve[-1])
            continue

        sig = int(df['signal'].iat[i])
        # entry at next bar open if available
        if i + 1 >= len(df):
            break
        entry_idx = i + 1
        entry_price = df['Open'].iat[entry_idx]
        atr = df['atr14'].iat[entry_idx] if not np.isnan(df['atr14'].iat[entry_idx]) else 0
        if atr == 0:
            atr = df['Close'].diff().abs().rolling(14).mean().iat[entry_idx]

        if sig == 1:
            stop = entry_price - atr_mult * atr
            if stop <= 0:
                stop = entry_price * 0.98
            risk_per_share = entry_price - stop
            if risk_per_share <= 0:
                continue
            risk_amount = capital * (risk_pct / 100.0)
            qty = max(1, int(risk_amount / risk_per_share))
            target = entry_price + rr_target * (entry_price - stop)

            exit_price = None
            exit_idx = None
            for j in range(entry_idx, min(len(df), entry_idx + max_hold)):
                # check intrabar hits using High/Low
                if df['Low'].iat[j] <= stop:
                    exit_price = stop
                    exit_idx = j
                    break
                elif df['High'].iat[j] >= target:
                    exit_price = target
                    exit_idx = j
                    break
            if exit_price is None:
                # exit at close of last bar
                exit_idx = min(len(df) - 1, entry_idx + max_hold - 1)
                exit_price = df['Close'].iat[exit_idx]

            profit = (exit_price - entry_price) * qty
            pct_return_on_cap = profit / capital
            cash += profit
            trades.append({
                'entry_time': df.index[entry_idx],
                'exit_time': df.index[exit_idx],
                'side': 'LONG',
                'entry': entry_price,
                'exit': exit_price,
                'qty': qty,
                'pnl': profit,
                'pnl_pct': pct_return_on_cap,
            })
            equity_curve.append(cash)

        elif sig == -1:
            stop = entry_price + atr_mult * atr
            risk_per_share = stop - entry_price
            if risk_per_share <= 0:
                continue
            risk_amount = capital * (risk_pct / 100.0)
            qty = max(1, int(risk_amount / risk_per_share))
            target = entry_price - rr_target * (stop - entry_price)

            exit_price = None
            exit_idx = None
            for j in range(entry_idx, min(len(df), entry_idx + max_hold)):
                if df['High'].iat[j] >= stop:
                    exit_price = stop
                    exit_idx = j
                    break
                elif df['Low'].iat[j] <= target:
                    exit_price = target
                    exit_idx = j
                    break
            if exit_price is None:
                exit_idx = min(len(df) - 1, entry_idx + max_hold - 1)
                exit_price = df['Close'].iat[exit_idx]

            # for shorts PnL is reversed
            profit = (entry_price - exit_price) * qty
            pct_return_on_cap = profit / capital
            cash += profit
            trades.append({
                'entry_time': df.index[entry_idx],
                'exit_time': df.index[exit_idx],
                'side': 'SHORT',
                'entry': entry_price,
                'exit': exit_price,
                'qty': qty,
                'pnl': profit,
                'pnl_pct': pct_return_on_cap,
            })
            equity_curve.append(cash)

    trades_df = pd.DataFrame(trades)
    equity = pd.Series(equity_curve[:len(df) + 1], index=list(df.index[: len(equity_curve) - 1]) + [df.index[-1]])
    return trades_df, equity

# ------------------------------
# Main: fetch data, compute, show
# ------------------------------

if run_backtest:
    with st.spinner("Fetching data and calculating indicators..."):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception as e:
            st.error(f"Failed to download data: {e}")
            st.stop()

        if df is None or df.empty:
            st.error("No data returned for this ticker / interval. Try different ticker or shorter period (for intraday intervals).")
            st.stop()

        df.index = pd.to_datetime(df.index)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        df = compute_indicators(df)
        df = generate_signals(df, spike_mult=spike_mult, atr_mult_sl=atr_mult_sl, rr_target=rr_target)

        # compute volume profile zones
        zones, edges, vol_by_bin = volume_profile(df, bins=24)

        st.success("Data fetched and indicators computed.")

    # Show summary metrics
    st.subheader(f"{ticker} — Data & Smart Signals Summary")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(df.tail(10))
    with col2:
        st.write("Top volume-price zones (HVNs):")
        for z in zones:
            st.write(f"{z[0]:.2f} to {z[1]:.2f} — volume: {int(z[2])}")

    # chart
    st.subheader("Price chart (indicators + signals)")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['Close'], label='Close')
    if show_indicators:
        ax.plot(df.index, df['ema20'], label='EMA20', linewidth=0.8)
        ax.plot(df.index, df['ema200'], label='EMA200', linewidth=0.8)
        ax.plot(df.index, df['bb_upper'], label='BB Upper', linewidth=0.5, alpha=0.6)
        ax.plot(df.index, df['bb_lower'], label='BB Lower', linewidth=0.5, alpha=0.6)
        ax.plot(df.index, df['vwap'], label='VWAP', linewidth=0.8)

    # mark signals
    longs = df[df['signal'] == 1]
    shorts = df[df['signal'] == -1]
    ax.scatter(longs.index, longs['Close'], marker='^', color='green', s=60, label='Long Signal')
    ax.scatter(shorts.index, shorts['Close'], marker='v', color='red', s=60, label='Short Signal')

    # mark big player volumes
    bigp = df[df['big_player']]
    ax.scatter(bigp.index, bigp['Close'], marker='o', facecolors='none', edgecolors='black', s=80, label='Big-player vol')

    ax.legend(loc='upper left')
    ax.set_title(f"{ticker} — Close with indicators & signals")
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    st.pyplot(fig)

    # volume profile bar
    st.subheader("Volume Profile (approx)")
    fig2, ax2 = plt.subplots(figsize=(4, 6))
    bins_center = (edges[:-1] + edges[1:]) / 2
    ax2.barh(bins_center, vol_by_bin)
    ax2.set_xlabel('Volume')
    ax2.set_ylabel('Price')
    st.pyplot(fig2)

    # trades & backtest
    st.subheader("Backtest (simple discrete trades)")
    trades_df, equity = backtest(df, capital=starting_capital, risk_pct=risk_percent, max_hold=max_hold_bars, atr_mult=atr_mult_sl, rr_target=rr_target)

    if trades_df.empty:
        st.warning("No trades generated with these parameters. Consider loosening conditions or changing timeframe.")
    else:
        st.write(trades_df)
        total_pnl = trades_df['pnl'].sum()
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss)

        st.metric("Total P&L (INR)", f"{total_pnl:.2f}")
        st.metric("Trades", f"{len(trades_df)}")
        st.metric("Win Rate", f"{win_rate * 100:.2f}%")
        st.metric("Expectancy (INR per trade)", f"{expectancy:.2f}")

        # equity curve
        st.subheader("Equity Curve")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        equity.plot(ax=ax3)
        ax3.set_ylabel('Equity (INR)')
        st.pyplot(fig3)

    # additional smart indicators panel
    st.subheader("Smart Trading Insights (what to watch)")
    insights = []
    recent = df.iloc[-30:]
    if recent['momentum_imminent'].any():
        insights.append("Momentum signals detected recently (squeeze release + vol pickup). Watch for directional breakout.")
    if recent['big_player'].any():
        insights.append("Big-player volume spikes observed recently — these often precede extended moves or mark liquidity events.")
    # zone insights
    if zones:
        top_zone = zones[0]
        insights.append(f"Top high-volume price area near {top_zone[0]:.2f} to {top_zone[1]:.2f}. Treat as strong support/resistance.")

    if len(insights) == 0:
        st.write("No immediate smart signals flagged in the recent window. Continue monitoring squeeze/volume.")
    else:
        for it in insights:
            st.info(it)

    st.markdown("---")
    st.write("**How to use this template**: Tweak indicators (EMA lengths, ATR multipliers, volume spike multiplier) and timeframe. Always forward-test and monitor slippage / transaction costs.")
    st.write("If you want I can help further: add option/futures position sizing, hook to a broker API, or refine the big-player detector to use exchange OI / block trade data (requires additional data source).")

else:
    st.info("Set parameters in the sidebar and click 'Fetch & Run Backtest' to compute signals and run the simple backtest.")

# EOF
