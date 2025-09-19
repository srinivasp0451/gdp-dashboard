# streamlit_swing_trader.py
# Robust Swing Trading + "Smart Trading" Facility
# Now uses file upload + column mapping (no yfinance) and includes a Live Recommendation
# No use of pandas_ta or talib. All indicators implemented from scratch.
# Requirements: streamlit, pandas, numpy, matplotlib, openpyxl (if using xlsx)
# Run: streamlit run streamlit_swing_trader.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Robust Swing Trader (File Upload)")

st.title("Robust Swing Trading + Smart Trading Facility — File Upload + Live Recommendation")
st.markdown(
    """
    Upload your OHLCV file (CSV or Excel). Map the columns and run the strategy.

    This version preserves the original logic (trend + pullback + squeeze/momentum + big-player detection)
    but replaces `yfinance` with a file uploader and adds a **Live Recommendation** panel based on the latest bar.

    **Notes**: This is a research/backtest/live-readiness template. Forward-test before trading real capital.
    """
)

# ------------------------------
# Sidebar: user inputs + file upload
# ------------------------------
with st.sidebar:
    st.header("Inputs")
    uploaded_file = st.file_uploader("Upload CSV / Excel with OHLCV data (Date/Time, Open, High, Low, Close, Volume)", type=["csv", "xls", "xlsx"])    
    st.markdown("---")

    st.subheader("Strategy Params")
    risk_percent = st.number_input("Risk per trade (% of capital)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
    starting_capital = st.number_input("Starting capital (INR)", value=100000.0, step=1000.0)
    max_hold_bars = st.number_input("Max holding bars (if TP/SL not hit)", value=20, min_value=1, step=1)
    spike_mult = st.number_input("Volume spike multiplier (big-player detector)", value=3.0, step=0.1)
    atr_mult_sl = st.number_input("Stop-loss ATR multiplier", value=2.0, step=0.1)
    rr_target = st.number_input("Target R:R (e.g., 2 means 2R)", value=2.0, step=0.1)
    show_indicators = st.checkbox("Show indicators on chart", value=True)

    st.markdown("---")
    st.write("Once uploaded & mapped, click 'Compute & Run' to compute signals and backtest.")
    run_backtest = st.button("Compute & Run")
    st.write("")

# ------------------------------
# Helpers: auto-detect & read file
# ------------------------------

def guess_column(cols, candidates):
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(cols_l):
            if cand.lower() == c:
                return cols[i]
    # fuzzy
    for cand in candidates:
        for i, c in enumerate(cols_l):
            if cand.lower() in c:
                return cols[i]
    return None


def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

# ------------------------------
# Indicator implementations (same as before)
# ------------------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
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
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vol = df['Volume']
    df2 = df.copy()
    df2['typ_vol'] = tp * vol
    df2['date_only'] = df2.index.normalize()
    df2['cum_tp_vol'] = df2.groupby('date_only')['typ_vol'].cumsum()
    df2['cum_vol'] = df2.groupby('date_only')['Volume'].cumsum()
    df2['vwap'] = df2['cum_tp_vol'] / df2['cum_vol']
    return df2['vwap']


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
    pct = df['Close'].pct_change().fillna(0)
    vpt = (pct * df['Volume']).cumsum()
    return vpt


def volume_profile(df, bins=24):
    prices = df['Close']
    vols = df['Volume']
    low = prices.min()
    high = prices.max()
    if high == low:
        edges = np.linspace(low - 1, high + 1, bins + 1)
    else:
        edges = np.linspace(low, high, bins + 1)
    inds = np.digitize(prices, edges) - 1
    vol_by_bin = np.zeros(bins)
    for i, v in zip(inds, vols):
        if 0 <= i < bins:
            vol_by_bin[i] += v
    top_idx = np.argsort(vol_by_bin)[-5:][::-1]
    zones = [(edges[i], edges[i + 1], vol_by_bin[i]) for i in top_idx]
    return zones, edges, vol_by_bin


def pivots(df, window=5):
    highs = df['High']
    lows = df['Low']
    ph = highs[(highs == highs.rolling(window, center=True).max())]
    pl = lows[(lows == lows.rolling(window, center=True).min())]
    return ph.dropna(), pl.dropna()

# ------------------------------
# Compute indicators and signals (vectorized to avoid ambiguous Series checks)
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
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['kc_mid'] = tp.ewm(span=20, adjust=False).mean()
    df['kc_upper'] = df['kc_mid'] + 1.5 * df['atr14']
    df['kc_lower'] = df['kc_mid'] - 1.5 * df['atr14']
    df['vwap'] = vwap(df)
    df['obv'] = obv(df)
    df['vpt'] = vpt(df)
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['squeeze_on'] = (df['bb_width'] < (df['kc_upper'] - df['kc_lower']))
    return df


def generate_signals(df, spike_mult=3.0, atr_mult_sl=2.0, rr_target=2.0):
    df = df.copy()
    # Rolling baseline for BB width
    bb_w_ma = df['bb_width'].rolling(20).mean()
    # Squeeze release and momentum
    df['squeeze_release'] = df['squeeze_on'].shift(1).fillna(False) & (df['bb_width'] > bb_w_ma * 1.5)
    df['vol_pickup'] = df['Volume'] > 1.2 * df['vol_ma20']
    df['momentum_imminent'] = df['squeeze_release'] & df['vol_pickup']
    # Big player detection
    df['vol_spike'] = df['Volume'] > spike_mult * df['vol_ma20']
    df['strong_move'] = (df['Close'] - df['Close'].shift(1)).abs() > 0.8 * df['atr14']
    df['big_player'] = df['vol_spike'] & df['strong_move']
    # Trend and pullback conditions
    df['above_ema200'] = df['Close'] > df['ema200']
    df['below_ema200'] = df['Close'] < df['ema200']
    df['to_ema20_pullback_long'] = df['Close'] <= df['ema20']
    df['to_ema20_pullback_short'] = df['Close'] >= df['ema20']
    df['rsi_rise'] = df['rsi14'] > df['rsi14'].shift(1)
    df['close_up'] = df['Close'] > df['Open']
    df['close_down'] = df['Close'] < df['Open']
    long_cond = (
        df['above_ema200'] & df['to_ema20_pullback_long'] & (df['rsi14'] > 45) & df['rsi_rise'] & df['close_up']
    )
    short_cond = (
        df['below_ema200'] & df['to_ema20_pullback_short'] & (df['rsi14'] < 55) & (~df['rsi_rise']) & df['close_down']
    )
    extra_ok = df['momentum_imminent'] | (~df['squeeze_on']) | (df['Volume'] > df['vol_ma20'] * 0.6)
    df['signal'] = 0
    df['signal_reason'] = ''
    df.loc[long_cond & extra_ok, 'signal'] = 1
    df.loc[short_cond & extra_ok, 'signal'] = -1
    df.loc[long_cond & extra_ok, 'signal_reason'] = 'Trend+Pullback to EMA20'
    df.loc[short_cond & extra_ok, 'signal_reason'] = 'Trend+Pullback to EMA20'
    return df

# ------------------------------
# Backtest engine (unchanged core logic)
# ------------------------------

def backtest(df, capital=100000, risk_pct=1.0, max_hold=20, atr_mult=2.0, rr_target=2.0):
    trades = []
    cash = capital
    equity_curve = [capital]
    for i in range(len(df)):
        if df['signal'].iat[i] == 0:
            equity_curve.append(equity_curve[-1])
            continue
        sig = int(df['signal'].iat[i])
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
                if df['Low'].iat[j] <= stop:
                    exit_price = stop
                    exit_idx = j
                    break
                elif df['High'].iat[j] >= target:
                    exit_price = target
                    exit_idx = j
                    break
            if exit_price is None:
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
    # build equity series aligned to df index (simple)
    equity_index = list(df.index[:len(equity_curve)])
    equity = pd.Series(equity_curve, index=equity_index)
    return trades_df, equity

# ------------------------------
# Main app logic: file upload, mapping, compute, backtest, live recommendation
# ------------------------------

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file with OHLCV data in the sidebar.")
else:
    try:
        raw = load_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

    st.subheader("File preview (first 10 rows)")
    st.write(raw.head(10))

    # Attempt to guess columns
    cols = list(raw.columns)
    guessed_date = guess_column(cols, ["date", "datetime", "timestamp", "time"])
    guessed_open = guess_column(cols, ["open", "opn"])
    guessed_high = guess_column(cols, ["high", "hi"])
    guessed_low = guess_column(cols, ["low", "lo"])
    guessed_close = guess_column(cols, ["close", "adj close", "last"])
    guessed_vol = guess_column(cols, ["volume", "vol"])

    st.subheader("Map columns")
    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox("Date/Time column", options=[None] + cols, index=cols.index(guessed_date) + 1 if guessed_date in cols else 0)
        open_col = st.selectbox("Open column", options=[None] + cols, index=cols.index(guessed_open) + 1 if guessed_open in cols else 0)
        high_col = st.selectbox("High column", options=[None] + cols, index=cols.index(guessed_high) + 1 if guessed_high in cols else 0)
    with c2:
        low_col = st.selectbox("Low column", options=[None] + cols, index=cols.index(guessed_low) + 1 if guessed_low in cols else 0)
        close_col = st.selectbox("Close column", options=[None] + cols, index=cols.index(guessed_close) + 1 if guessed_close in cols else 0)
        vol_col = st.selectbox("Volume column", options=[None] + cols, index=cols.index(guessed_vol) + 1 if guessed_vol in cols else 0)

    st.markdown("**Parsing options**")
    date_format = st.text_input("Date format (optional, strptime style). Leave blank to auto-parse.", value="")
    timezone = st.text_input("Timezone (for display only, optional)", value="")

    if not date_col or not open_col or not high_col or not low_col or not close_col:
        st.warning("Please map at least Date, Open, High, Low and Close columns.")
        st.stop()

    # Build dataframe with required columns
    working = raw[[date_col, open_col, high_col, low_col, close_col]].copy()
    if vol_col:
        working['Volume'] = raw[vol_col].fillna(0)
    else:
        working['Volume'] = 0

    # parse date
    try:
        if date_format.strip() == "":
            working[date_col] = pd.to_datetime(working[date_col], infer_datetime_format=True, errors='coerce')
        else:
            working[date_col] = pd.to_datetime(working[date_col], format=date_format, errors='coerce')
    except Exception as e:
        st.error(f"Failed to parse date column: {e}")
        st.stop()

    if working[date_col].isnull().any():
        st.warning("Some date values could not be parsed and will be dropped.")
        working = working.dropna(subset=[date_col])

    working = working.rename(columns={
        date_col: 'DateTime',
        open_col: 'Open',
        high_col: 'High',
        low_col: 'Low',
        close_col: 'Close'
    })
    working = working[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    working = working.sort_values('DateTime').set_index('DateTime')

    df = working.copy()

    if run_backtest:
        with st.spinner("Computing indicators and signals..."):
            df = compute_indicators(df)
            df = generate_signals(df, spike_mult=spike_mult, atr_mult_sl=atr_mult_sl, rr_target=rr_target)
            zones, edges, vol_by_bin = volume_profile(df, bins=24)
        st.success("Indicators & signals computed.")

        st.subheader("Latest data (last 10 rows)")
        st.write(df.tail(10))

        # Price chart
        st.subheader("Price chart (indicators + signals)")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df['Close'], label='Close')
        if show_indicators:
            ax.plot(df.index, df['ema20'], label='EMA20', linewidth=0.8)
            ax.plot(df.index, df['ema200'], label='EMA200', linewidth=0.8)
            ax.plot(df.index, df['bb_upper'], label='BB Upper', linewidth=0.5, alpha=0.6)
            ax.plot(df.index, df['bb_lower'], label='BB Lower', linewidth=0.5, alpha=0.6)
            ax.plot(df.index, df['vwap'], label='VWAP', linewidth=0.8)
        longs = df[df['signal'] == 1]
        shorts = df[df['signal'] == -1]
        ax.scatter(longs.index, longs['Close'], marker='^', color='green', s=60, label='Long Signal')
        ax.scatter(shorts.index, shorts['Close'], marker='v', color='red', s=60, label='Short Signal')
        bigp = df[df['big_player']]
        ax.scatter(bigp.index, bigp['Close'], marker='o', facecolors='none', edgecolors='black', s=80, label='Big-player vol')
        ax.legend(loc='upper left')
        ax.set_title(f"Price with indicators & signals")
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        st.pyplot(fig)

        # volume profile
        st.subheader("Volume Profile (approx)")
        fig2, ax2 = plt.subplots(figsize=(4, 6))
        bins_center = (edges[:-1] + edges[1:]) / 2
        ax2.barh(bins_center, vol_by_bin)
        ax2.set_xlabel('Volume')
        ax2.set_ylabel('Price')
        st.pyplot(fig2)

        # backtest
        st.subheader("Backtest (simple discrete trades)")
        trades_df, equity = backtest(df, capital=starting_capital, risk_pct=risk_percent, max_hold=max_hold_bars, atr_mult=atr_mult_sl, rr_target=rr_target)
        if trades_df.empty:
            st.warning("No trades generated with these parameters. Consider loosening conditions or changing timeframe/data sampling.")
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
            st.subheader("Equity Curve")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            equity.plot(ax=ax3)
            ax3.set_ylabel('Equity (INR)')
            st.pyplot(fig3)

        # Smart insights
        st.subheader("Smart Trading Insights (recent)")
        insights = []
        recent = df.iloc[-30:]
        if recent['momentum_imminent'].any():
            insights.append("Momentum signals detected recently (squeeze release + vol pickup). Watch for directional breakout.")
        if recent['big_player'].any():
            insights.append("Big-player volume spikes observed recently — these often precede extended moves or mark liquidity events.")
        if zones:
            top_zone = zones[0]
            insights.append(f"Top high-volume price area near {top_zone[0]:.2f} to {top_zone[1]:.2f}. Treat as strong support/resistance.")
        if len(insights) == 0:
            st.write("No immediate smart signals flagged in the recent window. Continue monitoring squeeze/volume.")
        else:
            for it in insights:
                st.info(it)

        st.markdown("---")
        st.subheader("Live Recommendation (based on latest bar)")
        if st.button("Get Live Recommendation"):
            last = df.iloc[-1]
            reasons = []
            score = 0.0
            # trend
            if last['above_ema200']:
                reasons.append('Trend: LONG (Close > EMA200)')
                score += 0.4
            elif last['below_ema200']:
                reasons.append('Trend: SHORT (Close < EMA200)')
                score += 0.4
            else:
                reasons.append('Trend: Neutral')
            # pullback
            if last['to_ema20_pullback_long'] and last['above_ema200']:
                reasons.append('Price near/below EMA20 (pullback) — possible entry on bounce')
                score += 0.15
            if last['to_ema20_pullback_short'] and last['below_ema200']:
                reasons.append('Price near/above EMA20 (pullback) — possible short on rejection')
                score += 0.15
            # momentum / big player
            if last['momentum_imminent']:
                reasons.append('Momentum imminent (squeeze release + vol pickup)')
                score += 0.2
            if last['big_player']:
                reasons.append('Big-player volume spike detected')
                score += 0.2
            # volume
            if last['Volume'] > last['vol_ma20']:
                reasons.append('Volume above 20-period average')
                score += 0.1
            # cap score at 1.0
            conf = min(score, 1.0)
            # determine recommendation
            rec = 'HOLD'
            if last['signal'] == 1:
                rec = 'BUY'
            elif last['signal'] == -1:
                rec = 'SELL'
            else:
                # infer from trend + momentum
                if last['above_ema200'] and (last['momentum_imminent'] or last['big_player']):
                    rec = 'WATCH/CONSIDER BUY'
                elif last['below_ema200'] and (last['momentum_imminent'] or last['big_player']):
                    rec = 'WATCH/CONSIDER SELL'
            # show
            st.success(f"Recommendation: {rec} — Confidence: {conf*100:.0f}%")
            st.write("Reasons:")
            for r in reasons:
                st.write(f"- {r}")

        st.write("---")
        st.write("**How to use**: Upload a fresh CSV/Excel (intraday or daily), map the columns, click 'Compute & Run'. Use 'Get Live Recommendation' to evaluate the latest bar. Tune the parameters in the sidebar to your universe/timeframe.")

    else:
        st.info("Map columns and click 'Compute & Run' in the sidebar to compute indicators, signals, and run backtest.")

# EOF
