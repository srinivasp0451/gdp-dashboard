# streamlit_swing_trader.py
# Robust Swing Trading + Smart Trading Facility — File Upload + Live Recommendation + Detailed Backtest Trade Log
# No pandas_ta / talib. All indicators implemented from scratch.
# Requirements: streamlit, pandas, numpy, matplotlib, openpyxl (if using xlsx)
# Run: streamlit run streamlit_swing_trader.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Robust Swing Trader — Trade Log & Live")

st.title("Robust Swing Trading + Smart Trading Facility — File Upload + Live Recommendation + Trade Journal")
st.markdown(
    """
    Upload OHLCV file (CSV/Excel), map columns, and run the strategy. This version adds:
      - Detailed trade log (entry/exit/time/qty/SL/TP/reason/confidence)
      - Summary statistics (total trades, wins, losses, accuracy, points, P&L, expectancy, max drawdown)
      - Live recommendation that returns a structured "trade object" (entry, target, SL, qty, confidence)

    The core strategy logic (trend + pullback + squeeze/momentum + big-player detection) is preserved.
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
    st.write("After upload & mapping click 'Compute & Run'")
    run_backtest = st.button("Compute & Run")

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
# Indicators (implementations)
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

# ------------------------------
# Compute indicators and signals (vectorized)
# Also compute a confidence score per row for probability-of-profit
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

    # Confidence score components (simple weighted heuristic)
    # weight: trend (0.4), momentum (0.2), big_player (0.2), volume (0.2)
    df['trend_score'] = np.where(df['Close'] > df['ema200'], 1.0, np.where(df['Close'] < df['ema200'], -1.0, 0.0))
    # momentum: squeeze release + bb expansion
    bb_w_ma = df['bb_width'].rolling(20).mean()
    df['squeeze_release'] = df['squeeze_on'].shift(1).fillna(False) & (df['bb_width'] > bb_w_ma * 1.5)
    df['vol_pickup'] = df['Volume'] > 1.2 * df['vol_ma20']
    df['momentum_flag'] = (df['squeeze_release'] & df['vol_pickup']).astype(int)
    df['big_player'] = ((df['Volume'] > spike_mult * df['vol_ma20']) & ((df['Close'] - df['Close'].shift(1)).abs() > 0.8 * df['atr14'])).astype(int)
    df['vol_sig'] = (df['Volume'] > df['vol_ma20']).astype(int)

    # normalize trend_score to positive-only for confidence calculation per direction
    df['conf_trend'] = np.where(df['trend_score'] > 0, 1.0, np.where(df['trend_score'] < 0, 1.0, 0.0))
    # raw confidence (0..1)
    df['conf_raw'] = (0.4 * df['conf_trend'] + 0.2 * df['momentum_flag'] + 0.2 * df['big_player'] + 0.2 * df['vol_sig'])
    df['conf_raw'] = df['conf_raw'].clip(0, 1)

    return df


def generate_signals(df):
    df = df.copy()
    # trend and pullback
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

    extra_ok = df['momentum_flag'].astype(bool) | (~df['squeeze_on']) | (df['Volume'] > df['vol_ma20'] * 0.6)

    df['signal'] = 0
    df['signal_reason'] = ''
    df.loc[long_cond & extra_ok, 'signal'] = 1
    df.loc[short_cond & extra_ok, 'signal'] = -1
    df.loc[long_cond & extra_ok, 'signal_reason'] = 'Trend+Pullback to EMA20'
    df.loc[short_cond & extra_ok, 'signal_reason'] = 'Trend+Pullback to EMA20'

    # attach confidence
    df['confidence'] = df['conf_raw']
    return df

# ------------------------------
# Backtest engine that builds a detailed trade log
# ------------------------------

def backtest(df, capital=100000, risk_pct=1.0, max_hold=20, atr_mult=2.0, rr_target=2.0):
    trades = []
    cash = capital
    equity_curve = [capital]

    for i in range(len(df)):
        # always append equity after processing bar i (initial equity already present at index 0)
        # but we append later after potential trade profit so for alignment we append current cash now
        equity_curve.append(cash)

        sig = int(df['signal'].iat[i])
        if sig == 0:
            continue
        if i + 1 >= len(df):
            # no next bar to enter
            continue

        entry_idx = i + 1
        entry_time = df.index[entry_idx]
        entry_price = df['Open'].iat[entry_idx]
        atr_val = df['atr14'].iat[entry_idx] if not np.isnan(df['atr14'].iat[entry_idx]) else 0
        if atr_val == 0:
            atr_val = df['Close'].diff().abs().rolling(14).mean().iat[entry_idx]

        # compute stop and target depending on side
        if sig == 1:
            stop = entry_price - atr_mult * atr_val
            if stop <= 0:
                stop = entry_price * 0.98
            risk_per_share = entry_price - stop
            if risk_per_share <= 0:
                continue
            risk_amount = capital * (risk_pct / 100.0)
            qty = max(1, int(risk_amount / risk_per_share))
            target = entry_price + rr_target * (entry_price - stop)

            # scan forward for hit
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

            pnl = (exit_price - entry_price) * qty
            points = (exit_price - entry_price)
            cash += pnl
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.index[exit_idx],
                'side': 'LONG',
                'entry': entry_price,
                'target': target,
                'stop': stop,
                'exit': exit_price,
                'qty': qty,
                'pnl': pnl,
                'points': points,
                'pnl_pct': pnl / capital if capital != 0 else 0,
                'reason': df['signal_reason'].iat[i],
                'confidence': float(df['confidence'].iat[i]),
                'win': pnl > 0
            })
            # update latest appended equity to reflect trade outcome
            equity_curve[-1] = cash

        elif sig == -1:
            stop = entry_price + atr_mult * atr_val
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

            pnl = (entry_price - exit_price) * qty
            points = (entry_price - exit_price)
            cash += pnl
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.index[exit_idx],
                'side': 'SHORT',
                'entry': entry_price,
                'target': target,
                'stop': stop,
                'exit': exit_price,
                'qty': qty,
                'pnl': pnl,
                'points': points,
                'pnl_pct': pnl / capital if capital != 0 else 0,
                'reason': df['signal_reason'].iat[i],
                'confidence': float(df['confidence'].iat[i]),
                'win': pnl > 0
            })
            equity_curve[-1] = cash

    # Align equity series length with df.index
    if len(equity_curve) == len(df) + 1:
        equity_series = pd.Series(equity_curve[1:], index=df.index)
    elif len(equity_curve) == len(df):
        equity_series = pd.Series(equity_curve, index=df.index)
    else:
        equity_series = pd.Series(equity_curve[-len(df):], index=df.index)

    trades_df = pd.DataFrame(trades)

    # Summary statistics
    if not trades_df.empty:
        total_trades = len(trades_df)
        wins = trades_df[trades_df['win']]
        losses = trades_df[~trades_df['win']]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        net_pnl = trades_df['pnl'].sum()
        total_points = trades_df['points'].sum()
        avg_win = wins['pnl'].mean() if not wins.empty else 0
        avg_loss = losses['pnl'].mean() if not losses.empty else 0
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss)
        # max drawdown from equity_series
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min()
    else:
        total_trades = 0
        win_rate = 0
        net_pnl = 0
        total_points = 0
        expectancy = 0
        max_dd = 0

    stats = {
        'total_trades': total_trades,
        'wins': len(wins) if not trades_df.empty else 0,
        'losses': len(losses) if not trades_df.empty else 0,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'total_points': total_points,
        'expectancy': expectancy,
        'max_drawdown': max_dd
    }

    return trades_df, equity_series, stats

# ------------------------------
# Build live trade suggestion (same format as backtest trades)
# Uses last closed candle as basis; entry is set to last Close (you can change to next open)
# ------------------------------

def build_live_trade(last_row, capital, risk_pct, atr_mult, rr_target):
    # last_row is a Series (last closed candle)
    side = int(last_row.get('signal', 0))
    if side == 0:
        return None
    entry_price = float(last_row['Close'])  # using last close as current price/entry basis
    atr_val = float(last_row.get('atr14', 0))
    if atr_val == 0:
        atr_val = np.nan
    if side == 1:
        stop = entry_price - atr_mult * atr_val if not np.isnan(atr_val) else entry_price * 0.98
        risk_per_share = entry_price - stop
        qty = max(1, int((capital * (risk_pct / 100.0)) / risk_per_share)) if risk_per_share > 0 else 1
        target = entry_price + rr_target * (entry_price - stop)
        reason = last_row.get('signal_reason', '')
    else:
        stop = entry_price + atr_mult * atr_val if not np.isnan(atr_val) else entry_price * 1.02
        risk_per_share = stop - entry_price
        qty = max(1, int((capital * (risk_pct / 100.0)) / risk_per_share)) if risk_per_share > 0 else 1
        target = entry_price - rr_target * (stop - entry_price)
        reason = last_row.get('signal_reason', '')

    conf = float(last_row.get('confidence', 0.0))

    trade = {
        'side': 'LONG' if side == 1 else 'SHORT',
        'entry_time': last_row.name,
        'entry': entry_price,
        'target': target,
        'stop': stop,
        'qty': qty,
        'reason': reason,
        'confidence': conf
    }
    return trade

# ------------------------------
# Main: file upload, mapping, compute, backtest, live recommendation
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

    date_format = st.text_input("Date format (optional, strptime style). Leave blank to auto-parse.", value="")

    if not date_col or not open_col or not high_col or not low_col or not close_col:
        st.warning("Please map at least Date, Open, High, Low and Close columns.")
        st.stop()

    working = raw[[date_col, open_col, high_col, low_col, close_col]].copy()
    if vol_col:
        working['Volume'] = raw[vol_col].fillna(0)
    else:
        working['Volume'] = 0

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
            df = generate_signals(df)
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
        bigp = df[df['big_player'] == 1]
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
        st.subheader("Backtest (detailed trade log)")
        trades_df, equity, stats = backtest(df, capital=starting_capital, risk_pct=risk_percent, max_hold=max_hold_bars, atr_mult=atr_mult_sl, rr_target=rr_target)

        if trades_df.empty:
            st.warning("No trades generated with these parameters. Consider loosening conditions or changing timeframe/data sampling.")
        else:
            # Show trade log with relevant columns
            display_cols = ['entry_time','exit_time','side','entry','target','stop','exit','qty','points','pnl','pnl_pct','reason','confidence','win']
            st.dataframe(trades_df[display_cols].sort_values('entry_time').reset_index(drop=True))

            st.subheader("Summary statistics")
            st.write(f"Total trades: {stats['total_trades']}")
            st.write(f"Wins: {stats['wins']}, Losses: {stats['losses']}")
            st.write(f"Win rate: {stats['win_rate']*100:.2f}%")
            st.write(f"Net P&L (INR): {stats['net_pnl']:.2f}")
            st.write(f"Total points: {stats['total_points']:.2f}")
            st.write(f"Expectancy (avg pnl per trade): {stats['expectancy']:.2f}")
            st.write(f"Max drawdown (fraction): {stats['max_drawdown']:.2%}")

            # equity curve
            st.subheader("Equity Curve")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            equity.plot(ax=ax3)
            ax3.set_ylabel('Equity (INR)')
            st.pyplot(fig3)

            # export
            st.markdown("**Export trade log**")
            csv = trades_df.to_csv(index=False).encode()
            st.download_button(label="Download trades CSV", data=csv, file_name='trade_log.csv', mime='text/csv')

        # Smart insights
        st.subheader("Smart Trading Insights (recent)")
        insights = []
        recent = df.iloc[-30:]
        if recent['momentum_flag'].any():
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
        st.subheader("Live Recommendation (structured trade-like output)")
        if st.button("Get Live Recommendation"):
            last = df.iloc[-1]
            live_trade = build_live_trade(last, capital=starting_capital, risk_pct=risk_percent, atr_mult=atr_mult_sl, rr_target=rr_target)
            if live_trade is None:
                st.warning("No actionable signal on the last candle.")
            else:
                st.success(f"Live suggestion — {live_trade['side']} — Confidence: {live_trade['confidence']*100:.0f}%")
                st.write("Entry time:", live_trade['entry_time'])
                st.write("Entry price:", live_trade['entry'])
                st.write("Target:", live_trade['target'])
                st.write("Stop:", live_trade['stop'])
                st.write("Qty (by risk%):", live_trade['qty'])
                st.write("Reason:", live_trade['reason'])
                # offer to export as CSV-like single-row for quick order entry
                live_df = pd.DataFrame([live_trade])
                st.download_button(label="Download live trade (CSV)", data=live_df.to_csv(index=False).encode(), file_name='live_trade.csv', mime='text/csv')

        st.write("---")
        st.write("**How to use**: Upload your intraday/daily data, map columns, click Compute & Run. Use Download buttons to export trades or live suggestion for manual order entry or broker integration.")

    else:
        st.info("Map columns and click 'Compute & Run' in the sidebar to compute indicators, signals, and run backtest.")

# EOF
