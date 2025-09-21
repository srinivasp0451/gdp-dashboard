# streamlit_swing_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pytz
import random
from itertools import product

st.set_page_config(page_title="Swing Trading Recommender — Fixed", layout="wide")

# ---------------------------
# Utilities: column mapping, timezone, basic indicators
# ---------------------------
def map_columns(df):
    cols_lower = {c: c.lower() for c in df.columns}
    df_col_lower = {c.lower(): c for c in df.columns}
    mapping = {}
    def find_candidate(key_words):
        for k, orig in df_col_lower.items():
            for kw in key_words:
                if kw in k:
                    return orig
        return None

    mapping['date'] = find_candidate(['date','time','timestamp','dt'])
    mapping['open']  = find_candidate(['open','o','openprice','open_price'])
    mapping['high']  = find_candidate(['high','h','highprice','high_price'])
    mapping['low']   = find_candidate(['low','l','lowprice','low_price'])
    mapping['close'] = find_candidate(['close','c','closeprice','close_price','last','price'])
    mapping['volume']= find_candidate(['volume','vol','quantity','qty','shares','turnover'])

    # last-resort fallbacks by position
    cols_list = list(df.columns)
    fallback = {'date':0,'open':1,'high':2,'low':3,'close':4,'volume':5}
    for k, pos in fallback.items():
        if mapping.get(k) is None and len(cols_list) > pos:
            mapping[k] = cols_list[pos]

    mapped = pd.DataFrame()
    for k in ['date','open','high','low','close','volume']:
        if mapping.get(k) in df.columns:
            mapped[k] = df[mapping[k]]
        else:
            mapped[k] = np.nan
    return mapped, mapping

def convert_to_datetime_index(df, date_col_values):
    """
    date_col_values: Series (original date column)
    returns df with index localized to Asia/Kolkata if possible
    """
    s = pd.to_datetime(date_col_values, errors='coerce', infer_datetime_format=True)
    # if entirely NaT try fallback parsing
    if s.isna().all():
        s = pd.to_datetime(date_col_values.astype(str), errors='coerce')
    # handle tz-aware vs naive: attempt to preserve tz if present else assume UTC then convert to IST
    try:
        if getattr(s.dt, 'tz', None) is None:
            s = s.dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
        s = s.dt.tz_convert('Asia/Kolkata')
    except Exception:
        # fallback: naive -> localize to Asia/Kolkata
        try:
            s = s.dt.tz_localize('Asia/Kolkata')
        except Exception:
            # last resort: naive timezone-unaware index
            s = pd.to_datetime(s)
    df = df.copy()
    df['date_ist'] = s
    df = df.set_index('date_ist')
    return df

def clean_numeric_columns(df):
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def compute_true_atr(df, period=14):
    # True Range
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr.fillna(method='bfill')

def add_basic_indicators(df):
    df = df.copy()
    df['ret'] = df['close'].pct_change().fillna(0)
    df['sma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma50'] = df['close'].rolling(50, min_periods=1).mean()
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['atr14'] = compute_true_atr(df, period=14)
    return df

# ---------------------------
# Pattern detectors (causal: only use data up to candle i)
# return positions (integer indices) of signals
# ---------------------------

def detect_double_top_bottom_pos(df_close, lookback=30, tolerance=0.02):
    pos_list = []
    L = len(df_close)
    for i in range(lookback, L):
        # window includes current candle (up to i)
        window = df_close.iloc[i-lookback+1:i+1]  # inclusive of i
        peak = window.max()
        trough = window.min()
        # count peaks near peak value
        near_peaks = window[window >= peak*(1 - tolerance)]
        if len(near_peaks) >= 2:
            # ensure after second peak there's a drop compared to peak
            if df_close.iloc[i] < peak*(1 - 0.005):
                pos_list.append((i, 'double_top'))
        near_bottoms = window[window <= trough*(1 + tolerance)]
        if len(near_bottoms) >= 2:
            if df_close.iloc[i] > trough*(1 + 0.005):
                pos_list.append((i, 'double_bottom'))
    return pos_list

def detect_head_shoulder_pos(df_close, lookback=60):
    pos_list = []
    L = len(df_close)
    for i in range(lookback, L):
        window = df_close.iloc[i-lookback+1:i+1]
        if len(window) < 10:
            continue
        # heuristic: head significantly higher than shoulders
        peak_val = window.max()
        peak_idx = window.idxmax()
        # split relative to peak
        pre = window[:window.index.get_loc(peak_idx)]
        post = window[window.index.get_loc(peak_idx)+1:]
        if len(pre) < 2 or len(post) < 2:
            continue
        left_peak = pre.max()
        right_peak = post.max() if not post.empty else -999999
        if left_peak < peak_val and right_peak < peak_val and abs(left_peak-right_peak)/max(left_peak, right_peak+1e-9) < 0.12:
            pos_list.append((i, 'head_shoulder'))
    return pos_list

def detect_triangle_pos(df_close, lookback=50):
    pos_list = []
    L = len(df_close)
    for i in range(lookback, L):
        window = df_close.iloc[i-lookback+1:i+1]
        x = np.arange(len(window))
        highs = window.rolling(5, min_periods=1).max().dropna()
        lows = window.rolling(5, min_periods=1).min().dropna()
        if len(highs) < 8 or len(lows) < 8:
            continue
        try:
            m_high, _ = np.polyfit(x[-len(highs):], highs.values, 1)
            m_low, _ = np.polyfit(x[-len(lows):], lows.values, 1)
            if m_high * m_low < 0 or (abs(m_high) < 0.0005 and abs(m_low) > 0.0005):
                pos_list.append((i, 'triangle_or_wedge'))
        except Exception:
            continue
    return pos_list

def detect_cup_handle_pos(df_close, lookback=200):
    pos_list = []
    L = len(df_close)
    for i in range(lookback, L):
        window = df_close.iloc[i-lookback+1:i+1]
        left = window[:len(window)//2]
        right = window[len(window)//2:]
        if len(left)==0 or len(right)==0:
            continue
        # a very rough curvature heuristic
        if left.min() < left.mean() and right.min() < right.mean():
            # small handle (final dip)
            if window.iloc[-1] < window.iloc[-5] * 1.03:
                pos_list.append((i, 'cup_handle'))
    return pos_list

def aggregate_patterns_positions(df):
    c = df['close']
    sigs = []
    sigs += detect_double_top_bottom_pos(c, lookback=30)
    sigs += detect_head_shoulder_pos(c, lookback=60)
    sigs += detect_triangle_pos(c, lookback=50)
    sigs += detect_cup_handle_pos(c, lookback=200)
    # aggregate per integer position
    agg = {}
    for pos, name in sigs:
        agg.setdefault(pos, []).append(name)
    return agg  # keys: integer pos -> list names

# ---------------------------
# Candle psych helpers (causal)
# ---------------------------
def is_bullish_bearish_candle_causal(df, pos):
    # use only data at pos and previous candle
    if pos < 0 or pos >= len(df):
        return None
    o = df['open'].iloc[pos]; c = df['close'].iloc[pos]; h = df['high'].iloc[pos]; l = df['low'].iloc[pos]
    body = abs(c-o); total = max(h-l, 1e-9)
    # hammer
    if (body/total) < 0.35 and (c - l)/total > 0.45 and c > o:
        return 'bull'
    if (body/total) < 0.35 and (h - c)/total > 0.45 and c < o:
        return 'bear'
    if pos > 0:
        prev_o = df['open'].iloc[pos-1]; prev_c = df['close'].iloc[pos-1]
        if c > o and prev_c < prev_o and (c - o) > (prev_o - prev_c):
            return 'bull'
        if c < o and prev_c > prev_o and (o - c) > (prev_c - prev_o):
            return 'bear'
    return None

# ---------------------------
# Signal generation (causal)
# ---------------------------
def generate_signals(df, params):
    """
    Use only data up to candle i to decide if a signal is present at i (entry on close i).
    Returns list of signals with 'pos' integer, side, price (close@pos), reason, prob
    """
    df = df.copy()
    # indicators already present (sma20, sma50, vol_ma20, atr14)
    # use shifted versions where needed to avoid peeking: e.g., vol_ma20_shift = vol_ma20.shift(0) is okay because vol_ma20 uses past+current
    pattern_dict = aggregate_patterns_positions(df)
    signals = []
    L = len(df)
    for i in range(0, L):
        reason_parts = []
        prob = 0.45  # baseline low
        side = None
        price = df['close'].iloc[i]
        # pattern vote
        if i in pattern_dict:
            pats = pattern_dict[i]
            if any('double_top' in p or 'head_shoulder' in p for p in pats):
                side = 'short'; reason_parts.append("Reversal pattern: "+",".join(pats)); prob += 0.18
            elif any('double_bottom' in p or 'cup_handle' in p for p in pats):
                side = 'long'; reason_parts.append("Reversal/accumulation: "+",".join(pats)); prob += 0.15
            else:
                # triangle/wedge, use trend
                if df['sma20'].iloc[i] > df['sma50'].iloc[i]:
                    side = 'long'; reason_parts.append("Triangle in uptrend"); prob += 0.07
                else:
                    side = 'short'; reason_parts.append("Triangle in downtrend"); prob += 0.07

        # volume: use previous vol_ma20 (conservative)
        vol_ma_prev = df['vol_ma20'].iloc[i-1] if i-1>=0 else df['vol_ma20'].iloc[i]
        if df['volume'].iloc[i] > (vol_ma_prev * params.get('vol_mult', 2.0)):
            prob += 0.06
            reason_parts.append("Volume spike vs prior 20")

        # candle psychology (causal)
        cb = is_bullish_bearish_candle_causal(df, i)
        if cb == 'bull':
            prob += 0.04
            reason_parts.append("Bullish candle shape")
            side = side or 'long'
        elif cb == 'bear':
            prob += 0.04
            reason_parts.append("Bearish candle shape")
            side = side or 'short'

        # trend SMA confluence (current candle)
        if df['sma20'].iloc[i] > df['sma50'].iloc[i]:
            prob += 0.02
            reason_parts.append("SMA20>50")
        else:
            prob -= 0.01

        # final threshold
        if prob >= params.get('min_prob', 0.55):
            signals.append({
                'pos': i,
                'index': df.index[i],
                'side': side if side else 'long',
                'price': price,
                'reason': "; ".join(reason_parts) if reason_parts else "Pattern/price action",
                'prob': round(min(prob, 0.95), 3)
            })
    # deduplicate by pos (keep highest prob if duplicates)
    sig_by_pos = {}
    for s in signals:
        p = s['pos']
        if p not in sig_by_pos or s['prob'] > sig_by_pos[p]['prob']:
            sig_by_pos[p] = s
    final = [sig_by_pos[p] for p in sorted(sig_by_pos.keys())]
    return final

# ---------------------------
# Backtester (strict no future leak)
# ---------------------------
def backtest_with_params(df, params, mode='both', end_pos=None):
    """
    df: must be sorted ascending.
    end_pos: integer index position (inclusive). If None use full df.
    Entries happen at close of signal candle (pos). Exits are searched in subsequent candles using intrabar high/low. 
    If both TP and SL touched within the same candle, use heuristic based on that candle's open to decide sequence; conservative fallback: SL first.
    """
    df = df.copy().reset_index(drop=False)
    # compute ATR if missing
    if 'atr14' not in df.columns:
        df['atr14'] = compute_true_atr(df.set_index('date_ist'), period=14).values
    signals = generate_signals(df.set_index('date_ist'), params)
    trades = []
    taken_until_pos = -1
    L = len(df)
    last_allowed_pos = end_pos if end_pos is not None else L-1

    for sig in signals:
        pos = int(sig['pos'])
        if pos <= taken_until_pos:
            continue
        if pos > last_allowed_pos:
            continue
        side = sig['side']
        if mode == 'long' and side != 'long':
            continue
        if mode == 'short' and side != 'short':
            continue
        entry_price = float(df.at[pos, 'close'])
        entry_time = df.at[pos, 'date_ist']
        # sl and tp based on ATR + pct constraints
        atr = float(df.at[pos, 'atr14']) if not pd.isna(df.at[pos, 'atr14']) else max(0.01*entry_price, 1.0)
        sl_pct = float(params.get('sl_pct', 0.02))
        sl_atr_mult = float(params.get('sl_atr_mult', 0.5))
        sl_dist = max(sl_pct * entry_price, sl_atr_mult * atr)
        tp_dist = max(float(params.get('tp_ratio', 2.0)) * sl_dist, float(params.get('min_points', 0.01)) * entry_price)
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        # Walk forward candles from next candle to find first breach
        exit_price = None; exit_time = None; exit_reason = None; exit_pos = None
        max_hold = int(params.get('max_hold', 20))
        for j in range(pos+1, min(L, pos+1+max_hold)):
            high = float(df.at[j,'high']); low = float(df.at[j,'low']); open_j = float(df.at[j,'open']); close_j = float(df.at[j,'close'])
            # For LONG
            if side == 'long':
                tp_hit = high >= tp
                sl_hit = low <= sl
                if tp_hit and sl_hit:
                    # both hit in same candle. infer which hit first using open price if possible
                    if open_j >= tp:
                        exit_price = tp; exit_reason = 'TP_same_candle_via_open'
                    elif open_j <= sl:
                        exit_price = sl; exit_reason = 'SL_same_candle_via_open'
                    else:
                        # conservative assume SL first
                        exit_price = sl; exit_reason = 'SL_same_candle_conservative'
                    exit_time = df.at[j, 'date_ist']; exit_pos = j
                    break
                elif tp_hit:
                    exit_price = tp; exit_reason = 'TP'
                    exit_time = df.at[j, 'date_ist']; exit_pos = j
                    break
                elif sl_hit:
                    exit_price = sl; exit_reason = 'SL'
                    exit_time = df.at[j, 'date_ist']; exit_pos = j
                    break
            else:  # SHORT
                tp_hit = low <= tp
                sl_hit = high >= sl
                if tp_hit and sl_hit:
                    if open_j <= tp:
                        exit_price = tp; exit_reason = 'TP_same_candle_via_open'
                    elif open_j >= sl:
                        exit_price = sl; exit_reason = 'SL_same_candle_via_open'
                    else:
                        exit_price = sl; exit_reason = 'SL_same_candle_conservative'
                    exit_time = df.at[j, 'date_ist']; exit_pos = j
                    break
                elif tp_hit:
                    exit_price = tp; exit_reason = 'TP'
                    exit_time = df.at[j, 'date_ist']; exit_pos = j
                    break
                elif sl_hit:
                    exit_price = sl; exit_reason = 'SL'
                    exit_time = df.at[j, 'date_ist']; exit_pos = j
                    break
        # If nothing hit, exit at close after max_hold or at last_allowed_pos whichever is earlier
        if exit_price is None:
            final_pos = min(pos+max_hold, last_allowed_pos)
            exit_price = float(df.at[final_pos, 'close'])
            exit_time = df.at[final_pos, 'date_ist']
            exit_reason = 'TIMEOUT'
            exit_pos = final_pos

        # pnl sign correct
        pnl = (exit_price - entry_price) if side == 'long' else (entry_price - exit_price)
        trades.append({
            'entry_pos': pos, 'exit_pos': exit_pos,
            'entry_time': entry_time, 'exit_time': exit_time,
            'side': side, 'entry': entry_price, 'exit': exit_price,
            'tp': tp, 'sl': sl, 'reason': sig.get('reason',''),
            'exit_reason': exit_reason, 'prob': sig.get('prob', 0.5),
            'pnl': pnl, 'duration_candles': exit_pos - pos
        })
        taken_until_pos = exit_pos

    tr_df = pd.DataFrame(trades)
    if tr_df.empty:
        metrics = {'net_pnl':0.0, 'total_trades':0, 'win_rate':0.0, 'avg_points':0.0, 'positive_trades':0, 'negative_trades':0}
    else:
        net = tr_df['pnl'].sum()
        total = len(tr_df)
        pos_count = (tr_df['pnl'] > 0).sum()
        neg_count = (tr_df['pnl'] <= 0).sum()
        win_rate = pos_count / total if total>0 else 0.0
        avg_pts = tr_df['pnl'].mean()
        metrics = {'net_pnl': float(net), 'total_trades': int(total), 'win_rate': float(win_rate),
                   'avg_points': float(avg_pts), 'positive_trades': int(pos_count), 'negative_trades': int(neg_count)}
    return tr_df, metrics

# ---------------------------
# Optimization
# ---------------------------
def random_search(df, param_space, n_iter, mode, end_pos, target_accuracy=0.6, min_points=0.0, prog=None):
    best = None; best_score = -1e12; results=[]
    for it in range(n_iter):
        params = {}
        for k, v in param_space.items():
            if isinstance(v, list):
                params[k] = random.choice(v)
            elif isinstance(v, tuple) and len(v)==2:
                params[k] = random.uniform(v[0], v[1])
            else:
                params[k] = v
        trades, metrics = backtest_with_params(df, params, mode=mode, end_pos=end_pos)
        score = metrics.get('net_pnl',0) + 1000*(metrics.get('win_rate',0) - target_accuracy)
        if metrics.get('avg_points',0) < min_points:
            score -= abs(min_points - metrics.get('avg_points',0)) * 100
        results.append((params, metrics, trades, score))
        if score > best_score:
            best_score = score
            best = (params, metrics, trades)
        if prog:
            prog.progress(int((it+1)/n_iter*100))
    return best, results

def grid_search(df, param_grid, mode, end_pos, target_accuracy=0.6, min_points=0.0, prog=None):
    keys = list(param_grid.keys())
    combos = list(product(*(param_grid[k] for k in keys)))
    best=None; best_score=-1e12; results=[]
    for idx, combo in enumerate(combos):
        params = {keys[i]: combo[i] for i in range(len(keys))}
        trades, metrics = backtest_with_params(df, params, mode=mode, end_pos=end_pos)
        score = metrics.get('net_pnl',0) + 1000*(metrics.get('win_rate',0) - target_accuracy)
        if metrics.get('avg_points',0) < min_points:
            score -= abs(min_points - metrics.get('avg_points',0)) * 100
        results.append((params, metrics, trades, score))
        if score > best_score:
            best_score = score
            best = (params, metrics, trades)
        if prog:
            prog.progress(int((idx+1)/len(combos)*100))
    return best, results

# ---------------------------
# Presentation helpers
# ---------------------------
def human_summary(df):
    if df.empty:
        return "No data."
    period_days = (df.index.max() - df.index.min()).days
    mean_return = df['ret'].mean() * 252
    vol = df['ret'].std() * np.sqrt(252)
    up = (df['ret']>0).mean()
    trend = 'bullish' if df['sma20'].iloc[-1] > df['sma50'].iloc[-1] else 'bearish'
    s = (f"This dataset spans {period_days} days from {df.index.min()} to {df.index.max()}. "
         f"Annualized mean return ~{mean_return:.2%} with volatility ~{vol:.2%}. "
         f"Positive days {up:.1%}. Current structure: {trend}. "
         "System will detect patterns, supply/demand, and optimize entries with strict no-future-leak rules.")
    return s

def trade_reason_text(tr):
    return (f"{tr['side'].upper()} entry {tr['entry']:.2f} @ {tr['entry_time']} -> exit {tr['exit']:.2f} @ {tr['exit_time']} "
            f"({tr['exit_reason']}). TP {tr['tp']:.2f} SL {tr['sl']:.2f}. PnL {tr['pnl']:.4f}. Logic: {tr['reason']}. Prob {tr['prob']}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Swing Trading Recommender — FIXED (No lookahead; entry on candle close)")

st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV or Excel", type=['csv','xlsx','xls'])
mode = st.sidebar.selectbox("Direction", ['both','long','short'])
search_method = st.sidebar.selectbox("Optimization Method", ['random_search','grid_search'])
n_iter = st.sidebar.number_input("Random search iterations", min_value=5, max_value=2000, value=60, step=5)
grid_limit = st.sidebar.number_input("Grid combos limit", min_value=1, max_value=2000, value=300)
desired_accuracy = st.sidebar.slider("Target minimum win rate", 0.0, 1.0, 0.65)
min_points = st.sidebar.number_input("Min average points/trade (fraction)", value=0.005, step=0.001, format="%.4f")

if uploaded_file is None:
    st.info("Upload OHLCV file (CSV / Excel). The app will map columns automatically.")
    st.stop()

# read file
try:
    if uploaded_file.name.lower().endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

mapped_df, mapping = map_columns(df_raw)
mapped_df['date_orig'] = df_raw[mapping.get('date')] if mapping.get('date') in df_raw.columns else df_raw.iloc[:,0]
mapped_df = clean_numeric_columns(mapped_df)

# convert to datetime index IST
try:
    df_idxed = convert_to_datetime_index(mapped_df, mapped_df['date_orig'])
except Exception as e:
    st.warning("Date parsing trouble: attempting simpler parse.")
    mapped_df['date_orig'] = pd.to_datetime(mapped_df['date_orig'], errors='coerce')
    df_idxed = mapped_df.set_index('date_orig')
df_idxed = df_idxed.sort_index(ascending=True)
df_idxed = clean_numeric_columns(df_idxed)
df_idxed = add_basic_indicators(df_idxed)

st.subheader("Data Preview & Mapping")
col1, col2 = st.columns([1,1])
with col1:
    st.write("Detected mapping:", mapping)
    st.write("Top 5:")
    st.write(df_idxed.head(5))
with col2:
    st.write("Bottom 5:")
    st.write(df_idxed.tail(5))
st.write("Date range:", df_idxed.index.min(), "to", df_idxed.index.max())
st.write("Price min/max:", df_idxed['close'].min(), df_idxed['close'].max())

# end-date control (user can simulate backtest stopping at a date)
end_default = df_idxed.index.max()
end_date = st.sidebar.date_input("Select backtest end date (inclusive)", value=end_default.date(), min_value=df_idxed.index.min().date(), max_value=df_idxed.index.max().date())
end_dt = pd.to_datetime(str(end_date)).tz_localize('Asia/Kolkata') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
# convert end_dt to position index
end_pos = df_idxed.reset_index().index[df_idxed.reset_index()['date_ist'] <= end_dt][-1] if not df_idxed.reset_index()[df_idxed.reset_index()['date_ist'] <= end_dt].empty else len(df_idxed)-1

st.subheader("Candlestick (last N candles)")
last_n = st.slider("Candles to show", 50, min(2000, len(df_idxed)), value=min(300, len(df_idxed)))
plot_df = df_idxed.iloc[-last_n:].reset_index()
fig = go.Figure(data=[go.Candlestick(x=plot_df['date_ist'], open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'])])
fig.update_layout(height=450, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Returns heatmap (Year vs Month)")
df_idxed['year'] = df_idxed.index.year
df_idxed['month'] = df_idxed.index.month
monthly = df_idxed.groupby(['year','month'])['ret'].sum().unstack(level=0).fillna(0).T
fig2, ax2 = plt.subplots(figsize=(10,3))
sns.heatmap(monthly, annot=True, fmt=".2f", cmap='RdYlGn', ax=ax2)
st.pyplot(fig2)

st.subheader("100-word summary")
st.write(human_summary(df_idxed))

# parameter search space
st.sidebar.header("Optimizer parameter ranges")
param_space = {
    'sl_pct': (0.004, 0.04),
    'sl_atr_mult': (0.2, 1.2),
    'tp_ratio': (1.0, 3.5),
    'min_prob': (0.52, 0.8),
    'vol_mult': [1.5, 2.0, 2.5],
    'sr_lookback': [5,10,20],
    'max_hold': [5,10,20,40],
    'min_points': (min_points, min_points+0.02)
}
st.sidebar.write("Default ranges set. Random search samples continuous ranges; grid discretizes.")

run_button = st.button("Run Optimization & Backtest")
progress_placeholder = st.sidebar.empty()

if run_button:
    prog = progress_placeholder.progress(0)
    if search_method == 'random_search':
        best, results = random_search(df_idxed, param_space, n_iter, mode, end_pos=end_pos, target_accuracy=desired_accuracy, min_points=min_points, prog=prog)
    else:
        # discretize continuous ranges into 3 points each
        param_grid = {}
        for k,v in param_space.items():
            if isinstance(v, tuple) and len(v)==2:
                param_grid[k] = list(np.linspace(v[0], v[1], 3))
            elif isinstance(v, list):
                param_grid[k] = v
            else:
                param_grid[k] = [v]
        combos = 1
        for k in param_grid: combos *= len(param_grid[k])
        if combos > grid_limit:
            st.error(f"Grid combos {combos} exceeds limit {grid_limit}. Reduce grid or increase limit.")
            prog.empty()
            st.stop()
        best, results = grid_search(df_idxed, param_grid, mode, end_pos=end_pos, target_accuracy=desired_accuracy, min_points=min_points, prog=prog)
    prog.empty()
    if not best:
        st.warning("No valid strategy found.")
        st.stop()
    best_params, best_metrics, best_trades = best
    st.success("Optimization completed — best strategy selected")
    st.subheader("Best parameters")
    st.json(best_params)
    st.subheader("Backtest metrics")
    st.metric("Net PnL (price units)", f"{best_metrics.get('net_pnl',0):.4f}")
    st.metric("Total trades", best_metrics.get('total_trades',0))
    st.metric("Win rate", f"{best_metrics.get('win_rate',0):.2%}")
    st.metric("Avg points/trade", f"{best_metrics.get('avg_points',0):.6f}")

    if not best_trades.empty:
        st.subheader("Trade log (most recent 200)")
        display_df = best_trades[['entry_time','exit_time','side','entry','exit','tp','sl','pnl','exit_reason','reason','prob','duration_candles']].sort_values('entry_time')
        st.dataframe(display_df.tail(200))
        st.subheader("Trade explanations (last 20)")
        for idx, tr in display_df.tail(20).iterrows():
            st.markdown(f"**Trade {idx+1}** — {tr['side'].upper()} | Entry {tr['entry']:.2f} | Exit {tr['exit']:.2f} | PnL {tr['pnl']:.4f}")
            st.write(trade_reason_text(tr))
    else:
        st.write("No trades executed with these parameters.")

    st.subheader("Backtest narrative")
    st.write((f"Backtest until {end_dt} yielded net PnL {best_metrics.get('net_pnl',0):.4f} across {best_metrics.get('total_trades',0)} trades. "
              f"Win rate {best_metrics.get('win_rate',0):.2%}. Strategy params: {best_params}. "
              "Entries are always on the close of the signal candle; exits respect intrabar TP/SL using high/low and conservative same-candle handling."))

    st.subheader("Live recommendation (based on last candle close & best strategy)")
    # find signal at last candle position if exists
    last_pos = len(df_idxed)-1
    # generate signals on full df but we will pick the one with pos == last_pos
    live_signals = generate_signals(df_idxed, best_params)
    cand = None
    for s in reversed(live_signals):
        if s['pos'] == last_pos:
            cand = s; break
    if cand is None:
        # pick latest signal before last but not after end_pos
        recent = [s for s in live_signals if s['pos'] <= last_pos]
        cand = recent[-1] if recent else None

    if cand is None:
        st.info("No live signal found in recent data using best strategy.")
    else:
        entry_price = float(df_idxed['close'].iloc[last_pos])
        atr = float(df_idxed['atr14'].iloc[last_pos]) if 'atr14' in df_idxed.columns else 0.0
        sl_dist = max(best_params.get('sl_pct',0.02)*entry_price, best_params.get('sl_atr_mult',0.5)*atr)
        tp_dist = max(best_params.get('tp_ratio',2.0)*sl_dist, best_params.get('min_points',0.01)*entry_price)
        if cand['side']=='long':
            sl = entry_price - sl_dist; tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist; tp = entry_price - tp_dist
        st.markdown(f"**{cand['side'].upper()}** entry at **{entry_price:.4f}** (close of last candle)")
        st.markdown(f"- Target (TP): **{tp:.4f}**")
        st.markdown(f"- Stop Loss (SL): **{sl:.4f}**")
        st.markdown(f"- Reason: {cand['reason']}")
        st.markdown(f"- Estimated prob: {cand['prob']:.2%}")
        st.markdown("**Strict rule**: enter on close of the last candle. No use of next candle open.")

st.sidebar.markdown("---")
st.sidebar.write("Fixes included: causal pattern detection, ATR-based SL, intrabar handling, deduplication, and conservative same-candle rules.")
