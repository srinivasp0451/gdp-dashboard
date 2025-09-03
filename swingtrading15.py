import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import random
from datetime import datetime

# ------------------- Helpers / Normalization -------------------

def normalize_df(df):
    # normalize common column names to Open/High/Low/Close/Volume/Date
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    for key, orig in cols.items():
        if key in ("open"):
            mapping[orig] = "Open"
        if key in ("high"):
            mapping[orig] = "High"
        if key in ("low"):
            mapping[orig] = "Low"
        if key in ("close"):
            mapping[orig] = "Close"
        if key in ("volume", "vol"):
            mapping[orig] = "Volume"
        if key in ("date", "datetime", "time"):
            mapping[orig] = "Date"
    df = df.rename(columns=mapping)
    # ensure required cols
    for required in ["Open","High","Low","Close"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    # parse date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.set_index("Date").sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    # make sure numeric
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ------------------- Technical Indicators (causal) -------------------

def compute_indicators(df, params):
    """Compute indicators required by signal generator. All calculations are causal (use only past/current data).
    Returns a DataFrame with indicator columns appended.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"].fillna(0)

    # SMA and EMA
    df[f"sma_{params['sma_fast']}"] = close.rolling(params['sma_fast'], min_periods=1).mean()
    df[f"sma_{params['sma_slow']}"] = close.rolling(params['sma_slow'], min_periods=1).mean()
    df[f"ema_{params['ema_fast']}"] = close.ewm(span=params['ema_fast'], adjust=False).mean()
    df[f"ema_{params['ema_slow']}"] = close.ewm(span=params['ema_slow'], adjust=False).mean()

    # MACD
    ema_f = df[f"ema_{params['ema_fast']}"]
    ema_s = df[f"ema_{params['ema_slow']}"]
    macd = ema_f - ema_s
    macd_signal = macd.ewm(span=params.get('macd_signal', 9), adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd - macd_signal

    # RSI (Wilder)
    period = params['rsi_period']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (on sma_fast)
    ma = df[f"sma_{params['sma_fast']}"]
    std = close.rolling(params['sma_fast'], min_periods=1).std()
    df['bb_upper'] = ma + params.get('bb_mult', 2) * std
    df['bb_lower'] = ma - params.get('bb_mult', 2) * std

    # Momentum
    df[f"mom_{params['mom_period']}"] = close - close.shift(params['mom_period'])

    # Stochastic
    stoch_k = ( (close - low.rolling(params['stoch_period'], min_periods=1).min()) /
                (high.rolling(params['stoch_period'], min_periods=1).max() - low.rolling(params['stoch_period'], min_periods=1).min()).replace(0, np.nan) ) * 100
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_k.rolling(3, min_periods=1).mean()

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * vol).fillna(0).cumsum()
    df['obv'] = obv

    # VWMA (volume weighted moving avg) and volume sma
    wsum = (close * vol).rolling(params.get('vwma_period', 14)).sum()
    vsum = vol.rolling(params.get('vwma_period', 14)).sum().replace(0, np.nan)
    df['vwma'] = (wsum / vsum).fillna(df['Close'])
    df['vol_sma'] = vol.rolling(params.get('vol_sma_period', 20), min_periods=1).mean()

    # CCI
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(params['cci_period'], min_periods=1).mean()
    md = tp.rolling(params['cci_period'], min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).replace(0, np.nan)
    df['cci'] = (tp - tp_ma) / (0.015 * md)

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr_{params['atr_period']}"] = tr.rolling(params['atr_period'], min_periods=1).mean()

    # ADX/PDI/MDI approximation (Wilder smoothing)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_for_adx = tr.ewm(alpha=1/params['adx_period'], adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/params['adx_period'], adjust=False).mean() / atr_for_adx)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/params['adx_period'], adjust=False).mean() / atr_for_adx)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df['adx'] = dx.ewm(alpha=1/params['adx_period'], adjust=False).mean()
    df['pdi'] = plus_di
    df['mdi'] = minus_di

    # clean up
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

# ------------------- Signal Generation (confluence voting) -------------------

def generate_confluence_signals(df_local, params, side="Both"):
    df_calc = compute_indicators(df_local, params)
    votes = []
    sig_series = pd.Series(0, index=df_calc.index)

    for idx in df_calc.index:
        row = df_calc.loc[idx]
        indicators_that_long = []
        indicators_that_short = []

        # 1) SMA crossover (fast > slow -> long)
        sma_fast = row[f"sma_{params['sma_fast']}"]
        sma_slow = row[f"sma_{params['sma_slow']}"]
        if not np.isnan(sma_fast) and not np.isnan(sma_slow):
            if sma_fast > sma_slow:
                indicators_that_long.append(f"SMA{params['sma_fast']}>{params['sma_slow']}")
            elif sma_fast < sma_slow:
                indicators_that_short.append(f"SMA{params['sma_fast']}<{params['sma_slow']}")

        # 2) EMA crossover
        ema_f = row[f"ema_{params['ema_fast']}"]
        ema_s = row[f"ema_{params['ema_slow']}"]
        if not np.isnan(ema_f) and not np.isnan(ema_s):
            if ema_f > ema_s:
                indicators_that_long.append(f"EMA{params['ema_fast']}>{params['ema_slow']}")
            elif ema_f < ema_s:
                indicators_that_short.append(f"EMA{params['ema_fast']}<{params['ema_slow']}")

        # 3) MACD histogram positive -> long
        if not np.isnan(row.get("macd_hist", np.nan)):
            if row["macd_hist"] > 0:
                indicators_that_long.append("MACD+")
            elif row["macd_hist"] < 0:
                indicators_that_short.append("MACD-")

        # 4) RSI strength
        rsi_val = row[f"rsi_{params['rsi_period']}"]
        if not np.isnan(rsi_val):
            if rsi_val < params.get('rsi_oversold', 35):
                indicators_that_long.append(f"RSI<{params.get('rsi_oversold', 35)}")
            elif rsi_val > params.get('rsi_overbought', 65):
                indicators_that_short.append(f"RSI>{params.get('rsi_overbought', 65)}")

        # 5) Bollinger
        price = row['Close']
        if not np.isnan(row['bb_upper']) and not np.isnan(row['bb_lower']):
            if price < row['bb_lower']:
                indicators_that_long.append("BB_Lower")
            elif price > row['bb_upper']:
                indicators_that_short.append("BB_Upper")

        # 6) Momentum
        mom = row[f"mom_{params['mom_period']}"]
        if not np.isnan(mom):
            if mom > 0:
                indicators_that_long.append(f"MOM+({params['mom_period']})")
            elif mom < 0:
                indicators_that_short.append(f"MOM-({params['mom_period']})")

        # 7) Stochastic
        if not np.isnan(row['stoch_k']) and not np.isnan(row['stoch_d']):
            if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < params.get('stoch_oversold', 30):
                indicators_that_long.append("STOCH")
            elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > params.get('stoch_overbought', 70):
                indicators_that_short.append("STOCH")

        # 8) ADX direction
        if not np.isnan(row['adx']):
            if row['adx'] > params.get('adx_threshold', 20) and row['pdi'] > row['mdi']:
                indicators_that_long.append("ADX+")
            elif row['adx'] > params.get('adx_threshold', 20) and row['mdi'] > row['pdi']:
                indicators_that_short.append("ADX-")

        # 9) OBV rising
        i = df_calc.index.get_indexer([idx])[0]
        obv_vote_long = False; obv_vote_short = False
        if i >= 3:
            recent = df_calc['obv'].iloc[max(0,i-2):i+1].mean()
            prev = df_calc['obv'].iloc[max(0,i-5):max(0,i-2)].mean()
            if recent > prev:
                obv_vote_long = True
            elif recent < prev:
                obv_vote_short = True
            if obv_vote_long: indicators_that_long.append('OBV')
            if obv_vote_short: indicators_that_short.append('OBV')

        # 10) VWAP/VWMA
        if not np.isnan(row.get('vwma', np.nan)):
            if row['Close'] > row['vwma']:
                indicators_that_long.append('VWMA')
            elif row['Close'] < row['vwma']:
                indicators_that_short.append('VWMA')

        # 11) CCI extremes
        if not np.isnan(row.get('cci', np.nan)):
            if row['cci'] < -100:
                indicators_that_long.append('CCI')
            elif row['cci'] > 100:
                indicators_that_short.append('CCI')

        # 12) Volume spike (simple)
        if not np.isnan(row.get('vol_sma', np.nan)) and row['Volume'] > 0:
            if row['Volume'] > row['vol_sma'] * params.get('vol_multiplier', 1.5):
                # if price closed up on spike, favor long; if closed down, favor short
                prev_close = df_calc['Close'].shift(1).loc[idx] if idx in df_calc.index and df_calc.index.get_loc(idx) > 0 else np.nan
                if not np.isnan(prev_close) and row['Close'] > prev_close:
                    indicators_that_long.append('VOL_SPIKE')
                elif not np.isnan(prev_close) and row['Close'] < prev_close:
                    indicators_that_short.append('VOL_SPIKE')

        # count confluences
        long_votes = len(set(indicators_that_long))
        short_votes = len(set(indicators_that_short))
        total_votes = max(long_votes, short_votes)

        vote_direction = 1 if long_votes > short_votes else (-1 if short_votes > long_votes else 0)
        # side filter
        if side == "Long" and vote_direction == -1:
            vote_direction = 0
            total_votes = 0
        if side == "Short" and vote_direction == 1:
            vote_direction = 0
            total_votes = 0

        final_sig = vote_direction if total_votes >= params['min_confluence'] else 0

        votes.append({
            'index': idx,
            'long_votes': long_votes,
            'short_votes': short_votes,
            'total_votes': total_votes,
            'direction': vote_direction,
            'signal': final_sig,
            'indicators_long': indicators_that_long,
            'indicators_short': indicators_that_short
        })
        sig_series.loc[idx] = final_sig

    votes_df = pd.DataFrame(votes).set_index('index')
    result = df_calc.join(votes_df)
    result['Signal'] = sig_series
    return result

# ------------------- Backtester (entry executed at SAME bar close) -------------------

def choose_primary_indicator(indicators_list):
    # priority for display if multiple indicators exist
    priority = ['EMA','SMA','MACD','RSI','BB','VWMA','VWAP','OBV','VOL_SPIKE','MOM','STOCH','ADX','CCI']
    for p in priority:
        for it in indicators_list:
            if p in it:
                return it
    return indicators_list[0] if indicators_list else ''


def backtest_point_strategy(df_signals, params):
    """
    Modified backtester: ENTRY is executed at the SAME bar's Close (no next-bar open used).
    Exits are still checked on subsequent bars (their H/L/Close) — that's not lookahead relative to entry.
    """
    trades = []
    in_pos = False
    pos_side = 0
    entry_price = None
    entry_date = None
    entry_details = None
    target = None
    sl = None

    # buy & hold baseline (points)
    first_price = df_signals['Close'].iloc[0]
    last_price = df_signals['Close'].iloc[-1]
    buy_hold_points = last_price - first_price

    # iterate rows in order -- we use i from 0..n-2 because exits look at future rows
    for i in range(len(df_signals)-1):
        row = df_signals.iloc[i]
        next_row = df_signals.iloc[i+1]
        sig = row['Signal']

        # ENTRY: when not in position and signal on THIS bar -> enter at THIS bar's Close (no future open used)
        if (not in_pos) and sig != 0:
            entry_date = row.name
            entry_price = row['Close']
            pos_side = sig
            atr_val = row.get(f"atr_{params['atr_period']}", np.nan)
            if np.isnan(atr_val) or atr_val == 0:
                atr_val = df_signals[f"atr_{params['atr_period']}"].median() or 1.0
            if pos_side == 1:
                target = entry_price + params['target_atr_mult'] * atr_val
                sl = entry_price - params['sl_atr_mult'] * atr_val
            else:
                target = entry_price - params['target_atr_mult'] * atr_val
                sl = entry_price + params['sl_atr_mult'] * atr_val

            indicators = (row['indicators_long'] if pos_side==1 else row['indicators_short'])
            primary = choose_primary_indicator(indicators)

            entry_details = {
                "Entry Date": entry_date, "Entry Price": entry_price,
                "Side": "Long" if pos_side==1 else "Short",
                "Indicators": indicators,
                "Primary Indicator": primary,
                "Confluences": row['total_votes']
            }
            in_pos = True
            continue

        # If in position, check exits on next_row (future after entry)
        if in_pos:
            h = next_row['High']; l = next_row['Low']; closep = next_row['Close']
            exit_price = None; exit_date = None; reason = None

            if pos_side == 1:
                if not pd.isna(h) and h >= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(l) and l <= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif next_row['Signal'] == -1 and next_row['total_votes'] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"
            else:
                if not pd.isna(l) and l <= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(h) and h >= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif next_row['Signal'] == 1 and next_row['total_votes'] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"

            # final day fallback: if last day and still in_pos, exit at close
            if (i+1) == (len(df_signals)-1) and in_pos and exit_price is None:
                exit_price = closep
                reason = "End of data"

            if exit_price is not None:
                exit_date = next_row.name
                points = (exit_price - entry_price) if pos_side == 1 else (entry_price - exit_price)
                trades.append({
                    **entry_details,
                    "Exit Date": exit_date,
                    "Exit Price": exit_price,
                    "Reason Exit": reason,
                    "Points": points,
                    "Hold Days": (pd.to_datetime(exit_date).date() - pd.to_datetime(entry_details['Entry Date']).date()).days
                })

                in_pos = False
                pos_side = 0
                entry_price = None
                entry_details = None
                target = None
                sl = None

    trades_df = pd.DataFrame(trades)
    total_points = trades_df['Points'].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = (trades_df['Points'] > 0).sum() if not trades_df.empty else 0
    prob_of_profit = (wins / num_trades) if num_trades>0 else 0.0

    percent_vs_buyhold = (total_points / (abs(buy_hold_points)+1e-9)) * 100 if abs(buy_hold_points) > 0 else np.nan

    summary = {
        'total_points': total_points,
        'num_trades': num_trades,
        'wins': wins,
        'prob_of_profit': prob_of_profit,
        'buy_hold_points': buy_hold_points,
        'pct_vs_buyhold': percent_vs_buyhold
    }
    return summary, trades_df

# ------------------- Parameter optimization (random search) -------------------

def sample_random_params(base):
    p = base.copy()
    # sample sensible ranges
    p['sma_fast'] = random.choice([5,8,10,12,15,20])
    p['sma_slow'] = random.choice([50,100,150,200])
    if p['sma_fast'] >= p['sma_slow']:
        p['sma_fast'] = max(5, p['sma_slow']//10)
    p['ema_fast'] = random.choice([5,9,12,15])
    p['ema_slow'] = random.choice([21,26,34,50])
    if p['ema_fast'] >= p['ema_slow']:
        p['ema_fast'] = max(5, p['ema_slow']//3)
    p['rsi_period'] = random.choice([7,9,14,21])
    p['mom_period'] = random.choice([5,10,20])
    p['atr_period'] = random.choice([7,14,21])
    p['target_atr_mult'] = round(random.uniform(0.6,3.0),2)
    p['sl_atr_mult'] = round(random.uniform(0.6,3.0),2)
    p['min_confluence'] = random.randint(1,6)
    p['vol_multiplier'] = round(random.uniform(1.0,3.0),2)
    return p


def optimize_parameters(df, base_params, n_iter, target_acc, target_points, side):
    best = None
    best_score = None
    results = []
    target_frac = target_acc
    for i in range(n_iter):
        p = sample_random_params(base_params)
        try:
            df_sig = generate_confluence_signals(df, p, side)
            summary, trades = backtest_point_strategy(df_sig, p)
        except Exception as e:
            continue
        prob = summary['prob_of_profit']
        # compute score based on abs diff in accuracy and prefer higher total_points
        score = abs(prob - target_frac) - 0.0001 * summary['total_points']
        results.append((p, summary, trades, score))
        if best is None or score < best_score:
            best = (p, summary, trades)
            best_score = score
        # early stop if meets both targets
        if prob >= target_frac and summary['total_points'] >= target_points:
            return p, summary, trades, True
    # no perfect meet; return best
    return best[0], best[1], best[2], False

# ------------------- Streamlit App -------------------

st.title("Backtester with Confluence + Accuracy Target (Entry on Close)")
st.markdown("Upload OHLCV CSV/XLSX (Date,Open,High,Low,Close,Volume). The app will run a randomized search to find parameters that reach user-specified accuracy% (probability of profit) and strategy returns.

**Note:** Entry is executed at the same bar's Close (no next-bar Open is used). Use the End Date selector below to restrict the data used for optimization/backtest.")

uploaded_file = st.file_uploader("Upload CSV or XLSX", type=['csv','xlsx'])
side = st.selectbox("Trade Side", options=["Both","Long","Short"], index=0)
random_iters = st.number_input("Random iterations (1-2000)", min_value=1, max_value=2000, value=200, step=1)
expected_returns = st.number_input("Expected strategy returns (total points)", value=0.0, step=1.0, format="%.2f")
expected_accuracy_pct = st.number_input("Expected accuracy % (probability of profit, e.g. 70)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
run_btn = st.button("Run Backtest & Optimize")

if uploaded_file is not None:
    try:
        if str(uploaded_file).lower().endswith('.xlsx') or hasattr(uploaded_file, 'getvalue') and ('xls' in uploaded_file.name.lower()):
            raw = pd.read_excel(uploaded_file)
        else:
            raw = pd.read_csv(uploaded_file)
        df = normalize_df(raw)
    except Exception as e:
        st.error(f"Failed to read/normalize file: {e}")
        st.stop()

    # allow user to select End Date (last date). Start date remains min date of data
    available_dates = [d.strftime('%Y-%m-%d') for d in df.index.unique()]
    default_idx = len(available_dates) - 1
    selected_end_str = st.selectbox("Select last date for backtest (start = min date)", options=available_dates, index=default_idx)
    selected_end_dt = pd.to_datetime(selected_end_str)
    # restrict data up to selected end date (inclusive)
    df = df.loc[:selected_end_dt].copy()

    if run_btn:
        with st.spinner("Running optimization on data up to selected end date..."):
            # show uploaded top/bottom 5 rows (original uploaded file)
            st.subheader("Uploaded file sample")
            try:
                st.write(f"Uploaded rows: {raw.shape[0]} Columns: {raw.shape[1]}")
                st.dataframe(raw.head(5))
                st.subheader("Uploaded file - bottom 5 rows")
                st.dataframe(raw.tail(5))
            except Exception:
                pass

            # base params (defaults)
            base_params = {
                'sma_fast': 10, 'sma_slow': 50,
                'ema_fast': 9, 'ema_slow': 21,
                'macd_signal': 9, 'rsi_period': 14, 'mom_period': 10,
                'stoch_period': 14, 'cci_period': 20, 'adx_period': 14,
                'atr_period': 14, 'target_atr_mult': 1.5, 'sl_atr_mult': 1.0,
                'min_confluence': 3, 'vol_multiplier': 1.5,
                'vwma_period': 14, 'vol_sma_period': 20
            }

            target_acc = expected_accuracy_pct/100.0
            target_points = expected_returns

            best_params, best_summary, best_trades, perfect = optimize_parameters(df, base_params, int(random_iters), target_acc, target_points, side)

            st.subheader("Optimization Result")
            st.write("Target accuracy:", expected_accuracy_pct, "% ; Target points:", target_points)
            st.write("Perfect match found:" , perfect)
            st.json(best_params)

            st.subheader("Summary (best candidate)")
            st.write(best_summary)

            if best_trades is None or best_trades.empty:
                st.info("No trades found with best parameters.")
            else:
                # preserve format + show Primary Indicator column
                best_trades_display = best_trades.copy()
                # Top 5 & Bottom 5
                st.subheader("Top 5 trades (by Points)")
                st.dataframe(best_trades_display.nlargest(5, 'Points'))
                st.subheader("Bottom 5 trades (by Points)")
                st.dataframe(best_trades_display.nsmallest(5, 'Points'))

                # Heatmap: monthly returns pivot table (Year x Month) - show percent returns
                best_trades_display['Exit Date'] = pd.to_datetime(best_trades_display['Exit Date'])
                best_trades_display['Year'] = best_trades_display['Exit Date'].dt.year
                best_trades_display['Month'] = best_trades_display['Exit Date'].dt.month
                monthly_points = best_trades_display.groupby(['Year','Month'])['Points'].sum().reset_index()

                # month start price from original df (restricted to selected end date)
                month_start = df['Close'].resample('MS').first().reset_index()
                month_start['Year'] = month_start['Date'].dt.year
                month_start['Month'] = month_start['Date'].dt.month
                month_start = month_start.rename(columns={'Close':'Month_Start_Close'})

                monthly = monthly_points.merge(month_start[['Year','Month','Month_Start_Close']], on=['Year','Month'], how='left')
                # if missing month_start_close, fallback to average close
                avg_close = df['Close'].mean()
                monthly['Month_Start_Close'] = monthly['Month_Start_Close'].fillna(avg_close)
                monthly['Pct_Return'] = (monthly['Points'] / monthly['Month_Start_Close']) * 100.0

                pivot_pct = monthly.pivot(index='Year', columns='Month', values='Pct_Return').fillna(0)
                # ensure months 1..12 present
                for m in range(1,13):
                    if m not in pivot_pct.columns:
                        pivot_pct[m] = 0
                pivot_pct = pivot_pct.reindex(sorted(pivot_pct.columns), axis=1)

                st.subheader("Monthly % returns heatmap (Year vs Month)")
                fig, ax = plt.subplots(figsize=(10, max(2, 0.6*len(pivot_pct.index)+1)))
                sns.heatmap(pivot_pct, annot=True, fmt='.2f', linewidths=0.5, ax=ax)
                ax.set_ylabel('Year')
                ax.set_xlabel('Month')
                st.pyplot(fig)

                st.subheader("All trades (best candidate)")
                st.dataframe(best_trades_display)

            # live recommendation (based on selected end date)
            latest_sig_df = generate_confluence_signals(df, best_params, side)
            latest_row = latest_sig_df.iloc[-1]

            # human friendly signal
            sig_val = int(latest_row['Signal'])
            sig_text = "Buy" if sig_val == 1 else ("Sell" if sig_val == -1 else "No Signal")

            # ENTRY is at last candle Close (selected_end_dt). We do NOT use next bar open for entry.
            entry_price_est = float(latest_row['Close'])
            atr_val = latest_row.get(f"atr_{best_params['atr_period']}", np.nan)
            if sig_val == 1:
                target_price = entry_price_est + best_params['target_atr_mult'] * atr_val
                sl_price = entry_price_est - best_params['sl_atr_mult'] * atr_val
                indicators_list = latest_row['indicators_long']
            elif sig_val == -1:
                target_price = entry_price_est - best_params['target_atr_mult'] * atr_val
                sl_price = entry_price_est + best_params['sl_atr_mult'] * atr_val
                indicators_list = latest_row['indicators_short']
            else:
                target_price = np.nan
                sl_price = np.nan
                indicators_list = []

            primary = choose_primary_indicator(indicators_list)
            confluences = int(latest_row.get('total_votes', 0))
            prob_of_profit = (best_summary.get('prob_of_profit', np.nan) * 100.0) if isinstance(best_summary, dict) else np.nan

            # build a short, human-readable reason/logic string
            def explain_indicator(it):
                if it.startswith('EMA'):
                    return f"{it}: short-term EMA above/below long-term EMA -> momentum signal"
                if it.startswith('SMA'):
                    return f"{it}: SMA crossover indicating trend bias"
                if 'MACD' in it:
                    return "MACD+: histogram >0 indicates bullish momentum; MACD- indicates bearish"
                if it.startswith('RSI'):
                    return "RSI extreme indicates overbought/oversold"
                if 'BB' in it:
                    return "BB band touch suggests mean-reversion/extreme"
                if 'VWMA' in it or 'VWAP' in it:
                    return "Price vs VWMA indicates trade direction with volume support"
                if 'OBV' in it:
                    return "OBV rising/falling shows accumulation/distribution"
                if it == 'VOL_SPIKE':
                    return "Volume spike with price direction — increased conviction"
                if it.startswith('MOM'):
                    return "Momentum positive/negative"
                if it == 'STOCH':
                    return "Stochastic crossover in extreme zones"
                if 'ADX' in it:
                    return "ADX indicates trend strength and direction via DI lines"
                if 'CCI' in it:
                    return "CCI in extremes suggests momentum reversal/continuation"
                return it

            reasons = [explain_indicator(ii) for ii in indicators_list]
            reason_text = (f"Primary: {primary}. ") + ("; ".join(reasons) if reasons else "No strong indicator explanation.")

            indicator_values = {
                'sma_fast': latest_row.get(f"sma_{best_params['sma_fast']}", np.nan),
                'sma_slow': latest_row.get(f"sma_{best_params['sma_slow']}", np.nan),
                'ema_fast': latest_row.get(f"ema_{best_params['ema_fast']}", np.nan),
                'ema_slow': latest_row.get(f"ema_{best_params['ema_slow']}", np.nan),
                'macd_hist': latest_row.get('macd_hist', np.nan),
                f"rsi_{best_params['rsi_period']}": latest_row.get(f"rsi_{best_params['rsi_period']}", np.nan),
                'bb_upper': latest_row.get('bb_upper', np.nan),
                'bb_lower': latest_row.get('bb_lower', np.nan),
                'obv': latest_row.get('obv', np.nan),
                'vwma': latest_row.get('vwma', np.nan),
                'cci': latest_row.get('cci', np.nan),
                'vol': latest_row.get('Volume', np.nan),
                'vol_sma': latest_row.get('vol_sma', np.nan),
                f"atr_{best_params['atr_period']}": latest_row.get(f"atr_{best_params['atr_period']}", np.nan)
            }

            st.subheader("Latest live recommendation (based on best params & selected end date)")
            st.markdown(f"**Date (end date used):** {selected_end_str}")
            st.markdown(f"**Signal:** {sig_text}")
            st.markdown(f"**Entry (at bar Close):** {entry_price_est:.2f}")
            st.markdown(f"**Target:** {target_price:.2f}  |  **Stop-loss:** {sl_price:.2f}")
            st.markdown(f"**Confluences (votes):** {confluences}  |  **Primary indicator:** {primary}")
            st.markdown(f"**Probability of profit (backtested):** {prob_of_profit:.2f}%")
            st.markdown("**Indicators that voted:**")
            st.write(indicators_list)
            st.markdown("**Reason / Logic (brief):**")
            st.write(reason_text)

            st.subheader("Latest indicator values (key ones)")
            ind_df = pd.DataFrame([indicator_values]).T.reset_index()
            ind_df.columns = ['Indicator', 'Value']
            st.dataframe(ind_df)

            st.success("Done")

    else:
        st.info("Select an End Date (defaults to max date), then click 'Run Backtest & Optimize' to start.")

else:
    st.info("Upload a CSV/XLSX to begin.")


# ------------------- End -------------------
