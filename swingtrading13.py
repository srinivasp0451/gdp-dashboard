"""
Streamlit app: Swing backtester with user-driven optimization (expected accuracy & expected return)

Changes made (per user request):
- Added input box for user to specify expected accuracy (%) and expected strategy return (%).
- Implemented a random-search optimizer that samples parameter combinations and picks the best set that meets user thresholds (if any).
- Removed "expectancy" metric from outputs (not computed/displayed).
- Added a volume indicator (volume EMA + volume spike rule) to signal generation and used it as a voting indicator.
- Expanded indicator names to include parameter values (e.g., EMA_9_21, SMA_5_20) so each trade log shows which exact indicator parameters contributed.
- Added a "Primary Indicator" column in trade logs (first indicator in the list) while preserving the original log fields.
- Displays Top 5 and Bottom 5 trades and a heatmap of monthly returns (month vs year). Works even with 1+ year of data.
- Inputs preserved/added: file upload, trade side, random iterations, expected strategy returns (%), expected accuracy (%), and a Run button.

Notes:
- This script assumes the uploaded file has OHLCV data. It tries to detect common column names (case-insensitive).
- Signals are generated using only current/past data. Entries are executed on the next bar's Open (no lookahead). Exits use the next bar's H/L range (typical EOD heuristic).
- The optimizer runs locally (random iterations). Keep iterations modest if dataset is large.

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random

# Column name constants (normalized)
O, H, L, C, V = "Open", "High", "Low", "Close", "Volume"

st.set_page_config(page_title="Swing Backtester + Accuracy Optimizer", layout="wide")
st.title("Swing Backtester with Expected-Accuracy Optimization")

# ---------- Utilities ----------

def detect_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to normalize common OHLCV and Date column names to O,H,L,C,V and set datetime index."""
    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    # date detection
    date_col = None
    for candidate in ("date", "datetime", "time"):
        if candidate in cols_lower:
            date_col = cols_lower[candidate]
            break
    # OHLCV detection
    for target, names in [(O, ["open", "o"]), (H, ["high", "h"]), (L, ["low", "l"]), (C, ["close", "c", "adj close", "closeprice"]), (V, ["volume", "vol"])]:
        for n in names:
            if n in cols_lower:
                col_map[cols_lower[n]] = target
                break
    # if no explicit OHLCV mapping found, try heuristics
    if C not in col_map.values() and len(df.columns) >= 4:
        # try to guess by position: date + O H L C V or O H L C
        possible = [c for c in df.columns if c not in ([date_col] if date_col else [])]
        if len(possible) >= 4:
            col_map[possible[0]] = O
            col_map[possible[1]] = H
            col_map[possible[2]] = L
            col_map[possible[3]] = C
            if len(possible) >= 5:
                col_map[possible[4]] = V

    df = df.rename(columns=col_map)

    # set datetime index
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        # if no date column but index looks like datetime, keep it
        if not isinstance(df.index, pd.DatetimeIndex):
            # try to infer from first column
            try:
                firstcol = df.columns[0]
                df[firstcol] = pd.to_datetime(df[firstcol])
                df = df.set_index(firstcol)
            except Exception:
                pass

    # ensure columns exist
    for required in [O, H, L, C]:
        if required not in df.columns:
            raise ValueError(f"Could not find required column for {required}. Found: {list(df.columns)}")

    # ensure volume column exists
    if V not in df.columns:
        df[V] = 0

    # sort index
    df = df.sort_index()
    return df


# ---------- Indicator calculations ----------

def compute_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    close = df[C]
    high = df[H]
    low = df[L]
    vol = df[V]

    # SMA & EMA
    df[f"sma_{params['sma_fast']}"] = close.rolling(params['sma_fast']).mean()
    df[f"sma_{params['sma_slow']}"] = close.rolling(params['sma_slow']).mean()
    df[f"ema_{params['ema_fast']}"] = close.ewm(span=params['ema_fast'], adjust=False).mean()
    df[f"ema_{params['ema_slow']}"] = close.ewm(span=params['ema_slow'], adjust=False).mean()

    # MACD (fast, slow, signal)
    macd_fast = close.ewm(span=params.get('macd_fast',12), adjust=False).mean()
    macd_slow = close.ewm(span=params.get('macd_slow',26), adjust=False).mean()
    macd = macd_fast - macd_slow
    macd_signal = macd.ewm(span=params.get('macd_signal',9), adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd - macd_signal

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rsi_period = params['rsi_period']
    ma_up = up.ewm(alpha=1/rsi_period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    df[f"rsi_{rsi_period}"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_period = params.get('bb_period', 20)
    tp = (close + high + low) / 3
    mb = tp.rolling(bb_period).mean()
    sd = tp.rolling(bb_period).std()
    df['bb_middle'] = mb
    df['bb_upper'] = mb + 2 * sd
    df['bb_lower'] = mb - 2 * sd

    # Momentum
    mom_period = params['mom_period']
    df[f"mom_{mom_period}"] = close.diff(mom_period)

    # Stochastic oscillator (fast)
    stoch_k_period = params.get('stoch_k_period', 14)
    stoch_d_period = params.get('stoch_d_period', 3)
    low_min = low.rolling(stoch_k_period).min()
    high_max = high.rolling(stoch_k_period).max()
    df['stoch_k'] = 100 * ((close - low_min) / (high_max - low_min + 1e-9))
    df['stoch_d'] = df['stoch_k'].rolling(stoch_d_period).mean()

    # ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr_{params['atr_period']}"] = tr.rolling(params['atr_period']).mean()

    # OBV
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df['obv'] = obv

    # VWAP: cumulative typical price * vol / cumulative vol (works as daily-series VWAP)
    typ = (high + low + close) / 3
    cum_typ_vol = (typ * vol).cumsum()
    cum_vol = vol.cumsum().replace(0, np.nan)
    df['vwap'] = cum_typ_vol / cum_vol

    # CCI
    cci_period = params.get('cci_period', 20)
    tp_sma = tp.rolling(cci_period).mean()
    tp_mean_dev = tp.rolling(cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['cci'] = (tp - tp_sma) / (0.015 * tp_mean_dev + 1e-9)

    # ADX (simplified) and PDI/MDI
    # Following a simple Welles Wilder smoothing
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    atr = tr.rolling(params['atr_period']).mean() + 1e-9
    plus_di = 100 * (plus_dm.ewm(alpha=1/params['atr_period'], adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/params['atr_period'], adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.ewm(alpha=1/params['atr_period'], adjust=False).mean()
    df['pdi'] = plus_di
    df['mdi'] = minus_di
    df['adx'] = adx

    # Volume EMA and volume spike indicator
    vol_ema = vol.ewm(span=params.get('vol_ema_period', 20), adjust=False).mean()
    df['vol_ema'] = vol_ema
    df['vol_spike'] = (vol > (vol_ema * params.get('vol_spike_mult', 1.5))).astype(int)

    # ensure no infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ---------- Signal generation (confluence voting) ----------

def generate_confluence_signals(df_local: pd.DataFrame, params: dict, side="Both") -> pd.DataFrame:
    df_calc = compute_indicators(df_local, params)
    votes = []
    sig_series = pd.Series(0, index=df_calc.index)

    # Prepare convenient names
    sma_fast_col = f"sma_{params['sma_fast']}"
    sma_slow_col = f"sma_{params['sma_slow']}"
    ema_fast_col = f"ema_{params['ema_fast']}"
    ema_slow_col = f"ema_{params['ema_slow']}"
    rsi_col = f"rsi_{params['rsi_period']}"
    mom_col = f"mom_{params['mom_period']}"
    atr_col = f"atr_{params['atr_period']}"

    for idx in df_calc.index:
        row = df_calc.loc[idx]
        indicators_that_long = []
        indicators_that_short = []

        # 1) SMA crossover (fast > slow -> long)
        if not pd.isna(row[sma_fast_col]) and not pd.isna(row[sma_slow_col]):
            if row[sma_fast_col] > row[sma_slow_col]:
                indicators_that_long.append(f"SMA_{params['sma_fast']}_{params['sma_slow']}")
            elif row[sma_fast_col] < row[sma_slow_col]:
                indicators_that_short.append(f"SMA_{params['sma_fast']}_{params['sma_slow']}")

        # 2) EMA crossover
        if not pd.isna(row[ema_fast_col]) and not pd.isna(row[ema_slow_col]):
            if row[ema_fast_col] > row[ema_slow_col]:
                indicators_that_long.append(f"EMA_{params['ema_fast']}_{params['ema_slow']}")
            elif row[ema_fast_col] < row[ema_slow_col]:
                indicators_that_short.append(f"EMA_{params['ema_fast']}_{params['ema_slow']}")

        # 3) MACD histogram positive -> long
        if not pd.isna(row.get('macd_hist', np.nan)):
            if row['macd_hist'] > 0:
                indicators_that_long.append(f"MACD_{params.get('macd_fast',12)}_{params.get('macd_slow',26)}")
            elif row['macd_hist'] < 0:
                indicators_that_short.append(f"MACD_{params.get('macd_fast',12)}_{params.get('macd_slow',26)}")

        # 4) RSI extremes: <35 long, >65 short
        if not pd.isna(row.get(rsi_col, np.nan)):
            if row[rsi_col] < params.get('rsi_oversold', 35):
                indicators_that_long.append(f"RSI_{params['rsi_period']}")
            elif row[rsi_col] > params.get('rsi_overbought', 65):
                indicators_that_short.append(f"RSI_{params['rsi_period']}")

        # 5) Bollinger: price crossing bands
        if not pd.isna(row.get('bb_upper', np.nan)) and not pd.isna(row.get('bb_lower', np.nan)):
            price = row[C]
            if price < row['bb_lower']:
                indicators_that_long.append(f"BB_Lower_{params.get('bb_period',20)}")
            elif price > row['bb_upper']:
                indicators_that_short.append(f"BB_Upper_{params.get('bb_period',20)}")

        # 6) Momentum
        if not pd.isna(row.get(mom_col, np.nan)):
            if row[mom_col] > 0:
                indicators_that_long.append(f"MOM_{params['mom_period']}")
            elif row[mom_col] < 0:
                indicators_that_short.append(f"MOM_{params['mom_period']}")

        # 7) Stochastic
        if not pd.isna(row.get('stoch_k', np.nan)) and not pd.isna(row.get('stoch_d', np.nan)):
            if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 30:
                indicators_that_long.append(f"STOCH_{params.get('stoch_k_period',14)}")
            elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 70:
                indicators_that_short.append(f"STOCH_{params.get('stoch_k_period',14)}")

        # 8) ADX directional
        if not pd.isna(row.get('adx', np.nan)) and not pd.isna(row.get('pdi', np.nan)) and not pd.isna(row.get('mdi', np.nan)):
            if row['adx'] > 20 and row['pdi'] > row['mdi']:
                indicators_that_long.append("ADX+")
            elif row['adx'] > 20 and row['mdi'] > row['pdi']:
                indicators_that_short.append("ADX-")

        # 9) OBV trend (recent 3 vs previous 3)
        i = df_calc.index.get_indexer([idx])[0]
        if i >= 2:
            recent = df_calc['obv'].iloc[max(0, i-2):i+1].mean()
            prev = df_calc['obv'].iloc[max(0, i-5):max(0,i-2)].mean() if i >= 3 else np.nan
            if not np.isnan(prev) and recent > prev:
                indicators_that_long.append('OBV')
            elif not np.isnan(prev) and recent < prev:
                indicators_that_short.append('OBV')

        # 10) VWAP
        if not pd.isna(row.get('vwap', np.nan)):
            if row[C] > row['vwap']:
                indicators_that_long.append('VWAP')
            elif row[C] < row['vwap']:
                indicators_that_short.append('VWAP')

        # 11) CCI extremes
        if not pd.isna(row.get('cci', np.nan)):
            if row['cci'] < -100:
                indicators_that_long.append(f"CCI_{params.get('cci_period',20)}")
            elif row['cci'] > 100:
                indicators_that_short.append(f"CCI_{params.get('cci_period',20)}")

        # 12) Volume spike (volume indicator requested)
        if not pd.isna(row.get('vol_spike', np.nan)) and row.get('vol_spike', 0) == 1:
            # treat as bullish long vote when price action supports
            if row[C] > row.get('vwap', row[C]):
                indicators_that_long.append(f"VOL_SPIKE_{params.get('vol_ema_period',20)}")
            else:
                indicators_that_short.append(f"VOL_SPIKE_{params.get('vol_ema_period',20)}")

        # count confluences
        long_votes = len(set(indicators_that_long))
        short_votes = len(set(indicators_that_short))
        total_votes = max(long_votes, short_votes)

        vote_direction = 1 if long_votes > short_votes else (-1 if short_votes > long_votes else 0)
        # apply side filter
        if side == "Long" and vote_direction == -1:
            vote_direction = 0
            total_votes = 0
        if side == "Short" and vote_direction == 1:
            vote_direction = 0
            total_votes = 0

        # produce final signal only if votes >= threshold
        final_sig = vote_direction if total_votes >= params['min_confluence'] else 0

        votes.append({
            "index": idx,
            "long_votes": long_votes,
            "short_votes": short_votes,
            "total_votes": total_votes,
            "direction": vote_direction,
            "signal": final_sig,
            "indicators_long": indicators_that_long,
            "indicators_short": indicators_that_short
        })
        sig_series.loc[idx] = final_sig

    votes_df = pd.DataFrame(votes).set_index("index")
    result = df_calc.join(votes_df)
    result["Signal"] = sig_series
    return result


# ---------- Backtester (kept logic intact, small additions) ----------

def backtest_point_strategy(df_signals: pd.DataFrame, params: dict):
    trades = []
    in_pos = False
    pos_side = 0
    entry_price = None
    entry_date = None
    entry_details = None
    target = None
    sl = None

    # buy & hold baseline (points)
    first_price = df_signals[C].iloc[0]
    last_price = df_signals[C].iloc[-1]
    buy_hold_points = last_price - first_price

    # iterate rows in order
    for i in range(len(df_signals)-1):
        row = df_signals.iloc[i]
        next_row = df_signals.iloc[i+1]
        sig = row["Signal"]

        # If not in position and there's a signal -> enter next day's open
        if (not in_pos) and sig != 0:
            entry_date = next_row.name  # next day's timestamp (index)
            entry_price = next_row[O] if not pd.isna(next_row[O]) else next_row[C]
            pos_side = sig  # 1 long, -1 short
            # set sl/target based on ATR
            atr_val = row.get(f"atr_{params['atr_period']}", np.nan)
            if np.isnan(atr_val) or atr_val == 0:
                atr_val = df_signals[f"atr_{params['atr_period']}"].median() or 1.0
            if pos_side == 1:
                target = entry_price + params['target_atr_mult'] * atr_val
                sl = entry_price - params['sl_atr_mult'] * atr_val
            else:
                target = entry_price - params['target_atr_mult'] * atr_val
                sl = entry_price + params['sl_atr_mult'] * atr_val

            entry_details = {
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Side": "Long" if pos_side==1 else "Short",
                "Indicators": (row["indicators_long"] if pos_side==1 else row["indicators_short"]),
                "Confluences": row["total_votes"]
            }
            in_pos = True
            continue

        # If in position, check exit conditions for next_row (we check next_row price range)
        if in_pos:
            h = next_row[H]; l = next_row[L]; closep = next_row[C]
            exit_price = None; exit_date = None; reason = None

            if pos_side == 1:
                if not pd.isna(h) and h >= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(l) and l <= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif next_row["Signal"] == -1 and next_row["total_votes"] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"
            else:  # short
                if not pd.isna(l) and l <= target:
                    exit_price = target
                    reason = "Target hit"
                elif not pd.isna(h) and h >= sl:
                    exit_price = sl
                    reason = "Stopped"
                elif next_row["Signal"] == 1 and next_row["total_votes"] >= params['min_confluence']:
                    exit_price = closep
                    reason = "Opposite signal"

            # final day fallback: if last day and still in_pos, exit at close
            if (i+1) == (len(df_signals)-1) and in_pos and exit_price is None:
                exit_price = closep
                reason = "End of data"

            if exit_price is not None:
                exit_date = next_row.name
                points = (exit_price - entry_price) if pos_side == 1 else (entry_price - exit_price)
                primary_indicator = None
                if entry_details and entry_details.get('Indicators'):
                    primary_indicator = entry_details['Indicators'][0] if len(entry_details['Indicators'])>0 else None

                trades.append({
                    **entry_details,
                    "Exit Date": exit_date,
                    "Exit Price": exit_price,
                    "Reason Exit": reason,
                    "Points": points,
                    "Hold Days": (exit_date.date() - entry_details["Entry Date"].date()).days,
                    "Primary Indicator": primary_indicator
                })
                # reset
                in_pos = False
                pos_side = 0
                entry_price = None
                entry_details = None
                target = None
                sl = None

    trades_df = pd.DataFrame(trades)
    total_points = trades_df["Points"].sum() if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    wins = (trades_df["Points"] > 0).sum() if not trades_df.empty else 0
    prob_of_profit = (wins / num_trades) if num_trades>0 else 0.0

    buy_hold_points = last_price - first_price if 'first_price' in locals() else (df_signals[C].iloc[-1] - df_signals[C].iloc[0])
    percent_vs_buyhold = (total_points / (abs(buy_hold_points)+1e-9)) * 100 if abs(buy_hold_points) > 0 else np.nan

    summary = {
        "total_points": total_points,
        "num_trades": num_trades,
        "wins": int(wins),
        "prob_of_profit": prob_of_profit,
        "buy_hold_points": buy_hold_points,
        "pct_vs_buyhold": percent_vs_buyhold
    }
    return summary, trades_df


# ---------- Optimization / evaluation ----------

def evaluate_params(df: pd.DataFrame, params: dict, side: str):
    df_signals = generate_confluence_signals(df, params, side)
    summary, trades_df = backtest_point_strategy(df_signals, params)
    return summary, trades_df, df_signals


def random_search_optimizer(df: pd.DataFrame, side: str, iterations: int, expected_acc: float, expected_return: float, seed: int=42):
    random.seed(seed)
    best = None
    passing = []
    records = []

    for it in range(iterations):
        # sample params
        p = {
            'sma_fast': random.choice([5,8,10,12]),
            'sma_slow': random.choice([20,30,50]),
            'ema_fast': random.choice([5,9,12]),
            'ema_slow': random.choice([21,26,34]),
            'rsi_period': random.choice([7,14,21]),
            'mom_period': random.choice([5,10,14]),
            'atr_period': random.choice([10,14,20]),
            'target_atr_mult': round(random.uniform(0.8, 3.0),2),
            'sl_atr_mult': round(random.uniform(0.8, 3.0),2),
            'min_confluence': random.choice([2,3,4]),
            'vol_ema_period': random.choice([10,20,30]),
            'vol_spike_mult': round(random.uniform(1.2, 3.0),2),
            'macd_fast':12,'macd_slow':26,'macd_signal':9,
            'bb_period':20,'cci_period':20
        }
        # sanity: ensure sma_fast < sma_slow etc
        if p['sma_fast'] >= p['sma_slow']:
            p['sma_fast'], p['sma_slow'] = min(p['sma_fast'], p['sma_slow']), max(p['sma_fast'], p['sma_slow'])
        if p['ema_fast'] >= p['ema_slow']:
            p['ema_fast'], p['ema_slow'] = min(p['ema_fast'], p['ema_slow']), max(p['ema_fast'], p['ema_slow'])

        # compute indicators & backtest
        try:
            summary, trades_df, df_signals = evaluate_params(df, p, side)
        except Exception as e:
            # ignore combos that error
            continue

        rec = {**p, **summary}
        records.append(rec)

        meets_acc = (summary['prob_of_profit']*100) >= expected_acc if expected_acc and expected_acc>0 else True
        meets_ret = summary['pct_vs_buyhold'] >= expected_return if expected_return and not np.isnan(summary['pct_vs_buyhold']) and expected_return>0 else True
        if meets_acc and meets_ret:
            passing.append((p, summary, trades_df))

        # track best by combined score (prob + normalized return)
        score = summary['prob_of_profit']*100 + (summary['pct_vs_buyhold'] if not np.isnan(summary['pct_vs_buyhold']) else 0)
        if best is None or score > best[0]:
            best = (score, p, summary, trades_df)

    # if passing candidates exist choose one with highest pct_vs_buyhold
    if passing:
        passing_sorted = sorted(passing, key=lambda x: (x[1]['pct_vs_buyhold'], x[1]['prob_of_profit']), reverse=True)
        chosen = passing_sorted[0]
        return {'status':'found', 'params': chosen[0], 'summary': chosen[1], 'trades': chosen[2], 'records': pd.DataFrame(records)}
    else:
        # return best found
        if best is not None:
            return {'status':'best_only', 'params': best[1], 'summary': best[2], 'trades': best[3], 'records': pd.DataFrame(records)}
        else:
            return {'status':'none', 'records': pd.DataFrame(records)}


# ---------- Streamlit UI ----------

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload OHLCV CSV/XLSX file", type=['csv','xlsx'])
trade_side = st.sidebar.selectbox("Trade side", options=['Both','Long','Short'])
random_iterations = st.sidebar.number_input("Random iterations", min_value=1, max_value=2000, value=100, step=1)
expected_strategy_return = st.sidebar.number_input("Expected strategy return (%)", value=0.0, step=0.1)
expected_accuracy = st.sidebar.number_input("Expected accuracy (%)", value=0.0, step=0.1)
run_button = st.sidebar.button("Run Backtest & Optimize")

if uploaded is not None and run_button:
    try:
        if uploaded.name.lower().endswith('.csv'):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
        df = detect_and_normalize_columns(df_raw)
    except Exception as e:
        st.error(f"Error reading file or detecting columns: {e}")
        st.stop()

    # default params (used if user doesn't want random search)
    default_params = {
        'sma_fast':5,'sma_slow':20,'ema_fast':9,'ema_slow':21,'rsi_period':14,
        'mom_period':10,'atr_period':14,'target_atr_mult':1.5,'sl_atr_mult':1.5,'min_confluence':3,
        'vol_ema_period':20,'vol_spike_mult':1.5,'macd_fast':12,'macd_slow':26,'macd_signal':9,'bb_period':20,'cci_period':20
    }

    with st.spinner('Running optimization and backtests...'):
        result = random_search_optimizer(df, trade_side, int(random_iterations), float(expected_accuracy), float(expected_strategy_return))

    if result['status'] == 'none':
        st.warning('No valid parameter combinations found in the random search.')
    else:
        chosen_params = result['params']
        chosen_summary = result['summary']
        trades_df = result['trades']

        st.subheader('Chosen Parameters & Summary')
        st.write(chosen_params)
        st.metric('Total Points', chosen_summary['total_points'])
        st.metric('Number of Trades', chosen_summary['num_trades'])
        st.metric('Wins', chosen_summary['wins'])
        st.metric('Prob. of Profit (%)', f"{chosen_summary['prob_of_profit']*100:.2f}")
        st.metric('Pct vs Buy & Hold (%)', f"{chosen_summary['pct_vs_buyhold']:.2f}")

        # show records of search (optional)
        if 'records' in result and not result['records'].empty:
            st.subheader('Random Search Summary (sample of tried combinations)')
            st.dataframe(result['records'].sort_values(by=['prob_of_profit'], ascending=False).head(10))

        st.subheader('Trades (logs) — format preserved with added Primary Indicator')
        if trades_df is None or trades_df.empty:
            st.info('No trades generated for chosen parameter set.')
        else:
            # make sure Entry/Exit Date columns are datetime
            trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date'])
            trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date'])
            st.dataframe(trades_df)

            # Top 5 and Bottom 5
            st.subheader('Top 5 Trades')
            st.dataframe(trades_df.sort_values('Points', ascending=False).head(5))
            st.subheader('Bottom 5 Trades')
            st.dataframe(trades_df.sort_values('Points', ascending=True).head(5))

            # Heatmap: monthly returns (month vs year)
            trades_df['Entry Year'] = trades_df['Entry Date'].dt.year
            trades_df['Entry Month'] = trades_df['Entry Date'].dt.month
            pivot = trades_df.groupby(['Entry Year','Entry Month'])['Points'].sum().unstack(fill_value=0)
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(12, max(2, pivot.shape[0]*0.6)))
                sns.heatmap(pivot, annot=True, fmt='.1f', linewidths=0.5, cmap='RdYlGn', center=0, ax=ax)
                ax.set_title('Monthly Returns Heatmap (Points) — Year vs Month')
                ax.set_xlabel('Month')
                ax.set_ylabel('Year')
                st.pyplot(fig)

    st.success('Done. Review results above.')

else:
    st.info('Upload a CSV/XLSX with OHLCV data and press Run. Typical columns: Date, Open, High, Low, Close, Volume.')


# End of script
