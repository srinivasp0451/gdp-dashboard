# hpe_rf_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="HPE-RF — Hybrid Predictive Engine (Random Forest)")

st.title("HPE-RF — Hybrid Predictive Engine (Random Forest)")

# --------------------------
# Utilities: safe yfinance fetch (cached)
# --------------------------
@st.cache_data(ttl=600)
def download_yf(ticker, interval="1d", start=None, end=None, period=None):
    try:
        if start or end:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            df = yf.download(ticker, period=period or "2y", interval=interval, progress=False)
    except Exception as e:
        st.error(f"yfinance download error: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if 'Date' not in df.columns and 'Datetime' in df.columns:
        df.rename(columns={'Datetime':'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    for c in ['Open','High','Low','Close']:
        if c not in df.columns:
            return pd.DataFrame()
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df = df[['Date','Open','High','Low','Close','Volume']].dropna().reset_index(drop=True)
    return df

# --------------------------
# Feature engineering (adapts if volume missing)
# --------------------------
def add_features(df, use_volume_features=True, bb_n=20, atr_n=14, ema_short=20, ema_long=50):
    df = df.copy().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    # basic returns
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_3'] = df['Close'].pct_change(3)
    df['ret_5'] = df['Close'].pct_change(5)
    # ATR
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(atr_n, min_periods=1).mean().fillna(method='bfill')
    df['TR_PCT'] = df['ATR'] / df['Close']
    # candle body
    df['body'] = (df['Close'] - df['Open']).abs()
    df['body_strength'] = df['body'] / (df['ATR'] + 1e-9)
    df['range_ratio'] = (df['High'] - df['Low']) / (df['ATR'] + 1e-9)
    # EMAs & slopes
    df['ema_short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df['ema_long'] = df['Close'].ewm(span=ema_long, adjust=False).mean()
    df['ema_slope'] = (df['ema_short'] - df['ema_long']) / (df['ema_long'] + 1e-9)
    # Bollinger width
    ma = df['Close'].rolling(bb_n).mean()
    std = df['Close'].rolling(bb_n).std()
    df['bbw'] = ((ma + 2*std) - (ma - 2*std)) / (ma + 1e-9)
    # momentum acceleration
    df['mom_3'] = df['Close'].pct_change(3)
    df['mom_5'] = df['Close'].pct_change(5)
    df['mom_accel'] = df['mom_3'] - df['mom_5']
    # volume-based features (only if meaningful)
    if use_volume_features:
        df['vol_change'] = df['Volume'].pct_change(1).fillna(0)
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df.loc[i,'Close'] > df.loc[i-1,'Close']:
                obv.append(obv[-1] + df.loc[i,'Volume'])
            elif df.loc[i,'Close'] < df.loc[i-1,'Close']:
                obv.append(obv[-1] - df.loc[i,'Volume'])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        df['obv_20_ma'] = pd.Series(df['OBV']).rolling(20).mean().fillna(0)
    else:
        df['vol_change'] = 0.0
        df['OBV'] = 0.0
        df['obv_20_ma'] = 0.0
    # drop initial NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df

# --------------------------
# Label construction (dependent variable)
# Predict big up move or big down move over next N bars (horizon)
# --------------------------
def build_labels(df, horizon=5, up_threshold=0.02, down_threshold=-0.02):
    df = df.copy().reset_index(drop=True)
    df['future_ret'] = df['Close'].shift(-horizon) / df['Close'] - 1
    def label_row(x):
        if pd.isna(x):
            return np.nan
        if x >= up_threshold:
            return 1
        elif x <= down_threshold:
            return -1
        else:
            return 0
    df['label'] = df['future_ret'].apply(label_row)
    df = df.dropna(subset=['label']).reset_index(drop=True)
    return df

# --------------------------
# Model training + predict function
# --------------------------
def train_rf(X, y, n_estimators=200, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    return rf

# --------------------------
# Backtester: uses predictions (class outputs) to simulate trades (last-close entries)
# --------------------------
def backtest_predictions(df, preds, target_atr=2.0, sl_atr=1.0, max_hold=14, capital=100000):
    """
    df: original df with features & ATR
    preds: series aligned with df.index, values in {1, -1, 0}
    """
    df2 = df.copy().reset_index(drop=True)
    df2['pred'] = preds
    trades = []
    pos = None
    for i in range(len(df2)):
        r = df2.loc[i]
        if pos is None and r['pred'] != 0:
            pos = {
                'entry_idx': i,
                'entry_date': r['Date'],
                'entry_price': r['Close'],
                'side': 'LONG' if r['pred']==1 else 'SHORT',
                'atr': r['ATR']
            }
            if pos['atr'] <= 0 or np.isnan(pos['atr']):
                pos['atr'] = 1e-6
            if pos['side']=='LONG':
                pos['target'] = pos['entry_price'] + target_atr * pos['atr']
                pos['sl'] = pos['entry_price'] - sl_atr * pos['atr']
            else:
                pos['target'] = pos['entry_price'] - target_atr * pos['atr']
                pos['sl'] = pos['entry_price'] + sl_atr * pos['atr']
            continue

        if pos is not None:
            price = r['Close']
            exited = False
            exit_price = price
            exit_reason = None
            holding = i - pos['entry_idx']
            if pos['side']=='LONG':
                if price >= pos['target']:
                    exit_price = pos['target']; exit_reason='Target Hit'; exited=True
                elif price <= pos['sl']:
                    exit_price = pos['sl']; exit_reason='SL Hit'; exited=True
            else:
                if price <= pos['target']:
                    exit_price = pos['target']; exit_reason='Target Hit'; exited=True
                elif price >= pos['sl']:
                    exit_price = pos['sl']; exit_reason='SL Hit'; exited=True
            # forced exit
            if not exited and holding >= max_hold:
                exit_price = price; exit_reason='Max Hold'; exited=True
            # exit on opposite prediction at same time (conservative)
            if not exited and r['pred'] != 0:
                if (pos['side']=='LONG' and r['pred']==-1) or (pos['side']=='SHORT' and r['pred']==1):
                    exit_price = price; exit_reason='Opposite Pred'; exited=True

            if exited:
                pnl = (exit_price - pos['entry_price']) if pos['side']=='LONG' else (pos['entry_price'] - exit_price)
                pnl_pct = (pnl / pos['entry_price']) * 100
                trades.append({
                    'entry_date': pos['entry_date'],
                    'entry_price': pos['entry_price'],
                    'exit_date': r['Date'],
                    'exit_price': exit_price,
                    'side': pos['side'],
                    'target': pos['target'],
                    'sl': pos['sl'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_bars': holding,
                    'exit_reason': exit_reason
                })
                pos = None
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {'total_trades':0,'positive_trades':0,'negative_trades':0,'accuracy':0.0,'total_pnl':0.0,'total_pnl_pct':0.0,'strategy_points':0.0,'bh_points':0.0}
    else:
        pos_ct = (trades_df['pnl']>0).sum()
        neg_ct = (trades_df['pnl']<=0).sum()
        total_pnl = trades_df['pnl'].sum()
        buyhold = df2['Close'].iloc[-1] - df2['Close'].iloc[0]
        summary = {
            'total_trades': int(len(trades_df)),
            'positive_trades': int(pos_ct),
            'negative_trades': int(neg_ct),
            'accuracy': float(pos_ct / len(trades_df)),
            'total_pnl': float(total_pnl),
            'total_pnl_pct': float((total_pnl / float(capital)) * 100),
            'strategy_points': float(trades_df['pnl'].sum()),
            'bh_points': float(buyhold)
        }
    return trades_df, summary

# --------------------------
# Walk-forward training + OOS evaluation (retrain RF per fold)
# --------------------------
def walk_forward_rf(df, features, label_col='label', n_splits=4, test_frac=0.2, rf_params=None, prob_threshold=0.6):
    """
    Chronological rolling walk-forward:
      - split data into n_splits chronological test chunks
      - for each fold: train on data before test chunk, predict on test chunk
      - collect OOS predictions and per-fold metrics
    """
    if rf_params is None:
        rf_params = {'n_estimators':200,'random_state':42}
    n = len(df)
    fold_size = int(n / (n_splits + 1))
    all_oos_preds = pd.Series(index=df.index, dtype=float)
    fold_results = []
    feature_importances = []
    for k in range(n_splits):
        train_end = (k+1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end - test_start < 10:
            continue
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        X_train = train_df[features].values
        y_train = train_df[label_col].values
        X_test = test_df[features].values
        # drop samples where label==0? We'll train multiclass {1,0,-1}
        clf = RandomForestClassifier(**rf_params, n_jobs=-1, class_weight='balanced')
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        # classes order
        classes = clf.classes_
        # map to predictions: if prob of 1 >= prob_threshold -> predict 1, elif prob(-1) >= prob_threshold -> -1 else 0
        preds = []
        for p in proba:
            pred = 0
            # identify index of class 1 and -1 in classes
            p1 = p[classes.tolist().index(1)] if 1 in classes else 0
            pm1 = p[classes.tolist().index(-1)] if -1 in classes else 0
            if p1 >= prob_threshold and p1 > pm1:
                pred = 1
            elif pm1 >= prob_threshold and pm1 > p1:
                pred = -1
            else:
                pred = 0
            preds.append(pred)
        preds = np.array(preds)
        # place into full-index series
        all_oos_preds.iloc[test_df.index] = preds
        # metrics
        y_true = test_df[label_col].values
        # only for non-zero events
        mask_nonzero = y_true != 0
        acc = accuracy_score(y_true[mask_nonzero], preds[mask_nonzero]) if mask_nonzero.sum()>0 else 0.0
        pos = precision_score(y_true, preds, labels=[1], average='macro', zero_division=0)
        neg = precision_score(y_true, preds, labels=[-1], average='macro', zero_division=0)
        fold_results.append({'fold':k+1, 'train_end_idx':train_end, 'test_start':test_start, 'test_end':test_end,
                             'test_acc_nonzero': acc, 'prec_long': pos, 'prec_short': neg, 'test_trades': (preds!=0).sum()})
        feature_importances.append(clf.feature_importances_)
    # For remaining indices (not predicted), fill 0
    all_oos_preds = all_oos_preds.fillna(0).astype(int)
    # aggregate feature importances
    if feature_importances:
        avg_imp = np.mean(feature_importances, axis=0)
    else:
        avg_imp = np.zeros(len(features))
    return all_oos_preds, pd.DataFrame(fold_results), avg_imp

# --------------------------
# Heatmap helper
# --------------------------
def monthly_yearly(df):
    df = df.copy().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    month_close = df.set_index('Date').resample('M')['Close'].last().reset_index()
    month_close['Year'] = month_close['Date'].dt.year
    month_close['Month'] = month_close['Date'].dt.month
    month_close['ret'] = month_close['Close'].pct_change() * 100
    pivot_month = month_close.pivot(index='Year', columns='Month', values='ret')
    year_close = df.set_index('Date').resample('Y')['Close'].last().reset_index()
    year_close['ret'] = year_close['Close'].pct_change() * 100
    pivot_year = year_close[['Date','ret']].set_index(year_close['Date'].dt.year)['ret']
    return pivot_month, pivot_year

# --------------------------
# UI: Inputs
# --------------------------
with st.sidebar:
    st.header("Data & Model Settings")
    data_source = st.selectbox("Primary data source", ["Upload CSV (recommended)", "yfinance (optional)"])
    uploaded_file = st.file_uploader("Upload CSV (Date,Open,High,Low,Close,Volume)", type=['csv']) if data_source.startswith("Upload") else None
    yf_ticker = st.text_input("yfinance ticker", value="^NSEI")
    interval = st.selectbox("Interval (yfinance)", ["1d","60m","15m"], index=0)
    use_dates = st.checkbox("Use start/end (yfinance)", True)
    if use_dates:
        start = st.date_input("Start date", value=datetime.now().date() - timedelta(days=365*5))
        end = st.date_input("End date", value=datetime.now().date())
        start_s = start.strftime("%Y-%m-%d"); end_s = end.strftime("%Y-%m-%d")
    else:
        period = st.selectbox("Period (yfinance)", ["1y","2y","5y","10y","max"], index=1)
        start_s = end_s = None
    bb_n = st.number_input("BB/KC window (n)", value=20, min_value=5)
    atr_n = st.number_input("ATR period", value=14, min_value=1)
    ema_short = st.number_input("EMA short (for features)", value=20, min_value=5)
    ema_long = st.number_input("EMA long (regime)", value=50, min_value=10)
    st.markdown("---")
    st.subheader("Label & Backtest")
    horizon = st.number_input("Prediction horizon (bars)", value=5, min_value=1)
    up_thr = st.number_input("Up threshold (fraction)", value=0.02, format="%.4f")
    down_thr = st.number_input("Down threshold (fraction)", value=-0.02, format="%.4f")
    prob_threshold = st.slider("Probability threshold for signal (RF)", 0.50, 0.95, 0.60, 0.05)
    target_atr = st.number_input("Target (x ATR)", value=2.0)
    sl_atr = st.number_input("SL (x ATR)", value=1.0)
    max_hold = st.number_input("Max holding bars", value=12, min_value=1)
    capital = st.number_input("Capital (for %)", value=100000)
    st.markdown("---")
    st.subheader("Walk-forward")
    wf_splits = st.slider("WF splits", 3, 8, value=4)
    run_button = st.button("Fetch / Run (safe single click)")

# --------------------------
# Data load (explicit)
# --------------------------
if 'raw' not in st.session_state:
    st.session_state.raw = pd.DataFrame()

if run_button:
    if data_source.startswith("Upload"):
        if uploaded_file is None:
            st.error("Please upload CSV file.")
            st.stop()
        raw = pd.read_csv(uploaded_file)
        if 'Date' not in raw.columns and 'date' in raw.columns:
            raw.rename(columns={'date':'Date'}, inplace=True)
        raw['Date'] = pd.to_datetime(raw['Date'])
        for c in ['Open','High','Low','Close']:
            if c not in raw.columns:
                st.error(f"CSV missing column {c}")
                st.stop()
        if 'Volume' not in raw.columns:
            raw['Volume'] = 0
        raw = raw[['Date','Open','High','Low','Close','Volume']].dropna().reset_index(drop=True)
        st.session_state.raw = raw.copy()
    else:
        raw = download_yf(yf_ticker, interval=interval, start=start_s, end=end_s, period=None)
        if raw.empty:
            st.error("yfinance returned no data. Try upload or different ticker/period.")
            st.stop()
        st.session_state.raw = raw.copy()
    st.success("Data loaded into session.")

df_raw = st.session_state.raw.copy()
if df_raw.empty:
    st.info("No data loaded yet. Upload CSV or use yfinance and press 'Fetch / Run'.")
    st.stop()

# --------------------------
# Preprocess and features
# --------------------------
# detect volume availability
use_volume = True
if (df_raw['Volume'].sum() == 0) or df_raw['Volume'].isna().all():
    use_volume = False

df_feat = add_features(df_raw, use_volume_features=use_volume, bb_n=bb_n, atr_n=atr_n, ema_short=ema_short, ema_long=ema_long)

# build labels
df_labeled = build_labels(df_feat, horizon=horizon, up_threshold=up_thr, down_threshold=down_thr)

# assemble feature list (drop target columns)
feat_cols = ['ret_1','ret_3','ret_5','TR_PCT','body_strength','range_ratio','ema_slope','bbw','mom_3','mom_5','mom_accel']
if use_volume:
    feat_cols += ['vol_change','OBV','obv_20_ma']
# ensure features present
feat_cols = [c for c in feat_cols if c in df_labeled.columns]

# --------------------------
# Walk-forward RF training + OOS predictions
# --------------------------
with st.spinner("Performing walk-forward training & OOS predictions..."):
    oos_preds, wf_df, avg_imp = walk_forward_rf(df_labeled, features=feat_cols, label_col='label', n_splits=wf_splits, prob_threshold=prob_threshold)

# align preds with labeled df
df_labeled['pred'] = oos_preds.values

# --------------------------
# Backtest using predictions (last-close entries)
# --------------------------
with st.spinner("Backtesting predicted signals..."):
    trades_df, summary = backtest_predictions(df_labeled, df_labeled['pred'], target_atr=target_atr, sl_atr=sl_atr, max_hold=max_hold, capital=capital)

# --------------------------
# Heatmaps
# --------------------------
pivot_month, pivot_year = monthly_yearly(df_raw)

# --------------------------
# UI: Results display
# --------------------------
# top metrics
col1, col2, col3 = st.columns([1.2,1,1])
with col1:
    st.subheader("Backtest Summary")
    st.write(f"- Total trades: **{summary['total_trades']}**")
    st.write(f"- Positive trades: **{summary['positive_trades']}**")
    st.write(f"- Negative trades: **{summary['negative_trades']}**")
    st.write(f"- Accuracy: **{summary['accuracy']*100:.2f}%**")
with col2:
    st.subheader("PnL & Comparison")
    st.write(f"- Strategy total PnL (points): **{summary['strategy_points']:.2f}**")
    st.write(f"- Strategy PnL (% of capital): **{summary['total_pnl_pct']:.4f}%**")
    st.write(f"- Buy & Hold (first->last points): **{summary['bh_points']:.2f}**")
with col3:
    st.subheader("Walk-Forward (folds)")
    if wf_df.empty:
        st.write("No WF folds produced (insufficient data?)")
    else:
        st.write(f"- folds: {len(wf_df)}")
        st.write(f"- avg OOS trades per fold: {wf_df['test_trades'].mean():.1f}")
        st.dataframe(wf_df, use_container_width=True, height=160)

# Live recommendation (use last trained fold model logic)
st.subheader("Live Recommendation (based on last available features & last fold RF)")

# For live prediction, train RF on entire labeled data with same params then predict last row (conservative)
if len(df_labeled) > 50:
    X_all = df_labeled[feat_cols].values
    y_all = df_labeled['label'].values
    rf_live = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_live.fit(X_all, y_all)
    last_feat = df_feat.tail(1)
    # need to create same features & label-building context for last row
    last_row = add_features(df_raw.tail(1).append(df_raw.tail(1), ignore_index=True), use_volume_features=use_volume, bb_n=bb_n, atr_n=atr_n, ema_short=ema_short, ema_long=ema_long).iloc[0:1]
    # fallback: use last row from df_feat
    last_row = df_feat.iloc[[-1]][feat_cols]
    proba = rf_live.predict_proba(last_row.values)[0]
    classes = rf_live.classes_
    p_up = proba[classes.tolist().index(1)] if 1 in classes else 0
    p_dn = proba[classes.tolist().index(-1)] if -1 in classes else 0
    pred_label = 1 if p_up>=prob_threshold and p_up>p_dn else (-1 if p_dn>=prob_threshold and p_dn>p_up else 0)
    if pred_label == 0:
        st.info("No confident prediction (probabilities below threshold).")
    else:
        side = 'LONG' if pred_label==1 else 'SHORT'
        entry = float(df_raw['Close'].iloc[-1])
        atr = float(df_feat['ATR'].iloc[-1])
        target = entry + target_atr * atr if side == 'LONG' else entry - target_atr * atr
        sl = entry - sl_atr * atr if side == 'LONG' else entry + sl_atr * atr
        st.json({
            'entry_date_time': str(df_raw['Date'].iloc[-1]),
            'side': side,
            'levels': entry,
            'target': float(target),
            'sl': float(sl),
            'prob_up': float(p_up),
            'prob_down': float(p_dn)
        })
else:
    st.info("Not enough labeled data to train live model.")

# Chart with signals (predictions overlay)
st.subheader("Price chart with predicted triggers (OOS preds)")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df_raw['Date'], df_raw['Close'], label='Close')
# overlay predicted long/short from df_labeled (align indices)
label_idx = df_labeled.index
ax.scatter(df_labeled.loc[df_labeled['pred']==1,'Date'], df_labeled.loc[df_labeled['pred']==1,'Close'], marker='^', label='Pred Long', s=40, color='green')
ax.scatter(df_labeled.loc[df_labeled['pred']==-1,'Date'], df_labeled.loc[df_labeled['pred']==-1,'Close'], marker='v', label='Pred Short', s=40, color='red')
ax.set_title("Price & Predicted Triggers (OOS)")
ax.legend()
st.pyplot(fig)

# Top/bottom trades + full trade log
st.subheader("Top 5 / Bottom 5 trades")
if trades_df.empty:
    st.info("No trades executed by predictions.")
else:
    top5 = trades_df.sort_values(by='pnl', ascending=False).head(5)
    bot5 = trades_df.sort_values(by='pnl', ascending=True).head(5)
    c1, c2 = st.columns(2)
    with c1:
        st.write("Top 5 profitable trades")
        st.dataframe(top5, height=250)
    with c2:
        st.write("Bottom 5 losing trades")
        st.dataframe(bot5, height=250)

st.subheader("Full trade log (scrollable)")
if trades_df.empty:
    st.write("No trades to show.")
else:
    st.dataframe(trades_df.reset_index(drop=True), height=300, use_container_width=True)

# Raw original data
st.subheader("Original Raw Data (scrollable)")
st.dataframe(df_raw.reset_index(drop=True), height=300, use_container_width=True)

# Feature importances
st.subheader("Feature importances (avg across WF folds)")
if len(avg_imp)>0:
    imp_df = pd.DataFrame({'feature': feat_cols, 'importance': avg_imp})
    imp_df = imp_df.sort_values('importance', ascending=False)
    st.dataframe(imp_df, use_container_width=True)
    fig_imp, ax_imp = plt.subplots(figsize=(6,3))
    ax_imp.barh(imp_df['feature'].iloc[::-1], imp_df['importance'].iloc[::-1])
    ax_imp.set_title("Feature Importances")
    st.pyplot(fig_imp)
else:
    st.write("No feature importance (WF produced no models).")

# Heatmaps
st.subheader("Monthly returns heatmap")
if not pivot_month.empty:
    fig2, ax2 = plt.subplots(figsize=(10,4))
    im = ax2.imshow(pivot_month.fillna(0).values, aspect='auto', cmap='RdYlGn', vmin=-30, vmax=30)
    ax2.set_yticks(np.arange(pivot_month.shape[0])); ax2.set_yticklabels(pivot_month.index)
    ax2.set_xticks(np.arange(12)); ax2.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12])
    ax2.set_title("Monthly returns (%)")
    plt.colorbar(im, ax=ax2)
    st.pyplot(fig2)
else:
    st.write("Not enough data for monthly heatmap.")

st.subheader("Yearly returns")
if not pivot_year.empty:
    st.dataframe(pivot_year.rename("Yearly % Return").to_frame())
else:
    st.write("Not enough data for yearly returns.")

# Notes
st.markdown("---")
