# streamlit_swing_trading_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Universal Swing Trading App", layout="wide")

# ---------------------- UTILITY FUNCTIONS ----------------------

@st.cache_data
def fetch_data_yf(ticker, period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    df['Volume'] = df['Volume'].replace(0, np.nan)  # handle zero volumes
    return df

def calculate_features(df, ema_short=10, ema_long=50, atr_window=14, rsi_window=14):
    df = df.copy()
    
    # EMA
    df['EMA_short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=ema_long, adjust=False).mean()
    
    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(atr_window).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100/(1+rs))
    
    # Momentum
    df['Momentum'] = df['Close'].pct_change(periods=1)
    
    df.fillna(0,inplace=True)
    return df

def label_data(df, horizon=5, pct_thr=0.02):
    df = df.copy()
    df['future_close'] = df['Close'].shift(-horizon)
    df['pct_change'] = (df['future_close'] - df['Close']) / df['Close']
    df['label'] = 0
    df.loc[df['pct_change'] >= pct_thr, 'label'] = 1
    df.loc[df['pct_change'] <= -pct_thr, 'label'] = -1
    df.dropna(inplace=True)
    return df

def backtest_model(df, features, n_splits=4, capital=100000):
    X = df[features].values
    y = df['label'].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    trades = []
    oos_pnls = []
    oos_accs = []
    fold = 1
    total_steps = n_splits
    progress_bar = st.progress(0)
    for train_index, test_index in tscv.split(X):
        st.write(f"Walkforward fold {fold}/{total_steps}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        oos_accs.append(accuracy)
        
        df_test = df.iloc[test_index].copy()
        df_test['pred'] = y_pred
        
        # Simulate trades
        for idx, row in df_test.iterrows():
            if row['pred'] == 0: continue
            entry = row['Close']
            sl = entry * 0.99 if row['pred']==1 else entry * 1.01
            target = entry * 1.02 if row['pred']==1 else entry * 0.98
            exit_price = target if row['pred']==1 else sl
            pnl = (exit_price - entry) if row['pred']==1 else (entry - exit_price)
            trades.append({'Date': row['Date'], 'Signal': 'Long' if row['pred']==1 else 'Short',
                           'Entry': entry, 'Target': target, 'SL': sl, 'PnL': pnl,
                           'Reason': 'Model prediction', 'Probability': max(model.predict_proba([row[features].values])[0])})
        
        oos_pnls.append(sum([t['PnL'] for t in trades]))
        fold +=1
        progress_bar.progress(int(fold/total_steps*100))
    return trades, oos_pnls, oos_accs

def generate_heatmaps(df):
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_returns = df.groupby('YearMonth')['Close'].last().pct_change().fillna(0)
    df['Year'] = df['Date'].dt.year
    yearly_returns = df.groupby('Year')['Close'].last().pct_change().fillna(0)
    return monthly_returns, yearly_returns

# ---------------------- STREAMLIT UI ----------------------

st.title("🔥 Universal Swing Trading App")

# Data Input
st.sidebar.header("Data Input")
data_option = st.sidebar.radio("Select data source:", ["Upload CSV/Excel", "YFinance"])
if data_option=="Upload CSV/Excel":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
else:
    ticker = st.sidebar.text_input("YFinance Ticker (e.g., ^NSEI)", value="^NSEI")
    df = fetch_data_yf(ticker)
    df['Date'] = pd.to_datetime(df['Date'])

st.write("### Original Data (scrollable)")
st.dataframe(df, use_container_width=True)

# Feature Engineering and Labeling
st.sidebar.header("Model & Feature Options")
atr_window = st.sidebar.number_input("ATR Window", value=14, min_value=5,max_value=50)
ema_short = st.sidebar.number_input("EMA Short", value=10)
ema_long = st.sidebar.number_input("EMA Long", value=50)
rsi_window = st.sidebar.number_input("RSI Window", value=14)
horizon = st.sidebar.number_input("Label Horizon (bars)", value=5)
pct_thr = st.sidebar.number_input("Label % Threshold", value=2.0)/100

df_feat = calculate_features(df, ema_short, ema_long, atr_window, rsi_window)
df_label = label_data(df_feat, horizon, pct_thr)

# Features for model
features = ['EMA_short','EMA_long','ATR','RSI','Momentum']

if st.button("Run Swing Strategy"):
    st.info("Running Walkforward Backtest & Live Signals...")
    trades, oos_pnls, oos_accs = backtest_model(df_label, features)
    
    # Backtest Summary
    total_trades = len(trades)
    positive_trades = len([t for t in trades if t['PnL']>0])
    negative_trades = len([t for t in trades if t['PnL']<=0])
    accuracy = round(positive_trades/total_trades*100,2) if total_trades>0 else 0
    total_pnl = sum([t['PnL'] for t in trades])
    total_pnl_pct = round(total_pnl/100000*100,4)
    buy_hold = df_label['Close'].iloc[-1] - df_label['Close'].iloc[0]
    
    st.subheader("Backtest Summary")
    st.write(f"Total trades: {total_trades}")
    st.write(f"Positive trades: {positive_trades}")
    st.write(f"Negative trades: {negative_trades}")
    st.write(f"Accuracy: {accuracy}%")
    st.write(f"Strategy total PnL (points): {round(total_pnl,2)}")
    st.write(f"Strategy PnL (% of capital): {total_pnl_pct}%")
    st.write(f"Buy & Hold (first->last points): {round(buy_hold,2)}")
    
    # Walkforward Summary
    st.subheader("Walkforward Summary")
    st.write(f"WF folds: {len(oos_pnls)}")
    st.write(f"Avg OOS pnl per fold: {round(np.mean(oos_pnls),2)}")
    st.write(f"Avg OOS accuracy: {round(np.mean(oos_accs)*100,2)}%")
    
    # Trades Table
    st.subheader("Trade Details")
    trades_df = pd.DataFrame(trades)
    st.write("Top 5 trades")
    st.dataframe(trades_df.head())
    st.write("Bottom 5 trades")
    st.dataframe(trades_df.tail())
    st.write("Full trades (scrollable)")
    st.dataframe(trades_df, use_container_width=True)
    
    # Heatmaps
    monthly_returns, yearly_returns = generate_heatmaps(df_label)
    st.subheader("Monthly Returns Heatmap")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(monthly_returns.values.reshape(-1,1), annot=True, fmt=".2%", cmap="RdYlGn", ax=ax)
    st.pyplot(fig)
    
    st.subheader("Yearly Returns Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    sns.heatmap(yearly_returns.values.reshape(-1,1), annot=True, fmt=".2%", cmap="RdYlGn", ax=ax2)
    st.pyplot(fig2)
    
    # Live Recommendation
    st.subheader("Live Recommendation (Last Candle)")
    last_row = df_label.iloc[-1]
    X_live = last_row[features].values.reshape(1,-1)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(df_label[features], df_label['label'])
    pred = model.predict(X_live)[0]
    prob = max(model.predict_proba(X_live)[0])
    entry = last_row['Close']
    target = entry*1.02 if pred==1 else entry*0.98
    sl = entry*0.99 if pred==1 else entry*1.01
    signal = "Long" if pred==1 else "Short" if pred==-1 else "No Signal"
    st.write({
        "Date": last_row['Date'],
        "Signal": signal,
        "Entry": round(entry,2),
        "Target": round(target,2),
        "SL": round(sl,2),
        "Reason": "Model prediction",
        "Probability": round(prob*100,2)
    })
