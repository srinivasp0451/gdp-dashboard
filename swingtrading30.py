import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Universal Swing Trading Backtest & Live Signal App")

# ----------------------------- #
# Load Data
# ----------------------------- #
upload_file = st.file_uploader("Upload CSV file with OHLCV", type=["csv"])
if upload_file is not None:
    df = pd.read_csv(upload_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
else:
    st.warning("Please upload CSV file")
    st.stop()

# ----------------------------- #
# Feature Engineering
# ----------------------------- #
def add_features(df, ema_short=10, ema_long=50, atr_period=14, rsi_period=14):
    df['EMA_short'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=ema_long, adjust=False).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(atr_period).mean().fillna(0)
    delta = df['Close'].diff()
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(rsi_period).mean().fillna(0)
    avg_loss = pd.Series(loss).rolling(rsi_period).mean().fillna(0)
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    if 'Volume' not in df.columns or df['Volume'].sum()==0:
        df['Volume'] = df['Close'].pct_change().abs()*100000  # surrogate volume
    df.fillna(0, inplace=True)
    return df

df = add_features(df)

# ----------------------------- #
# Label Generation
# ----------------------------- #
horizon = 5
up_pct = 1.0
down_pct = 1.0
df['Future_Close'] = df['Close'].shift(-horizon)
df['Return'] = ((df['Future_Close'] - df['Close'])/df['Close'])*100
df['Label'] = np.where(df['Return']>=up_pct,1,
                       np.where(df['Return']<=-down_pct,-1,0))
df.drop(['Future_Close','Return'], axis=1, inplace=True)

# ----------------------------- #
# Features & Labels
# ----------------------------- #
features = ['Close','Open','High','Low','Volume','EMA_short','EMA_long','ATR','RSI']
X = df[features]
y = df['Label']

# ----------------------------- #
# Walkforward Prediction
# ----------------------------- #
splits = TimeSeriesSplit(n_splits=4)
df['Pred'] = 0
df['Prob'] = 0.0
fold_num = 1
progress_bar = st.progress(0)
for train_idx, test_idx in splits.split(X):
    st.info(f"Training fold {fold_num}/4...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    df.loc[test_idx,'Pred'] = model.predict(X.iloc[test_idx])
    df.loc[test_idx,'Prob'] = model.predict_proba(X.iloc[test_idx]).max(axis=1)
    fold_num += 1
    #progress_bar.progress(fold_num*25)
    # Calculate progress % safely
    progress_pct = int((fold_num / total_folds) * 100)
    progress_bar.progress(min(progress_pct, 100))  # never exceed 100

# ----------------------------- #
# Backtest Simulation
# ----------------------------- #
def backtest(df, horizon=5, atr_mult=1.5):
    trades = []
    for i in range(len(df)-horizon):
        row = df.iloc[i]
        if row['Pred']==1:
            entry = row['Close']
            sl = entry - atr_mult*row['ATR']
            target = entry + atr_mult*row['ATR']
            exit_price = min(df['Low'].iloc[i:i+horizon].min(), sl)
            exit_price = target if df['High'].iloc[i:i+horizon].max()>=target else exit_price
            pnl = exit_price-entry
            trades.append({'Date':row['Date'],'Side':'Long','Entry':entry,
                           'SL':sl,'Target':target,'Exit':exit_price,'PnL':pnl,
                           'Reason':'Signal','Prob':row['Prob']})
        elif row['Pred']==-1:
            entry = row['Close']
            sl = entry + atr_mult*row['ATR']
            target = entry - atr_mult*row['ATR']
            exit_price = max(df['High'].iloc[i:i+horizon].max(), sl)
            exit_price = target if df['Low'].iloc[i:i+horizon].min()<=target else exit_price
            pnl = entry-exit_price
            trades.append({'Date':row['Date'],'Side':'Short','Entry':entry,
                           'SL':sl,'Target':target,'Exit':exit_price,'PnL':pnl,
                           'Reason':'Signal','Prob':row['Prob']})
    trades_df = pd.DataFrame(trades)
    return trades_df

trades_df = backtest(df)

# ----------------------------- #
# Backtest Summary
# ----------------------------- #
total_trades = len(trades_df)
pos_trades = len(trades_df[trades_df['PnL']>0])
neg_trades = total_trades - pos_trades
acc = (pos_trades/total_trades*100) if total_trades>0 else 0
total_pnl = trades_df['PnL'].sum()
buy_hold = df['Close'].iloc[-1]-df['Close'].iloc[0]

st.subheader("Backtest Summary")
st.write({
    'Total trades': total_trades,
    'Positive trades': pos_trades,
    'Negative trades': neg_trades,
    'Accuracy': acc,
    'Strategy total PnL (points)': total_pnl,
    'Buy & Hold points': buy_hold
})

# ----------------------------- #
# Display trades & original data
# ----------------------------- #
st.subheader("Top 5 / Bottom 5 Trades")
st.dataframe(pd.concat([trades_df.head(), trades_df.tail()]))

st.subheader("Full Trades Table")
st.dataframe(trades_df)

st.subheader("Original Data")
st.dataframe(df)

# ----------------------------- #
# Monthly & Yearly Heatmaps
# ----------------------------- #
df['Month'] = df['Date'].dt.to_period('M')
df['Year'] = df['Date'].dt.to_period('Y')
monthly = df.groupby('Month')['Close'].last().pct_change().fillna(0)*100
yearly = df.groupby('Year')['Close'].last().pct_change().fillna(0)*100

fig, axes = plt.subplots(1,2,figsize=(14,4))
sns.heatmap(monthly.values.reshape(-1,1), annot=True, fmt=".2f", cmap='RdYlGn', ax=axes[0])
axes[0].set_yticks(range(len(monthly)))
axes[0].set_yticklabels(monthly.index.astype(str))
axes[0].set_title("Monthly Returns (%)")
sns.heatmap(yearly.values.reshape(-1,1), annot=True, fmt=".2f", cmap='RdYlGn', ax=axes[1])
axes[1].set_yticks(range(len(yearly)))
axes[1].set_yticklabels(yearly.index.astype(str))
axes[1].set_title("Yearly Returns (%)")
st.pyplot(fig)
