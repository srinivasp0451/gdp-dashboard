import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Option Chain Live Analysis", layout="wide")

st.title("📊 NIFTY Option Chain Analysis + Live Recommendations")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("Upload Option Chain File (CSV or XLSX)", type=["csv", "xlsx"])
if not uploaded_file:
    st.warning("Please upload your option chain file first.")
    st.stop()

# -------- READ FILE --------
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file, skiprows=1)
else:
    df = pd.read_excel(uploaded_file, skiprows=1)

# Clean column names
df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]

# --- Auto-split CE/PE based on STRIKE location ---
strike_index = df.columns.get_loc('STRIKE')
ce_cols = df.columns[:strike_index].tolist() + ['STRIKE']
pe_cols = ['STRIKE'] + df.columns[strike_index+1:].tolist()

df_ce = df[ce_cols].copy()
df_pe = df[pe_cols].copy()

# Prefix columns
df_ce.columns = ['CE ' + c if c != 'STRIKE' else c for c in df_ce.columns]
df_pe.columns = ['PE ' + c if c != 'STRIKE' else c for c in df_pe.columns]

# Merge
df_all = pd.merge(df_ce, df_pe, on='STRIKE', how='inner')

# -------- NUMERIC CONVERSIONS --------
for col in df_all.columns:
    if col != 'STRIKE':
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
df_all['STRIKE'] = pd.to_numeric(df_all['STRIKE'], errors='coerce')
df_all = df_all.dropna(subset=['STRIKE'])

# -------- DYNAMIC SPOT DETECTION --------
spot_guess = None

# 1️⃣ ATM logic: strike with minimum CE/PE LTP difference
if 'CE LTP' in df_all.columns and 'PE LTP' in df_all.columns:
    df_all['LTP_DIFF'] = abs(df_all['CE LTP'] - df_all['PE LTP'])
    if not df_all['LTP_DIFF'].isna().all():
        atm_row = df_all.loc[df_all['LTP_DIFF'].idxmin()]
        spot_guess = float(atm_row['STRIKE'])

# 2️⃣ Fallback: Max total volume
if spot_guess is None or pd.isna(spot_guess):
    if 'CE VOLUME' in df_all.columns and 'PE VOLUME' in df_all.columns:
        df_all['TOT_VOL'] = df_all['CE VOLUME'].fillna(0) + df_all['PE VOLUME'].fillna(0)
        vol_row = df_all.loc[df_all['TOT_VOL'].idxmax()]
        spot_guess = float(vol_row['STRIKE'])

# 3️⃣ Final fallback: median strike
if spot_guess is None or pd.isna(spot_guess):
    spot_guess = float(df_all['STRIKE'].median())

st.info(f"Estimated Spot Price ≈ **{spot_guess}**")

# -------- FIND BEST TRADES --------
def find_best(df_all, spot):
    window = 200
    near_df = df_all[df_all['STRIKE'].between(spot - window, spot + window)]

    # CE candidate: high volume, positive OI change
    ce_df = near_df[(near_df['CE CHNG IN OI'] > 0) & (near_df['CE IV'] > 10)]
    best_ce = ce_df.sort_values(['CE CHNG IN OI','CE VOLUME','CE IV'], ascending=False).head(1)

    # PE candidate: high volume, positive OI change
    pe_df = near_df[(near_df['PE CHNG IN OI'] > 0) & (near_df['PE IV'] > 10)]
    best_pe = pe_df.sort_values(['PE CHNG IN OI','PE VOLUME','PE IV'], ascending=False).head(1)

    trades = []

    if not best_ce.empty:
        row = best_ce.iloc[0]
        entry = row['CE LTP']
        target = round(entry * 1.35, 2)
        sl = round(entry * 0.8, 2)
        prob = min(90, round((row['CE IV'] / (row['CE IV'] + row['PE IV'])) * 100, 2))
        logic = f"High CE volume ({row['CE VOLUME']}) with OI up by {row['CE CHNG IN OI']} and IV {row['CE IV']}% — upside bias."
        trades.append({"Type": "BUY CE", "Strike": int(row['STRIKE']), "Entry": entry, "Target": target, "SL": sl, "Prob%": prob, "Logic": logic})

    if not best_pe.empty:
        row = best_pe.iloc[0]
        entry = row['PE LTP']
        target = round(entry * 1.35, 2)
        sl = round(entry * 0.8, 2)
        prob = min(90, round((row['PE IV'] / (row['PE IV'] + row['CE IV'])) * 100, 2))
        logic = f"High PE volume ({row['PE VOLUME']}) with OI up by {row['PE CHNG IN OI']} and IV {row['PE IV']}% — downside bias."
        trades.append({"Type": "BUY PE", "Strike": int(row['STRIKE']), "Entry": entry, "Target": target, "SL": sl, "Prob%": prob, "Logic": logic})

    return pd.DataFrame(trades)

rec_df = find_best(df_all, spot_guess)

st.subheader("📈 Live Recommendations")
if rec_df.empty:
    st.warning("No suitable CE/PE trades found near spot based on current filter conditions.")
else:
    st.dataframe(rec_df)
    for _, row in rec_df.iterrows():
        st.markdown(f"**{row['Type']} {row['Strike']}** — Entry: `{row['Entry']}` | Target: `{row['Target']}` | SL: `{row['SL']}` | Prob: **{row['Prob%']}%**\n\n**Logic:** {row['Logic']}\n")

# -------- STRADDLE PREMIUM & % CHANGE --------
st.subheader("Straddle Premium & % Change vs Previous Strike")
premiums, pct_changes = [], [np.nan]
strikes = sorted(df_all['STRIKE'].unique())

for i, s in enumerate(strikes):
    r = df_all[df_all['STRIKE'] == s]
    ce, pe = r['CE LTP'].values[0], r['PE LTP'].values[0]
    total = ce + pe
    premiums.append(total)
    pct_changes.append(((total - premiums[i-1]) / premiums[i-1]) * 100 if i > 0 else np.nan)

fig, ax1 = plt.subplots()
ax1.bar([str(int(s)) for s in strikes], premiums, color='blue', alpha=0.6)
ax2 = ax1.twinx()
ax2.plot([str(int(s)) for s in strikes], pct_changes, color='orange', marker='o')
ax1.set_ylabel("Straddle Premium")
ax2.set_ylabel("% change")
ax1.set_title("Straddle Premium & % Change vs Strike")
st.pyplot(fig)

# -------- OI / OI CHANGE / VOLUME CHARTS --------
def plot_bar(title, ce_vals, pe_vals, strike_vals):
    fig, ax = plt.subplots()
    ax.bar(strike_vals, ce_vals, color='green', alpha=0.6, label='CE')
    ax.bar(strike_vals, -np.array(pe_vals), color='red', alpha=0.6, label='PE')
    ax.set_title(f"{title} (CE up, PE down)")
    ax.set_xlabel("Strike Price")
    ax.legend()
    st.pyplot(fig)

st.subheader("OI Bar Chart")
plot_bar("Open Interest", df_all['CE OI'], df_all['PE OI'], df_all['STRIKE'])

st.subheader("OI Change Bar Chart")
plot_bar("OI Change", df_all['CE CHNG IN OI'], df_all['PE CHNG IN OI'], df_all['STRIKE'])

st.subheader("Volume Bar Chart (Around Spot)")
near = df_all[df_all['STRIKE'].between(spot_guess - 250, spot_guess + 250)]
plot_bar("Volume", near['CE VOLUME'], near['PE VOLUME'], near['STRIKE'])
