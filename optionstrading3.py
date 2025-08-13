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

# Clean col names
df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]

# --- Split CE/PE based on STRIKE col position ---
strike_index = df.columns.get_loc('STRIKE')
ce_cols = df.columns[:strike_index].tolist() + ['STRIKE']
pe_cols = ['STRIKE'] + df.columns[strike_index+1:].tolist()

df_ce = df[ce_cols].copy()
df_pe = df[pe_cols].copy()

df_ce.columns = ['CE ' + c if c != 'STRIKE' else c for c in df_ce.columns]
df_pe.columns = ['PE ' + c if c != 'STRIKE' else c for c in df_pe.columns]

df_all = pd.merge(df_ce, df_pe, on='STRIKE', how='inner')

# -------- NORMALISE COLUMN NAMES --------
df_all.columns = (
    df_all.columns
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
    .str.upper()
)

# -------- MAKE NUMERIC --------
for col in df_all.columns:
    if col != 'STRIKE':
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
df_all['STRIKE'] = pd.to_numeric(df_all['STRIKE'], errors='coerce')
df_all = df_all.dropna(subset=['STRIKE'])

# -------- FIND COLUMN NAMES DYNAMICALLY --------
def get_col(part):
    for col in df_all.columns:
        if part in col:
            return col
    st.error(f"Column containing '{part}' not found!")
    st.stop()

col_ce_oi      = get_col('CE OI')
col_pe_oi      = get_col('PE OI')
col_ce_chngoi  = get_col('CE CHNG IN OI')
col_pe_chngoi  = get_col('PE CHNG IN OI')
col_ce_vol     = get_col('CE VOLUME')
col_pe_vol     = get_col('PE VOLUME')
col_ce_iv      = get_col('CE IV')
col_pe_iv      = get_col('PE IV')
col_ce_ltp     = get_col('CE LTP')
col_pe_ltp     = get_col('PE LTP')

# -------- SAFE SPOT DETECTION --------
spot_guess = None

if col_ce_ltp in df_all.columns and col_pe_ltp in df_all.columns:
    df_all['LTP_DIFF'] = abs(df_all[col_ce_ltp] - df_all[col_pe_ltp])
    if not df_all['LTP_DIFF'].isna().all():
        spot_guess = float(df_all.loc[df_all['LTP_DIFF'].idxmin(), 'STRIKE'])

if spot_guess is None or pd.isna(spot_guess):
    df_all['TOT_VOL'] = df_all[col_ce_vol].fillna(0) + df_all[col_pe_vol].fillna(0)
    spot_guess = float(df_all.loc[df_all['TOT_VOL'].idxmax(), 'STRIKE'])

if spot_guess is None or pd.isna(spot_guess):
    spot_guess = float(df_all['STRIKE'].median())

st.info(f"Estimated Spot Price ≈ **{spot_guess}**")

# -------- TRADE FINDER --------
def find_best(df_all, spot):
    window = 200
    near_df = df_all[df_all['STRIKE'].between(spot - window, spot + window)]
    trades = []

    # CE candidate
    ce_df = near_df[(near_df[col_ce_chngoi] > 0) & (near_df[col_ce_iv] > 10)]
    best_ce = ce_df.sort_values([col_ce_chngoi, col_ce_vol, col_ce_iv], ascending=False).head(1)

    if not best_ce.empty:
        row = best_ce.iloc[0]
        entry = row[col_ce_ltp]
        trades.append({
            "Type": "BUY CE",
            "Strike": int(row['STRIKE']),
            "Entry": entry,
            "Target": round(entry * 1.35, 2),
            "SL": round(entry * 0.8, 2),
            "Prob%": min(90, round((row[col_ce_iv] / (row[col_ce_iv] + row[col_pe_iv])) * 100, 2)),
            "Logic": f"High CE Vol {row[col_ce_vol]} & OI +{row[col_ce_chngoi]}, IV {row[col_ce_iv]}%"
        })

    # PE candidate
    pe_df = near_df[(near_df[col_pe_chngoi] > 0) & (near_df[col_pe_iv] > 10)]
    best_pe = pe_df.sort_values([col_pe_chngoi, col_pe_vol, col_pe_iv], ascending=False).head(1)

    if not best_pe.empty:
        row = best_pe.iloc[0]
        entry = row[col_pe_ltp]
        trades.append({
            "Type": "BUY PE",
            "Strike": int(row['STRIKE']),
            "Entry": entry,
            "Target": round(entry * 1.35, 2),
            "SL": round(entry * 0.8, 2),
            "Prob%": min(90, round((row[col_pe_iv] / (row[col_pe_iv] + row[col_ce_iv])) * 100, 2)),
            "Logic": f"High PE Vol {row[col_pe_vol]} & OI +{row[col_pe_chngoi]}, IV {row[col_pe_iv]}%"
        })

    return pd.DataFrame(trades)

rec_df = find_best(df_all, spot_guess)

st.subheader("📈 Live Recommendations")
if rec_df.empty:
    st.warning("No suitable CE/PE trades found near spot.")
else:
    st.dataframe(rec_df)
    for _, row in rec_df.iterrows():
        st.markdown(f"**{row['Type']} {row['Strike']}** — Entry `{row['Entry']}` | Target `{row['Target']}` | SL `{row['SL']}` | Prob **{row['Prob%']}%**\n\nLogic: {row['Logic']}")

# -------- STRADDLE PREMIUM & % CHANGE --------
st.subheader("Straddle Premium & % Change")
premiums, pct_changes = [], []
strikes = sorted(df_all['STRIKE'].unique())
for i, s in enumerate(strikes):
    r = df_all[df_all['STRIKE'] == s]
    total = r[col_ce_ltp].values[0] + r[col_pe_ltp].values[0]
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

# -------- OI / OI Change / Volume Charts --------
def plot_bar(title, ce_vals, pe_vals, strikes):
    fig, ax = plt.subplots()
    ax.bar(strikes, ce_vals, color='green', alpha=0.6, label='CE')
    ax.bar(strikes, -np.array(pe_vals), color='red', alpha=0.6, label='PE')
    ax.set_title(f"{title} (CE up, PE down)")
    ax.set_xlabel("Strike")
    ax.legend()
    st.pyplot(fig)

st.subheader("OI Bar Chart")
plot_bar("Open Interest", df_all[col_ce_oi], df_all[col_pe_oi], df_all['STRIKE'])

st.subheader("OI Change Bar Chart")
plot_bar("OI Change", df_all[col_ce_chngoi], df_all[col_pe_chngoi], df_all['STRIKE'])

st.subheader("Volume Bar Chart (Around Spot)")
near = df_all[df_all['STRIKE'].between(spot_guess - 250, spot_guess + 250)]
plot_bar("Volume", near[col_ce_vol], near[col_pe_vol], near['STRIKE'])
