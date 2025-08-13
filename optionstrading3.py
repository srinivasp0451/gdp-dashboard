import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Option Chain Safe Mode")
st.title("📊 NIFTY Option Chain Analysis — Safe Mode (No Crashes)")

def safe_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(',', '', regex=False).replace('-', np.nan),
        errors='coerce'
    )

# ---- File upload ----
uploaded_file = st.file_uploader("Upload Option Chain CSV or XLSX", type=['csv', 'xlsx'])
if not uploaded_file:
    st.stop()

# ---- Read file ----
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file, skiprows=1)
else:
    df = pd.read_excel(uploaded_file, skiprows=1)

# ---- Clean columns ----
df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]

# ---- Split CE/PE sides ----
strike_index = df.columns.get_loc('STRIKE')
df_ce = df[df.columns[:strike_index].tolist() + ['STRIKE']].copy()
df_pe = df[['STRIKE'] + df.columns[strike_index+1:].tolist()].copy()

df_ce.columns = ['CE ' + c if c != 'STRIKE' else c for c in df_ce.columns]
df_pe.columns = ['PE ' + c if c != 'STRIKE' else c for c in df_pe.columns]

df_all = pd.merge(df_ce, df_pe, on='STRIKE', how='inner')

# ---- Normalise column names ----
df_all.columns = (
    df_all.columns
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
    .str.upper()
)

# ---- Convert all numeric-like columns ----
for col in df_all.columns:
    df_all[col] = safe_numeric(df_all[col]) if col != 'STRIKE' else pd.to_numeric(df_all[col], errors='coerce')
df_all = df_all.dropna(subset=['STRIKE'])

# ---- Dynamic column lookup ----
def get_col(part):
    matches = [c for c in df_all.columns if part in c]
    return matches[0] if matches else None

col_ce_oi = get_col('CE OI')
col_pe_oi = get_col('PE OI')
col_ce_chngoi = get_col('CE CHNG IN OI')
col_pe_chngoi = get_col('PE CHNG IN OI')
col_ce_vol = get_col('CE VOLUME')
col_pe_vol = get_col('PE VOLUME')
col_ce_iv = get_col('CE IV')
col_pe_iv = get_col('PE IV')
col_ce_ltp = get_col('CE LTP')
col_pe_ltp = get_col('PE LTP')

# ---- Safe spot detection ----
spot_guess = None
if col_ce_ltp and col_pe_ltp:
    temp = df_all.copy()
    temp['LTP_DIFF'] = abs(temp[col_ce_ltp] - temp[col_pe_ltp])
    temp = temp.dropna(subset=['LTP_DIFF'])
    if not temp.empty:
        spot_guess = float(temp.loc[temp['LTP_DIFF'].idxmin(), 'STRIKE'])

if spot_guess is None and col_ce_vol and col_pe_vol:
    temp = df_all.copy()
    temp['TOT_VOL'] = temp[col_ce_vol].fillna(0) + temp[col_pe_vol].fillna(0)
    temp = temp.dropna(subset=['TOT_VOL'])
    if not temp.empty:
        spot_guess = float(temp.loc[temp['TOT_VOL'].idxmax(), 'STRIKE'])

if spot_guess is None and not df_all.empty:
    spot_guess = float(df_all['STRIKE'].median())

if spot_guess is None:
    st.error("⚠ Could not determine spot price — please check file format")
    st.stop()

st.info(f"Estimated Spot Price ≈ **{spot_guess}**")

# ---- Find trades safely ----
def find_best(df_all, spot):
    trades = []
    near_df = df_all[df_all['STRIKE'].between(spot-200, spot+200)]
    if near_df.empty:
        return pd.DataFrame()

    if col_ce_chngoi and col_ce_iv and col_ce_vol:
        ce_df = near_df[(near_df[col_ce_chngoi] > 0) & (near_df[col_ce_iv] > 10)]
        if not ce_df.empty:
            row = ce_df.sort_values([col_ce_chngoi, col_ce_vol, col_ce_iv], ascending=False).iloc[0]
            trades.append({
                "Type": "BUY CE",
                "Strike": int(row['STRIKE']),
                "Entry": row[col_ce_ltp],
                "Target": round(row[col_ce_ltp]*1.35, 2),
                "SL": round(row[col_ce_ltp]*0.8, 2),
                "Prob%": min(90, round((row[col_ce_iv]/(row[col_ce_iv]+row[col_pe_iv]))*100, 2)),
                "Logic": f"High CE Vol {row[col_ce_vol]} & OI+{row[col_ce_chngoi]}, IV {row[col_ce_iv]}%"
            })

    if col_pe_chngoi and col_pe_iv and col_pe_vol:
        pe_df = near_df[(near_df[col_pe_chngoi] > 0) & (near_df[col_pe_iv] > 10)]
        if not pe_df.empty:
            row = pe_df.sort_values([col_pe_chngoi, col_pe_vol, col_pe_iv], ascending=False).iloc[0]
            trades.append({
                "Type": "BUY PE",
                "Strike": int(row['STRIKE']),
                "Entry": row[col_pe_ltp],
                "Target": round(row[col_pe_ltp]*1.35, 2),
                "SL": round(row[col_pe_ltp]*0.8, 2),
                "Prob%": min(90, round((row[col_pe_iv]/(row[col_pe_iv]+row[col_ce_iv]))*100, 2)),
                "Logic": f"High PE Vol {row[col_pe_vol]} & OI+{row[col_pe_chngoi]}, IV {row[col_pe_iv]}%"
            })

    return pd.DataFrame(trades)

rec_df = find_best(df_all, spot_guess)
st.subheader("📈 Live Recommendations")
if rec_df.empty:
    st.warning("No trade signals found near spot with current filters.")
else:
    st.dataframe(rec_df)

# ---- Straddle Premium chart ----
if col_ce_ltp and col_pe_ltp:
    st.subheader("Straddle Premium & % Change vs Strike")
    strikes = sorted(df_all['STRIKE'].unique())
    premiums = []
    pct_changes = [np.nan]
    for i,s in enumerate(strikes):
        r = df_all[df_all['STRIKE']==s]
        total = r[col_ce_ltp].values[0] + r[col_pe_ltp].values[0]
        premiums.append(total)
        if i>0:
            pct_changes.append(((total-premiums[i-1])/premiums[i-1])*100 if premiums[i-1] else np.nan)
    fig, ax1 = plt.subplots()
    ax1.bar([str(int(s)) for s in strikes], premiums, color='blue', alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot([str(int(s)) for s in strikes], pct_changes, color='orange', marker='o')
    st.pyplot(fig)

# ---- OI, OI Change, Volume ----
def plot_bar(title, ce_vals, pe_vals, strike_vals):
    if ce_vals is not None and pe_vals is not None:
        fig, ax = plt.subplots()
        ax.bar(strike_vals, ce_vals, color='green', alpha=0.6, label='CE')
        ax.bar(strike_vals, -np.array(pe_vals), color='red', alpha=0.6, label='PE')
        ax.set_title(f"{title} (CE up, PE down)")
        ax.legend()
        st.pyplot(fig)

st.subheader("OI Bar Chart")
plot_bar("Open Interest", df_all[col_ce_oi], df_all[col_pe_oi], df_all['STRIKE'])

st.subheader("OI Change Bar Chart")
plot_bar("OI Change", df_all[col_ce_chngoi], df_all[col_pe_chngoi], df_all['STRIKE'])

st.subheader("Volume Bar Chart (Around Spot)")
near = df_all[df_all['STRIKE'].between(spot_guess-250, spot_guess+250)]
plot_bar("Volume", near[col_ce_vol], near[col_pe_vol], near['STRIKE'])
