import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Option Chain Analysis — FINAL SAFE MODE")
st.title("📊 NIFTY Option Chain Analysis — Final Crash-Proof Version")

# ---------- HELPER FUNCTIONS ----------
def safe_numeric(series):
    """Clean commas, convert '-' to NaN, then to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(',', '', regex=False).replace('-', np.nan),
        errors='coerce'
    )

def get_col(part):
    matches = [c for c in df_all.columns if part in c]
    return matches[0] if matches else None

def plot_bar(title, ce_vals, pe_vals, strikes):
    """Safe bar plot, skip if data length mismatch."""
    if ce_vals is None or pe_vals is None:
        st.warning(f"Skipping {title} — columns not found")
        return
    if len(strikes) == 0 or len(strikes) != len(ce_vals) or len(strikes) != len(pe_vals):
        st.warning(f"Skipping {title} — data length mismatch or empty")
        return
    fig, ax = plt.subplots()
    ax.bar(strikes, ce_vals, color='green', alpha=0.6, label='CE')
    ax.bar(strikes, -np.array(pe_vals), color='red', alpha=0.6, label='PE')
    ax.set_title(f"{title} (Green=CE up, Red=PE down)")
    ax.legend()
    st.pyplot(fig)

# ---------- UPLOAD FILE ----------
uploaded = st.file_uploader("Upload Option Chain CSV/XLSX", type=['csv', 'xlsx'])
if not uploaded:
    st.stop()

# ---------- READ FILE ----------
if uploaded.name.endswith(".csv"):
    df = pd.read_csv(uploaded, skiprows=1)
else:
    df = pd.read_excel(uploaded, skiprows=1)

# Column cleaning
df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]

# ---------- SPLIT CE/PE ----------
strike_idx = df.columns.get_loc('STRIKE')
df_ce = df[df.columns[:strike_idx].tolist() + ['STRIKE']].copy()
df_pe = df[['STRIKE'] + df.columns[strike_idx+1:].tolist()].copy()
df_ce.columns = ['CE ' + c if c != 'STRIKE' else c for c in df_ce.columns]
df_pe.columns = ['PE ' + c if c != 'STRIKE' else c for c in df_pe.columns]

df_all = pd.merge(df_ce, df_pe, on='STRIKE', how='inner')

# Normalise names
df_all.columns = df_all.columns.str.strip().str.replace(r'\s+', ' ', regex=True).str.upper()

# Convert numerics
for col in df_all.columns:
    df_all[col] = safe_numeric(df_all[col]) if col != 'STRIKE' else pd.to_numeric(df_all[col], errors='coerce')
df_all = df_all.dropna(subset=['STRIKE'])

# ---------- MAP COLUMNS ----------
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

# ---------- SPOT DETECTION ----------
spot_guess = None
# ATM method
if col_ce_ltp and col_pe_ltp:
    temp = df_all.dropna(subset=[col_ce_ltp, col_pe_ltp])
    if not temp.empty:
        temp['LTP_DIFF'] = abs(temp[col_ce_ltp] - temp[col_pe_ltp])
        spot_guess = float(temp.loc[temp['LTP_DIFF'].idxmin(), 'STRIKE'])
# Max volume fallback
if (spot_guess is None or pd.isna(spot_guess)) and col_ce_vol and col_pe_vol:
    temp = df_all.dropna(subset=[col_ce_vol, col_pe_vol])
    if not temp.empty:
        temp['TOT_VOL'] = temp[col_ce_vol].fillna(0) + temp[col_pe_vol].fillna(0)
        spot_guess = float(temp.loc[temp['TOT_VOL'].idxmax(), 'STRIKE'])
# Median final fallback
if spot_guess is None or pd.isna(spot_guess):
    spot_guess = float(df_all['STRIKE'].median())

st.info(f"Estimated Spot Price ≈ **{spot_guess}**")

# ---------- RECOMMENDATIONS ----------
def find_best(df_all, spot):
    trades = []
    near_df = df_all[df_all['STRIKE'].between(spot-200, spot+200)]
    if near_df.empty:
        return pd.DataFrame()

    # CE
    if all([col_ce_chngoi, col_ce_iv, col_ce_vol, col_ce_ltp]):
        ce_df = near_df[(near_df[col_ce_chngoi] > 0) & (near_df[col_ce_iv] > 10)]
        if not ce_df.empty:
            row = ce_df.sort_values([col_ce_chngoi, col_ce_vol, col_ce_iv], ascending=False).iloc[0]
            trades.append({
                "Type": "BUY CE", "Strike": int(row['STRIKE']),
                "Entry": row[col_ce_ltp],
                "Target": round(row[col_ce_ltp]*1.35, 2),
                "SL": round(row[col_ce_ltp]*0.8, 2),
                "Prob%": min(90, round((row[col_ce_iv]/(row[col_ce_iv]+row[col_pe_iv]))*100, 2)) if row[col_pe_iv]>0 else None,
                "Logic": f"High CE Vol {row[col_ce_vol]} & OI+{row[col_ce_chngoi]}, IV {row[col_ce_iv]}%"
            })
    # PE
    if all([col_pe_chngoi, col_pe_iv, col_pe_vol, col_pe_ltp]):
        pe_df = near_df[(near_df[col_pe_chngoi] > 0) & (near_df[col_pe_iv] > 10)]
        if not pe_df.empty:
            row = pe_df.sort_values([col_pe_chngoi, col_pe_vol, col_pe_iv], ascending=False).iloc[0]
            trades.append({
                "Type": "BUY PE", "Strike": int(row['STRIKE']),
                "Entry": row[col_pe_ltp],
                "Target": round(row[col_pe_ltp]*1.35, 2),
                "SL": round(row[col_pe_ltp]*0.8, 2),
                "Prob%": min(90, round((row[col_pe_iv]/(row[col_pe_iv]+row[col_ce_iv]))*100, 2)) if row[col_ce_iv]>0 else None,
                "Logic": f"High PE Vol {row[col_pe_vol]} & OI+{row[col_pe_chngoi]}, IV {row[col_pe_iv]}%"
            })
    return pd.DataFrame(trades)

rec_df = find_best(df_all, spot_guess)
st.subheader("📈 Live Recommendations")
st.dataframe(rec_df) if not rec_df.empty else st.warning("No trade signals near spot.")

# ---------- STRADDLE PREMIUM ----------
if col_ce_ltp and col_pe_ltp:
    strikes = sorted(df_all['STRIKE'].unique())
    premiums = []
    pct_changes = []
    for i, s in enumerate(strikes):
        r = df_all[df_all['STRIKE']==s]
        if r.empty: continue
        total = r[col_ce_ltp].values[0] + r[col_pe_ltp].values[0]
        premiums.append(total)
        pct_changes.append(np.nan if i==0 else ((total - premiums[i-1])/premiums[i-1])*100 if premiums[i-1] else np.nan)
    if len(strikes) == len(premiums) and premiums:
        fig, ax1 = plt.subplots()
        ax1.bar([str(int(s)) for s in strikes], premiums, color='blue', alpha=0.6)
        ax2 = ax1.twinx()
        ax2.plot([str(int(s)) for s in strikes], pct_changes, color='orange', marker='o')
        ax1.set_title("Straddle Premium & % Change")
        st.pyplot(fig)
    else:
        st.warning("Skipping Straddle Premium chart — data mismatch")

# ---------- OI / OI Change / Volume ----------
st.subheader("OI")
plot_bar("Open Interest", df_all[col_ce_oi], df_all[col_pe_oi], df_all['STRIKE'])
st.subheader("OI Change")
plot_bar("OI Change", df_all[col_ce_chngoi], df_all[col_pe_chngoi], df_all['STRIKE'])
st.subheader("Volume (Around Spot)")
near = df_all[df_all['STRIKE'].between(spot_guess-250, spot_guess+250)]
plot_bar("Volume", near[col_ce_vol], near[col_pe_vol], near['STRIKE'])
