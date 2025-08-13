import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Option Chain Live Recommendations", layout="wide")

st.title("📊 NIFTY Option Chain Analysis + Live Recommendations")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("Upload Option Chain File (CSV or XLSX)", type=["csv", "xlsx"])
if not uploaded_file:
    st.warning("Please upload your option chain file first.")
    st.stop()

# Read file
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file, skiprows=1)
else:
    df = pd.read_excel(uploaded_file, skiprows=1)

# Clean column names
df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]

# -------- DATA PREP --------
df_ce = df[['STRIKE','OI','CHNG IN OI','VOLUME','IV','LTP']].rename(
    columns={'OI':'CE OI','CHNG IN OI':'CE CHNGOI','VOLUME':'CE VOLUME','IV':'CE IV','LTP':'CE LTP'}
)
df_pe = df[['STRIKE',df.columns[-6],df.columns[-5],df.columns[-4],df.columns[-3],df.columns[-2]]]
df_pe.columns = ['STRIKE','PE OI','PE CHNGOI','PE VOLUME','PE IV','PE LTP']

df_all = df_ce.merge(df_pe, on='STRIKE')

# Auto-spot from max CE and PE volume/OI near each other
spot_guess = df_all.loc[df_all['CE VOLUME'] == df_all['CE VOLUME'].max(), 'STRIKE'].iloc[0]
st.info(f"Estimated Spot Price ≈ **{spot_guess}**")

# -------- FIND BEST TRADES --------
def find_best(df_all, spot):
    # Focus on strikes around spot (+/- 200)
    window = 200
    near_df = df_all[df_all['STRIKE'].between(spot-window, spot+window)]

    # CE candidate = Highest CE volume + positive CE CHNGOI + decent IV
    ce_df = near_df[(near_df['CE CHNGOI'] > 0) & (near_df['CE IV'] > 10)]
    best_ce = ce_df.sort_values(['CE CHNGOI','CE VOLUME','CE IV'], ascending=False).head(1)

    # PE candidate = Highest PE volume + positive PE CHNGOI + decent IV
    pe_df = near_df[(near_df['PE CHNGOI'] > 0) & (near_df['PE IV'] > 10)]
    best_pe = pe_df.sort_values(['PE CHNGOI','PE VOLUME','PE IV'], ascending=False).head(1)

    trades = []

    # CE Trade
    if not best_ce.empty:
        row = best_ce.iloc[0]
        entry = row['CE LTP']
        target = round(entry * 1.35, 2)
        sl = round(entry * 0.8, 2)
        prob = min(90, round((row['CE IV'] / (row['CE IV'] + row['PE IV'])) * 100, 2))
        logic = (
            f"High CE volume ({row['CE VOLUME']}) with OI up by {row['CE CHNGOI']} "
            f"and IV {row['CE IV']}% — suggests traders betting on upside."
        )
        trades.append({
            "Type": "BUY CE",
            "Strike": int(row['STRIKE']),
            "Entry": entry,
            "Target": target,
            "SL": sl,
            "Prob%": prob,
            "Logic": logic
        })

    # PE Trade
    if not best_pe.empty:
        row = best_pe.iloc[0]
        entry = row['PE LTP']
        target = round(entry * 1.35, 2)
        sl = round(entry * 0.8, 2)
        prob = min(90, round((row['PE IV'] / (row['PE IV'] + row['CE IV'])) * 100, 2))
        logic = (
            f"High PE volume ({row['PE VOLUME']}) with OI up by {row['PE CHNGOI']} "
            f"and IV {row['PE IV']}% — suggests traders betting on downside."
        )
        trades.append({
            "Type": "BUY PE",
            "Strike": int(row['STRIKE']),
            "Entry": entry,
            "Target": target,
            "SL": sl,
            "Prob%": prob,
            "Logic": logic
        })

    return pd.DataFrame(trades)

rec_df = find_best(df_all, spot_guess)

st.subheader("📈 Live Recommendations")
st.dataframe(rec_df)

# -------- HUMAN READABLE LOGIC --------
for _, row in rec_df.iterrows():
    st.markdown(
        f"**{row['Type']} {row['Strike']}** — Entry: {row['Entry']} | Target: {row['Target']} | SL: {row['SL']} | "
        f"Prob. of Profit: **{row['Prob%']}%**\n\n**Logic:** {row['Logic']}\n"
    )

# -------- STRADDLE PREMIUM --------
st.subheader("Straddle Premium & % Change vs Previous Strike")
premiums, pct_changes = [], [np.nan]
strikes = sorted(df_all['STRIKE'].unique())

for i, s in enumerate(strikes):
    row = df_all[df_all['STRIKE'] == s]
    ce = row['CE LTP'].values[0] if not row.empty else 0
    pe = row['PE LTP'].values[0] if not row.empty else 0
    total = ce + pe
    premiums.append(total)
    if i > 0:
        pct_changes.append(((total - premiums[i-1]) / premiums[i-1]) * 100 if premiums[i-1] else 0)

fig, ax1 = plt.subplots()
ax1.bar([str(int(s)) for s in strikes], premiums, color='blue', alpha=0.6)
ax2 = ax1.twinx()
ax2.plot([str(int(s)) for s in strikes], pct_changes, color='orange', marker='o')
ax1.set_ylabel("Straddle Premium")
ax2.set_ylabel("% change")
ax1.set_title("Straddle Premium & % Change vs Strike")
st.pyplot(fig)

# -------- OI, OI Change, Volume --------
def plot_bar(name, ce_values, pe_values):
    fig, ax = plt.subplots()
    ax.bar(df_all['STRIKE'], ce_values, color='green', alpha=0.6, label='CE')
    ax.bar(df_all['STRIKE'], -np.array(pe_values), color='red', alpha=0.6, label='PE')
    ax.set_title(f"{name} (Green=CE up, Red=PE down)")
    ax.set_xlabel("Strike")
    ax.legend()
    st.pyplot(fig)

st.subheader("OI Bar Chart")
plot_bar("Open Interest", df_all['CE OI'], df_all['PE OI'])

st.subheader("OI Change Bar Chart")
plot_bar("OI Change", df_all['CE CHNGOI'], df_all['PE CHNGOI'])

st.subheader("Volume Bar Chart (Around Spot)")
near_strikes = df_all[df_all['STRIKE'].between(spot_guess-250, spot_guess+250)]
plot_bar("Volume", near_strikes['CE VOLUME'], near_strikes['PE VOLUME'])
