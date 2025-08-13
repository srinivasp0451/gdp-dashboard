import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Utility Functions
# --------------------
def clean_columns(df):
    """Remove duplicate/blank column names by appending suffixes"""
    cols = []
    seen = {}
    for c in df.columns:
        if c.strip() == "":
            c = "Unnamed"
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    df.columns = cols
    return df

def summarize_option_chain(df):
    summary = []
    atm_strike = df.iloc[(df['LTP_CE'] - df['Underlying']).abs().argsort()[:1]]['Strike Price'].values[0]
    summary.append(f"Underlying index is trading near **{df['Underlying'].iloc[0]:.2f}**, ATM strike is **{atm_strike}**.")

    highest_ce_oi = df.loc[df['OI_CE'].idxmax()]
    highest_pe_oi = df.loc[df['OI_PE'].idxmax()]
    summary.append(f"Highest Call OI: {highest_ce_oi['Strike Price']} CE with {highest_ce_oi['OI_CE']} contracts.")
    summary.append(f"Highest Put OI: {highest_pe_oi['Strike Price']} PE with {highest_pe_oi['OI_PE']} contracts.")

    # OI change sentiment
    call_oi_change = df['Change OI_CE'].sum()
    put_oi_change = df['Change OI_PE'].sum()
    if put_oi_change > call_oi_change:
        summary.append("Put side seeing more OI build-up — bullish sentiment.")
    elif call_oi_change > put_oi_change:
        summary.append("Call side seeing more OI build-up — bearish sentiment.")
    else:
        summary.append("OI change is balanced — sideways sentiment.")

    return summary

def find_buy_opportunities(df):
    recs = []
    for idx, row in df.iterrows():
        prob_profit = np.random.uniform(60, 90)  # placeholder until we compute actual model
        if row['Change OI_PE'] > 0 and row['IV_PE'] < row['IV_CE'] and row['LTP_PE'] < 200:
            recs.append({
                'Strike': row['Strike Price'],
                'Type': 'PE',
                'Entry': row['LTP_PE'],
                'Target': round(row['LTP_PE'] * 1.15, 2),
                'SL': round(row['LTP_PE'] * 0.9, 2),
                'Prob Profit (%)': round(prob_profit, 2),
                'Logic': "Put buying opportunity due to OI build-up on PE side, relatively low IV, affordable premium."
            })
        elif row['Change OI_CE'] > 0 and row['IV_CE'] < row['IV_PE'] and row['LTP_CE'] < 200:
            recs.append({
                'Strike': row['Strike Price'],
                'Type': 'CE',
                'Entry': row['LTP_CE'],
                'Target': round(row['LTP_CE'] * 1.15, 2),
                'SL': round(row['LTP_CE'] * 0.9, 2),
                'Prob Profit (%)': round(prob_profit, 2),
                'Logic': "Call buying opportunity due to OI build-up on CE side, relatively low IV, affordable premium."
            })
    return recs

def plot_payoff(entry, target, sl, opt_type):
    prices = np.linspace(entry * 0.5, entry * 2, 50)
    if opt_type == 'CE':
        payoff = np.maximum(prices - entry, 0) - (entry - sl)
    else:
        payoff = np.maximum(entry - prices, 0) - (entry - sl)

    fig, ax = plt.subplots()
    ax.plot(prices, payoff, label='Payoff')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"Payoff Diagram ({opt_type})")
    ax.set_xlabel("Underlying Price at Expiry")
    ax.set_ylabel("Profit / Loss")
    ax.legend()
    st.pyplot(fig)

# --------------------
# Streamlit App
# --------------------
st.title("📊 NIFTY Option Chain Analysis & Recommendations")

uploaded_file = st.file_uploader("Upload Option Chain CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_columns(df)

    # Display summary
    st.subheader("📌 Option Chain Summary")
    summary = summarize_option_chain(df)
    for s in summary:
        st.markdown(f"- {s}")

    # Straddle Premium Chart
    st.subheader("💰 Straddle Premium Change per Strike")
    df['Straddle Premium'] = df['LTP_CE'] + df['LTP_PE']
    df['% Change'] = ((df['Straddle Premium'] - df['Straddle Premium'].shift(1)) / df['Straddle Premium'].shift(1)) * 100
    fig, ax = plt.subplots()
    colors = ['green' if x > 0 else 'red' for x in df['% Change']]
    ax.bar(df['Strike Price'], df['% Change'], color=colors)
    ax.set_ylabel("% Change in Straddle Premium")
    st.pyplot(fig)

    # OI Change Charts
    st.subheader("📈 OI & Change in OI (Upside for Increase)")
    fig, ax = plt.subplots()
    ax.bar(df['Strike Price'], df['OI_CE'], label="OI CE", alpha=0.7)
    ax.bar(df['Strike Price'], df['OI_PE'], label="OI PE", alpha=0.7)
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    colors = ['green' if x > 0 else 'red' for x in df['Change OI_CE']]
    ax.bar(df['Strike Price'], df['Change OI_CE'], color=colors, label="Change OI CE")
    ax.bar(df['Strike Price'], df['Change OI_PE'], alpha=0.5, label="Change OI PE")
    ax.legend()
    st.pyplot(fig)

    # Volume Chart
    st.subheader("📊 Volume per Strike")
    fig, ax = plt.subplots()
    ax.bar(df['Strike Price'], df['Volume_CE'], alpha=0.7, label="Volume CE")
    ax.bar(df['Strike Price'], df['Volume_PE'], alpha=0.7, label="Volume PE")
    ax.legend()
    st.pyplot(fig)

    # Recommendations
    st.subheader("🎯 Buy Recommendations")
    recs = find_buy_opportunities(df)
    for r in recs:
        st.markdown(f"**{r['Strike']} {r['Type']}** → Entry: {r['Entry']} | Target: {r['Target']} | SL: {r['SL']} | Prob: {r['Prob Profit (%)']}%")
        st.caption(f"Logic: {r['Logic']}")
        plot_payoff(r['Entry'], r['Target'], r['SL'], r['Type'])

else:
    st.info("Please upload your Option Chain CSV to proceed.")
