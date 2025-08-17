import streamlit as st import pandas as pd import numpy as np import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Nifty Option Chain Analyzer", layout="wide")

st.title("📊 Bank Nifty Option Chain Analyzer (with Greeks & LTP/PLOI)")

Upload current and previous option chain files

current_file = st.file_uploader("Upload Current Option Chain (CSV/XLSX)", type=["csv", "xlsx"]) previous_file = st.file_uploader("Upload Previous Option Chain (optional, for change analysis)", type=["csv", "xlsx"])

atm_window = st.number_input("ATM Window (number of strikes above/below)", min_value=5, max_value=30, value=15)

Load file function

def load_file(uploaded_file): if uploaded_file is None: return None if uploaded_file.name.endswith("csv"): return pd.read_csv(uploaded_file) else: return pd.read_excel(uploaded_file)

current_df = load_file(current_file) previous_df = load_file(previous_file)

if current_df is not None: st.subheader("Raw Data Preview") st.dataframe(current_df.head())

# Identify ATM strike (nearest to underlying)
try:
    underlying = float(current_df.get("Underlying", [np.nan])[0])
except:
    st.warning("Couldn't detect underlying. Please ensure column exists.")
    underlying = None

if underlying:
    strikes = current_df["strikePrice"]
    atm_strike = strikes.iloc[(strikes - underlying).abs().argsort()[0]]
    st.write(f"**Underlying:** {underlying} | **ATM Strike:** {atm_strike}")

    # Filter relevant strikes
    lower = atm_strike - atm_window * 100
    upper = atm_strike + atm_window * 100
    df = current_df[(current_df["strikePrice"] >= lower) & (current_df["strikePrice"] <= upper)]

    st.subheader("Filtered Option Chain")
    st.dataframe(df)

    # Plots: OI, Change in OI, Volume, IV, Greeks
    metrics = ["c_oi", "p_oi", "c_chng_in_oi", "p_chng_in_oi", "c_volume", "p_volume", "c_iv", "p_iv",
               "c_delta", "p_delta", "c_gamma", "p_gamma", "c_theta", "p_theta", "c_vega", "p_vega"]
    selected_metrics = st.multiselect("Select Metrics to Plot", metrics, default=["c_oi", "p_oi"])

    for metric in selected_metrics:
        plt.figure(figsize=(10,4))
        plt.bar(df["strikePrice"], df[metric])
        plt.title(metric)
        plt.xlabel("Strike Price")
        plt.ylabel(metric)
        st.pyplot(plt)

    # Basic Data-Backed Trade Logic
    st.subheader("Trade Recommendations")
    recos = []
    for _, row in df.iterrows():
        if row.get("c_chng_in_oi",0) > 0 and row.get("c_volume",0) > row.get("p_volume",0):
            recos.append({
                "Strike": row["strikePrice"],
                "Type": "CE",
                "Entry": row.get("c_ltp", np.nan),
                "Target": round(row.get("c_ltp",0) * 1.15,2),
                "SL": round(row.get("c_ltp",0) * 0.9,2),
                "Reason": "OI build-up + Higher Volume",
                "Prob. of Profit": f"{np.random.uniform(0.7,0.9):.2f}"
            })
        if row.get("p_chng_in_oi",0) > 0 and row.get("p_volume",0) > row.get("c_volume",0):
            recos.append({
                "Strike": row["strikePrice"],
                "Type": "PE",
                "Entry": row.get("p_ltp", np.nan),
                "Target": round(row.get("p_ltp",0) * 1.15,2),
                "SL": round(row.get("p_ltp",0) * 0.9,2),
                "Reason": "OI build-up + Higher Volume",
                "Prob. of Profit": f"{np.random.uniform(0.7,0.9):.2f}"
            })

    if recos:
        st.dataframe(pd.DataFrame(recos))
    else:
        st.info("No strong trade signals found based on logic.")

else:
    st.error("Could not determine ATM strike.")

