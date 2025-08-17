import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# Streamlit App Setup
# ================================
st.set_page_config(page_title="BankNifty Option Chain Analysis", layout="wide")
st.title("📊 BankNifty Option Chain with Greeks & Trade Recommendations")

# ================================
# File Upload Section
# ================================
st.sidebar.header("Upload Option Chain Files")
file1 = st.sidebar.file_uploader("Upload Option Chain CSV (LTP, OI, Volume etc.)", type=["csv"])
file2 = st.sidebar.file_uploader("Upload Option Chain CSV (Greeks Data)", type=["csv"])

if file1 and file2:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Merge both files if they share common columns (e.g., strike price)
    if "strikePrice" in df1.columns and "strikePrice" in df2.columns:
        df = pd.merge(df1, df2, on="strikePrice", how="inner")
    else:
        st.error("Both files must have a 'strikePrice' column to merge.")
        st.stop()

    st.subheader("🔍 Raw Uploaded Data")
    st.dataframe(df.head(50))

    # ================================
    # ATM Strike Detection
    # ================================
    if "LTP" in df.columns:
        atm_strike = df.loc[df["LTP"].sub(df["LTP"].mean()).abs().idxmin(), "strikePrice"]
        st.write(f"⚡ Detected ATM Strike: **{atm_strike}**")

    # ================================
    # Configurable ATM Window (±N strikes)
    # ================================
    n_strikes = st.sidebar.number_input("ATM Window (±N strikes)", min_value=5, max_value=30, value=15)
    df_window = df[(df["strikePrice"] >= atm_strike - n_strikes*100) &
                   (df["strikePrice"] <= atm_strike + n_strikes*100)]

    st.subheader("📑 Filtered Option Chain (Around ATM)")
    st.dataframe(df_window)

    # ================================
    # Plot OI, Volume, IV, Greeks
    # ================================
    st.subheader("📈 Data Exploration Charts")

    fig, ax = plt.subplots(figsize=(12, 6))
    if "OI" in df_window.columns:
        ax.bar(df_window["strikePrice"], df_window["OI"], color="blue", alpha=0.6, label="Open Interest")
    if "Volume" in df_window.columns:
        ax.bar(df_window["strikePrice"], df_window["Volume"], color="orange", alpha=0.4, label="Volume")
    ax.legend()
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Value")
    ax.set_title("OI & Volume Distribution")
    st.pyplot(fig)

    # ================================
    # Trade Recommendations (Example Logic)
    # ================================
    st.subheader("🎯 Trade Recommendations")

    recommendations = []
    for _, row in df_window.iterrows():
        if "Delta" in row and "Gamma" in row and "IV" in row:
            if row["Delta"] > 0.5 and row["IV"] < df_window["IV"].mean():
                recommendations.append({
                    "Strike": row["strikePrice"],
                    "Type": "CE",
                    "Entry": row["LTP"],
                    "Target": round(row["LTP"] * 1.15, 2),
                    "StopLoss": round(row["LTP"] * 0.9, 2),
                    "Reason": f"Strong Delta {row['Delta']:.2f}, Low IV {row['IV']:.2f}",
                    "Probability of Profit": "High"
                })

    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df)
    else:
        st.info("No strong trade recommendations found for the current data.")
