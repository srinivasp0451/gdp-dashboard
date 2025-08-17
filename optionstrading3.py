import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit app setup
st.set_page_config(page_title="BankNifty Option Chain Analysis", layout="wide")
st.title("📊 BankNifty Option Chain with Greeks & Trade Recommendations")

# File upload
uploaded_file = st.file_uploader("Upload BankNifty Option Chain CSV", type=["csv"])

# Configurable ATM window
atm_window = st.number_input("Select ±N Strikes around ATM", min_value=5, max_value=30, value=15, step=1)

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "").str.replace("_", "")

    # Ensure strike column exists
    if "strikeprice" not in df.columns:
        st.error("CSV must have a 'strikePrice' (or similar) column!")
        st.stop()

    # Show cleaned data
    st.subheader("📑 Cleaned Option Chain Data")
    st.dataframe(df.head(20))

    # Detect ATM (closest strike to LTP)
    if "c_ltp" in df.columns and "p_ltp" in df.columns:
        df["atm_diff"] = (df["c_ltp"] - df["p_ltp"]).abs()
        atm_strike = df.loc[df["atm_diff"].idxmin(), "strikeprice"]
    else:
        st.error("CSV must contain 'c_ltp' and 'p_ltp' columns for ATM detection.")
        st.stop()

    st.write(f"**Detected ATM Strike:** {atm_strike}")

    # Filter relevant strikes (± atm_window)
    strikes = df["strikeprice"].unique()
    strikes.sort()
    filtered = df[(df["strikeprice"] >= atm_strike - atm_window*100) &
                  (df["strikeprice"] <= atm_strike + atm_window*100)]

    st.subheader("📌 Filtered Option Chain (±N strikes)")
    st.dataframe(filtered)

    # Summaries
    st.subheader("📊 OI, Change in OI & Volume")
    fig, ax = plt.subplots(figsize=(12, 6))

    if "c_oi" in filtered.columns and "p_oi" in filtered.columns:
        ax.bar(filtered["strikeprice"], filtered["c_oi"], alpha=0.5, label="CE OI", color="blue")
        ax.bar(filtered["strikeprice"], -filtered["p_oi"], alpha=0.5, label="PE OI", color="red")

    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Open Interest")
    ax.set_title("Open Interest (CE vs PE)")
    ax.legend()
    st.pyplot(fig)

    # Trade Recommendation Logic
    st.subheader("🎯 Trade Recommendations")

    recs = []
    for _, row in filtered.iterrows():
        if "c_oi" in row and "c_chnginoi" in row and "c_iv" in row:
            if row["c_oi"] > 50000 and row["c_chnginoi"] > 10000 and row["c_iv"] < 20:
                recs.append({
                    "Strike": row["strikeprice"],
                    "Type": "CE",
                    "Entry": row.get("c_ltp", None),
                    "Target": round(row.get("c_ltp", 0) * 1.15, 2),
                    "SL": round(row.get("c_ltp", 0) * 0.9, 2),
                    "Reason": "High OI + OI Buildup + Low IV"
                })
        if "p_oi" in row and "p_chnginoi" in row and "p_iv" in row:
            if row["p_oi"] > 50000 and row["p_chnginoi"] > 10000 and row["p_iv"] < 20:
                recs.append({
                    "Strike": row["strikeprice"],
                    "Type": "PE",
                    "Entry": row.get("p_ltp", None),
                    "Target": round(row.get("p_ltp", 0) * 1.15, 2),
                    "SL": round(row.get("p_ltp", 0) * 0.9, 2),
                    "Reason": "High OI + OI Buildup + Low IV"
                })

    if recs:
        st.write(pd.DataFrame(recs))
    else:
        st.info("No strong trade opportunities found based on filters.")
