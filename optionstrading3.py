import streamlit as st import pandas as pd import matplotlib.pyplot as plt

Streamlit app setup

st.set_page_config(page_title="BankNifty Option Chain Analysis", layout="wide") st.title("BankNifty Option Chain with Greeks & Trade Recommendations")

File upload section

st.sidebar.header("Upload Option Chain Files") uploaded_file1 = st.sidebar.file_uploader("Upload Option Chain CSV (File 1)", type=["csv"]) uploaded_file2 = st.sidebar.file_uploader("Upload Option Chain CSV (File 2)", type=["csv"])

Configurable ATM window

atm_window = st.sidebar.number_input( "ATM Window (number of strikes above/below)", min_value=5, max_value=30, value=15 )

Load data function

def load_data(uploaded_file): if uploaded_file is not None: df = pd.read_csv(uploaded_file) df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns] return df return None

Analysis function

def analyze_option_chain(df): if df is None: return None

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# Identify ATM strike based on closest LTP difference
if "c_ltp" in df.columns and "p_ltp" in df.columns:
    df["ltp_diff"] = (df["c_ltp"] - df["p_ltp"]).abs()
    atm_strike = df.loc[df["ltp_diff"].idxmin(), "strike_price"]
    st.write(f"**Identified ATM Strike Price:** {atm_strike}")

    filtered_df = df[(df["strike_price"] >= atm_strike - atm_window * 100) & 
                     (df["strike_price"] <= atm_strike + atm_window * 100)]
else:
    st.warning("Missing LTP columns for ATM detection")
    filtered_df = df

st.subheader("Filtered Data (±ATM Window)")
st.dataframe(filtered_df)

# Plot OI & Change in OI
if "c_oi" in filtered_df.columns and "p_oi" in filtered_df.columns:
    st.subheader("Open Interest (OI) Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(filtered_df["strike_price"], filtered_df["c_oi"], width=50, label="CE OI", alpha=0.7)
    ax.bar(filtered_df["strike_price"], filtered_df["p_oi"], width=50, label="PE OI", alpha=0.7)
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Open Interest")
    ax.legend()
    st.pyplot(fig)

# Return processed data
return filtered_df

Trade recommendation function

def generate_recommendations(df): if df is None: return

st.subheader("Trade Recommendations")
recommendations = []

for _, row in df.iterrows():
    try:
        if row.get("c_oi", 0) > row.get("p_oi", 0) and row.get("c_delta", 0) > 0.5:
            rec = {
                "Instrument": f"{row['strike_price']} CE",
                "Entry (LTP)": row.get("c_ltp", 0),
                "Target": round(row.get("c_ltp", 0) * 1.15, 2),
                "Stop Loss": round(row.get("c_ltp", 0) * 0.9, 2),
                "Reason": "Bullish OI build-up with positive delta",
                "Confidence": min(1.0, row.get("c_delta", 0))
            }
            recommendations.append(rec)
        elif row.get("p_oi", 0) > row.get("c_oi", 0) and row.get("p_delta", 0) < -0.5:
            rec = {
                "Instrument": f"{row['strike_price']} PE",
                "Entry (LTP)": row.get("p_ltp", 0),
                "Target": round(row.get("p_ltp", 0) * 1.15, 2),
                "Stop Loss": round(row.get("p_ltp", 0) * 0.9, 2),
                "Reason": "Bearish OI build-up with negative delta",
                "Confidence": min(1.0, abs(row.get("p_delta", 0)))
            }
            recommendations.append(rec)
    except Exception:
        continue

if recommendations:
    st.dataframe(pd.DataFrame(recommendations))
else:
    st.write("No strong trade setups found.")

Run app logic

df1 = load_data(uploaded_file1) df2 = load_data(uploaded_file2)

if df1 is not None: st.header("Analysis of File 1") processed1 = analyze_option_chain(df1) generate_recommendations(processed1)

if df2 is not None: st.header("Analysis of File 2") processed2 = analyze_option_chain(df2) generate_recommendations(processed2)

