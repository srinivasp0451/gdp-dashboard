import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Streamlit App Setup ----------------
st.set_page_config(page_title="BankNifty Option Chain Analysis", layout="wide")
st.title("📊 BankNifty Option Chain with Greeks & Trade Recommendations")

# ---------------- File Upload Section ----------------
st.sidebar.header("Upload Files")

file1 = st.sidebar.file_uploader("Upload Option Chain File (with c_ltp & p_ltp)", type=["csv"])
file2 = st.sidebar.file_uploader("Upload Greeks / PLOI File", type=["csv"])

if file1 is not None and file2 is not None:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Validate strikePrice column
    if 'strikePrice' not in df1.columns or 'strikePrice' not in df2.columns:
        st.error("❌ Both files must have a 'strikePrice' column to merge.")
    elif 'c_ltp' not in df1.columns or 'p_ltp' not in df1.columns:
        st.error("❌ Option chain file must contain 'c_ltp' and 'p_ltp' columns for ATM detection.")
    else:
        # Merge on strikePrice
        df = pd.merge(df1, df2, on="strikePrice", how="inner")

        st.success("✅ Files uploaded and merged successfully!")
        st.write("### Merged Data Preview")
        st.dataframe(df.head(20))

        # Example ATM Detection
        df['atm_diff'] = (df['c_ltp'] - df['p_ltp']).abs()
        atm_strike = df.loc[df['atm_diff'].idxmin(), 'strikePrice']
        st.info(f"ATM Strike Detected: **{atm_strike}**")

        # ---------------- Data Exploration ----------------
        st.subheader("Exploratory Data Analysis")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select a column to visualize", numeric_cols)
            fig, ax = plt.subplots()
            ax.plot(df['strikePrice'], df[selected_col], marker='o')
            ax.set_xlabel("Strike Price")
            ax.set_ylabel(selected_col)
            ax.set_title(f"{selected_col} vs Strike Price")
            st.pyplot(fig)
        else:
            st.warning("⚠ No numeric columns found for plotting.")

else:
    st.warning("⚠ Please upload **both files** to proceed.")
