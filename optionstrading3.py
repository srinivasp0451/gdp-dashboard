import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="BankNifty Option Chain Analysis", layout="wide")
st.title("📊 BankNifty Option Chain with Greeks & Trade Recommendations")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload BankNifty Option Chain CSV", type=["csv"])

# ---------------------------
# Spot Price Input (default = 55341)
# ---------------------------
spot_price = st.number_input("Enter Current Spot Price", value=55341)

# ---------------------------
# ATM Window Config
# ---------------------------
atm_window = st.number_input("ATM Window (± strikes)", value=15, step=1)

if uploaded_file:
    # ---------------------------
    # Load Data
    # ---------------------------
    df = pd.read_csv(uploaded_file)

    # Ensure lowercase col names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ["strikeprice", "c_ltp", "p_ltp", "c_oi", "p_oi", "c_chng_in_oi", "p_chng_in_oi",
                     "c_iv", "p_iv", "c_volume", "p_volume", "delta", "gamma", "theta", "vega"]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"CSV missing columns: {missing}")
    else:
        # ---------------------------
        # ATM Filtering
        # ---------------------------
        atm_strike = min(df['strikeprice'], key=lambda x: abs(x - spot_price))
        lower = atm_strike - atm_window * 100
        upper = atm_strike + atm_window * 100
        df = df[(df['strikeprice'] >= lower) & (df['strikeprice'] <= upper)]

        # ---------------------------
        # PCR Calculation
        # ---------------------------
        total_ce_oi = df['c_oi'].sum()
        total_pe_oi = df['p_oi'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi != 0 else None

        # ---------------------------
        # Summary in Layman Terms
        # ---------------------------
        st.subheader("📌 Option Chain Summary (Layman Terms)")
        bullishness = "bullish" if pcr > 1 else "bearish" if pcr < 1 else "neutral"
        summary = f"""
        The current spot price is around {spot_price}. The ATM strike is {atm_strike}.
        Total Call OI = {total_ce_oi:,}, Total Put OI = {total_pe_oi:,}, giving a PCR of {round(pcr,2) if pcr else 'NA'}.
        A PCR above 1 indicates more Put writing, suggesting bullishness. Below 1 indicates bearishness.
        In this case, market looks **{bullishness}**. Traders should watch the build-up in OI along with Greeks
        like Delta and Gamma for momentum confirmation. Implied Volatility (IV) shows expected volatility;
        higher IV means options are expensive. Volume activity also indicates active strikes.
        """
        st.write(summary)

        # ---------------------------
        # Recommendations
        # ---------------------------
        st.subheader("💡 Trade Recommendations")
        recos = []
        for _, row in df.iterrows():
            if row['p_chng_in_oi'] > 0 and row['p_iv'] < row['c_iv']:
                recos.append(f"📈 Buy PE {row['strikeprice']} (Put writing + Lower IV)")
            if row['c_chng_in_oi'] > 0 and row['c_iv'] < row['p_iv']:
                recos.append(f"📈 Buy CE {row['strikeprice']} (Call writing + Lower IV)")

        if recos:
            for r in recos:
                st.write(r)
        else:
            st.write("No strong trade recommendations at the moment.")

        # ---------------------------
        # Data Exploration
        # ---------------------------
        st.subheader("📊 Data Exploration Charts")

        col1, col2 = st.columns(2)
        with col1:
            st.write("OI Comparison")
            df_melted = df.melt(id_vars="strikeprice", value_vars=["c_oi", "p_oi"], var_name="type", value_name="oi")
            plt.figure(figsize=(10,4))
            sns.barplot(data=df_melted, x="strikeprice", y="oi", hue="type")
            plt.xticks(rotation=90)
            st.pyplot(plt)

        with col2:
            st.write("Change in OI")
            df_melted = df.melt(id_vars="strikeprice", value_vars=["c_chng_in_oi", "p_chng_in_oi"],
                                var_name="type", value_name="oi_change")
            plt.figure(figsize=(10,4))
            sns.barplot(data=df_melted, x="strikeprice", y="oi_change", hue="type")
            plt.xticks(rotation=90)
            st.pyplot(plt)

        # ---------------------------
        # Heatmap of Greeks
        # ---------------------------
        st.subheader("🔥 Heatmap of Greeks")
        greek_cols = ["delta", "gamma", "theta", "vega"]
        plt.figure(figsize=(8,6))
        sns.heatmap(df[greek_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
