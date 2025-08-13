import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Nifty Option Chain Analysis")

# --- Upload CSV ---
st.sidebar.header("Upload Option Chain CSV")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Remove spaces

    st.title("📊 Nifty Option Chain Detailed Analysis")
    st.write("This app analyzes your option chain data and provides **clear, data-backed trade recommendations** for buying opportunities.")

    # --- Clean Data ---
    numeric_cols = [c for c in df.columns if df[c].dtype != 'object']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=["Strike Price"])

    # --- Summary Analysis ---
    ce_oi_total = df["CE OI"].sum()
    pe_oi_total = df["PE OI"].sum()
    ce_vol_total = df["CE Volume"].sum()
    pe_vol_total = df["PE Volume"].sum()

    if pe_oi_total > ce_oi_total:
        sentiment = "Bullish bias — Put writers dominating"
    elif ce_oi_total > pe_oi_total:
        sentiment = "Bearish bias — Call writers dominating"
    else:
        sentiment = "Neutral"

    st.subheader("📌 Market Summary")
    st.markdown(f"""
    **Total Call OI:** {ce_oi_total:,.0f}  
    **Total Put OI:** {pe_oi_total:,.0f}  
    **Call Volume:** {ce_vol_total:,.0f}  
    **Put Volume:** {pe_vol_total:,.0f}  
    **Sentiment:** {sentiment}  
    """)

    # --- Find Best Buying Opportunities ---
    recommendations = []
    spot_price = df["Underlying Value"].iloc[0] if "Underlying Value" in df.columns else None

    for _, row in df.iterrows():
        if spot_price:
            distance = abs(row["Strike Price"] - spot_price)
            if distance <= 200:  # Near ATM
                if row["CE Change in OI"] > 0 and row["CE Volume"] > 1000:
                    entry = row["CE LTP"]
                    target = round(entry * 1.15, 2)
                    sl = round(entry * 0.90, 2)
                    prob = np.clip(80 - (distance/spot_price)*100, 60, 90)
                    recommendations.append({
                        "Type": "BUY CE",
                        "Strike": row["Strike Price"],
                        "Entry": entry,
                        "Target": target,
                        "SL": sl,
                        "Probability (%)": prob,
                        "Reason": "High volume & OI build-up near ATM"
                    })
                if row["PE Change in OI"] > 0 and row["PE Volume"] > 1000:
                    entry = row["PE LTP"]
                    target = round(entry * 1.15, 2)
                    sl = round(entry * 0.90, 2)
                    prob = np.clip(80 - (distance/spot_price)*100, 60, 90)
                    recommendations.append({
                        "Type": "BUY PE",
                        "Strike": row["Strike Price"],
                        "Entry": entry,
                        "Target": target,
                        "SL": sl,
                        "Probability (%)": prob,
                        "Reason": "High volume & OI build-up near ATM"
                    })

    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        st.subheader("🎯 Trade Recommendations")
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.warning("No strong buying opportunities found based on the current criteria.")

    # --- Straddle Premium Chart ---
    st.subheader("💰 Straddle Premium per Strike")
    df["Straddle Premium"] = df["CE LTP"] + df["PE LTP"]
    df["% Change"] = df["Straddle Premium"].pct_change().fillna(0) * 100
    fig_straddle = px.bar(df, x="Strike Price", y="Straddle Premium",
                          text=df["% Change"].apply(lambda x: f"{x:.2f}%"),
                          color=df["% Change"], color_continuous_scale=["red", "green"])
    st.plotly_chart(fig_straddle, use_container_width=True)

    # --- OI & Change in OI Chart ---
    st.subheader("📈 OI & Change in OI")
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(x=df["Strike Price"], y=df["CE OI"], name="CE OI", marker_color="blue"))
    fig_oi.add_trace(go.Bar(x=df["Strike Price"], y=df["PE OI"], name="PE OI", marker_color="orange"))
    fig_oi.add_trace(go.Bar(x=df["Strike Price"], y=df["CE Change in OI"], name="CE Change OI", marker_color="cyan"))
    fig_oi.add_trace(go.Bar(x=df["Strike Price"], y=df["PE Change in OI"], name="PE Change OI", marker_color="pink"))
    st.plotly_chart(fig_oi, use_container_width=True)

    # --- Volume Chart ---
    st.subheader("📊 Volume per Strike")
    fig_vol = px.bar(df, x="Strike Price", y=["CE Volume", "PE Volume"], barmode="group")
    st.plotly_chart(fig_vol, use_container_width=True)

    # --- Payoff Chart ---
    st.subheader("📉 Payoff Chart for Buying a Strike")
    strike_choice = st.selectbox("Select Strike Price", df["Strike Price"].unique())
    option_type = st.selectbox("Option Type", ["CALL", "PUT"])
    ltp = df.loc[df["Strike Price"] == strike_choice, f"{option_type[0]}E LTP"].values[0]

    prices = np.linspace(spot_price - 500, spot_price + 500, 50)
    if option_type == "CALL":
        payoff = np.maximum(prices - strike_choice, 0) - ltp
    else:
        payoff = np.maximum(strike_choice - prices, 0) - ltp

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=prices, y=payoff, mode="lines", name="Payoff"))
    fig_payoff.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_payoff, use_container_width=True)

else:
    st.info("Please upload your Option Chain CSV to begin analysis.")
