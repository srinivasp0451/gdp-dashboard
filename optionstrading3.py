import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go

# -----------------------
# SAFE GET COLUMN FUNCTION
# -----------------------
def get_col_safe(df, col_name):
    """Return the column if exists, else a zero-filled Series."""
    if col_name in df.columns:
        return df[col_name].fillna(0)
    else:
        return pd.Series([0] * len(df), index=df.index)

# -----------------------
# FETCH NSE OPTION CHAIN DATA
# -----------------------
def fetch_option_chain(symbol):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        response = session.get(url, headers=headers, timeout=5)
        data = response.json()

        records = []
        for item in data["records"]["data"]:
            strike_price = item.get("strikePrice", None)
            expiry_date = item.get("expiryDate", None)

            ce_data = item.get("CE", {})
            pe_data = item.get("PE", {})

            records.append({
                "expiryDate": expiry_date,
                "strikePrice": strike_price,
                "c_oi": ce_data.get("openInterest"),
                "c_chng_in_oi": ce_data.get("changeinOpenInterest"),
                "c_volume": ce_data.get("totalTradedVolume"),
                "c_ltp": ce_data.get("lastPrice"),
                "p_oi": pe_data.get("openInterest"),
                "p_chng_in_oi": pe_data.get("changeinOpenInterest"),
                "p_volume": pe_data.get("totalTradedVolume"),
                "p_ltp": pe_data.get("lastPrice"),
            })

        return pd.DataFrame(records)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# -----------------------
# APP UI
# -----------------------
st.set_page_config(page_title="NSE Option Chain Analysis", layout="wide")
st.title("📈 NSE Option Chain Analysis")

# User Inputs
symbol_map = {
    "NIFTY 50": "NIFTY",
    "BANK NIFTY": "BANKNIFTY",
    "FIN NIFTY": "FINNIFTY",
    "MIDCAP NIFTY": "MIDCPNIFTY",
    "SENSEX": "SENSEX"
}

symbol_name = st.selectbox("Select Index", list(symbol_map.keys()))
symbol = symbol_map[symbol_name]

# Fetch Data
df = fetch_option_chain(symbol)

if not df.empty:
    # Expiry selection
    expiries = sorted(df["expiryDate"].dropna().unique())
    selected_expiry = st.selectbox("Select Expiry Date", expiries)
    df = df[df["expiryDate"] == selected_expiry].reset_index(drop=True)

    # Ensure calculations safe
    df["total_oi_change"] = get_col_safe(df, "c_chng_in_oi") + get_col_safe(df, "p_chng_in_oi")
    df["total_volume"] = get_col_safe(df, "c_volume") + get_col_safe(df, "p_volume")

    # ATM detection
    df["diff"] = abs(get_col_safe(df, "c_ltp") - get_col_safe(df, "p_ltp"))
    atm_strike = df.loc[df["diff"].idxmin(), "strikePrice"]

    # Probability of Profit (dummy calc for now)
    df["prob_profit"] = np.round(
        100 * (1 - (abs(df["strikePrice"] - atm_strike) / atm_strike)), 2
    )

    # -----------------------
    # DISPLAY TOP INFO
    # -----------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Index", symbol_name)
    with col2:
        st.metric("ATM Strike", atm_strike)
    with col3:
        st.metric("Date/Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # -----------------------
    # OI & Change in OI Plots
    # -----------------------
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["strikePrice"], y=get_col_safe(df, "c_oi"),
        name="CE OI", marker_color="green"
    ))
    fig.add_trace(go.Bar(
        x=df["strikePrice"], y=get_col_safe(df, "p_oi"),
        name="PE OI", marker_color="red"
    ))
    fig.update_layout(title="Open Interest", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["strikePrice"], y=get_col_safe(df, "c_chng_in_oi"),
        name="CE Chng OI", marker_color="lightgreen"
    ))
    fig2.add_trace(go.Bar(
        x=df["strikePrice"], y=get_col_safe(df, "p_chng_in_oi"),
        name="PE Chng OI", marker_color="pink"
    ))
    fig2.update_layout(title="Change in Open Interest", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------
    # CE & PE Premiums with Probability
    # -----------------------
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["strikePrice"], y=get_col_safe(df, "c_ltp"),
        mode="lines+markers", name="CE Premium", line=dict(color="green")
    ))
    fig3.add_trace(go.Scatter(
        x=df["strikePrice"], y=get_col_safe(df, "p_ltp"),
        mode="lines+markers", name="PE Premium", line=dict(color="red")
    ))
    fig3.update_layout(title="CE & PE Premiums")
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------
    # Data Table
    # -----------------------
    st.subheader("Option Chain Data with Probability of Profit")
    st.dataframe(df[[
        "strikePrice", "c_oi", "p_oi",
        "c_chng_in_oi", "p_chng_in_oi",
        "c_ltp", "p_ltp", "prob_profit"
    ]])

else:
    st.warning("No data available for the selected index.")
