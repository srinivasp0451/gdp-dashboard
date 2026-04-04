"""
╔══════════════════════════════════════════════════════════════════════╗
║           Dhan IP Registration Utility — SEBI Mandate                ║
║                                                                      ║
║  Run this ONCE per IP address (or whenever your public IP changes).  ║
║  After registering, the same IP is whitelisted permanently until     ║
║  you revoke it in the Dhan Developer Portal.                         ║
║                                                                      ║
║  Install : pip install streamlit requests                            ║
║  Run     : streamlit run dhan_register_ip.py                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import requests
import datetime
import json

st.set_page_config(
    page_title="Dhan IP Registration",
    page_icon="🔐",
    layout="centered",
)

st.markdown("""
<style>
.block-container{padding-top:1.5rem!important}
.reg-title{
    font-size:1.6rem;font-weight:900;
    background:linear-gradient(90deg,#64ffda,#00b4d8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.ip-box{
    background:#1a1f35;border:1px solid #2d3560;border-radius:12px;
    padding:16px 22px;margin:12px 0;font-family:monospace;
}
.step{
    background:#151a2d;border-left:3px solid #64ffda;
    border-radius:7px;padding:10px 14px;margin:8px 0;font-size:.9rem;
    color:#ccd6f6;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="reg-title">🔐 Dhan IP Registration</div>', unsafe_allow_html=True)
st.caption("SEBI mandates that every machine placing orders via Dhan API must "
           "have its public IP registered. Run this **once per IP address**.")

st.divider()

# ── When to run ───────────────────────────────────────────────────────────────
st.markdown("#### 📋 When do you need to run this?")
st.markdown("""
<div class="step">✅ <b>One-time</b> if your internet connection has a <b>static / fixed public IP</b>
  (office network, leased line, most cloud VMs). Register once and never again.</div>
<div class="step">🔄 <b>Re-run when IP changes</b> — home broadband / 4G / VPN users often
  get a new public IP each session. Run this again after any IP change before placing orders.</div>
<div class="step">🖥️ <b>Per machine</b> — if you deploy Smart Investing on a new server or laptop,
  register that machine's IP even if you registered a different machine before.</div>
""", unsafe_allow_html=True)

st.divider()

# ── Step 1: Detect public IP ──────────────────────────────────────────────────
st.markdown("#### Step 1 — Detect Your Public IP")

if st.button("🔍 Detect My Public IP", use_container_width=True):
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=8)
        ip_data = r.json()
        public_ip = ip_data.get("ip", "Unknown")
        st.session_state["public_ip"] = public_ip

        # Also get geo info
        try:
            geo = requests.get(f"https://ipapi.co/{public_ip}/json/", timeout=6).json()
            st.session_state["ip_geo"] = geo
        except Exception:
            st.session_state["ip_geo"] = {}
    except Exception as e:
        st.error(f"Could not detect IP: {e}")

if "public_ip" in st.session_state:
    ip   = st.session_state["public_ip"]
    geo  = st.session_state.get("ip_geo", {})
    city = geo.get("city", "—")
    isp  = geo.get("org", "—")
    st.markdown(f"""
<div class="ip-box">
  <div style="color:#8892b0;font-size:.75rem;text-transform:uppercase;letter-spacing:1px">
    Your Public IP Address</div>
  <div style="color:#64ffda;font-size:1.5rem;font-weight:700;margin:4px 0">{ip}</div>
  <div style="color:#8892b0;font-size:.8rem">📍 {city} &nbsp;|&nbsp; 🌐 {isp}</div>
  <div style="color:#ffd166;font-size:.78rem;margin-top:6px">
    ⚠️ This is the IP that must be whitelisted in Dhan Developer Portal.</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Step 2: Enter Dhan credentials ───────────────────────────────────────────
st.markdown("#### Step 2 — Enter Dhan Credentials")
st.caption("Used only for the API registration call. Not stored anywhere.")

col1, col2 = st.columns(2)
client_id    = col1.text_input("Client ID",    type="password", placeholder="Your Dhan client ID")
access_token = col2.text_input("Access Token", type="password", placeholder="Your Dhan access token")

st.divider()

# ── Step 3: Register ──────────────────────────────────────────────────────────
st.markdown("#### Step 3 — Register IP with Dhan API")

if st.button("🚀 Register IP Now", type="primary", use_container_width=True):
    if not client_id or not access_token:
        st.error("Please enter both Client ID and Access Token.")
    elif "public_ip" not in st.session_state:
        st.warning("Please detect your public IP first (Step 1).")
    else:
        with st.spinner("Calling Dhan API…"):
            result = {}
            result["public_ip"] = st.session_state["public_ip"]
            result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                # Dhan eDIS / TPIN initiation endpoint
                url = "https://api.dhan.co/edis/tpin"
                headers = {
                    "access-token": access_token,
                    "client-id":    client_id,
                    "Content-Type": "application/json",
                }
                resp = requests.post(url, headers=headers, timeout=12)
                result["http_status"] = resp.status_code
                result["response"]    = resp.text[:400]

                if resp.status_code in (200, 201, 202):
                    st.success(f"✅ API call succeeded (HTTP {resp.status_code}). "
                               f"IP `{result['public_ip']}` registration initiated.")
                elif resp.status_code == 401:
                    st.warning("⚠️ HTTP 401 — Invalid credentials. Check Client ID / Access Token.")
                elif resp.status_code == 403:
                    st.warning("⚠️ HTTP 403 — IP may already be registered, or credentials insufficient.")
                else:
                    st.info(f"ℹ️ HTTP {resp.status_code} — See raw response below.")

            except requests.exceptions.Timeout:
                st.error("Request timed out. Check your network and try again.")
                result["error"] = "timeout"
            except Exception as e:
                st.error(f"Request failed: {e}")
                result["error"] = str(e)

            st.session_state["reg_result"] = result

        # Show raw result
        if "reg_result" in st.session_state:
            with st.expander("📄 Raw API Response", expanded=True):
                st.json(st.session_state["reg_result"])

st.divider()

# ── Step 4: Manual Portal Whitelist ──────────────────────────────────────────
st.markdown("#### Step 4 — Whitelist in Dhan Developer Portal (required)")
st.info(
    "The API call alone is **not sufficient**. You must also **manually add your IP** "
    "in the Dhan Developer Portal:\n\n"
    "1. Go to [developer.dhan.co](https://developer.dhan.co/) and log in.\n"
    "2. Navigate to **API Access → Manage IP Whitelist**.\n"
    "3. Click **Add IP** and enter your public IP shown above.\n"
    "4. Save and wait ~2 minutes for it to take effect.\n\n"
    "After this, your `Smart Investing` app can place live orders without any further registration — "
    "**unless your IP changes**."
)

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="color:#8892b0;font-size:.75rem;text-align:center;padding:8px 0">
  This utility is part of the <b>Smart Investing</b> platform.<br>
  IP registration data is never stored or transmitted to any server other than Dhan's API.
</div>
""", unsafe_allow_html=True)
