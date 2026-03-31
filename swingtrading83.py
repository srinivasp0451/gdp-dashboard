"""
NIFTY EXPIRY DAY -- LIVE ROLLING ATM STRADDLE TABLE
Data Source : NSE India Live Option Chain (no yfinance, no delay)
Run         : streamlit run nifty_expiry_straddle.py
Install     : pip install streamlit requests pandas pytz
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, date, time as dtime
import pytz
import time

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# CONFIG
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
IST = pytz.timezone("Asia/Kolkata")

SESSION_BINS = [
    {"bin": 1, "label": "Opening",      "start": (9, 15),  "end": (9, 45)},
    {"bin": 2, "label": "Morning",      "start": (9, 45),  "end": (11, 0)},
    {"bin": 3, "label": "Mid-Session",  "start": (11, 0),  "end": (13, 0)},
    {"bin": 4, "label": "Afternoon",    "start": (13, 0),  "end": (14, 30)},
    {"bin": 5, "label": "Expiry Close", "start": (14, 30), "end": (15, 31)},
]

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# NSE SESSION
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def create_nse_session():
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    for url in ["https://www.nseindia.com", "https://www.nseindia.com/option-chain"]:
        try:
            session.get(url, timeout=10)
            time.sleep(0.5)
        except Exception:
            pass
    return session


def safe_get(session, url):
    for attempt in range(3):
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.JSONDecodeError:
            time.sleep(1)
        except Exception as e:
            if attempt == 2:
                st.warning("NSE fetch failed: " + str(e))
            time.sleep(1 + attempt)
    return None


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# DATA FETCH
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def fetch_all_data():
    result = {
        "ok": False,
        "spot": None,
        "vix": None,
        "expiry": None,
        "records": [],
        "error": "",
        "timestamp": datetime.now(IST).strftime("%H:%M:%S"),
    }

    session = create_nse_session()

    # India VIX
    vix_data = safe_get(session, "https://www.nseindia.com/api/allIndices")
    if vix_data:
        for idx in vix_data.get("data", []):
            if "VIX" in idx.get("index", "").upper():
                try:
                    result["vix"] = round(float(idx.get("last", 0)), 2)
                except Exception:
                    pass
                break

    # Option Chain
    oc_data = safe_get(
        session,
        "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    )
    if oc_data is None:
        result["error"] = "Option chain fetch failed. NSE may be blocking -- retry in 60s."
        return result

    records_root = oc_data.get("records", {})
    spot_raw = records_root.get("underlyingValue", 0)
    if not spot_raw:
        result["error"] = "Spot price missing in NSE response."
        return result

    result["spot"] = round(float(spot_raw), 2)
    expiry_dates = records_root.get("expiryDates", [])
    nearest = get_nearest_expiry(expiry_dates)
    result["expiry"] = nearest

    all_records = records_root.get("data", [])
    result["records"] = [r for r in all_records if r.get("expiryDate") == nearest]
    result["ok"] = True
    return result


def get_nearest_expiry(expiry_dates):
    today = date.today()
    for exp in expiry_dates:
        try:
            d = datetime.strptime(exp, "%d-%b-%Y").date()
            if d >= today:
                return exp
        except Exception:
            pass
    return expiry_dates[0] if expiry_dates else None


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# OPTION CHAIN CALCULATIONS
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def find_atm(records, spot):
    strikes = [r["strikePrice"] for r in records if "strikePrice" in r]
    if not strikes:
        return 0
    return min(strikes, key=lambda x: abs(x - spot))


def straddle_premium(records, atm):
    for r in records:
        if r.get("strikePrice") == atm:
            ce = float(r.get("CE", {}).get("lastPrice", 0) or 0)
            pe = float(r.get("PE", {}).get("lastPrice", 0) or 0)
            return round(ce + pe, 2), round(ce, 2), round(pe, 2)
    return 0.0, 0.0, 0.0


def calc_pcr(records):
    ce_oi = sum(r.get("CE", {}).get("openInterest", 0) or 0 for r in records)
    pe_oi = sum(r.get("PE", {}).get("openInterest", 0) or 0 for r in records)
    if ce_oi == 0:
        return 0.0
    return round(pe_oi / ce_oi, 2)


def calc_total_oi(records):
    total = sum(
        (r.get("CE", {}).get("openInterest", 0) or 0)
        + (r.get("PE", {}).get("openInterest", 0) or 0)
        for r in records
    )
    if total >= 10000000:
        return str(round(total / 10000000, 2)) + "Cr"
    return str(round(total / 100000, 1)) + "L"


def calc_max_pain(records):
    strikes = sorted({r["strikePrice"] for r in records if "strikePrice" in r})
    if not strikes:
        return 0

    ce_oi = {}
    pe_oi = {}
    for r in records:
        s = r.get("strikePrice")
        if s is not None:
            ce_oi[s] = r.get("CE", {}).get("openInterest", 0) or 0
            pe_oi[s] = r.get("PE", {}).get("openInterest", 0) or 0

    min_loss = float("inf")
    pain_strike = 0

    for test in strikes:
        loss = 0
        for s in strikes:
            if test > s:
                loss += (test - s) * ce_oi.get(s, 0)
            if test < s:
                loss += (s - test) * pe_oi.get(s, 0)
        if loss < min_loss:
            min_loss = loss
            pain_strike = test

    return pain_strike


def calc_atm_oi(records, atm):
    for r in records:
        if r.get("strikePrice") == atm:
            ce = r.get("CE", {}).get("openInterest", 0) or 0
            pe = r.get("PE", {}).get("openInterest", 0) or 0
            return str(round(ce / 100)) + "K", str(round(pe / 100)) + "K"
    return "--", "--"


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# TIME BIN LOGIC
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def get_current_bin():
    now = datetime.now(IST).time()
    for b in SESSION_BINS:
        if dtime(*b["start"]) <= now < dtime(*b["end"]):
            return b["bin"]
    if now >= dtime(15, 31):
        return 6   # post-market
    return 0       # pre-market


def market_status():
    now = datetime.now(IST).time()
    if now < dtime(9, 15):
        return "Pre-Market"
    if now >= dtime(15, 30):
        return "Market Closed"
    return "Market LIVE"


def bin_time_str(b):
    sh, sm = b["start"]
    eh, em = b["end"]
    return "%02d:%02d - %02d:%02d %s" % (sh, sm, eh, em, b["label"])


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# OBSERVATIONS ENGINE
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def generate_observations(snapshots, spot, vix, max_pain, pcr, straddle, spot_open):
    obs = []

    if not snapshots and get_current_bin() == 0:
        obs.append("Waiting for market open at 9:15 IST. The table will populate once the opening bell rings.")
        return obs

    # VIX
    if vix:
        if vix >= 28:
            obs.append(
                "VIX CRITICAL at " + str(vix) + ": Far above the normal expiry-day range of 12-18. "
                "Extreme fear in the market. Option premiums are massively inflated. "
                "Straddle sellers face high risk. Buyers benefit from elevated IV but need a large move to profit."
            )
        elif vix >= 22:
            obs.append(
                "VIX Elevated at " + str(vix) + ": Market is fearful. Straddle premiums are significantly "
                "higher than calm expiry days. Watch for IV crush after any event resolves."
            )
        elif vix >= 16:
            obs.append(
                "VIX Moderate at " + str(vix) + ": Slightly above comfort zone. Premiums are mildly inflated. "
                "Normal expiry theta decay expected."
            )
        else:
            obs.append(
                "VIX Calm at " + str(vix) + ": Ideal expiry conditions for sellers. Premiums are in the "
                "expected range and theta decay is the dominant force today."
            )

    # Spot vs Max Pain
    if spot and max_pain:
        diff = spot - max_pain
        abs_diff = abs(diff)
        pct_diff = round(abs_diff / max_pain * 100, 2)
        if abs_diff < 80:
            obs.append(
                "Spot (" + str(round(spot)) + ") is within " + str(round(abs_diff)) + " pts of Max Pain (" + str(max_pain) + "). "
                "Market is gravitating toward max pain -- this is where option writers lose the least. "
                "Expect resistance to breakouts from here until expiry."
            )
        elif diff < 0:
            obs.append(
                "Spot (" + str(round(spot)) + ") is " + str(round(abs_diff)) + " pts BELOW Max Pain (" + str(max_pain) + ") [" + str(pct_diff) + "%]. "
                "Bears fully in control, overriding max pain magnetism. Strong institutional selling. "
                "A pullback is possible but not guaranteed -- large FII short positions can keep spot pinned low."
            )
        else:
            obs.append(
                "Spot (" + str(round(spot)) + ") is " + str(round(abs_diff)) + " pts ABOVE Max Pain (" + str(max_pain) + ") [" + str(pct_diff) + "%]. "
                "Bulls pushing above max pain. CE writers are under pressure. "
                "Watch for profit-taking near the max pain level."
            )

    # PCR
    if pcr:
        if pcr < 0.65:
            obs.append(
                "PCR at " + str(pcr) + " -- Extremely Bearish. Calls dominating put OI by a wide margin. "
                "Aggressive put buying or call writing at scale. High probability of continued bearish pressure."
            )
        elif pcr < 0.85:
            obs.append(
                "PCR at " + str(pcr) + " -- Bearish Bias. More puts than calls in OI, "
                "indicating fear and downside expectation."
            )
        elif pcr <= 1.15:
            obs.append(
                "PCR at " + str(pcr) + " -- Neutral. Balanced call and put OI. "
                "Market undecided on direction -- range-bound action likely near ATM."
            )
        else:
            obs.append(
                "PCR at " + str(pcr) + " -- Bullish. Active put writing signals confidence in holding current "
                "levels or moving up. Watch for reversal if PCR spikes above 1.5."
            )

    # Straddle decay trend
    bins_done = sorted(snapshots.keys())
    premiums = [snapshots[b]["straddle"] for b in bins_done if snapshots[b].get("straddle")]
    if len(premiums) >= 2:
        first_p = premiums[0]
        last_p = premiums[-1]
        decay = first_p - last_p
        decay_pct = round(decay / first_p * 100, 1) if first_p > 0 else 0
        if decay_pct >= 60:
            intensity = "Aggressive"
            comment = "Theta is dominating -- straddle melting fast. Option buyers losing money rapidly."
        elif decay_pct >= 30:
            intensity = "Moderate"
            comment = "Normal expiry-day decay. Straddle losing value steadily."
        else:
            intensity = "Slow"
            comment = "Premium is sticky -- possibly due to high VIX or an ongoing directional move."
        obs.append(
            "Theta Decay [" + intensity + "]: Straddle dropped from Rs." + str(first_p) +
            " (opening) to Rs." + str(last_p) + " now. " +
            "That is Rs." + str(round(decay)) + " (" + str(decay_pct) + "%) eroded. " + comment
        )

    # Spot movement from open
    if spot_open and spot:
        chg = spot - spot_open
        pct = round(chg / spot_open * 100, 2)
        direction = "UP" if chg > 0 else "DOWN"
        bias = (
            "Bullish momentum -- call writers under pressure."
            if chg > 0
            else "Bearish momentum -- put writers bleeding. OTM puts may have turned ITM."
        )
        obs.append(
            "Spot moved " + direction + " " + str(round(abs(chg))) + " pts (" + str(abs(pct)) + "%) "
            "from day-open of " + str(round(spot_open)) + " to current " + str(round(spot)) + ". " + bias
        )

    # ATM shift
    if len(bins_done) >= 2:
        first_atm = snapshots[bins_done[0]]["atm"]
        last_atm = snapshots[bins_done[-1]]["atm"]
        atm_shift = abs(last_atm - first_atm)
        if atm_shift > 0:
            direction = "lower" if last_atm < first_atm else "higher"
            leg_comment = (
                "CE leg is now worthless -- full loss on that side for straddle buyer."
                if direction == "lower"
                else "PE leg is now worthless -- full loss on that side for straddle buyer."
            )
            obs.append(
                "Rolling ATM shifted " + direction + " by " + str(atm_shift) + " pts "
                "(" + str(first_atm) + " -> " + str(last_atm) + "). " + leg_comment
            )

    # Strategy signal for early bins
    current_bin = get_current_bin()
    if current_bin in [1, 2] and straddle:
        if vix and vix > 22:
            obs.append(
                "STRATEGY WARNING: Do NOT buy straddle at open with VIX at " + str(vix) + ". "
                "High VIX = inflated premium = IV crush will erode your straddle even if market moves. "
                "Better approach: Wait for VIX to spike further and sell OTM strangles, OR "
                "wait for a directional breakout and buy only the directional leg."
            )
        else:
            breakeven = round(straddle * 1.5)
            obs.append(
                "Strategy Signal (Bin " + str(current_bin) + "): Straddle at Rs." + str(straddle) +
                " with VIX " + str(vix) + ". "
                "Spot needs to move more than " + str(breakeven) + " pts from ATM for buyer to profit. "
                "Otherwise, theta wins."
            )

    if not obs:
        obs.append("Collecting data -- observations will appear once at least one bin is captured.")

    return obs


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# SESSION STATE INIT
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
if "snapshots" not in st.session_state:
    st.session_state.snapshots = {}
if "spot_open" not in st.session_state:
    st.session_state.spot_open = None
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = None


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# PAGE SETUP
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.set_page_config(
    page_title="Nifty Expiry Straddle Table",
    page_icon="\ud83d\udcca",
    layout="wide",
)

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 10px 14px;
}
.obs-card {
    background: #f0f4ff;
    border-left: 4px solid #4361ee;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 14px;
    line-height: 1.6;
}
.header-band {
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    color: white;
    padding: 16px 20px;
    border-radius: 10px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# HEADER
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
now_ist = datetime.now(IST)
mkt = market_status()
current_bin = get_current_bin()

st.markdown(
    "<div class='header-band'>"
    "<span style='font-size:22px;font-weight:700'>\ud83d\udcca Nifty Expiry Day -- Live Rolling ATM Straddle Table</span><br>"
    "<span style='font-size:13px;opacity:0.8'>Live NSE data | No yfinance | Auto 5-bin capture | "
    + now_ist.strftime("%d-%b-%Y %H:%M:%S IST") + " | " + mkt + "</span>"
    "</div>",
    unsafe_allow_html=True
)

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# CONTROLS
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
with c1:
    auto_ref = st.toggle("Auto Refresh", value=True)
with c2:
    refresh_interval = st.selectbox("Interval (s)", [30, 60, 120], index=1, label_visibility="collapsed")
with c3:
    if st.button("Fetch Now", type="primary"):
        st.session_state.refresh_count += 1
        st.rerun()
with c4:
    last = st.session_state.last_fetch_time or "Never"
    st.caption("Refresh #" + str(st.session_state.refresh_count) + " | Last fetch: " + last)

st.markdown("---")

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# FETCH LIVE DATA
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
with st.spinner("Connecting to NSE India -- fetching live option chain..."):
    data = fetch_all_data()

if not data["ok"]:
    st.error("NSE Error: " + data["error"])
    st.info(
        "NSE blocks rapid repeated requests. Wait 30-60s and retry. "
        "If this persists outside market hours, NSE may be in maintenance mode."
    )
    st.stop()

spot      = data["spot"]
vix       = data["vix"]
records   = data["records"]
expiry    = data["expiry"]
fetch_ts  = data["timestamp"]

atm                   = find_atm(records, spot)
straddle, ce_px, pe_px = straddle_premium(records, atm)
pcr                   = calc_pcr(records)
total_oi              = calc_total_oi(records)
max_pain              = calc_max_pain(records)
atm_ce_oi, atm_pe_oi = calc_atm_oi(records, atm)

st.session_state.last_fetch_time = fetch_ts

# Store spot at first live capture
if st.session_state.spot_open is None and current_bin in [1, 2]:
    st.session_state.spot_open = spot

# Build and store snapshot for current bin
snapshot = {
    "vix":      vix,
    "spot":     spot,
    "atm":      atm,
    "straddle": straddle,
    "ce_px":    ce_px,
    "pe_px":    pe_px,
    "max_pain": max_pain,
    "pcr":      pcr,
    "total_oi": total_oi,
    "atm_ce":   atm_ce_oi,
    "atm_pe":   atm_pe_oi,
    "time":     now_ist.strftime("%H:%M"),
}

if current_bin in range(1, 7):
    st.session_state.snapshots[current_bin] = snapshot
    if current_bin == 6 and 5 not in st.session_state.snapshots:
        st.session_state.snapshots[5] = snapshot

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# LIVE METRICS
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.subheader("Live Snapshot -- Expiry: " + str(expiry))

m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)

spot_delta = None
if st.session_state.spot_open:
    delta_val = round(spot - st.session_state.spot_open, 2)
    spot_delta = str(delta_val) + " from open"

m1.metric("Nifty Spot",    str(spot),    spot_delta)
m2.metric("India VIX",     str(vix))
m3.metric("ATM Strike",    str(atm))
m4.metric("CE (ATM)",      "Rs." + str(ce_px))
m5.metric("PE (ATM)",      "Rs." + str(pe_px))
m6.metric("Straddle",      "Rs." + str(straddle))
m7.metric("Max Pain",      str(max_pain))
m8.metric("PCR (OI)",      str(pcr))

st.markdown("---")

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# 5-BIN TABLE
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.subheader("5-Bin Expiry Session Table (Rolling ATM Straddle)")

rows = []
for b in SESSION_BINS:
    bn      = b["bin"]
    label   = "Bin " + str(bn) + " | " + bin_time_str(b)
    is_live = (bn == current_bin)
    is_past = (bn < current_bin) or (current_bin == 6)

    if bn in st.session_state.snapshots:
        s = st.session_state.snapshots[bn]
        status = "LIVE" if is_live else "Captured"
        row = {
            "Session": label,
            "Status":       status,
            "India VIX":    s["vix"],
            "Nifty Spot":   s["spot"],
            "ATM Strike":   s["atm"],
            "CE (Rs.)":     s["ce_px"],
            "PE (Rs.)":     s["pe_px"],
            "Straddle (Rs.)": s["straddle"],
            "Max Pain":     s["max_pain"],
            "PCR (OI)":     s["pcr"],
            "Total OI":     s["total_oi"],
            "ATM CE OI":    s["atm_ce"],
            "ATM PE OI":    s["atm_pe"],
            "At":           s["time"],
        }
    #elif
