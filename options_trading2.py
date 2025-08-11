import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from datetime import datetime, time
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Nifty Options Analysis Dashboard", page_icon="📈", layout="wide")

class OptionsAnalyzer:
    def __init__(self):
        self.current_time = datetime.now().time()

    def load_and_clean_data(self, uploaded_file):
        """Load NSE-style option chain CSV starting from STRIKE header line."""
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')

        header_idx = None
        for i, line in enumerate(lines):
            if 'STRIKE' in line.upper():
                header_idx = i
                break
        if header_idx is None:
            st.error("No STRIKE column found — verify file format.")
            return None, None

        csv_content = '\n'.join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_content))
        df.dropna(axis=1, how='all', inplace=True)
        df.columns = [str(c).strip() for c in df.columns]
        return self.parse_chain(df)

    def parse_chain(self, df):
        """Parse NSE option chain into separate Calls & Puts DataFrames."""
        strike_index = None
        for i, col in enumerate(df.columns):
            if 'STRIKE' in col.upper():
                strike_index = i
                break
        if strike_index is None:
            return pd.DataFrame(), pd.DataFrame()

        calls, puts = [], []
        for _, row in df.iterrows():
            strike = self.safe_num(row.iloc[strike_index])
            if strike == 0:
                continue
            try:
                call = {
                    'strike': strike,
                    'oi': self.safe_num(row.iloc[1]),
                    'chng_oi': self.safe_num(row.iloc[2]),
                    'volume': self.safe_num(row.iloc[3]),
                    'iv': self.safe_num(row.iloc[4]),
                    'ltp': self.safe_num(row.iloc[5]),
                    'chng': self.safe_num(row.iloc[6])
                }
                put = {
                    'strike': strike,
                    'ltp': self.safe_num(row.iloc[strike_index+6]),
                    'iv': self.safe_num(row.iloc[strike_index+7]),
                    'volume': self.safe_num(row.iloc[strike_index+8]),
                    'chng_oi': self.safe_num(row.iloc[strike_index+9]),
                    'oi': self.safe_num(row.iloc[strike_index+10]),
                    'chng': self.safe_num(row.iloc[strike_index+5])
                }
                if call['ltp'] > 0:
                    calls.append(call)
                if put['ltp'] > 0:
                    puts.append(put)
            except:
                continue
        return pd.DataFrame(calls), pd.DataFrame(puts)

    def safe_num(self, val):
        try:
            if pd.isna(val) or val in ['', '-']:
                return 0
            return float(str(val).replace(',', '').strip())
        except:
            return 0

    def get_market_phase(self):
        if time(9, 15) <= self.current_time <= time(10, 0): return 'OPENING'
        elif time(14, 30) <= self.current_time <= time(15, 30): return 'CLOSING'
        return 'MID_SESSION'

    def analyze_signal(self, row, spot, option_type):
        strike, ltp, volume, oi, chng_oi, iv = row['strike'], row['ltp'], row['volume'], row['oi'], row['chng_oi'], row['iv']
        moneyness = ((spot - strike) / spot * 100) if option_type == 'CALL' else ((strike - spot) / spot * 100)
        pop = max(5, min(95, 50 + (moneyness if option_type == 'CALL' else -moneyness) * 2 - abs(moneyness) * 0.5))

        volume_score = min(100, (volume / 500000) * 100)
        oi_score = min(100, (oi / 200000) * 100)
        iv_score = max(0, 100 - abs(iv - 15) * 5)
        score = (volume_score + oi_score + iv_score) / 3

        entry_cond = []
        if volume > 100000: entry_cond.append("High Volume")
        if oi > 100000: entry_cond.append("OI Build-up")
        if 10 <= iv <= 20: entry_cond.append("Optimal IV")
        if chng_oi > 5000:
            entry_cond.append("Positive OI Change" if option_type == 'CALL' else "Negative OI Change")

        risk = 'LOW' if abs(moneyness) < 1 else 'MEDIUM' if abs(moneyness) < 3 else 'HIGH'
        target1, target2 = round(ltp * 1.5, 2), round(ltp * 2, 2)
        stop_loss = round(ltp * 0.7, 2)

        if score > 75 and len(entry_cond) >= 2: rec = "STRONG BUY"
        elif score > 60 and len(entry_cond) >= 2: rec = "BUY"
        elif score > 50 and len(entry_cond) >= 1: rec = "WEAK BUY"
        else: rec = "AVOID"

        return {
            'option_type': option_type, 'strike': strike, 'ltp': ltp,
            'stop_loss': stop_loss, 'target_1': target1, 'target_2': target2,
            'probability': round(pop, 1), 'recommendation': rec, 'score': score,
            'risk_level': risk, 'entry_conditions': entry_cond
        }

    def get_recommendations(self, calls_df, puts_df, spot):
        recs = [self.analyze_signal(r, spot, 'CALL') for _, r in calls_df.iterrows()]
        recs += [self.analyze_signal(r, spot, 'PUT') for _, r in puts_df.iterrows()]
        return sorted(recs, key=lambda x: -x['score'])[:10]

    def create_charts(self, calls_df, puts_df, spot):
        fig_ltp = go.Figure()
        fig_ltp.add_trace(go.Scatter(x=calls_df['strike'], y=calls_df['ltp'], name='CALL LTP', line=dict(color='red')))
        fig_ltp.add_trace(go.Scatter(x=puts_df['strike'], y=puts_df['ltp'], name='PUT LTP', line=dict(color='green')))
        fig_ltp.add_vline(x=spot, line_dash="dash", line_color="blue")
        fig_ltp.update_layout(title="Call & Put Premiums (LTP)")

        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['oi'], name='Call OI', marker_color='red'))
        fig_oi.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['oi'], name='Put OI', marker_color='green'))
        fig_oi.update_layout(barmode='group', title="Open Interest")

        fig_ch_oi = go.Figure()
        fig_ch_oi.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['chng_oi'], name='Call ΔOI', marker_color='red'))
        fig_ch_oi.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['chng_oi'], name='Put ΔOI', marker_color='green'))
        fig_ch_oi.update_layout(barmode='group', title="Change in OI")

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['volume'], name='Call Volume', marker_color='red'))
        fig_vol.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['volume'], name='Put Volume', marker_color='green'))
        fig_vol.update_layout(barmode='group', title="Volume")

        return fig_ltp, fig_oi, fig_ch_oi, fig_vol

# -------- Streamlit App --------
def main():
    st.title("📊 Nifty Options Chain Analysis & Recommendations")
    analyzer = OptionsAnalyzer()

    uploaded_file = st.sidebar.file_uploader("Upload Option Chain CSV", type=['csv'])
    screenshot_file = st.sidebar.file_uploader("Upload Index Screenshot (optional)", type=['png','jpg','jpeg'])
    spot_price = st.sidebar.number_input("Current Spot Price", min_value=20000, max_value=30000, value=24500, step=1)

    if uploaded_file:
        calls_df, puts_df = analyzer.load_and_clean_data(uploaded_file)
        if calls_df is not None and not calls_df.empty:
            # Summary
            total_call_oi, total_put_oi = calls_df['oi'].sum(), puts_df['oi'].sum()
            pcr_oi = total_put_oi / total_call_oi if total_call_oi else 0
            sentiment = "Bullish" if pcr_oi > 1 else "Bearish" if pcr_oi < 1 else "Neutral"
            st.subheader("Summary - Option Chain Insights")
            st.write(f"**PCR (OI)**: {pcr_oi:.2f} → **{sentiment}** sentiment")
            st.write(f"Max Call OI at Strike: {calls_df.loc[calls_df['oi'].idxmax(), 'strike']}")
            st.write(f"Max Put OI at Strike: {puts_df.loc[puts_df['oi'].idxmax(), 'strike']}")
            st.write(f"Highest Call Volume at: {calls_df.loc[calls_df['volume'].idxmax(), 'strike']}")
            st.write(f"Highest Put Volume at: {puts_df.loc[puts_df['volume'].idxmax(), 'strike']}")

            if screenshot_file:
                st.subheader("Uploaded Index Chart Analysis")
                img = Image.open(screenshot_file)
                st.image(img, caption="Index chart snapshot")
                st.write("**Chart Observation:** Use visible price trend, S/R levels to confirm option chain signals.")
            else:
                st.info("No index chart uploaded.")

            recs = analyzer.get_recommendations(calls_df, puts_df, spot_price)
            st.subheader("🎯 Trade Recommendations")
            st.caption("Logic = High Volume + OI build-up + Optimal IV + Positive ΔOI + scoring from Volume, OI & IV deviation.")
            for r in recs:
                st.markdown(f"**{r['recommendation']}** → {r['option_type']} {r['strike']} @ ₹{r['ltp']} | "
                            f"SL ₹{r['stop_loss']} | T1 ₹{r['target_1']} | T2 ₹{r['target_2']} "
                            f"| POP {r['probability']}% | Risk {r['risk_level']}")
                if r['entry_conditions']:
                    st.write("Conditions met:", ", ".join(r['entry_conditions']))
                st.progress(int(r['score']))

            fig_ltp, fig_oi, fig_ch_oi, fig_vol = analyzer.create_charts(calls_df, puts_df, spot_price)
            st.plotly_chart(fig_ltp, use_container_width=True)
            st.plotly_chart(fig_oi, use_container_width=True)
            st.plotly_chart(fig_ch_oi, use_container_width=True)
            st.plotly_chart(fig_vol, use_container_width=True)

            st.subheader("📋 Raw Data")
            t1, t2 = st.tabs(["Calls", "Puts"])
            with t1: st.dataframe(calls_df)
            with t2: st.dataframe(puts_df)
        else:
            st.error("Could not parse CSV.")
    else:
        st.info("Please upload the Option Chain CSV.")

if __name__ == "__main__":
    main()
