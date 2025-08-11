import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from datetime import datetime, time
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# -------- SETTINGS --------
st.set_page_config(page_title="Nifty Options Analysis Dashboard",
                   page_icon="📈",
                   layout="wide")

# -------- CLASS --------
class OptionsAnalyzer:
    def __init__(self):
        self.current_time = datetime.now().time()

    def load_and_clean_data(self, uploaded_file):
        """Load NSE-style option chain CSV starting from STRIKE header line."""
        try:
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

            calls_df, puts_df = self.parse_nse_chain(df)
            return calls_df, puts_df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None, None

    def parse_nse_chain(self, df):
        """Parse option chain into separate Calls and Puts dataframes."""
        strike_index = None
        for i, col in enumerate(df.columns):
            if 'STRIKE' in col.upper():
                strike_index = i
                break
        if strike_index is None:
            return pd.DataFrame(), pd.DataFrame()

        calls_data, puts_data = [], []
        for _, row in df.iterrows():
            try:
                strike = self.safe_num(row.iloc[strike_index])
                if strike == 0:
                    continue
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
                    'chng': self.safe_num(row.iloc[strike_index+5]),
                    'ltp': self.safe_num(row.iloc[strike_index+6]),
                    'iv': self.safe_num(row.iloc[strike_index+7]),
                    'volume': self.safe_num(row.iloc[strike_index+8]),
                    'chng_oi': self.safe_num(row.iloc[strike_index+9]),
                    'oi': self.safe_num(row.iloc[strike_index+10])
                }
                if call['ltp'] > 0:
                    calls_data.append(call)
                if put['ltp'] > 0:
                    puts_data.append(put)
            except:
                continue
        return pd.DataFrame(calls_data), pd.DataFrame(puts_data)

    def safe_num(self, val):
        try:
            if pd.isna(val) or val in ['', '-']:
                return 0
            return float(str(val).replace(',', '').strip())
        except:
            return 0

    def get_market_phase(self):
        if time(9, 15) <= self.current_time <= time(10, 0):
            return 'OPENING'
        elif time(14, 30) <= self.current_time <= time(15, 30):
            return 'CLOSING'
        else:
            return 'MID_SESSION'

    def analyze_signal(self, row, current_spot, option_type):
        """Generate recommendation for a single row."""
        strike, ltp, volume, oi, chng_oi, iv = row['strike'], row['ltp'], row['volume'], row['oi'], row['chng_oi'], row['iv']
        moneyness = ((current_spot - strike) / current_spot * 100) if option_type == 'CALL' else ((strike - current_spot) / current_spot * 100)

        # POP estimation
        pop = max(5, min(95, 50 + (moneyness * -1 if option_type == 'PUT' else moneyness) * 2 - abs(moneyness) * 0.5))

        # Scoring
        volume_score = min(100, (volume / 500000) * 100)
        oi_score = min(100, (oi / 200000) * 100)
        iv_score = max(0, 100 - abs(iv - 15) * 5)
        score = (volume_score + oi_score + iv_score) / 3

        # Entry logic
        entry_cond = []
        if volume > 100000: entry_cond.append("High Volume")
        if oi > 100000: entry_cond.append("OI Build-up")
        if 10 <= iv <= 20: entry_cond.append("Optimal IV")
        if chng_oi > 5000:
            entry_cond.append("Positive OI Change" if option_type == 'CALL' else "Negative OI Change")

        # Risk & Targets
        risk_level = 'LOW' if abs(moneyness) < 1 else 'MEDIUM' if abs(moneyness) < 3 else 'HIGH'
        target1 = round(ltp * 1.5, 2)
        target2 = round(ltp * 2, 2)
        stop_loss = round(ltp * 0.7, 2)

        # Recommendation
        if score > 75 and len(entry_cond) >= 2:
            rec = "STRONG BUY"
        elif score > 60 and len(entry_cond) >= 2:
            rec = "BUY"
        elif score > 50 and len(entry_cond) >= 1:
            rec = "WEAK BUY"
        else:
            rec = "AVOID"

        return {
            'option_type': option_type, 'strike': strike, 'ltp': ltp,
            'stop_loss': stop_loss, 'target_1': target1, 'target_2': target2,
            'probability': round(pop, 1), 'recommendation': rec, 'score': score,
            'risk_level': risk_level, 'entry_conditions': entry_cond
        }

    def get_recommendations(self, calls_df, puts_df, spot):
        recs = []
        for _, r in calls_df.iterrows():
            recs.append(self.analyze_signal(r, spot, 'CALL'))
        for _, r in puts_df.iterrows():
            recs.append(self.analyze_signal(r, spot, 'PUT'))
        return sorted(recs, key=lambda x: -x['score'])[:10]

    def create_charts(self, calls_df, puts_df, spot):
        # LTP Chart
        fig_ltp = go.Figure()
        fig_ltp.add_trace(go.Scatter(x=calls_df['strike'], y=calls_df['ltp'], name='CALL LTP', line=dict(color='red')))
        fig_ltp.add_trace(go.Scatter(x=puts_df['strike'], y=puts_df['ltp'], name='PUT LTP', line=dict(color='green')))
        fig_ltp.add_vline(x=spot, line_dash="dash", line_color="blue")
        fig_ltp.update_layout(title="Call & Put Premiums", xaxis_title="Strike", yaxis_title="LTP")

        # OI Chart
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['oi'], name='Call OI', marker_color='red'))
        fig_oi.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['oi'], name='Put OI', marker_color='green'))
        fig_oi.update_layout(barmode='group', title="Open Interest")

        # Change in OI Chart
        fig_chg_oi = go.Figure()
        fig_chg_oi.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['chng_oi'], name='Call ΔOI', marker_color='red'))
        fig_chg_oi.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['chng_oi'], name='Put ΔOI', marker_color='green'))
        fig_chg_oi.update_layout(barmode='group', title="Change in OI")

        # Volume Chart
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['volume'], name='Call Volume', marker_color='red'))
        fig_vol.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['volume'], name='Put Volume', marker_color='green'))
        fig_vol.update_layout(barmode='group', title="Volume")

        return fig_ltp, fig_oi, fig_chg_oi, fig_vol

# --------- APP ----------
def main():
    st.title("📊 Nifty Options Chain Analysis & Recommendations")

    analyzer = OptionsAnalyzer()
    uploaded_file = st.sidebar.file_uploader("Upload Option Chain CSV", type=['csv'])
    screenshot_file = st.sidebar.file_uploader("Upload Index Screenshot (optional)", type=['png', 'jpg', 'jpeg'])
    spot_price = st.sidebar.number_input("Current Spot Price", min_value=20000, max_value=30000, value=24500, step=1)

    if uploaded_file:
        calls_df, puts_df = analyzer.load_and_clean_data(uploaded_file)
        if calls_df is not None and not calls_df.empty:
            fig_ltp, fig_oi, fig_chg_oi, fig_vol = analyzer.create_charts(calls_df, puts_df, spot_price)

            # Summary: Option Chain
            total_call_oi = calls_df['oi'].sum()
            total_put_oi = puts_df['oi'].sum()
            pcr_oi = total_put_oi / total_call_oi if total_call_oi else 0
            sentiment = "Bullish" if pcr_oi > 1 else "Bearish" if pcr_oi < 1 else "Neutral"
            st.subheader("Summary - Option Chain Insights")
            st.write(f"**PCR (OI):** {pcr_oi:.2f} → Market Sentiment: **{sentiment}**")
            st.write(f"- Call OI Concentration: Strike {calls_df.loc[calls_df['oi'].idxmax(), 'strike']}")
            st.write(f"- Put OI Concentration: Strike {puts_df.loc[puts_df['oi'].idxmax(), 'strike']}")
            st.write(f"- Highest Call Volume @ {calls_df.loc[calls_df['volume'].idxmax(), 'strike']}")
            st.write(f"- Highest Put Volume @ {puts_df.loc[puts_df['volume'].idxmax(), 'strike']}")

            # Chart (image) discussion placeholder
            if screenshot_file:
                st.subheader("Uploaded Index Chart")
                img = Image.open(screenshot_file)
                st.image(img, caption="Index chart provided by user")
                st.write("**Manual Observation:** Visual inspection of index trend, support/resistance from chart to be combined with Option Chain view.")
            else:
                st.info("No chart uploaded — only Option Chain data is analyzed.")

            # Recommendations
            recs = analyzer.get_recommendations(calls_df, puts_df, spot_price)
            st.subheader("🎯 Top Trade Recommendations (Logic-driven)")
            st.caption("Logic: Score computed from Volume, OI, IV proximity, ΔOI; Entry conditions must be met (High Vol, OI build-up, Optimal IV, Positive ΔOI). Targets/SL from LTP.")
            for r in recs:
                st.markdown(f"**{r['recommendation']}** → {r['option_type']} {r['strike']} @ ₹{r['ltp']} | SL ₹{r['stop_loss']} | T1 ₹{r['target_1']} | T2 ₹{r['target_2']} | POP {r['probability']}% | Risk {r['risk_level']}")
                if r['entry_conditions']:
                    st.write("Conditions:", ", ".join(r['entry_conditions']))
                st.progress(int(r['score']))

            # Display charts
            st.plotly_chart(fig_ltp, use_container_width=True)
            st.plotly_chart(fig_oi, use_container_width=True)
            st.plotly_chart(fig_chg_oi, use_container_width=True)
            st.plotly_chart(fig_vol, use_container_width=True)

            # Raw Data
            st.subheader("Raw Parsed Data")
            t1, t2 = st.tabs(["Calls", "Puts"])
            with t1:
                st.dataframe(calls_df)
            with t2:
                st.dataframe(puts_df)
        else:
            st.error("Could not parse the CSV.")
    else:
        st.info("Please upload the Option Chain CSV to start analysis.")

if __name__ == "__main__":
    main()
