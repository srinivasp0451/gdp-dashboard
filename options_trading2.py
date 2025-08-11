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

# Session storage for premium vs time data
if "premium_history" not in st.session_state:
    st.session_state.premium_history = {"time": [], "calls": {}, "puts": {}}

class OptionsAnalyzer:
    def parse_chain(self, df):
        # Identify strike column
        df.columns = [c.strip().upper() for c in df.columns]
        strike_index = df.columns.get_loc("STRIKE")
        calls, puts = [], []

        for _, row in df.iterrows():
            try:
                strike = self.safe_num(row.iloc[strike_index])
                if strike == 0:
                    continue
                # Correct NSE offsets for CALLS before STRIKE
                call_oi = self.safe_num(row.iloc[strike_index - 11])
                call_chng_oi = self.safe_num(row.iloc[strike_index - 10])
                call_vol = self.safe_num(row.iloc[strike_index - 9])
                call_iv = self.safe_num(row.iloc[strike_index - 8])
                call_ltp = self.safe_num(row.iloc[strike_index - 7])
                call_chng = self.safe_num(row.iloc[strike_index - 6])

                # Correct offsets for PUTS after STRIKE
                put_chng = self.safe_num(row.iloc[strike_index + 5])
                put_ltp = self.safe_num(row.iloc[strike_index + 6])
                put_iv = self.safe_num(row.iloc[strike_index + 7])
                put_vol = self.safe_num(row.iloc[strike_index + 8])
                put_chng_oi = self.safe_num(row.iloc[strike_index + 9])
                put_oi = self.safe_num(row.iloc[strike_index + 10])

                if call_ltp > 0:
                    calls.append({
                        "strike": strike, "oi": call_oi, "chng_oi": call_chng_oi,
                        "volume": call_vol, "iv": call_iv, "ltp": call_ltp, "chng": call_chng
                    })
                if put_ltp > 0:
                    puts.append({
                        "strike": strike, "oi": put_oi, "chng_oi": put_chng_oi,
                        "volume": put_vol, "iv": put_iv, "ltp": put_ltp, "chng": put_chng
                    })
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

    def load_and_clean_data(self, uploaded_file):
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')
        header_idx = None
        for i, line in enumerate(lines):
            if 'STRIKE' in line.upper():
                header_idx = i
                break
        if header_idx is None:
            st.error("Couldn't find STRIKE header in file.")
            return None, None
        csv_content = '\n'.join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_content))
        df.dropna(axis=1, how='all', inplace=True)
        return self.parse_chain(df)

    def analyze_signal(self, row, spot, option_type, max_oi, max_vol):
        strike, ltp, volume, oi, chng_oi, iv = row['strike'], row['ltp'], row['volume'], row['oi'], row['chng_oi'], row['iv']
        moneyness = ((spot - strike) / spot * 100) if option_type == 'CALL' else ((strike - spot) / spot * 100)

        # Improved POP: base from moneyness + OI/Vol weight
        base_pop = 50 - abs(moneyness) * 2
        oi_factor = (oi / max_oi) * 20 if max_oi else 0
        vol_factor = min(10, (volume / max_vol) * 10) if max_vol else 0
        chg_factor = 5 if chng_oi > 0 else -5 if chng_oi < 0 else 0
        pop = max(5, min(95, base_pop + oi_factor + vol_factor + chg_factor))

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
            'option_type': option_type, 'strike': strike, 'ltp': ltp, 'stop_loss': stop_loss,
            'target_1': target1, 'target_2': target2, 'probability': round(pop, 1),
            'recommendation': rec, 'score': score, 'risk_level': risk, 'entry_conditions': entry_cond
        }

    def get_recommendations(self, calls_df, puts_df, spot):
        max_oi_call, max_vol_call = calls_df['oi'].max(), calls_df['volume'].max()
        max_oi_put, max_vol_put = puts_df['oi'].max(), puts_df['volume'].max()

        recs = [self.analyze_signal(r, spot, 'CALL', max_oi_call, max_vol_call) for _, r in calls_df.iterrows()]
        recs += [self.analyze_signal(r, spot, 'PUT', max_oi_put, max_vol_put) for _, r in puts_df.iterrows()]
        return sorted(recs, key=lambda x: -x['score'])[:10]

    def update_premium_history(self, calls_df, puts_df):
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.premium_history["time"].append(timestamp)
        for df, side in [(calls_df, "calls"), (puts_df, "puts")]:
            for _, r in df.iterrows():
                strike = str(int(r['strike']))
                if strike not in st.session_state.premium_history[side]:
                    st.session_state.premium_history[side][strike] = []
                st.session_state.premium_history[side][strike].append(r['ltp'])

    def plot_premium_vs_time(self):
        fig = go.Figure()
        for strike, lst in st.session_state.premium_history["calls"].items():
            fig.add_trace(go.Scatter(x=st.session_state.premium_history["time"], y=lst, mode='lines+markers', name=f"CALL {strike}"))
        for strike, lst in st.session_state.premium_history["puts"].items():
            fig.add_trace(go.Scatter(x=st.session_state.premium_history["time"], y=lst, mode='lines+markers', name=f"PUT {strike}"))
        fig.update_layout(title="Premiums vs Time (Legend = Strike)", xaxis_title="Time", yaxis_title="LTP")
        return fig

    def summarize_chart(self, img):
        # very basic brightness-based trend check (placeholder)
        gray = img.convert("L")
        arr = np.array(gray)
        avg_brightness = arr.mean()
        if avg_brightness > 180:
            return "Chart appears bright - possible high bullish candles or uptrend bias."
        elif avg_brightness < 80:
            return "Chart appears darker - possible high bearish candles or downtrend bias."
        else:
            return "Chart has mixed brightness - possible sideways/consolidation."

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
            analyzer.update_premium_history(calls_df, puts_df)

            st.metric("Current Spot Price", f"{spot_price}")

            # Option chain summary
            total_call_oi, total_put_oi = calls_df['oi'].sum(), puts_df['oi'].sum()
            pcr_oi = total_put_oi / total_call_oi if total_call_oi else 0
            sentiment = "Bullish" if pcr_oi > 1 else "Bearish" if pcr_oi < 1 else "Neutral"
            st.subheader("Option Chain Summary")
            st.write(f"PCR (OI): {pcr_oi:.2f} → {sentiment}")
            st.write(f"Max CE OI @ {calls_df.loc[calls_df['oi'].idxmax(), 'strike']}")
            st.write(f"Max PE OI @ {puts_df.loc[puts_df['oi'].idxmax(), 'strike']}")

            # Chart summary
            if screenshot_file:
                img = Image.open(screenshot_file)
                st.image(img, caption="Uploaded Index Chart")
                chart_summary = analyzer.summarize_chart(img)
                st.subheader("Chart Summary")
                st.write(chart_summary)
            else:
                st.info("No chart uploaded.")

            # Recommendations
            recs = analyzer.get_recommendations(calls_df, puts_df, spot_price)
            st.subheader("Recommendations")
            for r in recs:
                st.markdown(
                    f"**{r['recommendation']}** {r['option_type']} {r['strike']} | "
                    f"Entry ₹{r['ltp']} | SL ₹{r['stop_loss']} | "
                    f"T1 ₹{r['target_1']} | T2 ₹{r['target_2']} | POP {r['probability']}% | Risk {r['risk_level']}"
                )
                st.write("Logic/Reason:", ", ".join(r['entry_conditions']))

            # Premium vs Time chart
            st.plotly_chart(analyzer.plot_premium_vs_time(), use_container_width=True)

        else:
            st.error("Error parsing uploaded CSV.")
    else:
        st.info("Please upload CSV to start.")

if __name__ == "__main__":
    main()
