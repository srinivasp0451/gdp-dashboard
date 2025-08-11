import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import cv2
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Nifty Options Chain Dashboard", layout="wide")

if "premium_history" not in st.session_state:
    st.session_state.premium_history = {"time": [], "calls": {}, "puts": {}}

class OptionsAnalyzer:
    def load_and_clean_data(self, uploaded_file):
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        lines = content.strip().split('\n')
        # Find the header row containing 'STRIKE'
        header_idx = next((i for i,l in enumerate(lines) if 'STRIKE' in l.upper()), None)
        if header_idx is None:
            st.error("No STRIKE column found.")
            return None, None
        csv_content = '\n'.join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_content))
        df.columns = [str(c).strip().upper() for c in df.columns]
        # Split into Calls and Puts by position of STRIKE column
        strike_col = 'STRIKE'
        strike_idx = df.columns.get_loc(strike_col)
        calls_block = df.iloc[:, :strike_idx]
        puts_block = df.iloc[:, strike_idx+1:]
        strikes = df[strike_col].apply(self.safe_num)
        
        # Map Calls
        calls_df = pd.DataFrame({
            "strike": strikes,
            "oi": calls_block["OI"].apply(self.safe_num),
            "chng_oi": calls_block["CHNG IN OI"].apply(self.safe_num),
            "volume": calls_block["VOLUME"].apply(self.safe_num),
            "iv": calls_block["IV"].apply(self.safe_num),
            "ltp": calls_block["LTP"].apply(self.safe_num),
            "chng": calls_block["CHNG"].apply(self.safe_num)
        })
        # Map Puts
        puts_df = pd.DataFrame({
            "strike": strikes,
            "oi": puts_block["OI"].apply(self.safe_num),
            "chng_oi": puts_block["CHNG IN OI"].apply(self.safe_num),
            "volume": puts_block["VOLUME"].apply(self.safe_num),
            "iv": puts_block["IV"].apply(self.safe_num),
            "ltp": puts_block["LTP"].apply(self.safe_num),
            "chng": puts_block["CHNG"].apply(self.safe_num)
        })
        # Drop rows where no strike
        calls_df = calls_df[calls_df["strike"]>0]
        puts_df = puts_df[puts_df["strike"]>0]
        return calls_df, puts_df

    def safe_num(self, val):
        try:
            if pd.isna(val) or str(val).strip() in ['', '-', '--']:
                return 0
            return float(str(val).replace(',', '').strip())
        except:
            return 0

    def autodetect_spot(self, calls_df, puts_df):
        merged = pd.merge(calls_df[['strike','ltp']], puts_df[['strike','ltp']], on='strike', suffixes=('_call','_put'))
        merged['total_premium'] = merged['ltp_call'] + merged['ltp_put']
        return merged.loc[merged['total_premium'].idxmin(), 'strike']

    def update_premium_history(self, calls_df, puts_df):
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.premium_history["time"].append(timestamp)
        for df, side in [(calls_df,"calls"), (puts_df,"puts")]:
            for _, r in df.iterrows():
                strike = str(int(r['strike']))
                st.session_state.premium_history[side].setdefault(strike, []).append(r['ltp'])

    def plot_premium_vs_time(self):
        fig = go.Figure()
        for strike, lst in st.session_state.premium_history["calls"].items():
            fig.add_trace(go.Scatter(x=st.session_state.premium_history["time"], y=lst, mode='lines+markers', name=f"CALL {strike}"))
        for strike, lst in st.session_state.premium_history["puts"].items():
            fig.add_trace(go.Scatter(x=st.session_state.premium_history["time"], y=lst, mode='lines+markers', name=f"PUT {strike}"))
        fig.update_layout(title="Premium vs Time", xaxis_title="Time", yaxis_title="LTP")
        return fig

    def create_bar(self, calls_df, puts_df, col, title):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=calls_df['strike'], y=calls_df[col], name=f"Call {col}", marker_color='red'))
        fig.add_trace(go.Bar(x=puts_df['strike'], y=puts_df[col], name=f"Put {col}", marker_color='green'))
        fig.update_layout(barmode='group', title=title)
        return fig

    def analyze_signal(self, row, spot, option_type, max_oi, max_vol):
        strike, ltp, volume, oi, chng_oi, iv = row['strike'], row['ltp'], row['volume'], row['oi'], row['chng_oi'], row['iv']
        moneyness = (spot-strike)/spot*100 if option_type=='CALL' else (strike-spot)/spot*100
        base_pop = 50 - abs(moneyness)*2
        oi_factor = (oi/max_oi)*20 if max_oi else 0
        vol_factor = min(10, (volume/max_vol)*10) if max_vol else 0
        chg_factor = 5 if chng_oi>0 else -5 if chng_oi<0 else 0
        pop = max(5, min(95, base_pop + oi_factor + vol_factor + chg_factor))

        entry_cond=[]
        if volume>100000: entry_cond.append("High Volume")
        if oi>100000: entry_cond.append("OI Build-up")
        if 10<=iv<=20: entry_cond.append("Optimal IV")
        if chng_oi>5000: entry_cond.append("Positive OI Change" if option_type=='CALL' else "Negative OI Change")

        risk = 'LOW' if abs(moneyness) <1 else 'MEDIUM' if abs(moneyness) <3 else 'HIGH'
        t1, t2 = round(ltp*1.5,2), round(ltp*2,2)
        sl = round(ltp*0.7,2)
        score = ( min(100,(volume/500000)*100) + min(100,(oi/200000)*100) + max(0,100-abs(iv-15)*5) ) / 3
        if score>75 and len(entry_cond)>=2: rec="STRONG BUY"
        elif score>60 and len(entry_cond)>=2: rec="BUY"
        elif score>50 and len(entry_cond)>=1: rec="WEAK BUY"
        else: rec="AVOID"
        return {"option_type":option_type,"strike":strike,"ltp":ltp,"stop_loss":sl,
                "target_1":t1,"target_2":t2,"probability":round(pop,1),"recommendation":rec,
                "score":score,"risk_level":risk,"entry_conditions":entry_cond}

    def get_recommendations(self, calls_df, puts_df, spot):
        max_oi_call, max_vol_call = calls_df['oi'].max(), calls_df['volume'].max()
        max_oi_put, max_vol_put = puts_df['oi'].max(), puts_df['volume'].max()
        recs = [self.analyze_signal(r, spot, 'CALL', max_oi_call, max_vol_call) for _,r in calls_df.iterrows()]
        recs += [self.analyze_signal(r, spot, 'PUT', max_oi_put, max_vol_put) for _,r in puts_df.iterrows()]
        return sorted(recs,key=lambda x:-x['score'])[:10]

    def summarize_chart(self, img):
        arr = np.array(img.convert("L"))
        slope = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=5).mean()
        bright = arr.mean()
        if slope>0 and bright>100: trend="Uptrend Bias"
        elif slope<0 and bright<100: trend="Downtrend Bias"
        else: trend="Sideways/Neutral"
        return f"Detected Chart Trend: {trend}"

# ------------ Streamlit App ------------
def main():
    st.title("📈 Nifty Option Chain Dashboard")
    analyzer = OptionsAnalyzer()

    upfile = st.sidebar.file_uploader("Upload Option Chain CSV", type=['csv'])
    chartfile = st.sidebar.file_uploader("Upload Index Chart Screenshot (optional)", type=['png','jpg','jpeg'])
    manual_spot = st.sidebar.number_input("Spot Override", 20000, 30000, 0)

    if upfile:
        calls_df, puts_df = analyzer.load_and_clean_data(upfile)
        if calls_df is not None and not calls_df.empty:
            spot = manual_spot if manual_spot>0 else analyzer.autodetect_spot(calls_df, puts_df)
            analyzer.update_premium_history(calls_df, puts_df)

            # Summary metrics
            st.metric("Current Spot", f"{spot}")
            total_call_oi, total_put_oi = calls_df['oi'].sum(), puts_df['oi'].sum()
            pcr_oi = total_put_oi/total_call_oi if total_call_oi else 0
            sentiment = "Bullish" if pcr_oi>1 else "Bearish" if pcr_oi<1 else "Neutral"
            st.write(f"PCR(OI): {pcr_oi:.2f} → {sentiment}")
            st.write(f"Max CE OI @ {calls_df.loc[calls_df['oi'].idxmax(), 'strike']}")
            st.write(f"Max PE OI @ {puts_df.loc[puts_df['oi'].idxmax(), 'strike']}")

            # Chart summary
            if chartfile:
                img = Image.open(chartfile)
                st.image(img, caption="Index Chart")
                st.write(analyzer.summarize_chart(img))

            # Recs
            st.subheader("Recommendations")
            recs = analyzer.get_recommendations(calls_df, puts_df, spot)
            for r in recs:
                st.markdown(f"**{r['recommendation']}** {r['option_type']} {r['strike']} | Entry ₹{r['ltp']} | SL ₹{r['stop_loss']} | T1 ₹{r['target_1']} | T2 ₹{r['target_2']} | POP {r['probability']}% | Risk {r['risk_level']}")
                st.write(f"Reason: {', '.join(r['entry_conditions'])}")

            # Charts
            st.plotly_chart(analyzer.plot_premium_vs_time(), use_container_width=True)
            st.plotly_chart(analyzer.create_bar(calls_df, puts_df, 'oi', "Open Interest"), use_container_width=True)
            st.plotly_chart(analyzer.create_bar(calls_df, puts_df, 'chng_oi', "Change in OI"), use_container_width=True)
            st.plotly_chart(analyzer.create_bar(calls_df, puts_df, 'volume', "Volume"), use_container_width=True)
        else:
            st.error("No valid rows parsed from CSV.")
    else:
        st.info("Upload the option chain CSV to start.")

if __name__=="__main__":
    main()
