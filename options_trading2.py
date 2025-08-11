import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time
import io
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nifty Options Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OptionsAnalyzer:
    def __init__(self):
        self.current_time = datetime.now().time()
        self.market_hours = (time(9, 15), time(15, 30))

    def load_and_clean_data(self, uploaded_file):
        """Load and parse NSE-style option chain CSV"""
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')

            # Find strike price header line (like STRIKE or STRIKE PRICE)
            header_idx = None
            for i, line in enumerate(lines):
                if 'STRIKE' in line.upper():
                    header_idx = i
                    break

            if header_idx is None:
                st.error("No STRIKE column found — check file")
                return None, None

            csv_content = '\n'.join(lines[header_idx:])
            df = pd.read_csv(io.StringIO(csv_content))

            # Remove any completely empty columns
            df.dropna(axis=1, how='all', inplace=True)

            st.write(f"✅ CSV loaded with shape: {df.shape}")
            st.dataframe(df.head(3))

            # Ensure column names are stripped
            df.columns = df.columns.map(lambda x: str(x).strip())

            calls_df, puts_df = self.parse_options_data(df)
            return calls_df, puts_df

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

    def parse_options_data(self, df):
        """Parse NSE option chain where calls are left of strike, puts on right"""
        try:
            strike_index = None
            for i, col in enumerate(df.columns):
                if 'STRIKE' in col.upper():
                    strike_index = i
                    break
            if strike_index is None:
                st.error("Strike column not found in parsed data")
                return pd.DataFrame(), pd.DataFrame()

            calls_data = []
            puts_data = []

            for _, row in df.iterrows():
                try:
                    strike = self.safe_numeric(row.iloc[strike_index])
                    if strike == 0:
                        continue

                    # CALLS: columns before STRIKE
                    call_part = row.iloc[0: strike_index]
                    # PUTS: columns after STRIKE
                    put_part = row.iloc[strike_index+1:]

                    # Map CALL part — expecting structure matching NSE order
                    call_data = {
                        'strike': strike,
                        'oi': self.safe_numeric(call_part.iloc[0]),
                        'chng_oi': self.safe_numeric(call_part.iloc[1]),
                        'volume': self.safe_numeric(call_part.iloc[2]),
                        'iv': self.safe_numeric(call_part.iloc[3]),
                        'ltp': self.safe_numeric(call_part.iloc[4]),
                        'chng': self.safe_numeric(call_part.iloc[5])
                    }
                    if call_data['ltp'] > 0:
                        calls_data.append(call_data)

                    # Map PUT part — expecting matching structure
                    put_data = {
                        'strike': strike,
                        'bid_qty': self.safe_numeric(put_part.iloc[0]),
                        'bid': self.safe_numeric(put_part.iloc[1]),
                        'ask': self.safe_numeric(put_part.iloc[2]),
                        'ask_qty': self.safe_numeric(put_part.iloc[3]),
                        'chng': self.safe_numeric(put_part.iloc[4]),
                        'ltp': self.safe_numeric(put_part.iloc[5]),
                        'iv': self.safe_numeric(put_part.iloc[6]),
                        'volume': self.safe_numeric(put_part.iloc[7]),
                        'chng_oi': self.safe_numeric(put_part.iloc[8]),
                        'oi': self.safe_numeric(put_part.iloc[9])
                    }
                    if put_data['ltp'] > 0:
                        puts_data.append({
                            'strike': strike,
                            'oi': put_data['oi'],
                            'chng_oi': put_data['chng_oi'],
                            'volume': put_data['volume'],
                            'iv': put_data['iv'],
                            'ltp': put_data['ltp'],
                            'chng': put_data['chng']
                        })
                except:
                    continue

            calls_df = pd.DataFrame(calls_data)
            puts_df = pd.DataFrame(puts_data)

            st.write(f"✅ Parsed {len(calls_df)} call rows, {len(puts_df)} put rows")
            return calls_df, puts_df

        except Exception as e:
            st.error(f"Error parsing data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def safe_numeric(self, value):
        """Convert to float after removing commas, handle missing"""
        try:
            if pd.isna(value) or value in ['', '-']:
                return 0
            return float(str(value).replace(',', '').strip())
        except:
            return 0

    def get_market_phase(self):
        """Determine current market phase"""
        c = self.current_time
        if time(9, 15) <= c <= time(10, 0):
            return 'OPENING'
        elif time(14, 30) <= c <= time(15, 30):
            return 'CLOSING'
        else:
            return 'MID_SESSION'

    def create_visualizations(self, calls_df, puts_df, current_spot):
        """Charts for LTP, OI, Volume and PCR"""
        fig_ltp = make_subplots(rows=2, cols=1,
            subplot_titles=('Call LTP', 'Put LTP'), vertical_spacing=0.08)
        fig_ltp.add_trace(go.Scatter(x=calls_df['strike'], y=calls_df['ltp'],
            mode='lines+markers', name='Call LTP', line=dict(color='green')), row=1, col=1)
        fig_ltp.add_trace(go.Scatter(x=puts_df['strike'], y=puts_df['ltp'],
            mode='lines+markers', name='Put LTP', line=dict(color='red')), row=2, col=1)
        fig_ltp.add_vline(x=current_spot, line_dash="dash", line_color="blue")
        fig_ltp.update_layout(height=600, title="Options Premium Analysis")

        fig_oi = make_subplots(rows=2, cols=2,
            subplot_titles=('Call OI', 'Put OI', 'Call OI Change', 'Put OI Change'),
            vertical_spacing=0.1, horizontal_spacing=0.1)
        fig_oi.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['oi'], name='Call OI'), row=1, col=1)
        fig_oi.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['oi'], name='Put OI'), row=1, col=2)
        fig_oi.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['chng_oi'], name='Call OI Chg'), row=2, col=1)
        fig_oi.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['chng_oi'], name='Put OI Chg'), row=2, col=2)
        fig_oi.update_layout(height=700, title="Open Interest Analysis")

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=calls_df['strike'], y=calls_df['volume'], name='Call Vol'))
        fig_vol.add_trace(go.Bar(x=puts_df['strike'], y=puts_df['volume'], name='Put Vol'))
        fig_vol.add_vline(x=current_spot, line_dash="dash", line_color="blue")
        fig_vol.update_layout(title="Volume Analysis", height=400)

        pcr_oi = puts_df['oi'].sum() / calls_df['oi'].sum() if calls_df['oi'].sum() else 0
        pcr_vol = puts_df['volume'].sum() / calls_df['volume'].sum() if calls_df['volume'].sum() else 0
        return fig_ltp, fig_oi, fig_vol, pcr_oi, pcr_vol

# ---------------- MAIN APP ---------------- #
def main():
    st.title("🚀 Nifty Options Analysis")
    analyzer = OptionsAnalyzer()
    st.sidebar.header("📂 Upload Options Chain CSV")

    uploaded_file = st.sidebar.file_uploader("Upload file", type=['csv'], help="Upload the options chain CSV from NSE")
    current_spot = st.sidebar.number_input("Current Nifty Spot", min_value=20000, max_value=30000, value=24500, step=1)

    if uploaded_file:
        calls_df, puts_df = analyzer.load_and_clean_data(uploaded_file)
        if calls_df is not None and not calls_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Current Spot", f"{current_spot}")
            with col2: st.metric("Market Phase", analyzer.get_market_phase())
            with col3: st.metric("PCR OI", f"{(puts_df['oi'].sum() / calls_df['oi'].sum()):.2f}" if calls_df['oi'].sum() else "N/A")

            st.markdown("---")
            fig_ltp, fig_oi, fig_vol, pcr_oi, pcr_vol = analyzer.create_visualizations(calls_df, puts_df, current_spot)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("PCR (OI)", f"{pcr_oi:.2f}")
            mc2.metric("PCR (Vol)", f"{pcr_vol:.2f}")
            mc3.metric("Sentiment", "Bullish" if pcr_oi < 1 else "Bearish")

            st.plotly_chart(fig_ltp, use_container_width=True)
            st.plotly_chart(fig_oi, use_container_width=True)
            st.plotly_chart(fig_vol, use_container_width=True)

            st.markdown("### 📋 Data Tables")
            t1, t2 = st.tabs(["Calls", "Puts"])
            with t1: st.dataframe(calls_df)
            with t2: st.dataframe(puts_df)
        else:
            st.error("Error parsing uploaded file")
    else:
        st.info("Please upload the latest options chain CSV.")

if __name__ == "__main__":
    main()
