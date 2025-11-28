import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Psychology-Enhanced AI Trader")

# NLTK Setup
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- 1. ASSET MANAGER 🌐 ---
class AssetManager:
    def __init__(self, ticker):
        self.ticker = ticker
        self.asset_class = self._detect_asset_class()

    def _detect_asset_class(self):
        if "-USD" in self.ticker or "BTC" in self.ticker: return "CRYPTO"
        elif "=X" in self.ticker: return "FOREX"
        elif ".NS" in self.ticker: return "INDIAN_EQUITY"
        else: return "US_EQUITY"

    def get_timeframes(self, style):
        if style == "Scalper": return ("1h", "15m", "5m")
        elif style == "Day Trader": return ("1d", "1h", "15m")
        else: return ("1wk", "1d", "4h")

# --- 2. PSYCHOLOGY AGENT (THE HUMAN ELEMENT) 🧠 ---
class PsychologyAgent:
    def __init__(self, df):
        self.df = df

    def analyze_smart_money(self):
        """Detects Institutional Activity via Volume & Wicks."""
        if self.df is None or self.df.empty: return {"status":
"UNKNOWN", "score": 0}

        last = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        # 1. Volume Anomaly (Smart Money Presence)
        # Is current volume 1.5x larger than average?
        vol_spike = last['Volume'] > (last['Vol_SMA'] * 1.5)

        # 2. Wick Analysis (Rejection Psychology)
        # Upper Wick = High - Max(Open, Close)
        body_size = abs(last['Close'] - last['Open'])
        upper_wick = last['High'] - max(last['Close'], last['Open'])
        lower_wick = min(last['Close'], last['Open']) - last['Low']

        # 3. Trap Detection
        psych_score = 0
        status = "Normal Activity"

        # BULL TRAP: Price Up, Low Volume, Big Upper Wick (Rejection)
        if upper_wick > (body_size * 2) and not vol_spike:
            status = "🚨 POSSIBLE BULL TRAP (Rejection)"
            psych_score = -2 # Strong Sell psychology

        # BEAR TRAP: Price Down, Low Volume, Big Lower Wick (Absorption)
        elif lower_wick > (body_size * 2) and not vol_spike:
            status = "🚨 POSSIBLE BEAR TRAP (Absorption)"
            psych_score = 2 # Strong Buy psychology

        # INSTITUTIONAL BUYING: Big Green Candle + Huge Volume
        elif last['Close'] > last['Open'] and vol_spike:
            status = "🐋 SMART MONEY BUYING (High Conviction)"
            psych_score = 3

        # INSTITUTIONAL SELLING: Big Red Candle + Huge Volume
        elif last['Close'] < last['Open'] and vol_spike:
            status = "🐋 SMART MONEY SELLING (High Conviction)"
            psych_score = -3

        return {"status": status, "score": psych_score, "vol_spike": vol_spike}

    def get_fear_greed(self, rsi):
        """Contrarian Logic: Be Fearful when others are Greedy."""
        if rsi > 80: return "EXTREME GREED (Euphoria)", -2 # Look to Short
        if rsi > 70: return "GREED", -1
        if rsi < 20: return "EXTREME FEAR (Capitulation)", 2 # Look to Buy
        if rsi < 30: return "FEAR", 1
        return "NEUTRAL", 0

# --- 3. TECHNICAL AGENT 🤖 ---
class TechnicalAgent:
    def __init__(self, ticker, timeframes):
        self.ticker = ticker
        self.timeframes = timeframes
        self.data = {}

    def fetch_data(self):
        status = []
        for tf in self.timeframes:
            # Period mapping
            period = "1mo" if tf in ["5m","15m"] else "2y"
            try:
                df = yf.download(self.ticker, period=period,
interval=tf, progress=False, auto_adjust=True)
                if df.empty: continue

                # Cleanup
                df = df.reset_index()
                df.columns = [str(c[0]).upper() if isinstance(c,
tuple) else str(c).upper() for c in df.columns]
                map_cols = {'DATETIME':'Date','DATE':'Date','CLOSE':'Close','OPEN':'Open','HIGH':'High','LOW':'Low','VOLUME':'Volume'}
                df = df.rename(columns={k:v for k,v in
map_cols.items() if k in df.columns})

                # Indicators
                df['EMA_50'] = df['Close'].ewm(span=50).mean()
                df['RSI'] = 100 - (100 / (1 +
(df['Close'].diff().clip(lower=0).ewm(span=14).mean() /
df['Close'].diff().clip(upper=0).abs().ewm(span=14).mean())))
                df['Vol_SMA'] = df['Volume'].rolling(20).mean()
                df['ATR'] = (df['High'] -
df['Low']).rolling(14).mean() # Simplified ATR

                self.data[tf] = df
                status.append(f"✅ {tf}")
            except Exception as e:
                status.append(f"❌ {tf}: {e}")
        return status

# --- 4. NEWS AGENT 📰 ---
class NewsAgent:
    def __init__(self, ticker):
        self.ticker = ticker
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self):
        try:
            news = yf.Ticker(self.ticker).news
            # st.write('News is :', news)
            if not news: return {"score": 0, "summary": "No News"}

            score = 0
            for n in news[:5]:
                st.write('n is:::',n['content']['title'])
                score += self.vader.polarity_scores(n['content']['title'])['compound']

            avg_score = score / 5
            sentiment = "NEUTRAL"
            if avg_score > 0.15: sentiment = "POSITIVE (News Catalyst)"
            if avg_score < -0.15: sentiment = "NEGATIVE (Bad Press)"

            return {"score": avg_score, "summary": sentiment}
        except:
            return {"score": 0, "summary": "Error"}

# --- 5. EXECUTION BOSS (PROFESSIONAL LOGIC) ⚖️ ---
class ProfessionalExecutionAgent:
    def __init__(self, tech_agent, news_data, psych_agent_entry):
        self.tech = tech_agent
        self.news = news_data
        self.psych = psych_agent_entry # Psychology of the Entry timeframe

    def decide(self):
        t1, t2, t3 = self.tech.timeframes

        # 1. Trend Analysis (HTF)
        df_trend = self.tech.data.get(t1)
        trend = "NEUTRAL"
        if df_trend is not None:
            last = df_trend.iloc[-1]
            trend = "BULLISH" if last['Close'] > last['EMA_50'] else "BEARISH"

        # 2. Entry Analysis (LTF)
        df_entry = self.tech.data.get(t3)
        if df_entry is None: return {"action": "ERROR"}

        entry_last = df_entry.iloc[-1]

        # 3. Psychology & Context
        psych_report = self.psych.analyze_smart_money()
        fg_label, fg_score = self.psych.get_fear_greed(entry_last['RSI'])
        news_score = self.news['score']

        # --- PROFESSIONAL DECISION MATRIX ---
        score = 0
        reasons = []

        # Trend Alignment (+1)
        if trend == "BULLISH": score += 1; reasons.append(f"Major Trend ({t1}) is Up")
        if trend == "BEARISH": score -= 1; reasons.append(f"Major Trend ({t1}) is Down")

        # Psychology Score (+/- 2 or 3) -> High Weight
        score += psych_report['score']
        if psych_report['score'] != 0: reasons.append(psych_report['status'])

        # Fear/Greed Contrarian Logic
        # Buying Fear in an Uptrend is a Pro move. Buying Greed is a Noob move.
        if trend == "BULLISH" and "FEAR" in fg_label:
            score += 2
            reasons.append("Contrarian Buy: Buying the dip (Fear) in Uptrend.")
        elif trend == "BULLISH" and "GREED" in fg_label:
            score -= 2
            reasons.append("Caution: Market is Euphoric/Overextended.")

        # News Filter
        if news_score < -0.2: score -= 2; reasons.append("News Sentiment is Very Negative.")
        if news_score > 0.2: score += 1; reasons.append("News Sentiment is Positive.")

        # --- FINAL ACTION ---
        action = "WAIT"
        confidence = "Low"

        if score >= 3:
            action = "STRONG BUY"
            confidence = "High (Institutional Footprint)"
        elif score >= 1.5:
            action = "BUY"
            confidence = "Medium"
        elif score <= -3:
            action = "STRONG SELL"
            confidence = "High (Institutional Distribution)"
        elif score <= -1.5:
            action = "SELL"
            confidence = "Medium"
        else:
            action = "HOLD / SIT OUT"
            reasons.append("No clear edge found.")

        # Levels
        atr = entry_last['ATR']
        if np.isnan(atr): atr = entry_last['Close'] * 0.01

        # Pro Risk Management: Tighter stops if entering against trend (Counter-trend)
        sl_mult = 1.5 if (trend == "BULLISH" and "BUY" in action) else 1.0

        entry_p = entry_last['Close']
        sl = entry_p - (atr * sl_mult) if "BUY" in action else entry_p + (atr * sl_mult)
        tp = entry_p + (atr * sl_mult * 2) if "BUY" in action else entry_p - (atr * sl_mult * 2)

        return {
            "action": action,
            "conf": confidence,
            "reason": reasons,
            "entry": entry_p, "sl": sl, "tp": tp,
            "psych_status": psych_report['status'],
            "fg": fg_label
        }

# --- UI LAYER ---
def main():
    st.title("🧠 Psychology-Enhanced Pro Trader")
    st.markdown("### Tracks Smart Money, Traps, and Sentiment")

    col1, col2 = st.columns(2)
    ticker = col1.text_input("Ticker", "BTC-USD")
    style = col2.selectbox("Style", ["Scalper", "Day Trader", "Swing Trader"], index=1)

    if st.button("🧠 Analyze Like a Pro"):
        am = AssetManager(ticker)
        tfs = am.get_timeframes(style)

        # 1. Tech Analysis
        tech = TechnicalAgent(ticker, tfs)
        with st.spinner("Analyzing Market Structure..."):
            tech.fetch_data()

        # 2. News Analysis
        news = NewsAgent(ticker)
        with st.spinner("Gauging Sentiment..."):
            news_data = news.analyze()

        # 3. Psych Analysis (On Entry TF)
        if tfs[2] in tech.data:
            entry_df = tech.data[tfs[2]]
            psych = PsychologyAgent(entry_df)

            # 4. Boss Decision
            boss = ProfessionalExecutionAgent(tech, news_data, psych)
            res = boss.decide()

            # --- DISPLAY ---

            # SCORECARD
            st.divider()
            c1, c2, c3 = st.columns(3)

            # Sentiment Card
            c1.markdown("### 📰 Sentiment")
            s_col = "green" if news_data['score'] > 0 else "red"
            c1.markdown(f":{s_col}[{news_data['summary']}]")

            # Psychology Card
            c2.markdown("### 🧠 Psychology")
            p_col = "orange"
            if "GREED" in res['fg']: p_col = "red"
            if "FEAR" in res['fg']: p_col = "green"
            c2.markdown(f"Crowd: :{p_col}[{res['fg']}]")
            c2.caption(f"Smart Money: {res['psych_status']}")

            # Decision Card
            c3.markdown("### 🤖 Decision")
            d_col = "gray"
            if "BUY" in res['action']: d_col = "green"
            if "SELL" in res['action']: d_col = "red"
            c3.markdown(f"## :{d_col}[{res['action']}]")
            c3.caption(f"Confidence: {res['conf']}")

            st.info(f"**Logic:** {' | '.join(res['reason'])}")

            # Trade Plan
            if "BUY" in res['action'] or "SELL" in res['action']:
                st.subheader("🎯 Professional Trade Plan")
                mp1, mp2, mp3 = st.columns(3)
                mp1.metric("Entry", f"{res['entry']:.2f}")
                mp2.metric("Stop Loss", f"{res['sl']:.2f}")
                mp3.metric("Take Profit (2R)", f"{res['tp']:.2f}")

            # Chart with Traps
            st.subheader(f"Price Action Analysis ({tfs[2]})")
            fig = go.Figure(data=[go.Candlestick(x=entry_df['Date'], open=entry_df['Open'], high=entry_df['High'], low=entry_df['Low'], close=entry_df['Close'])])
            fig.update_layout(height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Data fetch failed. Try a different ticker.")

if __name__ == "__main__":
    main()
