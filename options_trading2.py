import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, time
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
        """Load and clean options chain data with error handling"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Try different reading methods
            df = None
            
            # Method 1: Standard CSV reading
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                st.write(f"✅ CSV loaded with shape: {df.shape}")
            except:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                    st.write(f"✅ CSV loaded with latin1 encoding, shape: {df.shape}")
                except:
                    uploaded_file.seek(0)
                    # Method 2: Try with different separator
                    try:
                        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
                        st.write(f"✅ CSV loaded with semicolon separator, shape: {df.shape}")
                    except:
                        uploaded_file.seek(0)
                        # Method 3: Try reading as text first
                        content = uploaded_file.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        
                        # Find the header line and data
                        header_idx = 0
                        for i, line in enumerate(lines):
                            if 'STRIKE' in line.upper() or 'CALLS' in line.upper():
                                header_idx = i
                                break
                        
                        # Create DataFrame from lines
                        import io
                        csv_content = '\n'.join(lines[header_idx:])
                        df = pd.read_csv(io.StringIO(csv_content))
                        st.write(f"✅ CSV loaded from text parsing, shape: {df.shape}")
            
            if df is None or df.empty:
                st.error("Could not read the CSV file. Please check the format.")
                return None, None
            
            # Display first few rows for debugging
            st.write("**First 3 rows of data:**")
            st.dataframe(df.head(3))
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Display column names for debugging
            st.write(f"**Columns found:** {list(df.columns)}")
            
            # Parse the data based on structure
            calls_data, puts_data = self.parse_options_data(df)
            
            return calls_data, puts_data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.write("**Debug info:**")
            st.write(f"File type: {type(uploaded_file)}")
            st.write(f"File name: {uploaded_file.name}")
            return None, None
    
    def parse_options_data(self, df):
        """Parse options data from different CSV formats"""
        calls_data = []
        puts_data = []
        
        try:
            st.write("**Starting data parsing...**")
            
            # Find the strike column (should contain values like 22600, 22650, etc.)
            strike_col_idx = None
            for i, col in enumerate(df.columns):
                try:
                    # Convert column to numeric and check for strike-like values
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(values) > 0:
                        min_val, max_val = values.min(), values.max()
                        if 20000 <= min_val <= 30000 and 20000 <= max_val <= 30000:
                            strike_col_idx = i
                            st.write(f"✅ Strike column found at index {i}: {col}")
                            break
                except:
                    continue
            
            if strike_col_idx is None:
                # Try to find column with name containing 'STRIKE'
                for i, col in enumerate(df.columns):
                    if 'STRIKE' in str(col).upper():
                        strike_col_idx = i
                        st.write(f"✅ Strike column found by name at index {i}: {col}")
                        break
            
            if strike_col_idx is None:
                st.error("Could not identify strike price column")
                return pd.DataFrame(), pd.DataFrame()
            
            # Based on your CSV format, the structure appears to be:
            # Columns 0-10: CALLS data, Column 11: STRIKE, Columns 12-22: PUTS data
            
            total_cols = len(df.columns)
            st.write(f"Total columns: {total_cols}, Strike column at: {strike_col_idx}")
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    # Get strike price
                    strike = pd.to_numeric(row.iloc[strike_col_idx], errors='coerce')
                    if pd.isna(strike) or strike == 0:
                        continue
                    
                    # Parse CALLS data (left side of strike)
                    if strike_col_idx > 0:
                        try:
                            # Typical order: OI, CHNG_OI, VOLUME, IV, LTP, CHNG, BID_QTY, BID, ASK, ASK_QTY
                            call_oi = self.safe_numeric(row.iloc[0])
                            call_chng_oi = self.safe_numeric(row.iloc[1])  
                            call_volume = self.safe_numeric(row.iloc[2])
                            call_iv = self.safe_numeric(row.iloc[3])
                            call_ltp = self.safe_numeric(row.iloc[4])
                            call_chng = self.safe_numeric(row.iloc[5]) if strike_col_idx > 5 else 0
                            
                            if call_ltp > 0:  # Only add if LTP exists
                                calls_data.append({
                                    'strike': strike,
                                    'oi': call_oi,
                                    'chng_oi': call_chng_oi,
                                    'volume': call_volume,
                                    'iv': call_iv,
                                    'ltp': call_ltp,
                                    'chng': call_chng
                                })
                        except Exception as e:
                            st.write(f"Error parsing calls for strike {strike}: {e}")
                    
                    # Parse PUTS data (right side of strike)
                    if strike_col_idx < total_cols - 1:
                        try:
                            # PUTS data starts after strike column
                            puts_start = strike_col_idx + 1
                            
                            # Try to extract put data
                            if puts_start + 5 < total_cols:
                                put_oi = self.safe_numeric(row.iloc[puts_start + 10]) if puts_start + 10 < total_cols else self.safe_numeric(row.iloc[-1])
                                put_chng_oi = self.safe_numeric(row.iloc[puts_start + 9]) if puts_start + 9 < total_cols else self.safe_numeric(row.iloc[-2])
                                put_volume = self.safe_numeric(row.iloc[puts_start + 8]) if puts_start + 8 < total_cols else self.safe_numeric(row.iloc[-3])
                                put_iv = self.safe_numeric(row.iloc[puts_start + 7]) if puts_start + 7 < total_cols else self.safe_numeric(row.iloc[-4])
                                put_ltp = self.safe_numeric(row.iloc[puts_start + 6]) if puts_start + 6 < total_cols else self.safe_numeric(row.iloc[-5])
                                put_chng = self.safe_numeric(row.iloc[puts_start + 5]) if puts_start + 5 < total_cols else self.safe_numeric(row.iloc[-6])
                                
                                if put_ltp > 0:  # Only add if LTP exists
                                    puts_data.append({
                                        'strike': strike,
                                        'oi': put_oi,
                                        'chng_oi': put_chng_oi,
                                        'volume': put_volume,
                                        'iv': put_iv,
                                        'ltp': put_ltp,
                                        'chng': put_chng
                                    })
                        except Exception as e:
                            st.write(f"Error parsing puts for strike {strike}: {e}")
                            
                except Exception as e:
                    continue
            
            # Convert to DataFrames
            calls_df = pd.DataFrame(calls_data)
            puts_df = pd.DataFrame(puts_data)
            
            st.write(f"✅ Parsed {len(calls_df)} call options and {len(puts_df)} put options")
            
            # Display sample data
            if not calls_df.empty:
                st.write("**Sample Call Options:**")
                st.dataframe(calls_df.head(3))
            
            if not puts_df.empty:
                st.write("**Sample Put Options:**")
                st.dataframe(puts_df.head(3))
            
            return calls_df, puts_df
            
        except Exception as e:
            st.error(f"Error parsing options data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def safe_numeric(self, value):
        """Safely convert value to numeric, handling commas and invalid data"""
        try:
            if pd.isna(value) or value == '' or value == '-':
                return 0
            # Remove commas and convert to float
            clean_value = str(value).replace(',', '').replace('"', '').strip()
            return float(clean_value) if clean_value != '' else 0
        except:
            return 0
    
    def calculate_probabilities(self, df, current_spot, option_type='call'):
        """Calculate probability of profit using Black-Scholes approximation"""
        probabilities = []
        
        for _, row in df.iterrows():
            try:
                strike = row['strike']
                iv = row['iv'] / 100 if row['iv'] > 1 else row['iv']
                
                # Simple probability calculation based on moneyness and IV
                if option_type == 'call':
                    moneyness = current_spot / strike
                    prob = max(0, min(100, 100 * (1 - np.exp(-2 * max(0, moneyness - 1) / (iv + 0.01)))))
                else:
                    moneyness = strike / current_spot
                    prob = max(0, min(100, 100 * (1 - np.exp(-2 * max(0, moneyness - 1) / (iv + 0.01)))))
                
                probabilities.append(prob)
            except:
                probabilities.append(0)
        
        return probabilities
    
    def get_entry_exit_signals(self, calls_df, puts_df, current_spot):
        """Generate precise entry, target, and stop-loss recommendations"""
        recommendations = []
        
        # Market time check
        is_market_open = self.market_hours[0] <= self.current_time <= self.market_hours[1]
        market_phase = self.get_market_phase()
        
        # ATM and nearby strikes analysis
        atm_strike = round(current_spot / 50) * 50
        
        # Analyze calls
        for _, row in calls_df.iterrows():
            if abs(row['strike'] - current_spot) <= 300:  # Within 300 points
                signal = self.analyze_option_signal(row, current_spot, 'CALL', market_phase)
                if signal['recommendation'] != 'AVOID':
                    recommendations.append(signal)
        
        # Analyze puts
        for _, row in puts_df.iterrows():
            if abs(row['strike'] - current_spot) <= 300:  # Within 300 points
                signal = self.analyze_option_signal(row, current_spot, 'PUT', market_phase)
                if signal['recommendation'] != 'AVOID':
                    recommendations.append(signal)
        
        # Sort by score (higher is better)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def analyze_option_signal(self, row, current_spot, option_type, market_phase):
        """Detailed analysis for individual option"""
        strike = row['strike']
        ltp = row['ltp']
        volume = row['volume']
        oi = row['oi']
        chng_oi = row.get('chng_oi', 0)
        iv = row['iv']
        
        # Calculate moneyness
        if option_type == 'CALL':
            moneyness = (current_spot - strike) / current_spot * 100
        else:
            moneyness = (strike - current_spot) / current_spot * 100
        
        # Scoring factors
        volume_score = min(100, (volume / 100000) * 20)  # Volume in lakhs
        oi_score = min(100, (oi / 50000) * 20)
        iv_score = max(0, 100 - abs(iv - 15) * 5)  # Optimal IV around 15%
        
        # OI change analysis
        oi_signal = 'NEUTRAL'
        if chng_oi > 10000:
            oi_signal = 'BULLISH' if option_type == 'CALL' else 'BEARISH'
        elif chng_oi < -10000:
            oi_signal = 'BEARISH' if option_type == 'CALL' else 'BULLISH'
        
        # Entry conditions
        entry_conditions = []
        
        # Volume condition
        if volume > 50000:
            entry_conditions.append("High Volume")
            
        # OI condition
        if oi > 25000:
            entry_conditions.append("Good OI")
            
        # IV condition
        if 10 <= iv <= 20:
            entry_conditions.append("Optimal IV")
            
        # Time-based conditions
        if market_phase == 'OPENING':
            if volume > 100000:
                entry_conditions.append("Opening Momentum")
        elif market_phase == 'CLOSING':
            if abs(moneyness) < 2:  # Near ATM
                entry_conditions.append("Closing Play")
        
        # Calculate targets and stop loss
        if option_type == 'CALL':
            if moneyness > 2:  # ITM
                target_1 = ltp * 1.3
                target_2 = ltp * 1.6
                stop_loss = ltp * 0.75
            elif abs(moneyness) <= 2:  # ATM
                target_1 = ltp * 1.5
                target_2 = ltp * 2.0
                stop_loss = ltp * 0.7
            else:  # OTM
                target_1 = ltp * 1.8
                target_2 = ltp * 3.0
                stop_loss = ltp * 0.6
        else:  # PUT
            if moneyness > 2:  # ITM
                target_1 = ltp * 1.3
                target_2 = ltp * 1.6
                stop_loss = ltp * 0.75
            elif abs(moneyness) <= 2:  # ATM
                target_1 = ltp * 1.5
                target_2 = ltp * 2.0
                stop_loss = ltp * 0.7
            else:  # OTM
                target_1 = ltp * 1.8
                target_2 = ltp * 3.0
                stop_loss = ltp * 0.6
        
        # Overall score
        total_score = (volume_score + oi_score + iv_score) / 3
        
        # Recommendation logic
        if len(entry_conditions) >= 2 and total_score > 50:
            if total_score > 80:
                recommendation = 'STRONG BUY'
            elif total_score > 65:
                recommendation = 'BUY'
            else:
                recommendation = 'WEAK BUY'
        else:
            recommendation = 'AVOID'
        
        # Calculate probability
        if option_type == 'CALL':
            probability = max(5, min(95, 50 + moneyness * 2 - (abs(moneyness) * 0.5)))
        else:
            probability = max(5, min(95, 50 - moneyness * 2 - (abs(moneyness) * 0.5)))
        
        return {
            'option_type': option_type,
            'strike': strike,
            'ltp': ltp,
            'volume': volume,
            'oi': oi,
            'iv': iv,
            'moneyness': moneyness,
            'recommendation': recommendation,
            'probability': round(probability, 1),
            'target_1': round(target_1, 2),
            'target_2': round(target_2, 2),
            'stop_loss': round(stop_loss, 2),
            'entry_conditions': entry_conditions,
            'oi_signal': oi_signal,
            'score': total_score,
            'risk_level': 'LOW' if abs(moneyness) < 1 else 'MEDIUM' if abs(moneyness) < 3 else 'HIGH'
        }
    
    def create_sample_data(self):
        """Create sample data based on the uploaded document"""
        # Sample data extracted from your CSV
        calls_data = [
            {'strike': 24000, 'oi': 7773, 'chng_oi': -490, 'volume': 28334, 'iv': 14.89, 'ltp': 551.80, 'chng': 124.75},
            {'strike': 24050, 'oi': 316, 'chng_oi': 19, 'volume': 2296, 'iv': 12.71, 'ltp': 496.00, 'chng': 111.45},
            {'strike': 24100, 'oi': 1764, 'chng_oi': 331, 'volume': 17876, 'iv': 13.76, 'ltp': 453.90, 'chng': 111.95},
            {'strike': 24150, 'oi': 930, 'chng_oi': 194, 'volume': 15757, 'iv': 13.65, 'ltp': 408.15, 'chng': 110.85},
            {'strike': 24200, 'oi': 7908, 'chng_oi': 1478, 'volume': 127077, 'iv': 13.40, 'ltp': 362.65, 'chng': 100.05},
            {'strike': 24250, 'oi': 2356, 'chng_oi': 410, 'volume': 90086, 'iv': 13.02, 'ltp': 318.35, 'chng': 92.45},
            {'strike': 24300, 'oi': 20186, 'chng_oi': 2532, 'volume': 563836, 'iv': 12.92, 'ltp': 275.65, 'chng': 82.60},
            {'strike': 24350, 'oi': 15825, 'chng_oi': 1565, 'volume': 741989, 'iv': 12.65, 'ltp': 235.25, 'chng': 73.75},
            {'strike': 24400, 'oi': 66342, 'chng_oi': 7589, 'volume': 2570834, 'iv': 12.42, 'ltp': 197.45, 'chng': 60.95},
            {'strike': 24450, 'oi': 58308, 'chng_oi': 12395, 'volume': 2119322, 'iv': 12.23, 'ltp': 163.00, 'chng': 50.70},
            {'strike': 24500, 'oi': 179988, 'chng_oi': 32574, 'volume': 3231326, 'iv': 12.09, 'ltp': 131.45, 'chng': 40.80},
            {'strike': 24550, 'oi': 80136, 'chng_oi': 29150, 'volume': 1452173, 'iv': 12.09, 'ltp': 104.15, 'chng': 30.80},
            {'strike': 24600, 'oi': 141779, 'chng_oi': 2461, 'volume': 1952996, 'iv': 12.06, 'ltp': 82.20, 'chng': 23.35},
            {'strike': 24650, 'oi': 55622, 'chng_oi': 8648, 'volume': 964499, 'iv': 12.17, 'ltp': 63.00, 'chng': 16.10},
            {'strike': 24700, 'oi': 107374, 'chng_oi': 2382, 'volume': 1293020, 'iv': 12.22, 'ltp': 48.15, 'chng': 11.10},
            {'strike': 24750, 'oi': 48415, 'chng_oi': 9905, 'volume': 746369, 'iv': 12.30, 'ltp': 35.95, 'chng': 6.65},
            {'strike': 24800, 'oi': 107819, 'chng_oi': 2985, 'volume': 1081317, 'iv': 12.43, 'ltp': 26.85, 'chng': 3.85},
            {'strike': 24850, 'oi': 39820, 'chng_oi': 5508, 'volume': 537632, 'iv': 12.61, 'ltp': 20.00, 'chng': 1.80},
            {'strike': 24900, 'oi': 84509, 'chng_oi': 4714, 'volume': 726360, 'iv': 12.79, 'ltp': 14.45, 'chng': 0.15},
            {'strike': 24950, 'oi': 34699, 'chng_oi': 583, 'volume': 434233, 'iv': 13.01, 'ltp': 10.95, 'chng': -0.60},
            {'strike': 25000, 'oi': 165009, 'chng_oi': -592, 'volume': 949617, 'iv': 13.28, 'ltp': 8.15, 'chng': -1.35},
        ]
        
        puts_data = [
            {'strike': 24000, 'oi': 16950, 'chng_oi': 51415, 'volume': 1072766, 'iv': 15.94, 'ltp': 8.95, 'chng': -17.45},
            {'strike': 24050, 'oi': 16800, 'chng_oi': 13175, 'volume': 453551, 'iv': 15.46, 'ltp': 10.90, 'chng': -21.20},
            {'strike': 24100, 'oi': 9075, 'chng_oi': 22226, 'volume': 815541, 'iv': 14.97, 'ltp': 13.30, 'chng': -25.55},
            {'strike': 24150, 'oi': 7425, 'chng_oi': 22132, 'volume': 607644, 'iv': 14.46, 'ltp': 16.45, 'chng': -32.00},
            {'strike': 24200, 'oi': 1275, 'chng_oi': 41627, 'volume': 1180144, 'iv': 14.12, 'ltp': 20.75, 'chng': -38.20},
            {'strike': 24250, 'oi': 2175, 'chng_oi': 28432, 'volume': 814589, 'iv': 13.70, 'ltp': 26.30, 'chng': -46.05},
            {'strike': 24300, 'oi': 4950, 'chng_oi': 42074, 'volume': 1712849, 'iv': 13.45, 'ltp': 33.85, 'chng': -54.70},
            {'strike': 24350, 'oi': 675, 'chng_oi': 49544, 'volume': 1571658, 'iv': 13.07, 'ltp': 43.10, 'chng': -63.50},
            {'strike': 24400, 'oi': 1650, 'chng_oi': 113457, 'volume': 3117448, 'iv': 12.83, 'ltp': 55.60, 'chng': -71.85},
            {'strike': 24450, 'oi': 1050, 'chng_oi': 66731, 'volume': 1697127, 'iv': 12.61, 'ltp': 70.20, 'chng': -81.75},
            {'strike': 24500, 'oi': 3150, 'chng_oi': 84614, 'volume': 1791956, 'iv': 12.45, 'ltp': 88.90, 'chng': -92.65},
            {'strike': 24550, 'oi': 1200, 'chng_oi': 20328, 'volume': 397028, 'iv': 12.42, 'ltp': 112.20, 'chng': -102.00},
            {'strike': 24600, 'oi': 525, 'chng_oi': 13128, 'volume': 424063, 'iv': 12.41, 'ltp': 139.20, 'chng': -110.95},
            {'strike': 24650, 'oi': 600, 'chng_oi': 2624, 'volume': 78232, 'iv': 12.53, 'ltp': 171.00, 'chng': -118.65},
            {'strike': 24700, 'oi': 75, 'chng_oi': 3427, 'volume': 122032, 'iv': 12.57, 'ltp': 205.75, 'chng': -121.75},
            {'strike': 24750, 'oi': 225, 'chng_oi': 1192, 'volume': 19365, 'iv': 12.82, 'ltp': 244.50, 'chng': -131.10},
            {'strike': 24800, 'oi': 225, 'chng_oi': -149, 'volume': 34501, 'iv': 13.09, 'ltp': 284.00, 'chng': -131.15},
            {'strike': 24850, 'oi': 75, 'chng_oi': 168, 'volume': 3152, 'iv': 13.12, 'ltp': 328.45, 'chng': -135.15},
            {'strike': 24900, 'oi': 75, 'chng_oi': -617, 'volume': 5806, 'iv': 13.50, 'ltp': 371.95, 'chng': -138.25},
            {'strike': 24950, 'oi': 225, 'chng_oi': 61, 'volume': 484, 'iv': 14.98, 'ltp': 422.65, 'chng': -137.65},
            {'strike': 25000, 'oi': 450, 'chng_oi': -405, 'volume': 8092, 'iv': 14.40, 'ltp': 466.20, 'chng': -136.20},
        ]
        
        calls_df = pd.DataFrame(calls_data)
        puts_df = pd.DataFrame(puts_data)
        
        return calls_df, puts_df
        """Determine current market phase"""
        current = self.current_time
        if time(9, 15) <= current <= time(10, 0):
            return 'OPENING'
        elif time(14, 30) <= current <= time(15, 30):
            return 'CLOSING'
        else:
            return 'MID_SESSION'
    
    def create_visualizations(self, calls_df, puts_df, current_spot):
        """Create comprehensive visualization plots"""
        
        # 1. CE PE LTP Plot
        fig_ltp = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Call Options (CE) - LTP', 'Put Options (PE) - LTP'),
            vertical_spacing=0.08
        )
        
        fig_ltp.add_trace(
            go.Scatter(x=calls_df['strike'], y=calls_df['ltp'], 
                      mode='lines+markers', name='CE LTP', 
                      line=dict(color='green', width=3)), row=1, col=1
        )
        
        fig_ltp.add_trace(
            go.Scatter(x=puts_df['strike'], y=puts_df['ltp'], 
                      mode='lines+markers', name='PE LTP', 
                      line=dict(color='red', width=3)), row=2, col=1
        )
        
        # Add current spot line
        fig_ltp.add_vline(x=current_spot, line_dash="dash", 
                         line_color="blue", annotation_text="Current Spot")
        
        fig_ltp.update_layout(height=600, title="Options Premium Analysis")
        
        # 2. Open Interest Analysis
        fig_oi = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Call OI', 'Put OI', 'Call OI Change', 'Put OI Change'),
            vertical_spacing=0.1, horizontal_spacing=0.1
        )
        
        fig_oi.add_trace(
            go.Bar(x=calls_df['strike'], y=calls_df['oi'], 
                  name='Call OI', marker_color='lightgreen'), row=1, col=1
        )
        
        fig_oi.add_trace(
            go.Bar(x=puts_df['strike'], y=puts_df['oi'], 
                  name='Put OI', marker_color='lightcoral'), row=1, col=2
        )
        
        fig_oi.add_trace(
            go.Bar(x=calls_df['strike'], y=calls_df['chng_oi'], 
                  name='Call OI Change', 
                  marker_color=['green' if x > 0 else 'red' for x in calls_df['chng_oi']]), 
            row=2, col=1
        )
        
        fig_oi.add_trace(
            go.Bar(x=puts_df['strike'], y=puts_df['chng_oi'], 
                  name='Put OI Change',
                  marker_color=['green' if x > 0 else 'red' for x in puts_df['chng_oi']]), 
            row=2, col=2
        )
        
        fig_oi.update_layout(height=700, title="Open Interest Analysis")
        
        # 3. Volume Analysis
        fig_vol = go.Figure()
        
        fig_vol.add_trace(
            go.Bar(x=calls_df['strike'], y=calls_df['volume'], 
                  name='Call Volume', marker_color='rgba(0,255,0,0.7)')
        )
        
        fig_vol.add_trace(
            go.Bar(x=puts_df['strike'], y=puts_df['volume'], 
                  name='Put Volume', marker_color='rgba(255,0,0,0.7)')
        )
        
        fig_vol.add_vline(x=current_spot, line_dash="dash", 
                         line_color="blue", annotation_text="Current Spot")
        
        fig_vol.update_layout(
            title="Volume Analysis", 
            xaxis_title="Strike Price", 
            yaxis_title="Volume",
            height=400
        )
        
        # 4. PCR Analysis
        total_call_oi = calls_df['oi'].sum()
        total_put_oi = puts_df['oi'].sum()
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        total_call_vol = calls_df['volume'].sum()
        total_put_vol = puts_df['volume'].sum()
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        return fig_ltp, fig_oi, fig_vol, pcr_oi, pcr_vol

# Main Streamlit App
def main():
    st.title("🚀 Nifty Options Analysis & Trading Signals")
    st.markdown("---")
    
    # Initialize analyzer
    analyzer = OptionsAnalyzer()
    
    # Sidebar
    st.sidebar.header("📊 Data Input")
    
    # Data input method selection
    input_method = st.sidebar.radio(
        "Choose Data Input Method:",
        ["Upload CSV File", "Use Sample Data", "Manual Entry"]
    )
    
    calls_df, puts_df = None, None
    
    if input_method == "Upload CSV File":
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Options Chain CSV", 
            type=['csv'],
            help="Upload the latest Nifty options chain data"
        )
        
        if uploaded_file is not None:
            # Load data
            with st.spinner("Loading and analyzing data..."):
                calls_df, puts_df = analyzer.load_and_clean_data(uploaded_file)
    
    elif input_method == "Use Sample Data":
        # Use the data from the document
        st.sidebar.info("Using sample data from your uploaded document")
        calls_df, puts_df = analyzer.create_sample_data()
    
    elif input_method == "Manual Entry":
        st.sidebar.info("Manual entry feature - coming soon!")
        st.sidebar.markdown("For now, please use CSV upload or sample data.")
    
    # Manual current spot input
    current_spot = st.sidebar.number_input(
        "Current Nifty Spot Price", 
        min_value=20000, max_value=30000, 
        value=24526, step=1,
        help="Enter current Nifty spot price"
    )
    
    
    if calls_df is not None and puts_df is not None and not calls_df.empty:
            
            # Market Overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Nifty", f"{current_spot:,.0f}")
            
            with col2:
                market_phase = analyzer.get_market_phase()
                st.metric("Market Phase", market_phase)
            
            with col3:
                total_call_vol = calls_df['volume'].sum()
                st.metric("Total Call Volume", f"{total_call_vol:,.0f}")
            
            with col4:
                total_put_vol = puts_df['volume'].sum()
                st.metric("Total Put Volume", f"{total_put_vol:,.0f}")
            
            st.markdown("---")
            
            # Generate recommendations
            with st.spinner("Generating trading signals..."):
                recommendations = analyzer.get_entry_exit_signals(calls_df, puts_df, current_spot)
            
            # Display recommendations
            st.header("🎯 Trading Recommendations")
            
            if recommendations:
                for i, rec in enumerate(recommendations[:5]):  # Top 5
                    
                    # Color coding based on recommendation
                    if rec['recommendation'] == 'STRONG BUY':
                        color = "green"
                        emoji = "🟢"
                    elif rec['recommendation'] == 'BUY':
                        color = "blue"
                        emoji = "🔵"
                    elif rec['recommendation'] == 'WEAK BUY':
                        color = "orange"
                        emoji = "🟡"
                    else:
                        color = "gray"
                        emoji = "⚪"
                    
                    with st.expander(f"{emoji} {rec['option_type']} {rec['strike']} @ ₹{rec['ltp']} - {rec['recommendation']}", expanded=(i==0)):
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**📈 Entry Details**")
                            st.write(f"**Entry Price:** ₹{rec['ltp']}")
                            st.write(f"**Probability:** {rec['probability']}%")
                            st.write(f"**Risk Level:** {rec['risk_level']}")
                            st.write(f"**Moneyness:** {rec['moneyness']:.1f}%")
                        
                        with col2:
                            st.markdown("**🎯 Targets & Stop Loss**")
                            st.write(f"**Target 1:** ₹{rec['target_1']}")
                            st.write(f"**Target 2:** ₹{rec['target_2']}")
                            st.write(f"**Stop Loss:** ₹{rec['stop_loss']}")
                            
                            profit_1 = (rec['target_1'] - rec['ltp']) / rec['ltp'] * 100
                            st.write(f"**Profit Potential:** {profit_1:.0f}%")
                        
                        with col3:
                            st.markdown("**📊 Market Data**")
                            st.write(f"**Volume:** {rec['volume']:,.0f}")
                            st.write(f"**Open Interest:** {rec['oi']:,.0f}")
                            st.write(f"**IV:** {rec['iv']:.1f}%")
                            st.write(f"**OI Signal:** {rec['oi_signal']}")
                        
                        # Entry conditions
                        if rec['entry_conditions']:
                            st.markdown("**✅ Entry Conditions Met:**")
                            for condition in rec['entry_conditions']:
                                st.write(f"• {condition}")
                        
                        # Strategy
                        st.markdown("**📋 Strategy:**")
                        if rec['option_type'] == 'CALL':
                            strategy_text = f"Buy {rec['option_type']} if Nifty shows upward momentum. "
                        else:
                            strategy_text = f"Buy {rec['option_type']} if Nifty shows downward pressure. "
                        
                        strategy_text += f"Target profit of {profit_1:.0f}% with strict stop loss at {rec['stop_loss']}."
                        st.write(strategy_text)
            
            st.markdown("---")
            
            # Visualizations
            st.header("📊 Market Analysis Charts")
            
            with st.spinner("Creating visualizations..."):
                fig_ltp, fig_oi, fig_vol, pcr_oi, pcr_vol = analyzer.create_visualizations(calls_df, puts_df, current_spot)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PCR (OI)", f"{pcr_oi:.2f}", 
                         help="Put-Call Ratio based on Open Interest")
            with col2:
                st.metric("PCR (Volume)", f"{pcr_vol:.2f}", 
                         help="Put-Call Ratio based on Volume")
            with col3:
                sentiment = "Bullish" if pcr_oi < 1.0 else "Bearish"
                st.metric("Market Sentiment", sentiment)
            
            # Display charts
            st.plotly_chart(fig_ltp, use_container_width=True)
            st.plotly_chart(fig_oi, use_container_width=True)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Data tables
            st.header("📋 Raw Data")
            
            tab1, tab2 = st.tabs(["Call Options", "Put Options"])
            
            with tab1:
                st.dataframe(calls_df.style.format({
                    'ltp': '₹{:.2f}',
                    'volume': '{:,.0f}',
                    'oi': '{:,.0f}',
                    'iv': '{:.2f}%'
                }), use_container_width=True)
            
            with tab2:
                st.dataframe(puts_df.style.format({
                    'ltp': '₹{:.2f}',
                    'volume': '{:,.0f}',
                    'oi': '{:,.0f}',
                    'iv': '{:.2f}%'
                }), use_container_width=True)
        
        else:
            st.error("Could not parse the uploaded file. Please check the format.")
    
    else:
        st.info("👆 Please upload an options chain CSV file to begin analysis.")
        
        # Sample data format info
        st.markdown("### Expected CSV Format:")
        st.markdown("""
        The CSV should contain options chain data with columns for:
        - Strike prices
        - Call and Put LTP (Last Traded Price)
        - Volume data
        - Open Interest (OI)
        - Implied Volatility (IV)
        - Change in OI
        """)
    
    # Risk disclaimer
    st.markdown("---")
    st.markdown("### ⚠️ Risk Disclaimer")
    st.markdown("""
    - **High Risk:** Options trading involves substantial risk and may not be suitable for all investors
    - **No Guarantee:** Past performance does not guarantee future results
    - **Use Stop Loss:** Always use proper risk management and position sizing
    - **Market Volatility:** Options prices can change rapidly due to various factors
    - **Consult Advisor:** Consider consulting with a financial advisor before making investment decisions
    """)

if __name__ == "__main__":
    main()
