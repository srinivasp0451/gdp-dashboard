import streamlit as st
import pandas as pd
import requests
from io import StringIO
import datetime
import sqlite3
from io import StringIO
import time

from dhanhq import dhanhq, marketfeed
import nest_asyncio
import time

nest_asyncio.apply()


security_id1 = 0
security_id2 = 0
order_client_id=''
order_access_token=''
data_client_id=''
data_access_token=''
profit_threshold=''
loss_threshold=''
total_profit = 0.0
total_loss = 0.0

market_feed_value = marketfeed.NSE


# Function to load the CSV data from the URL
def load_csv_data():
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    response = requests.get(url)
    data = StringIO(response.text)
    df = pd.read_csv(data)
    return df

# Function to filter the data based on selected index and expiry
def filter_data(df, selected_index, expiry_date, strike_price, option_type):
    # Filter out rows with NaN values in relevant columns
    df = df.dropna(subset=['SEM_CUSTOM_SYMBOL', 'SEM_STRIKE_PRICE', 'SEM_EXPIRY_DATE', 'SEM_OPTION_TYPE'])
   
   
    # Filter by exchange (BSE) for consistency with the sample code
    #df = df[df['SEM_EXM_EXCH_ID'] == 'BSE']
   
   
    # Filter by custom symbol (e.g., 'SENSEX 18 FEB' or 'NIFTY')
    if selected_index == 'Nifty':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('nifty')]
    elif selected_index == 'Sensex':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('sensex')]
    elif selected_index == 'Bank Nifty':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('banknifty')]
    elif selected_index == 'Fin Nifty':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('finnifty')]
    elif selected_index == 'Midcap Nifty':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('midcpnifty')]
    elif selected_index == 'Bankex':
        df = df[df['SEM_CUSTOM_SYMBOL'].str.lower().str.startswith('bankex')]
   
    # Convert the SEM_EXPIRY_DATE to datetime format
    df['SEM_EXPIRY_DATE'] = pd.to_datetime(df['SEM_EXPIRY_DATE'], errors='coerce')

   
    # Filter by expiry date
    if expiry_date:
        df = df[df['SEM_EXPIRY_DATE'].dt.date == expiry_date]
   
    # Filter by strike price and option type (call/put)
    if strike_price and option_type:
        df = df[((df['SEM_STRIKE_PRICE'] == strike_price) & (df['SEM_OPTION_TYPE'] == option_type))]

    return df

# Streamlit UI
st.title("Algo Trading")

# Dropdown to select Nifty or Sensex
selected_index = st.selectbox("Select Index", ["Nifty", "Sensex","Bank Nifty","Fin Nifty","Midcap Nifty","Bankex"],index=1)

# Calendar widget to select expiry date
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.date(2025, 1, 1))

# Dropdown for Option Type (CE or PE)
option_type1 = st.selectbox("Select Option Type", ["CE", "PE"],index=0)
option_type2 = st.selectbox("Select Option Type", ["CE", "PE"],index=1)

# Dropdown for selecting strike price (you can manually add options or make it dynamic later)
strike_price1 = st.number_input("Select Strike Price", min_value=0, step=50,value=23650)
strike_price2 = st.number_input("Select Strike Price", min_value=0, step=50,value=strike_price1-100)

# Fetch the data from the CSV URL
df = load_csv_data()


# Input fields for Entry Price, Stop Loss, Target, etc.
entry_price1 = st.number_input("Entry Price for ce", min_value=0, step=1,value=1)
entry_price2 = st.number_input("Entry Price for pe", min_value=0, step=1,value=1)
less_than_or_greater_than1 = st.selectbox("Select above or below for ce", [">=", "<="])
less_than_or_greater_than2 = st.selectbox("Select above or below for pe", [">=", "<="])
stop_loss_distance1 = st.number_input("Stop Loss Distance ce", min_value=0, step=1,value=5)
stop_loss_distance2 = st.number_input("Stop Loss Distance pe", min_value=0, step=1,value=5)
target_distance1 = st.number_input("Target Distance ce", min_value=0, step=1,value=5)
target_distance2 = st.number_input("Target Distance pe", min_value=0, step=1,value=5)
quantity1 = st.number_input("Quantity ce", min_value=1, step=1, value=75)
quantity2 = st.number_input("Quantity pe", min_value=1, step=1, value=75)
profit_threshold1 = st.number_input("Profit Threshold ce", min_value=1, step=1,value=1000)
profit_threshold2 = st.number_input("Profit Threshold pe", min_value=1, step=1,value=1000)
total_profit = st.number_input("Total Profit Threshold for ce and pe", min_value=1, step=1,value=2000)
loss_threshold1 = st.number_input("Loss Threshold ce", min_value=0, step=1,value=500)
loss_threshold2 = st.number_input("Loss Threshold pe", min_value=0, step=1,value=500)
total_loss = st.number_input("Total Loss Threshold for ce and pe", min_value=0, step=1,value=1000)

# Dropdown for selecting whether to use trailing stop loss or not
use_trailing_stop_loss1 = st.selectbox("Use Trailing Stop Loss for ce?", ["Yes","No"],index=0)
use_trailing_stop_loss2 = st.selectbox("Use Trailing Stop Loss for pe?", ["Yes","No"],index=0)

# Select backtesting or live trading
trade_mode = st.selectbox("Select Trade Mode", ["Backtesting", "Live Trading"])



# Inputs for Live Trading (client ID and access token for live trading)
if trade_mode == "Live Trading":
    order_client_id = st.text_input("Client ID (for placing orders)", type="password",value='22305184')
    order_access_token = st.text_input("Access Token (for placing orders)", type="password",value='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ2MzQwNzQzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.dfxL1pGEwjTX1WIYWpLIPYAvvwXn6KTCUrng295eAvhXDTV2QgnZnKB-HRT9MuZ_n75tJueAaDZsDPinr9p2Mg')
else:
    data_client_id = "1104779876"
    data_access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ2MzQwNzQzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.dfxL1pGEwjTX1WIYWpLIPYAvvwXn6KTCUrng295eAvhXDTV2QgnZnKB-HRT9MuZ_n75tJueAaDZsDPinr9p2Mg"


# Display confirmation of selected options
st.subheader("Selected Configuration for Confirmation:")
st.write(f"Index: {selected_index}")
st.write(f"Option Type1: {option_type1}")
st.write(f"Option Type2: {option_type2}")
st.write(f"Strike Price ce: {strike_price1}")
st.write(f"Strike Price pe: {strike_price2}")
st.write(f"Expiry Date: {expiry_date}")
st.write(f"Entry Price ce: {entry_price1}")
st.write(f"Entry Price pe: {entry_price2}")
st.write(f"Entry Quantity ce: {quantity1}")
st.write(f"Entry Quantity pe: {quantity2}")
st.write(f"Profit Threshold ce: {profit_threshold1}")
st.write(f"Profit Threshold pe: {profit_threshold2}")
st.write(f"Loss Threshold ce: {loss_threshold2}")
st.write(f"Loss Threshold pe: {loss_threshold2}")
st.write(f"Stop Loss Distance ce: {stop_loss_distance1}")
st.write(f"Stop Loss Distance pe: {stop_loss_distance2}")
st.write(f"Target Distance ce: {target_distance1}")
st.write(f"Target Distance pe: {target_distance2}")
st.write(f"Use Trailing Stop Loss ce: {use_trailing_stop_loss1}")
st.write(f"Use Trailing Stop Loss pe: {use_trailing_stop_loss2}")
st.write(f"Trade Mode: {trade_mode}")


# Filtered data based on selection
filtered_df = filter_data(df, selected_index, expiry_date, strike_price1, option_type1)

#print("filtered data will be ",filtered_df)

# Option chain display
if not filtered_df.empty:
    # Show the corresponding security ID
    security_id1 = filtered_df.iloc[0]['SEM_SMST_SECURITY_ID']
    st.write(f"Security ID for Strike Price {strike_price1} and Option Type {option_type1}: {security_id1}")
    st.dataframe(filtered_df)
else:
    st.write("No data available for the selected criteria.")
st.write(f"Security id: {security_id1}")


# Filtered data based on selection
filtered_df = filter_data(df, selected_index, expiry_date, strike_price2, option_type2)

#print("filtered data will be ",filtered_df)

# Option chain display
if not filtered_df.empty:
    # Show the corresponding security ID
    security_id2 = filtered_df.iloc[0]['SEM_SMST_SECURITY_ID']
    st.write(f"Security ID for Strike Price {strike_price2} and Option Type {option_type2}: {security_id2}")
    st.dataframe(filtered_df)
else:
    st.write("No data available for the selected criteria.")
st.write(f"Security id: {security_id2}")


# Default values for the variables
selected_index = selected_index  # Default index, can be selected by user
option_type1 = option_type1  # Default option type (CE or PE), can be selected by user
option_type2 = option_type2
strike_price1 = strike_price1  # Default strike price, can be selected by user
strike_price2 = strike_price2
expiry_date = datetime.date.today()  # Default expiry date, can be selected by user

entry_price1 = entry_price1  # Example entry price
entry_price2 = entry_price2
stop_loss_distance1 = stop_loss_distance1  # Example stop loss distance (in points)
stop_loss_distance2 = stop_loss_distance2
target_distance1 = target_distance1  # Example target distance (in points)
target_distance2 = target_distance2

order_status1 = "not_placed"  # Default order status
order_status2 = "not_placed"
entry_time = datetime.datetime.now()  # Default entry time (current time)
exit_time = None  # Exit time will be set later when the trade is closed
profit_or_loss1 = 0  # Initial profit/loss is zero
profit_or_loss2 = 0
profit_or_loss = 0
profit_points = 0  # Initial profit points
loss_points = 0  # Initial loss points
backtest = True  # Default to backtesting,  set to False for live trading

profit_placeholder = st.empty()
ltp_placeholder_ce = st.empty()
ltp_placeholder_pe = st.empty()
trailing_placeholder = st.empty()
trailing_placeholder_ce = st.empty()
trailing_placeholder_pe = st.empty()

profit_placeholder_ce = st.empty()
profit_placeholder_pe = st.empty()


shared_state = {
    'exit_flag': False
    }


# Confirm the configuration before proceeding
if st.button("Start") and security_id1 and security_id2:
    if trade_mode == "Live Trading" and order_client_id and order_access_token:
       

        # Place order function
        def place_order(symbol, qty, price, order_type="buy", exchange_segment=None, product_type=None):
           
            dhan = dhanhq(client_id=order_client_id,access_token=order_access_token)
            
            if exchange_segment is None:
                if selected_index in ["Nifty", "Bank Nifty","Fin Nifty","Midcap Nifty"]:
                    exchange_segment = dhan.NSE_FNO  # Futures and Options segment
                else:
                    exchange_segment = dhan.BSE_FNO

            if product_type is None:
                product_type = dhan.INTRA  # Intraday product type

            print(f"symbol {symbol} {qty} {price} {order_type} {exchange_segment} {product_type}")
            print(f"order client id {order_client_id} token {order_access_token}")

            # Use Dhan's place_order method for placing orders
            order_data = dhan.place_order(
                security_id=int(symbol),  # Security ID of the option (symbol passed as argument)
                exchange_segment=exchange_segment,  # Exchange segment for Futures & Options
                transaction_type=dhan.BUY if order_type.lower() == "buy" else dhan.SELL,
                quantity=qty,  # Quantity of the option contracts
                order_type=dhan.MARKET,  # Market order
                product_type=product_type,  # Product type (e.g., INTRA for intraday)
                price=price  # Price at which to place the order (0 for market orders)
            )
            print("order is", order_data)
            st.write(f"Order Details: {order_data}")
            return order_data


        from dhanhq import dhanhq, marketfeed
        import nest_asyncio
        import time
        import threading

        nest_asyncio.apply()

        # Add your Dhan Client ID and Access Token

        # Define trade parameters
        entry_price1 = entry_price1  # Example entry price for NIFTY 50 23000CE
        entry_price2 = entry_price2
        stop_loss_distance1 = stop_loss_distance1  # Trailing stop loss distance (in points)
        stop_loss_distance2 = stop_loss_distance2
        target_distance1 = target_distance1  # Initial target distance (in points)
        target_distance2 = target_distance2
        trailing_stop_loss_price1 = 0.0
        trailing_stop_loss_price2 = 0.0
        quantity1 = quantity1
        quantity2 = quantity2
        security_id1 = security_id1  # 75300 PE Example security_id for options
        security_id2 = security_id2
        #security_id = 844230
        profit_threshold1 = profit_threshold1
        profit_threshold2 = profit_threshold2

        loss_threshold1 = loss_threshold1
        loss_threshold2 = loss_threshold2
        
        profit_or_loss1 = 0.0
        profit_or_loss2 = 0.0


        print(f"profit threshold1 {profit_threshold1}")
        print(f"profit threshold2 {profit_threshold2}")

        print(f"loss threshold1 {loss_threshold1}")
        print(f"loss threshold2 {loss_threshold2}")

        if selected_index in ['Nifty','Bank Nifty','Fin Nifty','Midcap Nifty']:
            market_feed_value = marketfeed.NSE_FNO  # Futures and Options segment
        else:
            market_feed_value = marketfeed.BSE_FNO
        
        instruments1 = [(market_feed_value, str(security_id1), marketfeed.Ticker)]  # Ticker Data
        instruments2 = [(market_feed_value, str(security_id2), marketfeed.Ticker)]  # Ticker Data
        version = "v2"  # Mention Version and set to latest version 'v2'

        # Define order status variables
        order_status1 = "not_placed"
        order_status2 = "not_placed"  # Can be 'not_placed', 'placed', 'target_hit', 'stop_loss_hit'
        highest_price1 = 0
        highest_price2 = 0  # To track the highest price reached after entry
        current_target1 = entry_price1 + target_distance1  # Initial target based on entry price
        current_target2 = entry_price2 + target_distance2


        data_client_id = "1104779876"
        data_access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ2MzQwNzQzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.dfxL1pGEwjTX1WIYWpLIPYAvvwXn6KTCUrng295eAvhXDTV2QgnZnKB-HRT9MuZ_n75tJueAaDZsDPinr9p2Mg"

        ltp_placeholder_pe = st.empty()
        ltp_placeholder_ce  = st.empty()
        

        # Main trading loop
        try:
            data = marketfeed.DhanFeed(data_client_id, data_access_token, instruments1, version)
            data2 = marketfeed.DhanFeed(data_client_id, data_access_token, instruments2, version)
            st.write("Fetching Data for backtesting")
            print(f"security id {security_id1}")

            def monitor_ce_trade(state):

                while not state["exit_flag"]:
                    
                    data.run_forever()
                    response1 = data.get_data()
                    # print(response)
                

                    if 'LTP' in response1.keys():
                        ltp1 = response1['LTP']
                        #st.write(f"LTP {ltp}")
                        ltp_placeholder_ce.markdown(f"{selected_index} {strike_price1} {option_type1}   LTP:   {ltp1}")

                        # Place buy order if LTP reaches the entry price
                        if order_status1 == "not_placed":
                            if less_than_or_greater_than1 == ">=":
                                if float(ltp1) >= entry_price1:
                                    st.write(f"{float(ltp1)} >= {entry_price1}")
                                    st.write("LTP reached entry price, placing order...")
                                    print("LTP reached entry price, placing order...")
                                    place_order(str(security_id1), quantity1, float(ltp1), "buy")
                                    
                                    order_status1 = "placed"
                                    highest_price1 = float(ltp1)  # Set highest price to entry price
                                    print(f"Buy order placed at {ltp1}")
                                    st.write(f"Buy order placed at {ltp1}")
                                    entry_price1 = float(ltp1)
                                
                            else:
                                if float(ltp1) <= entry_price1:
                                    st.write("LTP reached entry price or below, placing order...")
                                    print("LTP reached entry price, placing order...")
                                    # Place buy order
                                    place_order(str(security_id1), quantity1, float(ltp1), "buy")
                                    order_status1 = "placed"
                                    highest_price1 = float(ltp1)  # Set highest price to entry price
                                    print(f"Buy order placed at {ltp1}")
                                    st.write(f"Buy order placed at {ltp1}")
                                    entry_price1 = float(ltp1)

                        # Check if target or stop loss (with trailing) is hit
                        if order_status1 == "placed":

                            # Calculate profit/loss based on amount
                            profit_or_loss1 = (float(ltp1) - entry_price1) * quantity1
                            print(f"Current Profit/Loss: {profit_or_loss1}")
                            profit_placeholder_ce.markdown(f"Current Profit/Loss: {profit_or_loss1}")

                            # Trailing Target: If the price increases, increase the target
                            if float(ltp1) >= current_target1:
                                current_target1 = float(ltp1) + target_distance1  # Increase the target
                                print(f"Target adjusted to {current_target1}")
                                st.write(f"Target adjusted to {current_target1}")

                            # Update highest price and trailing stop loss if LTP goes higher
                            if float(ltp1) > highest_price1:
                                highest_price1 = float(ltp1)  # New highest price

                            if use_trailing_stop_loss1=="Yes":
                                # Calculate the trailing stop loss based on the highest price
                                trailing_stop_loss_price1 = highest_price1 - stop_loss_distance1
                                print(f"Entry price: {entry_price1} Trailing Stop Loss at: {trailing_stop_loss_price1} highest price: {highest_price1} target: {current_target1}")
                                trailing_placeholder_ce.markdown(f"Entry price: {entry_price1} Trailing Stop Loss at: {trailing_stop_loss_price1} highest price: {highest_price1} target: {current_target1}")
                            else:
                                trailing_stop_loss_price1 = entry_price1-stop_loss_distance1
                                current_target1 = entry_price1+target_distance1
                                trailing_placeholder_ce.markdown(f"Entry price: {entry_price1} Fixed Stop Loss at: {trailing_stop_loss_price1} highest price: {highest_price1} target: {current_target1}")

                            # Check if the price hits the trailing target
                            if float(ltp1) >= current_target1:
                                print("Trailing target hit! Exiting trade.")
                                st.write("Trailing target hit! Exiting trade.")
                                order_status1 = "target_hit"
                                # Exit the trade (place a sell order)
                                place_order(str(security_id1), quantity1, float(ltp1), "sell")
                            
                                print(f"Exited at {ltp1}")
                                st.write(f"Exited at {ltp1}")
                                print(f"Current Profit/Loss: {profit_or_loss1}")
                                st.write(f"Current Profit/Loss: {profit_or_loss1}")

                                # order_status = "not_placed"  # Default order status
                                # entry_time = datetime.datetime.now()  # Default entry time (current time)
                                # exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                                profit_or_loss1 = profit_or_loss1  # Initial profit/loss is zero
                                
                            
                                data.disconnect()
                                break

                            # If the price falls below the trailing stop loss, exit the trade
                            if float(ltp1) <= trailing_stop_loss_price1:
                                print("Trailing stop loss hit! Exiting trade.")
                                st.write("Trailing stop loss hit! Exiting trade.")
                                order_status1 = "stop_loss_hit"
                                # Exit the trade (place a sell order)
                                place_order(str(security_id1), quantity1, float(ltp1), "sell")
                            
                                print(f"Exited at {ltp1}")
                                st.write(f"Exited at {ltp1}")
                                print(f"Current Profit/Loss: {profit_or_loss1}")
                                st.write(f"Current Profit/Loss: {profit_or_loss1}")
                                profit_or_loss1 = profit_or_loss1  # Initial profit/loss is zero
                                # profit_points = float(ltp) - entry_price  # Initial profit points

                                data.disconnect()
                                break



                            # Check if profit or loss threshold is exceeded
                            if profit_or_loss1 >= profit_threshold1:
                                print(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss1}.")
                                st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss1}.")
                                order_status1 = "target_hit"
                                # Exit the trade (place a sell order)
                                place_order(str(security_id1), quantity1, float(ltp1), "sell")
                                print(f"Exited at {ltp1}")
                                st.write(f"Exited at {ltp1}")

                                profit_or_loss1 = profit_or_loss1  # Initial profit/loss is zero

                                data.disconnect()
                                break
                            elif profit_or_loss1 <= -loss_threshold1:
                                print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss1}.")
                                st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss1}.")
                                order_status1 = "stop_loss_hit"
                                # Exit the trade (place a sell order)
                                place_order(str(security_id1), quantity1, float(ltp1), "sell")
                                print(f"Exited at {ltp1}")
                                st.write(f"Exited at {ltp1}")

                                profit_or_loss1 = profit_or_loss1  # Initial profit/loss is zero
                                

                                data.disconnect()
                                break
                    # Check combined PnL after CE trade
                    total_pnl = profit_or_loss1 + profit_or_loss2
                    if total_pnl >= total_profit or total_pnl <= -total_loss:
                        state["exit_flag"] = True
                        break

            def monitor_pe_trade(state):

                while not state['exit_flag']:
                    
                    data2.run_forever()
                    response = data2.get_data()
                    # print(response)
                

                    if 'LTP' in response.keys():
                        ltp2 = response['LTP']
                        #st.write(f"LTP {ltp}")
                        ltp_placeholder_pe.markdown(f"{selected_index} {strike_price2} {option_type2}   LTP:   {ltp2}")

                        # Place buy order if LTP reaches the entry price
                        if order_status2 == "not_placed":
                            if less_than_or_greater_than2 == ">=":
                                if float(ltp2) >= entry_price2:
                                    st.write(f"{float(ltp2)} >= {entry_price2}")
                                    st.write("LTP reached entry price, placing order...")
                                    print("LTP reached entry price, placing order...")
                                    # Place buy order
                                    place_order(str(security_id2), quantity2, float(ltp2), "buy")
                                    order_status2 = "placed"
                                    highest_price2 = float(ltp2)  # Set highest price to entry price
                                    print(f"Buy order placed at {ltp2}")
                                    st.write(f"Buy order placed at {ltp2}")
                                    entry_price2 = float(ltp2)

                            else:
                                if float(ltp2) <= entry_price2:
                                    st.write("LTP reached entry price or below, placing order...")
                                    print("LTP reached entry price, placing order...")
                                    # Place buy order
                                    place_order(str(security_id2), quantity2, float(ltp2), "buy")
                                    order_status2 = "placed"
                                    highest_price2 = float(ltp2)  # Set highest price to entry price
                                    print(f"Buy order placed at {ltp2}")
                                    st.write(f"Buy order placed at {ltp2}")
                                    entry_price2 = float(ltp2)

                        # Check if target or stop loss (with trailing) is hit
                        if order_status2 == "placed":

                            # Calculate profit/loss based on amount
                            profit_or_loss2 = (float(ltp2) - entry_price2) * quantity2
                            print(f"Current Profit/Loss: {profit_or_loss2}")
                            profit_placeholder_pe.markdown(f"Current Profit/Loss: {profit_or_loss2}")

                            # Trailing Target: If the price increases, increase the target
                            if float(ltp2) >= current_target2:
                                current_target2 = float(ltp2) + target_distance2  # Increase the target
                                print(f"Target adjusted to {current_target2}")
                                st.write(f"Target adjusted to {current_target2}")

                            # Update highest price and trailing stop loss if LTP goes higher
                            if float(ltp2) > highest_price2:
                                highest_price2 = float(ltp2)  # New highest price

                            if use_trailing_stop_loss2=="Yes":
                                # Calculate the trailing stop loss based on the highest price
                                trailing_stop_loss_price2 = highest_price2 - stop_loss_distance2
                                print(f"Entry price: {entry_price2} Trailing Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}")
                                trailing_placeholder_pe.markdown(f"Entry price: {entry_price2} Trailing Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}")
                            else:
                                trailing_stop_loss_price2 = entry_price2-stop_loss_distance2
                                current_target2 = entry_price2+target_distance2
                                trailing_placeholder_pe.markdown(f"Entry price: {entry_price2} Fixed Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}")

                            # Check if the price hits the trailing target
                            if float(ltp2) >= current_target2:
                                print("Trailing target hit! Exiting trade.")
                                st.write("Trailing target hit! Exiting trade.")
                                order_status2 = "target_hit"
                                # Exit the trade (place a sell order)
                                place_order(str(security_id2), quantity2, float(ltp2), "sell")
                                print(f"Exited at {ltp2}")
                                st.write(f"Exited at {ltp2}")
                                print(f"Current Profit/Loss: {profit_or_loss2}")
                                st.write(f"Current Profit/Loss: {profit_or_loss2}")

                                profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                                
                            
                                data2.disconnect()
                                break

                            # If the price falls below the trailing stop loss, exit the trade
                            if float(ltp2) <= trailing_stop_loss_price2:
                                print("Trailing stop loss hit! Exiting trade.")
                                st.write("Trailing stop loss hit! Exiting trade.")
                                place_order(str(security_id2), quantity2, float(ltp2), "sell")
                                order_status2 = "stop_loss_hit"
                            
                                print(f"Exited at {ltp2}")
                                st.write(f"Exited at {ltp2}")
                                print(f"Current Profit/Loss: {profit_or_loss2}")
                                st.write(f"Current Profit/Loss: {profit_or_loss2}")
                                
                                profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                                
                                data2.disconnect()
                                break



                            # Check if profit or loss threshold is exceeded
                            if profit_or_loss2 >= profit_threshold2:
                                print(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss2}.")
                                st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss2}.")
                                order_status2 = "target_hit"
                                # Exit the trade (place a sell order)
                                place_order(str(security_id2), quantity2, float(ltp2), "sell")
                                print(f"Exited at {ltp2}")
                                st.write(f"Exited at {ltp2}")

                                profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                                data2.disconnect()
                                break
                            elif profit_or_loss2 <= -loss_threshold2:
                                print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss2}.")
                                st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss2}.")
                                order_status2 = "stop_loss_hit"
                                # Exit the trade (place a sell order)
                                place_order(str(security_id2), quantity2, float(ltp2), "sell")
                                print(f"Exited at {ltp2}")
                                st.write(f"Exited at {ltp2}")

                                profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                                

                                data2.disconnect()
                                break

                    # Check combined PnL after CE trade
                    total_pnl = profit_or_loss1 + profit_or_loss2
                    if total_pnl >= total_profit or total_pnl <= -total_loss:
                        state["exit_flag"] = True
                        break

            # Start threads
            thread_ce = threading.Thread(target=monitor_ce_trade,args=(shared_state,))
            thread_pe = threading.Thread(target=monitor_pe_trade,args=(shared_state,))

            thread_ce.start()
            thread_pe.start()

                       

        except Exception as e:
            print(e)
            print("Exception occured")
            print("Execution interrupted by user. Disconnecting...")
            st.write("Execution interrupted by user. Disconnecting...")
           
            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id1), 16)]

                data2.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id1), 16)]

                data2.unsubscribe_symbols(unsub_instruments)

            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id2), 16)]

                data2.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id2), 16)]

                data2.unsubscribe_symbols(unsub_instruments)


            data2.disconnect()  # This ensures disconnect when the program is forcefully stopped.
           
           

        finally:
           
            #Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id1), 16)]

                data2.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id1), 16)]

                data2.unsubscribe_symbols(unsub_instruments)

            #Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id2), 16)]

                data2.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id2), 16)]

                data2.unsubscribe_symbols(unsub_instruments)

            data2.disconnect()

        # Close Connection
        data2.disconnect()

    elif trade_mode == "Backtesting":
        from dhanhq import dhanhq, marketfeed
        import nest_asyncio
        import time
        import threading

        nest_asyncio.apply()

        # Add your Dhan Client ID and Access Token

        # Define trade parameters
        entry_price1 = entry_price1  # Example entry price for NIFTY 50 23000CE
        entry_price2 = entry_price2
        stop_loss_distance1 = stop_loss_distance1  # Trailing stop loss distance (in points)
        stop_loss_distance2 = stop_loss_distance2
        target_distance1 = target_distance1  # Initial target distance (in points)
        target_distance2 = target_distance2
        trailing_stop_loss_price1 = 0.0
        trailing_stop_loss_price2 = 0.0
        quantity1 = quantity1
        quantity2 = quantity2
        security_id1 = security_id1  # 75300 PE Example security_id for options
        security_id2 = security_id2
        #security_id = 844230
        profit_threshold1 = profit_threshold1
        profit_threshold2 = profit_threshold2

        loss_threshold1 = loss_threshold1
        loss_threshold2 = loss_threshold2
        
        profit_or_loss1 = 0.0
        profit_or_loss2 = 0.0


        print(f"profit threshold1 {profit_threshold1}")
        print(f"profit threshold2 {profit_threshold2}")

        print(f"loss threshold1 {loss_threshold1}")
        print(f"loss threshold2 {loss_threshold2}")

        if selected_index in ['Nifty','Bank Nifty','Fin Nifty','Midcap Nifty']:
            market_feed_value = marketfeed.NSE_FNO  # Futures and Options segment
        else:
            market_feed_value = marketfeed.BSE_FNO
        
        instruments1 = [(market_feed_value, str(security_id1), marketfeed.Ticker)]  # Ticker Data
        instruments2 = [(market_feed_value, str(security_id2), marketfeed.Ticker)]  # Ticker Data
        version = "v2"  # Mention Version and set to latest version 'v2'

        # Define order status variables
        order_status1 = "not_placed"
        order_status2 = "not_placed"  # Can be 'not_placed', 'placed', 'target_hit', 'stop_loss_hit'
        highest_price1 = 0
        highest_price2 = 0  # To track the highest price reached after entry
        current_target1 = entry_price1 + target_distance1  # Initial target based on entry price
        current_target2 = entry_price2 + target_distance2


        data_client_id = "1104779876"
        data_access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ2MzQwNzQzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.dfxL1pGEwjTX1WIYWpLIPYAvvwXn6KTCUrng295eAvhXDTV2QgnZnKB-HRT9MuZ_n75tJueAaDZsDPinr9p2Mg"

        ltp_placeholder_pe = st.empty()
        ltp_placeholder_ce  = st.empty()
        

        # Main trading loop
        try:
            data = marketfeed.DhanFeed(data_client_id, data_access_token, instruments1, version)
            data2 = marketfeed.DhanFeed(data_client_id, data_access_token, instruments2, version)
            st.write("Fetching Data for backtesting")
            print(f"security id {security_id1}")

            def monitor_ce_trade(state):
           

                while not state['exit_flag']:
                    
                    data.run_forever()
                    response1 = data.get_data()
                    # print(response)
                

                    if 'LTP' in response1.keys():
                        ltp1 = response1['LTP']
                        #st.write(f"LTP {ltp}")
                        ltp_placeholder_ce.markdown(f"{selected_index} {strike_price1} {option_type1}   LTP:   {ltp1}")

                        # Place buy order if LTP reaches the entry price
                        if order_status1 == "not_placed":
                            if less_than_or_greater_than1 == ">=":
                                if float(ltp1) >= entry_price1:
                                    st.write(f"{float(ltp1)} >= {entry_price1}")
                                    st.write("LTP reached entry price, placing order...")
                                    print("LTP reached entry price, placing order...")
                                    
                                    order_status1 = "placed"
                                    highest_price1 = float(ltp1)  # Set highest price to entry price
                                    print(f"Buy order placed at {ltp1}")
                                    st.write(f"Buy order placed at {ltp1}")
                                    entry_price1 = float(ltp1)
                                
                            else:
                                if float(ltp1) <= entry_price1:
                                    st.write("LTP reached entry price or below, placing order...")
                                    print("LTP reached entry price, placing order...")
                                    # Place buy order
                                
                                    order_status1 = "placed"
                                    highest_price1 = float(ltp1)  # Set highest price to entry price
                                    print(f"Buy order placed at {ltp1}")
                                    st.write(f"Buy order placed at {ltp1}")
                                    entry_price1 = float(ltp1)

                        # Check if target or stop loss (with trailing) is hit
                        if order_status1 == "placed":

                            # Calculate profit/loss based on amount
                            profit_or_loss1 = (float(ltp1) - entry_price1) * quantity1
                            print(f"Current Profit/Loss: {profit_or_loss1}")
                            profit_placeholder_ce.markdown(f"Current Profit/Loss: {profit_or_loss1}")

                            # Trailing Target: If the price increases, increase the target
                            if float(ltp1) >= current_target1:
                                current_target1 = float(ltp1) + target_distance1  # Increase the target
                                print(f"Target adjusted to {current_target1}")
                                st.write(f"Target adjusted to {current_target1}")

                            # Update highest price and trailing stop loss if LTP goes higher
                            if float(ltp1) > highest_price1:
                                highest_price1 = float(ltp1)  # New highest price

                            if use_trailing_stop_loss1=="Yes":
                                # Calculate the trailing stop loss based on the highest price
                                trailing_stop_loss_price1 = highest_price1 - stop_loss_distance1
                                print(f"Entry price: {entry_price1} Trailing Stop Loss at: {trailing_stop_loss_price1} highest price: {highest_price1} target: {current_target1}")
                                trailing_placeholder_ce.markdown(f"Entry price: {entry_price1} Trailing Stop Loss at: {trailing_stop_loss_price1} highest price: {highest_price1} target: {current_target1}")
                            else:
                                trailing_stop_loss_price1 = entry_price1-stop_loss_distance1
                                current_target1 = entry_price1+target_distance1
                                trailing_placeholder_ce.markdown(f"Entry price: {entry_price1} Fixed Stop Loss at: {trailing_stop_loss_price1} highest price: {highest_price1} target: {current_target1}")

                            # Check if the price hits the trailing target
                            if float(ltp1) >= current_target1:
                                print("Trailing target hit! Exiting trade.")
                                st.write("Trailing target hit! Exiting trade.")
                                order_status1 = "target_hit"
                                # Exit the trade (place a sell order)
                            
                                print(f"Exited at {ltp1}")
                                st.write(f"Exited at {ltp1}")
                                print(f"Current Profit/Loss: {profit_or_loss1}")
                                st.write(f"Current Profit/Loss: {profit_or_loss1}")

                                # order_status = "not_placed"  # Default order status
                                # entry_time = datetime.datetime.now()  # Default entry time (current time)
                                # exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                                profit_or_loss1 = profit_or_loss1  # Initial profit/loss is zero
                                
                            
                                data.disconnect()
                                break

                            # If the price falls below the trailing stop loss, exit the trade
                            if float(ltp1) <= trailing_stop_loss_price1:
                                print("Trailing stop loss hit! Exiting trade.")
                                st.write("Trailing stop loss hit! Exiting trade.")
                                order_status1 = "stop_loss_hit"
                                # Exit the trade (place a sell order)
                            
                                print(f"Exited at {ltp1}")
                                st.write(f"Exited at {ltp1}")
                                print(f"Current Profit/Loss: {profit_or_loss1}")
                                st.write(f"Current Profit/Loss: {profit_or_loss1}")
                                profit_or_loss1 = profit_or_loss1  # Initial profit/loss is zero
                                # profit_points = float(ltp) - entry_price  # Initial profit points

                                data.disconnect()
                                break



                            # Check if profit or loss threshold is exceeded
                            if profit_or_loss1 >= profit_threshold1:
                                print(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss1}.")
                                st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss1}.")
                                order_status1 = "target_hit"
                                # Exit the trade (place a sell order)
                            
                                print(f"Exited at {ltp1}")
                                st.write(f"Exited at {ltp1}")

                                profit_or_loss1 = profit_or_loss1  # Initial profit/loss is zero

                                data.disconnect()
                                break
                            elif profit_or_loss1 <= -loss_threshold1:
                                print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss1}.")
                                st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss1}.")
                                order_status1 = "stop_loss_hit"
                                # Exit the trade (place a sell order)
                            
                                print(f"Exited at {ltp1}")
                                st.write(f"Exited at {ltp1}")

                                profit_or_loss1 = profit_or_loss1  # Initial profit/loss is zero
                                

                                data.disconnect()
                                break

                    # Check combined PnL after CE trade
                    total_pnl = profit_or_loss1 + profit_or_loss2
                    if total_pnl >= total_profit or total_pnl <= -total_loss:
                        state["exit_flag"] = True
                        break


            def monitor_pe_trade(state):

                while not state['exit_flag']:
                    
                    data2.run_forever()
                    response = data2.get_data()
                    # print(response)
                

                    if 'LTP' in response.keys():
                        ltp2 = response['LTP']
                        #st.write(f"LTP {ltp}")
                        ltp_placeholder_pe.markdown(f"{selected_index} {strike_price2} {option_type2}   LTP:   {ltp2}")

                        # Place buy order if LTP reaches the entry price
                        if order_status2 == "not_placed":
                            if less_than_or_greater_than2 == ">=":
                                if float(ltp2) >= entry_price2:
                                    st.write(f"{float(ltp2)} >= {entry_price2}")
                                    st.write("LTP reached entry price, placing order...")
                                    print("LTP reached entry price, placing order...")
                                    # Place buy order
                                
                                    order_status2 = "placed"
                                    highest_price2 = float(ltp2)  # Set highest price to entry price
                                    print(f"Buy order placed at {ltp2}")
                                    st.write(f"Buy order placed at {ltp2}")
                                    entry_price2 = float(ltp2)

                            else:
                                if float(ltp2) <= entry_price2:
                                    st.write("LTP reached entry price or below, placing order...")
                                    print("LTP reached entry price, placing order...")
                                    # Place buy order
                                
                                    order_status2 = "placed"
                                    highest_price2 = float(ltp2)  # Set highest price to entry price
                                    print(f"Buy order placed at {ltp2}")
                                    st.write(f"Buy order placed at {ltp2}")
                                    entry_price2 = float(ltp2)

                        # Check if target or stop loss (with trailing) is hit
                        if order_status2 == "placed":

                            # Calculate profit/loss based on amount
                            profit_or_loss2 = (float(ltp2) - entry_price2) * quantity2
                            print(f"Current Profit/Loss: {profit_or_loss2}")
                            profit_placeholder_pe.markdown(f"Current Profit/Loss: {profit_or_loss2}")

                            # Trailing Target: If the price increases, increase the target
                            if float(ltp2) >= current_target2:
                                current_target2 = float(ltp2) + target_distance2  # Increase the target
                                print(f"Target adjusted to {current_target2}")
                                st.write(f"Target adjusted to {current_target2}")

                            # Update highest price and trailing stop loss if LTP goes higher
                            if float(ltp2) > highest_price2:
                                highest_price2 = float(ltp2)  # New highest price

                            if use_trailing_stop_loss2=="Yes":
                                # Calculate the trailing stop loss based on the highest price
                                trailing_stop_loss_price2 = highest_price2 - stop_loss_distance2
                                print(f"Entry price: {entry_price2} Trailing Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}")
                                trailing_placeholder_pe.markdown(f"Entry price: {entry_price2} Trailing Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}")
                            else:
                                trailing_stop_loss_price2 = entry_price2-stop_loss_distance2
                                current_target2 = entry_price2+target_distance2
                                trailing_placeholder_pe.markdown(f"Entry price: {entry_price2} Fixed Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}")

                            # Check if the price hits the trailing target
                            if float(ltp2) >= current_target2:
                                print("Trailing target hit! Exiting trade.")
                                st.write("Trailing target hit! Exiting trade.")
                                order_status2 = "target_hit"
                                # Exit the trade (place a sell order)
                            
                                print(f"Exited at {ltp2}")
                                st.write(f"Exited at {ltp2}")
                                print(f"Current Profit/Loss: {profit_or_loss2}")
                                st.write(f"Current Profit/Loss: {profit_or_loss2}")

                                profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                                
                            
                                data2.disconnect()
                                break

                            # If the price falls below the trailing stop loss, exit the trade
                            if float(ltp2) <= trailing_stop_loss_price2:
                                print("Trailing stop loss hit! Exiting trade.")
                                st.write("Trailing stop loss hit! Exiting trade.")
                                order_status2 = "stop_loss_hit"
                            
                                print(f"Exited at {ltp2}")
                                st.write(f"Exited at {ltp2}")
                                print(f"Current Profit/Loss: {profit_or_loss2}")
                                st.write(f"Current Profit/Loss: {profit_or_loss2}")
                                
                                profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                                
                                data2.disconnect()
                                break



                            # Check if profit or loss threshold is exceeded
                            if profit_or_loss2 >= profit_threshold2:
                                print(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss2}.")
                                st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss2}.")
                                order_status2 = "target_hit"
                                # Exit the trade (place a sell order)
                            
                                print(f"Exited at {ltp2}")
                                st.write(f"Exited at {ltp2}")

                                profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                                data2.disconnect()
                                break
                            elif profit_or_loss2 <= -loss_threshold2:
                                print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss2}.")
                                st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss2}.")
                                order_status2 = "stop_loss_hit"
                                # Exit the trade (place a sell order)
                            
                                print(f"Exited at {ltp2}")
                                st.write(f"Exited at {ltp2}")

                                profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                                

                                data2.disconnect()
                                break


                    # Check combined PnL after CE trade
                    total_pnl = profit_or_loss1 + profit_or_loss2
                    if total_pnl >= total_profit or total_pnl <= -total_loss:
                        state["exit_flag"] = True
                        break

            # Start threads
            thread_ce = threading.Thread(target=monitor_ce_trade,args=(shared_state,))
            thread_pe = threading.Thread(target=monitor_pe_trade,args=(shared_state,))

            thread_ce.start()
            thread_pe.start()

           

        except Exception as e:
            print(e)
            print("Exception occured")
            print("Execution interrupted by user. Disconnecting...")
            st.write("Execution interrupted by user. Disconnecting...")
           
            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id1), 16)]

                data2.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id1), 16)]

                data2.unsubscribe_symbols(unsub_instruments)

            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id2), 16)]

                data2.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id2), 16)]

                data2.unsubscribe_symbols(unsub_instruments)


            data2.disconnect()  # This ensures disconnect when the program is forcefully stopped.
           
           

        finally:
           
            #Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id1), 16)]

                data2.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id1), 16)]

                data2.unsubscribe_symbols(unsub_instruments)

            #Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id2), 16)]

                data2.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id2), 16)]

                data2.unsubscribe_symbols(unsub_instruments)

            data2.disconnect()

        # Close Connection
        data2.disconnect()
