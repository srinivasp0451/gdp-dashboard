# import streamlit as st
# pip install dhanhq
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

import time
import datetime
import pandas as pd


nest_asyncio.apply()



# Initialize client
client_code = "1104779876"
token_id = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ4OTk5NTU2LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.Xuuz1YAxf317M3YEE40pn3Cbz5B8Qly_S-hxutBg-YJL2oY8D4oEWb-d0AB2IbC1NUjF4n9PX9Aqox-OW-njIQ"


security_id = 0
security_id2 = 0
order_client_id=''
order_access_token=''
data_client_id=''
data_access_token=''
profit_threshold=''
profit_threshold2 =''
loss_threshold=''
loss_threshold2 = ''
market_feed_value = marketfeed.NSE
tradesymbol = ''
total_profit_threshold = 0
total_loss_threshold=0

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
# st.title("Algo Trading")

# Dropdown to select Nifty or Sensex
# selected_index = st.selectbox("Select Index", ["Nifty","Sensex", "Bank Nifty","Fin Nifty","Midcap Nifty","Bankex"])
selected_index = 'Sensex'
selected_index2 = 'Sensex'

# selected_index = 'Nifty'
# selected_index2 = 'Nifty'

# selected_index = 'Bank Nifty'
# selected_index2 = 'Bank Nifty'

# Calendar widget to select expiry date
# expiry_date = st.date_input("Select Expiry Date", min_value=datetime.date(2025, 1, 1))

expiry_date = datetime.date(2025, 5, 29)
expiry_date = datetime.date(2025, 5, 27)
# Dropdown for Option Type (CE or PE)
# option_type = st.selectbox("Select Option Type", ["CE", "PE"])

option_type = 'CE'
option_type2 = 'PE'
# Dropdown for selecting strike price (you can manually add options or make it dynamic later)
# strike_price = st.number_input("Select Strike Price", min_value=0, step=50,value=23250)

strike_price = 81400
strike_price2 = 81400

# strike_price = 80200
# strike_price2 = 80400

# strike_price = 55500
# strike_price2 = 55500

# Fetch the data from the CSV URL
df = load_csv_data()






# Input fields for Entry Price, Stop Loss, Target, etc.
# entry_price = st.number_input("Entry Price", min_value=0, step=1,value=1)
entry_price = 1
entry_price2 = 1
# less_than_or_greater_than = st.selectbox("Select above or below", [">=", "<="])
less_than_or_greater_than = '>='
less_than_or_greater_than2 = '>='
# stop_loss_distance = st.number_input("Stop Loss Distance", min_value=0, step=1,value=5)

stop_loss_distance = 30
stop_loss_distance2=30

# target_distance = st.number_input("Target Distance", min_value=0, step=1,value=3)
target_distance = 50
target_distance2 = 50
# quantity = st.number_input("Quantity", min_value=1, step=1, value=75)
quantity = 20
quantity2=20

# profit_threshold = st.number_input("Profit Threshold", min_value=1, step=1,value=5000)
profit_threshold =1000
profit_threshold2=1000

# loss_threshold = st.number_input("Loss Threshold", min_value=0, step=1,value=350)
loss_threshold = 300
loss_threshold2 = 300


total_profit_threshold = 2000
total_loss_threshold = 500

highest_profit =0
highest_profit2 = 0
max_total_profit = 0



max_highest_profit1 = 0
max_highest_profit2 = 0

brokerage1 = 50
brokerage2 = 50

# timeframe = st.text_input("Time Frame",value=1)
timeframe = '1'

# Dropdown for selecting whether to use trailing stop loss or not
# use_trailing_stop_loss = st.selectbox("Use Trailing Stop Loss?", ["No","Yes"])
# use_trailing_stop_loss = 'No' # 'Yes' 'No'

use_trailing_stop_loss = 'Yes'
use_trailing_stop_loss2 = 'Yes'

use_trailing_stop_loss = 'No'
use_trailing_stop_loss2 = 'No'
# Select backtesting or live trading
# trade_mode = st.selectbox("Select Trade Mode", ["Live Trading","Backtesting"])

# trade_mode ='Live Trading' #Live Trading  Backtesting
# trade_mode ='Backtesting'
trade_mode ='Live Trading'
trade_mode = 'Live Trading'

# trade_mode ='Backtesting'


# Inputs for Live Trading (client ID and access token for live trading)
if trade_mode == "Live Trading":
    # order_client_id = st.text_input("Client ID (for placing orders)", type="password",value='22305184')
    order_client_id = '22305184'
    # order_access_token = st.text_input("Access Token (for placing orders)", type="password",value='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQzNjcyMTgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.I-Hl-IKVl4dioMwp5Qhl-7duuX7BrIXJIc6v6kuLX7g3zMKcNCeAGFstRrbo2N7vDn2WCmY90YxPbmQsnquhpg')
    order_access_token = token_id
else:
    # data_client_id = "1104779876"
    data_client_id = "1104779876"
    # data_access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ2MTcwMzYzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.0PUWmWDNk8BffYk6-fvKaUzzzqR4wuNay27jfx7jXbNPgmt_h1ashEjseUaigRkYN41bzfzjI3oNrDoNcV0gEw"
    data_access_token = token_id



# Filtered data based on selection
filtered_df = filter_data(df, selected_index, expiry_date, strike_price, option_type)
filtered_df2 = filter_data(df, selected_index, expiry_date, strike_price2, option_type2)

#print("filtered data will be ",filtered_df)

# Option chain display
if not filtered_df.empty:
    # Show the corresponding security ID
    security_id = filtered_df.iloc[0]['SEM_SMST_SECURITY_ID']
    print(f"Security ID for Strike Price {strike_price} and Option Type {option_type}: {security_id}")
    tradesymbol = filtered_df.iloc[0]['SEM_CUSTOM_SYMBOL']
    # st.write(tradesymbol)
    # st.dataframe(filtered_df)
    # print(filtered_df)
else:
    # st.write("No data available for the selected criteria.")
    print("No data available for the selected criteria.")

# Option chain display
if not filtered_df2.empty:
    # Show the corresponding security ID
    security_id2 = filtered_df2.iloc[0]['SEM_SMST_SECURITY_ID']
    print(f"Security ID2 for Strike Price {strike_price2} and Option Type {option_type2}: {security_id2}")
    tradesymbol2 = filtered_df2.iloc[0]['SEM_CUSTOM_SYMBOL']
    # st.write(tradesymbol)
    # st.dataframe(filtered_df)
    # print(filtered_df)
else:
    # st.write("No data available for the selected criteria.")
    print("No data available for the selected criteria.")

# st.write(f"Trade Symbol: {tradesymbol}")
# st.write(f"Timeframe: {timeframe}")
# st.write(f"Security id: {security_id}")



# Default values for the variables
selected_index = selected_index  # Default index, can be selected by user
option_type = option_type  # Default option type (CE or PE), can be selected by user
strike_price = strike_price  # Default strike price, can be selected by user
expiry_date = datetime.date.today()  # Default expiry date, can be selected by user
print('Expiry date',expiry_date)
entry_price = entry_price  # Example entry price
stop_loss_distance = stop_loss_distance  # Example stop loss distance (in points)
target_distance = target_distance  # Example target distance (in points)
order_status = "not_placed"  # Default order status
entry_time = datetime.datetime.now()  # Default entry time (current time)
exit_time = None  # Exit time will be set later when the trade is closed
profit_or_loss = 0  # Initial profit/loss is zero
profit_points = 0  # Initial profit points
loss_points = 0  # Initial loss points
backtest = True  # Default to backtesting,  set to False for live trading
capital_used = entry_price*quantity  # Example capital used
time_spent_in_trade = None  # Will calculate once the trade is closed
notes = "Example trade with simple strategy"  # Notes for the trade
day_of_week = entry_time.strftime("%A")  # Day of the week (e.g., Monday)
month_of_year = entry_time.strftime("%B")  # Month of the year (e.g., February)
traded_date = datetime.datetime.today().date()




exchange = 'NFO'    
exchange = 'BFO'
# timeframe = '5'
    
# timeframe = '5'


# Confirm the configuration before proceeding
if security_id and security_id2:
    if trade_mode == "Live Trading" and order_client_id and order_access_token:
       

        # Add your Dhan Client ID and Access Token

        # Define trade parameters
        entry_price = entry_price  # Example entry price for NIFTY 50 23000CE
        stop_loss_distance = stop_loss_distance  # Trailing stop loss distance (in points)
        target_distance = target_distance  # Initial target distance (in points)
        quantity = quantity
        security_id = security_id  # 75300 PE Example security_id for options
        #security_id = 844230
        profit_threshold = profit_threshold
        loss_threshold = -loss_threshold
        instruments = [(marketfeed.BSE_FNO, str(security_id), marketfeed.Ticker)]  # Ticker Data
        version = "v2"  # Mention Version and set to latest version 'v2'

        # Define order status variables
        order_status = "not_placed"  # Can be 'not_placed', 'placed', 'target_hit', 'stop_loss_hit'
        highest_price = 0  # To track the highest price reached after entry
        current_target = entry_price + target_distance  # Initial target based on entry price

        # Place order function
        def place_order(symbol, qty, price, order_type="buy", exchange_segment=None, product_type=None):
           
            dhan = dhanhq(client_id=order_client_id,access_token=order_access_token)

            """
            Place a real-time market order with Dhan API using the official structure.
            Args:
                symbol (str): The symbol (e.g., NIFTY or specific option)
                qty (int): The quantity to buy or sell
                price (float): The price at which to place the order
                order_type (str): "buy" or "sell"
                exchange_segment (str): The exchange segment (default to None, to use NSE_FNO if not provided)
                product_type (str): The product type (default to None, to use INTRA if not provided)
            """
            # Default values if not provided
            if exchange_segment is None:
                if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                    exchange_segment = dhan.NSE_FNO  # Futures and Options segment
                else:
                    exchange_segment = dhan.BSE_FNO

            if product_type is None:
                product_type = dhan.INTRA  # Intraday product type

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
            # print('order status::',order_data['status'])
            # print("order id", order_data['orderId'])

            return order_data


        # access_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQxMDgxMDEzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.AjXo7XYfSqoc38AelIa22TNqZV_Doul3RtB_IkhDp7yEmQ69NPCTXotUUg7KmWLYDvWJXjz8ZN5DiULee6xe4w'
        # client_id = '1104779876'

        # Main trading loop
        from dhanhq import dhanhq, marketfeed
        import nest_asyncio
        import time

        nest_asyncio.apply()

        # Add your Dhan Client ID and Access Token

        # Define trade parameters
        entry_price = entry_price  # Example entry price for NIFTY 50 23000CE
        entry_price2 = entry_price2
        stop_loss_distance = stop_loss_distance  # Trailing stop loss distance (in points)
        stop_loss_distance2 = stop_loss_distance2
        target_distance = target_distance  # Initial target distance (in points)
        target_distance2 = target_distance2
        quantity = quantity
        quantity2 =quantity2
        security_id = security_id  # 75300 PE Example security_id for options
        security_id2 = security_id2  
        #security_id = 844230
        profit_threshold = profit_threshold
        profit_threshold2 = profit_threshold2

        loss_threshold = loss_threshold
        loss_threshold2 = loss_threshold2
        print(f"profit threshold {profit_threshold}")
        print(f"loss threshold {loss_threshold}")
        if selected_index in ['Nifty','Bank Nifty','Fin Nifty','Midcap Nifty']:
            market_feed_value = marketfeed.NSE_FNO  # Futures and Options segment
        else:
            market_feed_value = marketfeed.BSE_FNO
        instruments = [(market_feed_value, str(security_id), marketfeed.Ticker)]  # Ticker Data
        instruments2 = [(market_feed_value, str(security_id2), marketfeed.Ticker)]  # Ticker Data
        version = "v2"  # Mention Version and set to latest version 'v2'

        # Define order status variables
        order_status = "not_placed"  # Can be 'not_placed', 'placed', 'target_hit', 'stop_loss_hit'
        order_status2 = 'not_placed'
        highest_price = 0  # To track the highest price reached after entry
        current_target = entry_price + target_distance  # Initial target based on entry price
        highest_price2 = 0
        current_target2 = entry_price2 + target_distance2

        trade1_not_exit=True
        trade2_not_exit=True


        data_client_id = "1104779876"
        data_access_token = token_id


        # Main trading loop
        try:
            data = marketfeed.DhanFeed(data_client_id, data_access_token, instruments, version)
            data2 = marketfeed.DhanFeed(data_client_id, data_access_token, instruments2, version)
            print("Fetching Data for backtesting")
            print(f"security id1 {security_id}")
            print(f"security id2 {security_id2}")

            from datetime import timedelta

            start_time = datetime.datetime.now()
            visited_once = False
           

            while(trade1_not_exit | trade2_not_exit):

                data.run_forever()
                response = data.get_data()
                # print(response)

                data2.run_forever()
                response2 = data2.get_data()
               

                if(('LTP' in response.keys()) and (trade1_not_exit)):
                    ltp = response['LTP']
                    #st.write(f"LTP {ltp}")
                    print(f"{selected_index} {strike_price} {option_type}   LTP1:   {ltp}",flush=True,end='\r')

                    # Place buy order if LTP reaches the entry price
                    if order_status == "not_placed":
                        if less_than_or_greater_than == ">=":
                            if float(ltp) >= entry_price:
                                
                                print(f"{float(ltp)} >= {entry_price}")
                                # st.write("LTP reached entry price, and ema cross over placing order...")
                                print("LTP reached entry price, placing order...")
                                # Place buy order
                                place_order(security_id, quantity, float(ltp), "buy")                            
                                order_status = "placed"
                                highest_price = float(ltp)  # Set highest price to entry price
                                print(f"Buy order placed at {ltp}")
                                # st.write(f"Buy order placed at {ltp}")
                                entry_price = float(ltp)

                                # order_status = "not_placed"  # Default order status
                                entry_time = datetime.datetime.now()  # Default entry time (current time)
                                # exit_time = None  # Exit time will be set later when the trade is closed
                                # profit_or_loss = 0  # Initial profit/loss is zero
                                # profit_points = 0  # Initial profit points
                                # loss_points = 0  # Initial loss points
                                backtest = True  # Default to backtesting,  set to False for live trading
                               
                        else:
                            if float(ltp) <= entry_price:
                                # st.write("LTP reached entry price or below, and ema cross over placing order...")
                                print("LTP reached entry price , placing order...")
                                # Place buy order
                                place_order(security_id, quantity, float(ltp), "buy")
                                order_status = "placed"
                                highest_price = float(ltp)  # Set highest price to entry price
                                print(f"Buy order placed at {ltp}")
                                # st.write(f"Buy order placed at {ltp}")
                                entry_price = float(ltp)

                                # order_status = "not_placed"  # Default order status
                                entry_time = datetime.datetime.now()  # Default entry time (current time)
                                # exit_time = None  # Exit time will be set later when the trade is closed
                                # profit_or_loss = 0  # Initial profit/loss is zero
                                # profit_points = 0  # Initial profit points
                                # loss_points = 0  # Initial loss points
                                backtest = False  # Default to backtesting,  set to False for live trading


                    # Check if target or stop loss (with trailing) is hit
                    if order_status == "placed" and trade1_not_exit:

                        # Calculate profit/loss based on amount
                        profit_or_loss = (float(ltp) - entry_price) * quantity
                        print(f"Current Profit/Loss1: {profit_or_loss}",flush=True, end='\r')
                        # profit_placeholder.markdown(f"Current Profit/Loss: {profit_or_loss}")

                        # Trailing Target: If the price increases, increase the target
                        if float(ltp) >= current_target:
                            current_target = float(ltp) + target_distance  # Increase the target
                            print(f"Target1 adjusted to {current_target}")
                            # st.write(f"Target adjusted to {current_target}")

                        # Update highest price and trailing stop loss if LTP goes higher
                        if float(ltp) > highest_price:
                            highest_price = float(ltp)  # New highest price

                        if use_trailing_stop_loss=="Yes":
                            # Calculate the trailing stop loss based on the highest price
                            trailing_stop_loss_price = highest_price - stop_loss_distance
                            print(f"Entry price1: {entry_price} Trailing Stop Loss1 at: {trailing_stop_loss_price} highest price1: {highest_price} target1: {current_target}",flush=True, end='\r')
                            # trailing_placeholder.markdown(f"Entry price: {entry_price} Trailing Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")
                        else:
                            trailing_stop_loss_price = entry_price-stop_loss_distance
                            current_target = entry_price+target_distance
                            print(f"Entry price1: {entry_price} Fixed Stop Loss1 at: {trailing_stop_loss_price} highest price1: {highest_price} target1: {current_target}",flush=True, end='\r')
                        
                        # Check if the price hits the trailing target
                        if float(ltp) >= current_target:
                            print("Trailing target1 hit! Exiting trade.")
                            # st.write("Trailing target hit! Exiting trade.")
                            order_status = "target_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id, quantity, float(ltp), "sell")
                            print(f"Exited at {ltp}")
                            # st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss1: {profit_or_loss}")
                            trade1_not_exit = False
                            highest_profit = profit_or_loss
                            # st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            profit_points = float(ltp) - entry_price  # Initial profit points
                            
                           
                            data.disconnect()
                            # break

                        # If the price falls below the trailing stop loss, exit the trade
                        if float(ltp) <= trailing_stop_loss_price:
                            print("Trailing stop1 loss hit! Exiting trade.")
                            trade1_not_exit = False
                            # st.write("Trailing stop loss hit! Exiting trade.")
                            order_status = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id, quantity, float(ltp), "sell")
                            print(f"Exited at {ltp}")
                            # st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss1: {profit_or_loss}")
                            # st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            # profit_points = float(ltp) - entry_price  # Initial profit points
                            loss_points = float(ltp) - entry_price  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)

                            data.disconnect()
                            # break



                        # Check if profit or loss threshold is exceeded
                        if profit_or_loss >= profit_threshold:
                            print(f"Profit threshold1 reached! Exiting trade with profit of {profit_or_loss}.")
                            # st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss}.")
                            order_status = "target_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id, quantity, float(ltp), "sell")
                            print(f"Exited at {ltp}")
                            trade1_not_exit = False
                            highest_profit = profit_or_loss
                            # st.write(f"Exited at {ltp}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            profit_points = float(ltp) - entry_price  # Initial profit points
                            # loss_points = float(ltp) - entry_price  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)

                            data.disconnect()
                            # break
                        elif profit_or_loss <= -loss_threshold:
                            print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            order_status = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id, quantity, float(ltp), "sell")
                            print(f"Exited at {ltp}")
                            # st.write(f"Exited at {ltp}")
                            trade1_not_exit = False

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            # profit_points = float(ltp) - entry_price  # Initial profit points
                            loss_points = float(ltp) - entry_price  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)

                            data.disconnect()
                            # break

                        # if float(ltp) >= entry_price + 2:
                        #     # order_status ='not_placed'
                        #     # entry_price = (entry_price+float(ltp))/2
                        #     quantity = quantity+75
                        #     entry_price = (entry_price * quantity + float(ltp) * 75) / (quantity + 75)
                            
                            
                        #     place_order(security_id, quantity, float(ltp), "buy")
                        #     # place_order(security_id, quantity, float(ltp), "buy")
                        #     print(f'Quantityy1 increased to: {quantity}')
                        #     print(f"updated entry price1 : {entry_price}")
                        #     visited_once = True
                        #     # trade1_not_exit=True

                if(('LTP' in response2.keys()) and (trade2_not_exit)):
                    ltp2 = response2['LTP']
                    #st.write(f"LTP {ltp}")
                    print(f"{selected_index2} {strike_price2} {option_type2}   LTP2:   {ltp2}",flush=True,end='\r')

                    # Place buy order if LTP reaches the entry price
                    if order_status2 == "not_placed":
                        if less_than_or_greater_than2 == ">=":
                            if float(ltp2) >= entry_price2:
                                current_time2 = datetime.datetime.now()
                                #st.write(current_time)
                                
                                print(f"{float(ltp2)} >= {entry_price2}")
                                # st.write("LTP reached entry price, and ema cross over placing order...")
                                print("LTP reached entry price,  placing order...")
                                # Place buy order
                                place_order(security_id2, quantity2, float(ltp2), "buy")
                                order_status2 = "placed"
                                highest_price2 = float(ltp2)  # Set highest price to entry price
                                print(f"Buy order2 placed at {ltp2}")
                                # st.write(f"Buy order placed at {ltp}")
                                entry_price2 = float(ltp2)

                        else:
                            if float(ltp2) <= entry_price2:
                                
                                # st.write("LTP reached entry price or below, and ema cross over placing order...")
                                print("LTP reached entry price , placing order...")
                                # Place buy order
                                place_order(security_id2, quantity2, float(ltp2), "buy")
                                order_status2 = "placed"
                                highest_price2 = float(ltp2)  # Set highest price to entry price
                                print(f"Buy order2 placed at {ltp2}")
                                # st.write(f"Buy order placed at {ltp}")
                                entry_price2 = float(ltp2)



                    # Check if target or stop loss (with trailing) is hit
                    if order_status2 == "placed" and trade2_not_exit:

                        # Calculate profit/loss based on amount
                        profit_or_loss2 = (float(ltp2) - entry_price2) * quantity2
                        print(f"Current Profit/Loss2: {profit_or_loss2}",flush=True, end='\r')
                        # profit_placeholder.markdown(f"Current Profit/Loss: {profit_or_loss}")

                        # Trailing Target: If the price increases, increase the target
                        if float(ltp2) >= current_target2:
                            current_target2 = float(ltp2) + target_distance2  # Increase the target
                            print(f"Target2 adjusted to {current_target2}")
                            # st.write(f"Target adjusted to {current_target}")

                        # Update highest price and trailing stop loss if LTP goes higher
                        if float(ltp2) > highest_price2:
                            highest_price2 = float(ltp2)  # New highest price

                        if use_trailing_stop_loss2=="Yes":
                            # Calculate the trailing stop loss based on the highest price
                            trailing_stop_loss_price2 = highest_price2 - stop_loss_distance2
                            print(f"Entry price: {entry_price2} Trailing Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}",flush=True, end='\r')
                            # trailing_placeholder.markdown(f"Entry price: {entry_price} Trailing Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")
                        else:
                            trailing_stop_loss_price2 = entry_price2-stop_loss_distance2
                            current_target2 = entry_price2+target_distance2
                            print(f"Entry price: {entry_price2} Fixed Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}",flush=True, end='\r')

                        # Check if the price hits the trailing target
                        if float(ltp2) >= current_target2:
                            print("Trailing target hit! Exiting trade.")
                            # st.write("Trailing target hit! Exiting trade.")
                            order_status2 = "target_hit"
                            trade2_not_exit=False
                            # Exit the trade (place a sell order)
                            place_order(security_id2, quantity2, float(ltp2), "sell")
                            print(f"Exited at {ltp2}")
                            # st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss2: {profit_or_loss2}")
                            # st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time2 = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                            profit_points2 = float(ltp2) - entry_price2  # Initial profit points
                            highest_profit2 = profit_or_loss2
                            # loss_points = 0  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)
                           
                            data2.disconnect()
                            # break

                        # If the price falls below the trailing stop loss, exit the trade
                        if float(ltp2) <= trailing_stop_loss_price2:
                            print("Trailing stop loss hit! Exiting trade.")
                            # st.write("Trailing stop loss hit! Exiting trade.")
                            order_status2 = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id2, quantity2, float(ltp2), "sell")
                            print(f"Exited at {ltp2}")
                            # st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss2: {profit_or_loss2}")
                            trade2_not_exit=False
                            # st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            
                            profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                            # profit_points = float(ltp) - entry_price  # Initial profit points
                            loss_points2 = float(ltp2) - entry_price2  # Initial loss points
                            

                            data2.disconnect()
                            # break



                        # Check if profit or loss threshold is exceeded
                        if profit_or_loss2 >= profit_threshold2:
                            print(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss2}.")
                            # st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss}.")
                            order_status2 = "target_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id2, quantity2, float(ltp2), "sell")
                            print(f"Exited at {ltp2}")
                            # st.write(f"Exited at {ltp}")
                            trade2_not_exit=False

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time2 = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                            profit_points2 = float(ltp2) - entry_price2  # Initial profit points
                            highest_profit2 = profit_or_loss2
                            

                            data2.disconnect()
                            # break
                        elif profit_or_loss2 <= -loss_threshold2:
                            print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss2}.")
                            # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            order_status2 = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id2, quantity2, float(ltp2), "sell")
                            print(f"Exited at {ltp2}")
                            # st.write(f"Exited at {ltp}")
                            trade2_not_exit=False

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time2 = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                            # profit_points = float(ltp) - entry_price  # Initial profit points
                            loss_points2 = float(ltp2) - entry_price2  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)

                            data2.disconnect()
                            # break

                        # if float(ltp2) >= entry_price2 + 2:
                        #     # order_status ='not_placed'
                        #     # entry_price = (entry_price+float(ltp))/2
                        #     quantity2 = quantity2+75
                        #     entry_price2 = (entry_price2 * quantity2 + float(ltp2) * 75) / (quantity2 + 75)
                            
                            
                        #     place_order(security_id2, quantity2, float(ltp2), "buy")
                        #     print(f'Quantityy2 increased to : {quantity2}')
                        #     print(f"updated entry price2 : {entry_price2}")
                        #     visited_once = True

                temp = profit_or_loss+profit_or_loss2
                temp2 = highest_profit+highest_profit2

                
                if temp >= 800:
                    if temp > max_total_profit:
                        max_total_profit = temp

                

                if max_total_profit > 800 and temp <= 0.5 * max_total_profit:
                    
                    print(f"Total profit fell by 50% ! Exiting trade with profit of {temp}.")
                    # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                    if(order_status!='target_hit' or order_status!='stop_loss_hit'):
                        place_order(security_id, quantity, float(ltp), "sell")

                    if(order_status2!='target_hit' or order_status2!='stop_loss_hit'):
                        place_order(security_id2, quantity2, float(ltp2), "sell")
                    
                    # print(f"Exited at {ltp2}")
                    # st.write(f"Exited at {ltp}")
                    trade1_not_exit = False
                    trade2_not_exit = False

                    # order_status = "not_placed"  # Default order status
                    # entry_time = datetime.datetime.now()  # Default entry time (current time)
                    exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                    # profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                
                    data.disconnect()
                    # break


                print(f'total profit or loss : {profit_or_loss+profit_or_loss2}  CE: {profit_or_loss}  PE: {profit_or_loss2} qty1:{quantity} qty2:{quantity2}')
                # print(f'temp {temp}')

                # if((datetime.datetime.now() - start_time) == timedelta(seconds=30)):
                #     # your condition logic here
                #     if(not visited_once):
                #         print(f'No momentum exiting trade with total profit or loss {profit_or_loss+profit_or_loss2} CE: {profit_or_loss}  PE: {profit_or_loss2}')
                #         break 

                
                

                


                if temp >= total_profit_threshold:
                    print(f"Total profit threshold reached! Exiting trade with profit of {temp}.")
                    # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                    order_status = "target_hit"
                    # Exit the trade (place a sell order)
                    place_order(security_id2, quantity, float(ltp2), "sell")
                    print(f"Exited at {ltp}")
                    # st.write(f"Exited at {ltp}")
                    trade1_not_exit = False
                    trade2_not_exit = False

                    # order_status = "not_placed"  # Default order status
                    # entry_time = datetime.datetime.now()  # Default entry time (current time)
                    exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                    profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                    # profit_points = float(ltp) - entry_price  # Initial profit points
                    loss_points = float(ltp) - entry_price  # Initial loss points
                    # backtest = False  # Default to backtesting,  set to False for live trading

                    # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                    #              stop_loss_distance, target_distance,order_status, entry_time,
                    #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                    #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                    #             traded_date,use_trailing_stop_loss)

                    data.disconnect()
                    # break

                

                if temp <= -total_loss_threshold:
                    print(f"Total Loss threshold reached! Exiting trade with loss of {temp}.")
                    # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                    order_status = "stop_loss_hit"
                    # Exit the trade (place a sell order)
                    place_order(security_id2, quantity, float(ltp2), "sell")
                    print(f"Exited at {ltp}")
                    # st.write(f"Exited at {ltp}")
                    trade1_not_exit = False
                    trade2_not_exit = False

                    # order_status = "not_placed"  # Default order status
                    # entry_time = datetime.datetime.now()  # Default entry time (current time)
                    exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                    profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                    # profit_points = float(ltp) - entry_price  # Initial profit points
                    loss_points = float(ltp) - entry_price  # Initial loss points
                    # backtest = False  # Default to backtesting,  set to False for live trading

                    # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                    #              stop_loss_distance, target_distance,order_status, entry_time,
                    #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                    #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                    #             traded_date,use_trailing_stop_loss)

                    data.disconnect()
                    # break

           
        except KeyboardInterrupt:
            print("Execution interrupted by user. Disconnecting...")
            # st.write("Execution interrupted by user. Disconnecting...")
           
            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]
                unsub_instruments2 = [(marketfeed.NSE, str(security_id2), 16)]

                data.unsubscribe_symbols(unsub_instruments)
                data2.unsubscribe_symbols(unsub_instruments2)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]
                unsub_instruments2 = [(marketfeed.BSE, str(security_id2), 16)]

                data.unsubscribe_symbols(unsub_instruments)
                

                data2.unsubscribe_symbols(unsub_instruments2)

            data.disconnect()  # This ensures disconnect when the program is forcefully stopped.
            data2.disconnect()  #
           
        
           
           

        finally:
           
            #Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]
                unsub_instruments2 = [(marketfeed.NSE, str(security_id2), 16)]


                data.unsubscribe_symbols(unsub_instruments)
                data2.unsubscribe_symbols(unsub_instruments2)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]
                unsub_instruments2 = [(marketfeed.BSE, str(security_id2), 16)]


                data.unsubscribe_symbols(unsub_instruments)
                data2.unsubscribe_symbols(unsub_instruments2)

            data.disconnect()
            data2.disconnect()

        # Close Connection
        data.disconnect()
        data2.disconnect()

    elif trade_mode == "Backtesting":
        from dhanhq import dhanhq, marketfeed
        import nest_asyncio
        import time

        nest_asyncio.apply()

        # Add your Dhan Client ID and Access Token

        # Define trade parameters
        entry_price = entry_price  # Example entry price for NIFTY 50 23000CE
        entry_price2 = entry_price2
        stop_loss_distance = stop_loss_distance  # Trailing stop loss distance (in points)
        stop_loss_distance2 = stop_loss_distance2
        target_distance = target_distance  # Initial target distance (in points)
        target_distance2 = target_distance2
        quantity = quantity
        quantity2 =quantity2
        security_id = security_id  # 75300 PE Example security_id for options
        security_id2 = security_id2  
        #security_id = 844230
        profit_threshold = profit_threshold
        profit_threshold2 = profit_threshold2

        loss_threshold = loss_threshold
        loss_threshold2 = loss_threshold2
        print(f"profit threshold {profit_threshold}")
        print(f"loss threshold {loss_threshold}")
        if selected_index in ['Nifty','Bank Nifty','Fin Nifty','Midcap Nifty']:
            market_feed_value = marketfeed.NSE_FNO  # Futures and Options segment
        else:
            market_feed_value = marketfeed.BSE_FNO
        instruments = [(market_feed_value, str(security_id), marketfeed.Ticker)]  # Ticker Data
        instruments2 = [(market_feed_value, str(security_id2), marketfeed.Ticker)]  # Ticker Data
        version = "v2"  # Mention Version and set to latest version 'v2'

        # Define order status variables
        order_status = "not_placed"  # Can be 'not_placed', 'placed', 'target_hit', 'stop_loss_hit'
        order_status2 = 'not_placed'
        highest_price = 0  # To track the highest price reached after entry
        current_target = entry_price + target_distance  # Initial target based on entry price
        highest_price2 = 0
        current_target2 = entry_price2 + target_distance2

        trade1_not_exit=True
        trade2_not_exit=True


        data_client_id = "1104779876"
        data_access_token = token_id


        # Main trading loop
        try:
            data = marketfeed.DhanFeed(data_client_id, data_access_token, instruments, version)
            data2 = marketfeed.DhanFeed(data_client_id, data_access_token, instruments2, version)
            print("Fetching Data for backtesting")
            print(f"security id1 {security_id}")
            print(f"security id2 {security_id2}")

            from datetime import timedelta

            start_time = datetime.datetime.now()
            visited_once = False
           

            while(trade1_not_exit | trade2_not_exit):

                data.run_forever()
                response = data.get_data()
                # print(response)

                data2.run_forever()
                response2 = data2.get_data()
               

                if(('LTP' in response.keys()) and (trade1_not_exit)):
                    ltp = response['LTP']
                    #st.write(f"LTP {ltp}")
                    print(f"{selected_index} {strike_price} {option_type}   LTP1:   {ltp}",flush=True,end='\r')

                    # Place buy order if LTP reaches the entry price
                    if order_status == "not_placed":
                        if less_than_or_greater_than == ">=":
                            if float(ltp) >= entry_price:
                                
                                print(f"{float(ltp)} >= {entry_price}")
                                # st.write("LTP reached entry price, and ema cross over placing order...")
                                print("LTP reached entry price, placing order...")
                                # Place buy order
                            
                                order_status = "placed"
                                highest_price = float(ltp)  # Set highest price to entry price
                                print(f"Buy order placed at {ltp}")
                                # st.write(f"Buy order placed at {ltp}")
                                entry_price = float(ltp)

                                # order_status = "not_placed"  # Default order status
                                entry_time = datetime.datetime.now()  # Default entry time (current time)
                                # exit_time = None  # Exit time will be set later when the trade is closed
                                # profit_or_loss = 0  # Initial profit/loss is zero
                                # profit_points = 0  # Initial profit points
                                # loss_points = 0  # Initial loss points
                                backtest = True  # Default to backtesting,  set to False for live trading
                               
                        else:
                            if float(ltp) <= entry_price:
                                # st.write("LTP reached entry price or below, and ema cross over placing order...")
                                print("LTP reached entry price , placing order...")
                                # Place buy order
                            
                                order_status = "placed"
                                highest_price = float(ltp)  # Set highest price to entry price
                                print(f"Buy order placed at {ltp}")
                                # st.write(f"Buy order placed at {ltp}")
                                entry_price = float(ltp)

                                # order_status = "not_placed"  # Default order status
                                entry_time = datetime.datetime.now()  # Default entry time (current time)
                                # exit_time = None  # Exit time will be set later when the trade is closed
                                # profit_or_loss = 0  # Initial profit/loss is zero
                                # profit_points = 0  # Initial profit points
                                # loss_points = 0  # Initial loss points
                                backtest = False  # Default to backtesting,  set to False for live trading


                    # Check if target or stop loss (with trailing) is hit
                    if order_status == "placed" and trade1_not_exit:

                        # Calculate profit/loss based on amount
                        profit_or_loss = ((float(ltp) - entry_price) * quantity)
                        print(f"Current Profit/Loss1: {profit_or_loss}",flush=True, end='\r')
                        # profit_placeholder.markdown(f"Current Profit/Loss: {profit_or_loss}")

                        # Trailing Target: If the price increases, increase the target
                        if float(ltp) >= current_target:
                            current_target = float(ltp) + target_distance  # Increase the target
                            print(f"Target1 adjusted to {current_target}")
                            # st.write(f"Target adjusted to {current_target}")

                        # Update highest price and trailing stop loss if LTP goes higher
                        if float(ltp) > highest_price:
                            highest_price = float(ltp)  # New highest price

                        if use_trailing_stop_loss=="Yes":
                            # Calculate the trailing stop loss based on the highest price
                            trailing_stop_loss_price = highest_price - stop_loss_distance
                            print(f"Entry price1: {entry_price} Trailing Stop Loss1 at: {trailing_stop_loss_price} highest price1: {highest_price} target1: {current_target}",flush=True, end='\r')
                            # trailing_placeholder.markdown(f"Entry price: {entry_price} Trailing Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")
                        else:
                            trailing_stop_loss_price = entry_price-stop_loss_distance
                            current_target = entry_price+target_distance
                            print(f"Entry price1: {entry_price} Fixed Stop Loss1 at: {trailing_stop_loss_price} highest price1: {highest_price} target1: {current_target}",flush=True, end='\r')
                        
                        # Check if the price hits the trailing target
                        if float(ltp) >= current_target:
                            print("Trailing target1 hit! Exiting trade.")
                            # st.write("Trailing target hit! Exiting trade.")
                            order_status = "target_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp}")
                            # st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss1: {profit_or_loss}")
                            trade1_not_exit = False
                            highest_profit = profit_or_loss
                            # st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            profit_points = float(ltp) - entry_price  # Initial profit points
                            
                           
                            data.disconnect()
                            # break

                        # If the price falls below the trailing stop loss, exit the trade
                        if float(ltp) <= trailing_stop_loss_price:
                            print("Trailing stop1 loss hit! Exiting trade.")
                            trade1_not_exit = False
                            # st.write("Trailing stop loss hit! Exiting trade.")
                            order_status = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp}")
                            # st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss1: {profit_or_loss}")
                            # st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            # profit_points = float(ltp) - entry_price  # Initial profit points
                            loss_points = float(ltp) - entry_price  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)

                            data.disconnect()
                            # break



                        # Check if profit or loss threshold is exceeded
                        if profit_or_loss >= profit_threshold:
                            print(f"Profit threshold1 reached! Exiting trade with profit of {profit_or_loss}.")
                            # st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss}.")
                            order_status = "target_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp}")
                            trade1_not_exit = False
                            highest_profit = profit_or_loss
                            # st.write(f"Exited at {ltp}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            profit_points = float(ltp) - entry_price  # Initial profit points
                            # loss_points = float(ltp) - entry_price  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)

                            data.disconnect()
                            # break
                        elif profit_or_loss <= -loss_threshold:
                            print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            order_status = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp}")
                            # st.write(f"Exited at {ltp}")
                            trade1_not_exit = False

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            # profit_points = float(ltp) - entry_price  # Initial profit points
                            loss_points = float(ltp) - entry_price  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)

                            data.disconnect()
                            # break

                        # if(float(ltp) >= entry_price + 10):
                        #     # order_status ='not_placed'
                        #     # entry_price = (entry_price+float(ltp))/2
                        #     entry_price = (entry_price * quantity + float(ltp) * 75) / (quantity + 75)
                            
                        #     quantity = quantity+75
                        #     brokerage1 = brokerage1+50
                        #     print(f'Quantityy1 increased to: {quantity}')
                        #     print(f"updated entry price1 : {entry_price}")
                        #     visited_once = True
                        #     # trade1_not_exit=True

                if(('LTP' in response2.keys()) and (trade2_not_exit)):
                    ltp2 = response2['LTP']
                    #st.write(f"LTP {ltp}")
                    print(f"{selected_index2} {strike_price2} {option_type2}   LTP2:   {ltp2}",flush=True,end='\r')

                    # Place buy order if LTP reaches the entry price
                    if order_status2 == "not_placed":
                        if less_than_or_greater_than2 == ">=":
                            if float(ltp2) >= entry_price2:
                                current_time2 = datetime.datetime.now()
                                #st.write(current_time)
                                
                                print(f"{float(ltp2)} >= {entry_price2}")
                                # st.write("LTP reached entry price, and ema cross over placing order...")
                                print("LTP reached entry price,  placing order...")
                                # Place buy order
                            
                                order_status2 = "placed"
                                highest_price2 = float(ltp2)  # Set highest price to entry price
                                print(f"Buy order2 placed at {ltp2}")
                                # st.write(f"Buy order placed at {ltp}")
                                entry_price2 = float(ltp2)

                        else:
                            if float(ltp2) <= entry_price2:
                                
                                # st.write("LTP reached entry price or below, and ema cross over placing order...")
                                print("LTP reached entry price , placing order...")
                                # Place buy order
                            
                                order_status2 = "placed"
                                highest_price2 = float(ltp2)  # Set highest price to entry price
                                print(f"Buy order2 placed at {ltp2}")
                                # st.write(f"Buy order placed at {ltp}")
                                entry_price2 = float(ltp2)



                    # Check if target or stop loss (with trailing) is hit
                    if order_status2 == "placed" and trade2_not_exit:

                        # Calculate profit/loss based on amount
                        profit_or_loss2 = ((float(ltp2) - entry_price2) * quantity2)
                        print(f"Current Profit/Loss2: {profit_or_loss2}",flush=True, end='\r')
                        # profit_placeholder.markdown(f"Current Profit/Loss: {profit_or_loss}")

                        # Trailing Target: If the price increases, increase the target
                        if float(ltp2) >= current_target2:
                            current_target2 = float(ltp2) + target_distance2  # Increase the target
                            print(f"Target2 adjusted to {current_target2}")
                            # st.write(f"Target adjusted to {current_target}")

                        # Update highest price and trailing stop loss if LTP goes higher
                        if float(ltp2) > highest_price2:
                            highest_price2 = float(ltp2)  # New highest price

                        if use_trailing_stop_loss2=="Yes":
                            # Calculate the trailing stop loss based on the highest price
                            trailing_stop_loss_price2 = highest_price2 - stop_loss_distance2
                            print(f"Entry price: {entry_price2} Trailing Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}",flush=True, end='\r')
                            # trailing_placeholder.markdown(f"Entry price: {entry_price} Trailing Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")
                        else:
                            trailing_stop_loss_price2 = entry_price2-stop_loss_distance2
                            current_target2 = entry_price2+target_distance2
                            print(f"Entry price: {entry_price2} Fixed Stop Loss at: {trailing_stop_loss_price2} highest price: {highest_price2} target: {current_target2}",flush=True, end='\r')

                        # Check if the price hits the trailing target
                        if float(ltp2) >= current_target2:
                            print("Trailing target hit! Exiting trade.")
                            # st.write("Trailing target hit! Exiting trade.")
                            order_status2 = "target_hit"
                            trade2_not_exit=False
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp2}")
                            # st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss2: {profit_or_loss2}")
                            # st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time2 = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                            profit_points2 = float(ltp2) - entry_price2  # Initial profit points
                            highest_profit2 = profit_or_loss2
                            # loss_points = 0  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)
                           
                            data2.disconnect()
                            # break

                        # If the price falls below the trailing stop loss, exit the trade
                        if float(ltp2) <= trailing_stop_loss_price2:
                            print(f'iside trailing stop loss--------------------> {profit_or_loss2}')
                            print("Trailing stop loss hit! Exiting trade.")
                            # st.write("Trailing stop loss hit! Exiting trade.")
                            order_status2 = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp2}")
                            # st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss2: {profit_or_loss2}")
                            trade2_not_exit=False
                            # st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            
                            profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                            # profit_points = float(ltp) - entry_price  # Initial profit points
                            loss_points2 = float(ltp2) - entry_price2  # Initial loss points
                            

                            data2.disconnect()
                            # break



                        # Check if profit or loss threshold is exceeded
                        if profit_or_loss2 >= profit_threshold2:
                            print(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss2}.")
                            # st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss}.")
                            order_status2 = "target_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp2}")
                            # st.write(f"Exited at {ltp}")
                            trade2_not_exit=False

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time2 = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                            profit_points2 = float(ltp2) - entry_price2  # Initial profit points
                            highest_profit2 = profit_or_loss2
                            

                            data2.disconnect()
                            # break
                        elif profit_or_loss2 <= -loss_threshold2:
                            print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss2}.")
                            # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            order_status2 = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp2}")
                            # st.write(f"Exited at {ltp}")
                            trade2_not_exit=False

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time2 = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss2 = profit_or_loss2  # Initial profit/loss is zero
                            # profit_points = float(ltp) - entry_price  # Initial profit points
                            loss_points2 = float(ltp2) - entry_price2  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)

                            data2.disconnect()
                            # break

                        # if(float(ltp2) >= entry_price2 + 10):
                        #     # order_status ='not_placed'
                        #     # entry_price = (entry_price+float(ltp))/2
                        #     entry_price2 = (entry_price2 * quantity2 + float(ltp2) * 75) / (quantity2 + 75)
                            
                        #     quantity2 = quantity2+75
                        #     brokerage2 = brokerage2+50
                        #     print(f'Quantityy2 increased to : {quantity2}')
                        #     print(f"updated entry price2 : {entry_price2}")
                        #     visited_once = True

                        # elif float(ltp2) >= entry_price2 + 5:  # Condition to sell quantities
                        #     if quantity2 > 75:
                        #         quantity2 = 75  # Keep only 75 units
                        #         entry_price2 = float(ltp2)
                        #         print(f'initial target reached booking first profit!! booked profit: {profit_or_loss2}')
                        #         print(f'exited at {entry_price2}')

                temp = profit_or_loss+profit_or_loss2
                temp2 = highest_profit+highest_profit2

                

                
                
                if temp > max_total_profit:
                    max_total_profit = temp

                if profit_or_loss > max_highest_profit1:
                    max_highest_profit1 = profit_or_loss

                if profit_or_loss2 > max_highest_profit2:
                    max_highest_profit2 = profit_or_loss2

                if max_highest_profit1 > 800 and profit_or_loss <= 0.4 * max_highest_profit1:
                    
                    print(f"Trade1 Total profit fell by 70% ! Exiting trade with profit of {temp}.")
                    # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                    
                    print(f"Trade1 Exited at {ltp}")
                    # st.write(f"Exited at {ltp}")
                    trade1_not_exit = False
                    # trade2_not_exit = False

                    # order_status = "not_placed"  # Default order status
                    # entry_time = datetime.datetime.now()  # Default entry time (current time)
                    exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                    # profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                
                    data.disconnect()
                    # break

                if max_highest_profit2 > 800 and profit_or_loss2 <= 0.4 * max_highest_profit2:
                    
                    print(f"Trade2 Total profit fell by 70% ! Exiting trade with profit of {temp}.")
                    # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                    
                    print(f"Trade2 Exited at {ltp2}")
                    # st.write(f"Exited at {ltp}")
                    # trade1_not_exit = False
                    trade2_not_exit = False

                    # order_status = "not_placed"  # Default order status
                    # entry_time = datetime.datetime.now()  # Default entry time (current time)
                    exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                    # profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                
                    data.disconnect()
                    # break

                

                

                if max_total_profit > 1000 and temp <= 0.8 * max_total_profit:
                    
                    print(f"Total profit fell by 70% ! Exiting trade with profit of {temp}.")
                    # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                    
                    print(f"Exited at {ltp}")
                    # st.write(f"Exited at {ltp}")
                    trade1_not_exit = False
                    trade2_not_exit = False

                    # order_status = "not_placed"  # Default order status
                    # entry_time = datetime.datetime.now()  # Default entry time (current time)
                    exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                    # profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                
                    data.disconnect()
                    # break


                print(f'total profit or loss : {round((profit_or_loss+profit_or_loss2),2)}  CE: {round(profit_or_loss,2)}  PE: {round(profit_or_loss2,2)} qty1:{quantity} qty2:{quantity2} entry1:{round(entry_price,2)} entry2:{round(entry_price2,2)} ltp1: {round(float(ltp),2)} ltp2: {round(float(ltp2),2)}',flush=True,end='\r')
                # print(f'temp {temp}')

                # if((datetime.datetime.now() - start_time) == timedelta(seconds=30)):
                #     # your condition logic here
                #     if(not visited_once):
                #         print(f'No momentum exiting trade with total profit or loss {profit_or_loss+profit_or_loss2} CE: {profit_or_loss}  PE: {profit_or_loss2}')
                #         break 

                
                

                


                if temp >= total_profit_threshold:
                    print(f"Total profit threshold reached! Exiting trade with profit of {round(temp,2)}. max_total_profit: {max_total_profit}")
                    # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                    order_status = "target_hit"
                    # Exit the trade (place a sell order)
                    
                    print(f"Exited at {ltp}")
                    # st.write(f"Exited at {ltp}")
                    trade1_not_exit = False
                    trade2_not_exit = False

                    # order_status = "not_placed"  # Default order status
                    # entry_time = datetime.datetime.now()  # Default entry time (current time)
                    exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                    profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                    # profit_points = float(ltp) - entry_price  # Initial profit points
                    loss_points = float(ltp) - entry_price  # Initial loss points
                    # backtest = False  # Default to backtesting,  set to False for live trading

                    # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                    #              stop_loss_distance, target_distance,order_status, entry_time,
                    #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                    #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                    #             traded_date,use_trailing_stop_loss)

                    data.disconnect()
                    # break

                

                if temp <= -total_loss_threshold:
                    print(f"Total Loss threshold reached! Exiting trade with loss of {temp}.")
                    # st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                    order_status = "stop_loss_hit"
                    # Exit the trade (place a sell order)
                    
                    print(f"Exited at {ltp}")
                    # st.write(f"Exited at {ltp}")
                    trade1_not_exit = False
                    trade2_not_exit = False

                    # order_status = "not_placed"  # Default order status
                    # entry_time = datetime.datetime.now()  # Default entry time (current time)
                    exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                    profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                    # profit_points = float(ltp) - entry_price  # Initial profit points
                    loss_points = float(ltp) - entry_price  # Initial loss points
                    # backtest = False  # Default to backtesting,  set to False for live trading

                    # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                    #              stop_loss_distance, target_distance,order_status, entry_time,
                    #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                    #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                    #             traded_date,use_trailing_stop_loss)

                    data.disconnect()
                    # break

           
        except KeyboardInterrupt:
            print("Execution interrupted by user. Disconnecting...")
            # st.write("Execution interrupted by user. Disconnecting...")
           
            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]
                unsub_instruments2 = [(marketfeed.NSE, str(security_id2), 16)]

                data.unsubscribe_symbols(unsub_instruments)
                data2.unsubscribe_symbols(unsub_instruments2)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]
                unsub_instruments2 = [(marketfeed.BSE, str(security_id2), 16)]

                data.unsubscribe_symbols(unsub_instruments)
                

                data2.unsubscribe_symbols(unsub_instruments2)

            data.disconnect()  # This ensures disconnect when the program is forcefully stopped.
            data2.disconnect()  #
           
        
           
           

        finally:
           
            #Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]
                unsub_instruments2 = [(marketfeed.NSE, str(security_id2), 16)]


                data.unsubscribe_symbols(unsub_instruments)
                data2.unsubscribe_symbols(unsub_instruments2)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]
                unsub_instruments2 = [(marketfeed.BSE, str(security_id2), 16)]


                data.unsubscribe_symbols(unsub_instruments)
                data2.unsubscribe_symbols(unsub_instruments2)

            data.disconnect()
            data2.disconnect()

        # Close Connection
        data.disconnect()
        data2.disconnect()
