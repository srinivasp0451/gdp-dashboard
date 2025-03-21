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

import time
import datetime
import pandas as pd
from Dhan_Tradehull import Tradehull

nest_asyncio.apply()



# Initialize client
client_code = "1104779876"
token_id = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQzNjcyMTgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.I-Hl-IKVl4dioMwp5Qhl-7duuX7BrIXJIc6v6kuLX7g3zMKcNCeAGFstRrbo2N7vDn2WCmY90YxPbmQsnquhpg"
tsl = Tradehull(client_code, token_id)


security_id = 0
order_client_id=''
order_access_token=''
data_client_id=''
data_access_token=''
profit_threshold=''
loss_threshold=''
market_feed_value = marketfeed.NSE
tradesymbol = ''
# # Database setup for storing trade details
# def create_table():
#     conn = sqlite3.connect('trade_journal.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS trades (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     index_name TEXT,
#                     option_type TEXT,
#                     strike_price INTEGER,
#                     expiry_date DATE,
#                     entry_price FLOAT,
#                     stop_loss_distance INTEGER,
#                     target_distance INTEGER,
#                     order_status TEXT,
#                     entry_time TIMESTAMP,
#                     exit_time TIMESTAMP,
#                     profit_or_loss FLOAT,
#                     profit_points FLOAT,
#                     loss_points FLOAT,
#                     capital_used FLOAT,
#                     time_spent_in_trade TEXT,
#                     notes TEXT,
#                     day_of_week TEXT,
#                     month_of_year TEXT,
#                     backtest BOOLEAN,
#                     traded_date TEXT,
#                     use_trailing_stop_loss TEXT
#                  )''')
#     conn.commit()
#     conn.close()

# # Function to insert trade details into the database
# def insert_trade(index_name, option_type, strike_price, expiry_date, entry_price, stop_loss_distance, target_distance,
#                  order_status, entry_time, exit_time, profit_or_loss, profit_points, loss_points, capital_used,
#                  time_spent_in_trade, notes, day_of_week, month_of_year, backtest,traded_date,use_trailing_stop_loss):
#     conn = sqlite3.connect('trade_journal.db')
#     c = conn.cursor()
#     c.execute('''INSERT INTO trades (index_name, option_type, strike_price, expiry_date, entry_price, stop_loss_distance,
#                                       target_distance, order_status, entry_time, exit_time, profit_or_loss, profit_points,
#                                       loss_points, capital_used, time_spent_in_trade, notes, day_of_week, month_of_year, backtest,traded_date,use_trailing_stop_loss)
#                  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
#                  (index_name, option_type, strike_price, expiry_date, entry_price, stop_loss_distance, target_distance,
#                   order_status, entry_time, exit_time, profit_or_loss, profit_points, loss_points, capital_used,
#                   time_spent_in_trade, notes, day_of_week, month_of_year, backtest,traded_date,use_trailing_stop_loss))
#     conn.commit()
#     conn.close()


# create_table()

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
selected_index = st.selectbox("Select Index", ["Nifty","Sensex", "Bank Nifty","Fin Nifty","Midcap Nifty","Bankex"])

# Calendar widget to select expiry date
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.date(2025, 1, 1))

# Dropdown for Option Type (CE or PE)
option_type = st.selectbox("Select Option Type", ["CE", "PE"])

# Dropdown for selecting strike price (you can manually add options or make it dynamic later)
strike_price = st.number_input("Select Strike Price", min_value=0, step=50,value=23250)

# Fetch the data from the CSV URL
df = load_csv_data()






# Input fields for Entry Price, Stop Loss, Target, etc.
entry_price = st.number_input("Entry Price", min_value=0, step=1,value=1)
less_than_or_greater_than = st.selectbox("Select above or below", [">=", "<="])
stop_loss_distance = st.number_input("Stop Loss Distance", min_value=0, step=1,value=10)
target_distance = st.number_input("Target Distance", min_value=0, step=1,value=5)
quantity = st.number_input("Quantity", min_value=1, step=1, value=20)
profit_threshold = st.number_input("Profit Threshold", min_value=1, step=1,value=5000)
loss_threshold = st.number_input("Loss Threshold", min_value=0, step=1,value=350)
timeframe = st.text_input("Time Frame",value=5)

# Dropdown for selecting whether to use trailing stop loss or not
use_trailing_stop_loss = st.selectbox("Use Trailing Stop Loss?", ["No","Yes"])

# Select backtesting or live trading
trade_mode = st.selectbox("Select Trade Mode", ["Live Trading","Backtesting"])



# Inputs for Live Trading (client ID and access token for live trading)
if trade_mode == "Live Trading":
    order_client_id = st.text_input("Client ID (for placing orders)", type="password",value='22305184')
    order_access_token = st.text_input("Access Token (for placing orders)", type="password",value='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQzNjcyMTgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.I-Hl-IKVl4dioMwp5Qhl-7duuX7BrIXJIc6v6kuLX7g3zMKcNCeAGFstRrbo2N7vDn2WCmY90YxPbmQsnquhpg')
else:
    data_client_id = "1104779876"
    data_access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQzNjcyMTgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.I-Hl-IKVl4dioMwp5Qhl-7duuX7BrIXJIc6v6kuLX7g3zMKcNCeAGFstRrbo2N7vDn2WCmY90YxPbmQsnquhpg"



# Display confirmation of selected options
st.subheader("Selected Configuration for Confirmation:")
st.write(f"Index: {selected_index}")
st.write(f"Option Type: {option_type}")
st.write(f"Strike Price: {strike_price}")
st.write(f"Expiry Date: {expiry_date}")
st.write(f"Entry Price: {entry_price}")
st.write(f"Entry Quantity: {quantity}")
st.write(f"Profit Threshold: {profit_threshold}")
st.write(f"Loss Threshold: {loss_threshold}")
st.write(f"Stop Loss Distance: {stop_loss_distance}")
st.write(f"Target Distance: {target_distance}")
st.write(f"Use Trailing Stop Loss: {use_trailing_stop_loss}")
st.write(f"Trade Mode: {trade_mode}")



# Filtered data based on selection
filtered_df = filter_data(df, selected_index, expiry_date, strike_price, option_type)

#print("filtered data will be ",filtered_df)

# Option chain display
if not filtered_df.empty:
    # Show the corresponding security ID
    security_id = filtered_df.iloc[0]['SEM_SMST_SECURITY_ID']
    st.write(f"Security ID for Strike Price {strike_price} and Option Type {option_type}: {security_id}")
    tradesymbol = filtered_df.iloc[0]['SEM_CUSTOM_SYMBOL']
    # st.write(tradesymbol)
    st.dataframe(filtered_df)
else:
    st.write("No data available for the selected criteria.")

st.write(f"Trade Symbol: {tradesymbol}")
st.write(f"Timeframe: {timeframe}")
st.write(f"Security id: {security_id}")

# Button to start/stop trading
# start_button = st.button("Start")

# # Manage session state for stopping and starting the trading process
# if "is_trading_active" not in st.session_state:
#     st.session_state.is_trading_active = False

# if start_button:
#     st.session_state.is_trading_active = True
#     st.session_state.stop_requested = False
#     st.write("Trading has started...")

# if interrupt_button and st.session_state.is_trading_active:
#     st.session_state.stop_requested = True
#     st.write("Stopping trading...")
#     stop_trading = True  # Flag the trading process to stop


# Default values for the variables
selected_index = selected_index  # Default index, can be selected by user
option_type = option_type  # Default option type (CE or PE), can be selected by user
strike_price = strike_price  # Default strike price, can be selected by user
expiry_date = datetime.date.today()  # Default expiry date, can be selected by user

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
profit_placeholder = st.empty()
ltp_placeholder = st.empty()
trailing_placeholder = st.empty()


# Configurable variables
# tradingsymbol = 'NIFTY 27 MAR 22050 PUT'
# tradingsymbol = 'NIFTY 06 MAR 22000 PUT'


# tradingsymbol = 'NIFTY 06 MAR 22000 PUT'
exchange = 'NFO'
# timeframe = '5'

# EMA calculation
def calculate_ema(df, column, period):
    # print(f"{period} EMA: {df[column].ewm(span=period, adjust=False).mean()}")
    return df[column].ewm(span=period, adjust=False).mean()


# Fetch historical data
def fetch_data(tradingsymbol, exchange, timeframe):
    return tsl.get_historical_data(
        tradingsymbol=tradingsymbol,
        exchange=exchange,
        timeframe=timeframe
    )
# EMA crossover strategy with order execution
def generate_signals(df):

    df['ema9'] = calculate_ema(df, 'close', 9)
    df['ema20'] = calculate_ema(df, 'close', 20)

    # st.write(f"EMA1: {df['ema9']}")
    # st.write(f"EMA2: {df['ema20']}")

    # print("df::",df)

    old_candle = df.iloc[-2]
    latest_candle = df.iloc[-1]
    # print(f"Latest Candle :: {latest_candle}")
    # print(f"Old Candle :: {old_candle}")
    # print(f"ema 9 {latest_candle['ema9']}")
    # print(f"ema 9 {latest_candle['ema9']}")

    print(f"condition {latest_candle['ema9'] > latest_candle['ema20']}")
    print(f"condition {old_candle['ema9'] <= old_candle['ema20']}")

            
    if latest_candle['ema9'] > latest_candle['ema20'] and old_candle['ema9'] <= old_candle['ema20']:
    #if latest_candle['ema9'] > latest_candle['ema20']:
        st.write('EMA Crossovver')
        st.write(f"EMA 1: {latest_candle['ema9']}")
        st.write(f"EMA 2: {latest_candle['ema20']}")
        buy_signal =True
        return buy_signal
    elif latest_candle['ema9'] < latest_candle['ema20'] and old_candle['ema9']>= old_candle['ema20']:
        sell_signal =True
        buy_signal = False
        return buy_signal

    


# Confirm the configuration before proceeding
if st.button("Start") and security_id:
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
                security_id=symbol,  # Security ID of the option (symbol passed as argument)
                exchange_segment=exchange_segment,  # Exchange segment for Futures & Options
                transaction_type=dhan.BUY if order_type.lower() == "buy" else dhan.SELL,
                quantity=qty,  # Quantity of the option contracts
                order_type=dhan.LIMIT,  # Market order
                product_type=product_type,  # Product type (e.g., INTRA for intraday)
                price=price  # Price at which to place the order (0 for market orders)
            )
            print("order is", order_data)
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
        stop_loss_distance = stop_loss_distance  # Trailing stop loss distance (in points)
        target_distance = target_distance  # Initial target distance (in points)
        quantity = quantity
        security_id = security_id  # 75300 PE Example security_id for options
        #security_id = 844230
        profit_threshold = profit_threshold
        loss_threshold = loss_threshold
        print(f"profit threshold {profit_threshold}")
        print(f"loss threshold {loss_threshold}")
        if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
            market_feed_value = marketfeed.NSE_FNO  # Futures and Options segment
        else:
            market_feed_value = marketfeed.BSE_FNO
        instruments = [(market_feed_value, str(security_id), marketfeed.Ticker)]  # Ticker Data
        version = "v2"  # Mention Version and set to latest version 'v2'

        # Define order status variables
        order_status = "not_placed"  # Can be 'not_placed', 'placed', 'target_hit', 'stop_loss_hit'
        highest_price = 0  # To track the highest price reached after entry
        current_target = entry_price + target_distance  # Initial target based on entry price


        data_client_id = "1104779876"
        data_access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQzNjcyMTgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.I-Hl-IKVl4dioMwp5Qhl-7duuX7BrIXJIc6v6kuLX7g3zMKcNCeAGFstRrbo2N7vDn2WCmY90YxPbmQsnquhpg"
        # Main trading loop
        try:
            data = marketfeed.DhanFeed(data_client_id, data_access_token, instruments, version)
            st.write("Fetching Data for live trading")
            print(f"security id {security_id}")
           

            while True:
                # if st.button("Interrupt"):
                #     print("Execution interrupted by user. Disconnecting...")
                #     st.write("Execution interrupted by user. Disconnecting...")
                   
                #     # Unsubscribe instruments which are already active on connection
                #     if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                #         unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]

                #         data.unsubscribe_symbols(unsub_instruments)
                       
                #     else:
                #         unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]

                #         data.unsubscribe_symbols(unsub_instruments)

                #     data.disconnect()  # This ensures disconnect when the program is forcefully stopped.
                #     # break

                data.run_forever()
                response = data.get_data()
                # print(response)
               

                if 'LTP' in response.keys():
                    ltp = response['LTP']
                    #st.write(f"LTP {ltp}")
                    ltp_placeholder.markdown(f"{selected_index} {strike_price} {option_type}   LTP:   {ltp}")

                    # Place buy order if LTP reaches the entry price
                    if order_status == "not_placed":
                        if less_than_or_greater_than == ">=":
                            if float(ltp) >= entry_price:
                                current_time = datetime.datetime.now()
                                if current_time.minute % int(timeframe) == 0 and current_time.second == 0:
                                    # Configurable variables
                                    print(tradesymbol)
                                    print(f"exchane{exchange}")
                                    print(f"timeframe {timeframe}")
                                    
                                    fetched_df = fetch_data(tradesymbol,exchange,timeframe)
                                    st.write(f"fetched data:{fetched_df.tail(2)}")
                                    signal = generate_signals(fetched_df)
                                    # st.write("STATUS",signal)

                                    if(signal==True):
                                        st.write(f"{float(ltp)} >= {entry_price}")
                                        st.write("LTP reached above entry price, placing order...")
                                        print("LTP reached entry price, placing order...")
                                        # Place buy order
                                        place_order(security_id, quantity, float(ltp), "buy")
                                    
                                        order_status = "placed"
                                        highest_price = float(ltp)  # Set highest price to entry price
                                        print(f"Buy order placed at {ltp}")
                                        st.write(f"Buy order placed at {ltp}")
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
                                current_time = datetime.datetime.now()
                                if current_time.minute % int(timeframe) == 0 and current_time.second == 0:
                                    # Configurable variables
                                    print(tradesymbol)
                                    print(f"exchane{exchange}")
                                    print(f"timeframe {timeframe}")
                                    
                                    fetched_df = fetch_data(tradesymbol,exchange,timeframe)
                                    st.write(f"fetched data:{fetched_df.tail(2)}")
                                    signal = generate_signals(fetched_df)
                                    # st.write("STATUS",signal)

                                    if(signal==True):
                                        st.write("LTP reached entry price or below, placing order...")
                                        print("LTP reached entry price, placing order...")
                                        # Exit the trade (place a sell order)
                                        place_order(security_id, quantity, float(ltp), "buy")
                                    
                                        order_status = "placed"
                                        highest_price = float(ltp)  # Set highest price to entry price
                                        print(f"Buy order placed at {ltp}")
                                        st.write(f"Buy order placed at {ltp}")
                                        entry_price = float(ltp)

                                        # order_status = "not_placed"  # Default order status
                                        entry_time = datetime.datetime.now()  # Default entry time (current time)
                                        # exit_time = None  # Exit time will be set later when the trade is closed
                                        # profit_or_loss = 0  # Initial profit/loss is zero
                                        # profit_points = 0  # Initial profit points
                                        # loss_points = 0  # Initial loss points
                                        backtest = False  # Default to backtesting,  set to False for live trading


                    # Check if target or stop loss (with trailing) is hit
                    if order_status == "placed":

                        # Calculate profit/loss based on amount
                        profit_or_loss = (float(ltp) - entry_price) * quantity
                        print(f"Current Profit/Loss: {profit_or_loss}")
                        profit_placeholder.markdown(f"Current Profit/Loss: {profit_or_loss}")

                        # Trailing Target: If the price increases, increase the target
                        if float(ltp) >= current_target:
                            current_target = float(ltp) + target_distance  # Increase the target
                            print(f"Target adjusted to {current_target}")
                            st.write(f"Target adjusted to {current_target}")

                        # Update highest price and trailing stop loss if LTP goes higher
                        if float(ltp) > highest_price:
                            highest_price = float(ltp)  # New highest price

                        if use_trailing_stop_loss=="Yes":
                            # Calculate the trailing stop loss based on the highest price
                            trailing_stop_loss_price = highest_price - stop_loss_distance
                            print(f"Entry price: {entry_price} Trailing Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")
                            trailing_placeholder.markdown(f"Entry price: {entry_price} Trailing Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")
                        else:
                            trailing_stop_loss_price = entry_price-stop_loss_distance
                            current_target = entry_price+target_distance
                            trailing_placeholder.markdown(f"Entry price: {entry_price} Fixed Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")

                        # Check if the price hits the trailing target
                        if float(ltp) >= current_target:
                            print("Trailing target hit! Exiting trade.")
                            st.write("Trailing target hit! Exiting trade.")
                            order_status = "target_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id, quantity, float(ltp), "sell")
                           
                            print(f"Exited at {ltp}")
                            st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss: {profit_or_loss}")
                            st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            profit_points = float(ltp) - entry_price  # Initial profit points
                            # loss_points = 0  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)
                           
                            data.disconnect()
                            break

                        # If the price falls below the trailing stop loss, exit the trade
                        if float(ltp) <= trailing_stop_loss_price:
                            print("Trailing stop loss hit! Exiting trade.")
                            st.write("Trailing stop loss hit! Exiting trade.")
                            order_status = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id, quantity, float(ltp), "sell")
                            
                            print(f"Exited at {ltp}")
                            st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss: {profit_or_loss}")
                            st.write(f"Current Profit/Loss: {profit_or_loss}")

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
                            break



                        # Check if profit or loss threshold is exceeded
                        if profit_or_loss >= profit_threshold:
                            print(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss}.")
                            st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss}.")
                            order_status = "target_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id, quantity, float(ltp), "sell")
                           
                            print(f"Exited at {ltp}")
                            st.write(f"Exited at {ltp}")

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
                            break
                        elif profit_or_loss <= -loss_threshold:
                            print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            order_status = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                            place_order(security_id, quantity, float(ltp), "sell")
                           
                            print(f"Exited at {ltp}")
                            st.write(f"Exited at {ltp}")

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
                            break
           
        except KeyboardInterrupt:
            print("Execution interrupted by user. Disconnecting...")
            st.write("Execution interrupted by user. Disconnecting...")
           
            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)

            data.disconnect()  # This ensures disconnect when the program is forcefully stopped.
           
           

        except Exception as e:
            print(e)
            print("Exception occured")
            print("Execution interrupted by user. Disconnecting...")
            st.write("Execution interrupted by user. Disconnecting...")
           
            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)

            data.disconnect()  # This ensures disconnect when the program is forcefully stopped.
           
           

        finally:
           
            #Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)

            data.disconnect()

        # Close Connection
        data.disconnect()

    elif trade_mode == "Backtesting":
        from dhanhq import dhanhq, marketfeed
        import nest_asyncio
        import time

        nest_asyncio.apply()

        # Add your Dhan Client ID and Access Token

        # Define trade parameters
        entry_price = entry_price  # Example entry price for NIFTY 50 23000CE
        stop_loss_distance = stop_loss_distance  # Trailing stop loss distance (in points)
        target_distance = target_distance  # Initial target distance (in points)
        quantity = quantity
        security_id = security_id  # 75300 PE Example security_id for options
        #security_id = 844230
        profit_threshold = profit_threshold
        loss_threshold = loss_threshold
        print(f"profit threshold {profit_threshold}")
        print(f"loss threshold {loss_threshold}")
        if selected_index in ['Nifty','Bank Nifty','Fin Nifty','Midcap Nifty']:
            market_feed_value = marketfeed.NSE_FNO  # Futures and Options segment
        else:
            market_feed_value = marketfeed.BSE_FNO
        instruments = [(market_feed_value, str(security_id), marketfeed.Ticker)]  # Ticker Data
        version = "v2"  # Mention Version and set to latest version 'v2'

        # Define order status variables
        order_status = "not_placed"  # Can be 'not_placed', 'placed', 'target_hit', 'stop_loss_hit'
        highest_price = 0  # To track the highest price reached after entry
        current_target = entry_price + target_distance  # Initial target based on entry price


        data_client_id = "1104779876"
        data_access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQzNjcyMTgwLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNDc3OTg3NiJ9.I-Hl-IKVl4dioMwp5Qhl-7duuX7BrIXJIc6v6kuLX7g3zMKcNCeAGFstRrbo2N7vDn2WCmY90YxPbmQsnquhpg"



        # Main trading loop
        try:
            data = marketfeed.DhanFeed(data_client_id, data_access_token, instruments, version)
            st.write("Fetching Data for backtesting")
            print(f"security id {security_id}")
           

            while True:
                # if st.button("Interrupt"):
                #     print("Execution interrupted by user. Disconnecting...")
                #     st.write("Execution interrupted by user. Disconnecting...")
                   
                #     # Unsubscribe instruments which are already active on connection
                #     if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                #         unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]

                #         data.unsubscribe_symbols(unsub_instruments)
                       
                #     else:
                #         unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]

                #         data.unsubscribe_symbols(unsub_instruments)

                #     data.disconnect()  # This ensures disconnect when the program is forcefully stopped.
                #     # break

                data.run_forever()
                response = data.get_data()
                # print(response)
               

                if 'LTP' in response.keys():
                    ltp = response['LTP']
                    #st.write(f"LTP {ltp}")
                    ltp_placeholder.markdown(f"{selected_index} {strike_price} {option_type}   LTP:   {ltp}")

                    # Place buy order if LTP reaches the entry price
                    if order_status == "not_placed":
                        if less_than_or_greater_than == ">=":
                            if float(ltp) >= entry_price:
                                current_time = datetime.datetime.now()
                                #st.write(current_time)
                                if current_time.minute % int(timeframe) == 0 and current_time.second == 0:
                                    # Configurable variables
                                    print(tradesymbol)
                                    print(f"exchane{exchange}")
                                    print(f"timeframe {timeframe}")
                                    
                                    fetched_df = fetch_data(tradesymbol,exchange,timeframe)
                                    st.write(f"fetched data:{fetched_df.tail(2)}")
                                    signal = generate_signals(fetched_df)
                                    # st.write("STATUS",signal)
                                    if(signal==True):
                                        st.write(f"{float(ltp)} >= {entry_price}")
                                        st.write("LTP reached entry price, and ema cross over placing order...")
                                        print("LTP reached entry price, placing order...")
                                        # Place buy order
                                    
                                        order_status = "placed"
                                        highest_price = float(ltp)  # Set highest price to entry price
                                        print(f"Buy order placed at {ltp}")
                                        st.write(f"Buy order placed at {ltp}")
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
                                current_time = datetime.datetime.now()
                                if current_time.minute % int(timeframe) == 0 and current_time.second == 0:
                                    # Configurable variables
                                    print(tradesymbol)
                                    print(f"exchane{exchange}")
                                    print(f"timeframe {timeframe}")
                                    
                                    fetched_df = fetch_data(tradesymbol,exchange,timeframe)
                                    st.write(f"fetched data:{fetched_df.tail(2)}")
                                    signal = generate_signals(fetched_df)
                                    # st.write("STATUS",signal)
                                    if(signal==True):
                                        st.write("LTP reached entry price or below, and ema cross over placing order...")
                                        print("LTP reached entry price, placing order...")
                                        # Place buy order
                                    
                                        order_status = "placed"
                                        highest_price = float(ltp)  # Set highest price to entry price
                                        print(f"Buy order placed at {ltp}")
                                        st.write(f"Buy order placed at {ltp}")
                                        entry_price = float(ltp)

                                        # order_status = "not_placed"  # Default order status
                                        entry_time = datetime.datetime.now()  # Default entry time (current time)
                                        # exit_time = None  # Exit time will be set later when the trade is closed
                                        # profit_or_loss = 0  # Initial profit/loss is zero
                                        # profit_points = 0  # Initial profit points
                                        # loss_points = 0  # Initial loss points
                                        backtest = False  # Default to backtesting,  set to False for live trading


                    # Check if target or stop loss (with trailing) is hit
                    if order_status == "placed":

                        # Calculate profit/loss based on amount
                        profit_or_loss = (float(ltp) - entry_price) * quantity
                        print(f"Current Profit/Loss: {profit_or_loss}")
                        profit_placeholder.markdown(f"Current Profit/Loss: {profit_or_loss}")

                        # Trailing Target: If the price increases, increase the target
                        if float(ltp) >= current_target:
                            current_target = float(ltp) + target_distance  # Increase the target
                            print(f"Target adjusted to {current_target}")
                            st.write(f"Target adjusted to {current_target}")

                        # Update highest price and trailing stop loss if LTP goes higher
                        if float(ltp) > highest_price:
                            highest_price = float(ltp)  # New highest price

                        if use_trailing_stop_loss=="Yes":
                            # Calculate the trailing stop loss based on the highest price
                            trailing_stop_loss_price = highest_price - stop_loss_distance
                            print(f"Entry price: {entry_price} Trailing Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")
                            trailing_placeholder.markdown(f"Entry price: {entry_price} Trailing Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")
                        else:
                            trailing_stop_loss_price = entry_price-stop_loss_distance
                            current_target = entry_price+target_distance
                            trailing_placeholder.markdown(f"Entry price: {entry_price} Fixed Stop Loss at: {trailing_stop_loss_price} highest price: {highest_price} target: {current_target}")

                        # Check if the price hits the trailing target
                        if float(ltp) >= current_target:
                            print("Trailing target hit! Exiting trade.")
                            st.write("Trailing target hit! Exiting trade.")
                            order_status = "target_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp}")
                            st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss: {profit_or_loss}")
                            st.write(f"Current Profit/Loss: {profit_or_loss}")

                            # order_status = "not_placed"  # Default order status
                            # entry_time = datetime.datetime.now()  # Default entry time (current time)
                            exit_time = datetime.datetime.now()  # Exit time will be set later when the trade is closed
                            profit_or_loss = profit_or_loss  # Initial profit/loss is zero
                            profit_points = float(ltp) - entry_price  # Initial profit points
                            # loss_points = 0  # Initial loss points
                            # backtest = False  # Default to backtesting,  set to False for live trading

                            # insert_trade(selected_index, option_type, strike_price, expiry_date, entry_price,
                            #              stop_loss_distance, target_distance,order_status, entry_time,
                            #              exit_time, profit_or_loss, profit_points, loss_points, capital_used,
                            #             time_spent_in_trade, notes, day_of_week, month_of_year, backtest,
                            #             traded_date,use_trailing_stop_loss)
                           
                            data.disconnect()
                            break

                        # If the price falls below the trailing stop loss, exit the trade
                        if float(ltp) <= trailing_stop_loss_price:
                            print("Trailing stop loss hit! Exiting trade.")
                            st.write("Trailing stop loss hit! Exiting trade.")
                            order_status = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp}")
                            st.write(f"Exited at {ltp}")
                            print(f"Current Profit/Loss: {profit_or_loss}")
                            st.write(f"Current Profit/Loss: {profit_or_loss}")

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
                            break



                        # Check if profit or loss threshold is exceeded
                        if profit_or_loss >= profit_threshold:
                            print(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss}.")
                            st.write(f"Profit threshold reached! Exiting trade with profit of {profit_or_loss}.")
                            order_status = "target_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp}")
                            st.write(f"Exited at {ltp}")

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
                            break
                        elif profit_or_loss <= -loss_threshold:
                            print(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            st.write(f"Loss threshold reached! Exiting trade with loss of {profit_or_loss}.")
                            order_status = "stop_loss_hit"
                            # Exit the trade (place a sell order)
                           
                            print(f"Exited at {ltp}")
                            st.write(f"Exited at {ltp}")

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
                            break
           
        except KeyboardInterrupt:
            print("Execution interrupted by user. Disconnecting...")
            st.write("Execution interrupted by user. Disconnecting...")
           
            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)

            data.disconnect()  # This ensures disconnect when the program is forcefully stopped.
           
           

        except Exception as e:
            print(e)
            print("Exception occured")
            print("Execution interrupted by user. Disconnecting...")
            st.write("Execution interrupted by user. Disconnecting...")
           
            # Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)

            data.disconnect()  # This ensures disconnect when the program is forcefully stopped.
           
           

        finally:
           
            #Unsubscribe instruments which are already active on connection
            if selected_index in ['Nifty','BANKNIFTY','FINNIFTY','MIDCPNIFTY']:
                unsub_instruments = [(marketfeed.NSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)
               
            else:
                unsub_instruments = [(marketfeed.BSE, str(security_id), 16)]

                data.unsubscribe_symbols(unsub_instruments)

            data.disconnect()

        # Close Connection
        data.disconnect()
