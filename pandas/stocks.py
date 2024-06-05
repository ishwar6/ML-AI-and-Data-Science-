# If Trades were done when Indian Election Results were declared on 4th June when bump is shown. 

import pandas as pd
 
data = [
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Bata India Limited", "Symbol": "BATAINDIA", "Series": "EQ", 
     "Trade No": 2024060408433550, "Trade Time": "11:20:42 AM", "Quantity": 20, "Price (Rs.)": 1312.60, "Traded Value (Rs.)": 26252.00},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Bharat Dynamics Limited", "Symbol": "BDL", "Series": "EQ", 
     "Trade No": 2024060409051830, "Trade Time": "11:29:14 AM", "Quantity": 5, "Price (Rs.)": 1437.65, "Traded Value (Rs.)": 7188.25},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Bharti Airtel Limited", "Symbol": "BHARTIARTL", "Series": "EQ", 
     "Trade No": 2024060408295782, "Trade Time": "11:18:27 AM", "Quantity": 10, "Price (Rs.)": 1298.00, "Traded Value (Rs.)": 12980.00},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Coal India Limited", "Symbol": "COALINDIA", "Series": "EQ", 
     "Trade No": 2024060414391121, "Trade Time": "01:00:24 PM", "Quantity": 17, "Price (Rs.)": 438.00, "Traded Value (Rs.)": 7446.00},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Coal India Limited", "Symbol": "COALINDIA", "Series": "EQ", 
     "Trade No": 2024060414391119, "Trade Time": "01:00:24 PM", "Quantity": 9, "Price (Rs.)": 437.95, "Traded Value (Rs.)": 3941.55},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Coal India Limited", "Symbol": "COALINDIA", "Series": "EQ", 
     "Trade No": 2024060414391120, "Trade Time": "01:00:24 PM", "Quantity": 4, "Price (Rs.)": 437.95, "Traded Value (Rs.)": 1751.80},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "IDFC First Bank Limited", "Symbol": "IDFCFIRSTB", "Series": "EQ", 
     "Trade No": 2024060442296244, "Trade Time": "03:22:35 PM", "Quantity": 12, "Price (Rs.)": 72.50, "Traded Value (Rs.)": 870.00},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Indian Railway Catering And Tourism Corporation Limited", "Symbol": "IRCTC", "Series": "EQ", 
     "Trade No": 2024060442229001, "Trade Time": "03:21:29 PM", "Quantity": 1, "Price (Rs.)": 908.95, "Traded Value (Rs.)": 908.95},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Oil & Natural Gas Corporation Limited", "Symbol": "ONGC", "Series": "EQ", 
     "Trade No": 2024060447461335, "Trade Time": "11:18:35 AM", "Quantity": 50, "Price (Rs.)": 255.70, "Traded Value (Rs.)": 12785.00},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Reliance Industries Limited", "Symbol": "RELIANCE", "Series": "EQ", 
     "Trade No": 2024060467930846, "Trade Time": "11:20:03 AM", "Quantity": 9, "Price (Rs.)": 2817.50, "Traded Value (Rs.)": 25357.50},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Reliance Industries Limited", "Symbol": "RELIANCE", "Series": "EQ", 
     "Trade No": 2024060467930822, "Trade Time": "11:20:03 AM", "Quantity": 1, "Price (Rs.)": 2817.50, "Traded Value (Rs.)": 2817.50},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "State Bank of India", "Symbol": "SBIN", "Series": "EQ", 
     "Trade No": 2024060468614202, "Trade Time": "11:30:13 AM", "Quantity": 5, "Price (Rs.)": 799.20, "Traded Value (Rs.)": 3996.00},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "State Bank of India", "Symbol": "SBIN", "Series": "EQ", 
     "Trade No": 2024060473679219, "Trade Time": "12:55:19 PM", "Quantity": 10, "Price (Rs.)": 787.95, "Traded Value (Rs.)": 7879.50},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Vedanta Limited", "Symbol": "VEDL", "Series": "EQ", 
     "Trade No": 2024060481052526, "Trade Time": "03:20:44 PM", "Quantity": 20, "Price (Rs.)": 416.30, "Traded Value (Rs.)": 8326.00},
    
    {"TM Name": "UPSTOX SECURITIES PRIVATE LIMITED", "Client Code": "sdfaa3r34rsa", "Buy/Sell": "B", 
     "Name of the Security": "Vodafone Idea Limited", "Symbol": "IDEA", "Series": "EQ", 
     "Trade No": 2024060480490658, "Trade Time": "12:59:34 PM", "Quantity": 500, "Price (Rs.)": 13.45, "Traded Value (Rs.)": 6725.00},
]

 
df = pd.DataFrame(data)
 
total_traded_value = df["Traded Value (Rs.)"].sum()

 
total_quantity = df["Quantity"].sum()
total_trades = len(df)
average_price = df["Price (Rs.)"].mean()

total_traded_value, total_quantity, total_trades, average_price

# (129225.05, 673, 15, 950.08)

# Here is the summary of the trades:

# Total Traded Value: ₹129,225.05
# Total Quantity Traded: 673
# Total Number of Trades: 15
# Average Price: ₹950.08 
