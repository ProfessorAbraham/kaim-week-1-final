#importing liberaries
import pandas as pd
import talib
import pynance as pn
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import numpy as np
# Load and Prepare the Data
stock_data= pd.read_csv('../data/all_stock_data1.csv')  # Assuming you have a CSV file named 'stock_data.csv'
stock_data.head()
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2020-12-31'

# Get stock data using PyNance
stock_data = pn.data.get(stock_symbol, start_date, end_date)
print(stock_data)
# Calculate moving averages (50-day and 200-day)
stock_data['MA_20'] = talib.SMA(stock_data['Close'], timeperiod=20)
stock_data['MA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)

# Calculate relative strength index (RSI)
stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)

# Calculate moving average convergence divergence (MACD)
macd, macdsignal, macdhist = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
stock_data['MACD'] = macd
stock_data['MACD_signal'] = macdsignal
# Plotting the data
plt.figure(figsize=(12, 6))
# Plot AAPL closing price
plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
# Plot 20-day and 50-day moving averages
plt.plot(stock_data.index, stock_data['MA_20'], label='20-day MA')
plt.plot(stock_data.index, stock_data['MA_50'], label='50-day MA')

plt.figure(figsize=(12, 4))
plt.plot(stock_data.index, stock_data['RSI'], label='RSI', color='orange')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.title('AAPL Relative Strength Index (RSI)')
plt.legend()
plt.show()
# Plot MACD
plt.figure(figsize=(12, 4))
plt.plot(stock_data.index, stock_data['MACD'], label='MACD', color='blue')
plt.plot(stock_data.index, stock_data['MACD_signal'], label='MACD Signal', color='orange')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.title('AAPL Moving Average Convergence Divergence (MACD)')
plt.legend()
plt.show()
# Calculate financial metrics

# Calculate daily returns
daily_returns = stock_data['Close'].pct_change()

# Calculate annual return
annual_return = daily_returns.mean() * 252  # Assuming 252 trading days in a year

# Calculate volatility (annualized standard deviation)
volatility = daily_returns.std() * np.sqrt(252)  # Assuming 252 trading days in a year

# Print the calculated metrics
print("Annual Return:", annual_return)
print("Volatility:", volatility)