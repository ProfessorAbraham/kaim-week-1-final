import pandas as pd
import yfinance as yf
# Load the dataset
df = pd.read_csv('../data/raw_analyst_ratings.csv')
# Prepare the Data
df['date'] = pd.to_datetime(df['date'], format='ISO8601')
# Group the data by stock symbol and aggregate the minimum and maximum dates
grouped_data = df.groupby('stock')['date'].agg(['min', 'max'])
# Display the date range for each stock
print("Date Ranges for Each Stock:")
print(grouped_data)
# Initialize an empty DataFrame to store all stock data
all_stock_data1 = pd.DataFrame()
# Loop through each group (stock symbol) in grouped_data
for stock_symbol, date_range in grouped_data.iterrows():
    start_date = date_range['min']
    end_date = date_range['max']
    
    try:
        # Download stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        # Add stock symbol as a column in the DataFrame
        stock_data['Symbol'] = stock_symbol
        
        # Append stock data to the DataFrame containing all stock data
        all_stock_data1 = pd.concat([all_stock_data1, stock_data])
        
        print("Stock data for", stock_symbol, "downloaded successfully.")
    except Exception as e:
        print("Failed to download stock data for", stock_symbol, ":", str(e))

# Save all stock data to a CSV file
all_stock_data1.to_csv('all_stock_data1.csv')