import pandas as pd
from textblob import TextBlob
import yfinance as yf
import numpy as np
from scipy.stats import pearsonr
# Load news headlines data
news_data = pd.read_csv('../data/raw_analyst_ratings.csv')
stock_data= pd.read_csv('../data/all_stock_data2.csv')
print(news_data)
print(stock_data)
# Assuming 'news_data' is your DataFrame containing the 'date' column
news_data['date'] = pd.to_datetime(news_data['date'])  # Convert to datetime object
news_data['date'] = news_data['date'].dt.strftime('%Y-%m-%d')  # Format date as '%Y-%m-%
print(news_data['date'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='ISO8601')
print(stock_data['Date'])
# Apply sentiment analysis to news headlines
news_data['sentiment'] = news_data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
print(news_data['sentiment'])
#Calculate Stock Movements
# Compute daily percentage change in closing prices to represent stock movements
stock_data['Daily_Returns'] = stock_data['Close'].pct_change() * 100
# Correlation Analysis
# Aggregate sentiment scores if multiple articles appear on the same day
daily_sentiment = news_data.groupby('date')['sentiment'].mean().reset_index()

# Ensure both datasets have the same dates
merged_data = pd.concat([daily_sentiment, stock_data], axis=1)
print(merged_data)
# Check for missing values
missing_sentiment = merged_data['sentiment'].isnull().sum()
missing_returns = merged_data['Daily_Returns'].isnull().sum()
print("Missing values in sentiment:", missing_sentiment)
print("Missing values in Daily_Returns:", missing_returns)

# Check for infinite values
infinite_sentiment = np.isinf(merged_data['sentiment']).sum()
infinite_returns = np.isinf(merged_data['Daily_Returns']).sum()
print("Infinite values in sentiment:", infinite_sentiment)
print("Infinite values in Daily_Returns:", infinite_returns)
print("Merged dataset:")
print(merged_data)
print(len(merged_data['sentiment']))
print(len(merged_data['Daily_Returns']))

import numpy as np
from scipy.stats import pearsonr

# Check for NaN or infinite values in the arrays
nan_mask = np.isnan(merged_data['sentiment']) | np.isnan(merged_data['Daily_Returns'])
inf_mask = np.isinf(merged_data['sentiment']) | np.isinf(merged_data['Daily_Returns'])

# Combine NaN and infinite masks
invalid_mask = nan_mask | inf_mask

# Remove invalid values from both arrays
clean_sentiment = merged_data['sentiment'][~invalid_mask]
clean_returns = merged_data['Daily_Returns'][~invalid_mask]

# Calculate Pearson correlation coefficient between clean sentiment scores and stock returns
correlation_coefficient, p_value = pearsonr(clean_sentiment, clean_returns)
print("Pearson correlation coefficient:", correlation_coefficient)
print("p-value:", p_value)