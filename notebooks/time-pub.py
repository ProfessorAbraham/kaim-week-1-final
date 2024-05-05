import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
# Load the dataset
df = pd.read_csv('../data/raw_analyst_ratings.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
## Publication date trends
df['publication_date'] = pd.to_datetime(df['date'])
df['publication_day'] = df['publication_date'].dt.day_name()
publication_trends = df.groupby('publication_day').size()
# Time Series Analysis
## Publication frequency over time
df['publication_date'] = pd.to_datetime(df['publication_date'])
df['publication_time'] = df['publication_date'].dt.time
df.set_index('publication_date').resample('D').size().plot()
plt.title('Article Publication Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()# Publisher Analysis
## Most active publishers
articles_per_publisher = df['publisher'].value_counts()
print("Most Active Publishers:")
print(articles_per_publisher.head())
import matplotlib.pyplot as plt

# Publisher Analysis
## Most active publishers
articles_per_publisher = df['publisher'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
articles_per_publisher.head().plot(kind='bar', color='skyblue')
plt.title('Most Active Publishers')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
## Analysis of publishing times by publishers

publisher_time_analysis = df.groupby('publisher')['publication_time'].value_counts()
print("Analysis of Publishing Times by Publishers:")
print(publisher_time_analysis)