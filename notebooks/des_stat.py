import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
df = pd.read_csv('../data/raw_analyst_ratings.csv')

print(df)
df.head
df['headline_length'] = df['headline'].apply(len)
headline_stats = df['headline_length'].describe()
print("Headline Length Statistics:")
print(headline_stats)

## Count the number of articles per publisher
publisher_counts = df['publisher'].value_counts()
# Use pandas' groupby() function to count the number of articles per publisher
articles_per_publisher = df.groupby('publisher').size().reset_index(name='article_count')
most_active_publishers = articles_per_publisher.sort_values(by='article_count', ascending=False).head(5)
print("\nMost Active Publishers:")
print(most_active_publishers)

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
# Plotting publication frequency over time
plt.figure(figsize=(10, 6))
df.groupby('year').size().plot(kind='line', marker='o')
plt.title('Publication Frequency Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Articles')
plt.grid(True)
plt.show()