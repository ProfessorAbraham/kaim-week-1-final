# Import necessary libraries
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

# Ensure you have the necessary NLP libraries and datasets
nltk.download('punkt')
nltk.download('stopwords')

# Descriptive Statistics
## Textual lengths
df['headline_length'] = df['headline'].apply(len)

## Articles per publisher
articles_per_publisher = df['publisher'].value_counts()

## Publication date trends
df['publication_date'] = pd.to_datetime(df['date'])
df['publication_day'] = df['publication_date'].dt.day_name()
publication_trends = df.groupby('publication_day').size()

# Display Descriptive Statistics
print("Descriptive Statistics:")
print(df[['headline_length', 'publisher', 'publication_day']].describe())

# Text Analysis
## Sentiment Analysis
df['sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

## Topic Modeling - Word Cloud for visualization
stop_words = set(stopwords.words('english'))
text = ' '.join(df['headline'])
wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(text)

# Display Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Time Series Analysis
## Publication frequency over time
df.set_index('publication_date').resample('D').size().plot()
plt.title('Article Publication Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()

# Publisher Analysis
## Most active publishers
print("Most Active Publishers:")
print(articles_per_publisher.head())

## Publishing times analysis
if 'time' in df.columns:
    df['publication_time'] = pd.to_datetime(df['time']).dt.time
    sns.histplot(df['publication_time'].astype(str), kde=False)
    plt.title('Distribution of Publishing Times')
    plt.xlabel('Time of Day')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()